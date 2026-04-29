#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORKSPACE = Path('/root/.openclaw/workspace')
SCRIPTS = WORKSPACE / 'scripts'
OUT_DIR = WORKSPACE / 'tmp' / 'ch4_nomarkov_scale_dynamic_noise_observability_2026-04-08'
OUT_JSON = OUT_DIR / 'ch4_nomarkov_scale_dynamic_noise_observability_2026-04-08.json'
OUT_MD = OUT_DIR / 'ch4_nomarkov_scale_dynamic_noise_observability_2026-04-08.md'
OUT_FULL = OUT_DIR / 'fig_ch4_nomarkov_scale_dynamic_noise_observability_2026-04-08.png'
OUT_A = OUT_DIR / 'fig_ch4_nomarkov_scale_dynamic_noise_observability_panel_A_2026-04-08.png'
OUT_B = OUT_DIR / 'fig_ch4_nomarkov_scale_dynamic_noise_observability_panel_B_2026-04-08.png'
BASE12_PATH = SCRIPTS / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
GM_HELPER_PATH = SCRIPTS / 'alignvn_dar_truth_gm_helper_2026-03-31.py'
SEEDS = [0, 1, 2, 3, 4]
PROFILES = ['baseline', 'small_gm']
STATE_LABELS = [
    'phi_E', 'phi_N', 'phi_U',
    'dV_E', 'dV_N', 'dV_U',
    'eb_x', 'eb_y', 'eb_z',
    'db_x', 'db_y', 'db_z',
    'kg_x', 'kg_y', 'kg_z',
    'ka_x', 'ka_y', 'ka_z',
]
FAMILY_SPANS = [
    ('phi', 0, 2, '#2563eb'),
    ('dV', 3, 5, '#0891b2'),
    ('eb', 6, 8, '#7c3aed'),
    ('db', 9, 11, '#ea580c'),
    ('kg', 12, 14, '#b91c1c'),
    ('ka', 15, 17, '#0f766e'),
]

_base12 = None
_gm = None


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod



def load_base12():
    global _base12
    if _base12 is None:
        _base12 = load_module('nomarkov_scale_base12_20260408', BASE12_PATH)
    return _base12



def load_gm():
    global _gm
    if _gm is None:
        _gm = load_module('nomarkov_scale_gmhelper_20260408', GM_HELPER_PATH)
    return _gm



def build_initial_scales(base12) -> np.ndarray:
    imuerr = base12.build_imuerr()
    phi0 = np.array([0.1, 0.1, 0.5]) * base12.glv.deg
    dV0 = np.ones(3)
    init_eb = np.maximum(np.asarray(imuerr['eb']).reshape(3), 0.1 * base12.glv.dph)
    init_db = np.maximum(np.asarray(imuerr['db']).reshape(3), 1000.0 * base12.glv.ug)
    scale0 = np.full(3, 100.0 * base12.glv.ppm)
    return np.r_[phi0, dV0, init_eb, init_db, scale0, scale0]



def avnkfinit_18(nts: float, pos: np.ndarray, phi0: np.ndarray, imuerr: dict[str, np.ndarray], wvn: np.ndarray, scale_sigma: float) -> dict[str, Any]:
    base12 = load_base12()
    eth = base12.Earth(pos)
    web = np.asarray(imuerr['web']).reshape(3)
    wdb = np.asarray(imuerr['wdb']).reshape(3)
    eb = np.asarray(imuerr['eb']).reshape(3)
    db = np.asarray(imuerr['db']).reshape(3)

    init_eb_p = np.maximum(eb, 0.1 * base12.glv.dph)
    init_db_p = np.maximum(db, 1000 * base12.glv.ug)
    init_scale_p = np.full(3, scale_sigma)

    ft = np.zeros((18, 18))
    ft[0:3, 0:3] = base12.askew(-eth.wnie)

    qk = np.zeros((18, 18))
    qk[0:3, 0:3] = np.diag(web**2 * nts)
    qk[3:6, 3:6] = np.diag(wdb**2 * nts)
    q_scale = (1e-8) ** 2 * nts
    qk[12:15, 12:15] = np.eye(3) * q_scale
    qk[15:18, 15:18] = np.eye(3) * q_scale

    return {
        'n': 18,
        'm': 3,
        'nts': nts,
        'Qk': qk,
        'Rk': np.diag(wvn.reshape(3)) ** 2 / nts,
        'Pxk': np.diag(np.r_[phi0, np.ones(3), init_eb_p, init_db_p, init_scale_p, init_scale_p]) ** 2,
        'Phikk_1': np.eye(18) + ft * nts,
        'Hk': np.hstack([np.zeros((3, 3)), np.eye(3), np.zeros((3, 12))]),
        'xk': np.zeros(18),
    }



def alignvn_18state_iter(
    imu: np.ndarray,
    qnb: np.ndarray,
    pos: np.ndarray,
    phi0: np.ndarray,
    imuerr: dict[str, np.ndarray],
    wvn: np.ndarray,
    max_iter: int,
    truth_att: np.ndarray,
    wash_scale: float = 0.5,
    carry_att_seed: bool = True,
    scale_sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base12 = load_base12()
    if scale_sigma is None:
        scale_sigma = 100.0 * base12.glv.ppm

    imu_corr = imu.copy()
    nn = 2
    ts = float(imu_corr[1, -1] - imu_corr[0, -1])
    nts = nn * ts
    qnb_seed = base12.a2qua(qnb) if len(qnb) == 3 else np.asarray(qnb).reshape(4)

    length = (len(imu_corr) // nn) * nn
    imu_corr = imu_corr[:length]

    eth = base12.Earth(pos)
    cnn = base12.rv2m(-eth.wnie * nts / 2)

    final_attk = None
    final_xkpk = None
    for iteration in range(1, max_iter + 1):
        kf = avnkfinit_18(nts, pos, phi0, imuerr, wvn, scale_sigma)
        vn = np.zeros(3)
        qnbi = qnb_seed.copy()

        attk_rows: list[np.ndarray] = []
        xkpk_rows: list[np.ndarray] = []

        for k in range(0, length, nn):
            wvm = imu_corr[k:k + nn, 0:6]
            t = float(imu_corr[k + nn - 1, -1])
            phim, dvbm = base12.cnscl(wvm)

            cnb = base12.q2mat(qnbi)
            dvn = cnn @ cnb @ dvbm
            vn = vn + dvn + eth.gn * nts
            qnbi = base12.qupdt2(qnbi, phim, eth.wnin * nts)

            phi_k = kf['Phikk_1'].copy()
            cnbts = cnb * nts
            phi_k[3:6, 0:3] = base12.askew(dvn)
            phi_k[3:6, 9:12] = cnbts
            phi_k[0:3, 6:9] = -cnbts
            phi_k[0:3, 12:15] = -cnb @ np.diag(phim[0:3])
            phi_k[3:6, 15:18] = cnb @ np.diag(dvbm[0:3])
            kf['Phikk_1'] = phi_k

            kf = base12.kfupdate(kf, vn)

            qnbi = base12.qdelphi(qnbi, 0.91 * kf['xk'][0:3])
            kf['xk'][0:3] *= 0.09
            vn = vn - 0.91 * kf['xk'][3:6]
            kf['xk'][3:6] *= 0.09

            attk_rows.append(np.r_[base12.q2att(qnbi), vn, t])
            xkpk_rows.append(np.r_[kf['xk'], np.diag(kf['Pxk']), t])

        attk = np.asarray(attk_rows)
        xkpk = np.asarray(xkpk_rows)
        final_attk = attk
        final_xkpk = xkpk

        if iteration < max_iter:
            if carry_att_seed:
                qnb_seed = qnbi.copy()
            imu_corr[:, 0:3] -= wash_scale * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= wash_scale * kf['xk'][9:12] * ts
            for axis in range(3):
                imu_corr[:, axis] = imu_corr[:, axis] / (1.0 + wash_scale * kf['xk'][12 + axis])
                imu_corr[:, 3 + axis] = imu_corr[:, 3 + axis] / (1.0 + wash_scale * kf['xk'][15 + axis])

    assert final_attk is not None and final_xkpk is not None
    return final_attk[-1, 0:3], final_attk, final_xkpk



def build_noisy_imu(profile: str, seed: int, imu_clean: np.ndarray):
    base12 = load_base12()
    gm = load_gm()
    np.random.seed(seed)
    imuerr = gm.build_truth_imuerr_variant(profile=profile)
    base_keys = ('eb', 'db', 'web', 'wdb', 'dKg', 'dKa')
    imu_base = {key: np.array(imuerr[key], copy=True) for key in base_keys}
    imu_noisy = base12.imuadderr(imu_clean, imu_base)

    gyro_gm_last = np.zeros(3)
    accel_gm_last = np.zeros(3)
    if bool(imuerr.get('truth_gm_enabled', False)):
        ts = float(imu_noisy[1, -1] - imu_noisy[0, -1])
        n = int(imu_noisy.shape[0])
        gyro_bias = gm._generate_first_order_gm_bias(
            n=n,
            ts=ts,
            sigma=np.asarray(imuerr['truth_gm_gyro_sigma']),
            tau_s=np.asarray(imuerr['truth_gm_tau_g_s']),
        )
        accel_bias = gm._generate_first_order_gm_bias(
            n=n,
            ts=ts,
            sigma=np.asarray(imuerr['truth_gm_accel_sigma']),
            tau_s=np.asarray(imuerr['truth_gm_tau_a_s']),
        )
        imu_noisy[:, 0:3] += gyro_bias * ts
        imu_noisy[:, 3:6] += accel_bias * ts
        gyro_gm_last = gyro_bias[-1]
        accel_gm_last = accel_bias[-1]

    truth_final_eb = np.asarray(imuerr['eb']).reshape(3) + gyro_gm_last
    truth_final_db = np.asarray(imuerr['db']).reshape(3) + accel_gm_last
    truth_dkg = np.diag(np.asarray(imuerr['dKg']))
    truth_dka = np.diag(np.asarray(imuerr['dKa']))
    return imu_noisy, imuerr, truth_final_eb, truth_final_db, truth_dkg, truth_dka



def run_single(profile: str, seed: int) -> dict[str, Any]:
    base12 = load_base12()
    gm = load_gm()

    ts = 0.01
    max_iter = 5
    wash_scale = 0.5
    carry_att_seed = True
    att0 = np.array([0.0, 0.0, 0.0])
    pos0 = base12.posset(34, 116, 480, isdeg=1)
    rot_paras = base12.build_rot_paras()
    att_truth = base12.attrottt(att0, rot_paras, ts)
    imu_clean, _ = base12.avp2imu(att_truth, pos0)

    imu_noisy, imuerr, truth_eb, truth_db, truth_dkg, truth_dka = build_noisy_imu(profile, seed, imu_clean)

    phi = np.array([0.1, 0.1, 0.5]) * base12.glv.deg
    att0_guess = base12.q2att(base12.qaddphi(base12.a2qua(att0), phi))
    wvn = np.array([0.01, 0.01, 0.01])
    truth_att = att_truth[-1, 0:3]

    att_aligned, attk, xkpk = alignvn_18state_iter(
        imu=imu_noisy.copy(),
        qnb=att0_guess,
        pos=pos0,
        phi0=phi,
        imuerr=imuerr,
        wvn=wvn,
        max_iter=max_iter,
        truth_att=truth_att,
        wash_scale=wash_scale,
        carry_att_seed=carry_att_seed,
    )

    final_att = attk[-1, 0:3]
    final_vn = attk[-1, 3:6]
    final_x = xkpk[-1, 0:18]

    phi_err = base12.qq2phi(base12.a2qua(final_att), base12.a2qua(truth_att))
    dV_err = final_vn.copy()
    eb_err = final_x[6:9] - truth_eb
    db_err = final_x[9:12] - truth_db
    kg_err = final_x[12:15] - truth_dkg
    ka_err = final_x[15:18] - truth_dka

    err_vec = np.r_[phi_err, dV_err, eb_err, db_err, kg_err, ka_err]
    return {
        'profile': profile,
        'seed': seed,
        'profile_desc': gm.describe_truth_profile(profile),
        'state_abs_errors_native': [float(abs(x)) for x in err_vec],
        'display': {
            'phi_arcsec': [float(abs(x / base12.glv.sec)) for x in phi_err],
            'dV_mps': [float(abs(x)) for x in dV_err],
            'eb_dph': [float(abs(x / base12.glv.dph)) for x in eb_err],
            'db_ug': [float(abs(x / base12.glv.ug)) for x in db_err],
            'kg_ppm': [float(abs(x / base12.glv.ppm)) for x in kg_err],
            'ka_ppm': [float(abs(x / base12.glv.ppm)) for x in ka_err],
        },
    }



def normalize_inverse_log(values_a: np.ndarray, values_b: np.ndarray):
    eps = 1e-30
    concat = np.r_[values_a, values_b]
    logv = np.log10(np.maximum(concat, eps))
    lo = float(np.min(logv))
    hi = float(np.max(logv))
    if hi - lo < 1e-12:
        sa = np.ones_like(values_a)
        sb = np.ones_like(values_b)
    else:
        sa = (hi - np.log10(np.maximum(values_a, eps))) / (hi - lo)
        sb = (hi - np.log10(np.maximum(values_b, eps))) / (hi - lo)
    return sa, sb, {'log10_min': lo, 'log10_max': hi}



def family_summary_from_scores(scores: np.ndarray) -> list[dict[str, Any]]:
    out = []
    for fam, a, b, color in FAMILY_SPANS:
        out.append({'family': fam, 'color': color, 'mean_score': float(np.mean(scores[a:b+1]))})
    return out



def try_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend([
            '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
            '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf',
        ])
    candidates.extend([
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
    ])
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()



def text(draw: ImageDraw.ImageDraw, xy, s: str, font, fill='#111827', anchor=None):
    draw.text(xy, s, font=font, fill=fill, anchor=anchor)



def paste_rotated_text(img: Image.Image, center_xy, s: str, font, fill='#111827', angle=90):
    probe = Image.new('RGBA', (10, 10), (255, 255, 255, 0))
    pd = ImageDraw.Draw(probe)
    bbox = pd.textbbox((0, 0), s, font=font)
    w = max(1, bbox[2] - bbox[0])
    h = max(1, bbox[3] - bbox[1])
    pad = 8
    tile = Image.new('RGBA', (w + pad * 2, h + pad * 2), (255, 255, 255, 0))
    td = ImageDraw.Draw(tile)
    td.text((pad, pad), s, font=font, fill=fill)
    rot = tile.rotate(angle, expand=1)
    x, y = center_xy
    img.alpha_composite(rot, (int(x - rot.width / 2), int(y - rot.height / 2)))



def draw_axes(draw, box, y_ticks=5, y_max=1.0):
    x0, y0, x1, y1 = box
    label_font = try_font(18)
    draw.rectangle((x0, y0, x1, y1), fill='#ffffff', outline='#d1d5db', width=1)
    for i in range(y_ticks + 1):
        frac = i / y_ticks
        yy = y1 - (y1 - y0) * frac
        draw.line((x0, yy, x1, yy), fill='#e5e7eb', width=1)
        text(draw, (x0 - 12, yy), f'{frac * y_max:.1f}', label_font, '#6b7280', anchor='rm')
    draw.line((x0, y0, x0, y1), fill='#111827', width=2)
    draw.line((x0, y1, x1, y1), fill='#111827', width=2)



def make_full_png(payload: dict[str, Any]):
    width, height = 2600, 1520
    img = Image.new('RGBA', (width, height), '#ffffff')
    draw = ImageDraw.Draw(img)

    title_font = try_font(42, True)
    sub_font = try_font(24)
    small_font = try_font(18)
    label_font = try_font(13)

    baseline = payload['cases']['baseline']
    small_gm = payload['cases']['small_gm']

    text(draw, (60, 44), 'No-Markov + scale-states alignment — dynamic-noise observability proxy', title_font, '#111827')
    text(draw, (60, 96), 'Markov states removed, but diagonal scale-factor states kg/ka are retained. Score is a practical recoverability proxy under actual noisy DAR runs.', sub_font, '#374151')

    # Panel A
    px0, py0, px1, py1 = 110, 180, 2490, 940
    draw_axes(draw, (px0, py0, px1, py1))
    text(draw, (px0, py0 - 26), 'A. State-wise normalized score under small_gm (with kg/ka included)', try_font(30, True), '#111827')
    g_w = (px1 - px0) / len(STATE_LABELS)
    bar_w = max(18, int(g_w * 0.56))
    for fam, a, b, color in FAMILY_SPANS:
        sx0 = px0 + g_w * a
        sx1 = px0 + g_w * (b + 1)
        if a > 0:
            draw.line((sx0, py0, sx0, py1), fill='#cbd5e1', width=2)
        text(draw, ((sx0 + sx1) / 2, py0 - 6), fam, small_font, '#4b5563', anchor='mb')
    for i, row in enumerate(small_gm['state_rows']):
        cx = px0 + g_w * (i + 0.5)
        val = row['normalized_score']
        fam = next(f for f in FAMILY_SPANS if f[1] <= i <= f[2])
        color = fam[3]
        bh = (py1 - py0) * max(0.0, min(1.0, float(val)))
        bx0 = cx - bar_w / 2
        by0 = py1 - bh
        draw.rounded_rectangle((bx0, by0, bx0 + bar_w, py1), radius=3, fill=color, outline='#ffffff', width=1)
        text(draw, (cx, py1 + 16), STATE_LABELS[i], label_font, '#374151', anchor='ma')
    paste_rotated_text(img, (46, (py0 + py1) / 2), 'normalized practical observability / recoverability score', try_font(22, True), '#111827', angle=90)

    # Panel B
    fx0, fy0, fx1, fy1 = 110, 1060, 1720, 1410
    draw_axes(draw, (fx0, fy0, fx1, fy1))
    text(draw, (fx0, fy0 - 26), 'B. Family-mean score: baseline vs small_gm (with kg/ka)', try_font(30, True), '#111827')
    fam_labels = [row['family'] for row in baseline['family_summary']]
    base_map = {row['family']: row for row in baseline['family_summary']}
    gm_map = {row['family']: row for row in small_gm['family_summary']}
    g2 = (fx1 - fx0) / len(fam_labels)
    total_w = max(32, int(g2 * 0.72))
    bw = total_w / 2 - 4
    for i, fam in enumerate(fam_labels):
        cx = fx0 + g2 * (i + 0.5)
        for j, (name, val, color) in enumerate([
            ('baseline', base_map[fam]['mean_score'], '#2563eb'),
            ('small_gm', gm_map[fam]['mean_score'], '#f97316'),
        ]):
            bh = (fy1 - fy0) * max(0.0, min(1.0, float(val)))
            bx0 = cx - total_w / 2 + j * (bw + 8)
            by0 = fy1 - bh
            draw.rounded_rectangle((bx0, by0, bx0 + bw, fy1), radius=4, fill=color)
        text(draw, (cx, fy1 + 18), fam, try_font(16), '#374151', anchor='ma')
    paste_rotated_text(img, (46, (fy0 + fy1) / 2), 'mean score', try_font(22, True), '#111827', angle=90)
    lx, ly = 1240, 1010
    draw.rounded_rectangle((lx, ly, 1702, ly + 54), radius=12, fill='#ffffff', outline='#d1d5db')
    draw.rectangle((lx + 18, ly + 18, lx + 36, ly + 36), fill='#2563eb')
    text(draw, (lx + 46, ly + 27), 'baseline', try_font(18), '#374151', anchor='lm')
    draw.rectangle((lx + 198, ly + 18, lx + 216, ly + 36), fill='#f97316')
    text(draw, (lx + 226, ly + 27), 'small_gm', try_font(18), '#374151', anchor='lm')

    # Panel C notes
    nx0, ny0, nx1, ny1 = 1810, 1060, 2490, 1410
    draw.rounded_rectangle((nx0, ny0, nx1, ny1), radius=16, fill='#ffffff', outline='#d1d5db', width=1)
    text(draw, (nx0 + 18, ny0 + 18), 'C. Read-out', try_font(28, True), '#111827')
    lines = [
        f"small_gm gyro sigma = {small_gm['profile_desc']['gyro_sigma_dph']} dph",
        f"small_gm accel sigma = {small_gm['profile_desc']['accel_sigma_ug']} ug",
        f"tau_g = {small_gm['profile_desc']['tau_g_s']} s",
        f"tau_a = {small_gm['profile_desc']['tau_a_s']} s",
        '',
        f"top family under small_gm: {small_gm['top_family']}",
        f"bottom family under small_gm: {small_gm['bottom_family']}",
        '',
        'This version keeps scale-factor states',
        'kg/ka but still removes Markov states.',
        'So it answers: if scale states are present',
        'yet dynamic noise is still unmodeled, where',
        'does the practical recoverability remain?',
    ]
    y = ny0 + 64
    for line in lines:
        if line == '':
            y += 14
            continue
        text(draw, (nx0 + 18, y), line, try_font(20), '#374151')
        y += 28

    text(draw, (60, 1470), 'Higher score means smaller final state-recovery error relative to the corresponding initialization scale.', small_font, '#6b7280')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.convert('RGB').save(OUT_FULL, quality=95)
    # crop panels for direct sending
    img.crop((20, 130, 2580, 980)).convert('RGB').save(OUT_A, quality=95)
    img.crop((20, 980, 1750, 1450)).convert('RGB').save(OUT_B, quality=95)



def build_markdown(payload: dict[str, Any]) -> str:
    base = payload['cases']['baseline']
    gm = payload['cases']['small_gm']
    lines = [
        '# No-Markov + scale-states dynamic-noise observability / recoverability proxy (2026-04-08)',
        '',
        '- **Model**: DAR alignment with scale-factor states retained, Markov states removed.',
        '- **States**: phi / dV / eb / db / kg / ka.',
        '- **Path**: current Chapter-4 DAR path.',
        '- **Seeds**: ' + str(SEEDS),
        '- **Cases**: baseline vs small_gm dynamic noise.',
        '',
        '## Family means',
        '',
        '| family | baseline mean score | small_gm mean score |',
        '|---|---:|---:|',
    ]
    base_map = {row['family']: row for row in base['family_summary']}
    gm_map = {row['family']: row for row in gm['family_summary']}
    for fam in [x[0] for x in FAMILY_SPANS]:
        lines.append(f"| {fam} | {base_map[fam]['mean_score']:.3f} | {gm_map[fam]['mean_score']:.3f} |")
    lines += [
        '',
        '## Small_gm per-state scores',
        '',
        '| state | score | mean error ratio | mean display error |',
        '|---|---:|---:|---|',
    ]
    for row in gm['state_rows']:
        lines.append(f"| {row['state']} | {row['normalized_score']:.3f} | {row['mean_error_ratio']:.3e} | {row['display_error_text']} |")
    lines += [
        '',
        f"- strongest family under small_gm: **{gm['top_family']}**",
        f"- weakest family under small_gm: **{gm['bottom_family']}**",
        '',
        f'- full: `{OUT_FULL}`',
        f'- panel A: `{OUT_A}`',
        f'- panel B: `{OUT_B}`',
        f'- json: `{OUT_JSON}`',
        f'- md: `{OUT_MD}`',
    ]
    return '\n'.join(lines) + '\n'



def main():
    t0 = time.time()
    base12 = load_base12()
    scales = build_initial_scales(base12)

    runs: list[dict[str, Any]] = []
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for profile in PROFILES:
        rows = [run_single(profile, seed) for seed in SEEDS]
        by_profile[profile] = rows
        runs.extend(rows)

    base_err = np.mean(np.array([r['state_abs_errors_native'] for r in by_profile['baseline']], dtype=float), axis=0)
    gm_err = np.mean(np.array([r['state_abs_errors_native'] for r in by_profile['small_gm']], dtype=float), axis=0)
    base_ratio = base_err / scales
    gm_ratio = gm_err / scales
    base_score, gm_score, meta = normalize_inverse_log(base_ratio, gm_ratio)

    cases = {}
    for profile, ratio, score, mean_err in [
        ('baseline', base_ratio, base_score, base_err),
        ('small_gm', gm_ratio, gm_score, gm_err),
    ]:
        state_rows = []
        for i, state in enumerate(STATE_LABELS):
            if i < 3:
                display = f"{mean_err[i] / base12.glv.sec:.3f}\""
            elif i < 6:
                display = f"{mean_err[i]:.6f} m/s"
            elif i < 9:
                display = f"{mean_err[i] / base12.glv.dph:.6f} dph"
            elif i < 12:
                display = f"{mean_err[i] / base12.glv.ug:.3f} ug"
            else:
                display = f"{mean_err[i] / base12.glv.ppm:.3f} ppm"
            state_rows.append({
                'state': state,
                'mean_error_native': float(mean_err[i]),
                'mean_error_ratio': float(ratio[i]),
                'normalized_score': float(score[i]),
                'display_error_text': display,
            })
        fam_summary = family_summary_from_scores(score)
        top_family = max(fam_summary, key=lambda x: x['mean_score'])['family']
        bottom_family = min(fam_summary, key=lambda x: x['mean_score'])['family']
        cases[profile] = {
            'profile_desc': by_profile[profile][0]['profile_desc'],
            'state_rows': state_rows,
            'family_summary': fam_summary,
            'top_family': top_family,
            'bottom_family': bottom_family,
        }

    payload = {
        'task': 'ch4_nomarkov_scale_dynamic_noise_observability_proxy',
        'method_note': 'Practical recoverability score under actual noisy DAR runs, with scale-factor states retained but Markov states removed.',
        'profiles': PROFILES,
        'seeds': SEEDS,
        'state_labels': STATE_LABELS,
        'normalization_meta': meta,
        'initial_scales_native': [float(x) for x in scales],
        'cases': cases,
        'raw_runs': runs,
        'artifacts': {
            'full_png': str(OUT_FULL),
            'panel_a_png': str(OUT_A),
            'panel_b_png': str(OUT_B),
            'json': str(OUT_JSON),
            'md': str(OUT_MD),
        },
        'runtime_s': round(time.time() - t0, 3),
    }

    make_full_png(payload)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    OUT_MD.write_text(build_markdown(payload), encoding='utf-8')
    print(json.dumps({
        'panel_a': str(OUT_A),
        'panel_b': str(OUT_B),
        'small_gm_top_family': cases['small_gm']['top_family'],
        'small_gm_bottom_family': cases['small_gm']['bottom_family'],
        'baseline_top_family': cases['baseline']['top_family'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
