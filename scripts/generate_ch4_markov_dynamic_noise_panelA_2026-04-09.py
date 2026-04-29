#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORKSPACE = Path('/root/.openclaw/workspace')
SCRIPTS = WORKSPACE / 'scripts'
OUT_DIR = WORKSPACE / 'tmp' / 'ch4_markov_dynamic_noise_panelA_2026-04-09'
OUT_JSON = OUT_DIR / 'ch4_markov_dynamic_noise_panelA_2026-04-09.json'
OUT_MD = OUT_DIR / 'ch4_markov_dynamic_noise_panelA_2026-04-09.md'
OUT_PNG = OUT_DIR / 'fig_ch4_markov_dynamic_noise_panelA_2026-04-09.png'
HYBRID24_PATH = SCRIPTS / 'alignvn_dar_hybrid24_staged_py_2026-03-30.py'
GM_HELPER_PATH = SCRIPTS / 'alignvn_dar_truth_gm_helper_2026-03-31.py'
SEEDS = [0, 1, 2, 3, 4]
PROFILES = ['baseline', 'small_gm']
MAX_WORKERS = min(4, os.cpu_count() or 1)

STATE_LABELS = [
    'phi_E', 'phi_N', 'phi_U',
    'dV_E', 'dV_N', 'dV_U',
    'eb_x', 'eb_y', 'eb_z',
    'db_x', 'db_y', 'db_z',
    'ng_x', 'ng_y', 'ng_z',
    'xa_x', 'xa_y', 'xa_z',
    'kg_x', 'kg_y', 'kg_z',
    'ka_x', 'ka_y', 'ka_z',
]
FAMILY_SPANS = [
    ('phi', 0, 2, '#2563eb'),
    ('dV', 3, 5, '#0891b2'),
    ('eb', 6, 8, '#7c3aed'),
    ('db', 9, 11, '#ea580c'),
    ('ng', 12, 14, '#b91c1c'),
    ('xa', 15, 17, '#0f766e'),
    ('kg', 18, 20, '#92400e'),
    ('ka', 21, 23, '#0f766e'),
]


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod



def build_initial_scales(h24) -> np.ndarray:
    base12 = h24.load_base12()
    glv = h24.load_acc18().glv
    imuerr = base12.build_imuerr()
    phi0 = np.array([0.1, 0.1, 0.5]) * glv.deg
    dV0 = np.ones(3)
    init_eb = np.maximum(np.asarray(imuerr['eb']).reshape(3), 0.1 * glv.dph)
    init_db = np.maximum(np.asarray(imuerr['db']).reshape(3), 1000.0 * glv.ug)
    ng_sigma = np.array([0.05, 0.05, 0.05]) * glv.dph
    xa_sigma = np.maximum(np.array([0.01, 0.01, 0.01]) * glv.ug, 5.0 * glv.ug)
    scale0 = np.full(3, 100.0 * glv.ppm)
    return np.r_[phi0, dV0, init_eb, init_db, ng_sigma, xa_sigma, scale0, scale0]



def build_noisy_imu(base12, gm, profile: str, seed: int, imu_clean: np.ndarray):
    np.random.seed(seed)
    imuerr = gm.build_truth_imuerr_variant(profile=profile)
    imu_noisy = gm.apply_truth_imu_errors(imu_clean, imuerr)

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
        # apply_truth_imu_errors already injects these; here only recover final truth state.
        gyro_gm_last = gyro_bias[-1]
        accel_gm_last = accel_bias[-1]

    truth_eb = np.asarray(imuerr['eb']).reshape(3)
    truth_db = np.asarray(imuerr['db']).reshape(3)
    truth_ng = gyro_gm_last
    truth_xa = accel_gm_last
    truth_dkg = np.diag(np.asarray(imuerr['dKg']))
    truth_dka = np.diag(np.asarray(imuerr['dKa']))
    return imu_noisy, imuerr, truth_eb, truth_db, truth_ng, truth_xa, truth_dkg, truth_dka



def alignvn_24state_iter_with_state(h24, imu: np.ndarray, qnb: np.ndarray, pos: np.ndarray, phi0: np.ndarray,
                                    imuerr: dict[str, np.ndarray], wvn: np.ndarray, cfg, truth_att: np.ndarray):
    acc18 = h24.load_acc18()
    glv = acc18.glv

    imu_corr = imu.copy()
    nn = 2
    ts = float(imu_corr[1, -1] - imu_corr[0, -1])
    nts = nn * ts
    qnb_seed = acc18.a2qua(qnb) if len(qnb) == 3 else np.asarray(qnb).reshape(4)
    length = (len(imu_corr) // nn) * nn
    imu_corr = imu_corr[:length]

    eth = acc18.Earth(pos)
    cnn = acc18.rv2m(-eth.wnie * nts / 2)
    rot_gate_rad = cfg.rot_gate_dps * glv.deg

    final_att = None
    final_x = None
    final_vn = None
    for iteration in range(1, cfg.max_iter + 1):
        scale_active = (not cfg.staged_release) or (iteration >= cfg.release_iter)
        kf = h24.avnkfinit_24(
            nts, pos, phi0, imuerr, wvn,
            np.array(cfg.ng_sigma_dph) * glv.dph,
            np.array(cfg.tau_g_s),
            np.array(cfg.xa_sigma_ug) * glv.ug,
            np.array(cfg.tau_a_s),
            enable_scale_states=scale_active,
        )
        vn = np.zeros(3)
        qnbi = qnb_seed.copy()

        for k in range(0, length, nn):
            wvm = imu_corr[k:k + nn, 0:6]
            phim, dvbm = acc18.cnscl(wvm)
            cnb = acc18.q2mat(qnbi)
            dvn = cnn @ cnb @ dvbm
            vn = vn + dvn + eth.gn * nts
            qnbi = acc18.qupdt2(qnbi, phim, eth.wnin * nts)

            phi_k = kf['Phikk_1'].copy()
            cnbts = cnb * nts
            phi_k[3:6, 0:3] = acc18.askew(dvn)
            phi_k[3:6, 9:12] = cnbts
            phi_k[3:6, 15:18] = cnbts
            phi_k[0:3, 6:9] = -cnbts
            phi_k[0:3, 12:15] = -cnbts
            phi_k[12:15, 12:15] = np.diag(kf['fg'])
            phi_k[15:18, 15:18] = np.diag(kf['fa'])

            if scale_active:
                high_rot = np.max(np.abs(phim / nts)) > rot_gate_rad
                if high_rot:
                    phi_k[0:3, 18:21] = -cnb @ np.diag(phim[0:3])
                    phi_k[3:6, 21:24] = cnb @ np.diag(dvbm[0:3])
                else:
                    phi_k[0:3, 18:21] = 0.0
                    phi_k[3:6, 21:24] = 0.0
            else:
                phi_k[0:3, 18:21] = 0.0
                phi_k[3:6, 21:24] = 0.0

            kf['Phikk_1'] = phi_k
            kf = acc18.kfupdate(kf, vn)

            qnbi = acc18.qdelphi(qnbi, 0.91 * kf['xk'][0:3])
            kf['xk'][0:3] *= 0.09
            vn = vn - 0.91 * kf['xk'][3:6]
            kf['xk'][3:6] *= 0.09

            if not scale_active:
                kf['xk'][18:24] = 0.0

        final_att = acc18.q2att(qnbi)
        final_x = kf['xk'].copy()
        final_vn = vn.copy()

        if iteration < cfg.max_iter:
            if cfg.carry_att_seed:
                qnb_seed = qnbi.copy()
            imu_corr[:, 0:3] -= cfg.wash_scale * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= cfg.wash_scale * kf['xk'][9:12] * ts
            if scale_active and cfg.scale_wash_scale > 0.0:
                imu_corr = h24.apply_scale_wash(imu_corr, kf['xk'][18:21], kf['xk'][21:24], cfg.scale_wash_scale)

    assert final_att is not None and final_x is not None and final_vn is not None
    return final_att, final_vn, final_x



def run_single(task: tuple[str, int]) -> dict[str, Any]:
    profile, seed = task
    h24 = load_module(f'markov_panelA_h24_{os.getpid()}', HYBRID24_PATH)
    gm = load_module(f'markov_panelA_gm_{os.getpid()}', GM_HELPER_PATH)
    acc18 = h24.load_acc18()
    base12 = h24.load_base12()
    glv = acc18.glv

    ts = 0.01
    att0 = np.array([0.0, 0.0, 0.0])
    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    rot_paras = base12.build_rot_paras()
    att_truth = acc18.attrottt(att0, rot_paras, ts)
    imu_clean, _ = acc18.avp2imu(att_truth, pos0)

    imu_noisy, imuerr, truth_eb, truth_db, truth_ng, truth_xa, truth_dkg, truth_dka = build_noisy_imu(base12, gm, profile, seed, imu_clean)

    phi = np.array([0.1, 0.1, 0.5]) * glv.deg
    att0_guess = acc18.q2att(base12.qaddphi(acc18.a2qua(att0), phi))
    wvn = np.array([0.01, 0.01, 0.01])
    truth_att = att_truth[-1, 0:3]

    cfg = h24.Hybrid24Config(
        name='plain24_markov_with_scale_dynamic_noise',
        label='plain24 / Markov + scale',
        seeds=SEEDS,
        max_iter=5,
        wash_scale=0.5,
        scale_wash_scale=0.5,
        carry_att_seed=True,
        staged_release=False,
        release_iter=1,
        rot_gate_dps=5.0,
        note='Markov states ng/xa + scale states kg/ka all kept in the model; no staged release.',
    )

    final_att, final_vn, final_x = alignvn_24state_iter_with_state(
        h24=h24,
        imu=imu_noisy.copy(),
        qnb=att0_guess,
        pos=pos0,
        phi0=phi,
        imuerr=imuerr,
        wvn=wvn,
        cfg=cfg,
        truth_att=truth_att,
    )

    phi_err = acc18.qq2phi(acc18.a2qua(final_att), acc18.a2qua(truth_att))
    dV_err = final_vn.copy()
    eb_err = final_x[6:9] - truth_eb
    db_err = final_x[9:12] - truth_db
    ng_err = final_x[12:15] - truth_ng
    xa_err = final_x[15:18] - truth_xa
    kg_err = final_x[18:21] - truth_dkg
    ka_err = final_x[21:24] - truth_dka
    err_vec = np.r_[phi_err, dV_err, eb_err, db_err, ng_err, xa_err, kg_err, ka_err]

    return {
        'profile': profile,
        'seed': seed,
        'profile_desc': gm.describe_truth_profile(profile),
        'state_abs_errors_native': [float(abs(x)) for x in err_vec],
        'display': {
            'phi_arcsec': [float(abs(x / glv.sec)) for x in phi_err],
            'dV_mps': [float(abs(x)) for x in dV_err],
            'eb_dph': [float(abs(x / glv.dph)) for x in eb_err],
            'db_ug': [float(abs(x / glv.ug)) for x in db_err],
            'ng_dph': [float(abs(x / glv.dph)) for x in ng_err],
            'xa_ug': [float(abs(x / glv.ug)) for x in xa_err],
            'kg_ppm': [float(abs(x / glv.ppm)) for x in kg_err],
            'ka_ppm': [float(abs(x / glv.ppm)) for x in ka_err],
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



def make_png(payload: dict[str, Any]):
    width, height = 2650, 1080
    img = Image.new('RGBA', (width, height), '#ffffff')
    draw = ImageDraw.Draw(img)
    title_font = try_font(42, True)
    sub_font = try_font(24)
    small_font = try_font(18)
    label_font = try_font(13)

    case = payload['cases']['small_gm']
    text(draw, (60, 44), 'Markov + scale-states alignment — dynamic-noise A-panel', title_font, '#111827')
    text(draw, (60, 96), 'Model keeps ng/xa Markov states and kg/ka scale states. Score is a practical recoverability proxy under the actual small_gm DAR runs.', sub_font, '#374151')

    px0, py0, px1, py1 = 110, 180, 2540, 920
    draw_axes(draw, (px0, py0, px1, py1))
    text(draw, (px0, py0 - 26), 'A. State-wise normalized score under small_gm (Markov + kg/ka included)', try_font(30, True), '#111827')
    g_w = (px1 - px0) / len(STATE_LABELS)
    bar_w = max(18, int(g_w * 0.56))
    for fam, a, b, color in FAMILY_SPANS:
        sx0 = px0 + g_w * a
        sx1 = px0 + g_w * (b + 1)
        if a > 0:
            draw.line((sx0, py0, sx0, py1), fill='#cbd5e1', width=2)
        text(draw, ((sx0 + sx1) / 2, py0 - 6), fam, small_font, '#4b5563', anchor='mb')
    for i, row in enumerate(case['state_rows']):
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

    text(draw, (60, 1015), f"small_gm strongest family: {case['top_family']}   |   weakest family: {case['bottom_family']}", small_font, '#374151')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.convert('RGB').save(OUT_PNG, quality=95)



def build_markdown(payload: dict[str, Any]) -> str:
    case = payload['cases']['small_gm']
    lines = [
        '# Markov + scale-states dynamic-noise A-panel (2026-04-09)',
        '',
        '- **Model**: plain24 / Markov with kg/ka retained, no staged release.',
        '- **Path**: current Chapter-4 DAR path.',
        '- **Condition**: small_gm dynamic noise.',
        '- **Seeds**: ' + str(SEEDS),
        '',
        '| state | score | mean error ratio | mean display error |',
        '|---|---:|---:|---|',
    ]
    for row in case['state_rows']:
        lines.append(f"| {row['state']} | {row['normalized_score']:.3f} | {row['mean_error_ratio']:.3e} | {row['display_error_text']} |")
    lines += [
        '',
        f"- strongest family: **{case['top_family']}**",
        f"- weakest family: **{case['bottom_family']}**",
        f'- png: `{OUT_PNG}`',
        f'- json: `{OUT_JSON}`',
        f'- md: `{OUT_MD}`',
    ]
    return '\n'.join(lines) + '\n'



def main():
    t0 = time.time()
    h24 = load_module('markov_panelA_h24_main', HYBRID24_PATH)
    scales = build_initial_scales(h24)

    tasks = [(profile, seed) for profile in PROFILES for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as ex:
        runs = list(ex.map(run_single, tasks))

    by_profile: dict[str, list[dict[str, Any]]] = {p: [] for p in PROFILES}
    for row in runs:
        by_profile[row['profile']].append(row)
    for p in PROFILES:
        by_profile[p].sort(key=lambda r: r['seed'])

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
                display = f"{mean_err[i] / h24.load_acc18().glv.sec:.3f}\""
            elif i < 6:
                display = f"{mean_err[i]:.6f} m/s"
            elif i < 9:
                display = f"{mean_err[i] / h24.load_acc18().glv.dph:.6f} dph"
            elif i < 12:
                display = f"{mean_err[i] / h24.load_acc18().glv.ug:.3f} ug"
            elif i < 15:
                display = f"{mean_err[i] / h24.load_acc18().glv.dph:.6f} dph"
            elif i < 18:
                display = f"{mean_err[i] / h24.load_acc18().glv.ug:.3f} ug"
            elif i < 21:
                display = f"{mean_err[i] / h24.load_acc18().glv.ppm:.3f} ppm"
            else:
                display = f"{mean_err[i] / h24.load_acc18().glv.ppm:.3f} ppm"
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
        'task': 'ch4_markov_dynamic_noise_panelA',
        'method_note': 'Practical recoverability score under actual noisy DAR runs, with Markov states ng/xa and scale states kg/ka retained.',
        'profiles': PROFILES,
        'seeds': SEEDS,
        'state_labels': STATE_LABELS,
        'normalization_meta': meta,
        'initial_scales_native': [float(x) for x in scales],
        'cases': cases,
        'raw_runs': runs,
        'artifacts': {'png': str(OUT_PNG), 'json': str(OUT_JSON), 'md': str(OUT_MD)},
        'runtime_s': round(time.time() - t0, 3),
    }

    make_png(payload)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    OUT_MD.write_text(build_markdown(payload), encoding='utf-8')
    print(json.dumps({
        'png': str(OUT_PNG),
        'small_gm_top_family': cases['small_gm']['top_family'],
        'small_gm_bottom_family': cases['small_gm']['bottom_family'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
