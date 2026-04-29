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
OUT_DIR = WORKSPACE / 'tmp' / 'ch4_nomarkov_dynamic_noise_observability_2026-04-08'
OUT_JSON = OUT_DIR / 'ch4_nomarkov_dynamic_noise_observability_2026-04-08.json'
OUT_MD = OUT_DIR / 'ch4_nomarkov_dynamic_noise_observability_2026-04-08.md'
OUT_PNG = OUT_DIR / 'fig_ch4_nomarkov_dynamic_noise_observability_2026-04-08.png'
BASE12_PATH = SCRIPTS / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
GM_HELPER_PATH = SCRIPTS / 'alignvn_dar_truth_gm_helper_2026-03-31.py'
SEEDS = [0, 1, 2, 3, 4]
PROFILES = ['baseline', 'small_gm']
STATE_LABELS = ['phi_E', 'phi_N', 'phi_U', 'dV_E', 'dV_N', 'dV_U', 'eb_x', 'eb_y', 'eb_z', 'db_x', 'db_y', 'db_z']
FAMILY_SPANS = [
    ('phi', 0, 2, '#2563eb'),
    ('dV', 3, 5, '#0891b2'),
    ('eb', 6, 8, '#7c3aed'),
    ('db', 9, 11, '#ea580c'),
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
        _base12 = load_module('nomarkov_obs_base12_20260408', BASE12_PATH)
    return _base12



def load_gm():
    global _gm
    if _gm is None:
        _gm = load_module('nomarkov_obs_gmhelper_20260408', GM_HELPER_PATH)
    return _gm



def build_initial_scales(base12) -> np.ndarray:
    imuerr = base12.build_imuerr()
    phi0 = np.array([0.1, 0.1, 0.5]) * base12.glv.deg
    dV0 = np.ones(3)
    init_eb = np.maximum(np.asarray(imuerr['eb']).reshape(3), 0.1 * base12.glv.dph)
    init_db = np.maximum(np.asarray(imuerr['db']).reshape(3), 1000.0 * base12.glv.ug)
    return np.r_[phi0, dV0, init_eb, init_db]



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
    return imu_noisy, imuerr, truth_final_eb, truth_final_db



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

    imu_noisy, imuerr, truth_final_eb, truth_final_db = build_noisy_imu(profile, seed, imu_clean)

    phi = np.array([0.1, 0.1, 0.5]) * base12.glv.deg
    att0_guess = base12.q2att(base12.qaddphi(base12.a2qua(att0), phi))
    wvn = np.array([0.01, 0.01, 0.01])
    truth_att = att_truth[-1, 0:3]

    _att_aligned, attk, xkpk, _iter_logs = base12.alignvn_12state_iter(
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
    final_x = xkpk[-1, 0:12]
    final_p_diag = xkpk[-1, 12:24]

    phi_err = base12.qq2phi(base12.a2qua(final_att), base12.a2qua(truth_att))
    dV_err = final_vn.copy()  # final truth velocity is zero
    eb_err = final_x[6:9] - truth_final_eb
    db_err = final_x[9:12] - truth_final_db

    err_vec = np.r_[phi_err, dV_err, eb_err, db_err]
    return {
        'profile': profile,
        'seed': seed,
        'profile_desc': gm.describe_truth_profile(profile),
        'state_abs_errors_native': [float(abs(x)) for x in err_vec],
        'state_abs_errors_display': {
            'phi_arcsec': [float(abs(x / base12.glv.sec)) for x in phi_err],
            'dV_mps': [float(abs(x)) for x in dV_err],
            'eb_dph': [float(abs(x / base12.glv.dph)) for x in eb_err],
            'db_ug': [float(abs(x / base12.glv.ug)) for x in db_err],
        },
        'final_p_diag': [float(x) for x in final_p_diag],
        'final_sigma_native': [float(np.sqrt(max(x, 0.0))) for x in final_p_diag],
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
        out.append({
            'family': fam,
            'color': color,
            'mean_score': float(np.mean(scores[a:b+1])),
        })
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
    width, height = 2200, 1480
    img = Image.new('RGBA', (width, height), '#ffffff')
    draw = ImageDraw.Draw(img)

    title_font = try_font(42, True)
    sub_font = try_font(24)
    small_font = try_font(18)
    label_font = try_font(13)

    baseline = payload['cases']['baseline']
    small_gm = payload['cases']['small_gm']

    text(draw, (60, 44), 'No-Markov (12-state) alignment — dynamic-noise observability proxy', title_font, '#111827')
    text(draw, (60, 96), 'Formal Gramian itself does not directly encode truth-side GM drift, so this figure uses a practical recoverability score under the actual noisy DAR runs.', sub_font, '#374151')

    # Panel A: state-wise dynamic-noise score
    px0, py0, px1, py1 = 110, 180, 2130, 910
    draw_axes(draw, (px0, py0, px1, py1))
    text(draw, (px0, py0 - 26), 'A. State-wise normalized score under small_gm dynamic-noise case', try_font(30, True), '#111827')
    g_w = (px1 - px0) / len(STATE_LABELS)
    bar_w = max(22, int(g_w * 0.56))
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

    # Panel B: family means comparison
    fx0, fy0, fx1, fy1 = 110, 1030, 1450, 1360
    draw_axes(draw, (fx0, fy0, fx1, fy1))
    text(draw, (fx0, fy0 - 26), 'B. Family-mean score: baseline vs small_gm', try_font(30, True), '#111827')
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
    lx, ly = 1040, 988
    draw.rounded_rectangle((lx, ly, 1432, ly + 54), radius=12, fill='#ffffff', outline='#d1d5db')
    draw.rectangle((lx + 18, ly + 18, lx + 36, ly + 36), fill='#2563eb')
    text(draw, (lx + 46, ly + 27), 'baseline', try_font(18), '#374151', anchor='lm')
    draw.rectangle((lx + 158, ly + 18, lx + 176, ly + 36), fill='#f97316')
    text(draw, (lx + 186, ly + 27), 'small_gm', try_font(18), '#374151', anchor='lm')

    # Panel C: notes
    nx0, ny0, nx1, ny1 = 1530, 1030, 2130, 1360
    draw.rounded_rectangle((nx0, ny0, nx1, ny1), radius=16, fill='#ffffff', outline='#d1d5db', width=1)
    text(draw, (nx0 + 18, ny0 + 18), 'C. Read-out', try_font(28, True), '#111827')
    lines = [
        f"small_gm gyro sigma = {small_gm['profile_desc']['gyro_sigma_dph']} dph",
        f"small_gm accel sigma = {small_gm['profile_desc']['accel_sigma_ug']} ug",
        f"tau_g = {small_gm['profile_desc']['tau_g_s']} s",
        f"tau_a = {small_gm['profile_desc']['tau_a_s']} s",
        '',
        f"strongest family under small_gm: {small_gm['top_family']}",
        f"weakest family under small_gm: {small_gm['bottom_family']}",
        '',
        'Important caveat:',
        'this is not a pure formal Gramian view;',
        'it is a dynamic-noise practical score built',
        'from state-recovery error ratios over the',
        'actual no-Markov DAR runs.',
    ]
    y = ny0 + 64
    for line in lines:
        if line == '':
            y += 14
            continue
        text(draw, (nx0 + 18, y), line, try_font(20), '#374151')
        y += 28

    text(draw, (60, 1430), 'Higher score means smaller final state-recovery error relative to the corresponding initialization scale.', small_font, '#6b7280')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.convert('RGB').save(OUT_PNG, quality=95)



def build_markdown(payload: dict[str, Any]) -> str:
    base = payload['cases']['baseline']
    gm = payload['cases']['small_gm']
    lines = [
        '# No-Markov dynamic-noise observability / recoverability proxy (2026-04-08)',
        '',
        '- **Model**: 12-state DAR alignment, no Markov states.',
        '- **Path**: current Chapter-4 DAR path.',
        '- **Seeds**: ' + str(SEEDS),
        '- **Cases**: baseline vs small_gm dynamic noise.',
        '- **Important note**: because truth-side GM drift does not directly modify the model-side formal Gramian, this figure uses a practical state-recovery score rather than a strict pure-formal observability score.',
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
        f'- png: `{OUT_PNG}`',
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
            else:
                display = f"{mean_err[i] / base12.glv.ug:.3f} ug"
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
        'task': 'ch4_nomarkov_dynamic_noise_observability_proxy',
        'method_note': 'Practical recoverability score under actual noisy no-Markov DAR runs; not a pure formal Gramian because truth-side dynamic GM drift is not part of the 12-state model.',
        'profiles': PROFILES,
        'seeds': SEEDS,
        'state_labels': STATE_LABELS,
        'normalization_meta': meta,
        'initial_scales_native': [float(x) for x in scales],
        'cases': cases,
        'raw_runs': runs,
        'artifacts': {
            'png': str(OUT_PNG),
            'json': str(OUT_JSON),
            'md': str(OUT_MD),
        },
        'runtime_s': round(time.time() - t0, 3),
    }

    make_png(payload)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    OUT_MD.write_text(build_markdown(payload), encoding='utf-8')
    print(json.dumps({
        'png': str(OUT_PNG),
        'baseline_top_family': cases['baseline']['top_family'],
        'small_gm_top_family': cases['small_gm']['top_family'],
        'small_gm_bottom_family': cases['small_gm']['bottom_family'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
