#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORKSPACE = Path('/root/.openclaw/workspace')
SCRIPTS = WORKSPACE / 'scripts'
OUT_DIR = WORKSPACE / 'tmp' / 'ch4_nomarkov_dynamic_noise_compare_2026-04-08'
OUT_JSON = OUT_DIR / 'ch4_nomarkov_dynamic_noise_compare_2026-04-08.json'
OUT_MD = OUT_DIR / 'ch4_nomarkov_dynamic_noise_compare_2026-04-08.md'
OUT_FIG = OUT_DIR / 'fig_ch4_nomarkov_dynamic_noise_compare_2026-04-08.png'
BASE12_PATH = SCRIPTS / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
GM_HELPER_PATH = SCRIPTS / 'alignvn_dar_truth_gm_helper_2026-03-31.py'
SEEDS = [0, 1, 2, 3, 4]
COMPARE_PROFILES = ['baseline', 'small_gm']

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
        _base12 = load_module('alignvn_dar_12state_iterfix_cmp_20260408', BASE12_PATH)
    return _base12



def load_gm_helper():
    global _gm
    if _gm is None:
        _gm = load_module('alignvn_truth_gm_helper_cmp_20260408', GM_HELPER_PATH)
    return _gm



def run_single(seed: int, profile: str) -> dict[str, Any]:
    base12 = load_base12()
    gm = load_gm_helper()

    np.random.seed(seed)
    ts = 0.01
    max_iter = 5
    wash_scale = 0.5
    carry_att_seed = True

    att0 = np.array([0.0, 0.0, 0.0])
    pos0 = base12.posset(34, 116, 480, isdeg=1)
    rot_paras = base12.build_rot_paras()
    att_truth = base12.attrottt(att0, rot_paras, ts)
    imu_clean, _ = base12.avp2imu(att_truth, pos0)

    imuerr = gm.build_truth_imuerr_variant(profile=profile)
    imu_noisy = gm.apply_truth_imu_errors(imu_clean, imuerr)

    phi = np.array([0.1, 0.1, 0.5]) * base12.glv.deg
    att0_guess = base12.q2att(base12.qaddphi(base12.a2qua(att0), phi))
    wvn = np.array([0.01, 0.01, 0.01])
    truth_att = att_truth[-1, 0:3]

    att_aligned, _attk, _xkpk, iter_logs = base12.alignvn_12state_iter(
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

    final_att_err_arcsec = base12.qq2phi(base12.a2qua(att_aligned), base12.a2qua(truth_att)) / base12.glv.sec
    final_abs = np.abs(final_att_err_arcsec)
    return {
        'seed': seed,
        'profile': profile,
        'profile_desc': gm.describe_truth_profile(profile),
        'final_att_err_arcsec': [float(x) for x in final_att_err_arcsec],
        'final_att_err_abs_arcsec': [float(x) for x in final_abs],
        'final_pitch_abs_arcsec': float(final_abs[1]),
        'final_yaw_abs_arcsec': float(final_abs[2]),
        'final_norm_arcsec': float(np.linalg.norm(final_att_err_arcsec)),
        'iter_logs': [
            {
                'iteration': int(item.iteration),
                'att_err_arcsec': [float(x) for x in item.att_err_arcsec],
                'att_err_norm_arcsec': float(item.att_err_norm_arcsec),
                'yaw_abs_arcsec': float(item.yaw_abs_arcsec),
            }
            for item in iter_logs
        ],
    }



def summarize(profile: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    pitch = np.array([r['final_pitch_abs_arcsec'] for r in rows], dtype=float)
    yaw = np.array([r['final_yaw_abs_arcsec'] for r in rows], dtype=float)
    norm = np.array([r['final_norm_arcsec'] for r in rows], dtype=float)
    n_iter = len(rows[0]['iter_logs'])
    iter_mean_yaw = []
    iter_mean_norm = []
    for i in range(n_iter):
        iter_mean_yaw.append(float(np.mean([r['iter_logs'][i]['yaw_abs_arcsec'] for r in rows])))
        iter_mean_norm.append(float(np.mean([r['iter_logs'][i]['att_err_norm_arcsec'] for r in rows])))
    return {
        'profile': profile,
        'profile_desc': rows[0]['profile_desc'],
        'pitch_mean_abs_arcsec': float(np.mean(pitch)),
        'yaw_abs_mean_arcsec': float(np.mean(yaw)),
        'norm_mean_arcsec': float(np.mean(norm)),
        'yaw_abs_median_arcsec': float(np.median(yaw)),
        'yaw_abs_max_arcsec': float(np.max(yaw)),
        'per_seed_final_yaw_abs_arcsec': [float(x) for x in yaw],
        'per_seed_final_norm_arcsec': [float(x) for x in norm],
        'iter_mean_yaw_abs_arcsec': iter_mean_yaw,
        'iter_mean_norm_arcsec': iter_mean_norm,
    }



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



def draw_axes(draw: ImageDraw.ImageDraw, box, n_ticks=5, y_max=1.0, tick_fmt='{:.0f}', title=None, y_label=None):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill='white', outline='#d1d5db', width=2)
    pad_l, pad_r, pad_t, pad_b = 78, 26, 46, 62
    px0 = x0 + pad_l
    py0 = y0 + pad_t
    px1 = x1 - pad_r
    py1 = y1 - pad_b
    draw.line((px0, py0, px0, py1), fill='#111827', width=2)
    draw.line((px0, py1, px1, py1), fill='#111827', width=2)
    sub_font = try_font(22, False)
    small_font = try_font(18, False)
    if title:
        text(draw, (x0 + 18, y0 + 16), title, try_font(26, True))
    if y_label:
        text(draw, (x0 + 18, y0 + 48), y_label, small_font, fill='#4b5563')
    for i in range(n_ticks + 1):
        frac = i / n_ticks
        y = py1 - frac * (py1 - py0)
        draw.line((px0, y, px1, y), fill='#e5e7eb', width=1)
        label = tick_fmt.format(y_max * frac)
        text(draw, (px0 - 10, y), label, small_font, fill='#6b7280', anchor='rm')
    return px0, py0, px1, py1



def make_figure(payload: dict[str, Any]) -> None:
    width, height = 1800, 1220
    img = Image.new('RGB', (width, height), '#f8fafc')
    draw = ImageDraw.Draw(img)
    title_font = try_font(36, True)
    sub_font = try_font(22, False)
    body_font = try_font(20, False)
    small_font = try_font(18, False)

    baseline = payload['summaries']['baseline']
    noisy = payload['summaries']['small_gm']
    blue = '#2563eb'
    orange = '#f97316'

    text(draw, (46, 52), 'No-Markov (12-state) DAR alignment: before vs after dynamic noise', title_font)
    sub1 = 'Same dual-axis DAR path, same 12-state iterfix aligner, same seeds [0,1,2,3,4].'
    sub2 = 'Dynamic-noise condition uses truth-side small_gm drift only; filter model itself still has no Markov states.'
    text(draw, (46, 98), sub1, sub_font, fill='#374151')
    text(draw, (46, 130), sub2, sub_font, fill='#374151')

    # Panel A: per-seed final yaw
    axA = (40, 180, 880, 680)
    px0, py0, px1, py1 = draw_axes(
        draw, axA, n_ticks=6,
        y_max=max(max(baseline['per_seed_final_yaw_abs_arcsec']), max(noisy['per_seed_final_yaw_abs_arcsec'])) * 1.15,
        tick_fmt='{:.0f}',
        title='A. Final |yaw| per seed',
        y_label='arcsec'
    )
    y_max = max(max(baseline['per_seed_final_yaw_abs_arcsec']), max(noisy['per_seed_final_yaw_abs_arcsec'])) * 1.15
    group_w = (px1 - px0) / len(SEEDS)
    inner_w = group_w * 0.62
    bar_w = inner_w / 2 - 6
    for j, seed in enumerate(SEEDS):
        gx = px0 + j * group_w + (group_w - inner_w) / 2
        vals = [baseline['per_seed_final_yaw_abs_arcsec'][j], noisy['per_seed_final_yaw_abs_arcsec'][j]]
        cols = [blue, orange]
        for k, val in enumerate(vals):
            bx = gx + k * (bar_w + 12)
            bh = (val / y_max) * (py1 - py0)
            by = py1 - bh
            draw.rounded_rectangle((bx, by, bx + bar_w, py1), radius=6, fill=cols[k], outline=cols[k])
            text(draw, (bx + bar_w / 2, by - 8), f'{val:.1f}', small_font, fill='#374151', anchor='mb')
        text(draw, (gx + inner_w / 2, py1 + 28), f'seed {seed}', body_font, fill='#374151', anchor='mm')
    legend_y = axA[3] - 24
    draw.rounded_rectangle((axA[0] + 20, legend_y - 14, axA[0] + 38, legend_y + 4), radius=4, fill=blue)
    text(draw, (axA[0] + 48, legend_y - 6), 'baseline', body_font, fill='#374151')
    draw.rounded_rectangle((axA[0] + 190, legend_y - 14, axA[0] + 208, legend_y + 4), radius=4, fill=orange)
    text(draw, (axA[0] + 218, legend_y - 6), 'small_gm', body_font, fill='#374151')

    # Panel B: aggregate metrics
    axB = (920, 180, 1760, 680)
    metrics = [
        ('pitch mean abs', baseline['pitch_mean_abs_arcsec'], noisy['pitch_mean_abs_arcsec']),
        ('yaw mean abs', baseline['yaw_abs_mean_arcsec'], noisy['yaw_abs_mean_arcsec']),
        ('norm mean', baseline['norm_mean_arcsec'], noisy['norm_mean_arcsec']),
        ('yaw median', baseline['yaw_abs_median_arcsec'], noisy['yaw_abs_median_arcsec']),
        ('yaw max', baseline['yaw_abs_max_arcsec'], noisy['yaw_abs_max_arcsec']),
    ]
    y_max_b = max(max(x[1], x[2]) for x in metrics) * 1.18
    px0, py0, px1, py1 = draw_axes(draw, axB, n_ticks=6, y_max=y_max_b, tick_fmt='{:.0f}', title='B. Aggregate final metrics', y_label='arcsec')
    group_w = (px1 - px0) / len(metrics)
    inner_w = group_w * 0.68
    bar_w = inner_w / 2 - 6
    for j, (label, v1, v2) in enumerate(metrics):
        gx = px0 + j * group_w + (group_w - inner_w) / 2
        for k, (val, col) in enumerate([(v1, blue), (v2, orange)]):
            bx = gx + k * (bar_w + 12)
            bh = (val / y_max_b) * (py1 - py0)
            by = py1 - bh
            draw.rounded_rectangle((bx, by, bx + bar_w, py1), radius=6, fill=col, outline=col)
            text(draw, (bx + bar_w / 2, by - 8), f'{val:.1f}', small_font, fill='#374151', anchor='mb')
        # multiline labels
        words = label.split(' ')
        if len(words) >= 3:
            text(draw, (gx + inner_w / 2, py1 + 20), ' '.join(words[:2]), small_font, fill='#374151', anchor='mm')
            text(draw, (gx + inner_w / 2, py1 + 42), ' '.join(words[2:]), small_font, fill='#374151', anchor='mm')
        else:
            text(draw, (gx + inner_w / 2, py1 + 30), label, small_font, fill='#374151', anchor='mm')
    legend_y = axB[3] - 24
    draw.rounded_rectangle((axB[0] + 20, legend_y - 14, axB[0] + 38, legend_y + 4), radius=4, fill=blue)
    text(draw, (axB[0] + 48, legend_y - 6), 'baseline', body_font, fill='#374151')
    draw.rounded_rectangle((axB[0] + 190, legend_y - 14, axB[0] + 208, legend_y + 4), radius=4, fill=orange)
    text(draw, (axB[0] + 218, legend_y - 6), 'small_gm', body_font, fill='#374151')

    # Panel C: iteration-wise yaw curve
    axC = (40, 740, 1160, 1160)
    y_max_c = max(max(baseline['iter_mean_yaw_abs_arcsec']), max(noisy['iter_mean_yaw_abs_arcsec'])) * 1.18
    px0, py0, px1, py1 = draw_axes(draw, axC, n_ticks=6, y_max=y_max_c, tick_fmt='{:.0f}', title='C. Mean |yaw| by outer iteration', y_label='arcsec')
    n_iter = len(baseline['iter_mean_yaw_abs_arcsec'])
    def pt(idx: int, val: float):
        x = px0 + idx * (px1 - px0) / (n_iter - 1)
        y = py1 - (val / y_max_c) * (py1 - py0)
        return x, y
    for values, col, label in [
        (baseline['iter_mean_yaw_abs_arcsec'], blue, 'baseline'),
        (noisy['iter_mean_yaw_abs_arcsec'], orange, 'small_gm'),
    ]:
        pts = [pt(i, v) for i, v in enumerate(values)]
        draw.line(pts, fill=col, width=4)
        for i, (x, y) in enumerate(pts):
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill='white', outline=col, width=3)
            text(draw, (x, y - 10), f'{values[i]:.1f}', small_font, fill='#374151', anchor='mb')
        lx = axC[0] + 24 + (0 if label == 'baseline' else 180)
        ly = axC[1] + 18
        draw.line((lx, ly, lx + 36, ly), fill=col, width=4)
        draw.ellipse((lx + 12, ly - 6, lx + 24, ly + 6), fill='white', outline=col, width=3)
        text(draw, (lx + 48, ly - 10), label, body_font, fill='#374151')
    for i in range(n_iter):
        x = px0 + i * (px1 - px0) / (n_iter - 1)
        text(draw, (x, py1 + 26), f'iter {i + 1}', body_font, fill='#374151', anchor='mm')

    # Panel D: text notes
    axD = (1210, 740, 1760, 1160)
    draw.rounded_rectangle(axD, radius=18, fill='white', outline='#d1d5db', width=2)
    text(draw, (axD[0] + 18, axD[1] + 18), 'D. Read-out', try_font(26, True))
    lines = [
        f"Dynamic-noise profile: small_gm",
        f"gyro sigma = {noisy['profile_desc']['gyro_sigma_dph']} dph",
        f"accel sigma = {noisy['profile_desc']['accel_sigma_ug']} ug",
        f"tau_g = {noisy['profile_desc']['tau_g_s']} s",
        f"tau_a = {noisy['profile_desc']['tau_a_s']} s",
        '',
        f"Δ yaw mean = {noisy['yaw_abs_mean_arcsec'] - baseline['yaw_abs_mean_arcsec']:+.3f}\"",
        f"Δ norm mean = {noisy['norm_mean_arcsec'] - baseline['norm_mean_arcsec']:+.3f}\"",
        f"Δ pitch mean = {noisy['pitch_mean_abs_arcsec'] - baseline['pitch_mean_abs_arcsec']:+.3f}\"",
        '',
        'Interpretation:',
        'This figure isolates the effect of adding',
        'truth-side dynamic GM drift when the',
        'alignment filter still uses the plain',
        '12-state no-Markov model.',
    ]
    y = axD[1] + 64
    for line in lines:
        if line == '':
            y += 14
            continue
        text(draw, (axD[0] + 18, y), line, body_font, fill='#374151')
        y += 30

    img.save(OUT_FIG)



def build_markdown(payload: dict[str, Any]) -> str:
    baseline = payload['summaries']['baseline']
    noisy = payload['summaries']['small_gm']
    delta_pitch = noisy['pitch_mean_abs_arcsec'] - baseline['pitch_mean_abs_arcsec']
    delta_yaw = noisy['yaw_abs_mean_arcsec'] - baseline['yaw_abs_mean_arcsec']
    delta_norm = noisy['norm_mean_arcsec'] - baseline['norm_mean_arcsec']
    return f'''# Chapter 4 no-Markov dynamic-noise comparison (2026-04-08)

- **Goal**: compare the current no-Markov DAR aligner (12-state ordinary model) before/after adding truth-side dynamic GM drift.
- **Path**: same current Chapter-4 dual-axis DAR path.
- **Filter**: `alignvn_dar_12state_py_iterfix_2026-03-30.py`.
- **Conditions**: `baseline` vs `small_gm`.
- **Seeds**: {SEEDS}.

## Summary table

| condition | pitch mean abs (") | yaw abs mean (") | norm mean (") | yaw median (") | yaw max (") |
|---|---:|---:|---:|---:|---:|
| baseline | {baseline['pitch_mean_abs_arcsec']:.6f} | {baseline['yaw_abs_mean_arcsec']:.6f} | {baseline['norm_mean_arcsec']:.6f} | {baseline['yaw_abs_median_arcsec']:.6f} | {baseline['yaw_abs_max_arcsec']:.6f} |
| small_gm | {noisy['pitch_mean_abs_arcsec']:.6f} | {noisy['yaw_abs_mean_arcsec']:.6f} | {noisy['norm_mean_arcsec']:.6f} | {noisy['yaw_abs_median_arcsec']:.6f} | {noisy['yaw_abs_max_arcsec']:.6f} |
| delta (small_gm - baseline) | {delta_pitch:+.6f} | {delta_yaw:+.6f} | {delta_norm:+.6f} | {noisy['yaw_abs_median_arcsec'] - baseline['yaw_abs_median_arcsec']:+.6f} | {noisy['yaw_abs_max_arcsec'] - baseline['yaw_abs_max_arcsec']:+.6f} |

## Iteration-wise mean |yaw|

- baseline: {baseline['iter_mean_yaw_abs_arcsec']}
- small_gm: {noisy['iter_mean_yaw_abs_arcsec']}

## Dynamic-noise profile

- gyro sigma (dph): {noisy['profile_desc']['gyro_sigma_dph']}
- accel sigma (ug): {noisy['profile_desc']['accel_sigma_ug']}
- tau_g (s): {noisy['profile_desc']['tau_g_s']}
- tau_a (s): {noisy['profile_desc']['tau_a_s']}
- note: {noisy['profile_desc']['note']}

## Files

- figure: `{OUT_FIG}`
- json: `{OUT_JSON}`
- markdown: `{OUT_MD}`
'''



def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for profile in COMPARE_PROFILES:
        rows = [run_single(seed, profile) for seed in SEEDS]
        all_rows.extend(rows)
        summaries[profile] = summarize(profile, rows)

    payload = {
        'meta': {
            'date': '2026-04-08',
            'goal': 'compare no-Markov 12-state DAR alignment before/after adding truth-side dynamic GM drift',
            'filter_script': str(BASE12_PATH),
            'truth_gm_helper': str(GM_HELPER_PATH),
            'profiles': COMPARE_PROFILES,
            'seeds': SEEDS,
            'runtime_s': round(time.time() - t0, 3),
        },
        'summaries': summaries,
        'runs': all_rows,
        'artifacts': {
            'figure': str(OUT_FIG),
            'json': str(OUT_JSON),
            'markdown': str(OUT_MD),
        },
    }

    make_figure(payload)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    OUT_MD.write_text(build_markdown(payload), encoding='utf-8')
    print(json.dumps({
        'out_fig': str(OUT_FIG),
        'baseline': summaries['baseline'],
        'small_gm': summaries['small_gm'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
