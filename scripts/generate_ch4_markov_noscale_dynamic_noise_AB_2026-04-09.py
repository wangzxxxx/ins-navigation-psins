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
OUT_DIR = WORKSPACE / 'tmp' / 'ch4_markov_noscale_dynamic_noise_AB_2026-04-09'
OUT_JSON = OUT_DIR / 'ch4_markov_noscale_dynamic_noise_AB_2026-04-09.json'
OUT_MD = OUT_DIR / 'ch4_markov_noscale_dynamic_noise_AB_2026-04-09.md'
OUT_FULL = OUT_DIR / 'fig_ch4_markov_noscale_dynamic_noise_AB_2026-04-09.png'
OUT_A = OUT_DIR / 'fig_ch4_markov_noscale_dynamic_noise_panel_A_2026-04-09.png'
OUT_B = OUT_DIR / 'fig_ch4_markov_noscale_dynamic_noise_panel_B_2026-04-09.png'
ACC18_PATH = SCRIPTS / 'alignvn_dar_accel_colored_py_2026-03-30.py'
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
]
FAMILY_SPANS = [
    ('phi', 0, 2, '#2563eb'),
    ('dV', 3, 5, '#0891b2'),
    ('eb', 6, 8, '#7c3aed'),
    ('db', 9, 11, '#ea580c'),
    ('ng', 12, 14, '#b91c1c'),
    ('xa', 15, 17, '#0f766e'),
]


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod



def build_initial_scales(acc18_mod) -> np.ndarray:
    base12 = acc18_mod.load_iterfix12_module()
    glv = acc18_mod.glv
    imuerr = base12.build_imuerr()
    cfg_ng_sigma = np.array([0.05, 0.05, 0.05]) * glv.dph
    cfg_xa_sigma = np.maximum(np.array([0.01, 0.01, 0.01]) * glv.ug, 5.0 * glv.ug)
    phi0 = np.array([0.1, 0.1, 0.5]) * glv.deg
    dV0 = np.ones(3)
    init_eb = np.maximum(np.asarray(imuerr['eb']).reshape(3), 0.1 * glv.dph)
    init_db = np.maximum(np.asarray(imuerr['db']).reshape(3), 1000.0 * glv.ug)
    return np.r_[phi0, dV0, init_eb, init_db, cfg_ng_sigma, cfg_xa_sigma]



def build_noisy_imu(acc18_mod, gm, profile: str, seed: int, imu_clean: np.ndarray):
    base12 = acc18_mod.load_iterfix12_module()
    glv = acc18_mod.glv
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

    truth_eb = np.asarray(imuerr['eb']).reshape(3)
    truth_db = np.asarray(imuerr['db']).reshape(3)
    truth_ng = gyro_gm_last
    truth_xa = accel_gm_last
    return imu_noisy, imuerr, truth_eb, truth_db, truth_ng, truth_xa



def run_single(task: tuple[str, int]) -> dict[str, Any]:
    profile, seed = task
    acc18_mod = load_module(f'markov_noscale_acc18_{os.getpid()}', ACC18_PATH)
    gm = load_module(f'markov_noscale_gm_{os.getpid()}', GM_HELPER_PATH)
    base12 = acc18_mod.load_iterfix12_module()
    glv = acc18_mod.glv

    ts = 0.01
    att0 = np.array([0.0, 0.0, 0.0])
    pos0 = acc18_mod.posset(34, 116, 480, isdeg=1)
    rot_paras = base12.build_rot_paras()
    att_truth = acc18_mod.attrottt(att0, rot_paras, ts)
    imu_clean, _ = acc18_mod.avp2imu(att_truth, pos0)

    imu_noisy, imuerr, truth_eb, truth_db, truth_ng, truth_xa = build_noisy_imu(acc18_mod, gm, profile, seed, imu_clean)

    phi = np.array([0.1, 0.1, 0.5]) * glv.deg
    att0_guess = acc18_mod.q2att(base12.qaddphi(acc18_mod.a2qua(att0), phi))
    wvn = np.array([0.01, 0.01, 0.01])
    truth_att = att_truth[-1, 0:3]

    att_aligned, attk, xkpk, _iter_logs = acc18_mod.alignvn_18state_iter(
        imu=imu_noisy.copy(),
        qnb=att0_guess,
        pos=pos0,
        phi0=phi,
        imuerr=imuerr,
        wvn=wvn,
        max_iter=5,
        truth_att=truth_att,
        ng_sigma=np.array([0.05, 0.05, 0.05]) * glv.dph,
        tau_g_s=np.array([300.0, 300.0, 300.0]),
        xa_sigma=np.array([0.01, 0.01, 0.01]) * glv.ug,
        tau_a_s=np.array([100.0, 100.0, 100.0]),
        wash_scale=0.5,
        carry_att_seed=True,
    )

    final_att = att_aligned
    final_vn = attk[-1, 3:6]
    final_x = xkpk[-1, 0:18]

    phi_err = acc18_mod.qq2phi(acc18_mod.a2qua(final_att), acc18_mod.a2qua(truth_att))
    dV_err = final_vn.copy()
    eb_err = final_x[6:9] - truth_eb
    db_err = final_x[9:12] - truth_db
    ng_err = final_x[12:15] - truth_ng
    xa_err = final_x[15:18] - truth_xa
    err_vec = np.r_[phi_err, dV_err, eb_err, db_err, ng_err, xa_err]

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



def make_pngs(payload: dict[str, Any]):
    title_font = try_font(40, True)
    sub_font = try_font(22)
    small_font = try_font(18)
    label_font = try_font(13)

    case = payload['cases']['small_gm']
    base_case = payload['cases']['baseline']

    # Panel A
    imgA = Image.new('RGBA', (2200, 980), '#ffffff')
    draw = ImageDraw.Draw(imgA)
    text(draw, (50, 40), 'Markov / no-scale — dynamic-noise panel A', title_font, '#111827')
    text(draw, (50, 86), 'State-wise normalized score under small_gm, with Markov states ng/xa retained and scale-factor states removed.', sub_font, '#374151')
    px0, py0, px1, py1 = 100, 160, 2120, 860
    draw_axes(draw, (px0, py0, px1, py1))
    g_w = (px1 - px0) / len(STATE_LABELS)
    bar_w = max(20, int(g_w * 0.56))
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
    paste_rotated_text(imgA, (40, (py0 + py1) / 2), 'normalized practical observability / recoverability score', try_font(22, True), '#111827', angle=90)
    text(draw, (50, 920), f"small_gm strongest family: {case['top_family']}   |   weakest family: {case['bottom_family']}", small_font, '#374151')
    imgA.convert('RGB').save(OUT_A, quality=95)

    # Panel B
    imgB = Image.new('RGBA', (1700, 760), '#ffffff')
    draw = ImageDraw.Draw(imgB)
    text(draw, (50, 40), 'Markov / no-scale — dynamic-noise panel B', title_font, '#111827')
    text(draw, (50, 86), 'Family-mean score: baseline vs small_gm', sub_font, '#374151')
    fx0, fy0, fx1, fy1 = 110, 160, 1600, 610
    draw_axes(draw, (fx0, fy0, fx1, fy1))
    fam_labels = [row['family'] for row in base_case['family_summary']]
    base_map = {row['family']: row for row in base_case['family_summary']}
    gm_map = {row['family']: row for row in case['family_summary']}
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
    paste_rotated_text(imgB, (40, (fy0 + fy1) / 2), 'mean score', try_font(22, True), '#111827', angle=90)
    lx, ly = 1160, 120
    draw.rounded_rectangle((lx, ly, 1580, ly + 54), radius=12, fill='#ffffff', outline='#d1d5db')
    draw.rectangle((lx + 18, ly + 18, lx + 36, ly + 36), fill='#2563eb')
    text(draw, (lx + 46, ly + 27), 'baseline', try_font(18), '#374151', anchor='lm')
    draw.rectangle((lx + 178, ly + 18, lx + 196, ly + 36), fill='#f97316')
    text(draw, (lx + 206, ly + 27), 'small_gm', try_font(18), '#374151', anchor='lm')
    imgB.convert('RGB').save(OUT_B, quality=95)

    # Full combined for archive
    full = Image.new('RGBA', (2200, 1760), '#ffffff')
    full.alpha_composite(Image.open(OUT_A).convert('RGBA'), (0, 0))
    full.alpha_composite(Image.open(OUT_B).convert('RGBA'), (200, 980))
    full.convert('RGB').save(OUT_FULL, quality=95)



def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        '# Markov / no-scale dynamic-noise AB panels (2026-04-09)',
        '',
        '- **Model**: 18-state Markov alignment without scale-factor states.',
        '- **States**: phi / dV / eb / db / ng / xa.',
        '- **Condition pair**: baseline vs small_gm.',
        '- **Seeds**: ' + str(SEEDS),
        '',
        '## Files',
        f'- panel A: `{OUT_A}`',
        f'- panel B: `{OUT_B}`',
        f'- full: `{OUT_FULL}`',
        f'- json: `{OUT_JSON}`',
        f'- md: `{OUT_MD}`',
    ]
    return '\n'.join(lines) + '\n'



def main():
    t0 = time.time()
    acc18_mod = load_module('markov_noscale_acc18_main', ACC18_PATH)
    scales = build_initial_scales(acc18_mod)

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
            glv = acc18_mod.glv
            if i < 3:
                display = f"{mean_err[i] / glv.sec:.3f}\""
            elif i < 6:
                display = f"{mean_err[i]:.6f} m/s"
            elif i < 9:
                display = f"{mean_err[i] / glv.dph:.6f} dph"
            elif i < 12:
                display = f"{mean_err[i] / glv.ug:.3f} ug"
            elif i < 15:
                display = f"{mean_err[i] / glv.dph:.6f} dph"
            else:
                display = f"{mean_err[i] / glv.ug:.3f} ug"
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
        'task': 'ch4_markov_noscale_dynamic_noise_AB',
        'method_note': 'Practical recoverability score under actual noisy DAR runs, with Markov states retained and scale-factor states removed.',
        'profiles': PROFILES,
        'seeds': SEEDS,
        'state_labels': STATE_LABELS,
        'normalization_meta': meta,
        'initial_scales_native': [float(x) for x in scales],
        'cases': cases,
        'raw_runs': runs,
        'artifacts': {
            'panel_a_png': str(OUT_A),
            'panel_b_png': str(OUT_B),
            'full_png': str(OUT_FULL),
            'json': str(OUT_JSON),
            'md': str(OUT_MD),
        },
        'runtime_s': round(time.time() - t0, 3),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_pngs(payload)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    OUT_MD.write_text(build_markdown(payload), encoding='utf-8')
    print(json.dumps({
        'panel_a': str(OUT_A),
        'panel_b': str(OUT_B),
        'small_gm_top_family': cases['small_gm']['top_family'],
        'small_gm_bottom_family': cases['small_gm']['bottom_family'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
