#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORKSPACE = Path('/root/.openclaw/workspace')
HELPER_PATH = WORKSPACE / 'scripts' / 'compare_four_group_arcsec_sym20_2026-04-05.py'
OUT_DIR = WORKSPACE / 'tmp' / 'ch3_sym20_42state_observability_2026-04-07'
OUT_JSON = OUT_DIR / 'ch3_sym20_42state_observability_2026-04-07.json'
OUT_MD = OUT_DIR / 'ch3_sym20_42state_observability_2026-04-07.md'
OUT_PNG = OUT_DIR / 'fig_ch3_sym20_42state_observability_2026-04-07.png'

STATE_LABELS = [
    'phi_x','phi_y','phi_z','dv_x','dv_y','dv_z',
    'eb_x','eb_y','eb_z','db_x','db_y','db_z',
    'Kg00','Kg10','Kg20','Kg01','Kg11','Kg21','Kg02','Kg12','Kg22',
    'Ka_xx','Ka_xy','Ka_xz','Ka_yy','Ka_yz','Ka_zz',
    'Ka2_x','Ka2_y','Ka2_z','rx_x','rx_y','rx_z','ry_x','ry_y','ry_z',
    'bm_gx','bm_gy','bm_gz','bm_ax','bm_ay','bm_az',
]

FAMILY_SPANS = [
    ('phi', 0, 2, '#2563eb'),
    ('dv', 3, 5, '#0891b2'),
    ('eb', 6, 8, '#7c3aed'),
    ('db', 9, 11, '#ea580c'),
    ('Kg', 12, 20, '#c2410c'),
    ('Ka', 21, 26, '#0f766e'),
    ('Ka2', 27, 29, '#65a30d'),
    ('rx', 30, 32, '#be185d'),
    ('ry', 33, 35, '#9333ea'),
    ('bm_g', 36, 38, '#b91c1c'),
    ('bm_a', 39, 41, '#1d4ed8'),
]
FAMILY_OF_INDEX = {}
for fam, a, b, color in FAMILY_SPANS:
    for i in range(a, b + 1):
        FAMILY_OF_INDEX[i] = (fam, color)


def load_font(size: int, bold: bool = False):
    candidates = [
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf' if bold else '/usr/share/fonts/dejavu/DejaVuSans.ttf',
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def text(draw: ImageDraw.ImageDraw, xy, s, font, fill, anchor=None):
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


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def normalize_from_reduction(red: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    safe = np.maximum(np.asarray(red, dtype=float), 1e-18)
    log_red = np.log10(safe)
    lo = float(log_red.min())
    hi = float(log_red.max())
    if hi - lo < 1e-12:
        score = np.ones_like(log_red)
    else:
        score = (hi - log_red) / (hi - lo)
    return score, {'log10_min': lo, 'log10_max': hi}


def build_sym20_dataset():
    helper = load_module('sym20_obs_helper_20260407', HELPER_PATH)
    cal = helper.load_cal_source()
    h24 = helper.load_h24()
    acc18 = h24.load_acc18()

    ts = helper.TS
    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    att_truth = helper.build_sym20_att(acc18)
    imu, _ = acc18.avp2imu(att_truth, pos0)
    clbt_truth = cal.get_default_clbt()
    imu_clean = cal.imuclbt(imu, clbt_truth)

    return {
        'helper': helper,
        'cal': cal,
        'acc18': acc18,
        'ts': ts,
        'pos0': pos0,
        'att_truth': att_truth,
        'imu_clean': imu_clean,
    }


def init_alignment(dataset: dict[str, Any]) -> tuple[Any, float, int, np.ndarray]:
    cal = dataset['cal']
    imu_clean = dataset['imu_clean']
    ts = dataset['ts']
    pos0 = dataset['pos0']

    nn, _, nts, _ = cal.nnts(2, ts)
    frq2 = int(1 / ts / 2) - 1
    k = frq2
    upper = min(5 * 60 * 2 * frq2, len(imu_clean))
    for k in range(frq2, upper, 2 * frq2):
        ww = np.mean(imu_clean[k - frq2:k + frq2 + 1, 0:3], axis=0)
        if np.linalg.norm(ww) / ts > 20 * cal.glv.dph:
            break
    kstatic = max(frq2 * 2, k - 3 * frq2)
    _, _, _, qnb_init = cal.alignsb(imu_clean[frq2:kstatic, :], pos0)
    return qnb_init, nts, frq2, cal.imudot(imu_clean, 5.0)


def run_covariance_case(dataset: dict[str, Any], qnb_init, nts: float, frq2: int, dotwf, use_process_noise: bool) -> dict[str, Any]:
    cal = dataset['cal']
    acc18 = dataset['acc18']
    imu_clean = dataset['imu_clean']
    pos0 = dataset['pos0']
    ts = dataset['ts']

    bi_g = 0.002 * cal.glv.dph
    bi_a = 5.0 * cal.glv.ug
    tau_g = 300.0
    tau_a = 300.0

    kf = cal.clbtkfinit_42(nts, bi_g, tau_g, bi_a, tau_a)
    P = kf['Pxk'].copy()
    Hk = kf['Hk']
    Rk = kf['Rk']
    Qk = kf['Qk'].copy() if use_process_noise else np.zeros_like(kf['Qk'])
    sigma_init = np.sqrt(np.diag(kf['Pxk']))
    sigma_init = np.where(sigma_init < 1e-30, 1.0, sigma_init)

    eth = acc18.Earth(pos0)
    _ = eth  # kept for symmetry with legacy scripts
    wnie = cal.glv.wie * np.array([0, math.cos(pos0[0]), math.sin(pos0[0])])
    Cba = np.eye(3)
    nn = 2
    n = 42

    qnb = qnb_init.copy()
    t1s = 0.0

    for k in range(2 * frq2, len(imu_clean) - frq2, nn):
        k1 = k + nn - 1
        wm = imu_clean[k:k1 + 1, 0:3]
        vm = imu_clean[k:k1 + 1, 3:6]
        dwb = np.mean(dotwf[k:k1 + 1, 0:3], axis=0)

        phim, dvbm = cal.cnscl(np.hstack((wm, vm)))
        wb = phim / nts
        fb = dvbm / nts
        SS = cal.imulvS(wb, dwb, Cba)
        Cnb = cal.q2mat(qnb)
        Ft = cal.getFt_42(fb, wb, Cnb, wnie, SS, tau_g, tau_a)
        Phi = np.eye(n) + Ft * nts

        P = Phi @ P @ Phi.T + Qk

        t1s += nts
        if t1s > (0.2 - ts / 2):
            t1s = 0.0
            ww = np.mean(imu_clean[k - frq2:k + frq2 + 1, 0:3], axis=0)
            if np.linalg.norm(ww) / ts < 20 * cal.glv.dph:
                S = Hk @ P @ Hk.T + Rk
                K = P @ Hk.T @ np.linalg.inv(S)
                I_KH = np.eye(n) - K @ Hk
                P = I_KH @ P @ I_KH.T + K @ Rk @ K.T

        qnb = cal.qupdt2(qnb, phim, wnie * nts)

    sigma_final = np.sqrt(np.diag(P))
    reduction = sigma_final / sigma_init
    score, meta = normalize_from_reduction(reduction)

    rows = []
    for i, label in enumerate(STATE_LABELS):
        family, color = FAMILY_OF_INDEX[i]
        rows.append({
            'state': label,
            'family': family,
            'color': color,
            'sigma_init': float(sigma_init[i]),
            'sigma_final': float(sigma_final[i]),
            'reduction_ratio': float(reduction[i]),
            'normalized_score': float(score[i]),
        })

    fam_summary = []
    for fam, a, b, color in FAMILY_SPANS:
        idx = slice(a, b + 1)
        fam_summary.append({
            'family': fam,
            'color': color,
            'mean_reduction_ratio': float(np.mean(reduction[idx])),
            'mean_normalized_score': float(np.mean(score[idx])),
        })

    top_idx = np.argsort(-score)[:8]
    bottom_idx = np.argsort(score)[:8]

    return {
        'use_process_noise': bool(use_process_noise),
        'state_rows': rows,
        'family_summary': fam_summary,
        'normalization_meta': meta,
        'top8_states': [{'state': STATE_LABELS[i], 'score': float(score[i]), 'reduction_ratio': float(reduction[i])} for i in top_idx],
        'bottom8_states': [{'state': STATE_LABELS[i], 'score': float(score[i]), 'reduction_ratio': float(reduction[i])} for i in bottom_idx],
    }


def draw_axes(draw, box, y_ticks=5, y_max=1.0):
    x0, y0, x1, y1 = box
    label_font = load_font(18)
    draw.rectangle((x0, y0, x1, y1), fill='#ffffff', outline='#d1d5db', width=1)
    for i in range(y_ticks + 1):
        frac = i / y_ticks
        yy = y1 - (y1 - y0) * frac
        draw.line((x0, yy, x1, yy), fill='#e5e7eb', width=1)
        text(draw, (x0 - 12, yy), f'{frac * y_max:.1f}', label_font, '#6b7280', anchor='rm')
    draw.line((x0, y0, x0, y1), fill='#111827', width=2)
    draw.line((x0, y1, x1, y1), fill='#111827', width=2)


def make_png(payload: dict[str, Any]):
    width, height = 2300, 1500
    img = Image.new('RGBA', (width, height), '#ffffff')
    draw = ImageDraw.Draw(img)
    title_font = load_font(42, True)
    sub_font = load_font(24)
    small_font = load_font(18)
    label_font = load_font(13)

    text(draw, (60, 44), '20-position self-calibration (corrected symmetric20) — 42-state GM1 observability', title_font, '#111827')
    text(draw, (60, 96), 'Panel A uses the full-Q case as the main practical view; Panel B compares family means with/without process noise.', sub_font, '#374151')

    full_q = payload['cases']['full_q']
    no_q = payload['cases']['no_q']

    # Panel A: state-wise full-Q score
    px0, py0, px1, py1 = 110, 170, 2230, 930
    draw_axes(draw, (px0, py0, px1, py1))
    text(draw, (px0, py0 - 26), 'A. State-wise normalized observability score (full process noise case)', load_font(30, True), '#111827')
    g_w = (px1 - px0) / len(STATE_LABELS)
    bar_w = max(10, int(g_w * 0.52))
    values = [row['normalized_score'] for row in full_q['state_rows']]
    colors = [row['color'] for row in full_q['state_rows']]
    for fam, a, b, color in FAMILY_SPANS:
        sx0 = px0 + g_w * a
        sx1 = px0 + g_w * (b + 1)
        if a > 0:
            draw.line((sx0, py0, sx0, py1), fill='#cbd5e1', width=2)
        text(draw, ((sx0 + sx1) / 2, py0 - 6), fam, small_font, '#4b5563', anchor='mb')
    for i, (lab, val, color) in enumerate(zip(STATE_LABELS, values, colors)):
        cx = px0 + g_w * (i + 0.5)
        bh = (py1 - py0) * max(0.0, min(1.0, float(val)))
        bx0 = cx - bar_w / 2
        by0 = py1 - bh
        draw.rounded_rectangle((bx0, by0, bx0 + bar_w, py1), radius=3, fill=color, outline='#ffffff', width=1)
        text(draw, (cx, py1 + 16), lab, label_font, '#374151', anchor='ma')
    paste_rotated_text(img, (46, (py0 + py1) / 2), 'normalized observability score', load_font(24, True), '#111827', angle=90)

    # Panel B: family means no-Q vs full-Q
    fx0, fy0, fx1, fy1 = 110, 1050, 2230, 1400
    draw_axes(draw, (fx0, fy0, fx1, fy1))
    text(draw, (fx0, fy0 - 26), 'B. Family-mean score comparison: Q=0 vs full-Q', load_font(30, True), '#111827')
    fam_labels = [row['family'] for row in full_q['family_summary']]
    no_q_map = {row['family']: row for row in no_q['family_summary']}
    full_q_map = {row['family']: row for row in full_q['family_summary']}
    g2 = (fx1 - fx0) / len(fam_labels)
    total_w = max(24, int(g2 * 0.72))
    bw = total_w / 2 - 4
    for i, fam in enumerate(fam_labels):
        cx = fx0 + g2 * (i + 0.5)
        for j, (name, val, color) in enumerate([
            ('Q=0', no_q_map[fam]['mean_normalized_score'], '#2563eb'),
            ('full-Q', full_q_map[fam]['mean_normalized_score'], '#f97316'),
        ]):
            bh = (fy1 - fy0) * max(0.0, min(1.0, float(val)))
            bx0 = cx - total_w / 2 + j * (bw + 8)
            by0 = fy1 - bh
            draw.rounded_rectangle((bx0, by0, bx0 + bw, fy1), radius=4, fill=color)
        text(draw, (cx, fy1 + 18), fam, load_font(16), '#374151', anchor='ma')
    paste_rotated_text(img, (46, (fy0 + fy1) / 2), 'mean score', load_font(22, True), '#111827', angle=90)
    # legend
    lx, ly = 1830, 1008
    draw.rounded_rectangle((lx, ly, 2210, ly + 54), radius=12, fill='#ffffff', outline='#d1d5db')
    draw.rectangle((lx + 18, ly + 18, lx + 36, ly + 36), fill='#2563eb')
    text(draw, (lx + 46, ly + 27), 'Q=0', load_font(18), '#374151', anchor='lm')
    draw.rectangle((lx + 118, ly + 18, lx + 136, ly + 36), fill='#f97316')
    text(draw, (lx + 146, ly + 27), 'full-Q', load_font(18), '#374151', anchor='lm')

    text(draw, (60, 1455), 'State score here is a normalized inverse-log view of sigma_final / sigma_init: higher means stronger practical observability under the chosen 20-position trajectory.', small_font, '#6b7280')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    img.convert('RGB').save(OUT_PNG, quality=95)


def build_markdown(payload: dict[str, Any]) -> str:
    full_q = payload['cases']['full_q']
    lines = [
        '# 20-position corrected symmetric20 / 42-state GM1 observability analysis',
        '',
        '- Path: corrected symmetric20 (20-position, 1200 s)',
        '- Model: 42-state GM1 self-calibration filter baseline',
        '- Cases: `Q=0` (geometric/theoretical lower-bound style proxy) and `full-Q` (practical noise-floor view)',
        '',
        '## Full-Q top 8 states',
        '',
        '| state | score | sigma_final/sigma_init |',
        '|---|---:|---:|',
    ]
    for row in full_q['top8_states']:
        lines.append(f"| {row['state']} | {row['score']:.3f} | {row['reduction_ratio']:.3e} |")
    lines += [
        '',
        '## Full-Q bottom 8 states',
        '',
        '| state | score | sigma_final/sigma_init |',
        '|---|---:|---:|',
    ]
    for row in full_q['bottom8_states']:
        lines.append(f"| {row['state']} | {row['score']:.3f} | {row['reduction_ratio']:.3e} |")
    lines += [
        '',
        '## Family means',
        '',
        '| family | mean score (Q=0) | mean score (full-Q) |',
        '|---|---:|---:|',
    ]
    no_q_map = {row['family']: row for row in payload['cases']['no_q']['family_summary']}
    for row in full_q['family_summary']:
        fam = row['family']
        lines.append(f"| {fam} | {no_q_map[fam]['mean_normalized_score']:.3f} | {row['mean_normalized_score']:.3f} |")
    return '\n'.join(lines) + '\n'


def main():
    dataset = build_sym20_dataset()
    qnb_init, nts, frq2, dotwf = init_alignment(dataset)
    no_q = run_covariance_case(dataset, qnb_init, nts, frq2, dotwf, use_process_noise=False)
    full_q = run_covariance_case(dataset, qnb_init, nts, frq2, dotwf, use_process_noise=True)

    payload = {
        'task': 'ch3_sym20_42state_observability_analysis',
        'trajectory': {
            'name': 'corrected_symmetric20',
            'n_rows': 20,
            'att0_deg': [0.0, 0.0, 0.0],
            'pos0_deg_m': [34.0, 116.0, 480.0],
            'sample_dt_s': float(dataset['ts']),
            'duration_s': float(len(dataset['imu_clean']) * dataset['ts']),
        },
        'model': {
            'name': '42-state GM1 self-calibration baseline',
            'source': str((WORKSPACE / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py').resolve()),
            'noise': {
                'arw_dpsh': 0.005,
                'vrw_ugpsHz': 5.0,
                'bi_g_dph': 0.002,
                'bi_a_ug': 5.0,
                'tau_g_s': 300.0,
                'tau_a_s': 300.0,
            },
        },
        'cases': {
            'no_q': no_q,
            'full_q': full_q,
        },
        'artifacts': {
            'png': str(OUT_PNG),
            'md': str(OUT_MD),
            'json': str(OUT_JSON),
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    OUT_MD.write_text(build_markdown(payload), encoding='utf-8')
    make_png(payload)
    print(str(OUT_JSON))
    print(str(OUT_MD))
    print(str(OUT_PNG))


if __name__ == '__main__':
    main()
