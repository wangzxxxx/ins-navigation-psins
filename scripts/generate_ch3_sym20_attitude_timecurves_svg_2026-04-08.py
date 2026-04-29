#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
OUT_DIR = WORKSPACE / 'tmp' / 'ch3_sym20_attitude_curves_2026-04-08'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_SVG = OUT_DIR / 'fig_ch3_sym20_attitude_timecurves_2026-04-08.svg'
OUT_META = OUT_DIR / 'fig_ch3_sym20_attitude_timecurves_2026-04-08_svg_meta.json'

for p in [
    WORKSPACE,
    WORKSPACE / 'tmp_psins_py',
    WORKSPACE / 'psins_method_bench' / 'methods' / 'markov',
    WORKSPACE / 'psins_method_bench' / 'scripts',
]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module
import probe_ch3_corrected_symmetric20_front2_back11 as probe
from benchmark_ch3_12pos_goalA_repairs import rows_to_paras, orientation_faces

FACE_COLORS = {
    '+X': '#FDE2E2',
    '-X': '#FEF1F1',
    '+Y': '#E3F7EA',
    '-Y': '#F1FCF5',
    '+Z': '#E3F0FF',
    '-Z': '#F1F7FF',
}
TOP_STRIP_FILL = '#F8FAFC'
TOP_STRIP_OUTLINE = '#E2E8F0'
FONT_STACK = "'Noto Sans CJK SC','Noto Sans CJK','Microsoft YaHei','PingFang SC','Helvetica Neue',Arial,sans-serif"


def nice_limits(data: np.ndarray, min_span: float = 20.0, pad_frac: float = 0.08):
    dmin = float(np.min(data))
    dmax = float(np.max(data))
    span = dmax - dmin
    if span < min_span:
        mid = 0.5 * (dmax + dmin)
        half = 0.5 * min_span
        return mid - half, mid + half
    pad = max(min_span * 0.05, span * pad_frac)
    return dmin - pad, dmax + pad


def svg_text(x, y, text, size=18, fill='#111827', weight='400', anchor='start'):
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" fill="{fill}" '
        f'font-family="{FONT_STACK}" font-size="{size}" font-weight="{weight}" '
        f'text-anchor="{anchor}" dominant-baseline="middle">{escape(str(text))}</text>'
    )


def svg_rect(x, y, w, h, fill='none', stroke='none', sw=1, rx=0):
    extra = f' rx="{rx}" ry="{rx}"' if rx else ''
    return (
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}"{extra} '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" />'
    )


def svg_line(x1, y1, x2, y2, stroke='#000', sw=1):
    return f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="{stroke}" stroke-width="{sw}" />'


def svg_polyline(points, stroke='#000', sw=3, fill='none'):
    pts = ' '.join(f'{x:.2f},{y:.2f}' for x, y in points)
    return f'<polyline points="{pts}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" stroke-linejoin="round" stroke-linecap="round" />'


def main():
    mod = load_module('sym20_curve_svg_src', str(WORKSPACE / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'))
    candidate = probe.build_symmetric20_candidate(mod)
    paras = rows_to_paras(mod, candidate.all_rows)
    faces = orientation_faces(mod, paras)

    att0 = mod.np.array([0.0, 0.0, 0.0]) * mod.glv.deg
    ts = 0.01
    att = mod.attrottt(att0, paras, ts)
    t = att[:, 3]

    pitch = np.rad2deg(np.unwrap(att[:, 0]))
    roll = np.rad2deg(np.unwrap(att[:, 1]))
    yaw = np.rad2deg(np.unwrap(att[:, 2]))

    W, H = 2600, 1760
    left = 150
    right = 2480
    top_strip_y = 210
    top_strip_h = 88
    face_strip_y = 315
    face_strip_h = 78
    plot_h = 280
    panel_tops = [470, 840, 1210]
    TMAX = 1200.0

    def tx(sec: float) -> float:
        return left + (right - left) * sec / TMAX

    row_blocks = []
    for idx, row in enumerate(candidate.all_rows, start=1):
        t0 = (idx - 1) * 60.0
        t1 = idx * 60.0
        row_blocks.append((idx, t0, t1, row, faces[idx - 1]))

    panels = [
        ('横滚角 Roll', roll, '#DC2626'),
        ('俯仰角 Pitch', pitch, '#059669'),
        ('航向角 Yaw', yaw, '#2563EB'),
    ]

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        svg_rect(0, 0, W, H, fill='white'),
        svg_text(70, 52, '20位置标定路径姿态-时间变化曲线', size=46, fill='#111827', weight='700'),
        svg_text(70, 110, 'corrected symmetric20 · att0=(0°,0°,0°) · 总时长 1200 s · 20×60 s · 曲线按真实标定姿态序列生成', size=24, fill='#475569'),
    ]

    for idx, t0, t1, row, face in row_blocks:
        x0, x1 = tx(t0), tx(t1)
        parts.append(svg_rect(x0, top_strip_y, x1 - x0, top_strip_h, fill=TOP_STRIP_FILL, stroke=TOP_STRIP_OUTLINE, sw=2))
        parts.append(svg_text((x0 + x1) / 2, top_strip_y + 28, f'P{idx}', size=18, fill='#1F2937', weight='500', anchor='middle'))
        axis = row['axis']
        ang = int(round(row['angle_deg']))
        axis_label = 'X' if axis[0] != 0 else ('Y' if axis[1] != 0 else 'Z')
        parts.append(svg_text((x0 + x1) / 2, top_strip_y + 62, f'{axis_label}{ang:+d}°', size=16, fill='#64748B', anchor='middle'))

    for idx, t0, t1, row, face in row_blocks:
        x0, x1 = tx(t0), tx(t1)
        parts.append(svg_rect(x0, face_strip_y, x1 - x0, face_strip_h, fill=FACE_COLORS[face['face_name']], stroke=TOP_STRIP_OUTLINE, sw=2))
        parts.append(svg_text((x0 + x1) / 2, face_strip_y + 26, face['face_name'], size=18, fill='#111827', weight='500', anchor='middle'))

    for (title, data, color), top in zip(panels, panel_tops):
        ymin, ymax = nice_limits(data)
        parts.append(svg_rect(left - 26, top - 24, (right - left) + 50, plot_h + 76, fill='#FCFCFD', stroke='#E5E7EB', sw=2, rx=22))
        for idx, t0, t1, row, face in row_blocks:
            parts.append(svg_rect(tx(t0), top, tx(t1) - tx(t0), plot_h, fill=FACE_COLORS[face['face_name']]))
        for frac in np.linspace(0, 1, 5):
            y = top + plot_h * frac
            parts.append(svg_line(left, y, right, y, stroke='#E5E7EB', sw=2))
        for sec in range(0, 1201, 60):
            x = tx(float(sec))
            parts.append(svg_line(x, top, x, top + plot_h, stroke='#E5E7EB', sw=2))
        parts.append(svg_rect(left, top, right - left, plot_h, fill='none', stroke='#94A3B8', sw=3))
        parts.append(svg_text(left, top - 40, title, size=28, fill='#111827', weight='700'))
        parts.append(svg_text(right - 60, top - 38, 'deg', size=20, fill='#64748B'))

        for val in np.linspace(ymin, ymax, 5):
            yy = top + plot_h * (1 - (val - ymin) / (ymax - ymin))
            parts.append(svg_text(left - 18, yy, f'{val:.0f}', size=20, fill='#475569', anchor='end'))

        if title == '航向角 Yaw':
            for sec in range(0, 1201, 120):
                x = tx(float(sec))
                parts.append(svg_text(x, top + plot_h + 24, f'{sec}s', size=20, fill='#475569', anchor='middle'))

        pts = []
        step = max(1, len(t) // 5000)
        for i in range(0, len(t), step):
            x = tx(float(t[i]))
            y = top + plot_h * (1 - (float(data[i]) - ymin) / (ymax - ymin))
            pts.append((x, y))
        if len(pts) > 1:
            parts.append(svg_polyline(pts, stroke=color, sw=5))

    legend_y = 1588
    parts.append(svg_text(70, legend_y - 18, '图例', size=28, fill='#111827', weight='700'))
    legend_items = [
        ('+X 朝上', FACE_COLORS['+X']),
        ('-X 朝上', FACE_COLORS['-X']),
        ('+Y 朝上', FACE_COLORS['+Y']),
        ('-Y 朝上', FACE_COLORS['-Y']),
        ('+Z 朝上', FACE_COLORS['+Z']),
        ('-Z 朝上', FACE_COLORS['-Z']),
    ]
    x = 120
    y = legend_y
    for label, color in legend_items:
        parts.append(svg_rect(x, y, 28, 28, fill=color, stroke='#64748B', sw=1))
        parts.append(svg_text(x + 40, y + 14, label, size=18, fill='#334155'))
        x += 360
        if x > 2100:
            x = 120
            y += 40

    note = '说明：实线为实际姿态欧拉角随时间演化。这张图反映的是 corrected symmetric20 标定路径本身的姿态几何，而不是滤波误差收敛结果。'
    parts.append(svg_text(70, 1700, note, size=18, fill='#475569'))
    parts.append('</svg>')

    OUT_SVG.write_text('\n'.join(parts), encoding='utf-8')
    meta = {
        'title': '20位置标定路径姿态-时间变化曲线',
        'path_case': 'corrected symmetric20',
        'att0_deg': [0.0, 0.0, 0.0],
        'total_time_s': 1200.0,
        'output_svg': str(OUT_SVG),
        'roll_deg_range': [float(np.min(roll)), float(np.max(roll))],
        'pitch_deg_range': [float(np.min(pitch)), float(np.max(pitch))],
        'yaw_deg_range': [float(np.min(yaw)), float(np.max(yaw))],
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(OUT_SVG))


if __name__ == '__main__':
    main()
