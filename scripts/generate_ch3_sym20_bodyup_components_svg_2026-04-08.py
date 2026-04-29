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
OUT_SVG = OUT_DIR / 'fig_ch3_sym20_bodyup_components_2026-04-08.svg'
OUT_META = OUT_DIR / 'fig_ch3_sym20_bodyup_components_2026-04-08_meta.json'

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
from psins_py.math_utils import a2mat

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
    mod = load_module('sym20_bodyup_src', str(WORKSPACE / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'))
    candidate = probe.build_symmetric20_candidate(mod)
    paras = rows_to_paras(mod, candidate.all_rows)
    faces = orientation_faces(mod, paras)

    att0 = mod.np.array([0.0, 0.0, 0.0]) * mod.glv.deg
    ts = 0.01
    att = mod.attrottt(att0, paras, ts)
    t = att[:, 3]

    g_body = np.zeros((len(att), 3), dtype=float)
    g_nav = np.array([0.0, 0.0, 1.0])
    for i in range(len(att)):
        C = a2mat(att[i, :3])
        g_body[i, :] = C.T @ g_nav

    gx = g_body[:, 0]
    gy = g_body[:, 1]
    gz = g_body[:, 2]

    W, H = 2600, 1700
    left = 150
    right = 2480
    top_strip_y = 210
    top_strip_h = 88
    face_strip_y = 315
    face_strip_h = 78
    plot_h = 260
    panel_tops = [460, 800, 1140]
    TMAX = 1200.0
    ymin, ymax = -1.1, 1.1

    def tx(sec: float) -> float:
        return left + (right - left) * sec / TMAX

    def ty(val: float, top: float) -> float:
        return top + plot_h * (1 - (val - ymin) / (ymax - ymin))

    row_blocks = []
    for idx, row in enumerate(candidate.all_rows, start=1):
        t0 = (idx - 1) * 60.0
        t1 = idx * 60.0
        row_blocks.append((idx, t0, t1, row, faces[idx - 1]))

    panels = [
        ('Roll（X 向分量）', gx, '#DC2626'),
        ('Yaw（Y 向分量）', gy, '#059669'),
        ('Pitch（Z 向分量）', gz, '#2563EB'),
    ]

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        svg_rect(0, 0, W, H, fill='white'),
        svg_text(70, 52, '20位置标定路径朝向分量-时间变化曲线', size=44, fill='#111827', weight='700'),
        svg_text(70, 106, '这版保留机体 z 轴（朝上方向）在机体系 X/Y/Z 上的分量，但按 roll / yaw / pitch 的阅读顺序展示，便于和上图一起看。', size=22, fill='#475569'),
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
        parts.append(svg_text(left, top - 40, title, size=26, fill='#111827', weight='700'))

        for val in [-1.0, -0.5, 0.0, 0.5, 1.0]:
            yy = ty(val, top)
            parts.append(svg_text(left - 18, yy, f'{val:.1f}', size=20, fill='#475569', anchor='end'))

        if title.startswith('Pitch'):
            for sec in range(0, 1201, 120):
                x = tx(float(sec))
                parts.append(svg_text(x, top + plot_h + 24, f'{sec}s', size=20, fill='#475569', anchor='middle'))

        pts = []
        step = max(1, len(t) // 5000)
        for i in range(0, len(t), step):
            pts.append((tx(float(t[i])), ty(float(data[i]), top)))
        parts.append(svg_polyline(pts, stroke=color, sw=5))

    legend_y = 1510
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

    note = '解释：若机体 +X 朝上，则 X 向分量≈+1；若机体 -Z 朝上，则 Z 向分量≈-1。该图用朝上方向分量来展示多位置路径的姿态切换。'
    parts.append(svg_text(70, 1640, note, size=18, fill='#475569'))
    parts.append('</svg>')

    OUT_SVG.write_text('\n'.join(parts), encoding='utf-8')
    meta = {
        'title': '20位置标定路径朝向分量-时间变化曲线',
        'path_case': 'corrected symmetric20',
        'att0_deg': [0.0, 0.0, 0.0],
        'total_time_s': 1200.0,
        'output_svg': str(OUT_SVG),
        'segment_end_faces': [f['face_name'] for f in faces],
    }
    OUT_META.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(OUT_SVG))


if __name__ == '__main__':
    main()
