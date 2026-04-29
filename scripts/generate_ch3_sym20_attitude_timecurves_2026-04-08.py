#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WORKSPACE = Path('/root/.openclaw/workspace')
OUT_DIR = WORKSPACE / 'tmp' / 'ch3_sym20_attitude_curves_2026-04-08'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / 'fig_ch3_sym20_attitude_timecurves_2026-04-08.png'
OUT_JSON = OUT_DIR / 'fig_ch3_sym20_attitude_timecurves_2026-04-08_meta.json'

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

FONT_REG = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
]
FONT_BOLD = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf',
]

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


def load_font(size: int, bold: bool = False):
    cands = FONT_BOLD if bold else FONT_REG
    for path in cands:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def text_center(draw: ImageDraw.ImageDraw, box, text, font, fill):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    x0, y0, x1, y1 = box
    draw.text(((x0 + x1 - tw) / 2, (y0 + y1 - th) / 2), text, font=font, fill=fill)


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


def main():
    mod = load_module('sym20_curve_src', str(WORKSPACE / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'))
    candidate = probe.build_symmetric20_candidate(mod)
    paras = rows_to_paras(mod, candidate.all_rows)
    faces = orientation_faces(mod, paras)

    att0 = mod.np.array([0.0, 0.0, 0.0]) * mod.glv.deg
    ts = 0.01
    att = mod.attrottt(att0, paras, ts)
    t = att[:, 3]

    # PSINS order: [pitch, roll, yaw]
    pitch = np.rad2deg(np.unwrap(att[:, 0]))
    roll = np.rad2deg(np.unwrap(att[:, 1]))
    yaw = np.rad2deg(np.unwrap(att[:, 2]))

    W, H = 2600, 1760
    img = Image.new('RGB', (W, H), 'white')
    d = ImageDraw.Draw(img)

    f_title = load_font(46, True)
    f_sub = load_font(24)
    f_panel = load_font(28, True)
    f_axis = load_font(20)
    f_small = load_font(18)
    f_tiny = load_font(16)

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

    d.text((70, 42), '20位置标定路径姿态-时间变化曲线', font=f_title, fill='#111827')
    d.text((70, 102), 'corrected symmetric20 · att0=(0°,0°,0°) · 总时长 1200 s · 20×60 s · 曲线按真实标定姿态序列生成', font=f_sub, fill='#475569')

    row_blocks = []
    for idx, row in enumerate(candidate.all_rows, start=1):
        t0 = (idx - 1) * 60.0
        t1 = idx * 60.0
        row_blocks.append((idx, t0, t1, row, faces[idx - 1]))

    for idx, t0, t1, row, face in row_blocks:
        x0, x1 = tx(t0), tx(t1)
        d.rectangle((x0, top_strip_y, x1, top_strip_y + top_strip_h), fill=TOP_STRIP_FILL, outline=TOP_STRIP_OUTLINE, width=2)
        text_center(d, (x0, top_strip_y + 10, x1, top_strip_y + 42), f'P{idx}', f_small, '#1F2937')
        axis = row['axis']
        ang = int(round(row['angle_deg']))
        axis_label = 'X' if axis[0] != 0 else ('Y' if axis[1] != 0 else 'Z')
        motor_text = f'{axis_label}{ang:+d}°'
        text_center(d, (x0, top_strip_y + 46, x1, top_strip_y + 74), motor_text, f_tiny, '#64748B')

    for idx, t0, t1, row, face in row_blocks:
        x0, x1 = tx(t0), tx(t1)
        fc = FACE_COLORS[face['face_name']]
        d.rectangle((x0, face_strip_y, x1, face_strip_y + face_strip_h), fill=fc, outline=TOP_STRIP_OUTLINE, width=2)
        text_center(d, (x0, face_strip_y + 8, x1, face_strip_y + 36), face['face_name'], f_small, '#111827')

    panels = [
        ('横滚角 Roll', roll, '#DC2626'),
        ('俯仰角 Pitch', pitch, '#059669'),
        ('航向角 Yaw', yaw, '#2563EB'),
    ]

    for (title, data, color), top in zip(panels, panel_tops):
        ymin, ymax = nice_limits(data)
        d.rounded_rectangle((left - 26, top - 24, right + 24, top + plot_h + 52), radius=22, fill='#FCFCFD', outline='#E5E7EB', width=2)

        for idx, t0, t1, row, face in row_blocks:
            d.rectangle((tx(t0), top, tx(t1), top + plot_h), fill=FACE_COLORS[face['face_name']])

        for frac in np.linspace(0, 1, 5):
            y = top + plot_h * frac
            d.line((left, y, right, y), fill='#E5E7EB', width=2)
        for sec in range(0, 1201, 60):
            x = tx(float(sec))
            d.line((x, top, x, top + plot_h), fill='#E5E7EB', width=2)

        d.rectangle((left, top, right, top + plot_h), outline='#94A3B8', width=3)
        d.text((left, top - 56), title, font=f_panel, fill='#111827')
        d.text((right - 88, top - 52), 'deg', font=f_axis, fill='#64748B')

        for val in np.linspace(ymin, ymax, 5):
            yy = top + plot_h * (1 - (val - ymin) / (ymax - ymin))
            label = f'{val:.0f}'
            bbox = d.textbbox((0, 0), label, font=f_axis)
            tw = bbox[2] - bbox[0]
            d.text((left - tw - 18, yy - 10), label, font=f_axis, fill='#475569')

        if title == '航向角 Yaw':
            for sec in range(0, 1201, 120):
                x = tx(float(sec))
                label = f'{sec}s'
                bbox = d.textbbox((0, 0), label, font=f_axis)
                tw = bbox[2] - bbox[0]
                d.text((x - tw / 2, top + plot_h + 14), label, font=f_axis, fill='#475569')

        pts = []
        step = max(1, len(t) // 5000)
        for i in range(0, len(t), step):
            x = tx(float(t[i]))
            y = top + plot_h * (1 - (float(data[i]) - ymin) / (ymax - ymin))
            pts.append((x, y))
        if len(pts) > 1:
            d.line(pts, fill=color, width=5)

    # Legend
    legend_y = 1588
    d.text((70, legend_y - 42), '图例', font=f_panel, fill='#111827')
    legend_items = [
        ('+X 朝上', FACE_COLORS['+X']),
        ('-X 朝上', FACE_COLORS['-X']),
        ('+Y 朝上', FACE_COLORS['+Y']),
        ('-Y 朝上', FACE_COLORS['-Y']),
        ('+Z 朝上', FACE_COLORS['+Z']),
        ('-Z 朝上', FACE_COLORS['-Z']),
    ]
    x = 120
    for label, color in legend_items:
        d.rectangle((x, legend_y, x + 28, legend_y + 28), fill=color, outline='#64748B')
        d.text((x + 40, legend_y + 2), label, font=f_small, fill='#334155')
        x += 360
        if x > 2100:
            x = 120
            legend_y += 40

    note = (
        '说明：实线为实际姿态欧拉角随时间演化。'
        '这张图反映的是 corrected symmetric20 标定路径本身的姿态几何，而不是滤波误差收敛结果。'
    )
    d.text((70, 1700), note, font=f_small, fill='#475569')

    img.save(OUT_PNG)

    meta = {
        'title': '20位置标定路径姿态-时间变化曲线',
        'path_case': 'corrected symmetric20',
        'att0_deg': [0.0, 0.0, 0.0],
        'total_time_s': 1200.0,
        'n_rows': 20,
        'row_time_s': 60.0,
        'output_png': str(OUT_PNG),
        'roll_deg_range': [float(np.min(roll)), float(np.max(roll))],
        'pitch_deg_range': [float(np.min(pitch)), float(np.max(pitch))],
        'yaw_deg_range': [float(np.min(yaw)), float(np.max(yaw))],
        'faces': [{
            'row': idx,
            'label': row['label'],
            'face_name': face['face_name'],
            'axis': row['axis'],
            'angle_deg': row['angle_deg'],
        } for idx, row, face in zip(range(1, 21), candidate.all_rows, faces)],
    }
    OUT_JSON.write_text(__import__('json').dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(OUT_PNG))


if __name__ == '__main__':
    main()
