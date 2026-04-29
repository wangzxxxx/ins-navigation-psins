#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

WORKSPACE = Path('/root/.openclaw/workspace')
IN_JSON = WORKSPACE / 'tmp' / 'ch3_sym20_42state_observability_2026-04-07' / 'ch3_sym20_42state_observability_2026-04-07.json'
OUT_DIR = WORKSPACE / 'tmp' / 'ch3_sym20_42state_observability_2026-04-07'
OUT_A = OUT_DIR / 'fig_ch3_sym20_42state_observability_A_2026-04-07.png'
OUT_B = OUT_DIR / 'fig_ch3_sym20_42state_observability_B_2026-04-07.png'


def load_font(size: int, bold: bool = False):
    candidates = [
        '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc' if bold else '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
        '/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc' if bold else '/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc',
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


def make_panel_a(payload: dict):
    rows = payload['cases']['full_q']['state_rows']
    width, height = 2300, 980
    img = Image.new('RGBA', (width, height), '#ffffff')
    draw = ImageDraw.Draw(img)

    title_font = load_font(40, True)
    group_font = load_font(17)
    label_font = load_font(13)

    text(draw, (60, 44), 'A. 20位置 42状态单状态可观性分布图', title_font, '#111827')

    px0, py0, px1, py1 = 110, 150, 2230, 780
    draw_axes(draw, (px0, py0, px1, py1))

    labels = [r['state'] for r in rows]
    values = [r['normalized_score'] for r in rows]
    colors = [r['color'] for r in rows]

    # family spans inferred from contiguous blocks
    spans = []
    start = 0
    while start < len(rows):
        fam = rows[start]['family']
        color = rows[start]['color']
        end = start
        while end + 1 < len(rows) and rows[end + 1]['family'] == fam:
            end += 1
        spans.append((fam, start, end, color))
        start = end + 1

    g_w = (px1 - px0) / len(labels)
    bar_w = max(10, int(g_w * 0.52))
    for fam, a, b, color in spans:
        sx0 = px0 + g_w * a
        sx1 = px0 + g_w * (b + 1)
        if a > 0:
            draw.line((sx0, py0, sx0, py1), fill='#cbd5e1', width=2)
        text(draw, ((sx0 + sx1) / 2, py0 - 8), fam, group_font, '#4b5563', anchor='mb')

    for i, (lab, val, color) in enumerate(zip(labels, values, colors)):
        cx = px0 + g_w * (i + 0.5)
        bh = (py1 - py0) * max(0.0, min(1.0, float(val)))
        bx0 = cx - bar_w / 2
        by0 = py1 - bh
        draw.rounded_rectangle((bx0, by0, bx0 + bar_w, py1), radius=3, fill=color, outline='#ffffff', width=1)
        text(draw, (cx, py1 + 16), lab, label_font, '#374151', anchor='ma')

    paste_rotated_text(img, (46, (py0 + py1) / 2), 'normalized observability score', load_font(24, True), '#111827', angle=90)
    img.convert('RGB').save(OUT_A, quality=95)


def make_panel_b(payload: dict):
    full_q = payload['cases']['full_q']['family_summary']
    no_q = {r['family']: r for r in payload['cases']['no_q']['family_summary']}

    width, height = 1700, 900
    img = Image.new('RGBA', (width, height), '#ffffff')
    draw = ImageDraw.Draw(img)

    title_font = load_font(40, True)
    label_font = load_font(18)
    legend_font = load_font(18)

    text(draw, (60, 44), 'B. 20位置 42状态参数族平均可观性对比图', title_font, '#111827')

    px0, py0, px1, py1 = 120, 150, 1640, 730
    draw_axes(draw, (px0, py0, px1, py1))

    fam_labels = [r['family'] for r in full_q]
    g_w = (px1 - px0) / len(fam_labels)
    total_w = max(28, int(g_w * 0.72))
    bw = total_w / 2 - 4

    for i, fam in enumerate(fam_labels):
        cx = px0 + g_w * (i + 0.5)
        vals = [no_q[fam]['mean_normalized_score'], next(r['mean_normalized_score'] for r in full_q if r['family'] == fam)]
        cols = ['#2563eb', '#f97316']
        for j, (val, col) in enumerate(zip(vals, cols)):
            bh = (py1 - py0) * max(0.0, min(1.0, float(val)))
            bx0 = cx - total_w / 2 + j * (bw + 8)
            by0 = py1 - bh
            draw.rounded_rectangle((bx0, by0, bx0 + bw, py1), radius=4, fill=col)
        text(draw, (cx, py1 + 18), fam, label_font, '#374151', anchor='ma')

    paste_rotated_text(img, (50, (py0 + py1) / 2), 'mean normalized score', load_font(24, True), '#111827', angle=90)

    lx, ly = 1180, 92
    draw.rounded_rectangle((lx, ly, 1600, ly + 58), radius=12, fill='#ffffff', outline='#d1d5db')
    draw.rectangle((lx + 18, ly + 20, lx + 36, ly + 38), fill='#2563eb')
    text(draw, (lx + 46, ly + 29), 'Q = 0', legend_font, '#374151', anchor='lm')
    draw.rectangle((lx + 168, ly + 20, lx + 186, ly + 38), fill='#f97316')
    text(draw, (lx + 196, ly + 29), 'full-Q', legend_font, '#374151', anchor='lm')

    img.convert('RGB').save(OUT_B, quality=95)


def main():
    payload = json.loads(IN_JSON.read_text(encoding='utf-8'))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    make_panel_a(payload)
    make_panel_b(payload)
    print(OUT_A)
    print(OUT_B)


if __name__ == '__main__':
    main()
