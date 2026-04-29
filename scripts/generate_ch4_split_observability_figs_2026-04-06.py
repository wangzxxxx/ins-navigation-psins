#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

WORKSPACE = Path('/root/.openclaw/workspace')
IN_JSON = WORKSPACE / 'tmp' / 'ch4_llm_scd_rewrite_2026-04-06' / 'ch4_plain24_llm_scd_observability_2026-04-06.json'
OUT_B = WORKSPACE / 'tmp' / 'ch4_llm_scd_rewrite_2026-04-06' / 'fig_ch4_llm_scd_observability_fullstate_B.png'
OUT_C = WORKSPACE / 'tmp' / 'ch4_llm_scd_rewrite_2026-04-06' / 'fig_ch4_llm_scd_observability_competition_C.png'

FAMILY_COLORS = {
    'phi': '#2563eb',
    'dV': '#0891b2',
    'eb': '#7c3aed',
    'db': '#ea580c',
    'ng': '#dc2626',
    'xa': '#65a30d',
    'kg': '#c2410c',
    'ka': '#0f766e',
}


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


def rotated_label(label: str, font, fill='#374151', angle=50):
    tile = Image.new('RGBA', (190, 52), (255, 255, 255, 0))
    td = ImageDraw.Draw(tile)
    td.text((0, 10), label, font=font, fill=fill)
    return tile.rotate(angle, expand=1)


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


def make_b(payload):
    width, height = 1900, 980
    img = Image.new('RGBA', (width, height), '#ffffff')
    draw = ImageDraw.Draw(img)
    title_font = load_font(40, True)
    small_font = load_font(17)
    label_font = load_font(14)

    text(draw, (50, 42), 'B. Full 24-state formal direct observability bar chart', title_font, '#111827')

    # plot area
    px0, py0, px1, py1 = 115, 150, 1840, 770
    draw_axes(draw, (px0, py0, px1, py1))

    per_state = payload['per_state']
    labels = [row['state'] for row in per_state]
    values = [row['formal_direct_score'] for row in per_state]
    colors = [FAMILY_COLORS[row['family']] for row in per_state]

    g_w = (px1 - px0) / len(labels)
    bar_w = max(10, int(g_w * 0.46))
    family_spans = [('phi', 0, 2), ('dV', 3, 5), ('eb', 6, 8), ('db', 9, 11), ('ng', 12, 14), ('xa', 15, 17), ('kg', 18, 20), ('ka', 21, 23)]
    for fam, a, b in family_spans:
        sx0 = px0 + g_w * a
        sx1 = px0 + g_w * (b + 1)
        text(draw, ((sx0 + sx1) / 2, py0 - 16), fam, small_font, '#4b5563', anchor='mb')
        if a > 0:
            draw.line((sx0, py0, sx0, py1), fill='#cbd5e1', width=2)

    for i, (lab, val, color) in enumerate(zip(labels, values, colors)):
        cx = px0 + g_w * (i + 0.5)
        bh = (py1 - py0) * max(0.0, min(1.0, float(val)))
        bx0 = cx - bar_w / 2
        by0 = py1 - bh
        draw.rounded_rectangle((bx0, by0, bx0 + bar_w, py1), radius=4, fill=color, outline='#ffffff', width=1)
        text(draw, (cx, py1 + 16), lab, label_font, '#374151', anchor='ma')

    # y/x labels
    paste_rotated_text(img, (44, (py0 + py1) / 2), 'formal direct score', load_font(22, True), '#111827', angle=90)
    OUT_B.parent.mkdir(parents=True, exist_ok=True)
    img.convert('RGB').save(OUT_B, quality=95)


def make_c(payload):
    width, height = 1900, 980
    img = Image.new('RGBA', (width, height), '#ffffff')
    draw = ImageDraw.Draw(img)
    title_font = load_font(40, True)
    small_font = load_font(17)
    label_font = load_font(14)

    text(draw, (50, 42), 'C. Primary-state direct / conditional / competition-loss comparison', title_font, '#111827')

    px0, py0, px1, py1 = 115, 170, 1840, 770
    draw_axes(draw, (px0, py0, px1, py1))

    per_state = [row for row in payload['per_state'] if row['primary_direct_score'] is not None]
    labels = [row['state'] for row in per_state]
    dvals = [row['primary_direct_score'] for row in per_state]
    cvals = [row['conditional_primary_score'] for row in per_state]
    lvals = [row['competition_loss'] for row in per_state]

    g_w = (px1 - px0) / len(labels)
    total_w = max(20, int(g_w * 0.82))
    bar_w = total_w / 3 - 4
    series = [
        ('primary direct', dvals, '#2563eb'),
        ('conditional', cvals, '#f97316'),
        ('competition loss', lvals, '#ef4444'),
    ]

    family_spans = [('phi', 0, 2), ('dV', 3, 5), ('eb', 6, 8), ('db', 9, 11), ('ng', 12, 14), ('xa', 15, 17)]
    for fam, a, b in family_spans:
        sx0 = px0 + g_w * a
        sx1 = px0 + g_w * (b + 1)
        text(draw, ((sx0 + sx1) / 2, py0 - 16), fam, small_font, '#4b5563', anchor='mb')
        if a > 0:
            draw.line((sx0, py0, sx0, py1), fill='#cbd5e1', width=2)

    for i, lab in enumerate(labels):
        cx = px0 + g_w * (i + 0.5)
        for j, (_, vals, color) in enumerate(series):
            val = float(vals[i])
            bh = (py1 - py0) * max(0.0, min(1.0, val))
            bx0 = cx - total_w / 2 + j * (bar_w + 4)
            by0 = py1 - bh
            draw.rounded_rectangle((bx0, by0, bx0 + bar_w, py1), radius=4, fill=color, outline='#ffffff', width=1)
        text(draw, (cx, py1 + 16), lab, label_font, '#374151', anchor='ma')

    # legend
    lx, ly = 1260, 120
    draw.rounded_rectangle((lx, ly, 1835, ly + 52), radius=12, fill='#ffffff', outline='#d1d5db')
    for idx, (name, _, color) in enumerate(series):
        x = lx + 20 + idx * 185
        draw.rectangle((x, ly + 16, x + 18, ly + 34), fill=color)
        text(draw, (x + 28, ly + 25), name, small_font, '#374151', anchor='lm')

    paste_rotated_text(img, (44, (py0 + py1) / 2), 'score (same normalization)', load_font(22, True), '#111827', angle=90)

    OUT_C.parent.mkdir(parents=True, exist_ok=True)
    img.convert('RGB').save(OUT_C, quality=95)


if __name__ == '__main__':
    payload = json.loads(IN_JSON.read_text(encoding='utf-8'))
    make_b(payload)
    make_c(payload)
    print(OUT_B)
    print(OUT_C)
