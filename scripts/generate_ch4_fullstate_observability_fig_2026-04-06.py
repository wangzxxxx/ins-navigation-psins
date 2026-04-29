#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

WORKSPACE = Path('/root/.openclaw/workspace')
IN_JSON = WORKSPACE / 'tmp' / 'ch4_llm_scd_rewrite_2026-04-06' / 'ch4_plain24_llm_scd_observability_2026-04-06.json'
OUT_PNG = WORKSPACE / 'tmp' / 'ch4_llm_scd_rewrite_2026-04-06' / 'fig_ch4_llm_scd_observability_competition.png'

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


def panel(draw, box, title, labels, values, colors, ymax=1.0, rotate=False):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill='white', outline='#d1d5db', width=2)
    title_font = load_font(28, bold=True)
    label_font = load_font(18)
    small_font = load_font(16)
    text(draw, (x0 + 18, y0 + 18), title, title_font, '#111827')

    pad_l, pad_r, pad_t, pad_b = 78, 22, 70, 100 if rotate else 70
    px0, py0 = x0 + pad_l, y0 + pad_t
    px1, py1 = x1 - pad_r, y1 - pad_b
    pw, ph = px1 - px0, py1 - py0

    for i in range(6):
        frac = i / 5
        yy = py1 - ph * frac
        draw.line((px0, yy, px1, yy), fill='#e5e7eb', width=1)
        text(draw, (px0 - 10, yy), f'{frac:.1f}', small_font, '#6b7280', anchor='rm')
    draw.line((px0, py0, px0, py1), fill='#111827', width=2)
    draw.line((px0, py1, px1, py1), fill='#111827', width=2)

    n = max(len(labels), 1)
    g_w = pw / n
    bar_w = max(8, int(g_w * 0.72))
    for i, (lab, val, color) in enumerate(zip(labels, values, colors)):
        cx = px0 + g_w * (i + 0.5)
        bh = ph * max(0.0, min(ymax, float(val))) / ymax
        bx0 = cx - bar_w / 2
        by0 = py1 - bh
        draw.rounded_rectangle((bx0, by0, bx0 + bar_w, py1), radius=4, fill=color)
        if rotate:
            # draw rotated label on transparent tile
            tile = Image.new('RGBA', (180, 40), (255, 255, 255, 0))
            td = ImageDraw.Draw(tile)
            td.text((0, 8), lab, font=label_font, fill='#374151')
            tile = tile.rotate(55, expand=1)
            img.alpha_composite(tile, (int(cx - tile.width / 2), int(py1 + 8)))
        else:
            text(draw, (cx, py1 + 26), lab, label_font, '#374151', anchor='mm')


def panel_dual(draw, box, title, labels, vals_a, vals_b, colors, ymax=1.0):
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill='white', outline='#d1d5db', width=2)
    title_font = load_font(28, bold=True)
    label_font = load_font(18)
    small_font = load_font(16)
    text(draw, (x0 + 18, y0 + 18), title, title_font, '#111827')

    pad_l, pad_r, pad_t, pad_b = 78, 22, 70, 100
    px0, py0 = x0 + pad_l, y0 + pad_t
    px1, py1 = x1 - pad_r, y1 - pad_b
    pw, ph = px1 - px0, py1 - py0

    for i in range(6):
        frac = i / 5
        yy = py1 - ph * frac
        draw.line((px0, yy, px1, yy), fill='#e5e7eb', width=1)
        text(draw, (px0 - 10, yy), f'{frac:.1f}', small_font, '#6b7280', anchor='rm')
    draw.line((px0, py0, px0, py1), fill='#111827', width=2)
    draw.line((px0, py1, px1, py1), fill='#111827', width=2)

    n = max(len(labels), 1)
    g_w = pw / n
    total_w = max(16, int(g_w * 0.76))
    bw = total_w / 2 - 3
    for i, lab in enumerate(labels):
        cx = px0 + g_w * (i + 0.5)
        for j, val in enumerate((vals_a[i], vals_b[i])):
            if val is None:
                continue
            bh = ph * max(0.0, min(ymax, float(val))) / ymax
            bx0 = cx - total_w / 2 + j * (bw + 6)
            by0 = py1 - bh
            draw.rounded_rectangle((bx0, by0, bx0 + bw, py1), radius=4, fill=colors[j])
        text(draw, (cx, py1 + 26), lab, label_font, '#374151', anchor='mm')

    # legend
    legend_y = y0 + 52
    draw.rounded_rectangle((x1 - 310, legend_y - 8, x1 - 18, legend_y + 28), radius=10, fill='#f8fafc', outline='#e5e7eb')
    draw.rectangle((x1 - 290, legend_y, x1 - 272, legend_y + 18), fill=colors[0])
    text(draw, (x1 - 264, legend_y + 9), 'formal direct', load_font(16), '#374151', anchor='lm')
    draw.rectangle((x1 - 146, legend_y, x1 - 128, legend_y + 18), fill=colors[1])
    text(draw, (x1 - 120, legend_y + 9), 'conditional/loss', load_font(16), '#374151', anchor='lm')


if __name__ == '__main__':
    payload = json.loads(IN_JSON.read_text(encoding='utf-8'))
    family_summary = payload['family_summary']
    per_state = payload['per_state']

    width, height = 1800, 1280
    img = Image.new('RGBA', (width, height), '#f8fafc')
    draw = ImageDraw.Draw(img)
    title_font = load_font(42, bold=True)
    sub_font = load_font(22)
    text(draw, (50, 40), 'Current DAR / plain24 observability with full-state bar chart', title_font, '#111827')
    text(draw, (50, 92), 'Formal direct observability is shown for all 24 states; conditional / competition views are shown for primary states.', sub_font, '#374151')

    fam_labels = list(family_summary.keys())
    fam_direct = [family_summary[k]['formal_direct_score'] for k in fam_labels]
    fam_cond = [family_summary[k]['conditional_primary_score'] for k in fam_labels]
    fam_cond = [None if v is None else float(v) for v in fam_cond]

    state_labels = [row['state'] for row in per_state]
    state_direct = [row['formal_direct_score'] for row in per_state]
    state_colors = [FAMILY_COLORS[row['family']] for row in per_state]
    primary_labels = [row['state'] for row in per_state if row['conditional_primary_score'] is not None]
    primary_cond = [row['conditional_primary_score'] for row in per_state if row['conditional_primary_score'] is not None]
    primary_loss = [row['competition_loss'] for row in per_state if row['competition_loss'] is not None]

    panel_dual(draw, (40, 140, 1760, 410), 'A. Family-level observability summary', fam_labels, fam_direct, fam_cond, ['#2563eb', '#f97316'])
    panel(draw, (40, 440, 1760, 880), 'B. Full 24-state formal direct observability bar chart', state_labels, state_direct, state_colors, rotate=True)
    panel_dual(draw, (40, 910, 1760, 1230), 'C. Primary-state conditional observability and competition loss', primary_labels, primary_cond, primary_loss, ['#f97316', '#ef4444'])

    note_font = load_font(20)
    note = payload.get('best_llm_result_note') or {}
    if note:
        note_txt = f"Best current LLM+SCD candidate: {note['candidate']} | yaw abs mean {note['plain24_yaw_abs_mean_arcsec']:.3f}\" -> {note['yaw_abs_mean_arcsec']:.3f}\""
        text(draw, (50, 1260), note_txt, note_font, '#9a3412')

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    img.convert('RGB').save(OUT_PNG, quality=95)
    print(OUT_PNG)
