#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path('/root/.openclaw/workspace/tmp/ch4_custom_strategy_2026-04-07')
OUT_DIR.mkdir(parents=True, exist_ok=True)
PNG = OUT_DIR / 'fig_ch4_dar_strategy_custom_2026-04-07.png'

W, H = 2200, 1300
BG = 'white'
img = Image.new('RGB', (W, H), BG)
d = ImageDraw.Draw(img)

# font helpers
FONT_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    '/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf',
]
BOLD_CANDIDATES = [
    '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
    '/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf',
]

def load_font(size, bold=False):
    cands = BOLD_CANDIDATES if bold else FONT_CANDIDATES
    for p in cands:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

f_title = load_font(42, True)
f_sub = load_font(22, False)
f_h = load_font(26, True)
f_mid = load_font(21, False)
f_small = load_font(18, False)
f_bold = load_font(19, True)

# title
x_margin = 70

d.text((x_margin, 48), 'Chapter-4 custom DAR trajectory for alignment observability analysis', font=f_title, fill='#111827')
d.text((x_margin, 104), 'This is the actual path used in the current alignment experiments, not a generic rotation-modulation principle sketch.', font=f_sub, fill='#475569')

# rounded rectangle helper

def round_rect(xy, radius=24, fill='#F8FAFC', outline='#CBD5E1', width=3):
    d.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)

# state cards
card_y = 180
card_w = 420
card_h = 230
card_xs = [70, 590, 1110, 1630]
cards = [
    ('Block 1', 'Z UP', 'spin +720° / -720°\ndwell 10 s', '#EFF6FF'),
    ('Block 2', 'X UP', 'spin +720° / -720°\ndwell 10 s', '#EFF6FF'),
    ('Block 3', 'Z DOWN', 'spin +720° / -720°\ndwell 10 s', '#EFF6FF'),
    ('Terminal lock', 'Z UP', 'restore +90°\nstatic 100 s', '#ECFDF5'),
]
for (title, orient, detail, fill), x in zip(cards, card_xs):
    round_rect((x, card_y, x + card_w, card_y + card_h), fill=fill)
    d.text((x + 28, card_y + 24), title, font=f_h, fill='#111827')
    ow, oh = d.textbbox((0, 0), orient, font=load_font(34, True))[2:]
    d.text((x + (card_w - ow) / 2, card_y + 84), orient, font=load_font(34, True), fill='#1D4ED8')
    lines = detail.split('\n')
    for i, line in enumerate(lines):
        tw = d.textbbox((0, 0), line, font=f_mid)[2]
        d.text((x + (card_w - tw) / 2, card_y + 150 + i * 28), line, font=f_mid, fill='#374151')

# arrows between cards
arrow_y = card_y + card_h // 2
for i in range(3):
    x0 = card_xs[i] + card_w + 20
    x1 = card_xs[i + 1] - 20
    d.line((x0, arrow_y, x1, arrow_y), fill='#7C3AED', width=6)
    d.polygon([(x1, arrow_y), (x1 - 24, arrow_y - 13), (x1 - 24, arrow_y + 13)], fill='#7C3AED')
    label = '+90° flip about y' if i < 2 else 'restore +90°'
    tw = d.textbbox((0, 0), label, font=f_small)[2]
    d.text(((x0 + x1 - tw) / 2, arrow_y - 46), label, font=f_small, fill='#6D28D9')

# timeline
segments = [
    ('+720° spin', 30, '#4F46E5'),
    ('-720° spin', 30, '#F59E0B'),
    ('dwell', 10, '#9CA3AF'),
    ('+90° flip', 10, '#8B5CF6'),
    ('+720° spin', 30, '#4F46E5'),
    ('-720° spin', 30, '#F59E0B'),
    ('dwell', 10, '#9CA3AF'),
    ('+90° flip', 10, '#8B5CF6'),
    ('+720° spin', 30, '#4F46E5'),
    ('-720° spin', 30, '#F59E0B'),
    ('dwell', 10, '#9CA3AF'),
    ('+90° flip', 10, '#8B5CF6'),
    ('+90° restore', 10, '#A855F7'),
    ('final static lock', 100, '#10B981'),
]

d.text((x_margin, 500), 'Timeline (total 350 s)', font=load_font(30, True), fill='#111827')
d.text((x_margin, 545), 'Three modulation blocks are interleaved with 90° flips; the path ends with a long static locking window.', font=f_sub, fill='#475569')

tl_x0, tl_x1 = 70, 2130
tl_y0, tl_h = 620, 110
total = sum(dur for _, dur, _ in segments)
cur = tl_x0
for label, dur, color in segments:
    w = int((tl_x1 - tl_x0) * dur / total)
    d.rectangle((cur, tl_y0, cur + w, tl_y0 + tl_h), fill=color, outline='white', width=3)
    if w > 120:
        lines = [label, f'{dur}s']
        for j, line in enumerate(lines):
            bbox = d.textbbox((0, 0), line, font=f_small)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            d.text((cur + (w - tw) / 2, tl_y0 + 26 + j * 24), line, font=f_small,
                   fill='white' if color not in ['#9CA3AF'] else '#111827')
    cur += w

# ticks
for t in range(0, total + 1, 50):
    x = tl_x0 + int((tl_x1 - tl_x0) * t / total)
    d.line((x, tl_y0 + tl_h, x, tl_y0 + tl_h + 16), fill='#475569', width=2)
    label = f'{t}s'
    tw = d.textbbox((0, 0), label, font=f_small)[2]
    d.text((x - tw / 2, tl_y0 + tl_h + 22), label, font=f_small, fill='#475569')

# grouped stages
stages = [
    ('Modulation block 1', 0, 70),
    ('Modulation block 2', 80, 150),
    ('Modulation block 3', 160, 230),
    ('Terminal locking stage', 240, 350),
]
for name, a, b in stages:
    xa = tl_x0 + int((tl_x1 - tl_x0) * a / total)
    xb = tl_x0 + int((tl_x1 - tl_x0) * b / total)
    y = 790
    d.line((xa, y, xb, y), fill='#64748B', width=3)
    d.line((xa, y, xa, y - 18), fill='#64748B', width=3)
    d.line((xb, y, xb, y - 18), fill='#64748B', width=3)
    tw = d.textbbox((0, 0), name, font=f_small)[2]
    d.text(((xa + xb - tw) / 2, y + 10), name, font=f_small, fill='#334155')

# takeaways
box_y = 930
round_rect((70, box_y, 2130, 1200), fill='#F8FAFC')
d.text((95, box_y + 24), 'What makes this OUR trajectory rather than a generic modulation illustration', font=load_font(28, True), fill='#111827')
items = [
    '• symmetric spin pairs are used to modulate residual errors while limiting one-direction accumulation',
    '• short dwells are deliberately preserved so the filter gets cleaner observation windows after each strong excitation burst',
    '• repeated 90° flips re-distribute gravity / earth-rate projection geometry across axes instead of staying in one spin orientation',
    '• the long final static segment is reserved for tail convergence and final yaw locking rather than continuous spinning until the end',
]
for i, line in enumerate(items):
    d.text((110, box_y + 86 + i * 42), line, font=f_mid, fill='#374151')

img.save(PNG)
print(PNG)
