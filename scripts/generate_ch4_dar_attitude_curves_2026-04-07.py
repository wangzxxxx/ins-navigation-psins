#!/usr/bin/env python3
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import importlib.util
import sys
import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
OUT_DIR = WORKSPACE / 'tmp' / 'ch4_custom_strategy_2026-04-07'
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PNG = OUT_DIR / 'fig_ch4_dar_attitude_timecurves_2026-04-07.png'

MOD_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
spec = importlib.util.spec_from_file_location('dar12_for_curve', MOD_PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules['dar12_for_curve'] = mod
spec.loader.exec_module(mod)

att = mod.attrottt(mod.np.array([0.0, 0.0, 0.0]), mod.build_rot_paras(), 0.01)
mask = att[:, 3] <= 300.0 + 1e-9
att = att[mask]
t = att[:, 3]
# PSINS attitude order: [pitch, roll, yaw]
pitch = np.unwrap(att[:, 0]) / mod.glv.deg
roll = np.unwrap(att[:, 1]) / mod.glv.deg
yaw = np.unwrap(att[:, 2]) / mod.glv.deg

W, H = 2200, 1400
img = Image.new('RGB', (W, H), 'white')
d = ImageDraw.Draw(img)

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
f_sub = load_font(22)
f_panel = load_font(25, True)
f_axis = load_font(19)
f_small = load_font(17)

x_margin = 80

d.text((x_margin, 42), 'Chapter-4 DAR attitude-angle evolution under the strict 5-minute alignment trajectory', font=f_title, fill='#111827')
d.text((x_margin, 98), 'Curves are computed from the actual trajectory used here, but truncated to a strict 300 s window with a 50 s terminal static-lock stage.', font=f_sub, fill='#475569')

segments = [
    ('spin +720°', 30, '#DBEAFE'),
    ('spin -720°', 30, '#FDE68A'),
    ('dwell', 10, '#E5E7EB'),
    ('90° flip', 10, '#E9D5FF'),
    ('spin +720°', 30, '#DBEAFE'),
    ('spin -720°', 30, '#FDE68A'),
    ('dwell', 10, '#E5E7EB'),
    ('90° flip', 10, '#E9D5FF'),
    ('spin +720°', 30, '#DBEAFE'),
    ('spin -720°', 30, '#FDE68A'),
    ('dwell', 10, '#E5E7EB'),
    ('90° flip', 10, '#E9D5FF'),
    ('90° restore', 10, '#DDD6FE'),
    ('final static', 50, '#DCFCE7'),
]
segment_bounds = []
cum = 0.0
for label, dur, color in segments:
    segment_bounds.append((cum, cum + dur, label, color))
    cum += dur
TMAX = 300.0

plot_left = 170
plot_right = 2080

def tx(sec):
    return plot_left + (plot_right - plot_left) * sec / TMAX

# top strip
strip_y0 = 150
strip_h = 70
for t0, t1, label, color in segment_bounds:
    d.rectangle((tx(t0), strip_y0, tx(t1), strip_y0 + strip_h), fill=color, outline='white', width=2)
    if (tx(t1) - tx(t0)) > 95:
        text = label.replace('°', '')
        bbox = d.textbbox((0, 0), text, font=f_small)
        tw = bbox[2]-bbox[0]
        d.text(((tx(t0)+tx(t1)-tw)/2, strip_y0 + 24), text, font=f_small, fill='#374151')

blocks = [
    ('Z↑ modulation', 0, 70),
    ('X↑ modulation', 80, 150),
    ('Z↓ modulation', 160, 230),
    ('Z↑ terminal lock', 240, 300),
]
for name, a, b in blocks:
    xa, xb = tx(a), tx(b)
    d.line((xa, strip_y0 - 8, xb, strip_y0 - 8), fill='#64748B', width=3)
    d.line((xa, strip_y0 - 8, xa, strip_y0 - 18), fill='#64748B', width=3)
    d.line((xb, strip_y0 - 8, xb, strip_y0 - 18), fill='#64748B', width=3)
    bbox = d.textbbox((0,0), name, font=f_small)
    tw = bbox[2]-bbox[0]
    d.text(((xa+xb-tw)/2, strip_y0 - 46), name, font=f_small, fill='#334155')

panels = [
    ('Roll angle', roll, '#DC2626', (-5.0, 365.0)),
    ('Pitch angle', pitch, '#059669', (-5.0, 5.0)),
    ('Yaw angle', yaw, '#2563EB', (-760.0, 760.0)),
]
panel_tops = [280, 620, 960]
panel_h = 250

for (title, data, color, (ymin, ymax)), top in zip(panels, panel_tops):
    d.rounded_rectangle((plot_left-20, top-20, plot_right+20, top+panel_h+45), radius=18, fill='#FAFAFA', outline='#E5E7EB', width=2)
    for t0, t1, label, seg_color in segment_bounds:
        d.rectangle((tx(t0), top, tx(t1), top+panel_h), fill=seg_color)
    for frac in np.linspace(0, 1, 5):
        y = top + panel_h * frac
        d.line((plot_left, y, plot_right, y), fill='#E5E7EB', width=2)
    for sec in range(0, int(TMAX)+1, 50):
        x = tx(sec)
        d.line((x, top, x, top+panel_h), fill='#E5E7EB', width=2)
    d.rectangle((plot_left, top, plot_right, top+panel_h), outline='#94A3B8', width=3)
    d.text((plot_left, top-55), title, font=f_panel, fill='#111827')
    d.text((plot_right-140, top-52), 'deg', font=f_axis, fill='#64748B')

    tick_vals = np.linspace(ymin, ymax, 5)
    for val in tick_vals:
        yy = top + panel_h * (1 - (val - ymin) / (ymax - ymin))
        label = f'{val:.0f}'
        bbox = d.textbbox((0,0), label, font=f_axis)
        tw = bbox[2]-bbox[0]
        d.text((plot_left - tw - 18, yy - 10), label, font=f_axis, fill='#475569')
    if title == 'Yaw angle':
        for sec in range(0, int(TMAX)+1, 50):
            x = tx(sec)
            label = f'{sec}s'
            bbox = d.textbbox((0,0), label, font=f_axis)
            tw = bbox[2]-bbox[0]
            d.text((x - tw/2, top + panel_h + 12), label, font=f_axis, fill='#475569')
        d.text((plot_right - 80, top + panel_h + 12), 'time', font=f_axis, fill='#475569')

    pts = []
    step = 10
    for i in range(0, len(t), step):
        x = tx(float(t[i]))
        y = top + panel_h * (1 - (float(data[i]) - ymin) / (ymax - ymin))
        pts.append((x, y))
    if len(pts) > 1:
        d.line(pts, fill=color, width=5)

legend_y = 1280
legend_items = [
    ('roll', '#DC2626'), ('pitch', '#059669'), ('yaw', '#2563EB'),
    ('spin segment', '#DBEAFE'), ('reverse spin', '#FDE68A'), ('dwell / flip / static', '#E5E7EB')
]
x = 120
for label, color in legend_items:
    d.rectangle((x, legend_y, x+26, legend_y+26), fill=color, outline='#64748B')
    d.text((x+38, legend_y+2), label, font=f_small, fill='#334155')
    x += 240

note = 'Interpretation: yaw captures repeated ±720° modulation, roll captures the 90° reorientation sequence, and the final 50 s static-lock stage appears as a flat tail.'
d.text((80, 1330), note, font=f_small, fill='#475569')

img.save(OUT_PNG)
print(OUT_PNG)
