#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from xml.sax.saxutils import escape

WORKSPACE = Path('/root/.openclaw/workspace')
BASE_SCRIPT = WORKSPACE / 'psins_method_bench' / 'scripts' / 'generate_dualpath_three_method_state_convergence_v2_first3_2026-04-09.py'
OUT_DIR = WORKSPACE / 'tmp' / 'psins_dualpath_three_method_att_sigma_preview_2026-04-09'
OUT_JSON = OUT_DIR / 'summary.json'
MAX_POINTS = 1200
LEGEND_LINE_WIDTH = 2.8
TRACE_DT_S = 0.2


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


bp = load_module('dualpath_sigma_base_20260409', BASE_SCRIPT)


def downsample(x, ys, max_points=MAX_POINTS):
    n = len(x)
    if n <= max_points:
        return x, ys
    idx = [round(i * (n - 1) / (max_points - 1)) for i in range(max_points)]
    dedup = []
    seen = set()
    for i in idx:
        if i not in seen:
            seen.add(i)
            dedup.append(i)
    return [x[i] for i in dedup], [[arr[i] for i in dedup] for arr in ys]


def polyline(xs, ys):
    return ' '.join(f'{x:.2f},{y:.2f}' for x, y in zip(xs, ys))


def fmt_y(v: float) -> str:
    if abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 1e-3):
        return f'{v:.2e}'
    return f'{v:.4f}'


def fmt_time_tick(v: float) -> str:
    return str(int(round(v)))


def nice_ticks(vmin: float, vmax: float, n: int = 5):
    if math.isclose(vmin, vmax, rel_tol=0.0, abs_tol=1e-15):
        delta = 1.0 if abs(vmin) < 1e-12 else abs(vmin) * 0.1
        vmin -= delta
        vmax += delta
    return [vmin + (vmax - vmin) * i / (n - 1) for i in range(n)]


def percentile(seq, q):
    if not seq:
        return 0.0
    arr = sorted(seq)
    if len(arr) == 1:
        return arr[0]
    pos = (len(arr) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    frac = pos - lo
    return arr[lo] * (1 - frac) + arr[hi] * frac


def robust_range(values, low_q=0.0, high_q=1.0, min_pad_frac=0.08):
    if not values:
        return -1.0, 1.0
    lo = percentile(values, low_q)
    hi = percentile(values, high_q)
    lo = min(lo, min(values))
    hi = max(hi, max(values))
    if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-15):
        delta = 1.0 if abs(lo) < 1e-12 else abs(lo) * 0.1
        lo -= delta
        hi += delta
    pad = max((hi - lo) * min_pad_frac, 1e-12)
    return lo - pad, hi + pad


def log_ticks(vmin, vmax, n=5):
    if vmin <= 0:
        vmin = 1e-12
    if vmax <= 0:
        vmax = 1.0
    lo = math.log10(vmin)
    hi = math.log10(vmax)
    if math.isclose(lo, hi, abs_tol=1e-12):
        lo -= 1.0
        hi += 1.0
    return [10 ** (lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def map_log(v, vmin, vmax, y0, y1):
    v = max(v, vmin)
    lo = math.log10(vmin)
    hi = math.log10(vmax)
    if math.isclose(lo, hi, abs_tol=1e-15):
        return (y0 + y1) / 2
    return y1 - (math.log10(v) - lo) / (hi - lo) * (y1 - y0)


def draw_panel(parts, rect, title, x_ticks, y_ticks, x_mapper, y_mapper, x_label, y_label, iter_bounds=None):
    x0, y0, x1, y1 = rect
    parts.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="white" stroke="#d0d7de"/>')
    parts.append(f'<text x="{x0 + 8}" y="{y0 + 20}" class="panel-title">{escape(title)}</text>')
    plot_top = y0 + 28
    plot_bottom = y1 - 36
    plot_left = x0 + 78
    plot_right = x1 - 18
    for yt in y_ticks:
        yy = y_mapper(yt)
        parts.append(f'<line x1="{plot_left}" y1="{yy:.2f}" x2="{plot_right}" y2="{yy:.2f}" class="grid"/>')
        parts.append(f'<text x="{plot_left - 12}" y="{yy + 4:.2f}" class="tick" text-anchor="end">{escape(fmt_y(yt))}</text>')
    for xt in x_ticks:
        xx = x_mapper(xt)
        parts.append(f'<line x1="{xx:.2f}" y1="{plot_top}" x2="{xx:.2f}" y2="{plot_bottom}" class="grid"/>')
        parts.append(f'<text x="{xx:.2f}" y="{plot_bottom + 18}" class="tick" text-anchor="middle">{escape(fmt_time_tick(xt))}</text>')
    if iter_bounds:
        for bound in iter_bounds[:-1]:
            xx = x_mapper(bound)
            parts.append(f'<line x1="{xx:.2f}" y1="{plot_top}" x2="{xx:.2f}" y2="{plot_bottom}" class="iter"/>')
    parts.append(f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" class="axis"/>')
    parts.append(f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" class="axis"/>')
    parts.append(f'<text x="{(plot_left + plot_right)/2:.2f}" y="{y1 - 8}" class="small" text-anchor="middle">{escape(x_label)}</text>')
    y_label_x = x0 + 8
    y_label_y = (plot_top + plot_bottom) / 2
    parts.append(f'<text x="{y_label_x}" y="{y_label_y:.2f}" class="small" transform="rotate(-90 {y_label_x} {y_label_y:.2f})" text-anchor="middle">{escape(y_label)}</text>')


def render_sigma_preview(out_svg: Path, out_png: Path, axis_label: str, series_list):
    width = 1500
    height = 470
    gap = 18
    panel_w = 464
    panel_h = 320
    origin_x = 30
    origin_y = 40
    panels = [
        (origin_x, origin_y, origin_x + panel_w, origin_y + panel_h),
        (origin_x + panel_w + gap, origin_y, origin_x + 2 * panel_w + gap, origin_y + panel_h),
    ]

    all_x = [x for s in series_list for x in s['x']]
    x_min, x_max = min(all_x), max(all_x)
    x_ticks = nice_ticks(x_min, x_max, 6)
    iter_bounds = series_list[0]['iter_bounds_s'] if series_list else []
    if len(iter_bounds) >= 2:
        tail_start = max(iter_bounds[-2], x_max - 100.0)
    else:
        tail_start = max(x_min, x_max - 100.0)
    tail_end = x_max
    tail_ticks = nice_ticks(tail_start, tail_end, 5)

    sigma_all = [max(v, 1e-12) for s in series_list for v in s['y']]
    full_ymin, full_ymax = robust_range(sigma_all, low_q=0.0, high_q=1.0, min_pad_frac=0.08)
    tail_vals = [max(v, 1e-12) for s in series_list for x, v in zip(s['x'], s['y']) if x >= tail_start]
    tail_ymin, tail_ymax = robust_range(tail_vals, low_q=0.01, high_q=0.99, min_pad_frac=0.12)
    full_ticks = log_ticks(max(full_ymin, 1e-12), full_ymax, 5)
    tail_ticks_y = log_ticks(max(tail_ymin, 1e-12), tail_ymax, 5)

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<style>text{font-family:Arial,Helvetica,sans-serif;} .title{font-size:22px;font-weight:bold;} .panel-title{font-size:15px;font-weight:bold;fill:#223;} .tick{font-size:11px;fill:#555;} .small{font-size:12px;fill:#444;} .axis{stroke:#333;stroke-width:1;} .grid{stroke:#e5e7eb;stroke-width:1;} .iter{stroke:#94a3b8;stroke-width:1;stroke-dasharray:6 4;} .legend{font-size:12px;fill:#222;}</style>')
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    parts.append(f'<text x="18" y="26" class="title">{escape(axis_label)}</text>')

    def make_mappers(rect, xmin, xmax, ymin, ymax):
        x0, y0, x1, y1 = rect
        plot_top = y0 + 28
        plot_bottom = y1 - 36
        plot_left = x0 + 78
        plot_right = x1 - 18
        def mx(x):
            if math.isclose(xmin, xmax, abs_tol=1e-15):
                return (plot_left + plot_right) / 2
            return plot_left + (x - xmin) / (xmax - xmin) * (plot_right - plot_left)
        def my(y):
            return map_log(y, max(ymin, 1e-12), ymax, plot_top, plot_bottom)
        return mx, my

    mx1, my1 = make_mappers(panels[0], x_min, x_max, full_ymin, full_ymax)
    draw_panel(parts, panels[0], 'Approx. σ · first 3 rounds', x_ticks, full_ticks, mx1, my1, 'time (s)', 'sigma', iter_bounds=iter_bounds)

    mx2, my2 = make_mappers(panels[1], tail_start, tail_end, tail_ymin, tail_ymax)
    tail_iter = [b for b in iter_bounds if tail_start <= b <= tail_end]
    draw_panel(parts, panels[1], 'Approx. σ · round-3 last 100 s', tail_ticks, tail_ticks_y, mx2, my2, 'time (s)', 'sigma', iter_bounds=tail_iter)

    for series in series_list:
        style = bp.line_style(series['group_key'])
        xs1, [ys1] = downsample(series['x'], [series['y']])
        parts.append(f'<polyline points="{polyline([mx1(v) for v in xs1], [my1(max(v, max(full_ymin,1e-12))) for v in ys1])}" fill="none" {style}/>')
        tail_x = [x for x in series['x'] if x >= tail_start]
        tail_y = [y for x, y in zip(series['x'], series['y']) if x >= tail_start]
        xs2, [ys2] = downsample(tail_x, [tail_y], max_points=800)
        parts.append(f'<polyline points="{polyline([mx2(v) for v in xs2], [my2(max(v, max(tail_ymin,1e-12))) for v in ys2])}" fill="none" {style}/>')

    legend_x = 22
    legend_y = 382
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="980" height="42" fill="white" stroke="#d0d7de"/>')
    for i, series in enumerate(series_list):
        x = legend_x + 18 + i * 310
        y = legend_y + 22
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" {bp.line_style(series["group_key"], legend=True)}/>')
        parts.append(f'<text x="{x + 38}" y="{y + 4}" class="legend">{escape(series["label"])}</text>')

    parts.append('</svg>')
    out_svg.write_text('\n'.join(parts), encoding='utf-8')
    os.system(f'ffmpeg -y -loglevel error -i "{out_svg}" -frames:v 1 "{out_png}"')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shared = bp.build_shared_dual_dataset()
    groups = [bp.trace_scale18_first3(shared), bp.trace_plain24_first3(shared), bp.trace_purescd24_first3(shared)]
    mapping = [('phi_x', 'att_sigma_x_approx'), ('phi_y', 'att_sigma_y_approx'), ('phi_z', 'att_sigma_z_approx')]
    summary = {
        'task': 'dualpath_three_method_att_sigma_preview_2026_04_09',
        'note': 'Approximate attitude-error sigma using sigma(phi_x/phi_y/phi_z) from state covariance.',
        'display_labels': bp.GROUP_LABELS,
        'output_dir': str(OUT_DIR),
        'plots': [],
    }
    for state_label, axis_label in mapping:
        series_list = []
        for g in groups:
            info = g['state_by_label'][state_label]
            idx = info['index']
            sigma = [math.sqrt(abs(float(v[idx]))) * info['scale'] for v in g['p_trace']]
            x = [i * bp.TRACE_DT_S for i in range(len(g['x_trace']))]
            series_list.append({'group_key': g['group_key'], 'label': g['label'], 'x': x, 'y': sigma, 'iter_bounds_s': [b * bp.TRACE_DT_S for b in g['iter_bounds']]})
        svg = OUT_DIR / f'{axis_label}.svg'
        png = OUT_DIR / f'{axis_label}.png'
        render_sigma_preview(svg, png, axis_label, series_list)
        summary['plots'].append({'axis': axis_label, 'svg': str(svg), 'png': str(png)})
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
