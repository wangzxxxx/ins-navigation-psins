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
CUSTOM_DATASET_SCRIPT = WORKSPACE / 'psins_method_bench' / 'scripts' / 'generate_dualpath_three_method_att_err_preview_0p4_2026-04-13.py'
OUT_DIR = WORKSPACE / 'tmp' / 'psins_dualpath_three_method_state_sigma_combo_0p4_2026-04-13'
OUT_JSON = OUT_DIR / 'summary.json'
OUTER_ITERS = 5
SEED = 0
NOISE_TAG = '0.4'
NOTE = 'Three-panel combo state convergence for 0.4x custom noise: full estimate, tail zoom, and covariance sigma.'


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_module('dualpath_state_combo_base_0p4_20260413', BASE_SCRIPT)
custom = load_module('dualpath_state_combo_custom_0p4_20260413', CUSTOM_DATASET_SCRIPT)


def render_state_combo(out_svg: Path, out_png: Path, state_label: str, unit: str, series_list):
    width = 2200
    height = 470
    gap = 18
    panel_w = 680
    panel_h = 320
    origin_x = 30
    origin_y = 40
    panels = [
        (origin_x, origin_y, origin_x + panel_w, origin_y + panel_h),
        (origin_x + panel_w + gap, origin_y, origin_x + 2 * panel_w + gap, origin_y + panel_h),
        (origin_x + 2 * (panel_w + gap), origin_y, origin_x + 3 * panel_w + 2 * gap, origin_y + panel_h),
    ]

    all_x = [x for s in series_list for x in s['x']]
    x_min, x_max = min(all_x), max(all_x)
    x_ticks = base.nice_ticks(x_min, x_max, 6)
    iter_bounds = series_list[0]['iter_bounds_s'] if series_list else []
    last_round_idx = len(iter_bounds) if iter_bounds else OUTER_ITERS
    last_round_start = iter_bounds[-2] if len(iter_bounds) >= 2 else x_min
    tail_start = max(last_round_start, x_max - 100.0)
    tail_end = x_max
    tail_ticks = base.nice_ticks(tail_start, tail_end, 5)

    est_all = [v for s in series_list for v in s['est']]
    full_ymin, full_ymax = base.robust_range(est_all, low_q=0.0, high_q=1.0, min_pad_frac=0.08)
    tail_vals = [v for s in series_list for x, v in zip(s['x'], s['est']) if x >= tail_start]
    tail_ymin, tail_ymax = base.robust_range(tail_vals, low_q=0.0, high_q=1.0, min_pad_frac=0.12)

    sigma_all = [max(v, 1e-12) for s in series_list for v in s['sigma']]
    sigma_ymin, sigma_ymax = base.robust_range(sigma_all, low_q=0.0, high_q=1.0, min_pad_frac=0.08)
    sigma_ticks = base.log_ticks(max(sigma_ymin, 1e-12), sigma_ymax, 5)

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<style>text{font-family:Arial,Helvetica,sans-serif;} .title{font-size:22px;font-weight:bold;} .panel-title{font-size:15px;font-weight:bold;fill:#223;} .tick{font-size:11px;fill:#555;} .small{font-size:12px;fill:#444;} .axis{stroke:#333;stroke-width:1;} .grid{stroke:#e5e7eb;stroke-width:1;} .zero{stroke:#9aa0a6;stroke-width:1;stroke-dasharray:4 4;} .iter{stroke:#94a3b8;stroke-width:1;stroke-dasharray:6 4;} .legend{font-size:12px;fill:#222;}</style>')
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    parts.append(f'<text x="18" y="26" class="title">{escape(state_label)} · noise ×{NOISE_TAG} · seed={SEED}</text>')

    def make_linear_mappers(rect, xmin, xmax, ymin, ymax):
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
            return base.map_linear(y, ymin, ymax, plot_top, plot_bottom)

        return mx, my

    def make_log_mappers(rect, xmin, xmax, ymin, ymax):
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
            return base.map_log(y, max(ymin, 1e-12), ymax, plot_top, plot_bottom)

        return mx, my

    mx1, my1 = make_linear_mappers(panels[0], x_min, x_max, full_ymin, full_ymax)
    zero1 = my1(0.0) if full_ymin <= 0 <= full_ymax else None
    base.draw_panel(parts, panels[0], 'State convergence · full range', x_ticks, base.nice_ticks(full_ymin, full_ymax, 5), mx1, my1, 'time (s)', unit, zero_y=zero1, iter_bounds=iter_bounds)

    mx2, my2 = make_linear_mappers(panels[1], tail_start, tail_end, tail_ymin, tail_ymax)
    zero2 = my2(0.0) if tail_ymin <= 0 <= tail_ymax else None
    tail_iter = [b for b in iter_bounds if tail_start <= b <= tail_end]
    base.draw_panel(parts, panels[1], f'Tail-detail zoom · round-{last_round_idx} last 100 s', tail_ticks, base.nice_ticks(tail_ymin, tail_ymax, 5), mx2, my2, 'time (s)', unit, zero_y=zero2, iter_bounds=tail_iter)

    mx3, my3 = make_log_mappers(panels[2], x_min, x_max, sigma_ymin, sigma_ymax)
    base.draw_panel(parts, panels[2], 'Complete covariance convergence', x_ticks, sigma_ticks, mx3, my3, 'time (s)', 'sigma', zero_y=None, iter_bounds=iter_bounds)

    for series in series_list:
        style = base.line_style(series['group_key'])
        xs1, [ys1] = base.downsample(series['x'], [series['est']])
        parts.append(f'<polyline points="{base.polyline([mx1(v) for v in xs1], [my1(v) for v in ys1])}" fill="none" {style}/>')
        tail_x = [x for x in series['x'] if x >= tail_start]
        tail_y = [y for x, y in zip(series['x'], series['est']) if x >= tail_start]
        xs2, [ys2] = base.downsample(tail_x, [tail_y], max_points=800)
        parts.append(f'<polyline points="{base.polyline([mx2(v) for v in xs2], [my2(v) for v in ys2])}" fill="none" {style}/>')

    for series in series_list:
        style = base.line_style(series['group_key'])
        xs3, [ys3] = base.downsample(series['x'], [series['sigma']])
        parts.append(f'<polyline points="{base.polyline([mx3(v) for v in xs3], [my3(max(v, max(sigma_ymin, 1e-12))) for v in ys3])}" fill="none" {style}/>')

    legend_width = min(1200, max(360, 40 + 310 * len(series_list)))
    legend_x = 22
    legend_y = 382
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="42" fill="white" stroke="#d0d7de"/>')
    for i, series in enumerate(series_list):
        x = legend_x + 18 + i * 310
        y = legend_y + 22
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" {base.line_style(series["group_key"], legend=True)}/>')
        parts.append(f'<text x="{x + 38}" y="{y + 4}" class="legend">{escape(series["label"])}</text>')

    parts.append('</svg>')
    out_svg.write_text('\n'.join(parts), encoding='utf-8')
    os.system(f'ffmpeg -y -loglevel error -i "{out_svg}" -frames:v 1 "{out_png}"')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base.OUT_DIR = OUT_DIR
    base.OUT_JSON = OUT_JSON
    base.OUTER_ITERS = OUTER_ITERS
    base.build_shared_dual_dataset = custom.build_shared_dual_dataset_custom_noise_0p4

    shared = custom.build_shared_dual_dataset_custom_noise_0p4()
    groups = [
        base.trace_scale18_first3(shared),
        base.trace_plain24_first3(shared),
        base.trace_purescd24_first3(shared),
    ]

    state_labels = base.discover_state_labels(groups)
    summary = {
        'task': 'dualpath_three_method_state_sigma_combo_0p4_2026_04_13',
        'note': NOTE,
        'noise': '0.4x custom noise',
        'seed': SEED,
        'outer_iters': OUTER_ITERS,
        'display_labels': base.GROUP_LABELS,
        'output_dir': str(OUT_DIR),
        'plots': [],
    }

    for state_label in state_labels:
        series_list = []
        unit = ''
        for group in groups:
            s = base.extract_series(group, state_label)
            if s is not None:
                series_list.append(s)
                unit = s['unit']
        svg_path = OUT_DIR / f'{state_label}_combo.svg'
        png_path = OUT_DIR / f'{state_label}_combo.png'
        render_state_combo(svg_path, png_path, state_label, unit, series_list)
        summary['plots'].append({
            'state_label': state_label,
            'unit': unit,
            'svg': str(svg_path),
            'png': str(png_path),
            'groups_present': [s['group_key'] for s in series_list],
        })

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
