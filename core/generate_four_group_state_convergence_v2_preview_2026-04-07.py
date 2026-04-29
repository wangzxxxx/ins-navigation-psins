from __future__ import annotations

import json
import math
import os
import sys
import types
from pathlib import Path
from xml.sax.saxutils import escape

# Stub unavailable plotting deps before loading PSINS modules.
if 'matplotlib' not in sys.modules:
    matplotlib_stub = types.ModuleType('matplotlib')
    pyplot_stub = types.ModuleType('matplotlib.pyplot')
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules['matplotlib'] = matplotlib_stub
    sys.modules['matplotlib.pyplot'] = pyplot_stub
if 'seaborn' not in sys.modules:
    sys.modules['seaborn'] = types.ModuleType('seaborn')

ROOT = Path('/root/.openclaw/workspace')
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'
METHOD_DIR = ROOT / 'psins_method_bench' / 'methods' / 'markov'
TMP_DIR = ROOT / 'tmp'

for p in [ROOT, SCRIPTS_DIR, METHOD_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import compare_four_group_progression_19_20_custom_noise as custom_mod
import generate_four_group_custom_noise_state_convergence_svgs as base_mod

TRACE_DT_S = base_mod.TRACE_DT_S
REPORT_DATE = '2026-04-07'
DEFAULT_SELECTED_STATES = ['phi_x', 'dKg_xx', 'Ka2_x']
MAX_POINTS = 1200
SVG_ONLY = os.environ.get('SVG_ONLY', '0') == '1'
ALL_STATES = os.environ.get('ALL_STATES', '0') == '1'
GROUP_KEYS = [s.strip() for s in os.environ.get('GROUP_KEYS', '').split(',') if s.strip()]
OUTPUT_TAG = os.environ.get('OUTPUT_TAG', '')
STATE_KEYS = [s.strip() for s in os.environ.get('STATE_KEYS', '').split(',') if s.strip()]
LINE_OPACITY = 0.56
LINE_WIDTH = 1.5
LEGEND_LINE_WIDTH = 2.8
COLORS = {
    'g1_kf19': '#1769aa',
    'g2_markov19': '#d9822b',
    'g3_markov20': '#2b8a3e',
    'g4_round61_20': '#c92a2a',
}
LINE_DASHES = {
    'g1_kf19': None,
    'g2_markov19': '10 6',
    'g3_markov20': '3 5',
    'g4_round61_20': '14 5 3 5',
}


def line_style(group_key: str, legend: bool = False) -> str:
    color = COLORS.get(group_key, '#000')
    dash = LINE_DASHES.get(group_key)
    width = LEGEND_LINE_WIDTH if legend else LINE_WIDTH
    opacity = 1.0 if legend else LINE_OPACITY
    attrs = [
        f'stroke="{color}"',
        f'stroke-width="{width}"',
        f'stroke-opacity="{opacity}"',
        'stroke-linecap="round"',
        'stroke-linejoin="round"',
    ]
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    return ' '.join(attrs)


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


def robust_range(values, low_q=0.01, high_q=0.99, min_pad_frac=0.08):
    if not values:
        return -1.0, 1.0
    lo = percentile(values, low_q)
    hi = percentile(values, high_q)
    full_lo = min(values)
    full_hi = max(values)
    if full_lo < lo:
        lo = min(lo, full_lo)
    if full_hi > hi:
        hi = max(hi, full_hi)
    if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-15):
        delta = 1.0 if abs(lo) < 1e-12 else abs(lo) * 0.1
        lo -= delta
        hi += delta
    pad = max((hi - lo) * min_pad_frac, 1e-12)
    return lo - pad, hi + pad


def log_ticks(vmin, vmax, n=5):
    if vmin <= 0:
        vmin = min(v for v in [vmax * 1e-6, 1e-12] if v > 0)
    if vmax <= 0:
        vmax = 1.0
    lo = math.log10(vmin)
    hi = math.log10(vmax)
    if math.isclose(lo, hi, abs_tol=1e-12):
        lo -= 1.0
        hi += 1.0
    return [10 ** (lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def load_group_traces():
    cfg = base_mod.build_cfg()
    tag = base_mod.noise_tag()
    source_mod, compare_mod, shared_mod, compute_r61_mod, probe_r55_mod, r53_mod, r61_mod = custom_mod.load_modules(tag)
    cases = {
        'legacy19pos': compare_mod.build_legacy19pos_case(source_mod),
        'symmetric20': compare_mod.build_symmetric20_case(source_mod),
    }
    groups = []
    for group_key, label, case_key, method_key, n_states in base_mod.GROUP_SPECS:
        if GROUP_KEYS and group_key not in GROUP_KEYS:
            continue
        case = cases[case_key]
        dataset = custom_mod.build_dataset_custom(source_mod, case['paras'], case['att0_deg'], cfg)
        if method_key in ('kf36_noisy', 'markov42_noisy'):
            kwargs = {'n_states': 36} if method_key == 'kf36_noisy' else {
                'n_states': 42,
                'bi_g': dataset['bi_g'],
                'tau_g': dataset['tau_g'],
                'bi_a': dataset['bi_a'],
                'tau_a': dataset['tau_a'],
            }
            result = source_mod.run_calibration(
                dataset['imu_noisy'],
                dataset['pos0'],
                dataset['ts'],
                label=f'v2-{group_key}-{tag}',
                **kwargs,
            )
            p_trace = result[2].tolist()
            x_trace = result[3].tolist()
            iter_bounds = result[4]
        elif method_key == 'round61':
            candidate = r61_mod._pick_candidate()
            merged_candidate = r61_mod._merge_round61_candidate(candidate)
            patched_method = probe_r55_mod._build_patched_method(r53_mod, merged_candidate)
            result = list(r61_mod._run_internalized_hybrid_scd(
                patched_method,
                source_mod,
                dataset['imu_noisy'],
                dataset['pos0'],
                dataset['ts'],
                bi_g=dataset['bi_g'],
                bi_a=dataset['bi_a'],
                tau_g=dataset['tau_g'],
                tau_a=dataset['tau_a'],
                label=f'v2-{group_key}-{tag}',
                scd_cfg=merged_candidate['scd'],
            ))
            p_trace = result[2].tolist()
            x_trace = result[3].tolist()
            iter_bounds = result[4]['iter_bounds']
        else:
            raise KeyError(method_key)
        meta = base_mod.state_meta(n_states, source_mod.glv)
        state_by_label = {m['label']: {**m, 'index': i} for i, m in enumerate(meta)}
        groups.append({
            'group_key': group_key,
            'label': label,
            'n_states': n_states,
            'x_trace': x_trace,
            'p_trace': p_trace,
            'iter_bounds': iter_bounds,
            'state_by_label': state_by_label,
        })
    return groups, cfg


def extract_series(group, state_label):
    info = group['state_by_label'].get(state_label)
    if info is None:
        return None
    idx = info['index']
    x_vals = [i * TRACE_DT_S for i in range(len(group['x_trace']))]
    est = [float(v[idx]) * info['scale'] for v in group['x_trace']]
    sigma = [math.sqrt(abs(float(v[idx]))) * info['scale'] for v in group['p_trace']]
    return {
        'group_key': group['group_key'],
        'label': group['label'],
        'state_label': state_label,
        'unit': info['unit'],
        'x': x_vals,
        'est': est,
        'sigma': sigma,
        'iter_bounds_s': [b * TRACE_DT_S for b in group['iter_bounds']],
    }


def map_linear(v, vmin, vmax, y0, y1):
    if math.isclose(vmin, vmax, abs_tol=1e-15):
        return (y0 + y1) / 2
    return y1 - (v - vmin) / (vmax - vmin) * (y1 - y0)


def map_log(v, vmin, vmax, y0, y1):
    v = max(v, vmin)
    lo = math.log10(vmin)
    hi = math.log10(vmax)
    if math.isclose(lo, hi, abs_tol=1e-15):
        return (y0 + y1) / 2
    return y1 - (math.log10(v) - lo) / (hi - lo) * (y1 - y0)


def draw_panel(parts, rect, title, x_ticks, y_ticks, x_mapper, y_mapper, x_label, y_label, zero_y=None, iter_bounds=None, log_y=False):
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
    if zero_y is not None:
        parts.append(f'<line x1="{plot_left}" y1="{zero_y:.2f}" x2="{plot_right}" y2="{zero_y:.2f}" class="zero"/>')
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
    return plot_left, plot_top, plot_right, plot_bottom


def render_state_preview(out_svg: Path, out_png: Path | None, state_label: str, unit: str, series_list):
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
        (origin_x + 2 * (panel_w + gap), origin_y, origin_x + 3 * panel_w + 2 * gap, origin_y + panel_h),
    ]

    all_x = [x for s in series_list for x in s['x']]
    x_min, x_max = min(all_x), max(all_x)
    x_ticks = nice_ticks(x_min, x_max, 6)
    iter_bounds = series_list[0]['iter_bounds_s'] if series_list else []
    tail_start = iter_bounds[-2] if len(iter_bounds) >= 2 else x_max * 0.75
    tail_end = x_max
    tail_ticks = nice_ticks(tail_start, tail_end, 5)

    est_all = [v for s in series_list for v in s['est']]
    full_ymin, full_ymax = robust_range(est_all, low_q=0.0, high_q=1.0, min_pad_frac=0.08)
    tail_vals = [v for s in series_list for x, v in zip(s['x'], s['est']) if x >= tail_start]
    tail_ymin, tail_ymax = robust_range(tail_vals, low_q=0.0, high_q=1.0, min_pad_frac=0.12)
    sigma_all = [max(v, 1e-12) for s in series_list for v in s['sigma']]
    sigma_min = min(sigma_all)
    sigma_max = max(sigma_all)
    sigma_ticks = log_ticks(sigma_min, sigma_max, 5)

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<style>text{font-family:Arial,Helvetica,sans-serif;} .title{font-size:22px;font-weight:bold;} .subtitle{font-size:13px;fill:#444;} .panel-title{font-size:15px;font-weight:bold;fill:#223;} .tick{font-size:11px;fill:#555;} .small{font-size:12px;fill:#444;} .axis{stroke:#333;stroke-width:1;} .grid{stroke:#e5e7eb;stroke-width:1;} .zero{stroke:#9aa0a6;stroke-width:1;stroke-dasharray:4 4;} .iter{stroke:#94a3b8;stroke-width:1;stroke-dasharray:6 4;} .legend{font-size:12px;fill:#222;}</style>')
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    parts.append(f'<text x="18" y="26" class="title">{escape(state_label)}</text>')

    # Panel geometry helpers
    def make_mappers(rect, xmin, xmax, ymin, ymax, log_y=False):
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
            return map_log(y, ymin, ymax, plot_top, plot_bottom) if log_y else map_linear(y, ymin, ymax, plot_top, plot_bottom)
        return mx, my

    mx1, my1 = make_mappers(panels[0], x_min, x_max, full_ymin, full_ymax, log_y=False)
    zero1 = my1(0.0) if full_ymin <= 0 <= full_ymax else None
    draw_panel(parts, panels[0], 'Estimate · full range', x_ticks, nice_ticks(full_ymin, full_ymax, 5), mx1, my1, 'time (s)', unit, zero_y=zero1, iter_bounds=iter_bounds)

    mx2, my2 = make_mappers(panels[1], tail_start, tail_end, tail_ymin, tail_ymax, log_y=False)
    zero2 = my2(0.0) if tail_ymin <= 0 <= tail_ymax else None
    tail_iter = [b for b in iter_bounds if tail_start <= b <= tail_end]
    draw_panel(parts, panels[1], 'Estimate · final-iteration zoom', tail_ticks, nice_ticks(tail_ymin, tail_ymax, 5), mx2, my2, 'time (s)', unit, zero_y=zero2, iter_bounds=tail_iter)

    mx3, my3 = make_mappers(panels[2], x_min, x_max, sigma_min, sigma_max, log_y=True)
    draw_panel(parts, panels[2], 'σ half-width · separate panel', x_ticks, sigma_ticks, mx3, my3, 'time (s)', f'σ ({unit})', zero_y=None, iter_bounds=iter_bounds, log_y=True)

    for series in series_list:
        style = line_style(series['group_key'])
        x1s, [y1s] = downsample(series['x'], [series['est']])
        pts1 = polyline([mx1(v) for v in x1s], [my1(v) for v in y1s])
        parts.append(f'<polyline points="{pts1}" fill="none" {style}/>')

        tail_x = [x for x in series['x'] if x >= tail_start]
        tail_y = [y for x, y in zip(series['x'], series['est']) if x >= tail_start]
        tail_x_ds, [tail_y_ds] = downsample(tail_x, [tail_y], max_points=800)
        pts2 = polyline([mx2(v) for v in tail_x_ds], [my2(v) for v in tail_y_ds])
        parts.append(f'<polyline points="{pts2}" fill="none" {style}/>')

        x3s, [y3s] = downsample(series['x'], [series['sigma']])
        pts3 = polyline([mx3(v) for v in x3s], [my3(max(v, sigma_min)) for v in y3s])
        parts.append(f'<polyline points="{pts3}" fill="none" {style}/>')

    legend_x = 22
    legend_y = 382
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="700" height="66" fill="white" stroke="#d0d7de"/>')
    for i, series in enumerate(series_list):
        row = i // 2
        col = i % 2
        x = legend_x + 18 + col * 330
        y = legend_y + 22 + row * 24
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" {line_style(series["group_key"], legend=True)}/>')
        parts.append(f'<text x="{x + 38}" y="{y + 4}" class="legend">{escape(series["label"])}</text>')

    parts.append('</svg>')
    out_svg.write_text('\n'.join(parts), encoding='utf-8')
    if out_png is not None:
        os.system(f'ffmpeg -y -loglevel error -i "{out_svg}" -frames:v 1 "{out_png}"')


def discover_state_labels(groups):
    seen = set()
    labels = []
    for group in groups:
        for label in group['state_by_label'].keys():
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def main():
    groups, cfg = load_group_traces()
    state_labels = discover_state_labels(groups) if ALL_STATES else (STATE_KEYS if STATE_KEYS else DEFAULT_SELECTED_STATES)
    mode_name = 'full' if ALL_STATES else 'preview'
    suffix = f'_{OUTPUT_TAG}' if OUTPUT_TAG else ''
    out_dir = TMP_DIR / f'psins_four_group_state_convergence_v2_{mode_name}_{REPORT_DATE}{suffix}'
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        'report_date': REPORT_DATE,
        'selected_states': state_labels,
        'noise_config': cfg,
        'output_dir': str(out_dir),
        'plots': [],
        'svg_only': SVG_ONLY,
        'all_states': ALL_STATES,
        'group_keys': [g['group_key'] for g in groups],
    }
    for state_label in state_labels:
        series_list = []
        unit = ''
        for group in groups:
            s = extract_series(group, state_label)
            if s is not None:
                series_list.append(s)
                unit = s['unit']
        svg_path = out_dir / f'{state_label}_v2.svg'
        png_path = None if SVG_ONLY else out_dir / f'{state_label}_v2.png'
        render_state_preview(svg_path, png_path, state_label, unit, series_list)
        item = {
            'state_label': state_label,
            'unit': unit,
            'svg': str(svg_path),
        }
        if png_path is not None:
            item['png'] = str(png_path)
        summary['plots'].append(item)
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
