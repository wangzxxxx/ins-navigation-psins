from __future__ import annotations

import json
import math
import os
import sys
import types
from datetime import datetime
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

from common_markov import load_module
import compare_four_group_progression_19_20_custom_noise as custom_mod


ARW_DPSH = 0.0005
VRW_UGPSHZ = 0.5
BI_G_DPH = 0.0007
BI_A_UG = 5.0
TAU_G = 300.0
TAU_A = 300.0
SEED = 42
REPORT_DATE = '2026-04-07'
TRACE_DT_S = 0.2
MAX_SVG_POINTS = 1200

GROUP_SPECS = [
    ('g1_kf19', 'G1 普通模型 @19位置', 'legacy19pos', 'kf36_noisy', 36),
    ('g2_markov19', 'G2 Markov @19位置', 'legacy19pos', 'markov42_noisy', 42),
    ('g3_markov20', 'G3 Markov @20位置', 'symmetric20', 'markov42_noisy', 42),
    ('g4_round61_20', 'G4 Markov+LLM+SCD @20位置', 'symmetric20', 'round61', 42),
]


def build_cfg() -> dict:
    return {
        'arw_dpsh': ARW_DPSH,
        'vrw_ugpsHz': VRW_UGPSHZ,
        'bi_g_dph': BI_G_DPH,
        'bi_a_ug': BI_A_UG,
        'tau_g': TAU_G,
        'tau_a': TAU_A,
        'seed': SEED,
        'base_family': 'user_explicit_custom_noise',
    }


def noise_tag() -> str:
    return custom_mod.custom_suffix(build_cfg())


def state_meta(n_states: int, glv) -> list[dict]:
    meta = [
        {'label': 'phi_x', 'unit': 'deg', 'scale': 1.0 / glv.deg},
        {'label': 'phi_y', 'unit': 'deg', 'scale': 1.0 / glv.deg},
        {'label': 'phi_z', 'unit': 'deg', 'scale': 1.0 / glv.deg},
        {'label': 'dv_x', 'unit': 'm/s', 'scale': 1.0},
        {'label': 'dv_y', 'unit': 'm/s', 'scale': 1.0},
        {'label': 'dv_z', 'unit': 'm/s', 'scale': 1.0},
        {'label': 'eb_x', 'unit': 'dph', 'scale': 1.0 / glv.dph},
        {'label': 'eb_y', 'unit': 'dph', 'scale': 1.0 / glv.dph},
        {'label': 'eb_z', 'unit': 'dph', 'scale': 1.0 / glv.dph},
        {'label': 'db_x', 'unit': 'ug', 'scale': 1.0 / glv.ug},
        {'label': 'db_y', 'unit': 'ug', 'scale': 1.0 / glv.ug},
        {'label': 'db_z', 'unit': 'ug', 'scale': 1.0 / glv.ug},
        {'label': 'dKg_xx', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
        {'label': 'dKg_yx', 'unit': 'sec', 'scale': 1.0 / glv.sec},
        {'label': 'dKg_zx', 'unit': 'sec', 'scale': 1.0 / glv.sec},
        {'label': 'dKg_xy', 'unit': 'sec', 'scale': 1.0 / glv.sec},
        {'label': 'dKg_yy', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
        {'label': 'dKg_zy', 'unit': 'sec', 'scale': 1.0 / glv.sec},
        {'label': 'dKg_xz', 'unit': 'sec', 'scale': 1.0 / glv.sec},
        {'label': 'dKg_yz', 'unit': 'sec', 'scale': 1.0 / glv.sec},
        {'label': 'dKg_zz', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
        {'label': 'dKa_xx', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
        {'label': 'dKa_xy', 'unit': 'sec', 'scale': 1.0 / glv.sec},
        {'label': 'dKa_xz', 'unit': 'sec', 'scale': 1.0 / glv.sec},
        {'label': 'dKa_yy', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
        {'label': 'dKa_yz', 'unit': 'sec', 'scale': 1.0 / glv.sec},
        {'label': 'dKa_zz', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
        {'label': 'Ka2_x', 'unit': 'ug/g²', 'scale': 1.0 / glv.ugpg2},
        {'label': 'Ka2_y', 'unit': 'ug/g²', 'scale': 1.0 / glv.ugpg2},
        {'label': 'Ka2_z', 'unit': 'ug/g²', 'scale': 1.0 / glv.ugpg2},
        {'label': 'rx_x', 'unit': '—', 'scale': 1.0},
        {'label': 'rx_y', 'unit': '—', 'scale': 1.0},
        {'label': 'rx_z', 'unit': '—', 'scale': 1.0},
        {'label': 'ry_x', 'unit': '—', 'scale': 1.0},
        {'label': 'ry_y', 'unit': '—', 'scale': 1.0},
        {'label': 'ry_z', 'unit': '—', 'scale': 1.0},
    ]
    if n_states == 42:
        meta.extend([
            {'label': 'bm_g_x', 'unit': 'dph', 'scale': 1.0 / glv.dph},
            {'label': 'bm_g_y', 'unit': 'dph', 'scale': 1.0 / glv.dph},
            {'label': 'bm_g_z', 'unit': 'dph', 'scale': 1.0 / glv.dph},
            {'label': 'bm_a_x', 'unit': 'ug', 'scale': 1.0 / glv.ug},
            {'label': 'bm_a_y', 'unit': 'ug', 'scale': 1.0 / glv.ug},
            {'label': 'bm_a_z', 'unit': 'ug', 'scale': 1.0 / glv.ug},
        ])
    if len(meta) != n_states:
        raise ValueError(f'state meta length mismatch: {len(meta)} != {n_states}')
    return meta


def downsample_series(x, ys, max_points=MAX_SVG_POINTS):
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


def fmt(v: float) -> str:
    if abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 1e-3):
        return f'{v:.3e}'
    return f'{v:.4f}'


def nice_ticks(vmin: float, vmax: float, n: int = 5):
    if math.isclose(vmin, vmax, rel_tol=0.0, abs_tol=1e-15):
        delta = 1.0 if abs(vmin) < 1e-12 else abs(vmin) * 0.1
        vmin -= delta
        vmax += delta
    return [vmin + (vmax - vmin) * i / (n - 1) for i in range(n)]


def polyline_path(xs, ys):
    return ' '.join(f'{x:.2f},{y:.2f}' for x, y in zip(xs, ys))


def band_polygon(xs, lower, upper):
    pts = [(x, y) for x, y in zip(xs, upper)] + [(x, y) for x, y in zip(reversed(xs), reversed(lower))]
    return polyline_path([p[0] for p in pts], [p[1] for p in pts])


def render_state_svg(out_path: Path, method_label: str, state_label: str, unit: str, x_vals, est_vals, sigma_vals, iter_bounds_s):
    width = 1200
    height = 420
    left = 90
    right = 24
    top = 48
    bottom = 58
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_max = max(x_vals) if x_vals else 1.0
    x_min = min(x_vals) if x_vals else 0.0
    upper = [a + b for a, b in zip(est_vals, sigma_vals)]
    lower = [a - b for a, b in zip(est_vals, sigma_vals)]
    y_min = min(min(lower), 0.0)
    y_max = max(max(upper), 0.0)
    if math.isclose(y_min, y_max, rel_tol=0.0, abs_tol=1e-12):
        pad = 1.0 if abs(y_min) < 1e-12 else abs(y_min) * 0.15
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

    def mx(x):
        return left + (x - x_min) / (x_max - x_min if x_max != x_min else 1.0) * plot_w

    def my(y):
        return top + plot_h - (y - y_min) / (y_max - y_min if y_max != y_min else 1.0) * plot_h

    xs_px = [mx(v) for v in x_vals]
    est_px = [my(v) for v in est_vals]
    upper_px = [my(v) for v in upper]
    lower_px = [my(v) for v in lower]

    x_ticks = nice_ticks(x_min, x_max, 6)
    y_ticks = nice_ticks(y_min, y_max, 5)

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<style>text{font-family:Arial,Helvetica,sans-serif;} .small{font-size:12px;fill:#444;} .tick{font-size:11px;fill:#555;} .title{font-size:18px;font-weight:bold;} .subtitle{font-size:13px;fill:#333;} .axis{stroke:#333;stroke-width:1;} .grid{stroke:#ddd;stroke-width:1;} .zero{stroke:#999;stroke-width:1;stroke-dasharray:4 4;} .iter{stroke:#888;stroke-width:1;stroke-dasharray:6 4;} </style>')
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    parts.append(f'<text x="{left}" y="26" class="title">{escape(method_label)} · {escape(state_label)}</text>')
    parts.append(f'<text x="{left}" y="44" class="subtitle">estimate ± 1σ, unit = {escape(unit)}, x-axis = cumulative filter update time (s)</text>')

    # Grid + ticks
    for yt in y_ticks:
        y = my(yt)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" class="grid"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" class="tick" text-anchor="end">{escape(fmt(yt))}</text>')
    for xt in x_ticks:
        x = mx(xt)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" class="grid"/>')
        parts.append(f'<text x="{x:.2f}" y="{top + plot_h + 22}" class="tick" text-anchor="middle">{escape(fmt(xt))}</text>')

    # Zero line
    if y_min <= 0.0 <= y_max:
        y0 = my(0.0)
        parts.append(f'<line x1="{left}" y1="{y0:.2f}" x2="{left + plot_w}" y2="{y0:.2f}" class="zero"/>')

    # Iter boundaries
    for idx, bound_t in enumerate(iter_bounds_s[:-1], start=1):
        if x_min <= bound_t <= x_max:
            xb = mx(bound_t)
            parts.append(f'<line x1="{xb:.2f}" y1="{top}" x2="{xb:.2f}" y2="{top + plot_h}" class="iter"/>')
            parts.append(f'<text x="{xb + 4:.2f}" y="{top + 14}" class="small">iter {idx+1}</text>')

    # Axes
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" class="axis"/>')
    parts.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" class="axis"/>')

    # Band + lines
    if len(xs_px) >= 2:
        parts.append(f'<polygon points="{band_polygon(xs_px, lower_px, upper_px)}" fill="#f6c28b" fill-opacity="0.35" stroke="none"/>')
        parts.append(f'<polyline points="{polyline_path(xs_px, upper_px)}" fill="none" stroke="#d27a00" stroke-width="1.0" stroke-dasharray="4 4"/>')
        parts.append(f'<polyline points="{polyline_path(xs_px, lower_px)}" fill="none" stroke="#d27a00" stroke-width="1.0" stroke-dasharray="4 4"/>')
        parts.append(f'<polyline points="{polyline_path(xs_px, est_px)}" fill="none" stroke="#1769aa" stroke-width="1.6"/>')

    # Legend
    legend_x = left + plot_w - 180
    legend_y = top + 10
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="165" height="54" fill="white" fill-opacity="0.9" stroke="#ccc"/>')
    parts.append(f'<line x1="{legend_x + 10}" y1="{legend_y + 16}" x2="{legend_x + 38}" y2="{legend_y + 16}" stroke="#1769aa" stroke-width="2"/>')
    parts.append(f'<text x="{legend_x + 46}" y="{legend_y + 20}" class="small">estimate</text>')
    parts.append(f'<line x1="{legend_x + 10}" y1="{legend_y + 36}" x2="{legend_x + 38}" y2="{legend_y + 36}" stroke="#d27a00" stroke-width="2" stroke-dasharray="4 4"/>')
    parts.append(f'<text x="{legend_x + 46}" y="{legend_y + 40}" class="small">±1σ bound</text>')

    parts.append('</svg>')
    out_path.write_text('\n'.join(parts), encoding='utf-8')


def save_group_plots(group_dir: Path, method_label: str, x_trace, p_trace, iter_bounds, meta):
    plots = []
    x_raw = [i * TRACE_DT_S for i in range(len(x_trace))]
    iter_bounds_s = [b * TRACE_DT_S for b in iter_bounds]
    for idx, info in enumerate(meta):
        est = [float(v[idx]) * info['scale'] for v in x_trace]
        sigma = [math.sqrt(abs(float(v[idx]))) * info['scale'] for v in p_trace]
        xs, arrays = downsample_series(x_raw, [est, sigma])
        est_ds, sigma_ds = arrays
        file_name = f'{idx:02d}_{info["label"]}.svg'
        out_path = group_dir / file_name
        render_state_svg(out_path, method_label, info['label'], info['unit'], xs, est_ds, sigma_ds, iter_bounds_s)
        plots.append({
            'index': idx,
            'label': info['label'],
            'unit': info['unit'],
            'file': str(out_path),
            'downsampled_points': len(xs),
        })
    return plots


def write_readme(out_dir: Path, cfg: dict, results: dict):
    lines = []
    lines.append('# Four-method custom-noise state convergence plots')
    lines.append('')
    lines.append('## Noise config')
    lines.append('')
    for k in ['arw_dpsh', 'vrw_ugpsHz', 'bi_g_dph', 'bi_a_ug', 'tau_g', 'tau_a', 'seed']:
        lines.append(f'- {k}: `{cfg[k]}`')
    lines.append('')
    lines.append('## Output structure')
    lines.append('')
    for group_key, item in results.items():
        lines.append(f'- **{item["label"]}**: `{item["dir"]}` ({item["n_states"]} state plots)')
    lines.append('')
    lines.append('Each state file shows **estimate ± 1σ** with iteration boundaries marked by dashed vertical lines.')
    lines.append('')
    (out_dir / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def run_all() -> dict:
    cfg = build_cfg()
    tag = noise_tag()
    out_dir = TMP_DIR / f'psins_four_group_state_convergence_{REPORT_DATE}_{tag}'
    out_dir.mkdir(parents=True, exist_ok=True)

    source_mod, compare_mod, shared_mod, compute_r61_mod, probe_r55_mod, r53_mod, r61_mod = custom_mod.load_modules(tag)
    cases = {
        'legacy19pos': compare_mod.build_legacy19pos_case(source_mod),
        'symmetric20': compare_mod.build_symmetric20_case(source_mod),
    }

    results = {}
    for group_key, label, case_key, method_key, n_states in GROUP_SPECS:
        case = cases[case_key]
        dataset = custom_mod.build_dataset_custom(source_mod, case['paras'], case['att0_deg'], cfg)
        group_dir = out_dir / group_key
        group_dir.mkdir(parents=True, exist_ok=True)

        if method_key in ('kf36_noisy', 'markov42_noisy'):
            kwargs = {}
            if method_key == 'markov42_noisy':
                kwargs.update({
                    'n_states': 42,
                    'bi_g': dataset['bi_g'],
                    'tau_g': dataset['tau_g'],
                    'bi_a': dataset['bi_a'],
                    'tau_a': dataset['tau_a'],
                })
            else:
                kwargs.update({'n_states': 36})
            result = source_mod.run_calibration(
                dataset['imu_noisy'],
                dataset['pos0'],
                dataset['ts'],
                label=f'{group_key}-{tag}',
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
                label=f'{group_key}-{tag}',
                scd_cfg=merged_candidate['scd'],
            ))
            p_trace = result[2].tolist()
            x_trace = result[3].tolist()
            iter_bounds = result[4]['iter_bounds']
        else:
            raise KeyError(method_key)

        meta = state_meta(n_states, source_mod.glv)
        plots = save_group_plots(group_dir, label, x_trace, p_trace, iter_bounds, meta)
        manifest = {
            'group_key': group_key,
            'label': label,
            'case_key': case_key,
            'method_key': method_key,
            'n_states': n_states,
            'iter_bounds': iter_bounds,
            'trace_dt_s': TRACE_DT_S,
            'noise_tag': tag,
            'noise_config': cfg,
            'plots': plots,
        }
        (group_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        results[group_key] = {
            'label': label,
            'dir': str(group_dir),
            'n_states': n_states,
            'manifest': str(group_dir / 'manifest.json'),
        }

    summary = {
        'report_date': REPORT_DATE,
        'noise_tag': tag,
        'noise_config': cfg,
        'output_dir': str(out_dir),
        'groups': results,
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    write_readme(out_dir, cfg, results)
    return summary


def main():
    summary = run_all()
    print('__RESULT_JSON__=' + json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
