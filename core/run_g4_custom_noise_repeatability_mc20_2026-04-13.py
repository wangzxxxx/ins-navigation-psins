#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/root/.openclaw/workspace')
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'
METHOD_DIR = ROOT / 'psins_method_bench' / 'methods' / 'markov'
OUT_DIR = ROOT / 'tmp' / 'psins_repeatability' / 'g4_custom_noise_mc20_2026-04-13'

for p in [ROOT, SCRIPTS_DIR, METHOD_DIR, ROOT / 'tmp_psins_py']:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import compare_four_group_progression_19_20_custom_noise as base

ROUND61_DISPLAY = 'G4 Markov+LLM+SCD @20位置'
ROUND61_PLOT_LABEL = 'G4 Markov+LLM+SCD @20pos'


def sample_std(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0

FAMILY_SPECS = [
    {
        'family_key': 'eb',
        'title': '陀螺零偏 eb',
        'unit': 'dph',
        'params': ['eb_x', 'eb_y', 'eb_z'],
    },
    {
        'family_key': 'db',
        'title': '加速度计零偏 db',
        'unit': 'ug',
        'params': ['db_x', 'db_y', 'db_z'],
    },
    {
        'family_key': 'dKg_diag',
        'title': '陀螺比例系数主对角 dKg',
        'unit': 'ppm',
        'params': ['dKg_xx', 'dKg_yy', 'dKg_zz'],
    },
    {
        'family_key': 'dKg_offdiag',
        'title': '陀螺安装误差 / 非对角 dKg',
        'unit': 'sec',
        'params': ['dKg_yx', 'dKg_zx', 'dKg_xy', 'dKg_zy', 'dKg_xz', 'dKg_yz'],
    },
    {
        'family_key': 'dKa_diag',
        'title': '加速度计比例系数主对角 dKa',
        'unit': 'ppm',
        'params': ['dKa_xx', 'dKa_yy', 'dKa_zz'],
    },
    {
        'family_key': 'dKa_offdiag',
        'title': '加速度计安装误差 / 非对角 dKa',
        'unit': 'sec',
        'params': ['dKa_xy', 'dKa_xz', 'dKa_yz'],
    },
    {
        'family_key': 'Ka2',
        'title': '加速度计二阶项 Ka2',
        'unit': 'ug/g²',
        'params': ['Ka2_x', 'Ka2_y', 'Ka2_z'],
    },
    {
        'family_key': 'rx',
        'title': '杆臂误差系数 rx',
        'unit': 'sec',
        'params': ['rx_x', 'rx_y', 'rx_z'],
    },
    {
        'family_key': 'ry',
        'title': '杆臂误差系数 ry',
        'unit': 'sec',
        'params': ['ry_x', 'ry_y', 'ry_z'],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--arw-dpsh', type=float, default=0.0005)
    parser.add_argument('--vrw-ugpsHz', type=float, default=0.5)
    parser.add_argument('--bi-g-dph', type=float, default=0.0007)
    parser.add_argument('--bi-a-ug', type=float, default=5.0)
    parser.add_argument('--tau-g', type=float, default=300.0)
    parser.add_argument('--tau-a', type=float, default=300.0)
    parser.add_argument('--seed-start', type=int, default=42)
    parser.add_argument('--n-runs', type=int, default=20)
    parser.add_argument('--report-date', default='2026-04-13')
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def build_cfg(args: argparse.Namespace, seed: int) -> dict:
    return {
        'arw_dpsh': float(args.arw_dpsh),
        'vrw_ugpsHz': float(args.vrw_ugpsHz),
        'bi_g_dph': float(args.bi_g_dph),
        'bi_a_ug': float(args.bi_a_ug),
        'tau_g': float(args.tau_g),
        'tau_a': float(args.tau_a),
        'seed': int(seed),
        'base_family': 'user_explicit_custom_noise',
    }


def ensure_dirs() -> dict[str, Path]:
    raw_dir = OUT_DIR / 'raw_runs'
    plots_dir = OUT_DIR / 'plots'
    raw_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return {
        'root': OUT_DIR,
        'raw': raw_dir,
        'plots': plots_dir,
    }


def unit_scale(source_mod, param_name: str) -> tuple[float, str]:
    glv = source_mod.glv
    if param_name.startswith('eb_'):
        return (1.0 / float(glv.dph), 'dph')
    if param_name.startswith('db_'):
        return (1.0 / float(glv.ug), 'ug')
    if param_name.startswith('Ka2_'):
        return (1.0 / float(glv.ugpg2), 'ug/g²')
    if param_name.startswith('rx_') or param_name.startswith('ry_'):
        return (1.0 / float(glv.sec), 'sec')
    if param_name.startswith('dKg_'):
        if param_name[-2:] in {'xx', 'yy', 'zz'}:
            return (1.0 / float(glv.ppm), 'ppm')
        return (1.0 / float(glv.sec), 'sec')
    if param_name.startswith('dKa_'):
        if param_name[-2:] in {'xx', 'yy', 'zz'}:
            return (1.0 / float(glv.ppm), 'ppm')
        return (1.0 / float(glv.sec), 'sec')
    return (1.0, 'raw')


def run_or_reuse_g4_for_seed(cfg: dict, force_rerun: bool = False) -> tuple[dict, str, str]:
    suffix = base.custom_suffix(cfg)
    source_mod, compare_mod, shared_mod, compute_r61_mod, probe_r55_mod, r53_mod, r61_mod = base.load_modules(suffix)
    case20 = compare_mod.build_symmetric20_case(source_mod)
    g4_path = base.round61_sym20_output_path(compare_mod.CASE_TAGS['symmetric20'], suffix)

    if (not force_rerun) and g4_path.exists():
        payload = shared_mod._load_json(g4_path)
        extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
        if (
            base.noise_matches_explicit(payload, cfg)
            and extra.get('comparison_mode') == base.COMPARISON_MODE
            and extra.get('case_key') == 'symmetric20'
            and extra.get('group_key') == 'g4_round61_20'
        ):
            return payload, 'reused_verified', str(g4_path)

    dataset = base.build_dataset_custom(source_mod, case20['paras'], case20['att0_deg'], cfg)
    params = compute_r61_mod._param_specs(source_mod)

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
        label=f'R61-SYM20-{suffix.upper()}',
        scd_cfg=merged_candidate['scd'],
    ))
    extra = result[4] if len(result) >= 5 and isinstance(result[4], dict) else {}
    extra = dict(extra)
    extra.update({
        'comparison_mode': base.COMPARISON_MODE,
        'group_key': 'g4_round61_20',
        'case_key': 'symmetric20',
        'case_tag': compare_mod.CASE_TAGS['symmetric20'],
        'case_display_name': compare_mod.CASE_LABELS['symmetric20'],
        'att0_deg': case20['att0_deg'],
        'n_motion_rows': case20['n_motion_rows'],
        'claimed_position_count': case20['claimed_position_count'],
        'total_time_s': case20['total_time_s'],
        'timing_note': case20['timing_note'],
        'source_builder': case20['source_builder'],
        'source_reference': case20['source_reference'],
        'noise_tag': suffix,
        'noise_config': cfg,
        'round61_selected_candidate': candidate['name'],
        'round61_candidate_description': candidate['description'],
        'round61_candidate_rationale': candidate['rationale'],
        'transfer_note': 'Round61 hybrid method transferred onto corrected symmetric20 path without a new 20pos-specific search.',
    })
    payload = shared_mod.compute_payload(
        source_mod,
        result[0],
        params,
        variant=f'42state_gm1_round61_h_scd_state20_microtight_commit_{compare_mod.CASE_TAGS["symmetric20"]}_{suffix}',
        method_file=f'{base.R61_METHOD_FILE} on corrected symmetric20 path',
        extra=extra,
    )
    g4_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', str(g4_path)


def compute_family_record(payload: dict, source_mod, family_spec: dict) -> dict:
    params = family_spec['params']
    component_records = {}
    est_vals = []
    true_vals = []
    pct_vals = []
    for name in params:
        scale, unit = unit_scale(source_mod, name)
        item = payload['param_errors'][name]
        est_v = float(item['est']) * scale
        true_v = float(item['true']) * scale
        pct_v = float(item['pct_error'])
        component_records[name] = {
            'est_display': est_v,
            'true_display': true_v,
            'pct_error': pct_v,
            'unit': unit,
        }
        est_vals.append(est_v)
        true_vals.append(true_v)
        pct_vals.append(pct_v)

    norm_est = float(np.linalg.norm(np.asarray(est_vals, dtype=float)))
    norm_true = float(np.linalg.norm(np.asarray(true_vals, dtype=float)))
    norm_abs_error = abs(norm_est - norm_true)
    norm_pct_error = norm_abs_error / abs(norm_true) * 100.0 if abs(norm_true) > 1e-15 else 0.0
    return {
        'family_key': family_spec['family_key'],
        'title': family_spec['title'],
        'unit': family_spec['unit'],
        'components': component_records,
        'norm_est_display': norm_est,
        'norm_true_display': norm_true,
        'norm_abs_error_display': norm_abs_error,
        'norm_pct_error': norm_pct_error,
        'component_pct_mean': float(np.mean(np.asarray(pct_vals, dtype=float))),
        'component_pct_max': float(np.max(np.asarray(pct_vals, dtype=float))),
    }


def build_seed_row(seed: int, payload: dict, source_mod, json_path: str, status: str) -> dict:
    row = {
        'seed': int(seed),
        'status': status,
        'result_json': json_path,
        'overall': payload['overall'],
        'families': {},
        'param_errors': payload['param_errors'],
    }
    for family_spec in FAMILY_SPECS:
        row['families'][family_spec['family_key']] = compute_family_record(payload, source_mod, family_spec)
    return row


def make_panel(ax, values: np.ndarray, color: str, title: str, xlabel: str):
    mu = float(values.mean())
    sigma = sample_std(values)

    bins = min(10, max(6, len(values) // 2 if len(values) >= 8 else len(values)))
    ax.hist(values, bins=bins, color=color, alpha=0.68, edgecolor='white', linewidth=1.0)
    if sigma > 0:
        ax.axvspan(mu - sigma, mu + sigma, color='#C44E52', alpha=0.10, zorder=0)
        ax.axvline(mu - sigma, color='#C44E52', linestyle='--', linewidth=1.0, alpha=0.85)
        ax.axvline(mu + sigma, color='#C44E52', linestyle='--', linewidth=1.0, alpha=0.85, label='±1σ')
    ax.axvline(mu, color='black', linestyle='-', linewidth=1.2, alpha=0.95, label='mean')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('count')
    ax.grid(True, linestyle='--', alpha=0.20)
    ax.text(
        0.04,
        0.96,
        f'μ={mu:.3f}\nσ={sigma:.3f}',
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=8.7,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.82),
    )


def plot_family(seed_rows: list[dict], family_spec: dict, out_dir: Path) -> dict:
    family_key = family_spec['family_key']
    panels = list(family_spec['params']) + ['norm']
    n_panels = len(panels)
    if n_panels <= 4:
        ncols = 2
    else:
        ncols = 4
    nrows = int(math.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.8 * nrows), dpi=180)
    axes = np.atleast_1d(axes).ravel()
    palette = ['#4C78A8', '#F58518', '#54A24B', '#72B7B2', '#B279A2', '#E45756', '#9D755D', '#BAB0AC']

    for idx, panel_name in enumerate(panels):
        ax = axes[idx]
        color = palette[idx % len(palette)]
        if panel_name == 'norm':
            values = np.asarray([row['families'][family_key]['norm_est_display'] for row in seed_rows], dtype=float)
            xlabel = family_spec['unit']
            title = 'norm distribution'
        else:
            values = np.asarray([
                row['families'][family_key]['components'][panel_name]['est_display']
                for row in seed_rows
            ], dtype=float)
            xlabel = family_spec['unit']
            title = f'{panel_name} distribution'
        make_panel(ax, values, color, title, xlabel)

    for ax in axes[n_panels:]:
        ax.axis('off')

    fig.suptitle(f"{family_spec['family_key']} distribution ({ROUND61_PLOT_LABEL}, n={len(seed_rows)})", fontsize=14)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        uniq = {}
        for h, l in zip(handles, labels):
            uniq[l] = h
        fig.legend(list(uniq.values()), list(uniq.keys()), loc='upper center', ncol=min(3, len(uniq)), frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    png_path = out_dir / f'{family_key}_distribution.png'
    svg_path = out_dir / f'{family_key}_distribution.svg'
    fig.savefig(png_path, format='png', bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    return {
        'png': str(png_path),
        'svg': str(svg_path),
    }


def family_stats(seed_rows: list[dict], family_spec: dict) -> dict:
    family_key = family_spec['family_key']
    stats = {'components': {}, 'norm': {}}
    for name in family_spec['params']:
        vals = np.asarray([
            row['families'][family_key]['components'][name]['est_display']
            for row in seed_rows
        ], dtype=float)
        stats['components'][name] = {
            'true_display': float(seed_rows[0]['families'][family_key]['components'][name]['true_display']),
            'mean_est_display': float(vals.mean()),
            'std_est_display': sample_std(vals),
            'median_est_display': float(np.median(vals)),
            'min_est_display': float(vals.min()),
            'max_est_display': float(vals.max()),
            'unit': family_spec['unit'],
        }
    norm_vals = np.asarray([row['families'][family_key]['norm_est_display'] for row in seed_rows], dtype=float)
    stats['norm'] = {
        'true_display': float(seed_rows[0]['families'][family_key]['norm_true_display']),
        'mean_est_display': float(norm_vals.mean()),
        'std_est_display': sample_std(norm_vals),
        'median_est_display': float(np.median(norm_vals)),
        'min_est_display': float(norm_vals.min()),
        'max_est_display': float(norm_vals.max()),
        'unit': family_spec['unit'],
    }
    return stats


def write_family_csv(seed_rows: list[dict], family_spec: dict, out_dir: Path) -> str:
    family_key = family_spec['family_key']
    csv_path = out_dir / f'{family_key}_per_seed.csv'
    fieldnames = ['seed']
    for name in family_spec['params']:
        fieldnames.extend([
            f'{name}_est_display',
            f'{name}_true_display',
            f'{name}_pct_error',
        ])
    fieldnames.extend([
        'norm_est_display',
        'norm_true_display',
        'norm_pct_error',
        'unit',
    ])
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in seed_rows:
            fr = row['families'][family_key]
            out = {
                'seed': row['seed'],
                'norm_est_display': fr['norm_est_display'],
                'norm_true_display': fr['norm_true_display'],
                'norm_pct_error': fr['norm_pct_error'],
                'unit': fr['unit'],
            }
            for name in family_spec['params']:
                comp = fr['components'][name]
                out[f'{name}_est_display'] = comp['est_display']
                out[f'{name}_true_display'] = comp['true_display']
                out[f'{name}_pct_error'] = comp['pct_error']
            writer.writerow(out)
    return str(csv_path)


def write_wide_csv(seed_rows: list[dict], out_dir: Path) -> str:
    csv_path = out_dir / 'g4_custom_noise_mc20_wide.csv'
    fieldnames = ['seed', 'status', 'result_json', 'mean_pct_error', 'median_pct_error', 'max_pct_error']
    for family_spec in FAMILY_SPECS:
        family_key = family_spec['family_key']
        for name in family_spec['params']:
            fieldnames.extend([
                f'{name}_est_display',
                f'{name}_pct_error',
            ])
        fieldnames.append(f'{family_key}_norm_est_display')
        fieldnames.append(f'{family_key}_norm_pct_error')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in seed_rows:
            out = {
                'seed': row['seed'],
                'status': row['status'],
                'result_json': row['result_json'],
                'mean_pct_error': row['overall']['mean_pct_error'],
                'median_pct_error': row['overall']['median_pct_error'],
                'max_pct_error': row['overall']['max_pct_error'],
            }
            for family_spec in FAMILY_SPECS:
                family_key = family_spec['family_key']
                fr = row['families'][family_key]
                for name in family_spec['params']:
                    comp = fr['components'][name]
                    out[f'{name}_est_display'] = comp['est_display']
                    out[f'{name}_pct_error'] = comp['pct_error']
                out[f'{family_key}_norm_est_display'] = fr['norm_est_display']
                out[f'{family_key}_norm_pct_error'] = fr['norm_pct_error']
            writer.writerow(out)
    return str(csv_path)


def render_report(seed_rows: list[dict], payload: dict) -> str:
    cfg = payload['noise_config']
    overall_mean = np.asarray([row['overall']['mean_pct_error'] for row in seed_rows], dtype=float)
    overall_median = np.asarray([row['overall']['median_pct_error'] for row in seed_rows], dtype=float)
    overall_max = np.asarray([row['overall']['max_pct_error'] for row in seed_rows], dtype=float)

    lines = []
    lines.append('# G4 @ custom noise：20 次重复性实验')
    lines.append('')
    lines.append('## 1. 实验口径')
    lines.append('')
    lines.append(f"- 方法：{ROUND61_DISPLAY}")
    lines.append('- 路径：corrected symmetric20 (20位置)')
    lines.append('- 重复方式：改变随机种子，顺序运行 20 次')
    lines.append(f"- seeds：{payload['seeds']}")
    lines.append(f"- arw = {cfg['arw_dpsh']} dpsh")
    lines.append(f"- vrw = {cfg['vrw_ugpsHz']} ugpsHz")
    lines.append(f"- bi_g = {cfg['bi_g_dph']} dph")
    lines.append(f"- bi_a = {cfg['bi_a_ug']} ug")
    lines.append(f"- tau_g = {cfg['tau_g']} s, tau_a = {cfg['tau_a']} s")
    lines.append('')
    lines.append('## 2. overall 指标分布')
    lines.append('')
    lines.append(f"- mean%%: μ={overall_mean.mean():.6f}, σ={sample_std(overall_mean):.6f}, min={overall_mean.min():.6f}, max={overall_mean.max():.6f}")
    lines.append(f"- median%%: μ={overall_median.mean():.6f}, σ={sample_std(overall_median):.6f}, min={overall_median.min():.6f}, max={overall_median.max():.6f}")
    lines.append(f"- max%%: μ={overall_max.mean():.6f}, σ={sample_std(overall_max):.6f}, min={overall_max.min():.6f}, max={overall_max.max():.6f}")
    lines.append('')
    lines.append('## 3. 参数族图')
    lines.append('')
    for family_spec in FAMILY_SPECS:
        family_key = family_spec['family_key']
        stats = payload['family_stats'][family_key]
        lines.append(f"### {family_spec['title']}")
        lines.append(f"- unit: {family_spec['unit']}")
        lines.append(f"- norm: true={stats['norm']['true_display']:.6f}, μ={stats['norm']['mean_est_display']:.6f}, σ={stats['norm']['std_est_display']:.6f}")
        lines.append(f"- 图：`{payload['plots'][family_key]['png']}` / `{payload['plots'][family_key]['svg']}`")
        lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    dirs = ensure_dirs()
    seeds = list(range(int(args.seed_start), int(args.seed_start) + int(args.n_runs)))

    seed_rows = []
    for idx, seed in enumerate(seeds, start=1):
        cfg = build_cfg(args, seed)
        payload, status, json_path = run_or_reuse_g4_for_seed(cfg, force_rerun=args.force_rerun)
        suffix = base.custom_suffix(cfg)
        raw_path = dirs['raw'] / f'g4_repeat_seed{seed}_{suffix}.json'
        raw_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

        source_mod, *_ = base.load_modules(f'postproc_{suffix}')
        row = build_seed_row(seed, payload, source_mod, json_path, status)
        seed_rows.append(row)
        progress = {
            'completed': idx,
            'total': len(seeds),
            'last_seed': seed,
            'timestamp': datetime.now().isoformat(timespec='seconds'),
        }
        (dirs['root'] / 'progress.json').write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding='utf-8')

    plots = {}
    family_csvs = {}
    family_stats_payload = {}
    for family_spec in FAMILY_SPECS:
        family_key = family_spec['family_key']
        plots[family_key] = plot_family(seed_rows, family_spec, dirs['plots'])
        family_csvs[family_key] = write_family_csv(seed_rows, family_spec, dirs['root'])
        family_stats_payload[family_key] = family_stats(seed_rows, family_spec)

    wide_csv = write_wide_csv(seed_rows, dirs['root'])

    out_payload = {
        'experiment': 'g4_custom_noise_repeatability_mc20',
        'method': ROUND61_DISPLAY,
        'noise_config': build_cfg(args, seeds[0]) | {'seed': None},
        'seeds': seeds,
        'n_runs': len(seeds),
        'wide_csv': wide_csv,
        'family_csvs': family_csvs,
        'plots': plots,
        'family_stats': family_stats_payload,
        'per_seed': seed_rows,
    }
    summary_json = dirs['root'] / 'g4_custom_noise_mc20_summary.json'
    summary_json.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    report_md = dirs['root'] / 'g4_custom_noise_mc20_report.md'
    report_md.write_text(render_report(seed_rows, out_payload), encoding='utf-8')

    result = {
        'summary_json': str(summary_json),
        'report_md': str(report_md),
        'wide_csv': wide_csv,
        'plots_dir': str(dirs['plots']),
        'family_plot_count': len(plots),
        'n_runs': len(seeds),
        'seeds': seeds,
    }
    print('__RESULT_JSON__=' + json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
