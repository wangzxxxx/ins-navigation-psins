from __future__ import annotations

import argparse
import json
import sys
import types
from datetime import datetime
from pathlib import Path

if 'matplotlib' not in sys.modules:
    matplotlib_stub = types.ModuleType('matplotlib')
    pyplot_stub = types.ModuleType('matplotlib.pyplot')
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules['matplotlib'] = matplotlib_stub
    sys.modules['matplotlib.pyplot'] = pyplot_stub
if 'seaborn' not in sys.modules:
    sys.modules['seaborn'] = types.ModuleType('seaborn')

ROOT = Path('/root/.openclaw/workspace')
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
REPORTS_DIR = ROOT / 'reports'
METHOD_DIR = ROOT / 'psins_method_bench' / 'methods' / 'markov'
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'
COMPUTE_R61_FILE = SCRIPTS_DIR / 'compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors.py'
CH3_EXTRACT_FILE = ROOT / 'tmp_ch3_selfcal_extract.txt'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import load_module
from compare_four_methods_shared_noise import (
    _load_json,
    _noise_matches,
    build_output_paths,
    compute_payload,
    expected_noise_config,
)
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs

METHOD_KEYS = ['kf36_noisy', 'markov42_noisy']
METHOD_DISPLAY = {
    'kf36_noisy': 'KF36 baseline (36-state noisy baseline)',
    'markov42_noisy': 'Markov42 baseline (42-state GM1 baseline)',
}
PATH_KEYS = ['default_path', 'chapter3_12pos_reconstructed']
PATH_DISPLAY = {
    'default_path': 'Current default path (18-position shared dataset path)',
    'chapter3_12pos_reconstructed': 'Chapter-3 faithful reconstructed 12-position path',
}
PATH_TAG = {
    'default_path': 'default18',
    'chapter3_12pos_reconstructed': 'ch3faithful12',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=0.08)
    parser.add_argument('--report-date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def make_suffix(noise_scale: float) -> str:
    if abs(noise_scale - 1.0) < 1e-12:
        return 'noise1p0'
    if abs(noise_scale - 2.0) < 1e-12:
        return 'noise2p0'
    return f"noise{str(noise_scale).replace('.', 'p')}"


def build_default_path_paras(mod):
    paras = mod.np.array([
        [1, 0, 1, 0, 90, 9, 70, 70],
        [2, 0, 1, 0, 90, 9, 20, 20],
        [3, 0, 1, 0, 90, 9, 20, 20],
        [4, 0, 1, 0, -90, 9, 20, 20],
        [5, 0, 1, 0, -90, 9, 20, 20],
        [6, 0, 1, 0, -90, 9, 20, 20],
        [7, 0, 0, 1, 90, 9, 20, 20],
        [8, 1, 0, 0, 90, 9, 20, 20],
        [9, 1, 0, 0, 90, 9, 20, 20],
        [10, 1, 0, 0, 90, 9, 20, 20],
        [11, -1, 0, 0, 90, 9, 20, 20],
        [12, -1, 0, 0, 90, 9, 20, 20],
        [13, -1, 0, 0, 90, 9, 20, 20],
        [14, 0, 0, 1, 90, 9, 20, 20],
        [15, 0, 0, 1, 90, 9, 20, 20],
        [16, 0, 0, -1, 90, 9, 20, 20],
        [17, 0, 0, -1, 90, 9, 20, 20],
        [18, 0, 0, -1, 90, 9, 20, 20],
    ], dtype=float)
    paras[:, 4] = paras[:, 4] * mod.glv.deg
    return paras


# Exact axis-angle order recovered from tmp_ch3_selfcal_extract.txt.
# The thesis table provides only "rotation 10 s + dwell 90 s" per node,
# while the current simulator requires a split [Trot, T_pre, T_post].
# We therefore reconstruct each 100 s node as [10, 10, 80]:
# - preserves 12 nodes / 1200 s total / 10 s rotation budget
# - preserves 90 s static interval between consecutive rotations (80 + next 10)
# - provides the minimum initial static window needed by the current coarse alignment code.
def build_ch3_path_paras(mod):
    rows = [
        [1, 1, 0, 0, 90, 10, 10, 80],
        [2, 1, 0, 0, 90, 10, 10, 80],
        [3, 0, 1, 0, 90, 10, 10, 80],
        [4, 0, 0, 1, -90, 10, 10, 80],
        [5, 0, 0, 1, -90, 10, 10, 80],
        [6, 0, 1, 0, -90, 10, 10, 80],
        [7, 1, 0, 0, -90, 10, 10, 80],
        [8, 1, 0, 0, -90, 10, 10, 80],
        [9, 0, 1, 0, -90, 10, 10, 80],
        [10, 0, 0, 1, 90, 10, 10, 80],
        [11, 0, 0, 1, 90, 10, 10, 80],
        [12, 0, 1, 0, 90, 10, 10, 80],
    ]
    paras = mod.np.array(rows, dtype=float)
    paras[:, 4] = paras[:, 4] * mod.glv.deg
    return paras


# User clarified on 2026-04-02 that the chapter-3 twelve-position method starts from
# att0 = (0, 0, 0). Keep this helper as the single source of truth for the faithful12
# path-search / benchmark line so downstream search scripts inherit the same setup.
def build_ch3_initial_attitude(mod):
    return mod.np.array([0.0, 0.0, 0.0]) * mod.glv.deg


def paras_to_rows(mod, paras):
    rows = []
    for row in paras.tolist():
        rows.append({
            'pos_id': int(row[0]),
            'axis': [int(row[1]), int(row[2]), int(row[3])],
            'angle_deg': float(row[4] / mod.glv.deg),
            'rotation_time_s': float(row[5]),
            'pre_static_s': float(row[6]),
            'post_static_s': float(row[7]),
            'node_total_s': float(row[5] + row[6] + row[7]),
        })
    return rows


def summarize_path(mod, path_key: str, paras, provenance: dict) -> dict:
    rows = paras_to_rows(mod, paras)
    axis_counts = {}
    angle_counts = {}
    total_time_s = 0.0
    for row in rows:
        axis_key = str(row['axis'])
        angle_key = f"{row['angle_deg']:+.0f}"
        axis_counts[axis_key] = axis_counts.get(axis_key, 0) + 1
        angle_counts[angle_key] = angle_counts.get(angle_key, 0) + 1
        total_time_s += row['node_total_s']
    return {
        'path_key': path_key,
        'display_name': PATH_DISPLAY[path_key],
        'n_positions': len(rows),
        'total_time_s': total_time_s,
        'axis_counts': axis_counts,
        'angle_counts': angle_counts,
        'rows': rows,
        'provenance': provenance,
    }


def build_dataset_with_path(mod, noise_scale: float, paras):
    ts = 0.01
    att0 = build_ch3_initial_attitude(mod)
    pos0 = mod.posset(34.0, 0.0, 0.0)
    att = mod.attrottt(att0, paras, ts)
    imu, _ = mod.avp2imu(att, pos0)
    clbt_truth = mod.get_default_clbt()
    imu_clean = mod.imuclbt(imu, clbt_truth)

    cfg = expected_noise_config(noise_scale)
    arw = cfg['arw_dpsh'] * mod.glv.dpsh
    vrw = cfg['vrw_ugpsHz'] * mod.glv.ugpsHz
    bi_g = cfg['bi_g_dph'] * mod.glv.dph
    bi_a = cfg['bi_a_ug'] * mod.glv.ug

    imu_noisy = mod.imuadderr_full(
        imu_clean,
        ts,
        arw=arw,
        vrw=vrw,
        bi_g=bi_g,
        tau_g=cfg['tau_g'],
        bi_a=bi_a,
        tau_a=cfg['tau_a'],
        seed=cfg['seed'],
    )

    return {
        'ts': ts,
        'pos0': pos0,
        'imu_noisy': imu_noisy,
        'bi_g': bi_g,
        'bi_a': bi_a,
        'tau_g': cfg['tau_g'],
        'tau_a': cfg['tau_a'],
        'noise_scale': noise_scale,
        'noise_config': cfg,
    }


def path_method_output_path(path_key: str, method_key: str, noise_scale: float) -> Path:
    suffix = make_suffix(noise_scale)
    path_tag = PATH_TAG[path_key]
    if method_key == 'kf36_noisy':
        return RESULTS_DIR / f'KF36_{path_tag}_shared_{suffix}_param_errors.json'
    if method_key == 'markov42_noisy':
        return RESULTS_DIR / f'M_markov_42state_gm1_{path_tag}_shared_{suffix}_param_errors.json'
    raise KeyError(method_key)


def load_or_run_default_payload(method_key: str, noise_scale: float, report_date: str, force_rerun: bool) -> tuple[dict, str, Path]:
    expected_cfg = expected_noise_config(noise_scale)
    _, compare_paths = build_output_paths(noise_scale, report_date)
    if method_key == 'kf36_noisy':
        path = compare_paths['kf36_noisy']
    elif method_key == 'markov42_noisy':
        path = compare_paths['markov42_noisy']
    else:
        raise KeyError(method_key)

    if (not force_rerun) and path.exists():
        payload = _load_json(path)
        if _noise_matches(payload, expected_cfg):
            return payload, 'reused_verified', path
    raise FileNotFoundError(f'Default baseline missing or mismatched for {method_key}: {path}')


def run_path_method(source_mod, path_key: str, paras, method_key: str, noise_scale: float, force_rerun: bool = False) -> tuple[dict, str, Path]:
    out_path = path_method_output_path(path_key, method_key, noise_scale)
    expected_cfg = expected_noise_config(noise_scale)
    if (not force_rerun) and out_path.exists():
        payload = _load_json(out_path)
        if _noise_matches(payload, expected_cfg):
            extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
            if extra.get('path_key') == path_key:
                return payload, 'reused_verified', out_path

    dataset = build_dataset_with_path(source_mod, noise_scale, paras)
    params = _param_specs(source_mod)

    if method_key == 'kf36_noisy':
        result = source_mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=36,
            label=f'KF36-{path_key.upper()}-{make_suffix(noise_scale).upper()}',
        )
        payload = compute_payload(
            source_mod,
            result[0],
            params,
            variant=f'kf36_{PATH_TAG[path_key]}_shared_{make_suffix(noise_scale)}',
            method_file='source_mod.run_calibration(n_states=36) on chapter3/custom path',
            extra={
                'noise_scale': noise_scale,
                'noise_config': dataset['noise_config'],
                'comparison_mode': 'chapter3_12pos_vs_default_baselines',
                'path_key': path_key,
                'path_tag': PATH_TAG[path_key],
                'method_key': method_key,
            },
        )
    elif method_key == 'markov42_noisy':
        result = source_mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=42,
            bi_g=dataset['bi_g'],
            tau_g=dataset['tau_g'],
            bi_a=dataset['bi_a'],
            tau_a=dataset['tau_a'],
            label=f'MARKOV42-{path_key.upper()}-{make_suffix(noise_scale).upper()}',
        )
        payload = compute_payload(
            source_mod,
            result[0],
            params,
            variant=f'42state_gm1_{PATH_TAG[path_key]}_shared_{make_suffix(noise_scale)}',
            method_file='source_mod.run_calibration(n_states=42) on chapter3/custom path',
            extra={
                'noise_scale': noise_scale,
                'noise_config': dataset['noise_config'],
                'comparison_mode': 'chapter3_12pos_vs_default_baselines',
                'path_key': path_key,
                'path_tag': PATH_TAG[path_key],
                'method_key': method_key,
            },
        )
    else:
        raise KeyError(method_key)

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', out_path


def classify_delta(base_overall: dict, challenger_overall: dict) -> dict:
    metrics = {}
    better = 0
    worse = 0
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        base_v = float(base_overall[metric])
        new_v = float(challenger_overall[metric])
        delta = base_v - new_v
        improved = delta > 0
        metrics[metric] = {
            'default_path_value': base_v,
            'chapter3_path_value': new_v,
            'improvement_pct_points': delta,
            'relative_improvement_pct': (delta / base_v * 100.0) if abs(base_v) > 1e-15 else None,
            'chapter3_better': improved,
        }
        if improved:
            better += 1
        elif delta < 0:
            worse += 1
    if better == 3:
        verdict = 'better'
    elif worse == 3:
        verdict = 'worse'
    else:
        verdict = 'mixed'
    return {'metrics': metrics, 'verdict': verdict}


def compare_params(default_payload: dict, ch3_payload: dict) -> dict:
    better = []
    worse = []
    same = []
    for name in default_payload['param_order']:
        dv = float(default_payload['param_errors'][name]['pct_error'])
        cv = float(ch3_payload['param_errors'][name]['pct_error'])
        delta = dv - cv
        item = {
            'param': name,
            'default_pct_error': dv,
            'chapter3_pct_error': cv,
            'improvement_pct_points': delta,
            'relative_improvement_pct': (delta / dv * 100.0) if abs(dv) > 1e-15 else None,
        }
        if delta > 1e-12:
            better.append(item)
        elif delta < -1e-12:
            worse.append(item)
        else:
            same.append(item)
    better.sort(key=lambda x: x['improvement_pct_points'], reverse=True)
    worse.sort(key=lambda x: x['improvement_pct_points'])
    return {
        'better_count': len(better),
        'worse_count': len(worse),
        'same_count': len(same),
        'top_better': better[:10],
        'top_worse': worse[:10],
    }


def verdict_summary(path_swap_delta: dict) -> str:
    verdicts = [path_swap_delta[m]['overall']['verdict'] for m in METHOD_KEYS]
    if all(v == 'better' for v in verdicts):
        return 'chapter3_better_under_both_baselines'
    if all(v == 'worse' for v in verdicts):
        return 'chapter3_worse_under_both_baselines'
    return 'mixed_or_ambiguous'


def render_provenance_md(prov: dict) -> str:
    lines = []
    lines.append('# Chapter-3 12-position path provenance')
    lines.append('')
    lines.append(f"- source extract: `{prov['source_extract_file']}`")
    lines.append(f"- status: **{prov['reconstruction_status']}**")
    lines.append(f"- reason: {prov['reconstruction_reason']}")
    lines.append('')
    lines.append('## Recovered thesis path (axis-angle order)')
    lines.append('')
    lines.append('| idx | axis | angle_deg | rot_s | thesis_static_s | reconstructed_pre_s | reconstructed_post_s |')
    lines.append('|---|---|---:|---:|---:|---:|---:|')
    for row in prov['reconstructed_rows']:
        lines.append(
            f"| {row['pos_id']} | {row['axis']} | {row['angle_deg']:.0f} | {row['rotation_time_s']:.0f} | 90 | {row['pre_static_s']:.0f} | {row['post_static_s']:.0f} |"
        )
    lines.append('')
    lines.append('## Notes')
    lines.append('')
    for note in prov['notes']:
        lines.append(f'- {note}')
    lines.append('')
    return '\n'.join(lines) + '\n'


def render_report(summary: dict) -> str:
    lines = []
    lines.append('<callout emoji="🧭" background-color="light-blue">')
    lines.append('只替换标定路径，其余 truth model / noise family / seed / calibration code 保持不变。本报告只比较 baseline：KF36 与 Markov42。')
    lines.append('</callout>')
    lines.append('')

    cfg = summary['noise_config']
    lines.append('## 1. Fixed benchmark setup')
    lines.append('')
    lines.append(f"- noise_scale = {summary['noise_scale']}")
    lines.append(f"- seed = {cfg['seed']}")
    lines.append(f"- arw = {cfg['arw_dpsh']} dpsh")
    lines.append(f"- vrw = {cfg['vrw_ugpsHz']} ugpsHz")
    lines.append(f"- bi_g = {cfg['bi_g_dph']} dph")
    lines.append(f"- bi_a = {cfg['bi_a_ug']} ug")
    lines.append(f"- tau_g = tau_a = {cfg['tau_g']}")
    lines.append('- truth model = same `get_default_clbt()` source')
    lines.append('- dataset family = same shared-noise family; only path swapped')
    lines.append('')

    lines.append('## 2. Paths')
    lines.append('')
    lines.append('| path | n_positions | total_time_s | note |')
    lines.append('|---|---:|---:|---|')
    for path_key in PATH_KEYS:
        p = summary['paths'][path_key]
        note = p['provenance'].get('short_note', '')
        lines.append(f"| {p['display_name']} | {p['n_positions']} | {p['total_time_s']:.0f} | {note} |")
    lines.append('')

    lines.append('## 3. Overall metrics (30 params, lower is better)')
    lines.append('')
    lines.append('| method | default path mean/med/max | chapter3 path mean/med/max | verdict |')
    lines.append('|---|---|---|---|')
    for method_key in METHOD_KEYS:
        d = summary['runs']['default_path'][method_key]['overall']
        c = summary['runs']['chapter3_12pos_reconstructed'][method_key]['overall']
        verdict = summary['path_swap_delta'][method_key]['overall']['verdict']
        lines.append(
            f"| {METHOD_DISPLAY[method_key]} | {d['mean_pct_error']:.6f}/{d['median_pct_error']:.6f}/{d['max_pct_error']:.6f} | {c['mean_pct_error']:.6f}/{c['median_pct_error']:.6f}/{c['max_pct_error']:.6f} | **{verdict}** |"
        )
    lines.append('')

    lines.append('## 4. Direct answer')
    lines.append('')
    lines.append(f"- overall conclusion tag: **{summary['overall_conclusion']}**")
    for method_key in METHOD_KEYS:
        delta = summary['path_swap_delta'][method_key]['overall']
        mm = delta['metrics']
        lines.append(
            f"- **{METHOD_DISPLAY[method_key]}**: {delta['verdict']} — "
            f"mean {mm['mean_pct_error']['default_path_value']:.6f} → {mm['mean_pct_error']['chapter3_path_value']:.6f}, "
            f"median {mm['median_pct_error']['default_path_value']:.6f} → {mm['median_pct_error']['chapter3_path_value']:.6f}, "
            f"max {mm['max_pct_error']['default_path_value']:.6f} → {mm['max_pct_error']['chapter3_path_value']:.6f}."
        )
    lines.append('')

    lines.append('## 5. Param-count view')
    lines.append('')
    lines.append('| method | improved params | worsened params | unchanged |')
    lines.append('|---|---:|---:|---:|')
    for method_key in METHOD_KEYS:
        delta = summary['path_swap_delta'][method_key]['params']
        lines.append(f"| {METHOD_DISPLAY[method_key]} | {delta['better_count']} | {delta['worse_count']} | {delta['same_count']} |")
    lines.append('')

    lines.append('## 6. Output files')
    lines.append('')
    lines.append(f"- provenance_json: `{summary['files']['provenance_json']}`")
    lines.append(f"- provenance_md: `{summary['files']['provenance_md']}`")
    lines.append(f"- compare_json: `{summary['files']['compare_json']}`")
    lines.append(f"- report_md: `{summary['files']['report_md']}`")
    for path_key in PATH_KEYS:
        for method_key in METHOD_KEYS:
            lines.append(f"- run_json[{path_key}][{method_key}]: `{summary['files']['run_jsons'][path_key][method_key]}`")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    noise_scale = args.noise_scale
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    source_mod = load_module(f'bridge_baselines_source_{make_suffix(noise_scale)}', str(SOURCE_FILE))
    default_paras = build_default_path_paras(source_mod)
    ch3_paras = build_ch3_path_paras(source_mod)

    ch3_provenance = {
        'source_extract_file': str(CH3_EXTRACT_FILE),
        'reconstruction_status': 'faithful_reconstructed_12pos',
        'reconstruction_reason': 'Thesis artifact provides exact 12-node axis-angle order and 10 s + 90 s timing, but current simulator needs [rotation, pre-static, post-static]. Reconstructed as [10, 10, 80] per node to preserve 12 nodes / 1200 s / same axis-angle sequence while providing the minimum initial static window required by the current coarse-alignment code.',
        'short_note': 'exact axis-angle order recovered; timing faithfully reconstructed as [10,10,80] because simulator requires pre/post split.',
        'notes': [
            'Exact axis-angle sequence comes from the chapter-3 extracted table in tmp_ch3_selfcal_extract.txt.',
            'The thesis table is 12 nodes, each with 10 s rotation and 90 s dwell, totaling 1200 s.',
            'Current PSINS paras format is [axis, angle, Trot, T_pre, T_post], so an exact one-column dwell cannot be represented literally.',
            'Using [10, 10, 80] keeps total time 1200 s and keeps 90 s static between consecutive rotations (80 + next-node 10).',
            'A smaller T_pre would make the current coarse alignment slice too short; 10 s is the smallest clean reconstruction chosen here.',
        ],
        'reconstructed_rows': paras_to_rows(source_mod, ch3_paras),
    }

    default_provenance = {
        'source_extract_file': str(SOURCE_FILE),
        'reconstruction_status': 'existing_default_path',
        'reconstruction_reason': 'Current default shared dataset path copied directly from compare_four_methods_shared_noise.py / test_calibration_markov_pruned.py lineage.',
        'short_note': '18-position current default path reused as-is.',
        'notes': ['This is the current default path used by the shared-dataset PSINS comparison scripts.'],
    }

    paths_summary = {
        'default_path': summarize_path(source_mod, 'default_path', default_paras, default_provenance),
        'chapter3_12pos_reconstructed': summarize_path(source_mod, 'chapter3_12pos_reconstructed', ch3_paras, ch3_provenance),
    }

    suffix = make_suffix(noise_scale)
    provenance_json = RESULTS_DIR / f'ch3_12pos_path_provenance_{suffix}.json'
    provenance_md = REPORTS_DIR / f'psins_ch3_12pos_path_provenance_{args.report_date}_{suffix}.md'
    compare_json = RESULTS_DIR / f'compare_ch3_12pos_vs_default_baselines_{suffix}.json'
    report_md = REPORTS_DIR / f'psins_ch3_12pos_baselines_{suffix}_{args.report_date}.md'

    provenance_payload = {
        'noise_scale': noise_scale,
        'paths': paths_summary,
    }
    provenance_json.write_text(json.dumps(provenance_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    provenance_md.write_text(render_provenance_md(ch3_provenance), encoding='utf-8')

    runs = {path_key: {} for path_key in PATH_KEYS}
    execution = {path_key: {} for path_key in PATH_KEYS}
    run_jsons = {path_key: {} for path_key in PATH_KEYS}

    for method_key in METHOD_KEYS:
        payload, status, path = load_or_run_default_payload(method_key, noise_scale, args.report_date, force_rerun=args.force_rerun)
        runs['default_path'][method_key] = payload
        execution['default_path'][method_key] = status
        run_jsons['default_path'][method_key] = str(path)

    for method_key in METHOD_KEYS:
        payload, status, path = run_path_method(
            source_mod,
            'chapter3_12pos_reconstructed',
            ch3_paras,
            method_key,
            noise_scale,
            force_rerun=args.force_rerun,
        )
        runs['chapter3_12pos_reconstructed'][method_key] = payload
        execution['chapter3_12pos_reconstructed'][method_key] = status
        run_jsons['chapter3_12pos_reconstructed'][method_key] = str(path)

    path_swap_delta = {}
    for method_key in METHOD_KEYS:
        path_swap_delta[method_key] = {
            'overall': classify_delta(
                runs['default_path'][method_key]['overall'],
                runs['chapter3_12pos_reconstructed'][method_key]['overall'],
            ),
            'params': compare_params(
                runs['default_path'][method_key],
                runs['chapter3_12pos_reconstructed'][method_key],
            ),
        }

    summary = {
        'experiment': 'chapter3_12pos_vs_default_baselines',
        'noise_scale': noise_scale,
        'noise_config': expected_noise_config(noise_scale),
        'source_file': str(SOURCE_FILE),
        'report_date': args.report_date,
        'paths': paths_summary,
        'runs': {
            path_key: {
                method_key: {
                    'variant': runs[path_key][method_key]['variant'],
                    'overall': runs[path_key][method_key]['overall'],
                    'focus_scale_pct': runs[path_key][method_key]['focus_scale_pct'],
                    'lever_guard_pct': runs[path_key][method_key]['lever_guard_pct'],
                }
                for method_key in METHOD_KEYS
            }
            for path_key in PATH_KEYS
        },
        'path_swap_delta': path_swap_delta,
        'overall_conclusion': verdict_summary(path_swap_delta),
        'execution': execution,
        'files': {
            'provenance_json': str(provenance_json),
            'provenance_md': str(provenance_md),
            'compare_json': str(compare_json),
            'report_md': str(report_md),
            'run_jsons': run_jsons,
        },
    }

    compare_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_md.write_text(render_report(summary), encoding='utf-8')

    result = {
        'noise_scale': noise_scale,
        'compare_json': str(compare_json),
        'report_md': str(report_md),
        'provenance_json': str(provenance_json),
        'provenance_md': str(provenance_md),
        'execution': execution,
        'overall_conclusion': summary['overall_conclusion'],
        'kf36_default': runs['default_path']['kf36_noisy']['overall'],
        'kf36_ch3': runs['chapter3_12pos_reconstructed']['kf36_noisy']['overall'],
        'kf36_verdict': path_swap_delta['kf36_noisy']['overall']['verdict'],
        'markov42_default': runs['default_path']['markov42_noisy']['overall'],
        'markov42_ch3': runs['chapter3_12pos_reconstructed']['markov42_noisy']['overall'],
        'markov42_verdict': path_swap_delta['markov42_noisy']['overall']['verdict'],
    }
    print('__RESULT_JSON__=' + json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
