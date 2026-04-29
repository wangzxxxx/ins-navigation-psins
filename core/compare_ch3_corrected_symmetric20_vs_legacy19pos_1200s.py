from __future__ import annotations

import argparse
import json
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any

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
TMP_PSINS_DIR = ROOT / 'tmp_psins_py'
SOURCE_FILE = TMP_PSINS_DIR / 'psins_py' / 'test_calibration_markov_pruned.py'
LEGACY_19POS_FILE = TMP_PSINS_DIR / 'psins_py' / 'test_system_calibration_19pos.py'
SYMMETRIC20_PROBE_FILE = SCRIPTS_DIR / 'probe_ch3_corrected_symmetric20_front2_back11.py'

for p in [ROOT, TMP_PSINS_DIR, METHOD_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module
from compare_four_methods_shared_noise import (
    _load_json,
    _noise_matches,
    compute_payload,
    expected_noise_config,
    make_suffix,
)
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs

DEFAULT_NOISE_SCALE = 0.12
TARGET_TOTAL_S = 1200.0
COMPARISON_MODE = 'corrected_symmetric20_vs_legacy19pos_own_att0_normalized1200s'
METHOD_KEYS = ['markov42_noisy', 'kf36_noisy']
METHOD_LABELS = {
    'markov42_noisy': 'Markov42',
    'kf36_noisy': 'KF36',
}
CASE_KEYS = ['symmetric20', 'legacy19pos']
CASE_TAGS = {
    'symmetric20': 'ch3corrected_symmetric20_att0zero_1200s',
    'legacy19pos': 'legacy19pos_histatt0_scaled1200s',
}
CASE_LABELS = {
    'symmetric20': '20-position corrected symmetric scheme',
    'legacy19pos': 'legacy 19-position method',
}
KEY_PARAMS = ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'Ka2_y', 'Ka2_z', 'rx_y', 'ry_z']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=DEFAULT_NOISE_SCALE)
    parser.add_argument('--report-date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def paras_to_rows(mod, paras) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def rows_to_paras(mod, rows: list[dict[str, Any]]):
    arr = []
    for idx, row in enumerate(rows, start=1):
        arr.append([
            idx,
            int(row['axis'][0]),
            int(row['axis'][1]),
            int(row['axis'][2]),
            float(row['angle_deg']),
            float(row['rotation_time_s']),
            float(row['pre_static_s']),
            float(row['post_static_s']),
        ])
    paras = mod.np.array(arr, dtype=float)
    paras[:, 4] = paras[:, 4] * mod.glv.deg
    return paras


def build_symmetric20_case(mod) -> dict[str, Any]:
    probe_mod = load_module('compare_sym20_probe_module', str(SYMMETRIC20_PROBE_FILE))
    candidate = probe_mod.build_symmetric20_candidate(mod)
    paras = rows_to_paras(mod, candidate.all_rows)
    return {
        'case_key': 'symmetric20',
        'case_tag': CASE_TAGS['symmetric20'],
        'display_name': CASE_LABELS['symmetric20'],
        'att0_deg': [0.0, 0.0, 0.0],
        'paras': paras,
        'rows': candidate.all_rows,
        'n_motion_rows': len(candidate.all_rows),
        'claimed_position_count': len(candidate.all_rows),
        'total_time_s': float(candidate.total_time_s),
        'timing_note': 'uniform 20 × 60 s = 1200 s, row split 6 / 6 / 48 s',
        'source_builder': str(SYMMETRIC20_PROBE_FILE),
        'source_reference': 'anchor2_zpair_anchor11_xpair_symmetric20_60s',
        'rationale': candidate.rationale,
        'builder_method_tag': candidate.method_tag,
        'continuity_checks': candidate.continuity_checks,
    }


def build_legacy19pos_case(mod) -> dict[str, Any]:
    original_rows = [
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
    ]
    raw_total_s = float(sum(float(r[5] + r[6] + r[7]) for r in original_rows))
    scale = TARGET_TOTAL_S / raw_total_s
    scaled = mod.np.array(original_rows, dtype=float)
    scaled[:, 5:8] *= scale
    scaled[:, 4] *= mod.glv.deg
    return {
        'case_key': 'legacy19pos',
        'case_tag': CASE_TAGS['legacy19pos'],
        'display_name': CASE_LABELS['legacy19pos'],
        'att0_deg': [1.0, -91.0, -91.0],
        'paras': scaled,
        'rows': paras_to_rows(mod, scaled),
        'n_motion_rows': int(scaled.shape[0]),
        'claimed_position_count': 19,
        'total_time_s': float(sum(float(x) for x in scaled[:, 5:8].sum(axis=1))),
        'timing_note': 'original 982 s over 18 motion rows, scaled uniformly by 1200/982 to preserve row timing proportions',
        'source_builder': str(LEGACY_19POS_FILE),
        'source_reference': 'historical paras block + att0=[1,-91,-91] deg from original 19pos script',
        'rationale': 'Historical 19-position calibration path, evaluated with its original initial attitude and fair-duration normalization only.',
        'timing_scale_factor': float(scale),
        'raw_total_s_before_scaling': raw_total_s,
        'historical_note': 'Script name is 19pos, but the paras block contains 18 motion rows; the 19th position is the historical starting attitude/state.',
    }


def build_dataset(mod, paras, att0_deg: list[float], noise_scale: float) -> dict[str, Any]:
    ts = 0.01
    att0 = mod.np.array(att0_deg, dtype=float) * mod.glv.deg
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
        'att0_deg': att0_deg,
        'imu_noisy': imu_noisy,
        'bi_g': bi_g,
        'bi_a': bi_a,
        'tau_g': cfg['tau_g'],
        'tau_a': cfg['tau_a'],
        'noise_scale': noise_scale,
        'noise_config': cfg,
    }


def output_path(case_key: str, method_key: str, noise_scale: float) -> Path:
    suffix = make_suffix(noise_scale)
    if method_key == 'markov42_noisy':
        return RESULTS_DIR / f"M_markov_42state_gm1_{CASE_TAGS[case_key]}_shared_{suffix}_param_errors.json"
    if method_key == 'kf36_noisy':
        return RESULTS_DIR / f"KF36_{CASE_TAGS[case_key]}_shared_{suffix}_param_errors.json"
    raise KeyError(method_key)


def compact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        'overall': payload['overall'],
        'key_param_errors': {name: payload['param_errors'][name]['pct_error'] for name in KEY_PARAMS},
    }


def triplet_text(overall: dict[str, float]) -> str:
    return f"{overall['mean_pct_error']:.3f} / {overall['median_pct_error']:.3f} / {overall['max_pct_error']:.3f}"


def run_case_method(mod, case: dict[str, Any], method_key: str, noise_scale: float, force_rerun: bool = False) -> tuple[dict[str, Any], str, Path]:
    out_path = output_path(case['case_key'], method_key, noise_scale)
    expected_cfg = expected_noise_config(noise_scale)
    if (not force_rerun) and out_path.exists():
        payload = _load_json(out_path)
        extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
        if (
            _noise_matches(payload, expected_cfg)
            and extra.get('comparison_mode') == COMPARISON_MODE
            and extra.get('case_key') == case['case_key']
            and extra.get('method_key') == method_key
        ):
            return payload, 'reused_verified', out_path

    dataset = build_dataset(mod, case['paras'], case['att0_deg'], noise_scale)
    params = _param_specs(mod)

    if method_key == 'markov42_noisy':
        result = mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=42,
            bi_g=dataset['bi_g'],
            tau_g=dataset['tau_g'],
            bi_a=dataset['bi_a'],
            tau_a=dataset['tau_a'],
            label=f"MARKOV42-{case['case_tag'].upper()}-{make_suffix(noise_scale).upper()}",
        )
        method_file = 'source_mod.run_calibration(n_states=42)'
    elif method_key == 'kf36_noisy':
        result = mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=36,
            label=f"KF36-{case['case_tag'].upper()}-{make_suffix(noise_scale).upper()}",
        )
        method_file = 'source_mod.run_calibration(n_states=36)'
    else:
        raise KeyError(method_key)

    payload = compute_payload(
        mod,
        result[0],
        params,
        variant=f"{CASE_TAGS[case['case_key']]}_{method_key}_{make_suffix(noise_scale)}",
        method_file=method_file,
        extra={
            'comparison_mode': COMPARISON_MODE,
            'case_key': case['case_key'],
            'case_tag': case['case_tag'],
            'case_display_name': case['display_name'],
            'method_key': method_key,
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'att0_deg': case['att0_deg'],
            'n_motion_rows': case['n_motion_rows'],
            'claimed_position_count': case['claimed_position_count'],
            'total_time_s': case['total_time_s'],
            'timing_note': case['timing_note'],
            'source_builder': case['source_builder'],
            'source_reference': case['source_reference'],
            'rationale': case['rationale'],
            'timing_scale_factor': case.get('timing_scale_factor'),
            'raw_total_s_before_scaling': case.get('raw_total_s_before_scaling'),
            'builder_method_tag': case.get('builder_method_tag'),
        },
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', out_path


def overall_delta(sym_payload: dict[str, Any], legacy_payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    better_count = 0
    worse_count = 0
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        sym_v = float(sym_payload['overall'][metric])
        legacy_v = float(legacy_payload['overall'][metric])
        delta = legacy_v - sym_v
        entry = {
            'symmetric20_value': sym_v,
            'legacy19pos_value': legacy_v,
            'improvement_pct_points': delta,
            'relative_improvement_pct': (delta / legacy_v * 100.0) if abs(legacy_v) > 1e-15 else None,
            'symmetric20_better': delta > 0,
        }
        out[metric] = entry
        if delta > 1e-12:
            better_count += 1
        elif delta < -1e-12:
            worse_count += 1
    verdict = 'mixed'
    if better_count == 3:
        verdict = 'symmetric20_better'
    elif worse_count == 3:
        verdict = 'legacy19pos_better'
    return {'metrics': out, 'verdict': verdict}


def compare_params(sym_payload: dict[str, Any], legacy_payload: dict[str, Any]) -> dict[str, Any]:
    rows = []
    sym_better = 0
    legacy_better = 0
    ties = 0
    for name in sym_payload['param_order']:
        true_v = float(sym_payload['param_errors'][name]['true'])
        sym_est = float(sym_payload['param_errors'][name]['est'])
        legacy_est = float(legacy_payload['param_errors'][name]['est'])
        sym_pct = float(sym_payload['param_errors'][name]['pct_error'])
        legacy_pct = float(legacy_payload['param_errors'][name]['pct_error'])
        delta = legacy_pct - sym_pct
        if delta > 1e-12:
            winner = 'symmetric20'
            sym_better += 1
        elif delta < -1e-12:
            winner = 'legacy19pos'
            legacy_better += 1
        else:
            winner = 'tie'
            ties += 1
        rows.append({
            'param': name,
            'true': true_v,
            'symmetric20_est': sym_est,
            'symmetric20_pct_error': sym_pct,
            'legacy19pos_est': legacy_est,
            'legacy19pos_pct_error': legacy_pct,
            'improvement_pct_points': delta,
            'relative_improvement_pct': (delta / legacy_pct * 100.0) if abs(legacy_pct) > 1e-15 else None,
            'winner': winner,
        })
    top_symmetric20 = sorted(rows, key=lambda x: x['improvement_pct_points'], reverse=True)[:10]
    top_legacy19 = sorted(rows, key=lambda x: x['improvement_pct_points'])[:10]
    return {
        'rows': rows,
        'symmetric20_better_count': sym_better,
        'legacy19pos_better_count': legacy_better,
        'tie_count': ties,
        'top_symmetric20_advantages': top_symmetric20,
        'top_legacy19pos_advantages': top_legacy19,
    }


def method_verdict_text(method_key: str, overall: dict[str, Any], params: dict[str, Any]) -> str:
    if overall['verdict'] == 'symmetric20_better':
        return (
            f"{METHOD_LABELS[method_key]}: 20-pos wins all three overall metrics and also wins "
            f"{params['symmetric20_better_count']}/{len(params['rows'])} parameters on pct-error."
        )
    if overall['verdict'] == 'legacy19pos_better':
        return (
            f"{METHOD_LABELS[method_key]}: legacy 19-pos wins all three overall metrics and wins "
            f"{params['legacy19pos_better_count']}/{len(params['rows'])} parameters on pct-error."
        )
    return (
        f"{METHOD_LABELS[method_key]}: mixed overall result; parameter wins are "
        f"20-pos {params['symmetric20_better_count']} vs legacy19 {params['legacy19pos_better_count']} (ties {params['tie_count']})."
    )


def consensus_text(comparisons: dict[str, Any]) -> str:
    verdicts = [comparisons[m]['overall']['verdict'] for m in METHOD_KEYS]
    if all(v == 'symmetric20_better' for v in verdicts):
        return 'Across both Markov42 and KF36, the corrected 20-step symmetric scheme is better than the normalized legacy 19-position method at noise_scale=0.12.'
    if all(v == 'legacy19pos_better' for v in verdicts):
        return 'Across both Markov42 and KF36, the normalized legacy 19-position method is better than the corrected 20-step symmetric scheme at noise_scale=0.12.'
    return 'The comparison is estimator-dependent or mixed; inspect the full parameter tables for the exact trade-off pattern.'


def fmt_e(x: float) -> str:
    return f'{x:.6e}'


def fmt_pct(x: float) -> str:
    return f'{x:.6f}'


def render_case_table(cases: dict[str, Any]) -> list[str]:
    lines = []
    lines.append('| scheme | att0 (deg) | motion rows | claimed positions | total time (s) | timing note |')
    lines.append('|---|---|---:|---:|---:|---|')
    for case_key in CASE_KEYS:
        case = cases[case_key]
        att0 = '(' + ', '.join(f'{v:g}' for v in case['att0_deg']) + ')'
        lines.append(
            f"| {case['display_name']} | {att0} | {case['n_motion_rows']} | {case['claimed_position_count']} | {case['total_time_s']:.3f} | {case['timing_note']} |"
        )
    return lines


def render_overall_table(comparisons: dict[str, Any], runs: dict[str, Any]) -> list[str]:
    lines = []
    lines.append('| estimator | 20-pos mean/med/max | 19-pos mean/med/max | Δmean | Δmedian | Δmax | verdict |')
    lines.append('|---|---|---|---:|---:|---:|---|')
    for method_key in METHOD_KEYS:
        sym = runs['symmetric20'][method_key]['overall']
        legacy = runs['legacy19pos'][method_key]['overall']
        delta = comparisons[method_key]['overall']['metrics']
        lines.append(
            f"| {METHOD_LABELS[method_key]} | {triplet_text(sym)} | {triplet_text(legacy)} | "
            f"{delta['mean_pct_error']['improvement_pct_points']:+.3f} | "
            f"{delta['median_pct_error']['improvement_pct_points']:+.3f} | "
            f"{delta['max_pct_error']['improvement_pct_points']:+.3f} | {comparisons[method_key]['overall']['verdict']} |"
        )
    return lines


def render_param_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = []
    lines.append('| param | true | 20-pos est | 20-pos err% | 19-pos est | 19-pos err% | Δerr%(19-20) | better |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---|')
    for row in rows:
        lines.append(
            f"| {row['param']} | {fmt_e(row['true'])} | {fmt_e(row['symmetric20_est'])} | {fmt_pct(row['symmetric20_pct_error'])} | "
            f"{fmt_e(row['legacy19pos_est'])} | {fmt_pct(row['legacy19pos_pct_error'])} | {row['improvement_pct_points']:+.6f} | {row['winner']} |"
        )
    return lines


def render_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append('# Corrected symmetric 20-step vs legacy 19-position formal comparison')
    lines.append('')
    lines.append('## 1. Benchmark setup')
    lines.append('')
    lines.append(f"- noise_scale = **{summary['noise_scale']}**")
    lines.append('- same truth model = `get_default_clbt()` from `test_calibration_markov_pruned.py`')
    lines.append('- same noise family / seed = `expected_noise_config(noise_scale)` with seed 42')
    lines.append('- fairness rule = both paths normalized to **1200 s** total duration')
    lines.append('- important exception retained exactly as requested = each path keeps **its own correct att0**')
    lines.append('')
    lines.extend(render_case_table(summary['cases']))
    lines.append('')
    lines.append('## 2. Headline result')
    lines.append('')
    lines.extend(render_overall_table(summary['comparisons'], summary['runs']))
    lines.append('')
    lines.append(f"- **Consensus:** {summary['verdict']['consensus']}" )
    for method_key in METHOD_KEYS:
        lines.append(f"- {summary['verdict']['by_method'][method_key]}")
    lines.append('')
    lines.append('## 3. Path-specific notes')
    lines.append('')
    lines.append(f"- **20-pos corrected symmetric scheme** source/builder: `{summary['cases']['symmetric20']['source_builder']}`")
    lines.append(f"- **Legacy 19-pos** source: `{summary['cases']['legacy19pos']['source_builder']}`")
    lines.append(f"- Legacy timing scale factor = **{summary['cases']['legacy19pos']['timing_scale_factor']:.9f}** (982 s → 1200 s)")
    lines.append('- Legacy script naming note: the historical method is called “19-position”, while the paras block itself contains 18 motion rows plus the starting attitude/state.')
    lines.append('')
    for method_key in METHOD_KEYS:
        comp = summary['comparisons'][method_key]
        lines.append(f"## 4. Full parameter comparison — {METHOD_LABELS[method_key]}")
        lines.append('')
        lines.append(
            f"- parameter wins: 20-pos **{comp['params']['symmetric20_better_count']}**, "
            f"legacy19 **{comp['params']['legacy19pos_better_count']}**, ties **{comp['params']['tie_count']}**"
        )
        lines.append('')
        lines.extend(render_param_table(comp['params']['rows']))
        lines.append('')
    lines.append('## 5. Output files')
    lines.append('')
    for key, value in summary['files'].items():
        if isinstance(value, dict):
            for subk, subv in value.items():
                if isinstance(subv, dict):
                    for subsubk, subsubv in subv.items():
                        lines.append(f"- {key}.{subk}.{subsubk}: `{subsubv}`")
                else:
                    lines.append(f"- {key}.{subk}: `{subv}`")
        else:
            lines.append(f"- {key}: `{value}`")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module(f'compare_20vs19_source_{make_suffix(args.noise_scale)}', str(SOURCE_FILE))
    cases = {
        'symmetric20': build_symmetric20_case(mod),
        'legacy19pos': build_legacy19pos_case(mod),
    }

    runs: dict[str, Any] = {case_key: {} for case_key in CASE_KEYS}
    execution: dict[str, Any] = {case_key: {} for case_key in CASE_KEYS}
    run_jsons: dict[str, Any] = {case_key: {} for case_key in CASE_KEYS}

    for case_key in CASE_KEYS:
        for method_key in METHOD_KEYS:
            payload, status, path = run_case_method(mod, cases[case_key], method_key, args.noise_scale, force_rerun=args.force_rerun)
            runs[case_key][method_key] = compact_payload(payload)
            execution[case_key][method_key] = status
            run_jsons[case_key][method_key] = str(path)

    full_runs = {
        case_key: {
            method_key: _load_json(Path(run_jsons[case_key][method_key]))
            for method_key in METHOD_KEYS
        }
        for case_key in CASE_KEYS
    }

    comparisons = {}
    for method_key in METHOD_KEYS:
        sym_payload = full_runs['symmetric20'][method_key]
        legacy_payload = full_runs['legacy19pos'][method_key]
        comparisons[method_key] = {
            'overall': overall_delta(sym_payload, legacy_payload),
            'params': compare_params(sym_payload, legacy_payload),
        }

    verdict = {
        'by_method': {
            method_key: method_verdict_text(method_key, comparisons[method_key]['overall'], comparisons[method_key]['params'])
            for method_key in METHOD_KEYS
        },
        'consensus': consensus_text(comparisons),
    }

    suffix = make_suffix(args.noise_scale)
    summary_json = RESULTS_DIR / f'compare_corrected_symmetric20_vs_legacy19pos_{suffix}_1200s.json'
    report_md = REPORTS_DIR / f'psins_corrected_symmetric20_vs_legacy19pos_{args.report_date}_{suffix}_1200s.md'

    summary = {
        'experiment': 'corrected_symmetric20_vs_legacy19pos_formal_comparison',
        'comparison_mode': COMPARISON_MODE,
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'noise_config': expected_noise_config(args.noise_scale),
        'source_file': str(SOURCE_FILE),
        'legacy_source_file': str(LEGACY_19POS_FILE),
        'symmetric20_source_file': str(SYMMETRIC20_PROBE_FILE),
        'cases': {
            case_key: {
                key: value
                for key, value in cases[case_key].items()
                if key not in {'paras'}
            }
            for case_key in CASE_KEYS
        },
        'runs': runs,
        'comparisons': comparisons,
        'verdict': verdict,
        'execution': execution,
        'files': {
            'summary_json': str(summary_json),
            'report_md': str(report_md),
            'run_jsons': run_jsons,
        },
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_md.write_text(render_report(summary), encoding='utf-8')

    result = {
        'summary_json': str(summary_json),
        'report_md': str(report_md),
        'run_jsons': run_jsons,
        'execution': execution,
        'markov42': {
            'symmetric20': runs['symmetric20']['markov42_noisy']['overall'],
            'legacy19pos': runs['legacy19pos']['markov42_noisy']['overall'],
            'verdict': comparisons['markov42_noisy']['overall']['verdict'],
        },
        'kf36': {
            'symmetric20': runs['symmetric20']['kf36_noisy']['overall'],
            'legacy19pos': runs['legacy19pos']['kf36_noisy']['overall'],
            'verdict': comparisons['kf36_noisy']['overall']['verdict'],
        },
        'consensus': verdict['consensus'],
    }
    print('__RESULT_JSON__=' + json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
