from __future__ import annotations

import argparse
import copy
import json
import sys
import types
from datetime import datetime
from pathlib import Path
from statistics import median
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
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'
R61_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round61_h_scd_state20_microtight_commit.py'

COMPARE_SHARED_FILE = SCRIPTS_DIR / 'compare_four_methods_shared_noise.py'
COMPARE_CH3_FILE = SCRIPTS_DIR / 'compare_ch3_corrected_symmetric20_vs_legacy19pos_1200s.py'
COMPUTE_R61_FILE = SCRIPTS_DIR / 'compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors.py'
PROBE_R55_FILE = SCRIPTS_DIR / 'probe_round55_newline.py'
PROBE_R59_FILE = SCRIPTS_DIR / 'probe_round59_h_scd_hybrid.py'

for p in [ROOT, TMP_PSINS_DIR, METHOD_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module

METHOD_ORDER = ['kf36', 'markov42', 'pure_scd_neutral', 'round61']
METHOD_DISPLAY = {
    'kf36': 'KF36 baseline',
    'markov42': 'Markov42 baseline',
    'pure_scd_neutral': 'Pure SCD neutral baseline',
    'round61': 'Round61 transferred method',
}

CONDITION_ORDER = ['shared0p08', 'sym20_0p08', 'sym20_0p12']
CONDITION_DISPLAY = {
    'shared0p08': 'A. original shared path @ noise0p08',
    'sym20_0p08': 'B. corrected symmetric20 @ noise0p08',
    'sym20_0p12': 'C. corrected symmetric20 @ noise0p12',
}

TRANSITION_ORDER = [
    ('shared0p08', 'sym20_0p08', 'shared0p08_to_sym20_0p08'),
    ('sym20_0p08', 'sym20_0p12', 'sym20_0p08_to_sym20_0p12'),
]

METRICS = ['mean_pct_error', 'median_pct_error', 'max_pct_error']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _overall(payload: dict[str, Any]) -> dict[str, float]:
    ov = payload['overall']
    return {
        'mean_pct_error': float(ov['mean_pct_error']),
        'median_pct_error': float(ov['median_pct_error']),
        'max_pct_error': float(ov['max_pct_error']),
    }


def _fmt_pct(v: float) -> str:
    return f'{v:.6f}'


def _round61_vs_pure_block(cond_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for ck in CONDITION_ORDER:
        pure = cond_payloads[ck]['pure_scd_neutral']['overall']
        r61 = cond_payloads[ck]['round61']['overall']
        d = {m: float(r61[m] - pure[m]) for m in METRICS}
        out[ck] = {
            'round61_minus_pure_scd': d,
            'round61_better_on_metric': {m: bool(d[m] < 0.0) for m in METRICS},
        }
    return out


def _build_rankings(cond_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ranking_out: dict[str, Any] = {}
    for ck in CONDITION_ORDER:
        ranking_out[ck] = {
            'by_metric': {},
            'best_by_metric': {},
            'round61_rank': {},
        }
        for metric in METRICS:
            rows = sorted(
                [
                    {
                        'method': mk,
                        'value': float(cond_payloads[ck][mk]['overall'][metric]),
                    }
                    for mk in METHOD_ORDER
                ],
                key=lambda x: x['value'],
            )
            ranking_out[ck]['by_metric'][metric] = rows
            ranking_out[ck]['best_by_metric'][metric] = rows[0]
            ranking_out[ck]['round61_rank'][metric] = [x['method'] for x in rows].index('round61') + 1
    return ranking_out


def _build_transitions(cond_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for from_ck, to_ck, name in TRANSITION_ORDER:
        per_method = {}
        for mk in METHOD_ORDER:
            from_ov = cond_payloads[from_ck][mk]['overall']
            to_ov = cond_payloads[to_ck][mk]['overall']
            metric_delta = {}
            for metric in METRICS:
                fv = float(from_ov[metric])
                tv = float(to_ov[metric])
                delta = tv - fv
                metric_delta[metric] = {
                    'from_value': fv,
                    'to_value': tv,
                    'delta_pct_points': delta,
                    'relative_change_pct': (delta / fv * 100.0) if abs(fv) > 1e-15 else None,
                    'worse_if_positive': True,
                }
            per_method[mk] = metric_delta

        aggregate = {}
        for metric in METRICS:
            vals = [float(per_method[mk][metric]['delta_pct_points']) for mk in METHOD_ORDER]
            aggregate[metric] = {
                'mean_delta_pct_points': float(sum(vals) / len(vals)),
                'median_delta_pct_points': float(median(vals)),
                'max_delta_pct_points': float(max(vals)),
                'min_delta_pct_points': float(min(vals)),
            }

        out[name] = {
            'from_condition': from_ck,
            'to_condition': to_ck,
            'per_method': per_method,
            'aggregate': aggregate,
        }
    return out


def _build_round61_gap_effect(cond_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Gap definition per metric/condition:
      gap = round61_overall - best_other_method_overall
    Lower is better, so:
      gap < 0 => Round61 is better than all other 3 methods on that metric.
      gap > 0 => Round61 is behind the best non-Round61 method.
    """
    per_condition: dict[str, Any] = {}
    for ck in CONDITION_ORDER:
        metric_block: dict[str, Any] = {}
        for metric in METRICS:
            r61_val = float(cond_payloads[ck]['round61']['overall'][metric])
            others = [
                {
                    'method': mk,
                    'value': float(cond_payloads[ck][mk]['overall'][metric]),
                }
                for mk in METHOD_ORDER
                if mk != 'round61'
            ]
            best_other = min(others, key=lambda x: x['value'])
            gap = r61_val - float(best_other['value'])
            metric_block[metric] = {
                'round61_value': r61_val,
                'best_other_method': best_other['method'],
                'best_other_value': float(best_other['value']),
                'gap_round61_minus_best_other': gap,
                'round61_is_best': bool(gap < 0.0),
            }
        per_condition[ck] = metric_block

    transition_gap: dict[str, Any] = {}
    for from_ck, to_ck, tname in TRANSITION_ORDER:
        metric_delta: dict[str, Any] = {}
        for metric in METRICS:
            g_from = float(per_condition[from_ck][metric]['gap_round61_minus_best_other'])
            g_to = float(per_condition[to_ck][metric]['gap_round61_minus_best_other'])
            metric_delta[metric] = {
                'from_gap': g_from,
                'to_gap': g_to,
                'gap_delta': g_to - g_from,
                'worse_if_positive': True,
            }
        transition_gap[tname] = metric_delta

    return {
        'per_condition': per_condition,
        'transition_gap': transition_gap,
    }


def _causal_verdict(round61_gap: dict[str, Any]) -> dict[str, Any]:
    path_gap = round61_gap['transition_gap']['shared0p08_to_sym20_0p08']
    noise_gap = round61_gap['transition_gap']['sym20_0p08_to_sym20_0p12']

    # Interpretation target is disappearance of the old Round61 "good effect".
    # So use Round61-vs-best-other gap, and prioritize mean/median.
    path_strength = float((path_gap['mean_pct_error']['gap_delta'] + path_gap['median_pct_error']['gap_delta']) / 2.0)
    noise_strength = float((noise_gap['mean_pct_error']['gap_delta'] + noise_gap['median_pct_error']['gap_delta']) / 2.0)

    eps = 1e-12
    if path_strength > 0 and noise_strength > 0:
        ratio = path_strength / max(noise_strength, eps)
        if ratio >= 1.5:
            label = 'path'
            reason = 'Round61 advantage loss from A→B is clearly larger than the additional loss from B→C.'
        elif ratio <= (1.0 / 1.5):
            label = 'noise'
            reason = 'Additional Round61 loss from B→C is clearly larger than the A→B path shift.'
        else:
            label = 'both'
            reason = 'A→B and B→C both contribute similarly to Round61 advantage loss.'
    elif path_strength > 0 and noise_strength <= 0:
        label = 'path'
        reason = 'Round61 advantage mainly disappears already at A→B; B→C does not further worsen mean/median.'
    elif noise_strength > 0 and path_strength <= 0:
        label = 'noise'
        reason = 'Round61 advantage is stable across A→B but worsens at B→C.'
    else:
        label = 'both'
        reason = 'Round61 gap trend is mixed; no single leg dominates across mean/median.'

    return {
        'label': label,
        'path_strength_gap_delta_avg_mean_median': path_strength,
        'noise_strength_gap_delta_avg_mean_median': noise_strength,
        'path_gap_delta': path_gap,
        'noise_gap_delta': noise_gap,
        'reason': reason,
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append('# PSINS causal decomposition: shared0p08 → sym20(0p08, 0p12)')
    lines.append('')
    lines.append('## 1) Fixed methods and conditions')
    lines.append('')
    lines.append('- Methods fixed across all conditions: **KF36 / Markov42 / Pure SCD neutral / Round61 transferred**')
    lines.append('- Metrics: full-parameter overall **mean / median / max % error** (same style as old Feishu doc)')
    lines.append('- Conditions:')
    lines.append('  - A: original shared dataset @ noise0p08')
    lines.append('  - B: corrected symmetric20 path @ noise0p08, att0=(0,0,0)')
    lines.append('  - C: corrected symmetric20 path @ noise0p12, att0=(0,0,0)')
    lines.append('')

    lines.append('## 2) 4-method × 3-condition overall matrix')
    lines.append('')
    lines.append('| condition | KF36 (mean/med/max) | Markov42 (mean/med/max) | Pure SCD neutral (mean/med/max) | Round61 (mean/med/max) | best by mean | Round61 rank (mean/med/max) |')
    lines.append('|---|---|---|---|---|---|---|')

    for ck in CONDITION_ORDER:
        row = []
        for mk in METHOD_ORDER:
            ov = summary['matrix'][ck][mk]['overall']
            row.append(f"{ov['mean_pct_error']:.6f}/{ov['median_pct_error']:.6f}/{ov['max_pct_error']:.6f}")
        best_mean = METHOD_DISPLAY[summary['rankings'][ck]['best_by_metric']['mean_pct_error']['method']]
        rr = summary['rankings'][ck]['round61_rank']
        lines.append(
            f"| {CONDITION_DISPLAY[ck]} | {row[0]} | {row[1]} | {row[2]} | {row[3]} | {best_mean} | {rr['mean_pct_error']}/{rr['median_pct_error']}/{rr['max_pct_error']} |"
        )
    lines.append('')

    lines.append('## 3) Transition deltas (to - from, + means worse)')
    lines.append('')
    for _, _, tname in TRANSITION_ORDER:
        blk = summary['transitions'][tname]
        lines.append(f"### {tname}")
        lines.append('')
        lines.append('| method | Δmean | Δmedian | Δmax |')
        lines.append('|---|---:|---:|---:|')
        for mk in METHOD_ORDER:
            d = blk['per_method'][mk]
            lines.append(
                f"| {METHOD_DISPLAY[mk]} | {d['mean_pct_error']['delta_pct_points']:+.6f} | {d['median_pct_error']['delta_pct_points']:+.6f} | {d['max_pct_error']['delta_pct_points']:+.6f} |"
            )
        agg = blk['aggregate']
        lines.append(
            f"| **Aggregate median across methods** | **{agg['mean_pct_error']['median_delta_pct_points']:+.6f}** | **{agg['median_pct_error']['median_delta_pct_points']:+.6f}** | **{agg['max_pct_error']['median_delta_pct_points']:+.6f}** |"
        )
        lines.append('')

    lines.append('## 4) Round61 vs Pure SCD neutral (same condition)')
    lines.append('')
    lines.append('| condition | Δmean (R61-Pure) | Δmedian | Δmax | quick read |')
    lines.append('|---|---:|---:|---:|---|')
    for ck in CONDITION_ORDER:
        d = summary['round61_vs_pure'][ck]['round61_minus_pure_scd']
        better_count = sum(1 for m in METRICS if d[m] < 0.0)
        quick = 'Round61 better' if better_count >= 2 else ('Pure SCD better' if better_count <= 1 else 'mixed')
        lines.append(
            f"| {CONDITION_DISPLAY[ck]} | {d['mean_pct_error']:+.6f} | {d['median_pct_error']:+.6f} | {d['max_pct_error']:+.6f} | {quick} |"
        )
    lines.append('')

    lines.append('## 5) Causal verdict (target = Round61 old-good-effect disappearance)')
    lines.append('')
    cv = summary['causal_verdict']
    lines.append(f"- verdict: **{cv['label']}**")
    lines.append(f"- path-strength (A→B, Round61 gap delta avg over mean+median): **{cv['path_strength_gap_delta_avg_mean_median']:+.6f}**")
    lines.append(f"- noise-strength (B→C, Round61 gap delta avg over mean+median): **{cv['noise_strength_gap_delta_avg_mean_median']:+.6f}**")
    lines.append('- detailed Round61 gap deltas (to-from, + means Round61 falls further behind best non-Round61):')
    lines.append(
        f"  - A→B: Δgap_mean={cv['path_gap_delta']['mean_pct_error']['gap_delta']:+.6f}, "
        f"Δgap_median={cv['path_gap_delta']['median_pct_error']['gap_delta']:+.6f}, "
        f"Δgap_max={cv['path_gap_delta']['max_pct_error']['gap_delta']:+.6f}"
    )
    lines.append(
        f"  - B→C: Δgap_mean={cv['noise_gap_delta']['mean_pct_error']['gap_delta']:+.6f}, "
        f"Δgap_median={cv['noise_gap_delta']['median_pct_error']['gap_delta']:+.6f}, "
        f"Δgap_max={cv['noise_gap_delta']['max_pct_error']['gap_delta']:+.6f}"
    )
    lines.append(f"- reason: {cv['reason']}")
    lines.append('')

    lines.append('## 6) Artifact files')
    lines.append('')
    lines.append(f"- summary_json: `{summary['files']['summary_json']}`")
    lines.append(f"- report_md: `{summary['files']['report_md']}`")
    for ck in CONDITION_ORDER:
        lines.append(f"- condition `{ck}`:")
        for mk in METHOD_ORDER:
            lines.append(f"  - {mk}: `{summary['condition_result_jsons'][ck][mk]}`")
    lines.append('')
    return '\n'.join(lines) + '\n'


def _condition_noise_scale(condition_key: str) -> float:
    if condition_key == 'shared0p08':
        return 0.08
    if condition_key == 'sym20_0p08':
        return 0.08
    if condition_key == 'sym20_0p12':
        return 0.12
    raise KeyError(condition_key)


def _ensure_shared_method(
    method_key: str,
    noise_scale: float,
    force_rerun: bool,
    source_mod,
    shared_mod,
    compute_r61_mod,
    probe_r55_mod,
    probe_r59_mod,
) -> tuple[dict[str, Any], str, Path]:
    _, out = shared_mod.build_output_paths(noise_scale, report_date='2026-04-05')
    path_map = {
        'kf36': out['kf36_noisy'],
        'markov42': out['markov42_noisy'],
        'pure_scd_neutral': out['scd42_neutral'],
        'round61': out['round61'],
    }
    path = path_map[method_key]
    expected_cfg = shared_mod.expected_noise_config(noise_scale)

    if (not force_rerun) and path.exists():
        payload = shared_mod._load_json(path)
        if shared_mod._noise_matches(payload, expected_cfg):
            return payload, 'reused_verified', path

    dataset = shared_mod.build_shared_dataset(source_mod, noise_scale)
    params = compute_r61_mod._param_specs(source_mod)

    if method_key == 'kf36':
        res = source_mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=36,
            label=f'KF36-SHARED-{shared_mod.make_suffix(noise_scale).upper()}',
        )
        payload = shared_mod.compute_payload(
            source_mod,
            res[0],
            params,
            variant=f'kf36_shared_{shared_mod.make_suffix(noise_scale)}',
            method_file='source_mod.run_calibration(n_states=36)',
            extra={
                'noise_scale': noise_scale,
                'noise_config': dataset['noise_config'],
                'comparison_mode': 'shared_dataset_apples_to_apples',
                'mainline_rung': 'kf36_noisy',
            },
        )
    elif method_key == 'markov42':
        res = source_mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=42,
            bi_g=dataset['bi_g'],
            tau_g=dataset['tau_g'],
            bi_a=dataset['bi_a'],
            tau_a=dataset['tau_a'],
            label=f'MARKOV42-SHARED-{shared_mod.make_suffix(noise_scale).upper()}',
        )
        payload = shared_mod.compute_payload(
            source_mod,
            res[0],
            params,
            variant=f'42state_gm1_shared_{shared_mod.make_suffix(noise_scale)}',
            method_file='source_mod.run_calibration(n_states=42)',
            extra={
                'noise_scale': noise_scale,
                'noise_config': dataset['noise_config'],
                'comparison_mode': 'shared_dataset_apples_to_apples',
                'mainline_rung': 'markov42_noisy',
            },
        )
    elif method_key == 'pure_scd_neutral':
        neutral = shared_mod._build_neutral_scd_candidate()
        method_mod = load_module(
            f'causal_shared_r53_scd_{shared_mod.make_suffix(noise_scale)}',
            str(R53_METHOD_FILE),
        )
        method_mod = probe_r55_mod._build_patched_method(method_mod, neutral)
        res = list(probe_r59_mod._run_internalized_hybrid_scd(
            method_mod,
            source_mod,
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            bi_g=dataset['bi_g'],
            bi_a=dataset['bi_a'],
            tau_g=dataset['tau_g'],
            tau_a=dataset['tau_a'],
            label=f'SCD42-NEUTRAL-SHARED-{shared_mod.make_suffix(noise_scale).upper()}',
            scd_cfg=neutral['scd'],
        ))
        runtime = res[4] if len(res) >= 5 and isinstance(res[4], dict) else {}
        payload = shared_mod.compute_payload(
            source_mod,
            res[0],
            params,
            variant=f'42state_gm1_scdneutral_shared_{shared_mod.make_suffix(noise_scale)}',
            method_file='neutral_markov42_plus_once_scd_on_shared_dataset',
            extra={
                'noise_scale': noise_scale,
                'noise_config': dataset['noise_config'],
                'comparison_mode': 'shared_dataset_apples_to_apples',
                'mainline_rung': 'scd42_neutral',
                'selected_candidate': neutral['name'],
                'candidate_description': neutral['description'],
                'scd_cfg': copy.deepcopy(neutral['scd']),
                'iter_patches': copy.deepcopy(neutral['iter_patches']),
                'runtime_log': {
                    'schedule_log': runtime.get('schedule_log'),
                    'feedback_log': runtime.get('feedback_log'),
                    'scd_log': runtime.get('scd_log'),
                },
            },
        )
    elif method_key == 'round61':
        candidate = shared_mod._pick_round61_candidate()
        merged = shared_mod._merge_round61_candidate(candidate)
        method_mod = load_module(
            f'causal_shared_r53_r61_{shared_mod.make_suffix(noise_scale)}',
            str(R53_METHOD_FILE),
        )
        method_mod = probe_r55_mod._build_patched_method(method_mod, merged)
        r61_mod = load_module(
            f'causal_shared_r61_method_{shared_mod.make_suffix(noise_scale)}',
            str(R61_METHOD_FILE),
        )
        res = list(r61_mod._run_internalized_hybrid_scd(
            method_mod,
            source_mod,
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            bi_g=dataset['bi_g'],
            bi_a=dataset['bi_a'],
            tau_g=dataset['tau_g'],
            tau_a=dataset['tau_a'],
            label=f'R61-SHARED-{shared_mod.make_suffix(noise_scale).upper()}',
            scd_cfg=merged['scd'],
        ))
        extra = res[4] if len(res) >= 5 and isinstance(res[4], dict) else {}
        extra = dict(extra)
        extra.update({
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'comparison_mode': 'shared_dataset_apples_to_apples',
            'round61_selected_candidate': candidate['name'],
        })
        payload = shared_mod.compute_payload(
            source_mod,
            res[0],
            params,
            variant=f'42state_gm1_round61_h_scd_state20_microtight_commit_shared_{shared_mod.make_suffix(noise_scale)}',
            method_file=str(R61_METHOD_FILE),
            extra=extra,
        )
    else:
        raise KeyError(method_key)

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', path


def _ensure_sym20_method(
    condition_key: str,
    method_key: str,
    force_rerun: bool,
    source_mod,
    shared_mod,
    ch3_mod,
    compute_r61_mod,
    probe_r55_mod,
    probe_r59_mod,
) -> tuple[dict[str, Any], str, Path]:
    noise_scale = _condition_noise_scale(condition_key)
    suffix = shared_mod.make_suffix(noise_scale)
    case = ch3_mod.build_symmetric20_case(source_mod)
    expected_cfg = shared_mod.expected_noise_config(noise_scale)

    if method_key in {'kf36', 'markov42'}:
        payload, status, path = ch3_mod.run_case_method(
            source_mod,
            case,
            'kf36_noisy' if method_key == 'kf36' else 'markov42_noisy',
            noise_scale,
            force_rerun=force_rerun,
        )
        return payload, status, path

    if method_key == 'pure_scd_neutral':
        path = RESULTS_DIR / f"G4_pure_scd_neutral_{case['case_tag']}_shared_{suffix}_param_errors.json"
        if (not force_rerun) and path.exists():
            payload = _load_json(path)
            extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
            if (
                shared_mod._noise_matches(payload, expected_cfg)
                and extra.get('method_family') == 'pure_neutral_scd_baseline'
                and extra.get('path_case_tag') == case['case_tag']
            ):
                return payload, 'reused_verified', path

        dataset = ch3_mod.build_dataset(source_mod, case['paras'], case['att0_deg'], noise_scale)
        params = compute_r61_mod._param_specs(source_mod)
        neutral = shared_mod._build_neutral_scd_candidate()
        method_mod = load_module(
            f'causal_sym20_r53_scd_{suffix}',
            str(R53_METHOD_FILE),
        )
        method_mod = probe_r55_mod._build_patched_method(method_mod, neutral)
        res = list(probe_r59_mod._run_internalized_hybrid_scd(
            method_mod,
            source_mod,
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            bi_g=dataset['bi_g'],
            bi_a=dataset['bi_a'],
            tau_g=dataset['tau_g'],
            tau_a=dataset['tau_a'],
            label=f'PURE-SCD-SYM20-{suffix.upper()}',
            scd_cfg=neutral['scd'],
        ))
        runtime = res[4] if len(res) >= 5 and isinstance(res[4], dict) else {}

        payload = shared_mod.compute_payload(
            source_mod,
            res[0],
            params,
            variant=f'pure_scd_neutral_{case["case_tag"]}_{suffix}',
            method_file='round53_base + _build_patched_method(neutral) + _run_internalized_hybrid_scd',
            extra={
                'comparison_mode': 'causal_decompose_shared0p08_to_sym20_2026_04_05',
                'method_family': 'pure_neutral_scd_baseline',
                'path_case_key': case['case_key'],
                'path_case_tag': case['case_tag'],
                'path_case_display_name': case['display_name'],
                'att0_deg': case['att0_deg'],
                'n_motion_rows': case['n_motion_rows'],
                'claimed_position_count': case['claimed_position_count'],
                'total_time_s': case['total_time_s'],
                'timing_note': case['timing_note'],
                'source_builder': case['source_builder'],
                'source_reference': case['source_reference'],
                'builder_method_tag': case.get('builder_method_tag'),
                'rationale': case['rationale'],
                'noise_scale': noise_scale,
                'noise_config': dataset['noise_config'],
                'candidate_name': neutral['name'],
                'candidate_description': neutral['description'],
                'iter_patches': copy.deepcopy(neutral['iter_patches']),
                'scd_cfg': copy.deepcopy(neutral['scd']),
                'runtime_log': {
                    'schedule_log': runtime.get('schedule_log'),
                    'feedback_log': runtime.get('feedback_log'),
                    'scd_log': runtime.get('scd_log'),
                },
            },
        )
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return payload, 'rerun', path

    if method_key == 'round61':
        path = RESULTS_DIR / f'R61_42state_gm1_round61_h_scd_state20_microtight_commit_{case["case_tag"]}_shared_{suffix}_param_errors.json'
        if (not force_rerun) and path.exists():
            payload = _load_json(path)
            extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
            if (
                shared_mod._noise_matches(payload, expected_cfg)
                and extra.get('round61_selected_candidate') == 'r61_s20_08988_ryz00116'
                and (
                    extra.get('case_key') == 'symmetric20'
                    or extra.get('path_case_key') == 'symmetric20'
                    or extra.get('case_tag') == case['case_tag']
                )
            ):
                return payload, 'reused_verified', path

        dataset = ch3_mod.build_dataset(source_mod, case['paras'], case['att0_deg'], noise_scale)
        params = compute_r61_mod._param_specs(source_mod)

        candidate = shared_mod._pick_round61_candidate()
        merged = shared_mod._merge_round61_candidate(candidate)
        method_mod = load_module(
            f'causal_sym20_r53_r61_{suffix}',
            str(R53_METHOD_FILE),
        )
        method_mod = probe_r55_mod._build_patched_method(method_mod, merged)
        r61_mod = load_module(
            f'causal_sym20_r61_method_{suffix}',
            str(R61_METHOD_FILE),
        )
        res = list(r61_mod._run_internalized_hybrid_scd(
            method_mod,
            source_mod,
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            bi_g=dataset['bi_g'],
            bi_a=dataset['bi_a'],
            tau_g=dataset['tau_g'],
            tau_a=dataset['tau_a'],
            label=f'R61-SYM20-{suffix.upper()}',
            scd_cfg=merged['scd'],
        ))
        extra = res[4] if len(res) >= 5 and isinstance(res[4], dict) else {}
        extra = dict(extra)
        extra.update({
            'comparison_mode': 'causal_decompose_shared0p08_to_sym20_2026_04_05',
            'case_key': 'symmetric20',
            'case_tag': case['case_tag'],
            'case_display_name': case['display_name'],
            'att0_deg': case['att0_deg'],
            'n_motion_rows': case['n_motion_rows'],
            'claimed_position_count': case['claimed_position_count'],
            'total_time_s': case['total_time_s'],
            'timing_note': case['timing_note'],
            'source_builder': case['source_builder'],
            'source_reference': case['source_reference'],
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'round61_selected_candidate': candidate['name'],
            'round61_candidate_description': candidate['description'],
            'round61_candidate_rationale': candidate['rationale'],
            'transfer_note': 'Round61 transferred to corrected symmetric20 without retune.',
        })

        payload = shared_mod.compute_payload(
            source_mod,
            res[0],
            params,
            variant=f'42state_gm1_round61_h_scd_state20_microtight_commit_{case["case_tag"]}_{suffix}',
            method_file=f'{R61_METHOD_FILE} on corrected symmetric20 path',
            extra=extra,
        )
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        return payload, 'rerun', path

    raise KeyError(method_key)


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    shared_mod = load_module('causal_shared_mod_20260405', str(COMPARE_SHARED_FILE))
    ch3_mod = load_module('causal_ch3_mod_20260405', str(COMPARE_CH3_FILE))
    compute_r61_mod = load_module('causal_compute_r61_mod_20260405', str(COMPUTE_R61_FILE))
    probe_r55_mod = load_module('causal_probe_r55_mod_20260405', str(PROBE_R55_FILE))
    probe_r59_mod = load_module('causal_probe_r59_mod_20260405', str(PROBE_R59_FILE))

    source_mod = load_module('causal_source_mod_20260405', str(SOURCE_FILE))

    payloads: dict[str, dict[str, dict[str, Any]]] = {ck: {} for ck in CONDITION_ORDER}
    execution: dict[str, dict[str, str]] = {ck: {} for ck in CONDITION_ORDER}
    json_paths: dict[str, dict[str, str]] = {ck: {} for ck in CONDITION_ORDER}

    # Condition A (shared0p08)
    for mk in METHOD_ORDER:
        p, s, path = _ensure_shared_method(
            mk,
            noise_scale=0.08,
            force_rerun=args.force_rerun,
            source_mod=source_mod,
            shared_mod=shared_mod,
            compute_r61_mod=compute_r61_mod,
            probe_r55_mod=probe_r55_mod,
            probe_r59_mod=probe_r59_mod,
        )
        payloads['shared0p08'][mk] = p
        execution['shared0p08'][mk] = s
        json_paths['shared0p08'][mk] = str(path)

    # Conditions B/C (corrected symmetric20)
    for ck in ['sym20_0p08', 'sym20_0p12']:
        for mk in METHOD_ORDER:
            p, s, path = _ensure_sym20_method(
                ck,
                mk,
                force_rerun=args.force_rerun,
                source_mod=source_mod,
                shared_mod=shared_mod,
                ch3_mod=ch3_mod,
                compute_r61_mod=compute_r61_mod,
                probe_r55_mod=probe_r55_mod,
                probe_r59_mod=probe_r59_mod,
            )
            payloads[ck][mk] = p
            execution[ck][mk] = s
            json_paths[ck][mk] = str(path)

    matrix = {
        ck: {
            mk: {
                'overall': _overall(payloads[ck][mk]),
            }
            for mk in METHOD_ORDER
        }
        for ck in CONDITION_ORDER
    }

    rankings = _build_rankings(payloads)
    transitions = _build_transitions(payloads)
    round61_vs_pure = _round61_vs_pure_block(payloads)
    round61_gap_effect = _build_round61_gap_effect(payloads)
    causal = _causal_verdict(round61_gap_effect)

    summary_json = RESULTS_DIR / 'causal_decompose_shared0p08_to_sym20_2026-04-05_summary.json'
    report_md = REPORTS_DIR / f'psins_causal_decompose_shared0p08_to_sym20_{args.report_date}.md'

    summary = {
        'experiment': 'causal_decompose_shared0p08_to_sym20_2026_04_05',
        'comparison_goal': 'separate path effect (A→B) from noise effect (B→C) using fixed 4 methods',
        'methods_order': METHOD_ORDER,
        'conditions_order': CONDITION_ORDER,
        'method_display': METHOD_DISPLAY,
        'condition_display': CONDITION_DISPLAY,
        'matrix': matrix,
        'rankings': rankings,
        'transitions': transitions,
        'round61_vs_pure': round61_vs_pure,
        'round61_gap_effect': round61_gap_effect,
        'causal_verdict': causal,
        'condition_result_jsons': json_paths,
        'execution': execution,
        'minimal_fixes': 'none',
        'files': {
            'summary_json': str(summary_json),
            'report_md': str(report_md),
        },
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_md.write_text(_render_report(summary), encoding='utf-8')

    result = {
        'summary_json': str(summary_json),
        'report_md': str(report_md),
        'execution': execution,
        'causal_verdict': causal,
        'matrix': matrix,
        'round61_rank': {
            ck: rankings[ck]['round61_rank']
            for ck in CONDITION_ORDER
        },
        'condition_result_jsons': json_paths,
        'minimal_fixes': 'none',
    }
    print('__RESULT_JSON__=' + json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
