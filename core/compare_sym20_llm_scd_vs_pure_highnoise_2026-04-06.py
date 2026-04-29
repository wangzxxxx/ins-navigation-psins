from __future__ import annotations

import argparse
import copy
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
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'

COMPARE_CH3_FILE = SCRIPTS_DIR / 'compare_ch3_corrected_symmetric20_vs_legacy19pos_1200s.py'
COMPARE_SHARED_FILE = SCRIPTS_DIR / 'compare_four_methods_shared_noise.py'
COMPUTE_R61_FILE = SCRIPTS_DIR / 'compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors.py'
PROBE_R55_FILE = SCRIPTS_DIR / 'probe_round55_newline.py'
PROBE_R59_FILE = SCRIPTS_DIR / 'probe_round59_h_scd_hybrid.py'

for p in [ROOT, TMP_PSINS_DIR, METHOD_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module

EXPERIMENT_TAG = 'sym20_llm_scd_vs_pure_highnoise_2026_04_06'
DEFAULT_SCALES = [1.0, 2.0]

CANDIDATES = [
    {
        'name': 'candidate_A',
        'label': 'A',
        'description': 'selected scope, once_per_phase, alpha=0.999, transition_duration=2.0, bias_to_target=True',
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.999,
            'transition_duration': 2.0,
            'target': 'selected',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    },
    {
        'name': 'candidate_B',
        'label': 'B',
        'description': 'scale_block scope, once_per_phase, alpha=0.9995, transition_duration=2.0, bias_to_target=True',
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.9995,
            'transition_duration': 2.0,
            'target': 'scale_block',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    },
    {
        'name': 'candidate_C',
        'label': 'C',
        'description': 'scale_block scope, once_per_phase, alpha=0.999, transition_duration=4.0, bias_to_target=True',
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.999,
            'transition_duration': 4.0,
            'target': 'scale_block',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    },
    {
        'name': 'candidate_D',
        'label': 'D',
        'description': 'selected scope, repeat_after_transition, alpha=0.9998, transition_duration=2.0, bias_to_target=True',
        'scd': {
            'mode': 'repeat_after_transition',
            'alpha': 0.9998,
            'transition_duration': 2.0,
            'target': 'selected',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _candidate_signature(candidate: dict[str, Any]) -> str:
    return json.dumps({
        'name': candidate['name'],
        'label': candidate['label'],
        'description': candidate['description'],
        'scd': candidate['scd'],
    }, ensure_ascii=False, sort_keys=True)


def _overall_triplet(payload: dict[str, Any]) -> dict[str, float]:
    ov = payload['overall']
    return {
        'mean_pct_error': float(ov['mean_pct_error']),
        'median_pct_error': float(ov['median_pct_error']),
        'max_pct_error': float(ov['max_pct_error']),
    }


def _build_clean_candidate(compare_shared_mod, candidate: dict[str, Any]) -> dict[str, Any]:
    base = compare_shared_mod._build_neutral_scd_candidate()
    base['name'] = candidate['name']
    base['description'] = candidate['description']
    base['scd'] = copy.deepcopy(candidate['scd'])
    return base


def _pure_result_path(case_tag: str, suffix: str) -> Path:
    return RESULTS_DIR / f'G4_pure_scd_neutral_{case_tag}_shared_{suffix}_param_errors.json'


def _candidate_result_path(case_tag: str, candidate_name: str, suffix: str) -> Path:
    return RESULTS_DIR / f'SCD42_llm_clean_{candidate_name}_{case_tag}_shared_{suffix}_param_errors.json'


def _run_or_reuse_pure(
    *,
    compare_shared_mod,
    probe_r55_mod,
    probe_r59_mod,
    source_mod,
    params,
    case: dict[str, Any],
    dataset: dict[str, Any],
    noise_scale: float,
    suffix: str,
    force_rerun: bool,
) -> tuple[dict[str, Any], str, Path]:
    out = _pure_result_path(case['case_tag'], suffix)
    expected_cfg = compare_shared_mod.expected_noise_config(noise_scale)

    if (not force_rerun) and out.exists():
        payload = _load_json(out)
        extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
        if (
            compare_shared_mod._noise_matches(payload, expected_cfg)
            and extra.get('comparison_mode') == EXPERIMENT_TAG
            and extra.get('method_role') == 'pure_scd_baseline'
            and extra.get('path_case_tag') == case['case_tag']
        ):
            return payload, 'reused_verified', out

    neutral_candidate = compare_shared_mod._build_neutral_scd_candidate()
    method_mod = load_module(f'sym20_pure_r53_{suffix}', str(R53_METHOD_FILE))
    method_mod = probe_r55_mod._build_patched_method(method_mod, neutral_candidate)

    scd_result = list(probe_r59_mod._run_internalized_hybrid_scd(
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
        scd_cfg=neutral_candidate['scd'],
    ))

    runtime = scd_result[4] if len(scd_result) >= 5 and isinstance(scd_result[4], dict) else {}
    payload = compare_shared_mod.compute_payload(
        source_mod,
        scd_result[0],
        params,
        variant=f'pure_scd_neutral_{case["case_tag"]}_{suffix}',
        method_file='round53_base + _build_patched_method(neutral) + _run_internalized_hybrid_scd',
        extra={
            'comparison_mode': EXPERIMENT_TAG,
            'method_role': 'pure_scd_baseline',
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
            'candidate_name': neutral_candidate['name'],
            'candidate_description': neutral_candidate['description'],
            'iter_patches': copy.deepcopy(neutral_candidate['iter_patches']),
            'scd_cfg': copy.deepcopy(neutral_candidate['scd']),
            'runtime_log': {
                'schedule_log': runtime.get('schedule_log'),
                'feedback_log': runtime.get('feedback_log'),
                'scd_log': runtime.get('scd_log'),
            },
        },
    )
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', out


def _run_or_reuse_candidate(
    *,
    compare_shared_mod,
    probe_r55_mod,
    probe_r59_mod,
    source_mod,
    params,
    case: dict[str, Any],
    dataset: dict[str, Any],
    candidate: dict[str, Any],
    noise_scale: float,
    suffix: str,
    force_rerun: bool,
) -> tuple[dict[str, Any], str, Path]:
    out = _candidate_result_path(case['case_tag'], candidate['name'], suffix)
    expected_cfg = compare_shared_mod.expected_noise_config(noise_scale)
    candidate_sig = _candidate_signature(candidate)

    if (not force_rerun) and out.exists():
        payload = _load_json(out)
        extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
        if (
            compare_shared_mod._noise_matches(payload, expected_cfg)
            and extra.get('comparison_mode') == EXPERIMENT_TAG
            and extra.get('method_role') == 'llm_guided_scd'
            and extra.get('path_case_tag') == case['case_tag']
            and extra.get('candidate_signature') == candidate_sig
        ):
            return payload, 'reused_verified', out

    merged_candidate = _build_clean_candidate(compare_shared_mod, candidate)
    method_mod = load_module(f'sym20_llm_{candidate["name"]}_{suffix}', str(R53_METHOD_FILE))
    method_mod = probe_r55_mod._build_patched_method(method_mod, merged_candidate)

    scd_result = list(probe_r59_mod._run_internalized_hybrid_scd(
        method_mod,
        source_mod,
        dataset['imu_noisy'],
        dataset['pos0'],
        dataset['ts'],
        bi_g=dataset['bi_g'],
        bi_a=dataset['bi_a'],
        tau_g=dataset['tau_g'],
        tau_a=dataset['tau_a'],
        label=f'LLM-SCD-{candidate["label"]}-SYM20-{suffix.upper()}',
        scd_cfg=merged_candidate['scd'],
    ))

    runtime = scd_result[4] if len(scd_result) >= 5 and isinstance(scd_result[4], dict) else {}
    payload = compare_shared_mod.compute_payload(
        source_mod,
        scd_result[0],
        params,
        variant=f'llm_scd_clean_{candidate["name"]}_{case["case_tag"]}_{suffix}',
        method_file='probe_llm_scd_only_clean candidate family on corrected symmetric20 dataset',
        extra={
            'comparison_mode': EXPERIMENT_TAG,
            'method_role': 'llm_guided_scd',
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
            'candidate_name': candidate['name'],
            'candidate_label': candidate['label'],
            'candidate_description': candidate['description'],
            'candidate_signature': candidate_sig,
            'candidate_scd_cfg': copy.deepcopy(candidate['scd']),
            'runtime_log': {
                'schedule_log': runtime.get('schedule_log'),
                'feedback_log': runtime.get('feedback_log'),
                'scd_log': runtime.get('scd_log'),
            },
        },
    )
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', out


def _delta_vs_pure(pure: dict[str, float], llm: dict[str, float]) -> dict[str, Any]:
    metric_key_map = {
        'mean': 'mean_pct_error',
        'median': 'median_pct_error',
        'max': 'max_pct_error',
    }
    metrics: dict[str, Any] = {}
    for short, k in metric_key_map.items():
        delta = pure[k] - llm[k]  # positive => llm better
        metrics[short] = {
            'pure_value': pure[k],
            'llm_value': llm[k],
            'improvement_pct_points': delta,
            'llm_better': delta > 0.0,
        }
    return {
        'metrics': metrics,
        'beats_mean': metrics['mean']['llm_better'],
        'beats_median': metrics['median']['llm_better'],
        'beats_max': metrics['max']['llm_better'],
        'beats_all_three': all(metrics[k]['llm_better'] for k in ['mean', 'median', 'max']),
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append('# Sym20 high-noise check: LLM-guided SCD vs pure SCD')
    lines.append('')
    lines.append('## Setup')
    lines.append('')
    lines.append('- Path: corrected symmetric20 (20 positions, 1200 s)')
    lines.append('- att0: (0, 0, 0) deg')
    lines.append('- Noise scales: 1.0, 2.0')
    lines.append('- LLM-guided family: clean SCD-only candidate set A/B/C/D from 2026-03-30 line (`probe_llm_scd_only_clean.py`)')
    lines.append('')
    lines.append('## Results (lower is better; shown as mean / median / max)')
    lines.append('')
    lines.append('| noise | pure SCD | best LLM-guided SCD (candidate) | Δmean | Δmedian | Δmax | beats mean | beats median | beats max | beats all-three |')
    lines.append('|---|---|---|---:|---:|---:|---|---|---|---|')
    for rec in summary['scale_results']:
        pure = rec['pure_scd_overall']
        best = rec['best_llm_guided_scd']['overall']
        delta = rec['delta_vs_pure']['metrics']
        pure_t = f"{pure['mean_pct_error']:.6f}/{pure['median_pct_error']:.6f}/{pure['max_pct_error']:.6f}"
        best_t = f"{best['mean_pct_error']:.6f}/{best['median_pct_error']:.6f}/{best['max_pct_error']:.6f} ({rec['best_llm_guided_scd']['candidate_name']})"
        lines.append(
            f"| {rec['noise_scale']} | {pure_t} | {best_t} | "
            f"{delta['mean']['improvement_pct_points']:+.6f} | "
            f"{delta['median']['improvement_pct_points']:+.6f} | "
            f"{delta['max']['improvement_pct_points']:+.6f} | "
            f"{rec['delta_vs_pure']['beats_mean']} | {rec['delta_vs_pure']['beats_median']} | "
            f"{rec['delta_vs_pure']['beats_max']} | {rec['delta_vs_pure']['beats_all_three']} |"
        )
    lines.append('')
    lines.append('## Conclusion')
    lines.append('')
    lines.append(f"- {summary['conclusion']}")
    lines.append('')
    lines.append('## Files')
    lines.append('')
    lines.append(f"- summary_json: `{summary['files']['summary_json']}`")
    lines.append(f"- report_md: `{summary['files']['report_md']}`")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    compare_ch3_mod = load_module('sym20_highnoise_compare_ch3_mod', str(COMPARE_CH3_FILE))
    compare_shared_mod = load_module('sym20_highnoise_compare_shared_mod', str(COMPARE_SHARED_FILE))
    compute_r61_mod = load_module('sym20_highnoise_compute_r61_mod', str(COMPUTE_R61_FILE))
    probe_r55_mod = load_module('sym20_highnoise_probe_r55_mod', str(PROBE_R55_FILE))
    probe_r59_mod = load_module('sym20_highnoise_probe_r59_mod', str(PROBE_R59_FILE))

    scale_results: list[dict[str, Any]] = []
    for scale in DEFAULT_SCALES:
        suffix = compare_shared_mod.make_suffix(scale)
        source_mod = load_module(f'sym20_highnoise_source_{suffix}', str(SOURCE_FILE))
        case = compare_ch3_mod.build_symmetric20_case(source_mod)
        dataset = compare_ch3_mod.build_dataset(source_mod, case['paras'], case['att0_deg'], scale)
        params = compute_r61_mod._param_specs(source_mod)

        pure_payload, pure_status, pure_json = _run_or_reuse_pure(
            compare_shared_mod=compare_shared_mod,
            probe_r55_mod=probe_r55_mod,
            probe_r59_mod=probe_r59_mod,
            source_mod=source_mod,
            params=params,
            case=case,
            dataset=dataset,
            noise_scale=scale,
            suffix=suffix,
            force_rerun=args.force_rerun,
        )

        candidate_payloads: dict[str, dict[str, Any]] = {}
        candidate_status: dict[str, str] = {}
        candidate_jsons: dict[str, str] = {}
        for candidate in CANDIDATES:
            payload, status, out = _run_or_reuse_candidate(
                compare_shared_mod=compare_shared_mod,
                probe_r55_mod=probe_r55_mod,
                probe_r59_mod=probe_r59_mod,
                source_mod=source_mod,
                params=params,
                case=case,
                dataset=dataset,
                candidate=candidate,
                noise_scale=scale,
                suffix=suffix,
                force_rerun=args.force_rerun,
            )
            candidate_payloads[candidate['name']] = payload
            candidate_status[candidate['name']] = status
            candidate_jsons[candidate['name']] = str(out)

        best_name = min(
            candidate_payloads,
            key=lambda name: float(candidate_payloads[name]['overall']['mean_pct_error'])
        )
        best_payload = candidate_payloads[best_name]
        candidate_def = next(c for c in CANDIDATES if c['name'] == best_name)

        pure_overall = _overall_triplet(pure_payload)
        best_overall = _overall_triplet(best_payload)
        delta_vs_pure = _delta_vs_pure(pure_overall, best_overall)

        scale_results.append({
            'noise_scale': scale,
            'noise_tag': suffix,
            'noise_config': dataset['noise_config'],
            'path_case': {
                'case_key': case['case_key'],
                'case_tag': case['case_tag'],
                'display_name': case['display_name'],
                'att0_deg': case['att0_deg'],
                'n_motion_rows': case['n_motion_rows'],
                'claimed_position_count': case['claimed_position_count'],
                'total_time_s': case['total_time_s'],
                'timing_note': case['timing_note'],
                'source_builder': case['source_builder'],
                'source_reference': case['source_reference'],
                'builder_method_tag': case.get('builder_method_tag'),
            },
            'pure_scd_overall': pure_overall,
            'best_llm_guided_scd': {
                'candidate_name': best_name,
                'candidate_label': candidate_def['label'],
                'candidate_description': candidate_def['description'],
                'candidate_scd_cfg': copy.deepcopy(candidate_def['scd']),
                'overall': best_overall,
            },
            'delta_vs_pure': delta_vs_pure,
            'all_llm_candidates_overall': {
                name: _overall_triplet(payload)
                for name, payload in candidate_payloads.items()
            },
            'execution': {
                'pure_scd': pure_status,
                'llm_candidates': candidate_status,
            },
            'files': {
                'pure_scd_json': str(pure_json),
                'llm_candidate_jsons': candidate_jsons,
            },
        })

    helps_scales = [r['noise_scale'] for r in scale_results if r['delta_vs_pure']['beats_all_three']]
    if len(helps_scales) == len(scale_results):
        conclusion = 'LLM-guided SCD still helps under high noise on sym20, with all-three-metric wins at both 1.0 and 2.0.'
    elif helps_scales:
        conclusion = (
            'LLM-guided SCD still shows partial high-noise help on sym20, '
            f'but all-three-metric win appears only at scales: {helps_scales}.'
        )
    else:
        conclusion = 'LLM-guided SCD does not keep the earlier effect under high noise on sym20 (no all-three-metric win at 1.0/2.0).'

    summary_json = RESULTS_DIR / f'compare_sym20_llm_scd_vs_pure_highnoise_{args.report_date}.json'
    report_md = REPORTS_DIR / f'psins_sym20_llm_scd_vs_pure_highnoise_{args.report_date}.md'

    summary = {
        'experiment': EXPERIMENT_TAG,
        'report_date': args.report_date,
        'noise_scales': DEFAULT_SCALES,
        'candidate_family_source': str(SCRIPTS_DIR / 'probe_llm_scd_only_clean.py'),
        'selection_rule': 'best LLM-guided candidate selected by minimum overall mean_pct_error at each noise scale',
        'scale_results': scale_results,
        'helps_scales_all_three': helps_scales,
        'conclusion': conclusion,
        'code_fixes_needed': 'none',
        'files': {
            'summary_json': str(summary_json),
            'report_md': str(report_md),
        },
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_md.write_text(_render_report(summary), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps({
        'summary_json': str(summary_json),
        'report_md': str(report_md),
        'scale_results': [
            {
                'noise_scale': rec['noise_scale'],
                'pure_scd_overall': rec['pure_scd_overall'],
                'best_llm_guided_scd': rec['best_llm_guided_scd'],
                'delta_vs_pure': rec['delta_vs_pure'],
            }
            for rec in scale_results
        ],
        'helps_scales_all_three': helps_scales,
        'conclusion': conclusion,
        'code_fixes_needed': 'none',
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
