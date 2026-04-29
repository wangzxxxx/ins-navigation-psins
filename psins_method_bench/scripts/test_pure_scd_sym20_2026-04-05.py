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

G3_BASELINE_JSON = RESULTS_DIR / 'M_markov_42state_gm1_ch3corrected_symmetric20_att0zero_1200s_shared_noise0p12_param_errors.json'
BEST_G4_JSON = RESULTS_DIR / 'G4_sym20_retune_scd_scale_once_a0999_biaslink_commit_shared_noise0p12_param_errors.json'

for p in [ROOT, TMP_PSINS_DIR, METHOD_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=0.12)
    parser.add_argument('--report-date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _overall_triplet(payload: dict[str, Any]) -> dict[str, float]:
    ov = payload['overall']
    return {
        'mean_pct_error': float(ov['mean_pct_error']),
        'median_pct_error': float(ov['median_pct_error']),
        'max_pct_error': float(ov['max_pct_error']),
    }


def _compare_against(pure_payload: dict[str, Any], ref_payload: dict[str, Any], ref_label: str) -> dict[str, Any]:
    pure = _overall_triplet(pure_payload)
    ref = _overall_triplet(ref_payload)

    metrics = {}
    wins = 0
    for k in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        delta = ref[k] - pure[k]
        better = delta > 0.0
        if better:
            wins += 1
        metrics[k] = {
            'pure_scd': pure[k],
            'reference': ref[k],
            'improvement_pct_points': delta,
            'pure_scd_better': better,
        }

    return {
        'reference_label': ref_label,
        'wins_count': wins,
        'beats_all_overall_metrics': wins == 3,
        'beats_mean_metric': metrics['mean_pct_error']['pure_scd_better'],
        'metrics': metrics,
    }


def _render_report(summary: dict[str, Any]) -> str:
    pure = summary['pure_scd']['overall']
    cmp_g3 = summary['comparisons']['vs_g3']
    cmp_g4 = summary['comparisons']['vs_best_g4']

    def t(d: dict[str, float]) -> str:
        return f"{d['mean_pct_error']:.6f} / {d['median_pct_error']:.6f} / {d['max_pct_error']:.6f}"

    lines: list[str] = []
    lines.append('# Focused pure-SCD test on corrected symmetric20 (noise0p12)')
    lines.append('')
    lines.append('## Setup')
    lines.append('')
    lines.append('- Path: corrected symmetric20 (20-position, 1200 s)')
    lines.append('- att0: (0, 0, 0) deg')
    lines.append(f"- noise_scale: {summary['noise_scale']}")
    lines.append('- Method under test: pure neutral Markov+SCD baseline (no LLM, iter2 once-per-phase SCD only)')
    lines.append('')
    lines.append('## Overall metrics (mean / median / max, % error; lower is better)')
    lines.append('')
    lines.append(f"- Pure SCD: **{t(pure)}**")
    lines.append(f"- G3 Markov@20 baseline: **{t(summary['g3']['overall'])}**")
    lines.append(f"- Best G4 retune: **{t(summary['best_g4']['overall'])}**")
    lines.append('')
    lines.append('## Deltas (reference - pure SCD, + means pure SCD better)')
    lines.append('')
    lines.append('| comparison | Δmean | Δmedian | Δmax | pure wins count | beats all 3? | beats mean? |')
    lines.append('|---|---:|---:|---:|---:|---|---|')
    lines.append(
        f"| vs G3 Markov@20 | {cmp_g3['metrics']['mean_pct_error']['improvement_pct_points']:+.6f} | "
        f"{cmp_g3['metrics']['median_pct_error']['improvement_pct_points']:+.6f} | "
        f"{cmp_g3['metrics']['max_pct_error']['improvement_pct_points']:+.6f} | "
        f"{cmp_g3['wins_count']} | {cmp_g3['beats_all_overall_metrics']} | {cmp_g3['beats_mean_metric']} |"
    )
    lines.append(
        f"| vs best G4 retune | {cmp_g4['metrics']['mean_pct_error']['improvement_pct_points']:+.6f} | "
        f"{cmp_g4['metrics']['median_pct_error']['improvement_pct_points']:+.6f} | "
        f"{cmp_g4['metrics']['max_pct_error']['improvement_pct_points']:+.6f} | "
        f"{cmp_g4['wins_count']} | {cmp_g4['beats_all_overall_metrics']} | {cmp_g4['beats_mean_metric']} |"
    )
    lines.append('')
    lines.append('## Artifact files')
    lines.append('')
    lines.append(f"- pure_scd_json: `{summary['files']['pure_scd_json']}`")
    lines.append(f"- summary_json: `{summary['files']['summary_json']}`")
    lines.append(f"- report_md: `{summary['files']['report_md']}`")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    compare_ch3_mod = load_module('pure_scd_compare_ch3_mod', str(COMPARE_CH3_FILE))
    compare_shared_mod = load_module('pure_scd_compare_shared_mod', str(COMPARE_SHARED_FILE))
    compute_r61_mod = load_module('pure_scd_compute_r61_mod', str(COMPUTE_R61_FILE))
    probe_r55_mod = load_module('pure_scd_probe_r55_mod', str(PROBE_R55_FILE))
    probe_r59_mod = load_module('pure_scd_probe_r59_mod', str(PROBE_R59_FILE))

    source_mod = load_module(f'pure_scd_sym20_source_{compare_shared_mod.make_suffix(args.noise_scale)}', str(SOURCE_FILE))

    case = compare_ch3_mod.build_symmetric20_case(source_mod)
    dataset = compare_ch3_mod.build_dataset(source_mod, case['paras'], case['att0_deg'], args.noise_scale)
    params = compute_r61_mod._param_specs(source_mod)

    suffix = compare_shared_mod.make_suffix(args.noise_scale)
    pure_json = RESULTS_DIR / f'G4_pure_scd_neutral_ch3corrected_symmetric20_att0zero_1200s_shared_{suffix}_param_errors.json'

    expected_cfg = compare_shared_mod.expected_noise_config(args.noise_scale)
    pure_payload: dict[str, Any] | None = None
    execution = 'rerun'

    if (not args.force_rerun) and pure_json.exists():
        old = _load_json(pure_json)
        if compare_shared_mod._noise_matches(old, expected_cfg):
            extra = old.get('extra', {}) if isinstance(old, dict) else {}
            if (
                extra.get('comparison_mode') == 'pure_scd_sym20_focused_eval_2026_04_05'
                and extra.get('path_case_tag') == case['case_tag']
                and extra.get('method_family') == 'pure_neutral_scd_baseline'
            ):
                pure_payload = old
                execution = 'reused_verified'

    if pure_payload is None:
        neutral_candidate = compare_shared_mod._build_neutral_scd_candidate()
        method_mod = load_module(f'pure_scd_sym20_r53_{suffix}', str(R53_METHOD_FILE))
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

        pure_payload = compare_shared_mod.compute_payload(
            source_mod,
            scd_result[0],
            params,
            variant=f'pure_scd_neutral_{case["case_tag"]}_{suffix}',
            method_file='round53_base + _build_patched_method(neutral) + _run_internalized_hybrid_scd',
            extra={
                'comparison_mode': 'pure_scd_sym20_focused_eval_2026_04_05',
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
                'noise_scale': args.noise_scale,
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
                'baseline_g3_json': str(G3_BASELINE_JSON),
                'baseline_best_g4_json': str(BEST_G4_JSON),
            },
        )
        pure_json.write_text(json.dumps(pure_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    g3_payload = _load_json(G3_BASELINE_JSON)
    best_g4_payload = _load_json(BEST_G4_JSON)

    cmp_g3 = _compare_against(pure_payload, g3_payload, 'g3_markov20')
    cmp_g4 = _compare_against(pure_payload, best_g4_payload, 'g4_best_retune')

    summary_json = RESULTS_DIR / f'pure_scd_sym20_focused_compare_vs_g3_bestg4_{suffix}_2026-04-05.json'
    report_md = REPORTS_DIR / f'psins_pure_scd_sym20_vs_g3_bestg4_{args.report_date}_{suffix}.md'

    summary = {
        'experiment': 'pure_scd_sym20_focused_eval_2026_04_05',
        'noise_scale': args.noise_scale,
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
        'execution': execution,
        'pure_scd': {
            'json_path': str(pure_json),
            'overall': _overall_triplet(pure_payload),
            'focus_scale_pct': pure_payload.get('focus_scale_pct', {}),
            'lever_guard_pct': pure_payload.get('lever_guard_pct', {}),
        },
        'g3': {
            'json_path': str(G3_BASELINE_JSON),
            'overall': _overall_triplet(g3_payload),
        },
        'best_g4': {
            'json_path': str(BEST_G4_JSON),
            'overall': _overall_triplet(best_g4_payload),
        },
        'comparisons': {
            'vs_g3': cmp_g3,
            'vs_best_g4': cmp_g4,
        },
        'decision': {
            'pure_scd_beats_g3_all3': cmp_g3['beats_all_overall_metrics'],
            'pure_scd_beats_g3_mean': cmp_g3['beats_mean_metric'],
            'pure_scd_beats_best_g4_all3': cmp_g4['beats_all_overall_metrics'],
            'pure_scd_beats_best_g4_mean': cmp_g4['beats_mean_metric'],
        },
        'code_fixes_needed': 'none',
        'files': {
            'pure_scd_json': str(pure_json),
            'summary_json': str(summary_json),
            'report_md': str(report_md),
        },
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_md.write_text(_render_report(summary), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps({
        'execution': execution,
        'pure_scd_json': str(pure_json),
        'summary_json': str(summary_json),
        'report_md': str(report_md),
        'pure_scd_overall': summary['pure_scd']['overall'],
        'g3_overall': summary['g3']['overall'],
        'best_g4_overall': summary['best_g4']['overall'],
        'pure_scd_beats_g3_all3': summary['decision']['pure_scd_beats_g3_all3'],
        'pure_scd_beats_best_g4_all3': summary['decision']['pure_scd_beats_best_g4_all3'],
        'pure_scd_beats_g3_mean': summary['decision']['pure_scd_beats_g3_mean'],
        'pure_scd_beats_best_g4_mean': summary['decision']['pure_scd_beats_best_g4_mean'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
