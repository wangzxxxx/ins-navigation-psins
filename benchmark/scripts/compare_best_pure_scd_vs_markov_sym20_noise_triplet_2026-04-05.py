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

COMPARISON_MODE = 'best_pure_scd_vs_markov_sym20_noise_triplet_2026_04_05'
DEFAULT_NOISE_SCALES = [0.12, 0.10, 0.08]

BEST_PURE_SCD_CFG = {
    'mode': 'once_per_phase',
    'alpha': 0.9998,
    'transition_duration': 1.0,
    'target': 'selected',
    'bias_to_target': False,
    'apply_policy_names': ['iter2_commit'],
}

for p in [ROOT, TMP_PSINS_DIR, METHOD_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--noise-scales',
        nargs='*',
        type=float,
        default=DEFAULT_NOISE_SCALES,
        help='Noise scales to evaluate; default is 0.12 0.10 0.08',
    )
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


def _triplet_text(ov: dict[str, float]) -> str:
    return f"{ov['mean_pct_error']:.6f} / {ov['median_pct_error']:.6f} / {ov['max_pct_error']:.6f}"


def _best_cfg_signature() -> str:
    return json.dumps(BEST_PURE_SCD_CFG, ensure_ascii=False, sort_keys=True)


def _scd_result_path(compare_shared_mod, noise_scale: float) -> Path:
    suffix = compare_shared_mod.make_suffix(noise_scale)
    return RESULTS_DIR / f'G4_best_pure_scd_selected_once_a09998_td1_b0_sym20_shared_{suffix}_param_errors.json'


def _delta_markov_minus_scd(markov: dict[str, float], scd: dict[str, float]) -> dict[str, float]:
    return {
        k: float(markov[k]) - float(scd[k])
        for k in ['mean_pct_error', 'median_pct_error', 'max_pct_error']
    }


def _delta_scd_minus_markov(markov: dict[str, float], scd: dict[str, float]) -> dict[str, float]:
    return {
        k: float(scd[k]) - float(markov[k])
        for k in ['mean_pct_error', 'median_pct_error', 'max_pct_error']
    }


def _beats_flags(markov: dict[str, float], scd: dict[str, float]) -> dict[str, bool]:
    mean_b = scd['mean_pct_error'] < markov['mean_pct_error']
    med_b = scd['median_pct_error'] < markov['median_pct_error']
    max_b = scd['max_pct_error'] < markov['max_pct_error']
    return {
        'mean': bool(mean_b),
        'median': bool(med_b),
        'max': bool(max_b),
        'all_three': bool(mean_b and med_b and max_b),
    }


def _run_or_reuse_best_scd(
    *,
    noise_scale: float,
    force_rerun: bool,
    compare_shared_mod,
    compare_ch3_mod,
    compute_r61_mod,
    probe_r55_mod,
    probe_r59_mod,
    source_mod,
    case: dict[str, Any],
    markov_json_path: Path,
) -> tuple[dict[str, Any], str, Path]:
    out_path = _scd_result_path(compare_shared_mod, noise_scale)
    expected_cfg = compare_shared_mod.expected_noise_config(noise_scale)
    cfg_signature = _best_cfg_signature()

    if (not force_rerun) and out_path.exists():
        old = _load_json(out_path)
        extra = old.get('extra', {}) if isinstance(old, dict) else {}
        if (
            compare_shared_mod._noise_matches(old, expected_cfg)
            and extra.get('comparison_mode') == COMPARISON_MODE
            and extra.get('path_case_tag') == case['case_tag']
            and extra.get('best_scd_cfg_signature') == cfg_signature
        ):
            return old, 'reused_verified', out_path

    dataset = compare_ch3_mod.build_dataset(source_mod, case['paras'], case['att0_deg'], noise_scale)
    params = compute_r61_mod._param_specs(source_mod)

    candidate = compare_shared_mod._build_neutral_scd_candidate()
    candidate['name'] = 'best_pure_scd_selected_once_a09998_td1_b0'
    candidate['description'] = (
        "Best pure-SCD config from 2026-04-05 sweep: "
        "target=selected, mode=once_per_phase, alpha=0.9998, transition_duration=1.0, "
        "bias_to_target=False, apply_policy_names=['iter2_commit']"
    )
    candidate['scd'] = copy.deepcopy(BEST_PURE_SCD_CFG)

    method_mod = load_module(
        f'best_pure_scd_vs_markov_r53_{compare_shared_mod.make_suffix(noise_scale)}',
        str(R53_METHOD_FILE),
    )
    method_mod = probe_r55_mod._build_patched_method(method_mod, candidate)

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
        label=f"BEST-PURE-SCD-SYM20-{compare_shared_mod.make_suffix(noise_scale).upper()}",
        scd_cfg=BEST_PURE_SCD_CFG,
    ))

    runtime = scd_result[4] if len(scd_result) >= 5 and isinstance(scd_result[4], dict) else {}

    payload = compare_shared_mod.compute_payload(
        source_mod,
        scd_result[0],
        params,
        variant=(
            f"best_pure_scd_selected_once_a09998_td1_b0_"
            f"{case['case_tag']}_{compare_shared_mod.make_suffix(noise_scale)}"
        ),
        method_file='round53_base + _build_patched_method(neutral) + _run_internalized_hybrid_scd',
        extra={
            'comparison_mode': COMPARISON_MODE,
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
            'rationale': case.get('rationale'),
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'best_scd_cfg': copy.deepcopy(BEST_PURE_SCD_CFG),
            'best_scd_cfg_signature': cfg_signature,
            'iter_patches': copy.deepcopy(candidate['iter_patches']),
            'runtime_log': {
                'schedule_log': runtime.get('schedule_log'),
                'feedback_log': runtime.get('feedback_log'),
                'scd_log': runtime.get('scd_log'),
            },
            'markov_json': str(markov_json_path),
        },
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', out_path


def _render_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append('# corrected symmetric20: best pure-SCD vs G3 Markov@20 at low-noise triplet')
    lines.append('')
    lines.append('## Fixed setup')
    lines.append('')
    lines.append('- Path: corrected symmetric20 (20-position), att0=(0,0,0), total time=1200 s')
    lines.append('- State setting: fixed 42-state')
    lines.append(
        "- Best pure-SCD config (fixed): "
        "target=selected, mode=once_per_phase, alpha=0.9998, transition_duration=1.0, "
        "bias_to_target=False, apply_policy_names=['iter2_commit']"
    )
    lines.append(f"- Noise scales: {summary['noise_scales']}")
    lines.append("- Delta convention: **Δ = Markov - SCD** (positive means SCD is better)")
    lines.append('')

    lines.append('## Per-scale metrics')
    lines.append('')
    lines.append('| noise | Markov mean/med/max | SCD mean/med/max | Δmean | Δmedian | Δmax | SCD beats(mean/med/max) | all-three |')
    lines.append('|---:|---|---|---:|---:|---:|---|---|')
    for row in summary['scales']:
        dm = row['delta_markov_minus_scd']
        b = row['scd_beats_markov']
        beat_txt = f"{int(b['mean'])}/{int(b['median'])}/{int(b['max'])}"
        lines.append(
            f"| {row['noise_scale']:.2f} | {_triplet_text(row['markov']['overall'])} | {_triplet_text(row['best_pure_scd']['overall'])} | "
            f"{dm['mean_pct_error']:+.6f} | {dm['median_pct_error']:+.6f} | {dm['max_pct_error']:+.6f} | {beat_txt} | {b['all_three']} |"
        )
    lines.append('')

    lines.append('## Threshold verdict')
    lines.append('')
    if summary['first_all_three_overtake_scale'] is None:
        lines.append('- No full overtake found at 0.12 / 0.10 / 0.08 (SCD does not beat Markov on all three metrics at any tested scale).')
    else:
        lines.append(
            f"- First full overtake scale: **{summary['first_all_three_overtake_scale']:.2f}** "
            '(SCD beats Markov on mean / median / max simultaneously).'
        )
    lines.append(f"- Conclusion: {summary['trend_conclusion']}")
    lines.append('')

    lines.append('## Artifacts')
    lines.append('')
    lines.append(f"- script: `{summary['files']['script']}`")
    lines.append(f"- summary_json: `{summary['files']['summary_json']}`")
    lines.append(f"- report_md: `{summary['files']['report_md']}`")
    lines.append('')

    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    compare_ch3_mod = load_module('triplet_compare_ch3_mod', str(COMPARE_CH3_FILE))
    compare_shared_mod = load_module('triplet_compare_shared_mod', str(COMPARE_SHARED_FILE))
    compute_r61_mod = load_module('triplet_compute_r61_mod', str(COMPUTE_R61_FILE))
    probe_r55_mod = load_module('triplet_probe_r55_mod', str(PROBE_R55_FILE))
    probe_r59_mod = load_module('triplet_probe_r59_mod', str(PROBE_R59_FILE))

    scales = [float(x) for x in args.noise_scales]

    scale_rows: list[dict[str, Any]] = []
    code_fixes_needed = 'none'

    for scale in scales:
        suffix = compare_shared_mod.make_suffix(scale)
        source_mod = load_module(f'triplet_source_{suffix}', str(SOURCE_FILE))
        case = compare_ch3_mod.build_symmetric20_case(source_mod)

        markov_payload, markov_status, markov_path = compare_ch3_mod.run_case_method(
            source_mod,
            case,
            'markov42_noisy',
            scale,
            force_rerun=args.force_rerun,
        )

        scd_payload, scd_status, scd_path = _run_or_reuse_best_scd(
            noise_scale=scale,
            force_rerun=args.force_rerun,
            compare_shared_mod=compare_shared_mod,
            compare_ch3_mod=compare_ch3_mod,
            compute_r61_mod=compute_r61_mod,
            probe_r55_mod=probe_r55_mod,
            probe_r59_mod=probe_r59_mod,
            source_mod=source_mod,
            case=case,
            markov_json_path=markov_path,
        )

        markov_ov = _overall_triplet(markov_payload)
        scd_ov = _overall_triplet(scd_payload)
        delta_m_minus_s = _delta_markov_minus_scd(markov_ov, scd_ov)
        delta_s_minus_m = _delta_scd_minus_markov(markov_ov, scd_ov)
        beats = _beats_flags(markov_ov, scd_ov)

        scale_rows.append({
            'noise_scale': scale,
            'markov': {
                'json_path': str(markov_path),
                'status': markov_status,
                'overall': markov_ov,
            },
            'best_pure_scd': {
                'json_path': str(scd_path),
                'status': scd_status,
                'overall': scd_ov,
                'scd_cfg': copy.deepcopy(BEST_PURE_SCD_CFG),
            },
            'delta_markov_minus_scd': delta_m_minus_s,
            'delta_scd_minus_markov': delta_s_minus_m,
            'scd_beats_markov': beats,
        })

    first_all_three = None
    for row in scale_rows:
        if row['scd_beats_markov']['all_three']:
            first_all_three = float(row['noise_scale'])
            break

    any_mean = any(r['scd_beats_markov']['mean'] for r in scale_rows)
    any_med = any(r['scd_beats_markov']['median'] for r in scale_rows)
    any_max = any(r['scd_beats_markov']['max'] for r in scale_rows)
    any_all_three = any(r['scd_beats_markov']['all_three'] for r in scale_rows)

    if any_all_three:
        trend_conclusion = (
            f'Lowering noise helps enough for full SCD overtake, first observed at noise scale {first_all_three:.2f}.'
        )
    elif any_mean or any_med or any_max:
        trend_conclusion = (
            'Lower noise gives partial gains (some metrics flip), but no full mean/median/max overtake appears '
            'within 0.12→0.08.'
        )
    else:
        trend_conclusion = (
            'Lowering noise does not make this pure-SCD config surpass Markov on core overall metrics in the tested range.'
        )

    scale_tag = 'noise_triplet_0p12_0p10_0p08'
    summary_json = RESULTS_DIR / f'best_pure_scd_vs_markov_sym20_{args.report_date}_{scale_tag}_summary.json'
    report_md = REPORTS_DIR / f'psins_best_pure_scd_vs_markov_sym20_{args.report_date}_{scale_tag}.md'

    summary = {
        'task': 'best_pure_scd_vs_markov_sym20_noise_triplet_threshold_check',
        'comparison_mode': COMPARISON_MODE,
        'report_date': args.report_date,
        'noise_scales': scales,
        'setup': {
            'path_case_tag': 'ch3corrected_symmetric20_att0zero_1200s',
            'att0_deg': [0.0, 0.0, 0.0],
            'state_setting': 42,
            'best_pure_scd_cfg': copy.deepcopy(BEST_PURE_SCD_CFG),
            'evaluation_rule': 'overall mean / median / max (% error, lower is better)',
            'delta_definition': 'delta_markov_minus_scd = markov - scd (positive => scd better)',
        },
        'scales': scale_rows,
        'aggregate_beats': {
            'mean': any_mean,
            'median': any_med,
            'max': any_max,
            'all_three': any_all_three,
        },
        'first_all_three_overtake_scale': first_all_three,
        'trend_conclusion': trend_conclusion,
        'code_fixes_needed': code_fixes_needed,
        'files': {
            'script': str(Path(__file__)),
            'summary_json': str(summary_json),
            'report_md': str(report_md),
        },
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_md.write_text(_render_report(summary), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps({
        'summary_json': str(summary_json),
        'report_md': str(report_md),
        'scales': [
            {
                'noise_scale': row['noise_scale'],
                'markov_overall': row['markov']['overall'],
                'best_pure_scd_overall': row['best_pure_scd']['overall'],
                'delta_markov_minus_scd': row['delta_markov_minus_scd'],
                'scd_beats_markov': row['scd_beats_markov'],
                'markov_json': row['markov']['json_path'],
                'best_pure_scd_json': row['best_pure_scd']['json_path'],
                'markov_status': row['markov']['status'],
                'best_pure_scd_status': row['best_pure_scd']['status'],
            }
            for row in scale_rows
        ],
        'first_all_three_overtake_scale': first_all_three,
        'trend_conclusion': trend_conclusion,
        'code_fixes_needed': code_fixes_needed,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
