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
METHOD_DIR = ROOT / 'psins_method_bench' / 'methods' / 'markov'
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
REPORTS_DIR = ROOT / 'reports'
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'
COMPARE_SCRIPT = SCRIPTS_DIR / 'compare_ch3_corrected_symmetric20_vs_legacy19pos_1200s.py'
BASELINE_G3_JSON = RESULTS_DIR / 'M_markov_42state_gm1_ch3corrected_symmetric20_att0zero_1200s_shared_noise0p12_param_errors.json'

COMPARISON_MODE = 'g4_sym20_retune_r59_r60_r61_on_corrected_symmetric20_att0zero_1200s'
DEFAULT_NOISE_SCALE = 0.12
CURRENT_TRANSFERRED_ROUND61 = 'r61_s20_08988_ryz00116'

for p in [ROOT, ROOT / 'tmp_psins_py', METHOD_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module
from compare_four_methods_shared_noise import _load_json, _noise_matches, compute_payload, make_suffix
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs
from probe_round55_newline import _build_patched_method
from probe_round59_h_scd_hybrid import (
    HYBRID_CANDIDATES,
    _merge_hybrid_candidate,
    _run_internalized_hybrid_scd,
)
from probe_round60_conservative import (
    ROUND60_CONSERVATIVE_CANDIDATES,
    _merge_round60_candidate,
)
from probe_round61_hybrid_micro import ROUND61_CANDIDATES, _merge_round61_candidate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=DEFAULT_NOISE_SCALE)
    parser.add_argument('--report-date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def _candidate_registry() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for cand in HYBRID_CANDIDATES:
        merged = _merge_hybrid_candidate(cand)
        rows.append({
            'family': 'round59_hybrid',
            'source_name': cand['name'],
            'candidate': merged,
            'merge_source': 'probe_round59_h_scd_hybrid._merge_hybrid_candidate',
        })

    for cand in ROUND60_CONSERVATIVE_CANDIDATES:
        merged = _merge_round60_candidate(cand)
        rows.append({
            'family': 'round60_conservative',
            'source_name': cand['name'],
            'candidate': merged,
            'merge_source': 'probe_round60_conservative._merge_round60_candidate',
        })

    for cand in ROUND61_CANDIDATES:
        merged = _merge_round61_candidate(cand)
        rows.append({
            'family': 'round61_micro',
            'source_name': cand['name'],
            'candidate': merged,
            'merge_source': 'probe_round61_hybrid_micro._merge_round61_candidate',
        })

    return rows


def _candidate_output_path(candidate_name: str, noise_scale: float) -> Path:
    suffix = make_suffix(noise_scale)
    return RESULTS_DIR / f'G4_sym20_retune_{candidate_name}_shared_{suffix}_param_errors.json'


def _safe_float(v: Any) -> float:
    return float(v)


def _delta_vs_baseline(overall: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        out[k] = _safe_float(overall[k]) - _safe_float(baseline[k])
    return out


def _improve_vs_baseline(overall: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        out[k] = _safe_float(baseline[k]) - _safe_float(overall[k])
    return out


def _beats_flags(overall: dict[str, Any], baseline: dict[str, Any]) -> dict[str, bool]:
    mean_b = _safe_float(overall['mean_pct_error']) < _safe_float(baseline['mean_pct_error'])
    med_b = _safe_float(overall['median_pct_error']) < _safe_float(baseline['median_pct_error'])
    max_b = _safe_float(overall['max_pct_error']) < _safe_float(baseline['max_pct_error'])
    return {
        'mean': mean_b,
        'median': med_b,
        'max': max_b,
        'all_three': bool(mean_b and med_b and max_b),
    }


def _triplet(overall: dict[str, Any]) -> str:
    return f"{overall['mean_pct_error']:.6f} / {overall['median_pct_error']:.6f} / {overall['max_pct_error']:.6f}"


def _render_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append('# G4 retune on corrected symmetric20 (noise0p12, att0=(0,0,0))')
    lines.append('')
    lines.append('## 1) Fixed setup')
    lines.append('')
    lines.append(f"- Path: **{summary['path_case']['case_tag']}**")
    lines.append(f"- Att0: **{summary['path_case']['att0_deg']}**")
    lines.append(f"- Noise scale: **{summary['noise_scale']}**")
    lines.append(f"- G3 baseline JSON: `{summary['baseline']['json_path']}`")
    lines.append(f"- G3 baseline overall (mean / median / max): **{_triplet(summary['baseline']['overall'])}**")
    lines.append('')

    lines.append('## 2) Candidate ranking (sort: mean → max → median)')
    lines.append('')
    lines.append('| rank | family | candidate | mean | median | max | Δmean vs G3 | Δmedian vs G3 | Δmax vs G3 | beats(mean/med/max) |')
    lines.append('|---:|---|---|---:|---:|---:|---:|---:|---:|---|')
    for row in summary['ranking']:
        d = row['delta_vs_g3']
        b = row['beats_g3']
        beat_txt = f"{int(b['mean'])}/{int(b['median'])}/{int(b['max'])}"
        lines.append(
            f"| {row['rank']} | {row['family']} | {row['candidate_name']} | "
            f"{row['overall']['mean_pct_error']:.6f} | {row['overall']['median_pct_error']:.6f} | {row['overall']['max_pct_error']:.6f} | "
            f"{d['mean_pct_error']:+.6f} | {d['median_pct_error']:+.6f} | {d['max_pct_error']:+.6f} | {beat_txt} |"
        )
    lines.append('')

    lines.append('## 3) Headline verdict')
    lines.append('')
    lines.append(f"- Best candidate: **{summary['best_candidate']['candidate_name']}** ({summary['best_candidate']['family']})")
    lines.append(f"- Best overall: **{_triplet(summary['best_candidate']['overall'])}**")
    lines.append(
        f"- Beats G3? mean={summary['best_candidate']['beats_g3']['mean']}, "
        f"median={summary['best_candidate']['beats_g3']['median']}, max={summary['best_candidate']['beats_g3']['max']}, "
        f"all_three={summary['best_candidate']['beats_g3']['all_three']}"
    )
    lines.append(f"- Any candidate beats G3 on mean/median/max: {summary['any_beats_g3']} (all_three={summary['any_beats_g3_all_three']})")
    if summary.get('clear_all_three_winner'):
        lines.append(f"- ✅ Clear all-three winner: **{summary['clear_all_three_winner']['candidate_name']}**")
    else:
        gap = summary['nearest_gap_best_vs_g3']
        lines.append(
            '- ❌ No candidate beats G3 on all three metrics. '
            f"Nearest gap (best candidate vs G3): Δmean={gap['mean_pct_error']:+.6f}, "
            f"Δmedian={gap['median_pct_error']:+.6f}, Δmax={gap['max_pct_error']:+.6f}."
        )
    lines.append('')

    lines.append('## 4) Current transferred Round61 candidate position')
    lines.append('')
    lines.append(f"- Candidate: `{summary['current_transferred_round61']['candidate_name']}`")
    lines.append(f"- Rank: **{summary['current_transferred_round61']['rank']} / {summary['candidate_count']}**")
    lines.append(f"- Overall (mean / median / max): **{_triplet(summary['current_transferred_round61']['overall'])}**")
    lines.append(
        f"- vs G3: Δmean={summary['current_transferred_round61']['delta_vs_g3']['mean_pct_error']:+.6f}, "
        f"Δmedian={summary['current_transferred_round61']['delta_vs_g3']['median_pct_error']:+.6f}, "
        f"Δmax={summary['current_transferred_round61']['delta_vs_g3']['max_pct_error']:+.6f}"
    )
    lines.append('')

    lines.append('## 5) Artifacts')
    lines.append('')
    lines.append(f"- summary_json: `{summary['artifacts']['summary_json']}`")
    lines.append(f"- report_md: `{summary['artifacts']['report_md']}`")
    lines.append(f"- best_candidate_json: `{summary['artifacts']['best_candidate_json']}`")
    lines.append('')

    return '\n'.join(lines) + '\n'


def _load_compare_module(noise_scale: float):
    suffix = make_suffix(noise_scale)
    return load_module(f'compare_sym20_vs_legacy_mod_{suffix}', str(COMPARE_SCRIPT))


def _run_one_candidate(
    *,
    source_mod,
    params,
    dataset,
    case,
    noise_scale: float,
    baseline_overall: dict[str, Any],
    reg: dict[str, Any],
    idx: int,
    force_rerun: bool,
) -> dict[str, Any]:
    family = reg['family']
    merged = copy.deepcopy(reg['candidate'])
    candidate_name = merged['name']
    out_path = _candidate_output_path(candidate_name, noise_scale)

    if (not force_rerun) and out_path.exists():
        payload = _load_json(out_path)
        extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
        if (
            _noise_matches(payload, dataset['noise_config'])
            and extra.get('comparison_mode') == COMPARISON_MODE
            and extra.get('candidate_name') == candidate_name
            and extra.get('path_case_tag') == case['case_tag']
        ):
            overall = payload['overall']
            return {
                'family': family,
                'source_name': reg['source_name'],
                'candidate_name': candidate_name,
                'merge_source': reg['merge_source'],
                'overall': overall,
                'delta_vs_g3': _delta_vs_baseline(overall, baseline_overall),
                'improvement_vs_g3': _improve_vs_baseline(overall, baseline_overall),
                'beats_g3': _beats_flags(overall, baseline_overall),
                'result_json': str(out_path),
                'status': 'reused_verified',
            }

    method_mod = load_module(f'g4_sym20_r53_mod_{idx}_{candidate_name}', str(R53_METHOD_FILE))
    method_mod = _build_patched_method(method_mod, merged)

    result = _run_internalized_hybrid_scd(
        method_mod,
        source_mod,
        dataset['imu_noisy'],
        dataset['pos0'],
        dataset['ts'],
        bi_g=dataset['bi_g'],
        bi_a=dataset['bi_a'],
        tau_g=dataset['tau_g'],
        tau_a=dataset['tau_a'],
        label=f'G4-SYM20-{idx:02d}-{candidate_name}',
        scd_cfg=merged['scd'],
    )

    runtime_extra = result[4] if len(result) >= 5 and isinstance(result[4], dict) else {}
    payload = compute_payload(
        source_mod,
        result[0],
        params,
        variant=f'g4_sym20_retune_{candidate_name}_{make_suffix(noise_scale)}',
        method_file='round53_base + _build_patched_method + _run_internalized_hybrid_scd',
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
            'candidate_family': family,
            'source_name': reg['source_name'],
            'candidate_name': candidate_name,
            'merge_source': reg['merge_source'],
            'candidate_description': merged.get('description'),
            'candidate_rationale': merged.get('rationale'),
            'iter_patches': copy.deepcopy(merged.get('iter_patches', {})),
            'scd': copy.deepcopy(merged.get('scd', {})),
            'post_rx_y_mult': float(merged.get('post_rx_y_mult', 1.0)),
            'post_ry_z_mult': float(merged.get('post_ry_z_mult', 1.0)),
            'runtime_log': {
                'schedule_log': runtime_extra.get('schedule_log'),
                'feedback_log': runtime_extra.get('feedback_log'),
                'scd_log': runtime_extra.get('scd_log'),
            },
            'baseline_g3_json': str(BASELINE_G3_JSON),
        },
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    overall = payload['overall']
    return {
        'family': family,
        'source_name': reg['source_name'],
        'candidate_name': candidate_name,
        'merge_source': reg['merge_source'],
        'overall': overall,
        'delta_vs_g3': _delta_vs_baseline(overall, baseline_overall),
        'improvement_vs_g3': _improve_vs_baseline(overall, baseline_overall),
        'beats_g3': _beats_flags(overall, baseline_overall),
        'result_json': str(out_path),
        'status': 'rerun',
    }


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_payload = _load_json(BASELINE_G3_JSON)
    baseline_overall = baseline_payload['overall']

    compare_mod = _load_compare_module(args.noise_scale)
    source_mod = load_module(f'g4_sym20_source_{make_suffix(args.noise_scale)}', str(SOURCE_FILE))
    case = compare_mod.build_symmetric20_case(source_mod)
    dataset = compare_mod.build_dataset(source_mod, case['paras'], case['att0_deg'], args.noise_scale)
    params = _param_specs(source_mod)

    registry = _candidate_registry()
    rows: list[dict[str, Any]] = []
    for idx, reg in enumerate(registry, start=1):
        row = _run_one_candidate(
            source_mod=source_mod,
            params=params,
            dataset=dataset,
            case=case,
            noise_scale=args.noise_scale,
            baseline_overall=baseline_overall,
            reg=reg,
            idx=idx,
            force_rerun=args.force_rerun,
        )
        rows.append(row)

    rows_sorted = sorted(
        rows,
        key=lambda x: (
            float(x['overall']['mean_pct_error']),
            float(x['overall']['max_pct_error']),
            float(x['overall']['median_pct_error']),
        ),
    )
    for rank, row in enumerate(rows_sorted, start=1):
        row['rank'] = rank

    best = copy.deepcopy(rows_sorted[0])

    any_beats = {
        'mean': any(r['beats_g3']['mean'] for r in rows_sorted),
        'median': any(r['beats_g3']['median'] for r in rows_sorted),
        'max': any(r['beats_g3']['max'] for r in rows_sorted),
    }
    any_beats_all_three = any(r['beats_g3']['all_three'] for r in rows_sorted)

    clear_all_three_winner = None
    for r in rows_sorted:
        if r['beats_g3']['all_three']:
            clear_all_three_winner = {
                'candidate_name': r['candidate_name'],
                'family': r['family'],
                'rank': r['rank'],
                'overall': r['overall'],
                'result_json': r['result_json'],
            }
            break

    transferred_row = None
    for r in rows_sorted:
        if r['candidate_name'] == CURRENT_TRANSFERRED_ROUND61:
            transferred_row = copy.deepcopy(r)
            break
    if transferred_row is None:
        raise RuntimeError(f'Current transferred Round61 candidate not found: {CURRENT_TRANSFERRED_ROUND61}')

    suffix = make_suffix(args.noise_scale)
    summary_json = RESULTS_DIR / f'g4_sym20_retune_{args.report_date}_{suffix}_summary.json'
    report_md = REPORTS_DIR / f'psins_g4_sym20_retune_{args.report_date}_{suffix}.md'

    summary = {
        'task': 'g4_retune_on_corrected_symmetric20',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'noise_config': dataset['noise_config'],
        'comparison_mode': COMPARISON_MODE,
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
        'baseline': {
            'json_path': str(BASELINE_G3_JSON),
            'overall': baseline_overall,
            'variant': baseline_payload.get('variant'),
        },
        'candidate_count': len(rows_sorted),
        'ranking_rule': 'sort by mean_pct_error, then max_pct_error, then median_pct_error (ascending)',
        'ranking': rows_sorted,
        'best_candidate': best,
        'any_beats_g3': any_beats,
        'any_beats_g3_all_three': any_beats_all_three,
        'clear_all_three_winner': clear_all_three_winner,
        'nearest_gap_best_vs_g3': best['delta_vs_g3'],
        'current_transferred_round61': transferred_row,
        'artifacts': {
            'summary_json': str(summary_json),
            'report_md': str(report_md),
            'best_candidate_json': best['result_json'],
            'all_candidate_jsons': [r['result_json'] for r in rows_sorted],
        },
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_md.write_text(_render_report(summary), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps({
        'summary_json': str(summary_json),
        'report_md': str(report_md),
        'best_candidate': {
            'name': best['candidate_name'],
            'family': best['family'],
            'overall': best['overall'],
            'beats_g3': best['beats_g3'],
            'result_json': best['result_json'],
        },
        'any_beats_g3': any_beats,
        'any_beats_g3_all_three': any_beats_all_three,
        'current_transferred_round61': {
            'name': transferred_row['candidate_name'],
            'rank': transferred_row['rank'],
            'overall': transferred_row['overall'],
            'delta_vs_g3': transferred_row['delta_vs_g3'],
            'result_json': transferred_row['result_json'],
        },
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
