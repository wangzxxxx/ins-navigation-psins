from __future__ import annotations

import argparse
import copy
import json
import re
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
PROBE_R61_MICRO_FILE = SCRIPTS_DIR / 'probe_round61_hybrid_micro.py'

G3_BASELINE_JSON = RESULTS_DIR / 'M_markov_42state_gm1_ch3corrected_symmetric20_att0zero_1200s_shared_noise0p12_param_errors.json'
OLD_PURE_BEST_JSON = RESULTS_DIR / 'G4_pure_scd_sym20_sweep_phase2_bias_false_sel_once_a09998_td1_b0_from_phase1_sel_once_a09998_td1_b1_shared_noise0p12_param_errors.json'

COMPARISON_MODE = 'sym20_native_scd_round61_focused_search_2026_04_05'
ROUND61_REFERENCE_CANDIDATE = 'r61_s20_08988_ryz00116'

STAGE_A_ALPHAS = [0.9999, 0.99985, 0.9998, 0.99975, 0.9997]
STAGE_A_TDS = [0.8, 1.0, 1.2, 1.5]

for p in [ROOT, TMP_PSINS_DIR, METHOD_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module


METRIC_KEYS = ['mean_pct_error', 'median_pct_error', 'max_pct_error']


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


def _sanitize_name(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]+', '_', text)


def _alpha_tag(alpha: float) -> str:
    return f"a{int(round(alpha * 100000)):06d}"


def _td_tag(td: float) -> str:
    if abs(td - round(td)) < 1e-12:
        return f"td{int(round(td))}"
    return f"td{str(td).replace('.', 'p')}"


def _scaled_toward_old(*, neutral: float, old: float, scale: float) -> float:
    return float(neutral + (old - neutral) * scale)


def _candidate_signature(candidate: dict[str, Any]) -> str:
    keep = {
        'name': candidate['name'],
        'stage': candidate['stage'],
        'parent': candidate.get('parent'),
        'scd': candidate['scd'],
        'repair': candidate.get('repair', {}),
    }
    return json.dumps(keep, ensure_ascii=False, sort_keys=True)


def _is_old_pure_point(candidate: dict[str, Any]) -> bool:
    repair = candidate.get('repair', {})
    if repair.get('state20_alpha_mult') is not None:
        return False
    if repair.get('post_ry_z_mult') is not None:
        return False
    scd = candidate['scd']
    return (
        scd.get('target') == 'selected'
        and scd.get('mode') == 'once_per_phase'
        and abs(float(scd.get('alpha', 0.0)) - 0.9998) < 1e-12
        and abs(float(scd.get('transition_duration', 0.0)) - 1.0) < 1e-12
        and bool(scd.get('bias_to_target')) is False
        and list(scd.get('apply_policy_names', [])) == ['iter2_commit']
    )


def _inspect_round61_reference_knobs() -> dict[str, Any]:
    probe_r61_mod = load_module('sym20_native_round61_probe_ref', str(PROBE_R61_MICRO_FILE))

    ref_extra = None
    for candidate in probe_r61_mod.ROUND61_CANDIDATES:
        if candidate.get('name') == ROUND61_REFERENCE_CANDIDATE:
            ref_extra = copy.deepcopy(candidate)
            break
    if ref_extra is None:
        raise KeyError(f'Round61 reference candidate missing: {ROUND61_REFERENCE_CANDIDATE}')

    merged = probe_r61_mod._merge_round61_candidate(ref_extra)
    iter2_patch = merged.get('iter_patches', {}).get(1, {})
    state_alpha_mult = iter2_patch.get('state_alpha_mult', {}) if isinstance(iter2_patch, dict) else {}
    state20_alpha = float(state_alpha_mult.get(20, 1.0))
    post_ry_z_mult = float(merged.get('post_ry_z_mult', 1.0))

    return {
        'reference_candidate_name': ROUND61_REFERENCE_CANDIDATE,
        'inspected_from_script': str(PROBE_R61_MICRO_FILE),
        'inspected_extra_candidate': ref_extra,
        'inspected_merged_scd': copy.deepcopy(merged.get('scd', {})),
        'iter2_state20_alpha_mult_old': state20_alpha,
        'post_ry_z_mult_old': post_ry_z_mult,
        'neutral_state20_alpha_mult': 1.0,
        'neutral_post_ry_z_mult': 1.0,
        'delta_from_neutral': {
            'iter2_state20_alpha_mult': float(state20_alpha - 1.0),
            'post_ry_z_mult': float(post_ry_z_mult - 1.0),
        },
        'extracted_useful_knobs': ['iter2.state_alpha_mult[20]', 'post_ry_z_mult'],
    }


def _build_stage_a_candidates() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    idx = 0
    for alpha in STAGE_A_ALPHAS:
        for td in STAGE_A_TDS:
            idx += 1
            name = f"stageA_sel_once_b0_{_alpha_tag(alpha)}_{_td_tag(td)}"
            out.append({
                'name': name,
                'description': (
                    'Stage A base-local refinement: target=selected, mode=once_per_phase, '
                    f'alpha={alpha}, transition_duration={td}, bias_to_target=False, '
                    "apply_policy_names=['iter2_commit']"
                ),
                'stage': 'A',
                'stage_index': idx,
                'parent': None,
                'scd': {
                    'mode': 'once_per_phase',
                    'alpha': float(alpha),
                    'transition_duration': float(td),
                    'target': 'selected',
                    'bias_to_target': False,
                    'apply_policy_names': ['iter2_commit'],
                },
                'repair': {
                    'state20_alpha_mult': None,
                    'post_ry_z_mult': None,
                    'scaling': None,
                    'pattern': 'base_only',
                    'knobs': [],
                },
            })
    return out


def _build_stage_b_candidates(stage_a_bases: list[dict[str, Any]], round61_knobs: dict[str, Any]) -> list[dict[str, Any]]:
    # 2 bases × 6 variants = 12 candidates (bounded as requested)
    use_bases = stage_a_bases[:2]

    old_state20 = float(round61_knobs['iter2_state20_alpha_mult_old'])
    old_ryz = float(round61_knobs['post_ry_z_mult_old'])

    variants = [
        {'variant': 's20_25', 'scale': 0.25, 'state20': True, 'ryz': False},
        {'variant': 's20_50', 'scale': 0.50, 'state20': True, 'ryz': False},
        {'variant': 's20_75', 'scale': 0.75, 'state20': True, 'ryz': False},
        {'variant': 's20ryz_25', 'scale': 0.25, 'state20': True, 'ryz': True},
        {'variant': 's20ryz_50', 'scale': 0.50, 'state20': True, 'ryz': True},
        {'variant': 's20ryz_75', 'scale': 0.75, 'state20': True, 'ryz': True},
    ]

    out: list[dict[str, Any]] = []
    idx = 0
    for base in use_bases:
        for v in variants:
            idx += 1
            state20_alpha = _scaled_toward_old(neutral=1.0, old=old_state20, scale=float(v['scale'])) if v['state20'] else None
            post_ryz = _scaled_toward_old(neutral=1.0, old=old_ryz, scale=float(v['scale'])) if v['ryz'] else None
            name = f"stageB_{_sanitize_name(base['name'])}_{v['variant']}"
            out.append({
                'name': name,
                'description': (
                    f"Stage B tiny repair on {base['name']}: variant={v['variant']}, scale={v['scale']:.2f}, "
                    f"state20_alpha_mult={state20_alpha}, post_ry_z_mult={post_ryz}."
                ),
                'stage': 'B',
                'stage_index': idx,
                'parent': base['name'],
                'scd': copy.deepcopy(base['scd']),
                'repair': {
                    'state20_alpha_mult': state20_alpha,
                    'post_ry_z_mult': post_ryz,
                    'scaling': float(v['scale']),
                    'pattern': v['variant'],
                    'knobs': [
                        k for k, enabled in [
                            ('iter2.state_alpha_mult[20]', v['state20']),
                            ('post_ry_z_mult', v['ryz']),
                        ]
                        if enabled
                    ],
                },
            })
    return out


def _result_json_path(compare_shared_mod, candidate_name: str, noise_scale: float) -> Path:
    suffix = compare_shared_mod.make_suffix(noise_scale)
    return RESULTS_DIR / f'G4_sym20_native_scd_round61_{candidate_name}_shared_{suffix}_param_errors.json'


def _apply_repairs_to_neutral_candidate(neutral_candidate: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    patched = copy.deepcopy(neutral_candidate)
    repair = candidate.get('repair', {})

    state20_alpha = repair.get('state20_alpha_mult')
    if state20_alpha is not None:
        iter2 = patched.setdefault('iter_patches', {}).setdefault(1, {})
        state_alpha_mult = copy.deepcopy(iter2.get('state_alpha_mult', {}))
        state_alpha_mult[20] = float(state20_alpha)
        iter2['state_alpha_mult'] = state_alpha_mult

    post_ryz = repair.get('post_ry_z_mult')
    if post_ryz is not None:
        patched['post_ry_z_mult'] = float(post_ryz)

    return patched


def _run_one_candidate(
    *,
    idx: int,
    candidate: dict[str, Any],
    args: argparse.Namespace,
    compare_shared_mod,
    probe_r55_mod,
    probe_r59_mod,
    source_mod,
    case: dict[str, Any],
    dataset: dict[str, Any],
    params,
    round61_knobs: dict[str, Any],
) -> dict[str, Any]:
    result_path = _result_json_path(compare_shared_mod, candidate['name'], args.noise_scale)
    expected_cfg = compare_shared_mod.expected_noise_config(args.noise_scale)
    signature = _candidate_signature(candidate)

    if (not args.force_rerun) and result_path.exists():
        old = _load_json(result_path)
        extra = old.get('extra', {}) if isinstance(old, dict) else {}
        if (
            compare_shared_mod._noise_matches(old, expected_cfg)
            and extra.get('comparison_mode') == COMPARISON_MODE
            and extra.get('path_case_tag') == case['case_tag']
            and extra.get('candidate_signature') == signature
        ):
            return {
                'candidate': copy.deepcopy(candidate),
                'overall': _overall_triplet(old),
                'result_json': str(result_path),
                'status': 'reused_verified',
            }

    method_mod = load_module(
        f"sym20_native_round61_r53_{idx}_{_sanitize_name(candidate['name'])}",
        str(R53_METHOD_FILE),
    )
    neutral_candidate = compare_shared_mod._build_neutral_scd_candidate()
    neutral_candidate['name'] = candidate['name']
    neutral_candidate['description'] = candidate['description']
    neutral_candidate['scd'] = copy.deepcopy(candidate['scd'])
    neutral_candidate = _apply_repairs_to_neutral_candidate(neutral_candidate, candidate)

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
        label=f"SYM20-NATIVE-R61-{idx:02d}-{_sanitize_name(candidate['name']).upper()}",
        scd_cfg=candidate['scd'],
    ))
    runtime = scd_result[4] if len(scd_result) >= 5 and isinstance(scd_result[4], dict) else {}

    payload = compare_shared_mod.compute_payload(
        source_mod,
        scd_result[0],
        params,
        variant=f"g4_sym20_native_scd_round61_{candidate['name']}_{compare_shared_mod.make_suffix(args.noise_scale)}",
        method_file='round53_base + _build_patched_method(neutral+tiny_repair) + _run_internalized_hybrid_scd',
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
            'rationale': case['rationale'],
            'noise_scale': args.noise_scale,
            'noise_config': dataset['noise_config'],
            'candidate_name': candidate['name'],
            'candidate_stage': candidate['stage'],
            'candidate_parent': candidate.get('parent'),
            'candidate_stage_index': candidate.get('stage_index'),
            'candidate_description': candidate['description'],
            'candidate_signature': signature,
            'scd_cfg': copy.deepcopy(candidate['scd']),
            'repair': copy.deepcopy(candidate.get('repair', {})),
            'iter_patches': copy.deepcopy(neutral_candidate.get('iter_patches', {})),
            'post_ry_z_mult': float(neutral_candidate.get('post_ry_z_mult', 1.0)),
            'round61_reference_knobs': copy.deepcopy(round61_knobs),
            'runtime_log': {
                'schedule_log': runtime.get('schedule_log'),
                'feedback_log': runtime.get('feedback_log'),
                'scd_log': runtime.get('scd_log'),
            },
            'baseline_g3_json': str(G3_BASELINE_JSON),
            'baseline_old_pure_json': str(OLD_PURE_BEST_JSON),
        },
    )
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    return {
        'candidate': copy.deepcopy(candidate),
        'overall': _overall_triplet(payload),
        'result_json': str(result_path),
        'status': 'rerun',
    }


def _metric_deltas(candidate_overall: dict[str, float], reference_overall: dict[str, float]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in METRIC_KEYS:
        cand_v = float(candidate_overall[key])
        ref_v = float(reference_overall[key])
        improve = ref_v - cand_v
        metrics[key] = {
            'candidate_value': cand_v,
            'reference_value': ref_v,
            'improvement_pct_points': improve,
            'candidate_better': bool(cand_v < ref_v),
            'remaining_gap_pct_points': float(max(0.0, cand_v - ref_v)),
        }

    beats_mean = bool(metrics['mean_pct_error']['candidate_better'])
    beats_median = bool(metrics['median_pct_error']['candidate_better'])
    beats_max = bool(metrics['max_pct_error']['candidate_better'])
    return {
        'beats_mean': beats_mean,
        'beats_median': beats_median,
        'beats_max': beats_max,
        'beats_mean_median': bool(beats_mean and beats_median),
        'beats_all_three': bool(beats_mean and beats_median and beats_max),
        'metrics': metrics,
    }


def _weighted_closeness_score(candidate_overall: dict[str, float], g3_overall: dict[str, float]) -> dict[str, float]:
    gaps = {
        key: float(max(0.0, float(candidate_overall[key]) - float(g3_overall[key])))
        for key in METRIC_KEYS
    }
    score = 4.0 * gaps['mean_pct_error'] + 3.0 * gaps['median_pct_error'] + 1.0 * gaps['max_pct_error']
    return {
        'weighted_gap_score': float(score),
        'gaps': gaps,
    }


def _rank_rows(rows: list[dict[str, Any]], g3_overall: dict[str, float]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        r = copy.deepcopy(row)
        r['delta_vs_g3'] = _metric_deltas(r['overall'], g3_overall)
        closeness = _weighted_closeness_score(r['overall'], g3_overall)
        r['weighted_gap_score'] = float(closeness['weighted_gap_score'])
        r['remaining_gaps_to_g3'] = closeness['gaps']
        prepared.append(r)

    winners = [r for r in prepared if r['delta_vs_g3']['beats_mean_median']]
    others = [r for r in prepared if not r['delta_vs_g3']['beats_mean_median']]

    winners = sorted(
        winners,
        key=lambda x: (
            float(x['overall']['max_pct_error']),
            float(x['overall']['mean_pct_error']),
            float(x['overall']['median_pct_error']),
        ),
    )
    others = sorted(
        others,
        key=lambda x: (
            float(x['weighted_gap_score']),
            float(x['remaining_gaps_to_g3']['mean_pct_error']),
            float(x['remaining_gaps_to_g3']['median_pct_error']),
            float(x['remaining_gaps_to_g3']['max_pct_error']),
            float(x['overall']['mean_pct_error']),
            float(x['overall']['median_pct_error']),
            float(x['overall']['max_pct_error']),
        ),
    )

    ranked = winners + others
    for i, row in enumerate(ranked, start=1):
        row['rank'] = i
        row['ranking_group'] = 'beats_mean_median' if row['delta_vs_g3']['beats_mean_median'] else 'closest_weighted_gap'
    return ranked


def _triplet_text(overall: dict[str, float]) -> str:
    return f"{overall['mean_pct_error']:.6f} / {overall['median_pct_error']:.6f} / {overall['max_pct_error']:.6f}"


def _render_report(summary: dict[str, Any]) -> str:
    g3 = summary['baselines']['g3']['overall']
    old_pure = summary['baselines']['old_pure_best']['overall']
    best = summary['best_candidate']
    best_new = summary['best_new_candidate_beyond_old_point']

    lines: list[str] = []
    lines.append('# Sym20-native SCD focused search (Round61-style tiny repairs)')
    lines.append('')
    lines.append('## Setup')
    lines.append('')
    lines.append(f"- Path: `{summary['setup']['path_case_tag']}`")
    lines.append(f"- att0: `{summary['setup']['att0_deg']}`")
    lines.append(f"- noise_scale: `{summary['noise_scale']}`")
    lines.append('- Fixed SCD constraints: target=selected, mode=once_per_phase, bias_to_target=False, apply_policy_names=[iter2_commit]')
    lines.append('')
    lines.append('## Baselines (mean / median / max, % error)')
    lines.append('')
    lines.append(f"- G3 Markov@20: **{_triplet_text(g3)}**")
    lines.append(f"- Old pure-SCD local best: **{_triplet_text(old_pure)}**")
    lines.append('')
    lines.append('## Round60/Round61 tiny knob extraction (code-inspected)')
    lines.append('')
    ref = summary['round61_reference_knobs']
    lines.append(f"- reference candidate: `{ref['reference_candidate_name']}`")
    lines.append(f"- iter2 state20 alpha mult old: `{ref['iter2_state20_alpha_mult_old']}` (neutral=1.0)")
    lines.append(f"- post_ry_z_mult old: `{ref['post_ry_z_mult_old']}` (neutral=1.0)")
    lines.append('')
    lines.append('## Stage A results (20 candidates)')
    lines.append('')
    lines.append('| rankA | name | alpha | td | mean | median | max | beats(mean/median/max) | weighted_gap |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|---|---:|')
    for row in summary['stageA']['ranking']:
        scd = row['candidate']['scd']
        b = row['delta_vs_g3']
        lines.append(
            f"| {row['rank']} | {row['candidate']['name']} | {scd['alpha']:.5f} | {scd['transition_duration']:.1f} | "
            f"{row['overall']['mean_pct_error']:.6f} | {row['overall']['median_pct_error']:.6f} | {row['overall']['max_pct_error']:.6f} | "
            f"{int(b['beats_mean'])}/{int(b['beats_median'])}/{int(b['beats_max'])} | {row['weighted_gap_score']:.6f} |"
        )
    lines.append('')
    lines.append('## Stage B results (tiny repair, 12 candidates)')
    lines.append('')
    lines.append('| rankB | name | parent | pattern | state20_mult | post_ry_z_mult | mean | median | max | beats(mean/median/max) | weighted_gap |')
    lines.append('|---:|---|---|---|---:|---:|---:|---:|---:|---|---:|')
    for row in summary['stageB']['ranking']:
        rep = row['candidate']['repair']
        b = row['delta_vs_g3']
        s20 = rep['state20_alpha_mult']
        ryz = rep['post_ry_z_mult']
        lines.append(
            f"| {row['rank']} | {row['candidate']['name']} | {row['candidate'].get('parent')} | {rep['pattern']} | "
            f"{(f'{s20:.6f}' if s20 is not None else '-')} | {(f'{ryz:.6f}' if ryz is not None else '-')} | "
            f"{row['overall']['mean_pct_error']:.6f} | {row['overall']['median_pct_error']:.6f} | {row['overall']['max_pct_error']:.6f} | "
            f"{int(b['beats_mean'])}/{int(b['beats_median'])}/{int(b['beats_max'])} | {row['weighted_gap_score']:.6f} |"
        )
    lines.append('')
    lines.append('## Final ranking (A+B together)')
    lines.append('')
    lines.append('| rank | stage | name | mean | median | max | Δmean vs G3 | Δmedian vs G3 | Δmax vs G3 | weighted_gap | group |')
    lines.append('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|')
    for row in summary['ranking']:
        d = row['delta_vs_g3']['metrics']
        lines.append(
            f"| {row['rank']} | {row['candidate']['stage']} | {row['candidate']['name']} | "
            f"{row['overall']['mean_pct_error']:.6f} | {row['overall']['median_pct_error']:.6f} | {row['overall']['max_pct_error']:.6f} | "
            f"{d['mean_pct_error']['improvement_pct_points']:+.6f} | {d['median_pct_error']['improvement_pct_points']:+.6f} | {d['max_pct_error']['improvement_pct_points']:+.6f} | "
            f"{row['weighted_gap_score']:.6f} | {row['ranking_group']} |"
        )
    lines.append('')
    lines.append('## Conclusion')
    lines.append('')
    lines.append(f"- Best candidate: **{best['candidate']['name']}** ({best['candidate']['stage']})")
    lines.append(f"  - overall: **{_triplet_text(best['overall'])}**")
    lines.append(
        f"  - vs G3: Δmean={best['delta_vs_g3']['metrics']['mean_pct_error']['improvement_pct_points']:+.6f}, "
        f"Δmedian={best['delta_vs_g3']['metrics']['median_pct_error']['improvement_pct_points']:+.6f}, "
        f"Δmax={best['delta_vs_g3']['metrics']['max_pct_error']['improvement_pct_points']:+.6f}"
    )
    lines.append(f"- Best NEW candidate beyond old pure point: **{best_new['candidate']['name']}** ({best_new['candidate']['stage']})")
    lines.append(f"  - overall: **{_triplet_text(best_new['overall'])}**")
    lines.append(
        f"  - vs G3: Δmean={best_new['delta_vs_g3']['metrics']['mean_pct_error']['improvement_pct_points']:+.6f}, "
        f"Δmedian={best_new['delta_vs_g3']['metrics']['median_pct_error']['improvement_pct_points']:+.6f}, "
        f"Δmax={best_new['delta_vs_g3']['metrics']['max_pct_error']['improvement_pct_points']:+.6f}"
    )
    lines.append(
        f"- Any candidate beats G3: mean={summary['beats_g3']['mean']}, median={summary['beats_g3']['median']}, "
        f"max={summary['beats_g3']['max']}, mean+median={summary['beats_g3']['mean_median']}, all_three={summary['beats_g3']['all_three']}"
    )
    if not summary['beats_g3']['mean_median']:
        gap = summary['nearest_remaining_gap_to_mean_median_win']
        lines.append(
            f"- Nearest remaining gap to mean+median win: `{gap['candidate_name']}` needs "
            f"mean_gap={gap['remaining_gap_pct_points']['mean_pct_error']:.6f}, "
            f"median_gap={gap['remaining_gap_pct_points']['median_pct_error']:.6f}, "
            f"current max_gap={gap['remaining_gap_pct_points']['max_pct_error']:.6f}."
        )
    lines.append(f"- Reproduction verdict: **{summary['reproduction']['label']}** — {summary['reproduction']['reason']}")
    lines.append('')
    lines.append('## Artifacts')
    lines.append('')
    lines.append(f"- script: `{summary['files']['script']}`")
    lines.append(f"- summary json: `{summary['files']['summary_json']}`")
    lines.append(f"- report md: `{summary['files']['report_md']}`")
    lines.append(f"- best result json: `{summary['files']['best_result_json']}`")
    lines.append(f"- best new result json: `{summary['files']['best_new_result_json']}`")
    lines.append('')

    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    compare_ch3_mod = load_module('sym20_native_round61_compare_ch3_mod', str(COMPARE_CH3_FILE))
    compare_shared_mod = load_module('sym20_native_round61_compare_shared_mod', str(COMPARE_SHARED_FILE))
    compute_r61_mod = load_module('sym20_native_round61_compute_r61_mod', str(COMPUTE_R61_FILE))
    probe_r55_mod = load_module('sym20_native_round61_probe_r55_mod', str(PROBE_R55_FILE))
    probe_r59_mod = load_module('sym20_native_round61_probe_r59_mod', str(PROBE_R59_FILE))

    source_mod = load_module(
        f"sym20_native_round61_source_{compare_shared_mod.make_suffix(args.noise_scale)}",
        str(SOURCE_FILE),
    )

    case = compare_ch3_mod.build_symmetric20_case(source_mod)
    dataset = compare_ch3_mod.build_dataset(source_mod, case['paras'], case['att0_deg'], args.noise_scale)
    params = compute_r61_mod._param_specs(source_mod)

    baseline_g3 = _load_json(G3_BASELINE_JSON)
    baseline_old_pure = _load_json(OLD_PURE_BEST_JSON)
    g3_overall = _overall_triplet(baseline_g3)
    old_pure_overall = _overall_triplet(baseline_old_pure)

    round61_knobs = _inspect_round61_reference_knobs()

    # Stage A
    stage_a_candidates = _build_stage_a_candidates()
    stage_a_rows: list[dict[str, Any]] = []
    for i, cand in enumerate(stage_a_candidates, start=1):
        stage_a_rows.append(_run_one_candidate(
            idx=i,
            candidate=cand,
            args=args,
            compare_shared_mod=compare_shared_mod,
            probe_r55_mod=probe_r55_mod,
            probe_r59_mod=probe_r59_mod,
            source_mod=source_mod,
            case=case,
            dataset=dataset,
            params=params,
            round61_knobs=round61_knobs,
        ))

    stage_a_ranked = _rank_rows(stage_a_rows, g3_overall)
    stage_a_top_for_b = [copy.deepcopy(r['candidate']) for r in stage_a_ranked[:2]]

    # Stage B
    stage_b_candidates = _build_stage_b_candidates(stage_a_top_for_b, round61_knobs)
    stage_b_rows: list[dict[str, Any]] = []
    for j, cand in enumerate(stage_b_candidates, start=1):
        stage_b_rows.append(_run_one_candidate(
            idx=100 + j,
            candidate=cand,
            args=args,
            compare_shared_mod=compare_shared_mod,
            probe_r55_mod=probe_r55_mod,
            probe_r59_mod=probe_r59_mod,
            source_mod=source_mod,
            case=case,
            dataset=dataset,
            params=params,
            round61_knobs=round61_knobs,
        ))

    stage_b_ranked = _rank_rows(stage_b_rows, g3_overall)

    all_rows = stage_a_rows + stage_b_rows
    ranked = _rank_rows(all_rows, g3_overall)

    for row in ranked:
        row['delta_vs_old_pure_best'] = _metric_deltas(row['overall'], old_pure_overall)
        row['is_old_pure_point'] = _is_old_pure_point(row['candidate'])

    best = copy.deepcopy(ranked[0])

    best_new = None
    for row in ranked:
        if not row['is_old_pure_point']:
            best_new = copy.deepcopy(row)
            break
    if best_new is None:
        best_new = copy.deepcopy(best)

    beats_g3 = {
        'mean': any(r['delta_vs_g3']['beats_mean'] for r in ranked),
        'median': any(r['delta_vs_g3']['beats_median'] for r in ranked),
        'max': any(r['delta_vs_g3']['beats_max'] for r in ranked),
        'mean_median': any(r['delta_vs_g3']['beats_mean_median'] for r in ranked),
        'all_three': any(r['delta_vs_g3']['beats_all_three'] for r in ranked),
    }

    def _mean_median_gap_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
        gaps = row['remaining_gaps_to_g3']
        mm = float(gaps['mean_pct_error']) + float(gaps['median_pct_error'])
        return (
            mm,
            float(gaps['mean_pct_error']),
            float(gaps['median_pct_error']),
            float(gaps['max_pct_error']),
        )

    nearest_mm_gap_row = sorted(ranked, key=_mean_median_gap_key)[0]
    nearest_remaining_gap_to_mean_median_win = {
        'candidate_name': nearest_mm_gap_row['candidate']['name'],
        'stage': nearest_mm_gap_row['candidate']['stage'],
        'rank': nearest_mm_gap_row['rank'],
        'remaining_gap_pct_points': copy.deepcopy(nearest_mm_gap_row['remaining_gaps_to_g3']),
        'weighted_gap_score': float(nearest_mm_gap_row['weighted_gap_score']),
        'delta_vs_g3': copy.deepcopy(nearest_mm_gap_row['delta_vs_g3']),
    }

    if beats_g3['mean_median']:
        reproduction = {
            'success': True,
            'label': 'reproduced',
            'reason': 'At least one candidate beats G3 on both mean and median under corrected symmetric20.',
        }
    else:
        reproduction = {
            'success': False,
            'label': 'not_reproduced_yet',
            'reason': (
                'No candidate beats G3 on mean+median simultaneously; best remaining '
                f"mean/median gap is {nearest_mm_gap_row['remaining_gaps_to_g3']['mean_pct_error']:.6f} / "
                f"{nearest_mm_gap_row['remaining_gaps_to_g3']['median_pct_error']:.6f}."
            ),
        }

    suffix = compare_shared_mod.make_suffix(args.noise_scale)
    summary_json = RESULTS_DIR / f'g4_sym20_native_scd_round61_search_{args.report_date}_{suffix}_summary.json'
    report_md = REPORTS_DIR / f'psins_sym20_native_scd_round61_search_{args.report_date}_{suffix}.md'

    stage_a_ranked_names = [row['candidate']['name'] for row in stage_a_ranked]
    stage_b_ranked_names = [row['candidate']['name'] for row in stage_b_ranked]

    summary = {
        'experiment': 'sym20_native_scd_round61_focused_search_2026_04_05',
        'comparison_mode': COMPARISON_MODE,
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'noise_config': dataset['noise_config'],
        'ranking_rule': {
            'primary_target': 'candidates beating G3 on mean and median',
            'within_primary_sort': 'smaller max_pct_error first',
            'fallback_sort': 'weighted gap score = 4*mean_gap + 3*median_gap + 1*max_gap',
        },
        'setup': {
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
            'fixed_scd_constraints': {
                'target': 'selected',
                'mode': 'once_per_phase',
                'bias_to_target': False,
                'apply_policy_names': ['iter2_commit'],
            },
        },
        'round61_reference_knobs': round61_knobs,
        'baselines': {
            'g3': {
                'json_path': str(G3_BASELINE_JSON),
                'overall': g3_overall,
            },
            'old_pure_best': {
                'json_path': str(OLD_PURE_BEST_JSON),
                'overall': old_pure_overall,
            },
        },
        'stageA': {
            'count': len(stage_a_rows),
            'grid': {
                'alpha': STAGE_A_ALPHAS,
                'transition_duration': STAGE_A_TDS,
                'target': ['selected'],
                'mode': ['once_per_phase'],
                'bias_to_target': [False],
                'apply_policy_names': [['iter2_commit']],
            },
            'selected_for_stageB': stage_a_top_for_b,
            'ranking_names': stage_a_ranked_names,
            'ranking': stage_a_ranked,
        },
        'stageB': {
            'count': len(stage_b_rows),
            'candidate_cap': 12,
            'parent_bases': stage_a_top_for_b,
            'ranking_names': stage_b_ranked_names,
            'ranking': stage_b_ranked,
        },
        'total_tried_candidates': len(ranked),
        'ranking': ranked,
        'best_candidate': best,
        'best_new_candidate_beyond_old_point': best_new,
        'beats_g3': beats_g3,
        'nearest_remaining_gap_to_mean_median_win': nearest_remaining_gap_to_mean_median_win,
        'reproduction': reproduction,
        'code_fixes_needed': 'none',
        'files': {
            'script': str(Path(__file__)),
            'summary_json': str(summary_json),
            'report_md': str(report_md),
            'best_result_json': best['result_json'],
            'best_new_result_json': best_new['result_json'],
            'all_result_jsons': [r['result_json'] for r in ranked],
        },
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_md.write_text(_render_report(summary), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps({
        'summary_json': str(summary_json),
        'report_md': str(report_md),
        'best_candidate': {
            'name': best['candidate']['name'],
            'stage': best['candidate']['stage'],
            'overall': best['overall'],
            'delta_vs_g3': best['delta_vs_g3'],
            'result_json': best['result_json'],
        },
        'best_new_candidate_beyond_old_point': {
            'name': best_new['candidate']['name'],
            'stage': best_new['candidate']['stage'],
            'overall': best_new['overall'],
            'delta_vs_g3': best_new['delta_vs_g3'],
            'result_json': best_new['result_json'],
        },
        'beats_g3': beats_g3,
        'nearest_remaining_gap_to_mean_median_win': nearest_remaining_gap_to_mean_median_win,
        'reproduction': reproduction,
        'code_fixes_needed': summary['code_fixes_needed'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
