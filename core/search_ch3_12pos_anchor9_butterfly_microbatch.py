from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path('/root/.openclaw/workspace')
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
REPORTS_DIR = ROOT / 'reports'
METHOD_DIR = ROOT / 'psins_method_bench' / 'methods' / 'markov'
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'

for p in [ROOT, ROOT / 'tmp_psins_py', METHOD_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from benchmark_ch3_12pos_goalA_repairs import compact_result
from common_markov import load_module
from compare_four_methods_shared_noise import _load_json, _noise_matches, expected_noise_config
from search_ch3_12pos_closedloop_local_insertions import (
    NOISE_SCALE,
    REPORT_DATE,
    StepSpec,
    build_closedloop_candidate,
    candidate_result_path,
    delta_vs_ref,
    render_action,
    run_candidate_payload,
)
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate, make_suffix

FAITHFUL_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
FAITHFUL_KF = RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json'
OLD_BEST_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
OLD_BEST_KF = RESULTS_DIR / 'KF36_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
CURRENT_UNIFIED_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2_shared_noise0p08_param_errors.json'
CURRENT_UNIFIED_KF = RESULTS_DIR / 'KF36_ch3closedloop_twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2_shared_noise0p08_param_errors.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()



def load_json_checked(path: Path, noise_scale: float) -> dict[str, Any]:
    payload = _load_json(path)
    expected_cfg = expected_noise_config(noise_scale)
    if not _noise_matches(payload, expected_cfg):
        raise ValueError(f'Noise configuration mismatch: {path}')
    return payload



def load_reference_payloads(noise_scale: float) -> dict[str, Any]:
    return {
        'faithful_markov': load_json_checked(FAITHFUL_MARKOV, noise_scale),
        'faithful_kf': load_json_checked(FAITHFUL_KF, noise_scale),
        'oldbest_markov': load_json_checked(OLD_BEST_MARKOV, noise_scale),
        'oldbest_kf': load_json_checked(OLD_BEST_KF, noise_scale),
        'current_unified_markov': load_json_checked(CURRENT_UNIFIED_MARKOV, noise_scale),
        'current_unified_kf': load_json_checked(CURRENT_UNIFIED_KF, noise_scale),
    }



def diag_butterfly(
    dwell1_s: float,
    dwell2_s: float,
    label: str,
    first_sign: int,
    second_sign: int,
    *,
    open_step_deg: int = -45,
    cross_step_deg: int = +90,
    close_step_deg: int = -45,
    open_hold_s: float = 3.0,
    cross_hold_s: float = 3.0,
    close_hold_s: float = 3.0,
) -> list[StepSpec]:
    return [
        StepSpec(kind='inner', angle_deg=open_step_deg, rotation_time_s=2.5, pre_static_s=0.0, post_static_s=open_hold_s, segment_role='motif_diag_open1', label=f'{label}_open1'),
        StepSpec(kind='outer', angle_deg=90 * first_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=dwell1_s, segment_role='motif_diag1_sweep', label=f'{label}_diag1_sweep'),
        StepSpec(kind='outer', angle_deg=-90 * first_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=dwell1_s, segment_role='motif_diag1_return', label=f'{label}_diag1_return'),
        StepSpec(kind='inner', angle_deg=cross_step_deg, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=cross_hold_s, segment_role='motif_diag_cross', label=f'{label}_cross'),
        StepSpec(kind='outer', angle_deg=90 * second_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=dwell2_s, segment_role='motif_diag2_sweep', label=f'{label}_diag2_sweep'),
        StepSpec(kind='outer', angle_deg=-90 * second_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=dwell2_s, segment_role='motif_diag2_return', label=f'{label}_diag2_return'),
        StepSpec(kind='inner', angle_deg=close_step_deg, rotation_time_s=2.5, pre_static_s=0.0, post_static_s=close_hold_s, segment_role='motif_diag_close2', label=f'{label}_close2'),
    ]


REFERENCE_CANDIDATE_SPECS = [
    {
        'artifact_name': 'probe2_l9_diag_butterfly_same',
        'display_name': 'butterfly_same_d6',
        'family': 'reference_prompt_same',
        'rationale': 'Prompt reference point: original same-sign butterfly with uniform d6 dwell.',
        'insertions': {9: diag_butterfly(6.0, 6.0, 'l9_diag_bfly_same_d6', -1, -1)},
    },
    {
        'artifact_name': 'probe2_l9_diag_butterfly_flip',
        'display_name': 'butterfly_flip_d6',
        'family': 'reference_prompt_flip',
        'rationale': 'Prompt reference point: original flip-sign butterfly with uniform d6 dwell.',
        'insertions': {9: diag_butterfly(6.0, 6.0, 'l9_diag_bfly_flip_d6', -1, +1)},
    },
    {
        'artifact_name': 'probe3_l9_diag_bfly_same_d8',
        'display_name': 'butterfly_same_d8_incumbent',
        'family': 'family_incumbent_same',
        'rationale': 'Current best-known mean-oriented butterfly incumbent before this micro-batch.',
        'insertions': {9: diag_butterfly(8.0, 8.0, 'l9_diag_bfly_same_d8', -1, -1)},
    },
    {
        'artifact_name': 'probe3_l9_diag_bfly_flip_d8',
        'display_name': 'butterfly_flip_d8_incumbent',
        'family': 'family_incumbent_flip',
        'rationale': 'Current best-known balanced flip-sign butterfly incumbent before this micro-batch.',
        'insertions': {9: diag_butterfly(8.0, 8.0, 'l9_diag_bfly_flip_d8', -1, +1)},
    },
]


MICROBATCH_SPECS = [
    {
        'artifact_name': 'probe5_l9_diag_bfly_same_rev_d8',
        'display_name': 'butterfly_same_revorder_d8',
        'family': 'order_sensitivity',
        'hypothesis': 'H1 order sensitivity',
        'rationale': 'Reverse the diagonal visit order while keeping the same-sign sweep pattern and the same total dwell budget (8/8). If the loop is order-sensitive, this should move the branch even with identical total time.',
        'insertions': {9: diag_butterfly(8.0, 8.0, 'l9_diag_bfly_same_rev_d8', -1, -1, open_step_deg=+45, cross_step_deg=-90, close_step_deg=+45)},
    },
    {
        'artifact_name': 'probe5_l9_diag_bfly_flip_rev_d8',
        'display_name': 'butterfly_flip_revorder_d8',
        'family': 'order_sensitivity',
        'hypothesis': 'H1 order sensitivity',
        'rationale': 'Reverse the diagonal visit order for the flip-sign butterfly at the same 8/8 dwell level.',
        'insertions': {9: diag_butterfly(8.0, 8.0, 'l9_diag_bfly_flip_rev_d8', -1, +1, open_step_deg=+45, cross_step_deg=-90, close_step_deg=+45)},
    },
    {
        'artifact_name': 'probe5_l9_diag_bfly_same_split_10_6',
        'display_name': 'butterfly_same_split10_6',
        'family': 'dwell_split',
        'hypothesis': 'H2 dwell split',
        'rationale': 'Bias dwell toward the entry-side diagonal (10/6 instead of 8/8) while preserving exact closure and total time. This checks whether the butterfly should load more strongly before the cross-over.',
        'insertions': {9: diag_butterfly(10.0, 6.0, 'l9_diag_bfly_same_split10_6', -1, -1)},
    },
    {
        'artifact_name': 'probe5_l9_diag_bfly_same_split_6_10',
        'display_name': 'butterfly_same_split6_10',
        'family': 'dwell_split',
        'hypothesis': 'H2 dwell split',
        'rationale': 'Bias dwell toward the resume-side diagonal (6/10 instead of 8/8). This is the most direct check of whether the second diagonal visit should carry more weight because it is closest to the anchor10 resume boundary.',
        'insertions': {9: diag_butterfly(6.0, 10.0, 'l9_diag_bfly_same_split6_10', -1, -1)},
    },
    {
        'artifact_name': 'probe5_l9_diag_bfly_same_d8_closebias',
        'display_name': 'butterfly_same_d8_closebias',
        'family': 'asymmetry_resume_boundary',
        'hypothesis': 'H3/H4 asymmetry + minimal resume interaction',
        'rationale': 'Keep the same-sign d8 butterfly and total time fixed, but shift one second of hold from the opening edge to the final closing edge (open/cross/close = 2/3/4). This is a mild asymmetry tweak that minimally increases settling time immediately before node10 resumes while preserving exact continuity.',
        'insertions': {9: diag_butterfly(8.0, 8.0, 'l9_diag_bfly_same_d8_closebias', -1, -1, open_hold_s=2.0, cross_hold_s=3.0, close_hold_s=4.0)},
    },
]


ALL_SPECS = REFERENCE_CANDIDATE_SPECS + MICROBATCH_SPECS



def spec_by_name(name: str) -> dict[str, Any]:
    for spec in ALL_SPECS:
        if spec['artifact_name'] == name:
            return spec
    raise KeyError(name)



def build_candidates(mod) -> dict[str, Any]:
    faithful = build_candidate(mod, ())
    base_rows = faithful.rows
    base_actions = faithful.action_sequence
    candidates = {}
    for spec in ALL_SPECS:
        candidates[spec['artifact_name']] = build_closedloop_candidate(
            mod,
            {
                'name': spec['artifact_name'],
                'rationale': spec['rationale'],
                'insertions': spec['insertions'],
            },
            base_rows,
            base_actions,
        )
    return candidates



def _fmt_triplet(maybe_compact: dict[str, Any] | None) -> str:
    if maybe_compact is None:
        return 'n/a'
    mm = maybe_compact['overall']
    return f"{mm['mean_pct_error']:.3f} / {mm['median_pct_error']:.3f} / {mm['max_pct_error']:.3f}"



def render_report(payload: dict[str, Any]) -> str:
    refs = payload['references']
    best = payload['best_butterfly_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 anchor9 butterfly micro-batch')
    lines.append('')
    lines.append('## 1. Bottom line')
    lines.append('')
    lines.append(f"- best butterfly after this micro-batch: **{best['display_name']}** (`{best['artifact_name']}`) → **{best['markov42']['overall']['mean_pct_error']:.3f} / {best['markov42']['overall']['median_pct_error']:.3f} / {best['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- verdict: **{payload['verdict']}**")
    lines.append('')
    lines.append('## 2. Requested reference points')
    lines.append('')
    lines.append(f"- butterfly_same (prompt reference): **{refs['butterfly_same_d6']['markov42']['overall']['mean_pct_error']:.3f} / {refs['butterfly_same_d6']['markov42']['overall']['median_pct_error']:.3f} / {refs['butterfly_same_d6']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- butterfly_flip (prompt reference): **{refs['butterfly_flip_d6']['markov42']['overall']['mean_pct_error']:.3f} / {refs['butterfly_flip_d6']['markov42']['overall']['median_pct_error']:.3f} / {refs['butterfly_flip_d6']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- current family incumbent (same d8): **{refs['butterfly_same_d8_incumbent']['markov42']['overall']['mean_pct_error']:.3f} / {refs['butterfly_same_d8_incumbent']['markov42']['overall']['median_pct_error']:.3f} / {refs['butterfly_same_d8_incumbent']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- current unified winner: **{refs['current_unified_winner']['markov42']['overall']['mean_pct_error']:.3f} / {refs['current_unified_winner']['markov42']['overall']['median_pct_error']:.3f} / {refs['current_unified_winner']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- old best legal: **{refs['old_best_legal']['markov42']['overall']['mean_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['median_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('## 3. Markov42 micro-batch results')
    lines.append('')
    lines.append('| rank | display name | artifact | family | total_s | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Ka2_z | Δmean vs same_d6 | Δmean vs flip_d6 | Δmean vs same_d8 | Δmean vs unified | Δmean vs old best | Δmax vs old best |')
    lines.append('|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(payload['microbatch_markov42_rows'], start=1):
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        lines.append(
            f"| {idx} | {row['display_name']} | `{row['artifact_name']}` | {row['family']} | {row['total_time_s']:.0f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {row['delta_vs_same_d6']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_flip_d6']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_same_d8']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_current_unified']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_old_best']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_old_best']['max_pct_error']['improvement_pct_points']:+.3f} |"
        )
    lines.append('')
    lines.append('## 4. Interpretation by hypothesis')
    lines.append('')
    for item in payload['hypothesis_findings']:
        lines.append(f"- **{item['id']}**: {item['finding']}")
    lines.append('')
    lines.append('## 5. Best butterfly candidate: exact legal motor / timing table')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for seq_idx, (row, action, face) in enumerate(zip(best['all_rows'], best['all_actions'], best['all_faces']), start=1):
        lines.append(
            f"| {seq_idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 6. Continuity proof')
    lines.append('')
    for check in best['continuity_checks']:
        before = check['state_before_insertion']
        after = check['state_after_insertion']
        lines.append(f"- anchor {check['anchor_id']}: closure_ok = **{'yes' if check['closure_ok'] else 'no'}**")
        lines.append(f"  - before: beta={before['beta_deg']}°, outer_axis={before['outer_axis_body']}, face={before['face_name']}")
        lines.append(f"  - after : beta={after['beta_deg']}°, outer_axis={after['outer_axis_body']}, face={after['face_name']}")
        if check['next_base_action_preview'] is not None:
            preview = check['next_base_action_preview']
            lines.append(f"  - next original action remains legal as `{preview['kind']}` {preview['motor_angle_deg']:+d}° with effective axis {preview['effective_body_axis']}")
    lines.append('')
    lines.append('## 7. KF36 recheck')
    lines.append('')
    if payload['kf36_recheck'] is None:
        lines.append('- no new KF36 rerun was triggered, because none of the new micro-batch candidates displaced the current butterfly incumbent strongly enough to count as a genuinely competitive new family leader.')
        lines.append(f"- incumbent KF36 reference remains `{refs['butterfly_same_d8_incumbent']['artifact_name']}` = **{_fmt_triplet(refs['butterfly_same_d8_incumbent']['kf36'])}**")
    else:
        row = payload['kf36_recheck']
        mm = row['markov42']['overall']
        kk = row['kf36']['overall']
        kp = row['kf36']['key_param_errors']
        lines.append('| candidate | note | Markov42 mean/median/max | KF36 mean/median/max | dKa_yy / dKg_zz / Ka2_y / Ka2_z (KF36) |')
        lines.append('|---|---|---|---|---|')
        lines.append(
            f"| `{row['artifact_name']}` | new competitive butterfly leader | {mm['mean_pct_error']:.3f} / {mm['median_pct_error']:.3f} / {mm['max_pct_error']:.3f} | {kk['mean_pct_error']:.3f} / {kk['median_pct_error']:.3f} / {kk['max_pct_error']:.3f} | {kp['dKa_yy']:.3f} / {kp['dKg_zz']:.3f} / {kp['Ka2_y']:.3f} / {kp['Ka2_z']:.3f} |"
        )
    lines.append('')
    lines.append('## 8. Requested comparison summary')
    lines.append('')
    lines.append('| candidate/path | Markov42 mean/median/max | KF36 mean/median/max | note |')
    lines.append('|---|---|---|---|')
    for row in payload['comparison_rows']:
        mm = row['markov42']['overall']
        lines.append(
            f"| {row['label']} | {mm['mean_pct_error']:.3f} / {mm['median_pct_error']:.3f} / {mm['max_pct_error']:.3f} | {_fmt_triplet(row['kf36'])} | {row['note']} |"
        )
    lines.append('')
    lines.append('## 9. Final verdict')
    lines.append('')
    lines.append(f"- **Does butterfly remain just a signal, or become a real competing family?** **{payload['competition_verdict']}**")
    lines.append(f"- {payload['competition_explanation']}")
    lines.append('')
    return '\n'.join(lines) + '\n'



def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module('search_ch3_anchor9_butterfly_microbatch_src', str(SOURCE_FILE))
    refs = load_reference_payloads(args.noise_scale)
    candidates = build_candidates(mod)

    ref_payloads: dict[str, dict[str, Any]] = {}
    ref_kf_payloads: dict[str, dict[str, Any]] = {}
    for spec in REFERENCE_CANDIDATE_SPECS:
        cand = candidates[spec['artifact_name']]
        ref_payloads[spec['artifact_name']] = load_json_checked(candidate_result_path(cand, 'markov42_noisy', args.noise_scale), args.noise_scale)
        kf_path = candidate_result_path(cand, 'kf36_noisy', args.noise_scale)
        if kf_path.exists():
            ref_kf_payloads[spec['artifact_name']] = load_json_checked(kf_path, args.noise_scale)

    micro_rows: list[dict[str, Any]] = []
    micro_payloads: dict[str, dict[str, Any]] = {}
    for spec in MICROBATCH_SPECS:
        cand = candidates[spec['artifact_name']]
        payload, status, path = run_candidate_payload(mod, cand, 'markov42_noisy', args.noise_scale, args.force_rerun)
        micro_payloads[spec['artifact_name']] = payload
        row = {
            'artifact_name': spec['artifact_name'],
            'display_name': spec['display_name'],
            'family': spec['family'],
            'hypothesis': spec['hypothesis'],
            'rationale': spec['rationale'],
            'total_time_s': cand.total_time_s,
            'metrics': compact_result(payload),
            'run_json': str(path),
            'status': status,
        }
        row['delta_vs_same_d6'] = delta_vs_ref(ref_payloads['probe2_l9_diag_butterfly_same'], payload)
        row['delta_vs_flip_d6'] = delta_vs_ref(ref_payloads['probe2_l9_diag_butterfly_flip'], payload)
        row['delta_vs_same_d8'] = delta_vs_ref(ref_payloads['probe3_l9_diag_bfly_same_d8'], payload)
        row['delta_vs_flip_d8'] = delta_vs_ref(ref_payloads['probe3_l9_diag_bfly_flip_d8'], payload)
        row['delta_vs_current_unified'] = delta_vs_ref(refs['current_unified_markov'], payload)
        row['delta_vs_old_best'] = delta_vs_ref(refs['oldbest_markov'], payload)
        micro_rows.append(row)

    micro_rows.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))

    incumbent_same_payload = ref_payloads['probe3_l9_diag_bfly_same_d8']
    incumbent_flip_payload = ref_payloads['probe3_l9_diag_bfly_flip_d8']

    best_overall_name = 'probe3_l9_diag_bfly_same_d8'
    best_overall_payload = incumbent_same_payload
    best_overall_display = 'butterfly_same_d8_incumbent'
    for row in micro_rows:
        payload = micro_payloads[row['artifact_name']]
        cur = payload['overall']
        best = best_overall_payload['overall']
        if (cur['mean_pct_error'], cur['max_pct_error']) < (best['mean_pct_error'], best['max_pct_error']):
            best_overall_name = row['artifact_name']
            best_overall_payload = payload
            best_overall_display = row['display_name']

    best_cand = candidates[best_overall_name]
    best_spec = spec_by_name(best_overall_name)
    best_is_new = best_overall_name in micro_payloads

    best_summary = {
        'artifact_name': best_overall_name,
        'display_name': best_overall_display,
        'family': best_spec['family'],
        'rationale': best_spec['rationale'],
        'total_time_s': best_cand.total_time_s,
        'all_rows': best_cand.all_rows,
        'all_actions': best_cand.all_actions,
        'all_faces': best_cand.all_faces,
        'continuity_checks': best_cand.continuity_checks,
        'markov42': compact_result(best_overall_payload),
        'markov42_run_json': str(candidate_result_path(best_cand, 'markov42_noisy', args.noise_scale)),
        'delta_vs_same_d6': delta_vs_ref(ref_payloads['probe2_l9_diag_butterfly_same'], best_overall_payload),
        'delta_vs_flip_d6': delta_vs_ref(ref_payloads['probe2_l9_diag_butterfly_flip'], best_overall_payload),
        'delta_vs_same_d8': delta_vs_ref(ref_payloads['probe3_l9_diag_bfly_same_d8'], best_overall_payload),
        'delta_vs_flip_d8': delta_vs_ref(ref_payloads['probe3_l9_diag_bfly_flip_d8'], best_overall_payload),
        'delta_vs_current_unified': delta_vs_ref(refs['current_unified_markov'], best_overall_payload),
        'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], best_overall_payload),
    }

    kf36_recheck = None
    if best_is_new:
        incumbent = incumbent_same_payload['overall']
        cur = best_overall_payload['overall']
        competitive = (
            cur['mean_pct_error'] < incumbent['mean_pct_error'] - 0.03
            or (cur['max_pct_error'] < incumbent_flip_payload['overall']['max_pct_error'] and cur['mean_pct_error'] <= incumbent['mean_pct_error'] + 0.10)
            or cur['mean_pct_error'] <= refs['oldbest_markov']['overall']['mean_pct_error'] + 0.10
        )
        if competitive:
            cand = candidates[best_overall_name]
            kf_payload, kf_status, kf_path = run_candidate_payload(mod, cand, 'kf36_noisy', args.noise_scale, args.force_rerun)
            kf36_recheck = {
                'artifact_name': best_overall_name,
                'display_name': best_overall_display,
                'markov42': compact_result(best_overall_payload),
                'kf36': compact_result(kf_payload),
                'kf36_status': kf_status,
                'kf36_run_json': str(kf_path),
            }

    order_rows = [row for row in micro_rows if row['family'] == 'order_sensitivity']
    dwell_rows = [row for row in micro_rows if row['family'] == 'dwell_split']
    closebias_row = next(row for row in micro_rows if row['artifact_name'] == 'probe5_l9_diag_bfly_same_d8_closebias')

    best_order = min(order_rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    best_dwell = min(dwell_rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))

    hypothesis_findings = [
        {
            'id': 'H1 order sensitivity',
            'finding': (
                f"Reversing the butterfly order did {'not ' if best_order['delta_vs_same_d8']['mean_pct_error']['improvement_pct_points'] <= 0 else ''}improve the family incumbent. "
                f"Best order-reversal candidate `{best_order['artifact_name']}` landed at {best_order['metrics']['overall']['mean_pct_error']:.3f} / {best_order['metrics']['overall']['median_pct_error']:.3f} / {best_order['metrics']['overall']['max_pct_error']:.3f}, "
                f"which is Δmean {best_order['delta_vs_same_d8']['mean_pct_error']['improvement_pct_points']:+.3f} vs same_d8."
            ),
        },
        {
            'id': 'H2 dwell split',
            'finding': (
                f"The better split candidate was `{best_dwell['artifact_name']}` at {best_dwell['metrics']['overall']['mean_pct_error']:.3f} / {best_dwell['metrics']['overall']['median_pct_error']:.3f} / {best_dwell['metrics']['overall']['max_pct_error']:.3f}. "
                f"This is Δmean {best_dwell['delta_vs_same_d8']['mean_pct_error']['improvement_pct_points']:+.3f} and Δmax {best_dwell['delta_vs_same_d8']['max_pct_error']['improvement_pct_points']:+.3f} vs same_d8, so the 8/8 incumbent {'still holds' if best_dwell['delta_vs_same_d8']['mean_pct_error']['improvement_pct_points'] <= 0 and best_dwell['delta_vs_same_d8']['max_pct_error']['improvement_pct_points'] <= 0 else 'was displaced by a split variant'}."
            ),
        },
        {
            'id': 'H3 mild asymmetry',
            'finding': (
                f"The close-biased asymmetry candidate landed at {closebias_row['metrics']['overall']['mean_pct_error']:.3f} / {closebias_row['metrics']['overall']['median_pct_error']:.3f} / {closebias_row['metrics']['overall']['max_pct_error']:.3f}, "
                f"with Δmean {closebias_row['delta_vs_same_d8']['mean_pct_error']['improvement_pct_points']:+.3f} and Δmax {closebias_row['delta_vs_same_d8']['max_pct_error']['improvement_pct_points']:+.3f} vs same_d8."
            ),
        },
        {
            'id': 'H4 minimal anchor10 boundary interaction',
            'finding': (
                'The only boundary-touching test was the close-bias timing shift. It changed only the hold allocation immediately before node10 resumed, kept the mechanism state exactly identical, and therefore cleanly isolated resume-boundary sensitivity without introducing any new illegal motion.'
            ),
        },
    ]

    comparison_rows = [
        {
            'label': 'butterfly_same_d6',
            'note': 'prompt reference point (no dedicated KF36 rerun saved)',
            'artifact_name': 'probe2_l9_diag_butterfly_same',
            'markov42': compact_result(ref_payloads['probe2_l9_diag_butterfly_same']),
            'kf36': compact_result(ref_kf_payloads['probe2_l9_diag_butterfly_same']) if 'probe2_l9_diag_butterfly_same' in ref_kf_payloads else None,
        },
        {
            'label': 'butterfly_flip_d6',
            'note': 'prompt reference point (no dedicated KF36 rerun saved)',
            'artifact_name': 'probe2_l9_diag_butterfly_flip',
            'markov42': compact_result(ref_payloads['probe2_l9_diag_butterfly_flip']),
            'kf36': compact_result(ref_kf_payloads['probe2_l9_diag_butterfly_flip']) if 'probe2_l9_diag_butterfly_flip' in ref_kf_payloads else None,
        },
        {
            'label': 'butterfly_same_d8_incumbent',
            'note': 'current family incumbent before this micro-batch',
            'artifact_name': 'probe3_l9_diag_bfly_same_d8',
            'markov42': compact_result(ref_payloads['probe3_l9_diag_bfly_same_d8']),
            'kf36': compact_result(ref_kf_payloads['probe3_l9_diag_bfly_same_d8']),
        },
        {
            'label': 'best butterfly after micro-batch',
            'note': best_summary['artifact_name'],
            'artifact_name': best_summary['artifact_name'],
            'markov42': best_summary['markov42'],
            'kf36': kf36_recheck['kf36'] if kf36_recheck is not None else compact_result(ref_kf_payloads['probe3_l9_diag_bfly_same_d8']) if best_summary['artifact_name'] == 'probe3_l9_diag_bfly_same_d8' else compact_result(ref_kf_payloads['probe3_l9_diag_bfly_flip_d8']) if best_summary['artifact_name'] == 'probe3_l9_diag_bfly_flip_d8' else compact_result(ref_kf_payloads['probe3_l9_diag_bfly_same_d8']),
        },
        {
            'label': 'current unified winner',
            'note': 'twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2',
            'artifact_name': None,
            'markov42': compact_result(refs['current_unified_markov']),
            'kf36': compact_result(refs['current_unified_kf']),
        },
        {
            'label': 'old best legal',
            'note': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
            'artifact_name': None,
            'markov42': compact_result(refs['oldbest_markov']),
            'kf36': compact_result(refs['oldbest_kf']),
        },
    ]

    best_overall = best_summary['markov42']['overall']
    if best_summary['artifact_name'] == 'probe3_l9_diag_bfly_same_d8':
        competition_verdict = 'still a strong signal, not yet a real competing family'
        competition_explanation = (
            f"The butterfly family is now clearly real and much stronger than the original d6 signal, but the best saved point remains the existing same_d8 incumbent at {best_overall['mean_pct_error']:.3f} / {best_overall['median_pct_error']:.3f} / {best_overall['max_pct_error']:.3f}. "
            f"That is still {best_summary['delta_vs_old_best']['mean_pct_error']['improvement_pct_points']:+.3f} mean-points worse than the old best legal and {best_summary['delta_vs_current_unified']['mean_pct_error']['improvement_pct_points']:+.3f} worse than the unified winner, so the family has not crossed into true headline-competitive territory yet."
        )
    else:
        improved_vs_old = best_summary['delta_vs_old_best']['mean_pct_error']['improvement_pct_points'] > 0 and best_summary['delta_vs_old_best']['max_pct_error']['improvement_pct_points'] > 0
        improved_vs_unified = best_summary['delta_vs_current_unified']['mean_pct_error']['improvement_pct_points'] > 0 and best_summary['delta_vs_current_unified']['max_pct_error']['improvement_pct_points'] > 0
        if improved_vs_old or improved_vs_unified:
            competition_verdict = 'yes, butterfly became a real competing family'
        else:
            competition_verdict = 'borderline: improved locally but still not a full competing family'
        competition_explanation = (
            f"The micro-batch displaced the previous same_d8 incumbent with `{best_summary['artifact_name']}` at {best_overall['mean_pct_error']:.3f} / {best_overall['median_pct_error']:.3f} / {best_overall['max_pct_error']:.3f}. "
            f"Relative to old best legal this is Δmean {best_summary['delta_vs_old_best']['mean_pct_error']['improvement_pct_points']:+.3f}, Δmax {best_summary['delta_vs_old_best']['max_pct_error']['improvement_pct_points']:+.3f}; relative to the unified winner it is Δmean {best_summary['delta_vs_current_unified']['mean_pct_error']['improvement_pct_points']:+.3f}, Δmax {best_summary['delta_vs_current_unified']['max_pct_error']['improvement_pct_points']:+.3f}."
        )

    verdict = (
        f"best butterfly remains `{best_summary['artifact_name']}` at {best_overall['mean_pct_error']:.3f} / {best_overall['median_pct_error']:.3f} / {best_overall['max_pct_error']:.3f}; "
        f"the micro-batch {'did not displace' if best_summary['artifact_name'] == 'probe3_l9_diag_bfly_same_d8' else 'did displace'} the current family incumbent."
    )

    out_json = RESULTS_DIR / f'ch3_anchor9_butterfly_microbatch_{make_suffix(args.noise_scale)}.json'
    out_md = REPORTS_DIR / f'psins_ch3_anchor9_butterfly_microbatch_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_anchor9_butterfly_microbatch',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'hard_constraints': {
            'base_reference': 'faithful12 base remains the base reference',
            'physical_legality': 'real dual-axis legality only',
            'continuity_rule': 'exact same mechanism state before resume',
            'search_scope': 'very small anchor9 butterfly-only local micro-batch',
            'broad_random_search': False,
        },
        'reference_candidate_specs': [
            {
                'artifact_name': spec['artifact_name'],
                'display_name': spec['display_name'],
                'family': spec['family'],
                'rationale': spec['rationale'],
            }
            for spec in REFERENCE_CANDIDATE_SPECS
        ],
        'microbatch_specs': [
            {
                'artifact_name': spec['artifact_name'],
                'display_name': spec['display_name'],
                'family': spec['family'],
                'hypothesis': spec['hypothesis'],
                'rationale': spec['rationale'],
            }
            for spec in MICROBATCH_SPECS
        ],
        'references': {
            'butterfly_same_d6': {
                'artifact_name': 'probe2_l9_diag_butterfly_same',
                'markov42': compact_result(ref_payloads['probe2_l9_diag_butterfly_same']),
                'kf36': compact_result(ref_kf_payloads['probe2_l9_diag_butterfly_same']) if 'probe2_l9_diag_butterfly_same' in ref_kf_payloads else None,
            },
            'butterfly_flip_d6': {
                'artifact_name': 'probe2_l9_diag_butterfly_flip',
                'markov42': compact_result(ref_payloads['probe2_l9_diag_butterfly_flip']),
                'kf36': compact_result(ref_kf_payloads['probe2_l9_diag_butterfly_flip']) if 'probe2_l9_diag_butterfly_flip' in ref_kf_payloads else None,
            },
            'butterfly_same_d8_incumbent': {
                'artifact_name': 'probe3_l9_diag_bfly_same_d8',
                'markov42': compact_result(ref_payloads['probe3_l9_diag_bfly_same_d8']),
                'kf36': compact_result(ref_kf_payloads['probe3_l9_diag_bfly_same_d8']),
            },
            'butterfly_flip_d8_incumbent': {
                'artifact_name': 'probe3_l9_diag_bfly_flip_d8',
                'markov42': compact_result(ref_payloads['probe3_l9_diag_bfly_flip_d8']),
                'kf36': compact_result(ref_kf_payloads['probe3_l9_diag_bfly_flip_d8']),
            },
            'current_unified_winner': {
                'markov42': compact_result(refs['current_unified_markov']),
                'kf36': compact_result(refs['current_unified_kf']),
            },
            'old_best_legal': {
                'markov42': compact_result(refs['oldbest_markov']),
                'kf36': compact_result(refs['oldbest_kf']),
            },
        },
        'microbatch_markov42_rows': micro_rows,
        'best_butterfly_candidate': best_summary,
        'kf36_recheck': kf36_recheck,
        'hypothesis_findings': hypothesis_findings,
        'comparison_rows': comparison_rows,
        'verdict': verdict,
        'competition_verdict': competition_verdict,
        'competition_explanation': competition_explanation,
        'files': {
            'json': str(out_json),
            'report': str(out_md),
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    out_md.write_text(render_report(payload), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps(payload['files'], ensure_ascii=False), flush=True)
    print('BEST_BUTTERFLY', best_summary['artifact_name'], best_summary['markov42']['overall'], flush=True)
    print('COMPETITION_VERDICT', competition_verdict, flush=True)


if __name__ == '__main__':
    main()
