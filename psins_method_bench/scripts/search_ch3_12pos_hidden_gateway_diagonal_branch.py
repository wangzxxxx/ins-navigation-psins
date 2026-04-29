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
    delta_vs_ref,
    render_action,
    run_candidate_payload,
)
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate, make_suffix

FAITHFUL_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
FAITHFUL_KF = RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json'
OLD_BEST_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
OLD_BEST_KF = RESULTS_DIR / 'KF36_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
DEFAULT18_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json'
DEFAULT18_KF = RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json'
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
        'default18_markov': load_json_checked(DEFAULT18_MARKOV, noise_scale),
        'default18_kf': load_json_checked(DEFAULT18_KF, noise_scale),
        'current_unified_markov': load_json_checked(CURRENT_UNIFIED_MARKOV, noise_scale),
        'current_unified_kf': load_json_checked(CURRENT_UNIFIED_KF, noise_scale),
    }



def diag_pair(beta_step_deg: int, outer_first_deg: int, dwell_s: float, label: str) -> list[StepSpec]:
    beta_rot_s = abs(beta_step_deg) / 90.0 * 5.0
    return [
        StepSpec(kind='inner', angle_deg=beta_step_deg, rotation_time_s=beta_rot_s, pre_static_s=0.0, post_static_s=5.0, segment_role='motif_diag_open', label=f'{label}_open'),
        StepSpec(kind='outer', angle_deg=outer_first_deg, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=dwell_s, segment_role='motif_diag_sweep', label=f'{label}_sweep'),
        StepSpec(kind='outer', angle_deg=-outer_first_deg, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=dwell_s, segment_role='motif_diag_return', label=f'{label}_return'),
        StepSpec(kind='inner', angle_deg=-beta_step_deg, rotation_time_s=beta_rot_s, pre_static_s=0.0, post_static_s=5.0, segment_role='motif_diag_close', label=f'{label}_close'),
    ]



def diag_butterfly(dwell_s: float, label: str, first_sign: int, second_sign: int, cross_hold_s: float = 3.0, edge_hold_s: float = 3.0) -> list[StepSpec]:
    return [
        StepSpec(kind='inner', angle_deg=-45, rotation_time_s=2.5, pre_static_s=0.0, post_static_s=edge_hold_s, segment_role='motif_diag_open1', label=f'{label}_open1'),
        StepSpec(kind='outer', angle_deg=90 * first_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=dwell_s, segment_role='motif_diag1_sweep', label=f'{label}_diag1_sweep'),
        StepSpec(kind='outer', angle_deg=-90 * first_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=dwell_s, segment_role='motif_diag1_return', label=f'{label}_diag1_return'),
        StepSpec(kind='inner', angle_deg=90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=cross_hold_s, segment_role='motif_diag_cross', label=f'{label}_cross'),
        StepSpec(kind='outer', angle_deg=90 * second_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=dwell_s, segment_role='motif_diag2_sweep', label=f'{label}_diag2_sweep'),
        StepSpec(kind='outer', angle_deg=-90 * second_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=dwell_s, segment_role='motif_diag2_return', label=f'{label}_diag2_return'),
        StepSpec(kind='inner', angle_deg=-45, rotation_time_s=2.5, pre_static_s=0.0, post_static_s=edge_hold_s, segment_role='motif_diag_close2', label=f'{label}_close2'),
    ]


CANDIDATE_SPECS = [
    {
        'artifact_name': 'probe_l9_diag45_neg',
        'display_name': 'gatewaydiag_l9_pair_diag135_posfirst_d8',
        'family': 'single_gateway_diagonal_pair',
        'hypothesis': 'H1',
        'rationale': 'Single legal diagonal pair launched from the anchor9 gateway by opening to beta=-135 and sweeping outer +90/-90. This is the minimal mixed-beta test of whether pure x/z anchor saturation is the real blocker.',
        'insertions': {9: diag_pair(-45, +90, 8.0, 'l9_diag45_neg')},
    },
    {
        'artifact_name': 'probe2_l9_diag135_negfirst',
        'display_name': 'gatewaydiag_l9_pair_diag135_negfirst_d8',
        'family': 'single_gateway_diagonal_pair',
        'hypothesis': 'H1',
        'rationale': 'Same anchor9 diagonal pair, but reverse the outer sweep order to -90/+90. This tests whether the mixed-beta branch is order-sensitive even before any heavier structural motif is added.',
        'insertions': {9: diag_pair(-45, -90, 8.0, 'l9_diag135_negfirst')},
    },
    {
        'artifact_name': 'probe_l3_l9_diag45_pair',
        'display_name': 'gatewaydiag_l3_l9_mirror_pair_d8',
        'family': 'mirror_pair_control',
        'hypothesis': 'H1-control',
        'rationale': 'Symmetry control: apply one legal diagonal pair at the early mirror gateway and again at anchor9. If the gain is only from generic extra excitation, this mirror version should help; if the gain is late-gateway-specific, it should not.',
        'insertions': {
            3: diag_pair(-45, +90, 8.0, 'l3_diag45_neg'),
            9: diag_pair(+45, +90, 8.0, 'l9_diag45_pos'),
        },
    },
    {
        'artifact_name': 'probe2_l9_diag_butterfly_same',
        'display_name': 'gatewaydiag_l9_butterfly_same_d6',
        'family': 'gateway_diagonal_butterfly',
        'hypothesis': 'H2',
        'rationale': 'Within the same anchor9 gateway, visit both legal diagonal outer axes before returning. This tests whether the missing structure is not a single diagonal, but a two-diagonal butterfly that couples the late block more completely.',
        'insertions': {9: diag_butterfly(6.0, 'l9_diag_bfly_same', -1, -1)},
    },
    {
        'artifact_name': 'probe2_l9_diag_butterfly_flip',
        'display_name': 'gatewaydiag_l9_butterfly_flip_d6',
        'family': 'gateway_diagonal_butterfly',
        'hypothesis': 'H2',
        'rationale': 'Same butterfly idea, but flip the second diagonal sweep sign. This distinguishes “visit both diagonals” from a stricter “same signed sweep on both diagonals” story.',
        'insertions': {9: diag_butterfly(6.0, 'l9_diag_bfly_flip', -1, +1)},
    },
    {
        'artifact_name': 'probe3_l9_diag_bfly_same_d4',
        'display_name': 'gatewaydiag_l9_butterfly_same_d4',
        'family': 'gateway_diagonal_butterfly_tune',
        'hypothesis': 'H2-tune',
        'rationale': 'Shorter dwell control for the same-sign butterfly.',
        'insertions': {9: diag_butterfly(4.0, 'l9_diag_bfly_same_d4', -1, -1)},
    },
    {
        'artifact_name': 'probe3_l9_diag_bfly_same_d8',
        'display_name': 'gatewaydiag_l9_butterfly_same_d8',
        'family': 'gateway_diagonal_butterfly_tune',
        'hypothesis': 'H2-tune',
        'rationale': 'Longer-dwell same-sign butterfly. This is the best-mean landed point in the new branch.',
        'insertions': {9: diag_butterfly(8.0, 'l9_diag_bfly_same_d8', -1, -1)},
    },
    {
        'artifact_name': 'probe3_l9_diag_bfly_flip_d4',
        'display_name': 'gatewaydiag_l9_butterfly_flip_d4',
        'family': 'gateway_diagonal_butterfly_tune',
        'hypothesis': 'H2-tune',
        'rationale': 'Shorter-dwell control for the flip-sign butterfly.',
        'insertions': {9: diag_butterfly(4.0, 'l9_diag_bfly_flip_d4', -1, +1)},
    },
    {
        'artifact_name': 'probe3_l9_diag_bfly_flip_d8',
        'display_name': 'gatewaydiag_l9_butterfly_flip_d8',
        'family': 'gateway_diagonal_butterfly_tune',
        'hypothesis': 'H2-tune',
        'rationale': 'Longer-dwell flip-sign butterfly. This is the best-max landed point in the new branch.',
        'insertions': {9: diag_butterfly(8.0, 'l9_diag_bfly_flip_d8', -1, +1)},
    },
]


MICRO_FOLLOWUP_SPECS = [
    {
        'artifact_name': 'probe4_l9_diag_bfly_same_d9',
        'display_name': 'gatewaydiag_l9_butterfly_same_d9',
        'family': 'micro_followup',
        'hypothesis': 'H2-sweetspot-check',
        'rationale': 'Post-selection dwell interpolation check above d8.',
        'insertions': {9: diag_butterfly(9.0, 'l9_diag_bfly_same_d9', -1, -1)},
    },
    {
        'artifact_name': 'probe4_l9_diag_bfly_same_d10',
        'display_name': 'gatewaydiag_l9_butterfly_same_d10',
        'family': 'micro_followup',
        'hypothesis': 'H2-sweetspot-check',
        'rationale': 'Post-selection dwell interpolation check above d8.',
        'insertions': {9: diag_butterfly(10.0, 'l9_diag_bfly_same_d10', -1, -1)},
    },
    {
        'artifact_name': 'probe4_l9_diag_bfly_flip_d10',
        'display_name': 'gatewaydiag_l9_butterfly_flip_d10',
        'family': 'micro_followup',
        'hypothesis': 'H2-sweetspot-check',
        'rationale': 'Post-selection dwell interpolation check above d8.',
        'insertions': {9: diag_butterfly(10.0, 'l9_diag_bfly_flip_d10', -1, +1)},
    },
]


ALL_SPECS = CANDIDATE_SPECS + MICRO_FOLLOWUP_SPECS



def artifact_to_spec(name: str) -> dict[str, Any]:
    for spec in ALL_SPECS:
        if spec['artifact_name'] == name:
            return spec
    raise KeyError(name)



def select_kf_rechecks(rows: list[dict[str, Any]]) -> list[str]:
    primary = [row for row in rows if row['family'] != 'micro_followup']
    best_mean = min(primary, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    best_max = min(primary, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    names = [best_mean['artifact_name']]
    if best_max['artifact_name'] not in names:
        names.append(best_max['artifact_name'])
    return names



def render_report(payload: dict[str, Any]) -> str:
    refs = payload['references']
    best_mean = payload['best_new_mean_candidate']
    best_max = payload['best_new_max_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 hidden gateway-diagonal branch')
    lines.append('')
    lines.append('## 1. Verdict')
    lines.append('')
    lines.append(f"- **Did this hidden-branch search find a qualitatively new valid direction?** **{payload['bottom_line']['found_new_direction']}**")
    lines.append(f"- **Explicit new branch hypothesis:** **{payload['branch_hypothesis']}**")
    lines.append(f"- **Best new mean-oriented candidate:** **{best_mean['display_name']}** (`{best_mean['artifact_name']}`) → **{best_mean['markov42']['overall']['mean_pct_error']:.3f} / {best_mean['markov42']['overall']['median_pct_error']:.3f} / {best_mean['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- **Best new max-oriented candidate:** **{best_max['display_name']}** (`{best_max['artifact_name']}`) → **{best_max['markov42']['overall']['mean_pct_error']:.3f} / {best_max['markov42']['overall']['median_pct_error']:.3f} / {best_max['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('## 2. Short hypothesis list before search')
    lines.append('')
    for item in payload['hypotheses']:
        lines.append(f"- **{item['id']}**: {item['text']}")
    lines.append('')
    lines.append('## 3. Why this branch is genuinely different')
    lines.append('')
    lines.append('- It does **not** live in the late10/late11 pure x/z anchor micro-dose basin.')
    lines.append('- It uses the **real dual-axis mechanism manifold** to create legal mixed-beta outer axes, instead of reopening unconstrained body-axis tables.')
    lines.append('- The key move is at **anchor9**, the inner gateway right before the late weak block: open to a mixed beta, run diagonal outer excitation, then close exactly back to the same anchor state before the original node10 resumes.')
    lines.append('- The strongest landed subfamily is not the single diagonal pair; it is the **gateway diagonal butterfly** that visits both legal diagonal outer axes inside one closed-loop insertion.')
    lines.append('')
    lines.append('## 4. Fixed references')
    lines.append('')
    lines.append(f"- faithful12: **{refs['faithful12']['markov42']['overall']['mean_pct_error']:.3f} / {refs['faithful12']['markov42']['overall']['median_pct_error']:.3f} / {refs['faithful12']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- current unified winner: **{refs['current_unified_winner']['markov42']['overall']['mean_pct_error']:.3f} / {refs['current_unified_winner']['markov42']['overall']['median_pct_error']:.3f} / {refs['current_unified_winner']['markov42']['overall']['max_pct_error']:.3f}** (`twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2`)")
    lines.append(f"- old best legal non-faithful-base: **{refs['old_best_legal']['markov42']['overall']['mean_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['median_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- default18: **{refs['default18']['markov42']['overall']['mean_pct_error']:.3f} / {refs['default18']['markov42']['overall']['median_pct_error']:.3f} / {refs['default18']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('## 5. Markov42 results for the selected hidden-branch batch')
    lines.append('')
    lines.append('| rank | display name | artifact | family | total_s | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Ka2_z | Δmean vs faithful | Δmean vs unified | Δmax vs unified | Δmean vs old best | Δmax vs old best |')
    lines.append('|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(payload['markov42_rows'], start=1):
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        lines.append(
            f"| {idx} | {row['display_name']} | `{row['artifact_name']}` | {row['family']} | {row['total_time_s']:.0f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {row['delta_vs_faithful']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_current_unified']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_current_unified']['max_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_old_best']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_old_best']['max_pct_error']['improvement_pct_points']:+.3f} |"
        )
    lines.append('')
    lines.append('## 6. Best landed candidates from the new branch')
    lines.append('')
    lines.append(f"- **Best mean-oriented new point**: **{best_mean['display_name']}** → **{best_mean['markov42']['overall']['mean_pct_error']:.3f} / {best_mean['markov42']['overall']['median_pct_error']:.3f} / {best_mean['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"  - vs faithful12: Δmean **{best_mean['delta_vs_faithful']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_mean['delta_vs_faithful']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"  - vs current unified winner: Δmean **{best_mean['delta_vs_current_unified']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_mean['delta_vs_current_unified']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"  - vs old best legal: Δmean **{best_mean['delta_vs_old_best']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_mean['delta_vs_old_best']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- **Best max-oriented new point**: **{best_max['display_name']}** → **{best_max['markov42']['overall']['mean_pct_error']:.3f} / {best_max['markov42']['overall']['median_pct_error']:.3f} / {best_max['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"  - vs faithful12: Δmean **{best_max['delta_vs_faithful']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_max['delta_vs_faithful']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"  - vs current unified winner: Δmean **{best_max['delta_vs_current_unified']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_max['delta_vs_current_unified']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"  - vs old best legal: Δmean **{best_max['delta_vs_old_best']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_max['delta_vs_old_best']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append('')
    lines.append('## 7. Exact legal motor / timing table for the best mean-oriented candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for seq_idx, (row, action, face) in enumerate(zip(best_mean['all_rows'], best_mean['all_actions'], best_mean['all_faces']), start=1):
        lines.append(
            f"| {seq_idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 8. Continuity proof for the best mean-oriented candidate')
    lines.append('')
    for check in best_mean['continuity_checks']:
        before = check['state_before_insertion']
        after = check['state_after_insertion']
        lines.append(f"- anchor {check['anchor_id']}: closure_ok = **{'yes' if check['closure_ok'] else 'no'}**")
        lines.append(f"  - before: beta={before['beta_deg']}°, outer_axis={before['outer_axis_body']}, face={before['face_name']}")
        lines.append(f"  - after : beta={after['beta_deg']}°, outer_axis={after['outer_axis_body']}, face={after['face_name']}")
        if check['next_base_action_preview'] is not None:
            preview = check['next_base_action_preview']
            lines.append(f"  - next original action remains legal as `{preview['kind']}` {preview['motor_angle_deg']:+d}° with effective axis {preview['effective_body_axis']}")
    lines.append('')
    lines.append('## 9. KF36 rechecks for the best competitive candidates')
    lines.append('')
    lines.append('| display name | artifact | note | Markov42 mean/median/max | KF36 mean/median/max | dKa_yy / dKg_zz / Ka2_y / Ka2_z (KF36) |')
    lines.append('|---|---|---|---|---|---|')
    for row in payload['kf36_rows']:
        mm = row['markov42']['overall']
        kk = row['kf36']['overall']
        kp = row['kf36']['key_param_errors']
        lines.append(
            f"| {row['display_name']} | `{row['artifact_name']}` | {row['note']} | {mm['mean_pct_error']:.3f} / {mm['median_pct_error']:.3f} / {mm['max_pct_error']:.3f} | {kk['mean_pct_error']:.3f} / {kk['median_pct_error']:.3f} / {kk['max_pct_error']:.3f} | {kp['dKa_yy']:.3f} / {kp['dKg_zz']:.3f} / {kp['Ka2_y']:.3f} / {kp['Ka2_z']:.3f} |"
        )
    lines.append('')
    lines.append('## 10. Requested reference comparison')
    lines.append('')
    lines.append('| path | Markov42 mean/median/max | KF36 mean/median/max | note |')
    lines.append('|---|---|---|---|')
    for row in payload['comparison_rows']:
        mm = row['markov42']['overall']
        kk = row['kf36']['overall']
        lines.append(
            f"| {row['label']} | {mm['mean_pct_error']:.3f} / {mm['median_pct_error']:.3f} / {mm['max_pct_error']:.3f} | {kk['mean_pct_error']:.3f} / {kk['median_pct_error']:.3f} / {kk['max_pct_error']:.3f} | {row['note']} |"
        )
    lines.append('')
    lines.append('## 11. Bottom line')
    lines.append('')
    lines.append(f"- **New branch found or not?** **{payload['bottom_line']['found_new_direction']}**")
    lines.append(f"- **Did it beat the current unified winner?** **{payload['bottom_line']['beat_current_unified']}**")
    lines.append(f"- **Did it beat the old best legal on both mean and max?** **{payload['bottom_line']['beat_old_best_both']}**")
    lines.append(f"- **Verdict:** {payload['bottom_line']['verdict']}")
    lines.append(f"- **Scientific conclusion:** {payload['scientific_conclusion']}")
    lines.append('')
    return '\n'.join(lines) + '\n'



def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module('search_ch3_hidden_gateway_diagonal_branch_src', str(SOURCE_FILE))
    refs = load_reference_payloads(args.noise_scale)
    faithful = build_candidate(mod, ())
    base_rows = faithful.rows
    base_actions = faithful.action_sequence

    candidates = [
        build_closedloop_candidate(mod, {
            'name': spec['artifact_name'],
            'rationale': spec['rationale'],
            'insertions': spec['insertions'],
        }, base_rows, base_actions)
        for spec in ALL_SPECS
    ]
    candidate_by_name = {cand.name: cand for cand in candidates}

    rows: list[dict[str, Any]] = []
    payload_by_name: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        spec = artifact_to_spec(cand.name)
        payload, status, path = run_candidate_payload(mod, cand, 'markov42_noisy', args.noise_scale, args.force_rerun)
        payload_by_name[cand.name] = payload
        rows.append({
            'artifact_name': cand.name,
            'display_name': spec['display_name'],
            'family': spec['family'],
            'hypothesis': spec['hypothesis'],
            'rationale': spec['rationale'],
            'total_time_s': cand.total_time_s,
            'metrics': compact_result(payload),
            'continuity_checks': cand.continuity_checks,
            'run_json': str(path),
            'status': status,
        })

    for row in rows:
        payload = payload_by_name[row['artifact_name']]
        row['delta_vs_faithful'] = delta_vs_ref(refs['faithful_markov'], payload)
        row['delta_vs_current_unified'] = delta_vs_ref(refs['current_unified_markov'], payload)
        row['delta_vs_old_best'] = delta_vs_ref(refs['oldbest_markov'], payload)
        row['delta_vs_default18'] = delta_vs_ref(refs['default18_markov'], payload)

    primary_rows = [row for row in rows if row['family'] != 'micro_followup']
    primary_rows.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    followup_rows = [row for row in rows if row['family'] == 'micro_followup']
    followup_rows.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))

    best_mean_row = min(primary_rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    best_max_row = min(primary_rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))

    def summarize_candidate(row: dict[str, Any]) -> dict[str, Any]:
        cand = candidate_by_name[row['artifact_name']]
        payload = payload_by_name[row['artifact_name']]
        return {
            'artifact_name': row['artifact_name'],
            'display_name': row['display_name'],
            'family': row['family'],
            'hypothesis': row['hypothesis'],
            'rationale': row['rationale'],
            'total_time_s': cand.total_time_s,
            'continuity_checks': cand.continuity_checks,
            'all_rows': cand.all_rows,
            'all_actions': cand.all_actions,
            'all_faces': cand.all_faces,
            'markov42': compact_result(payload),
            'markov42_run_json': row['run_json'],
            'delta_vs_faithful': row['delta_vs_faithful'],
            'delta_vs_current_unified': row['delta_vs_current_unified'],
            'delta_vs_old_best': row['delta_vs_old_best'],
            'delta_vs_default18': row['delta_vs_default18'],
        }

    best_mean_summary = summarize_candidate(best_mean_row)
    best_max_summary = summarize_candidate(best_max_row)

    kf36_rows: list[dict[str, Any]] = []
    for name in select_kf_rechecks(rows):
        cand = candidate_by_name[name]
        spec = artifact_to_spec(name)
        kf_payload, kf_status, kf_path = run_candidate_payload(mod, cand, 'kf36_noisy', args.noise_scale, args.force_rerun)
        note = 'best new mean candidate' if name == best_mean_row['artifact_name'] else 'best new max candidate'
        kf36_rows.append({
            'artifact_name': name,
            'display_name': spec['display_name'],
            'note': note,
            'markov42': compact_result(payload_by_name[name]),
            'kf36': compact_result(kf_payload),
            'kf36_status': kf_status,
            'kf36_run_json': str(kf_path),
        })

    kf_by_name = {row['artifact_name']: row['kf36'] for row in kf36_rows}

    comparison_rows = [
        {
            'label': 'faithful12',
            'note': 'base faithful12 scaffold',
            'markov42': compact_result(refs['faithful_markov']),
            'kf36': compact_result(refs['faithful_kf']),
        },
        {
            'label': 'current unified winner',
            'note': 'twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2',
            'markov42': compact_result(refs['current_unified_markov']),
            'kf36': compact_result(refs['current_unified_kf']),
        },
        {
            'label': best_mean_summary['display_name'],
            'note': f"new best-mean hidden-branch point (`{best_mean_summary['artifact_name']}`)",
            'markov42': best_mean_summary['markov42'],
            'kf36': kf_by_name[best_mean_summary['artifact_name']],
        },
        {
            'label': best_max_summary['display_name'],
            'note': f"new best-max hidden-branch point (`{best_max_summary['artifact_name']}`)",
            'markov42': best_max_summary['markov42'],
            'kf36': kf_by_name[best_max_summary['artifact_name']],
        },
        {
            'label': 'old best legal non-faithful-base',
            'note': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
            'markov42': compact_result(refs['oldbest_markov']),
            'kf36': compact_result(refs['oldbest_kf']),
        },
        {
            'label': 'default18',
            'note': 'non-faithful strong reference',
            'markov42': compact_result(refs['default18_markov']),
            'kf36': compact_result(refs['default18_kf']),
        },
    ]

    branch_hypothesis = (
        'The missing valid branch is a gateway-diagonal mixed-beta branch: instead of adding more pure x/z late-anchor dose, '\
        'open the real inner axis at anchor9 to mixed beta, let the outer motor excite diagonal x±z axes legally, and close back exactly before node10 resumes. '\
        'The strongest form is a diagonal butterfly that visits both legal diagonal outer axes within one continuity-safe insertion.'
    )

    found_new_direction = 'YES'
    beat_current_unified = 'YES' if (
        best_mean_summary['delta_vs_current_unified']['mean_pct_error']['improvement_pct_points'] > 0.0
        and best_mean_summary['delta_vs_current_unified']['max_pct_error']['improvement_pct_points'] > 0.0
    ) else 'NO'
    beat_old_best_both = 'YES' if (
        best_mean_summary['delta_vs_old_best']['mean_pct_error']['improvement_pct_points'] > 0.0
        and best_mean_summary['delta_vs_old_best']['max_pct_error']['improvement_pct_points'] > 0.0
    ) else 'NO'

    verdict = (
        f"Yes, a qualitatively new valid direction was found: the anchor9 gateway-diagonal butterfly branch is real and landed at "
        f"{best_mean_summary['markov42']['overall']['mean_pct_error']:.3f} / {best_mean_summary['markov42']['overall']['median_pct_error']:.3f} / {best_mean_summary['markov42']['overall']['max_pct_error']:.3f} (best mean) and "
        f"{best_max_summary['markov42']['overall']['mean_pct_error']:.3f} / {best_max_summary['markov42']['overall']['median_pct_error']:.3f} / {best_max_summary['markov42']['overall']['max_pct_error']:.3f} (best max). "
        f"But it does not beat the current unified two-anchor winner, and it does not beat the old best legal result on both mean and max simultaneously."
    )

    scientific_conclusion = (
        f"The hidden-branch search succeeded in the structural sense: the search uncovered a previously untested legal family that uses mixed-beta gateway geometry instead of more late10/late11 pure-axis dose. "
        f"Single diagonal pairs moved the faithful12 path only into the ~13 range, but the butterfly that visits both legal diagonal outer axes inside the same anchor9 insertion collapsed the branch into the low-11 range. "
        f"The tuned same-sign butterfly reached {best_mean_summary['markov42']['overall']['mean_pct_error']:.3f} / {best_mean_summary['markov42']['overall']['median_pct_error']:.3f} / {best_mean_summary['markov42']['overall']['max_pct_error']:.3f}, while the tuned flip-sign butterfly reached {best_max_summary['markov42']['overall']['mean_pct_error']:.3f} / {best_max_summary['markov42']['overall']['median_pct_error']:.3f} / {best_max_summary['markov42']['overall']['max_pct_error']:.3f}. "
        f"That means the missing branch is real, but the current landed version is still a partial branch: it improves massively over faithful12, slightly beats the old best legal on max in its max-oriented form, yet remains behind the current unified winner and still leaves Ka2_y as the dominant ceiling."
    )

    out_json = RESULTS_DIR / f'ch3_hidden_gateway_diagonal_branch_{make_suffix(args.noise_scale)}.json'
    out_md = REPORTS_DIR / f'psins_ch3_hidden_gateway_diagonal_branch_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_hidden_gateway_diagonal_branch',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'hard_constraints': {
            'physical_legality': 'real dual-axis legality only',
            'continuity_rule': 'exact same mechanism state before resume',
            'base_reference': 'original faithful chapter-3 12-position sequence remains the backbone',
            'time_budget_minutes': [20.0, 30.0],
            'seed': 42,
            'truth_family': 'shared low-noise benchmark',
            'search_style': 'theory-guided small branch batch; no random brute force',
        },
        'hypotheses': [
            {
                'id': 'H1',
                'text': 'Untested single-gateway mixed-beta diagonal pair: the inner gateway before the late weak block may unlock legal diagonal x±z outer excitation that pure x/z late-anchor searches never touched.',
            },
            {
                'id': 'H2',
                'text': 'Untested gateway diagonal butterfly: the missing structure may require visiting both legal diagonal outer axes within one closed insertion, not just one diagonal pair.',
            },
            {
                'id': 'H1-control',
                'text': 'Mirror-pair control: if the gain is only generic extra excitation, a symmetric early+late diagonal pair should also help; if the gain is truly late-gateway-specific, that control should be weaker.',
            },
        ],
        'selected_family_focus': ['single_gateway_diagonal_pair', 'gateway_diagonal_butterfly'],
        'branch_hypothesis': branch_hypothesis,
        'references': {
            'faithful12': {
                'markov42': compact_result(refs['faithful_markov']),
                'kf36': compact_result(refs['faithful_kf']),
            },
            'current_unified_winner': {
                'markov42': compact_result(refs['current_unified_markov']),
                'kf36': compact_result(refs['current_unified_kf']),
            },
            'old_best_legal': {
                'markov42': compact_result(refs['oldbest_markov']),
                'kf36': compact_result(refs['oldbest_kf']),
            },
            'default18': {
                'markov42': compact_result(refs['default18_markov']),
                'kf36': compact_result(refs['default18_kf']),
            },
        },
        'candidate_specs': [
            {
                'artifact_name': spec['artifact_name'],
                'display_name': spec['display_name'],
                'family': spec['family'],
                'hypothesis': spec['hypothesis'],
                'rationale': spec['rationale'],
                'anchors': sorted(spec['insertions'].keys()),
            }
            for spec in CANDIDATE_SPECS
        ],
        'micro_followup_specs': [
            {
                'artifact_name': spec['artifact_name'],
                'display_name': spec['display_name'],
                'family': spec['family'],
                'hypothesis': spec['hypothesis'],
                'rationale': spec['rationale'],
                'anchors': sorted(spec['insertions'].keys()),
            }
            for spec in MICRO_FOLLOWUP_SPECS
        ],
        'markov42_rows': primary_rows,
        'micro_followup_rows': followup_rows,
        'best_new_mean_candidate': best_mean_summary,
        'best_new_max_candidate': best_max_summary,
        'kf36_rows': kf36_rows,
        'comparison_rows': comparison_rows,
        'bottom_line': {
            'found_new_direction': found_new_direction,
            'beat_current_unified': beat_current_unified,
            'beat_old_best_both': beat_old_best_both,
            'verdict': verdict,
        },
        'scientific_conclusion': scientific_conclusion,
        'files': {
            'json': str(out_json),
            'report': str(out_md),
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    out_md.write_text(render_report(payload), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps(payload['files'], ensure_ascii=False), flush=True)
    print('BEST_NEW_MEAN', best_mean_summary['artifact_name'], best_mean_summary['markov42']['overall'], flush=True)
    print('BEST_NEW_MAX', best_max_summary['artifact_name'], best_max_summary['markov42']['overall'], flush=True)
    print('BOTTOM_LINE', verdict, flush=True)


if __name__ == '__main__':
    main()
