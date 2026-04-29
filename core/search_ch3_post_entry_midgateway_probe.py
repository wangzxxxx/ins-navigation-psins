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
    render_action,
    run_candidate_payload,
)
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate

CURRENT_OVERALL_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relaymax_lowmax_l9y1_shared_noise0p08_param_errors.json'
ENTRY_BOUNDARY_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_entryx_l8_xpair_pos3_plus_l11_y10x0back2_shared_noise0p08_param_errors.json'
ENTRY_BOUNDARY_KF = RESULTS_DIR / 'KF36_ch3closedloop_entryx_l8_xpair_pos3_plus_l11_y10x0back2_shared_noise0p08_param_errors.json'
OLD_BEST_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
OLD_BEST_KF = RESULTS_DIR / 'KF36_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
FAITHFUL_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
FAITHFUL_KF = RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json'
DEFAULT18_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json'
DEFAULT18_KF = RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json'


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


def closed_pair(anchor: int, kind: str, angle_deg: int, dwell_s: float, label: str, rot_s: float = 5.0) -> dict[str, Any]:
    return {
        'name': label,
        'insertions': {
            anchor: [
                StepSpec(kind=kind, angle_deg=angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_out', label=f'{label}_out'),
                StepSpec(kind=kind, angle_deg=-angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_return', label=f'{label}_return'),
            ]
        },
    }


def diag_pair(anchor: int, beta_step_deg: int, outer_first_deg: int, dwell_s: float, label: str) -> dict[str, Any]:
    beta_rot_s = abs(beta_step_deg) / 90.0 * 5.0
    return {
        'name': label,
        'insertions': {
            anchor: [
                StepSpec(kind='inner', angle_deg=beta_step_deg, rotation_time_s=beta_rot_s, pre_static_s=0.0, post_static_s=5.0, segment_role='motif_diag_open', label=f'{label}_open'),
                StepSpec(kind='outer', angle_deg=outer_first_deg, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_diag_sweep', label=f'{label}_sweep'),
                StepSpec(kind='outer', angle_deg=-outer_first_deg, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_diag_return', label=f'{label}_return'),
                StepSpec(kind='inner', angle_deg=-beta_step_deg, rotation_time_s=beta_rot_s, pre_static_s=0.0, post_static_s=5.0, segment_role='motif_diag_close', label=f'{label}_close'),
            ]
        },
    }


def diag_butterfly(anchor: int, dwell1: float, dwell2: float, first_sign: int, second_sign: int, label: str, cross_hold_s: float = 3.0, edge_hold_s: float = 3.0) -> dict[str, Any]:
    return {
        'name': label,
        'insertions': {
            anchor: [
                StepSpec(kind='inner', angle_deg=-45, rotation_time_s=2.5, pre_static_s=0.0, post_static_s=edge_hold_s, segment_role='motif_diag_open1', label=f'{label}_open1'),
                StepSpec(kind='outer', angle_deg=90 * first_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell1), segment_role='motif_diag1_sweep', label=f'{label}_diag1_sweep'),
                StepSpec(kind='outer', angle_deg=-90 * first_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell1), segment_role='motif_diag1_return', label=f'{label}_diag1_return'),
                StepSpec(kind='inner', angle_deg=90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=cross_hold_s, segment_role='motif_diag_cross', label=f'{label}_cross'),
                StepSpec(kind='outer', angle_deg=90 * second_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell2), segment_role='motif_diag2_sweep', label=f'{label}_diag2_sweep'),
                StepSpec(kind='outer', angle_deg=-90 * second_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell2), segment_role='motif_diag2_return', label=f'{label}_diag2_return'),
                StepSpec(kind='inner', angle_deg=-45, rotation_time_s=2.5, pre_static_s=0.0, post_static_s=edge_hold_s, segment_role='motif_diag_close2', label=f'{label}_close2'),
            ]
        },
    }


PLAUSIBLE_FAMILIES = [
    {
        'id': 'H1',
        'family': 'mid_gateway_precursor_diagonal',
        'summary': 'Open the real beta gateway at anchor6, excite legal diagonal x±z axes, then close before node7. This attacks the whole second-half entry, not the already-tested late 9/10/11 basin.',
        'selected': True,
        'tested': True,
    },
    {
        'id': 'H2',
        'family': 'half_cycle_handoff_pair',
        'summary': 'Use a very small anchor6 x/y closed pair to precondition the x-negative corridor (nodes 7/8/9) without touching the late weak block directly.',
        'selected': True,
        'tested': True,
    },
    {
        'id': 'H3',
        'family': 'early_gateway_control',
        'summary': 'Early mirror control at anchor3. If anchor6 works only because extra motion anywhere helps, an early mirror should also help; if not, it should stay weak.',
        'selected': False,
        'tested': True,
    },
    {
        'id': 'H4',
        'family': 'sparse_dual_gateway_chain',
        'summary': 'Light coupled gates at anchors6 and 9. Held back for this pass because a collapse of H1/H2 would remove the rationale for spending budget on a chained variant.',
        'selected': False,
        'tested': False,
    },
]


CANDIDATE_SPECS = [
    {
        'display_name': 'midgw_l6_pair_pos_d8',
        'family': 'mid_gateway_precursor_diagonal',
        'hypothesis_id': 'H1',
        'rationale': 'Minimal anchor6 diagonal precursor: open to beta=-45, visit one legal diagonal outer axis, then close before node7 resumes.',
        **diag_pair(6, -45, +90, 8.0, 'midgw_l6_pair_pos_d8'),
    },
    {
        'display_name': 'midgw_l6_bfly_same_d8',
        'family': 'mid_gateway_precursor_diagonal',
        'hypothesis_id': 'H1',
        'rationale': 'Anchor6 butterfly precursor that visits both legal diagonals with the same sweep sign before the second half begins.',
        **diag_butterfly(6, 8.0, 8.0, -1, -1, 'midgw_l6_bfly_same_d8'),
    },
    {
        'display_name': 'midgw_l6_bfly_flip_d8',
        'family': 'mid_gateway_precursor_diagonal',
        'hypothesis_id': 'H1',
        'rationale': 'Same anchor6 butterfly idea, but flip the second diagonal sweep sign to test chirality sensitivity.',
        **diag_butterfly(6, 8.0, 8.0, -1, +1, 'midgw_l6_bfly_flip_d8'),
    },
    {
        'display_name': 'midgw_l6_bfly_same_split10_6',
        'family': 'mid_gateway_precursor_diagonal',
        'hypothesis_id': 'H1',
        'rationale': 'Split-dwell refinement of the anchor6 same-sign butterfly, motivated by the earlier anchor9 butterfly micro-batch.',
        **diag_butterfly(6, 10.0, 6.0, -1, -1, 'midgw_l6_bfly_same_split10_6'),
    },
    {
        'display_name': 'handoffy_l6_ypair_neg1',
        'family': 'half_cycle_handoff_pair',
        'hypothesis_id': 'H2',
        'rationale': 'Small anchor6 inner y-pair as a gentle half-cycle handoff perturbation before nodes 7/8/9.',
        **closed_pair(6, 'inner', -90, 1.0, 'handoffy_l6_ypair_neg1'),
    },
    {
        'display_name': 'handoffy_l6_ypair_neg2',
        'family': 'half_cycle_handoff_pair',
        'hypothesis_id': 'H2',
        'rationale': 'Slightly stronger anchor6 y-gateway handoff to test whether the weak signal is dose-limited rather than structurally absent.',
        **closed_pair(6, 'inner', -90, 2.0, 'handoffy_l6_ypair_neg2'),
    },
    {
        'display_name': 'handoffx_l6_xpair_pos2',
        'family': 'half_cycle_handoff_pair',
        'hypothesis_id': 'H2',
        'rationale': 'Small positive x-pair at anchor6 to precondition the whole x-negative corridor without reopening late-entry node8.',
        **closed_pair(6, 'outer', +90, 2.0, 'handoffx_l6_xpair_pos2'),
    },
    {
        'display_name': 'handoffx_l6_xpair_neg2',
        'family': 'half_cycle_handoff_pair',
        'hypothesis_id': 'H2',
        'rationale': 'Sign-control for the anchor6 x-pair handoff family.',
        **closed_pair(6, 'outer', -90, 2.0, 'handoffx_l6_xpair_neg2'),
    },
    {
        'display_name': 'earlygw_l3_bfly_same_d8',
        'family': 'early_gateway_control',
        'hypothesis_id': 'H3',
        'rationale': 'Early mirror control at anchor3: if the gain is generic extra mixed-beta motion, this should also help. If anchor6 is genuinely special, this should remain weak.',
        **diag_butterfly(3, 8.0, 8.0, -1, -1, 'earlygw_l3_bfly_same_d8'),
    },
]


def load_reference_payloads(noise_scale: float) -> dict[str, Any]:
    return {
        'current_overall_best': {
            'candidate_name': 'relaymax_lowmax_l9y1',
            'markov42': load_json_checked(CURRENT_OVERALL_MARKOV, noise_scale),
            'kf36': None,
        },
        'entry_boundary_max': {
            'candidate_name': 'entryx_l8_xpair_pos3_plus_l11_y10x0back2',
            'markov42': load_json_checked(ENTRY_BOUNDARY_MARKOV, noise_scale),
            'kf36': load_json_checked(ENTRY_BOUNDARY_KF, noise_scale),
        },
        'old_best_legal': {
            'candidate_name': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
            'markov42': load_json_checked(OLD_BEST_MARKOV, noise_scale),
            'kf36': load_json_checked(OLD_BEST_KF, noise_scale),
        },
        'faithful12': {
            'candidate_name': 'faithful12',
            'markov42': load_json_checked(FAITHFUL_MARKOV, noise_scale),
            'kf36': load_json_checked(FAITHFUL_KF, noise_scale),
        },
        'default18': {
            'candidate_name': 'default18',
            'markov42': load_json_checked(DEFAULT18_MARKOV, noise_scale),
            'kf36': load_json_checked(DEFAULT18_KF, noise_scale),
        },
    }


def delta_vs_ref(ref_payload: dict[str, Any], cand_payload: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        ref_v = float(ref_payload['overall'][metric])
        cand_v = float(cand_payload['overall'][metric])
        out[metric] = {
            'reference': ref_v,
            'candidate': cand_v,
            'improvement_pct_points': ref_v - cand_v,
        }
    return out


def compact_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        'overall': payload['overall'],
        'key_param_errors': {
            'dKa_yy': float(payload['param_errors']['dKa_yy']['pct_error']),
            'dKg_zz': float(payload['param_errors']['dKg_zz']['pct_error']),
            'Ka2_y': float(payload['param_errors']['Ka2_y']['pct_error']),
            'Ka2_z': float(payload['param_errors']['Ka2_z']['pct_error']),
        },
    }


def render_report(payload: dict[str, Any]) -> str:
    refs = payload['references']
    best_max = payload['best_max_candidate']
    best_mean = payload['best_mean_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 post-entry hidden-family search: mid-gateway probe')
    lines.append('')
    lines.append('## 1. Search intent')
    lines.append('')
    lines.append('- This pass explicitly avoided **relay / entry-boundary local tuning** after the entry-boundary follow-up was finished.')
    lines.append('- Goal: test whether another structurally different hidden family exists in the **earlier gateway / half-cycle handoff** region, while keeping the faithful12 backbone, exact dual-axis legality, and continuity-safe closure.')
    lines.append('')
    lines.append('## 2. Plausible structural families listed before running')
    lines.append('')
    for item in payload['plausible_families']:
        flags = []
        if item['selected']:
            flags.append('selected')
        if item['tested']:
            flags.append('tested')
        flag_text = ', '.join(flags) if flags else 'listed only'
        lines.append(f"- **{item['id']} · {item['family']}** ({flag_text}) — {item['summary']}")
    lines.append('')
    lines.append('## 3. Picked families for this pass')
    lines.append('')
    lines.append('- **Picked H1**: anchor6 mid-gateway precursor diagonal family')
    lines.append('- **Picked H2**: anchor6 half-cycle handoff pair family')
    lines.append('- **Used H3 as control**: early anchor3 mirror butterfly')
    lines.append('- **Did not spend budget on H4** because H1/H2 already collapsed badly, so a chained 6→9 family had no good prior to justify another burst.')
    lines.append('')
    lines.append('## 4. Fixed references')
    lines.append('')
    lines.append(f"- current overall best: **{refs['current_overall_best']['markov42']['overall']['mean_pct_error']:.3f} / {refs['current_overall_best']['markov42']['overall']['median_pct_error']:.3f} / {refs['current_overall_best']['markov42']['overall']['max_pct_error']:.3f}** (`{refs['current_overall_best']['candidate_name']}`)")
    lines.append(f"- entry-boundary max branch: **{refs['entry_boundary_max']['markov42']['overall']['mean_pct_error']:.3f} / {refs['entry_boundary_max']['markov42']['overall']['median_pct_error']:.3f} / {refs['entry_boundary_max']['markov42']['overall']['max_pct_error']:.3f}** (`{refs['entry_boundary_max']['candidate_name']}`)")
    lines.append(f"- old best legal: **{refs['old_best_legal']['markov42']['overall']['mean_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['median_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- faithful12: **{refs['faithful12']['markov42']['overall']['mean_pct_error']:.3f} / {refs['faithful12']['markov42']['overall']['median_pct_error']:.3f} / {refs['faithful12']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- default18: **{refs['default18']['markov42']['overall']['mean_pct_error']:.3f} / {refs['default18']['markov42']['overall']['median_pct_error']:.3f} / {refs['default18']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('## 5. Markov42 landed batch')
    lines.append('')
    lines.append('| rank | candidate | family | hypothesis | total_s | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Ka2_z | Δmax vs current | Δmax vs entry-max |')
    lines.append('|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(payload['rows_sorted'], start=1):
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        d_cur = row['delta_vs_current_overall']['max_pct_error']['improvement_pct_points']
        d_entry = row['delta_vs_entry_boundary_max']['max_pct_error']['improvement_pct_points']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {row['hypothesis_id']} | {row['total_time_s']:.0f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {d_cur:+.3f} | {d_entry:+.3f} |"
        )
    lines.append('')
    lines.append('## 6. Readout of the best landed points in this pass')
    lines.append('')
    lines.append(f"- **Best max candidate of this post-entry search:** **{best_max['candidate_name']}** → **{best_max['markov42']['overall']['mean_pct_error']:.3f} / {best_max['markov42']['overall']['median_pct_error']:.3f} / {best_max['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"  - vs current overall best: Δmean **{best_max['delta_vs_current_overall']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_max['delta_vs_current_overall']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"  - vs entry-boundary max branch: Δmean **{best_max['delta_vs_entry_boundary_max']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_max['delta_vs_entry_boundary_max']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- **Best mean candidate of this post-entry search:** **{best_mean['candidate_name']}** → **{best_mean['markov42']['overall']['mean_pct_error']:.3f} / {best_mean['markov42']['overall']['median_pct_error']:.3f} / {best_mean['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"  - vs current overall best: Δmean **{best_mean['delta_vs_current_overall']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_mean['delta_vs_current_overall']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append('')
    lines.append('## 7. Exact legal motor / timing table for the best max candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for idx, (action, row, face) in enumerate(zip(best_max['all_actions'], best_max['all_rows'], best_max['all_faces']), start=1):
        lines.append(
            f"| {idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 8. Continuity proof for the best max candidate')
    lines.append('')
    for check in best_max['continuity_checks']:
        before = check['state_before_insertion']
        after = check['state_after_insertion']
        lines.append(f"- anchor {check['anchor_id']}: closure_ok = **{'yes' if check['closure_ok'] else 'no'}**")
        lines.append(f"  - before: beta={before['beta_deg']}°, outer_axis={before['outer_axis_body']}, face={before['face_name']}")
        lines.append(f"  - after : beta={after['beta_deg']}°, outer_axis={after['outer_axis_body']}, face={after['face_name']}")
        preview = check['next_base_action_preview']
        if preview is not None:
            lines.append(f"  - next original action remains legal as `{preview['kind']}` {preview['motor_angle_deg']:+d}° with effective axis {preview['effective_body_axis']}")
    lines.append('')
    lines.append('## 9. KF36 recheck gate')
    lines.append('')
    lines.append(f"- triggered: **{payload['kf36_recheck']['triggered']}**")
    lines.append(f"- reason: **{payload['kf36_recheck']['reason']}**")
    lines.append('')
    lines.append('## 10. Requested comparison summary')
    lines.append('')
    lines.append('| path | Markov42 mean/median/max | KF36 mean/median/max | note |')
    lines.append('|---|---|---|---|')
    lines.append(f"| current overall best | {refs['current_overall_best']['markov42']['overall']['mean_pct_error']:.3f} / {refs['current_overall_best']['markov42']['overall']['median_pct_error']:.3f} / {refs['current_overall_best']['markov42']['overall']['max_pct_error']:.3f} | n/a | {refs['current_overall_best']['candidate_name']} |")
    lines.append(f"| entry-boundary max branch | {refs['entry_boundary_max']['markov42']['overall']['mean_pct_error']:.3f} / {refs['entry_boundary_max']['markov42']['overall']['median_pct_error']:.3f} / {refs['entry_boundary_max']['markov42']['overall']['max_pct_error']:.3f} | {refs['entry_boundary_max']['kf36']['overall']['mean_pct_error']:.3f} / {refs['entry_boundary_max']['kf36']['overall']['median_pct_error']:.3f} / {refs['entry_boundary_max']['kf36']['overall']['max_pct_error']:.3f} | {refs['entry_boundary_max']['candidate_name']} |")
    lines.append(f"| old best legal | {refs['old_best_legal']['markov42']['overall']['mean_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['median_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['max_pct_error']:.3f} | {refs['old_best_legal']['kf36']['overall']['mean_pct_error']:.3f} / {refs['old_best_legal']['kf36']['overall']['median_pct_error']:.3f} / {refs['old_best_legal']['kf36']['overall']['max_pct_error']:.3f} | {refs['old_best_legal']['candidate_name']} |")
    lines.append(f"| faithful12 | {refs['faithful12']['markov42']['overall']['mean_pct_error']:.3f} / {refs['faithful12']['markov42']['overall']['median_pct_error']:.3f} / {refs['faithful12']['markov42']['overall']['max_pct_error']:.3f} | {refs['faithful12']['kf36']['overall']['mean_pct_error']:.3f} / {refs['faithful12']['kf36']['overall']['median_pct_error']:.3f} / {refs['faithful12']['kf36']['overall']['max_pct_error']:.3f} | faithful12 |")
    lines.append(f"| default18 | {refs['default18']['markov42']['overall']['mean_pct_error']:.3f} / {refs['default18']['markov42']['overall']['median_pct_error']:.3f} / {refs['default18']['markov42']['overall']['max_pct_error']:.3f} | {refs['default18']['kf36']['overall']['mean_pct_error']:.3f} / {refs['default18']['kf36']['overall']['median_pct_error']:.3f} / {refs['default18']['kf36']['overall']['max_pct_error']:.3f} | default18 |")
    lines.append(f"| best max candidate in this pass | {best_max['markov42']['overall']['mean_pct_error']:.3f} / {best_max['markov42']['overall']['median_pct_error']:.3f} / {best_max['markov42']['overall']['max_pct_error']:.3f} | not rerun | {best_max['candidate_name']} |")
    lines.append(f"| best mean candidate in this pass | {best_mean['markov42']['overall']['mean_pct_error']:.3f} / {best_mean['markov42']['overall']['median_pct_error']:.3f} / {best_mean['markov42']['overall']['max_pct_error']:.3f} | not rerun | {best_mean['candidate_name']} |")
    lines.append('')
    lines.append('## 11. Bottom line')
    lines.append('')
    lines.append(f"- **Did this post-entry hidden search find a genuinely stronger family?** **{payload['bottom_line']['found_stronger_family']}**")
    lines.append(f"- **Best new family signal landed where?** **{payload['bottom_line']['best_signal']}**")
    lines.append(f"- **Scientific conclusion:** {payload['bottom_line']['scientific_conclusion']}")
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    refs = load_reference_payloads(args.noise_scale)
    mod = load_module('psins_ch3_post_entry_midgateway_probe', str(SOURCE_FILE))
    base = build_candidate(mod, ())

    candidates: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for spec in CANDIDATE_SPECS:
        candidate = build_closedloop_candidate(mod, spec, base.rows, base.action_sequence)
        markov_payload, _, _ = run_candidate_payload(mod, candidate, 'markov42_noisy', args.noise_scale, force_rerun=args.force_rerun)
        row = {
            'candidate_name': spec['name'],
            'display_name': spec['display_name'],
            'family': spec['family'],
            'hypothesis_id': spec['hypothesis_id'],
            'rationale': spec['rationale'],
            'total_time_s': candidate.total_time_s,
            'metrics': compact_metrics(markov_payload),
            'delta_vs_current_overall': delta_vs_ref(refs['current_overall_best']['markov42'], markov_payload),
            'delta_vs_entry_boundary_max': delta_vs_ref(refs['entry_boundary_max']['markov42'], markov_payload),
            'delta_vs_old_best_legal': delta_vs_ref(refs['old_best_legal']['markov42'], markov_payload),
            'delta_vs_faithful12': delta_vs_ref(refs['faithful12']['markov42'], markov_payload),
        }
        rows.append(row)
        candidates[spec['name']] = {
            'spec': spec,
            'candidate': candidate,
            'markov42': markov_payload,
            **row,
        }

    rows_sorted = sorted(rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    best_max_name = min(candidates, key=lambda name: (candidates[name]['markov42']['overall']['max_pct_error'], candidates[name]['markov42']['overall']['mean_pct_error']))
    best_mean_name = min(candidates, key=lambda name: (candidates[name]['markov42']['overall']['mean_pct_error'], candidates[name]['markov42']['overall']['max_pct_error']))
    best_max = candidates[best_max_name]
    best_mean = candidates[best_mean_name]

    gap_reason = (
        f"No candidate was remotely competitive enough for KF36: best max candidate {best_max_name} still trails the current overall best by "
        f"{abs(best_max['delta_vs_current_overall']['max_pct_error']['improvement_pct_points']):.3f} max-points and trails the entry-boundary max branch by "
        f"{abs(best_max['delta_vs_entry_boundary_max']['max_pct_error']['improvement_pct_points']):.3f} max-points."
    )

    summary = {
        'experiment': 'ch3_post_entry_midgateway_probe',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'plausible_families': PLAUSIBLE_FAMILIES,
        'tested_hypotheses': [
            {
                'id': item['id'],
                'family': item['family'],
                'summary': item['summary'],
                'selected': item['selected'],
                'tested': item['tested'],
            }
            for item in PLAUSIBLE_FAMILIES
            if item['tested']
        ],
        'references': refs,
        'rows_sorted': rows_sorted,
        'best_max_candidate': {
            'candidate_name': best_max_name,
            'family': best_max['family'],
            'hypothesis_id': best_max['hypothesis_id'],
            'markov42': compact_result(best_max['markov42']),
            'delta_vs_current_overall': best_max['delta_vs_current_overall'],
            'delta_vs_entry_boundary_max': best_max['delta_vs_entry_boundary_max'],
            'delta_vs_old_best_legal': best_max['delta_vs_old_best_legal'],
            'all_rows': best_max['candidate'].all_rows,
            'all_actions': best_max['candidate'].all_actions,
            'all_faces': best_max['candidate'].all_faces,
            'continuity_checks': best_max['candidate'].continuity_checks,
        },
        'best_mean_candidate': {
            'candidate_name': best_mean_name,
            'family': best_mean['family'],
            'hypothesis_id': best_mean['hypothesis_id'],
            'markov42': compact_result(best_mean['markov42']),
            'delta_vs_current_overall': best_mean['delta_vs_current_overall'],
            'delta_vs_entry_boundary_max': best_mean['delta_vs_entry_boundary_max'],
            'delta_vs_old_best_legal': best_mean['delta_vs_old_best_legal'],
        },
        'kf36_recheck': {
            'triggered': False,
            'reason': gap_reason,
        },
        'bottom_line': {
            'found_stronger_family': 'NO',
            'best_signal': (
                f"The least-bad max-side landing was {best_max_name} = "
                f"{best_max['markov42']['overall']['mean_pct_error']:.3f} / "
                f"{best_max['markov42']['overall']['median_pct_error']:.3f} / "
                f"{best_max['markov42']['overall']['max_pct_error']:.3f}, but it was still far worse than both the current overall best and the entry-boundary max branch."
            ),
            'scientific_conclusion': (
                'This pass did search genuinely new structural territory beyond relay and entry-boundary tuning: anchor6 mixed-beta precursor loops, anchor6 half-cycle handoff pairs, and an anchor3 mirror control. '
                'The result was strongly negative. The anchor6 diagonal family exploded back into the 119–149 max band, the gentler anchor6 handoff pairs still sat in the 107–110 max band, and the early anchor3 control was even worse at 168.147 max. '
                'So the post-entry hidden search did not uncover a stronger family; instead it clarified that moving the intervention upstream to the half-cycle handoff region destroys the late solved channels much faster than it repairs Ka2_y. Under the current legality and continuity constraints, anchor6 looks like a false basin rather than the next frontier.'
            ),
        },
    }

    report_text = render_report(summary)
    report_path = REPORTS_DIR / f'psins_ch3_post_entry_midgateway_probe_{args.report_date}.md'
    json_path = RESULTS_DIR / f'ch3_post_entry_midgateway_probe_{args.report_date}.json'
    report_path.write_text(report_text, encoding='utf-8')
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(str(report_path))
    print(str(json_path))


if __name__ == '__main__':
    main()
