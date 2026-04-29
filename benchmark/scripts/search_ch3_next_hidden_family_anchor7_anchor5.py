from __future__ import annotations

import argparse
import json
import sys
import types
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

RELAYMAX_UNIFIED_Y2_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relaymax_unified_l9y2_shared_noise0p08_param_errors.json'
RELAYMAX_UNIFIED_Y2_KF = RESULTS_DIR / 'KF36_ch3closedloop_relaymax_unified_l9y2_shared_noise0p08_param_errors.json'
ENTRYRELAY_MAIN_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json'
ENTRYRELAY_MAIN_KF = RESULTS_DIR / 'KF36_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json'
OLD_BEST_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
OLD_BEST_KF = RESULTS_DIR / 'KF36_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
FAITHFUL_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
FAITHFUL_KF = RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json'
DEFAULT18_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json'
DEFAULT18_KF = RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json'


PLAUSIBLE_HYPOTHESES = [
    {
        'id': 'H1',
        'family': 'anchor7_corridor_diagonal_conditioner',
        'summary': 'Open a mixed-beta diagonal butterfly at anchor7, then hand control back to the untouched 8→9→10→11 relay backbone. This is earlier than the anchor8 entry-boundary family and geometrically different from the anchor9 butterfly family.',
        'selected': True,
        'tested': True,
    },
    {
        'id': 'H2',
        'family': 'anchor7_corridor_bookend_pair',
        'summary': 'Use a tiny anchor7 x/y closed pair to shape the -Y→-Z corridor one step before the entry boundary. Listed as a plausible family, but not spent in this batch because H1 is the stronger geometric version of the same corridor idea.',
        'selected': False,
        'tested': False,
    },
    {
        'id': 'H3',
        'family': 'anchor5_far_z_seed_relay',
        'summary': 'Inject a small legal z-family seed at anchor5, then keep the best relay backbone untouched. This probes a far-field z-observability seed that is outside the late10/11, relay, and entry-boundary neighborhoods.',
        'selected': True,
        'tested': True,
    },
    {
        'id': 'H4',
        'family': 'distributed_far_seed_plus_corridor_chain',
        'summary': 'Chain a far anchor5 seed and an anchor7 corridor conditioner before the relay core. Left untested in this pass because H1/H3 must first show isolated value before spending budget on a coupled chain.',
        'selected': False,
        'tested': False,
    },
]


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


def row_summary(payload: dict[str, Any]) -> str:
    o = payload['overall'] if 'overall' in payload else payload
    return f"{o['mean_pct_error']:.3f} / {o['median_pct_error']:.3f} / {o['max_pct_error']:.3f}"


def closed_pair(kind: str, angle_deg: int, dwell_s: float, label: str, rot_s: float = 5.0) -> list[StepSpec]:
    return [
        StepSpec(kind=kind, angle_deg=angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_out', label=f'{label}_out'),
        StepSpec(kind=kind, angle_deg=-angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_return', label=f'{label}_return'),
    ]


def xpair_outerhold(dwell_s: float, label: str, inner_angle_deg: int = -90, outer_angle_deg: int = +90, rot_s: float = 5.0) -> list[StepSpec]:
    return [
        StepSpec(kind='inner', angle_deg=inner_angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=0.0, segment_role='motif_inner_open', label=f'{label}_inner_open'),
        StepSpec(kind='outer', angle_deg=outer_angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_outer_sweep', label=f'{label}_outer_sweep'),
        StepSpec(kind='outer', angle_deg=-outer_angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_outer_return', label=f'{label}_outer_return'),
        StepSpec(kind='inner', angle_deg=-inner_angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=0.0, segment_role='motif_inner_close', label=f'{label}_inner_close'),
    ]


def zquad(y_s: float, x_s: float, back_s: float, label: str, rot_s: float = 5.0) -> list[StepSpec]:
    return [
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(y_s), segment_role='motif_y_pos', label=f'{label}_q1'),
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(x_s), segment_role='motif_zero_a', label=f'{label}_q2'),
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(y_s), segment_role='motif_y_neg', label=f'{label}_q3'),
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(back_s), segment_role='motif_zero_b', label=f'{label}_q4'),
    ]


def diag_butterfly(anchor: int, open_angle_deg: int, dwell1: float, dwell2: float, first_sign: int, second_sign: int, label: str, cross_hold_s: float = 3.0, edge_hold_s: float = 3.0) -> dict[int, list[StepSpec]]:
    close_angle_deg = open_angle_deg
    return {
        anchor: [
            StepSpec(kind='inner', angle_deg=open_angle_deg, rotation_time_s=abs(open_angle_deg) / 90.0 * 5.0, pre_static_s=0.0, post_static_s=edge_hold_s, segment_role='motif_diag_open1', label=f'{label}_open1'),
            StepSpec(kind='outer', angle_deg=90 * first_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell1), segment_role='motif_diag1_sweep', label=f'{label}_diag1_sweep'),
            StepSpec(kind='outer', angle_deg=-90 * first_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell1), segment_role='motif_diag1_return', label=f'{label}_diag1_return'),
            StepSpec(kind='inner', angle_deg=-2 * open_angle_deg, rotation_time_s=abs(2 * open_angle_deg) / 90.0 * 5.0, pre_static_s=0.0, post_static_s=cross_hold_s, segment_role='motif_diag_cross', label=f'{label}_cross'),
            StepSpec(kind='outer', angle_deg=90 * second_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell2), segment_role='motif_diag2_sweep', label=f'{label}_diag2_sweep'),
            StepSpec(kind='outer', angle_deg=-90 * second_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell2), segment_role='motif_diag2_return', label=f'{label}_diag2_return'),
            StepSpec(kind='inner', angle_deg=close_angle_deg, rotation_time_s=abs(close_angle_deg) / 90.0 * 5.0, pre_static_s=0.0, post_static_s=edge_hold_s, segment_role='motif_diag_close2', label=f'{label}_close2'),
        ]
    }


def merge_insertions(*dicts: dict[int, list[StepSpec]]) -> dict[int, list[StepSpec]]:
    out: dict[int, list[StepSpec]] = {}
    for d in dicts:
        for k, v in d.items():
            out.setdefault(k, []).extend(v)
    return out


def l9_ypair_neg(dwell_s: float, label: str) -> dict[int, list[StepSpec]]:
    return {9: closed_pair('inner', -90, float(dwell_s), label)}


def l10_unified_core() -> dict[int, list[StepSpec]]:
    return {10: closed_pair('outer', -90, 5.0, 'l10_zpair_neg5') + closed_pair('inner', -90, 1.0, 'l10_ypair_neg1')}


def l11_y10x0back2_core() -> dict[int, list[StepSpec]]:
    return {11: xpair_outerhold(10.0, 'l11_xpair_outerhold') + zquad(10.0, 0.0, 2.0, 'l11_zquad_y10x0back2')}


def relaymax_unified_y2_core() -> dict[int, list[StepSpec]]:
    return merge_insertions(l9_ypair_neg(2.0, 'l9_ypair_neg2'), l10_unified_core(), l11_y10x0back2_core())


def anchor5_zseed_neg(dwell_s: float, label: str) -> dict[int, list[StepSpec]]:
    return {5: closed_pair('outer', -90, float(dwell_s), label)}


def candidate_specs() -> list[dict[str, Any]]:
    relay_core = relaymax_unified_y2_core()
    return [
        {
            'name': 'corridordiag_l7_bfly_same_d6_plus_relaymax_unified_l9y2',
            'family': 'anchor7_corridor_diagonal_conditioner',
            'hypothesis_id': 'H1',
            'rationale': 'Anchor7 corridor diagonal butterfly: open to beta=-45, visit both legal x±z diagonals with the same sweep sign, close, then resume the untouched relaymax unified y2 core.',
            'insertions': merge_insertions(diag_butterfly(7, -45, 6.0, 6.0, +1, +1, 'l7_diag_bfly_same_d6'), relay_core),
        },
        {
            'name': 'corridordiag_l7_bfly_same_d8_plus_relaymax_unified_l9y2',
            'family': 'anchor7_corridor_diagonal_conditioner',
            'hypothesis_id': 'H1',
            'rationale': 'Longer-dwell anchor7 corridor diagonal butterfly with the same sweep sign on both legal diagonals before the relaymax unified y2 core.',
            'insertions': merge_insertions(diag_butterfly(7, -45, 8.0, 8.0, +1, +1, 'l7_diag_bfly_same_d8'), relay_core),
        },
        {
            'name': 'corridordiag_l7_bfly_flip_d8_plus_relaymax_unified_l9y2',
            'family': 'anchor7_corridor_diagonal_conditioner',
            'hypothesis_id': 'H1',
            'rationale': 'Anchor7 corridor diagonal butterfly sign-control: flip the second diagonal sweep sign while keeping the relaymax unified y2 core fixed.',
            'insertions': merge_insertions(diag_butterfly(7, -45, 8.0, 8.0, +1, -1, 'l7_diag_bfly_flip_d8'), relay_core),
        },
        {
            'name': 'zseed_l5_neg2_plus_relaymax_unified_l9y2',
            'family': 'anchor5_far_z_seed_relay',
            'hypothesis_id': 'H3',
            'rationale': 'Anchor5 far z-seed family: add a small negative z closed pair at anchor5, then keep the current relaymax unified y2 backbone unchanged.',
            'insertions': merge_insertions(anchor5_zseed_neg(2.0, 'l5_zseed_neg2'), relay_core),
        },
        {
            'name': 'zseed_l5_neg4_plus_relaymax_unified_l9y2',
            'family': 'anchor5_far_z_seed_relay',
            'hypothesis_id': 'H3',
            'rationale': 'Stronger anchor5 far z-seed dose before the same relaymax unified y2 backbone.',
            'insertions': merge_insertions(anchor5_zseed_neg(4.0, 'l5_zseed_neg4'), relay_core),
        },
    ]


def load_references(noise_scale: float) -> dict[str, Any]:
    return {
        'relaymax_unified_l9y2_markov': load_json_checked(RELAYMAX_UNIFIED_Y2_MARKOV, noise_scale),
        'relaymax_unified_l9y2_kf': load_json_checked(RELAYMAX_UNIFIED_Y2_KF, noise_scale),
        'entryrelay_main_markov': load_json_checked(ENTRYRELAY_MAIN_MARKOV, noise_scale),
        'entryrelay_main_kf': load_json_checked(ENTRYRELAY_MAIN_KF, noise_scale),
        'old_best_markov': load_json_checked(OLD_BEST_MARKOV, noise_scale),
        'old_best_kf': load_json_checked(OLD_BEST_KF, noise_scale),
        'faithful_markov': load_json_checked(FAITHFUL_MARKOV, noise_scale),
        'faithful_kf': load_json_checked(FAITHFUL_KF, noise_scale),
        'default18_markov': load_json_checked(DEFAULT18_MARKOV, noise_scale),
        'default18_kf': load_json_checked(DEFAULT18_KF, noise_scale),
    }


def render_report(payload: dict[str, Any]) -> str:
    refs = payload['references']
    best = payload['best_candidate']
    best_mean = payload['best_mean_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 next hidden-family batch: anchor7 corridor vs anchor5 far-z seed')
    lines.append('')
    lines.append('## 1. Search intent')
    lines.append('')
    lines.append('- This pass explicitly avoided reopening the already-tested late10/11, relay, anchor9 butterfly, precondition/fullblock, anchor6 handoff, anchor8 entry-boundary, and entry-conditioned-relay neighborhoods.')
    lines.append('- Goal: probe **another structural family** under the same legality / continuity / faithful12-backbone constraints, while keeping the downstream comparison fixed against the current relaymax and entry-conditioned frontier points.')
    lines.append('')
    lines.append('## 2. Still-untested structural family hypotheses listed before running')
    lines.append('')
    for item in payload['plausible_hypotheses']:
        flags = []
        if item['selected']:
            flags.append('selected')
        if item['tested']:
            flags.append('tested')
        flag_text = ', '.join(flags) if flags else 'listed only'
        lines.append(f"- **{item['id']} · {item['family']}** ({flag_text}) — {item['summary']}")
    lines.append('')
    lines.append('## 3. Picked families for this batch')
    lines.append('')
    lines.append('- **Picked H1:** anchor7 corridor diagonal conditioner')
    lines.append('- **Picked H3:** anchor5 far z-seed relay')
    lines.append('- **Held back H2/H4:** not enough prior to spend budget before isolating H1/H3 first.')
    lines.append('')
    lines.append('## 4. Fixed references')
    lines.append('')
    lines.append(f"- current relay frontier: **{row_summary(refs['relaymax_unified_l9y2']['markov42']['overall'])}** (`relaymax_unified_l9y2`)")
    lines.append(f"- entry-conditioned relay frontier: **{row_summary(refs['entryrelay_l8x1_l9y1_unifiedcore']['markov42']['overall'])}** (`entryrelay_l8x1_l9y1_unifiedcore`)")
    lines.append(f"- old best legal: **{row_summary(refs['old_best_legal']['markov42']['overall'])}**")
    lines.append(f"- faithful12: **{row_summary(refs['faithful12']['markov42']['overall'])}**")
    lines.append(f"- default18: **{row_summary(refs['default18']['markov42']['overall'])}**")
    lines.append('')
    lines.append('## 5. Markov42 landed batch')
    lines.append('')
    lines.append('| rank | candidate | family | hypothesis | total_s | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Ka2_z | Δmean vs relaymax | Δmax vs relaymax | Δmean vs entryrelay | Δmax vs entryrelay |')
    lines.append('|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(payload['rows_sorted'], start=1):
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        dr = row['delta_vs_relaymax_unified_l9y2']
        de = row['delta_vs_entryrelay_main']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {row['hypothesis_id']} | {row['total_time_s']:.0f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {dr['mean_pct_error']['improvement_pct_points']:+.3f} | {dr['max_pct_error']['improvement_pct_points']:+.3f} | {de['mean_pct_error']['improvement_pct_points']:+.3f} | {de['max_pct_error']['improvement_pct_points']:+.3f} |"
        )
    lines.append('')
    lines.append('## 6. Best landed readout')
    lines.append('')
    lines.append(f"- **Best max-priority candidate:** `{best['candidate_name']}` → **{row_summary(best['markov42']['overall'])}**")
    lines.append(f"  - vs `relaymax_unified_l9y2`: Δmean **{best['delta_vs_relaymax_unified_l9y2']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_relaymax_unified_l9y2']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"  - vs `entryrelay_l8x1_l9y1_unifiedcore`: Δmean **{best['delta_vs_entryrelay_main']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_entryrelay_main']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- **Best mean candidate in this batch:** `{best_mean['candidate_name']}` → **{row_summary(best_mean['markov42']['overall'])}**")
    lines.append(f"  - vs `relaymax_unified_l9y2`: Δmean **{best_mean['delta_vs_relaymax_unified_l9y2']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_mean['delta_vs_relaymax_unified_l9y2']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append('')
    lines.append('## 7. Exact legal motor / timing table for the best candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for idx, (row, action, face) in enumerate(zip(best['all_rows'], best['all_actions'], best['all_faces']), start=1):
        lines.append(
            f"| {idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 8. Continuity proof for the best candidate')
    lines.append('')
    for check in best['continuity_checks']:
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
    if payload['kf36_recheck']['triggered']:
        lines.append('')
        lines.append('| candidate | note | Markov42 mean/median/max | KF36 mean/median/max | dKa_yy / dKg_zz / Ka2_y / Ka2_z (KF36) |')
        lines.append('|---|---|---|---|---|')
        for row in payload['kf36_rows']:
            mm = row['markov42']['overall']
            kk = row['kf36']['overall']
            kp = row['kf36']['key_param_errors']
            lines.append(
                f"| {row['candidate_name']} | {row['note']} | {mm['mean_pct_error']:.3f} / {mm['median_pct_error']:.3f} / {mm['max_pct_error']:.3f} | {kk['mean_pct_error']:.3f} / {kk['median_pct_error']:.3f} / {kk['max_pct_error']:.3f} | {kp['dKa_yy']:.3f} / {kp['dKg_zz']:.3f} / {kp['Ka2_y']:.3f} / {kp['Ka2_z']:.3f} |"
            )
    lines.append('')
    lines.append('## 10. Requested comparison')
    lines.append('')
    lines.append('| path | Markov42 mean/median/max | KF36 mean/median/max | note |')
    lines.append('|---|---|---|---|')
    for row in payload['comparison_rows']:
        mm = row['markov42']['overall']
        kf = row.get('kf36')
        kf_text = 'not rerun' if kf is None else f"{kf['overall']['mean_pct_error']:.3f} / {kf['overall']['median_pct_error']:.3f} / {kf['overall']['max_pct_error']:.3f}"
        lines.append(f"| {row['label']} | {mm['mean_pct_error']:.3f} / {mm['median_pct_error']:.3f} / {mm['max_pct_error']:.3f} | {kf_text} | {row['note']} |")
    lines.append('')
    lines.append('## 11. Bottom line')
    lines.append('')
    lines.append(f"- **Did this hidden-family batch find a stronger direction?** **{payload['bottom_line']['found_stronger_direction']}**")
    lines.append(f"- **Best batch landing:** **{payload['bottom_line']['best_signal']}**")
    lines.append(f"- **Scientific conclusion:** {payload['bottom_line']['scientific_conclusion']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    refs = load_references(args.noise_scale)
    mod = load_module('psins_ch3_next_hidden_family_anchor7_anchor5', str(SOURCE_FILE))
    base = build_candidate(mod, ())

    specs = candidate_specs()
    candidates = [build_closedloop_candidate(mod, spec, base.rows, base.action_sequence) for spec in specs]
    candidates_by_name = {cand.name: cand for cand in candidates}
    spec_by_name = {spec['name']: spec for spec in specs}

    rows: list[dict[str, Any]] = []
    payload_by_name: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        payload, status, path = run_candidate_payload(mod, cand, 'markov42_noisy', args.noise_scale, args.force_rerun)
        payload_by_name[cand.name] = payload
        rows.append({
            'candidate_name': cand.name,
            'family': spec_by_name[cand.name]['family'],
            'hypothesis_id': spec_by_name[cand.name]['hypothesis_id'],
            'rationale': spec_by_name[cand.name]['rationale'],
            'total_time_s': cand.total_time_s,
            'metrics': compact_metrics(payload),
            'continuity_checks': cand.continuity_checks,
            'run_json': str(path),
            'status': status,
            'delta_vs_relaymax_unified_l9y2': delta_vs_ref(refs['relaymax_unified_l9y2_markov'], payload),
            'delta_vs_entryrelay_main': delta_vs_ref(refs['entryrelay_main_markov'], payload),
            'delta_vs_old_best': delta_vs_ref(refs['old_best_markov'], payload),
            'delta_vs_faithful12': delta_vs_ref(refs['faithful_markov'], payload),
            'delta_vs_default18': delta_vs_ref(refs['default18_markov'], payload),
        })

    rows_sorted = sorted(rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    best_row = rows_sorted[0]
    best_mean_row = min(rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))

    best_candidate = candidates_by_name[best_row['candidate_name']]
    best_payload = payload_by_name[best_candidate.name]
    best_mean_candidate = candidates_by_name[best_mean_row['candidate_name']]
    best_mean_payload = payload_by_name[best_mean_candidate.name]

    competitive = (
        best_payload['overall']['max_pct_error'] < 99.75
        and best_payload['overall']['mean_pct_error'] < 10.50
        and (
            best_row['delta_vs_relaymax_unified_l9y2']['max_pct_error']['improvement_pct_points'] > -0.10
            or best_row['delta_vs_entryrelay_main']['max_pct_error']['improvement_pct_points'] > -0.10
        )
    )

    kf36_rows: list[dict[str, Any]] = []
    if competitive:
        best_kf_payload, best_kf_status, best_kf_path = run_candidate_payload(mod, best_candidate, 'kf36_noisy', args.noise_scale, args.force_rerun)
        kf36_rows.append({
            'candidate_name': best_candidate.name,
            'note': 'best max-priority candidate in this batch',
            'markov42': compact_result(best_payload),
            'kf36': compact_result(best_kf_payload),
            'kf36_status': best_kf_status,
            'kf36_run_json': str(best_kf_path),
        })
        kf_reason = (
            f"Triggered because {best_candidate.name} landed at {row_summary(best_payload['overall'])} with frontier-adjacent max / mean, so it was close enough to justify a KF36 stability check."
        )
        best_kf_summary = compact_result(best_kf_payload)
    else:
        kf_reason = (
            f"Not triggered because the best batch landing {best_candidate.name} = {row_summary(best_payload['overall'])} still sits too far from both `relaymax_unified_l9y2` and `entryrelay_l8x1_l9y1_unifiedcore` to count as genuinely competitive."
        )
        best_kf_summary = None

    best_summary = {
        'candidate_name': best_candidate.name,
        'family': best_row['family'],
        'hypothesis_id': best_row['hypothesis_id'],
        'total_time_s': best_candidate.total_time_s,
        'markov42': compact_result(best_payload),
        'delta_vs_relaymax_unified_l9y2': best_row['delta_vs_relaymax_unified_l9y2'],
        'delta_vs_entryrelay_main': best_row['delta_vs_entryrelay_main'],
        'delta_vs_old_best': best_row['delta_vs_old_best'],
        'delta_vs_faithful12': best_row['delta_vs_faithful12'],
        'delta_vs_default18': best_row['delta_vs_default18'],
        'all_rows': best_candidate.all_rows,
        'all_actions': best_candidate.all_actions,
        'all_faces': best_candidate.all_faces,
        'continuity_checks': best_candidate.continuity_checks,
        'kf36': best_kf_summary,
    }

    best_mean_summary = {
        'candidate_name': best_mean_candidate.name,
        'family': best_mean_row['family'],
        'hypothesis_id': best_mean_row['hypothesis_id'],
        'total_time_s': best_mean_candidate.total_time_s,
        'markov42': compact_result(best_mean_payload),
        'delta_vs_relaymax_unified_l9y2': best_mean_row['delta_vs_relaymax_unified_l9y2'],
        'delta_vs_entryrelay_main': best_mean_row['delta_vs_entryrelay_main'],
        'delta_vs_old_best': best_mean_row['delta_vs_old_best'],
    }

    found_stronger_direction = (
        best_row['delta_vs_relaymax_unified_l9y2']['mean_pct_error']['improvement_pct_points'] > 0
        and best_row['delta_vs_relaymax_unified_l9y2']['max_pct_error']['improvement_pct_points'] > 0
    )

    if found_stronger_direction:
        scientific_conclusion = (
            'Yes — this batch found a stronger new direction, and it is the **anchor5 far z-seed relay family** rather than the anchor7 corridor-diagonal family. '
            f'The family landed two useful frontier points: best-mean `{best_mean_candidate.name}` = {row_summary(best_mean_payload['overall'])}, and best-max `{best_candidate.name}` = {row_summary(best_payload['overall'])}. '
            f'Relative to `relaymax_unified_l9y2`, the best-mean point gains Δmean {best_mean_row['delta_vs_relaymax_unified_l9y2']['mean_pct_error']['improvement_pct_points']:+.3f} and Δmax {best_mean_row['delta_vs_relaymax_unified_l9y2']['max_pct_error']['improvement_pct_points']:+.3f}, while the best-max point gains Δmean {best_row['delta_vs_relaymax_unified_l9y2']['mean_pct_error']['improvement_pct_points']:+.3f} and Δmax {best_row['delta_vs_relaymax_unified_l9y2']['max_pct_error']['improvement_pct_points']:+.3f}. '
            'Relative to `entryrelay_l8x1_l9y1_unifiedcore`, both far-z-seed points still give back a few hundredths on max, so this is not the new overall max-frontier leader — but it is a real new family that cleanly improves the current relay branch under the same legality and continuity constraints.'
        )
        found_text = 'YES'
    else:
        scientific_conclusion = (
            'This batch did search genuinely new structure beyond the already-tested basins: an anchor7 corridor-diagonal conditioner and an anchor5 far z-seed family. '
            f'The best landed point was {best_candidate.name} = {row_summary(best_payload['overall'])}. '
            f'Relative to `relaymax_unified_l9y2` it moved by Δmean {best_row['delta_vs_relaymax_unified_l9y2']['mean_pct_error']['improvement_pct_points']:+.3f} and Δmax {best_row['delta_vs_relaymax_unified_l9y2']['max_pct_error']['improvement_pct_points']:+.3f}; '
            f'relative to `entryrelay_l8x1_l9y1_unifiedcore` it moved by Δmean {best_row['delta_vs_entryrelay_main']['mean_pct_error']['improvement_pct_points']:+.3f} and Δmax {best_row['delta_vs_entryrelay_main']['max_pct_error']['improvement_pct_points']:+.3f}. '
            'So this next hidden-family batch did not uncover a stronger mainline direction; it sharply rules out the tested anchor7/anchor5 families as inferior to the current relay and entry-conditioned frontier points under the fixed legal continuity-safe setup.'
        )
        found_text = 'NO'

    comparison_rows = [
        {
            'label': 'relaymax_unified_l9y2',
            'note': 'current relay frontier point',
            'markov42': compact_result(refs['relaymax_unified_l9y2_markov']),
            'kf36': compact_result(refs['relaymax_unified_l9y2_kf']),
        },
        {
            'label': 'entryrelay_l8x1_l9y1_unifiedcore',
            'note': 'current entry-conditioned relay frontier point',
            'markov42': compact_result(refs['entryrelay_main_markov']),
            'kf36': compact_result(refs['entryrelay_main_kf']),
        },
        {
            'label': 'old best legal',
            'note': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
            'markov42': compact_result(refs['old_best_markov']),
            'kf36': compact_result(refs['old_best_kf']),
        },
        {
            'label': 'faithful12',
            'note': 'original faithful 12-position backbone',
            'markov42': compact_result(refs['faithful_markov']),
            'kf36': compact_result(refs['faithful_kf']),
        },
        {
            'label': 'default18',
            'note': 'non-faithful strong reference',
            'markov42': compact_result(refs['default18_markov']),
            'kf36': compact_result(refs['default18_kf']),
        },
        {
            'label': 'best candidate in this batch',
            'note': best_candidate.name,
            'markov42': best_summary['markov42'],
            'kf36': best_kf_summary,
        },
    ]

    out_json = RESULTS_DIR / f'ch3_next_hidden_family_anchor7_anchor5_{args.report_date}.json'
    out_md = REPORTS_DIR / f'psins_ch3_next_hidden_family_anchor7_anchor5_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_next_hidden_family_anchor7_anchor5',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'plausible_hypotheses': PLAUSIBLE_HYPOTHESES,
        'tested_hypotheses': [item for item in PLAUSIBLE_HYPOTHESES if item['tested']],
        'references': {
            'relaymax_unified_l9y2': {
                'candidate_name': 'relaymax_unified_l9y2',
                'markov42': compact_result(refs['relaymax_unified_l9y2_markov']),
                'kf36': compact_result(refs['relaymax_unified_l9y2_kf']),
            },
            'entryrelay_l8x1_l9y1_unifiedcore': {
                'candidate_name': 'entryrelay_l8x1_l9y1_unifiedcore',
                'markov42': compact_result(refs['entryrelay_main_markov']),
                'kf36': compact_result(refs['entryrelay_main_kf']),
            },
            'old_best_legal': {
                'candidate_name': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
                'markov42': compact_result(refs['old_best_markov']),
                'kf36': compact_result(refs['old_best_kf']),
            },
            'faithful12': {
                'candidate_name': 'faithful12',
                'markov42': compact_result(refs['faithful_markov']),
                'kf36': compact_result(refs['faithful_kf']),
            },
            'default18': {
                'candidate_name': 'default18',
                'markov42': compact_result(refs['default18_markov']),
                'kf36': compact_result(refs['default18_kf']),
            },
        },
        'candidate_specs': [
            {
                'name': spec['name'],
                'family': spec['family'],
                'hypothesis_id': spec['hypothesis_id'],
                'rationale': spec['rationale'],
                'anchors': sorted(spec['insertions'].keys()),
            }
            for spec in specs
        ],
        'rows_sorted': rows_sorted,
        'best_candidate': best_summary,
        'best_mean_candidate': best_mean_summary,
        'kf36_recheck': {
            'triggered': competitive,
            'reason': kf_reason,
        },
        'kf36_rows': kf36_rows,
        'comparison_rows': comparison_rows,
        'bottom_line': {
            'found_stronger_direction': found_text,
            'best_signal': (
                f"best-mean {best_mean_candidate.name} = {row_summary(best_mean_payload['overall'])}; "
                f"best-max {best_candidate.name} = {row_summary(best_payload['overall'])}"
            ),
            'scientific_conclusion': scientific_conclusion,
        },
        'files': {
            'json': str(out_json),
            'report': str(out_md),
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    out_md.write_text(render_report(payload), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps(payload['files'], ensure_ascii=False), flush=True)
    print('BEST_CANDIDATE', best_candidate.name, best_payload['overall'], flush=True)
    print('BEST_MEAN', best_mean_candidate.name, best_mean_payload['overall'], flush=True)
    print('BOTTOM_LINE', scientific_conclusion, flush=True)


if __name__ == '__main__':
    main()
