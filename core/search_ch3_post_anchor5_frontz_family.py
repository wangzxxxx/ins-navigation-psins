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

CURRENT_MAINLINE_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_zseed_l5_neg6_plus_relaymax_unified_l9y2p5_shared_noise0p08_param_errors.json'
CURRENT_MAINLINE_KF = RESULTS_DIR / 'KF36_ch3closedloop_zseed_l5_neg6_plus_relaymax_unified_l9y2p5_shared_noise0p08_param_errors.json'
ENTRY_FRONTIER_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json'
ENTRY_FRONTIER_KF = RESULTS_DIR / 'KF36_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json'
OLD_BEST_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
OLD_BEST_KF = RESULTS_DIR / 'KF36_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
FAITHFUL_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
FAITHFUL_KF = RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json'
DEFAULT18_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json'
DEFAULT18_KF = RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json'

CURRENT_MAINLINE_NAME = 'zseed_l5_neg6_plus_relaymax_unified_l9y2p5'
ENTRY_FRONTIER_NAME = 'entryrelay_l8x1_l9y1_unifiedcore'

HYPOTHESES = [
    {
        'id': 'H1',
        'family': 'anchor4_front_z_precursor_seed',
        'summary': 'At anchor4, add a small legal z-family closed pair before the 4→5→6 front-half z corridor completes, then hand off to the untouched late relay core. This is earlier than the anchor5 far-z family and does not reuse the anchor8 / anchor9 / anchor10-11 neighborhoods.',
        'selected': True,
        'tested': True,
    },
    {
        'id': 'H2',
        'family': 'anchor4_front_z_diagonal_butterfly',
        'summary': 'At anchor4, open the real inner axis from beta=+90 to mixed beta and visit the two legal diagonal outer axes inside one exact-return butterfly before resuming the base path. This is a front-half mixed-beta family, distinct from the tested anchor9 butterfly basin.',
        'selected': True,
        'tested': True,
    },
    {
        'id': 'H3',
        'family': 'front_half_mirror_x_boundary_control',
        'summary': 'Mirror-style early x-boundary control around anchor2/3. Kept listed but not spent because it is less mechanistically coupled to the successful late relay backbone than H1/H2.',
        'selected': False,
        'tested': False,
    },
    {
        'id': 'H4',
        'family': 'distributed_front_corridor_chain',
        'summary': 'A coupled two-anchor front-half corridor chain across anchors4 and5. Held back until H1/H2 show isolated value, to avoid reopening the anchor5 basin without evidence.',
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


def dose_tag(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace('.', 'p')


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


def l10_core(y_dwell_s: float = 1.0) -> dict[int, list[StepSpec]]:
    return {10: closed_pair('outer', -90, 5.0, 'l10_zpair_neg5') + closed_pair('inner', -90, float(y_dwell_s), f'l10_ypair_neg{dose_tag(y_dwell_s)}')}


def l11_core(back_s: float = 2.0) -> dict[int, list[StepSpec]]:
    return {11: xpair_outerhold(10.0, 'l11_xpair_outerhold') + zquad(10.0, 0.0, float(back_s), f'l11_zquad_y10x0back{dose_tag(back_s)}')}


def relay_core(l9_dwell_s: float = 2.5, l10_y_dwell_s: float = 1.0, l11_back_s: float = 2.0) -> dict[int, list[StepSpec]]:
    return merge_insertions(
        l9_ypair_neg(l9_dwell_s, f'l9_ypair_neg{dose_tag(l9_dwell_s)}'),
        l10_core(l10_y_dwell_s),
        l11_core(l11_back_s),
    )


def anchor4_zseed(dose_abs_s: float, sign: str, label: str) -> dict[int, list[StepSpec]]:
    angle = -90 if sign == 'neg' else +90
    return {4: closed_pair('outer', angle, float(dose_abs_s), label)}


def make_spec(name: str, family: str, hypothesis_id: str, rationale: str, insertions: dict[int, list[StepSpec]], *, front_dose_s: float | None = None, front_mode: str = 'none', l9_dwell_s: float = 2.5) -> dict[str, Any]:
    return {
        'name': name,
        'family': family,
        'hypothesis_id': hypothesis_id,
        'rationale': rationale,
        'front_dose_s': front_dose_s,
        'front_mode': front_mode,
        'l9_dwell_s': l9_dwell_s,
        'insertions': insertions,
    }


def candidate_specs() -> list[dict[str, Any]]:
    core = relay_core(2.5)
    return [
        make_spec(
            'frontzseed_l4_neg2_plus_relaymax_unified_l9y2p5',
            'anchor4_front_z_precursor_seed',
            'H1',
            'Small anchor4 front-z precursor seed before the front-half z corridor completes; late relay core stays fixed.',
            merge_insertions(anchor4_zseed(2.0, 'neg', 'l4_zseed_neg2'), core),
            front_dose_s=2.0,
            front_mode='zseed_neg',
        ),
        make_spec(
            'frontzseed_l4_neg4_plus_relaymax_unified_l9y2p5',
            'anchor4_front_z_precursor_seed',
            'H1',
            'Medium anchor4 front-z precursor seed with the same untouched late relay core.',
            merge_insertions(anchor4_zseed(4.0, 'neg', 'l4_zseed_neg4'), core),
            front_dose_s=4.0,
            front_mode='zseed_neg',
        ),
        make_spec(
            'frontzseed_l4_neg6_plus_relaymax_unified_l9y2p5',
            'anchor4_front_z_precursor_seed',
            'H1',
            'Strong anchor4 front-z precursor seed to test whether the family has a usable dose ridge before the anchor5 basin even begins.',
            merge_insertions(anchor4_zseed(6.0, 'neg', 'l4_zseed_neg6'), core),
            front_dose_s=6.0,
            front_mode='zseed_neg',
        ),
        make_spec(
            'frontdiag_l4_bfly_same_d6_plus_relaymax_unified_l9y2p5',
            'anchor4_front_z_diagonal_butterfly',
            'H2',
            'Anchor4 mixed-beta diagonal butterfly, same-sign sweeps on the two legal front-half diagonals before the unchanged late relay core.',
            merge_insertions(diag_butterfly(4, -45, 6.0, 6.0, +1, +1, 'l4_diag_bfly_same_d6'), core),
            front_mode='diag_bfly_same',
        ),
        make_spec(
            'frontdiag_l4_bfly_flip_d6_plus_relaymax_unified_l9y2p5',
            'anchor4_front_z_diagonal_butterfly',
            'H2',
            'Anchor4 mixed-beta diagonal butterfly, sign-flipped second diagonal sweep as the control variant.',
            merge_insertions(diag_butterfly(4, -45, 6.0, 6.0, +1, -1, 'l4_diag_bfly_flip_d6'), core),
            front_mode='diag_bfly_flip',
        ),
    ]


def load_references(noise_scale: float) -> dict[str, Any]:
    return {
        'current_mainline_markov': load_json_checked(CURRENT_MAINLINE_MARKOV, noise_scale),
        'current_mainline_kf': load_json_checked(CURRENT_MAINLINE_KF, noise_scale),
        'entry_frontier_markov': load_json_checked(ENTRY_FRONTIER_MARKOV, noise_scale),
        'entry_frontier_kf': load_json_checked(ENTRY_FRONTIER_KF, noise_scale),
        'old_best_markov': load_json_checked(OLD_BEST_MARKOV, noise_scale),
        'old_best_kf': load_json_checked(OLD_BEST_KF, noise_scale),
        'faithful_markov': load_json_checked(FAITHFUL_MARKOV, noise_scale),
        'faithful_kf': load_json_checked(FAITHFUL_KF, noise_scale),
        'default18_markov': load_json_checked(DEFAULT18_MARKOV, noise_scale),
        'default18_kf': load_json_checked(DEFAULT18_KF, noise_scale),
    }


def render_report(payload: dict[str, Any]) -> str:
    best = payload['best_candidate']
    best_mean = payload['best_mean_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 post-anchor5 hidden-family batch: anchor4 front-z corridor')
    lines.append('')
    lines.append('## 1. Search intent')
    lines.append('')
    lines.append('- This pass explicitly avoided reopening the already-tested late10/late11 local basin, relay family, anchor9 butterfly basin, precondition/fullblock family, anchor6 handoff basin, anchor8 entry-boundary family, entry-conditioned relay family, and anchor5 far-z seed family.')
    lines.append('- New objective: probe a **genuinely different front-half family** centered on the anchor4 front-z corridor, while keeping exact dual-axis legality, continuity-safe closure, and the original 12-position backbone.')
    lines.append('')
    lines.append('## 2. Still-untested structural family hypotheses listed before running')
    lines.append('')
    for item in payload['hypotheses']:
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
    lines.append('- **Picked H1:** anchor4 front-z precursor seed')
    lines.append('- **Picked H2:** anchor4 front-z diagonal butterfly')
    lines.append('- **Held back H3/H4:** not enough prior to justify spending budget on weaker mirror-control or coupled-chain versions.')
    lines.append('')
    lines.append('## 4. Markov42 landed batch')
    lines.append('')
    lines.append('| rank | candidate | family | hypothesis | total_s | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Ka2_z | Δmean vs current mainline | Δmax vs current mainline | Δmean vs entry frontier | Δmax vs entry frontier |')
    lines.append('|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(payload['rows_sorted'], start=1):
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        dc = row['delta_vs_current_mainline']
        de = row['delta_vs_entry_frontier']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {row['hypothesis_id']} | {row['total_time_s']:.0f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {dc['mean_pct_error']['improvement_pct_points']:+.3f} | {dc['max_pct_error']['improvement_pct_points']:+.3f} | {de['mean_pct_error']['improvement_pct_points']:+.3f} | {de['max_pct_error']['improvement_pct_points']:+.3f} |"
        )
    lines.append('')
    lines.append('## 5. Best landed readout')
    lines.append('')
    lines.append(f"- **Best max-priority candidate:** `{best['candidate_name']}` → **{row_summary(best['markov42']['overall'])}**")
    lines.append(f"  - vs `{CURRENT_MAINLINE_NAME}`: Δmean **{best['delta_vs_current_mainline']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"  - vs `{ENTRY_FRONTIER_NAME}`: Δmean **{best['delta_vs_entry_frontier']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_entry_frontier']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- **Best mean candidate in this batch:** `{best_mean['candidate_name']}` → **{row_summary(best_mean['markov42']['overall'])}**")
    lines.append(f"  - vs `{CURRENT_MAINLINE_NAME}`: Δmean **{best_mean['delta_vs_current_mainline']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_mean['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append('')
    lines.append('## 6. Exact legal motor / timing table for the best candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for idx, (row, action, face) in enumerate(zip(best['all_rows'], best['all_actions'], best['all_faces']), start=1):
        lines.append(
            f"| {idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 7. Continuity proof for the best candidate')
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
    lines.append('## 8. KF36 recheck gate')
    lines.append('')
    lines.append(f"- triggered: **{payload['kf36_recheck']['triggered']}**")
    lines.append(f"- reason: **{payload['kf36_recheck']['reason']}**")
    if payload['kf36_recheck']['triggered']:
        lines.append('')
        lines.append('| candidate | note | Markov42 mean/median/max | KF36 mean/median/max |')
        lines.append('|---|---|---|---|')
        for row in payload['kf36_rows']:
            lines.append(f"| {row['candidate_name']} | {row['note']} | {row_summary(row['markov42']['overall'])} | {row_summary(row['kf36']['overall'])} |")
    lines.append('')
    lines.append('## 9. Requested comparison')
    lines.append('')
    lines.append('| path | Markov42 mean/median/max | KF36 mean/median/max | note |')
    lines.append('|---|---|---|---|')
    for row in payload['comparison_rows']:
        kf = row.get('kf36')
        kf_text = 'not rerun' if kf is None else row_summary(kf['overall'])
        lines.append(f"| {row['label']} | {row_summary(row['markov42']['overall'])} | {kf_text} | {row['note']} |")
    lines.append('')
    lines.append('## 10. Bottom line')
    lines.append('')
    lines.append(f"- **Did this post-anchor5 hidden-family batch find a stronger direction?** **{payload['bottom_line']['found_stronger_direction']}**")
    lines.append(f"- **Best batch landing:** **{payload['bottom_line']['best_signal']}**")
    lines.append(f"- **Scientific conclusion:** {payload['bottom_line']['scientific_conclusion']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    refs = load_references(args.noise_scale)
    mod = load_module('psins_ch3_post_anchor5_frontz_family', str(SOURCE_FILE))
    base = build_candidate(mod, ())

    specs = candidate_specs()
    candidates = [build_closedloop_candidate(mod, spec, base.rows, base.action_sequence) for spec in specs]
    spec_by_name = {spec['name']: spec for spec in specs}
    cand_by_name = {cand.name: cand for cand in candidates}

    rows: list[dict[str, Any]] = []
    payload_by_name: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        payload, status, path = run_candidate_payload(mod, cand, 'markov42_noisy', args.noise_scale, args.force_rerun)
        payload_by_name[cand.name] = payload
        spec = spec_by_name[cand.name]
        rows.append({
            'candidate_name': cand.name,
            'family': spec['family'],
            'hypothesis_id': spec['hypothesis_id'],
            'rationale': spec['rationale'],
            'front_dose_s': spec['front_dose_s'],
            'front_mode': spec['front_mode'],
            'l9_dwell_s': spec['l9_dwell_s'],
            'total_time_s': cand.total_time_s,
            'metrics': compact_metrics(payload),
            'continuity_checks': cand.continuity_checks,
            'run_json': str(path),
            'status': status,
            'delta_vs_current_mainline': delta_vs_ref(refs['current_mainline_markov'], payload),
            'delta_vs_entry_frontier': delta_vs_ref(refs['entry_frontier_markov'], payload),
            'delta_vs_old_best': delta_vs_ref(refs['old_best_markov'], payload),
            'delta_vs_faithful12': delta_vs_ref(refs['faithful_markov'], payload),
            'delta_vs_default18': delta_vs_ref(refs['default18_markov'], payload),
        })

    rows_sorted = sorted(rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    best_row = rows_sorted[0]
    best_mean_row = min(rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))

    best_candidate = cand_by_name[best_row['candidate_name']]
    best_payload = payload_by_name[best_candidate.name]
    best_mean_candidate = cand_by_name[best_mean_row['candidate_name']]
    best_mean_payload = payload_by_name[best_mean_candidate.name]

    competitive = (
        best_payload['overall']['max_pct_error'] < 99.62
        and best_payload['overall']['mean_pct_error'] < 9.60
        and (
            best_row['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points'] > -0.015
            or best_row['delta_vs_entry_frontier']['max_pct_error']['improvement_pct_points'] > -0.015
        )
    )

    kf36_rows: list[dict[str, Any]] = []
    if competitive:
        kf_payload, kf_status, kf_path = run_candidate_payload(mod, best_candidate, 'kf36_noisy', args.noise_scale, args.force_rerun)
        kf36_rows.append({
            'candidate_name': best_candidate.name,
            'note': 'best frontier-adjacent candidate in this batch',
            'markov42': compact_result(best_payload),
            'kf36': compact_result(kf_payload),
            'kf36_status': kf_status,
            'kf36_run_json': str(kf_path),
        })
        kf_reason = (
            f"Triggered because {best_candidate.name} landed close enough to the current frontiers at {row_summary(best_payload['overall'])}."
        )
        best_kf_summary = compact_result(kf_payload)
    else:
        kf_reason = (
            f"Not triggered because the best landing {best_candidate.name} = {row_summary(best_payload['overall'])} still sits too far from both `{CURRENT_MAINLINE_NAME}` and `{ENTRY_FRONTIER_NAME}` to count as genuinely competitive."
        )
        best_kf_summary = None

    best_summary = {
        'candidate_name': best_candidate.name,
        'family': best_row['family'],
        'hypothesis_id': best_row['hypothesis_id'],
        'total_time_s': best_candidate.total_time_s,
        'markov42': compact_result(best_payload),
        'kf36': best_kf_summary,
        'delta_vs_current_mainline': best_row['delta_vs_current_mainline'],
        'delta_vs_entry_frontier': best_row['delta_vs_entry_frontier'],
        'delta_vs_old_best': best_row['delta_vs_old_best'],
        'delta_vs_faithful12': best_row['delta_vs_faithful12'],
        'delta_vs_default18': best_row['delta_vs_default18'],
        'all_rows': best_candidate.all_rows,
        'all_actions': best_candidate.all_actions,
        'all_faces': best_candidate.all_faces,
        'continuity_checks': best_candidate.continuity_checks,
    }

    best_mean_summary = {
        'candidate_name': best_mean_candidate.name,
        'family': best_mean_row['family'],
        'hypothesis_id': best_mean_row['hypothesis_id'],
        'total_time_s': best_mean_candidate.total_time_s,
        'markov42': compact_result(best_mean_payload),
        'delta_vs_current_mainline': best_mean_row['delta_vs_current_mainline'],
        'delta_vs_entry_frontier': best_mean_row['delta_vs_entry_frontier'],
    }

    found_stronger_direction = (
        best_row['delta_vs_current_mainline']['mean_pct_error']['improvement_pct_points'] > 0
        and best_row['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points'] > 0
    )

    if found_stronger_direction:
        scientific_conclusion = (
            'Yes — this batch found a stronger new front-half direction. '
            f"The best landing `{best_candidate.name}` = {row_summary(best_payload['overall'])} beats `{CURRENT_MAINLINE_NAME}` on both mean and max under the same legality and continuity constraints."
        )
        found_text = 'YES'
    else:
        scientific_conclusion = (
            'This batch did probe genuinely new front-half structure beyond the already-pruned basins: an anchor4 front-z precursor seed family and an anchor4 mixed-beta diagonal butterfly family. '
            f"The best landed point was `{best_candidate.name}` = {row_summary(best_payload['overall'])}. "
            f"Relative to `{CURRENT_MAINLINE_NAME}` it moved by Δmean {best_row['delta_vs_current_mainline']['mean_pct_error']['improvement_pct_points']:+.3f} and Δmax {best_row['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points']:+.3f}; "
            f"relative to `{ENTRY_FRONTIER_NAME}` it moved by Δmean {best_row['delta_vs_entry_frontier']['mean_pct_error']['improvement_pct_points']:+.3f} and Δmax {best_row['delta_vs_entry_frontier']['max_pct_error']['improvement_pct_points']:+.3f}. "
            'So this post-anchor5 hidden-family batch did not uncover a stronger direction: the tested anchor4 front-half families remain clearly behind the current anchor5 mainline and the entry-conditioned max frontier.'
        )
        found_text = 'NO'

    comparison_rows = [
        {
            'label': CURRENT_MAINLINE_NAME,
            'note': 'current unified mainline family point',
            'markov42': compact_result(refs['current_mainline_markov']),
            'kf36': compact_result(refs['current_mainline_kf']),
        },
        {
            'label': ENTRY_FRONTIER_NAME,
            'note': 'current absolute max-frontier rival',
            'markov42': compact_result(refs['entry_frontier_markov']),
            'kf36': compact_result(refs['entry_frontier_kf']),
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

    out_json = RESULTS_DIR / f'ch3_post_anchor5_frontz_family_{args.report_date}.json'
    out_md = REPORTS_DIR / f'psins_ch3_post_anchor5_frontz_family_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_post_anchor5_frontz_family',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'hypotheses': HYPOTHESES,
        'tested_hypotheses': [item for item in HYPOTHESES if item['tested']],
        'references': {
            'current_mainline': {
                'candidate_name': CURRENT_MAINLINE_NAME,
                'markov42': compact_result(refs['current_mainline_markov']),
                'kf36': compact_result(refs['current_mainline_kf']),
            },
            'entry_frontier': {
                'candidate_name': ENTRY_FRONTIER_NAME,
                'markov42': compact_result(refs['entry_frontier_markov']),
                'kf36': compact_result(refs['entry_frontier_kf']),
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
                'front_dose_s': spec['front_dose_s'],
                'front_mode': spec['front_mode'],
                'l9_dwell_s': spec['l9_dwell_s'],
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
                f"best-max {best_candidate.name} = {row_summary(best_payload['overall'])}; "
                f"best-mean {best_mean_candidate.name} = {row_summary(best_mean_payload['overall'])}"
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
