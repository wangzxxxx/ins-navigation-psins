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
    load_reference_payloads,
    render_action,
    run_candidate_payload,
)
from search_ch3_12pos_closedloop_zquad_followup import xpair_outerhold, zquad_split
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate, make_suffix

RELAY_WINNER_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_y9_z10_x11_y9x0_shared_noise0p08_param_errors.json'
UNIFIED_WINNER_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2_shared_noise0p08_param_errors.json'
UNIFIED_WINNER_KF_RESULT = RESULTS_DIR / 'KF36_ch3closedloop_twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2_shared_noise0p08_param_errors.json'
OLD_BEST_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
OLD_BEST_KF_RESULT = RESULTS_DIR / 'KF36_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
DEFAULT18_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json'
DEFAULT18_KF_RESULT = RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json'
ENTRY_ANCHOR_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_entryx_l8_xpair_pos4_plus_l11_y8x0_shared_noise0p08_param_errors.json'
ENTRY_ANCHOR_KF_RESULT = RESULTS_DIR / 'KF36_ch3closedloop_entryx_l8_xpair_pos4_plus_l11_y8x0_shared_noise0p08_param_errors.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def closed_pair(kind: str, angle_deg: int, dwell_s: float, label: str, rot_s: float = 5.0) -> list[StepSpec]:
    return [
        StepSpec(kind=kind, angle_deg=angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_out', label=f'{label}_out'),
        StepSpec(kind=kind, angle_deg=-angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_return', label=f'{label}_return'),
    ]


def l8_xpair_pos(dwell_s: float, label: str) -> dict[int, list[StepSpec]]:
    return {8: closed_pair('outer', +90, dwell_s, label)}


def l11_y10x2(prefix: str = 'l11_entryfollow') -> dict[int, list[StepSpec]]:
    return {11: xpair_outerhold(10.0, f'{prefix}_xpair_outerhold') + zquad_split(10.0, 2.0, 10.0, 2.0, f'{prefix}_zquad_y10x2')}


def l11_y10x1(prefix: str = 'l11_entryfollow') -> dict[int, list[StepSpec]]:
    return {11: xpair_outerhold(10.0, f'{prefix}_xpair_outerhold') + zquad_split(10.0, 1.0, 10.0, 1.0, f'{prefix}_zquad_y10x1')}


def l11_y9x1(prefix: str = 'l11_entryfollow') -> dict[int, list[StepSpec]]:
    return {11: xpair_outerhold(10.0, f'{prefix}_xpair_outerhold') + zquad_split(9.0, 1.0, 9.0, 1.0, f'{prefix}_zquad_y9x1')}


def l11_y10x0back2(prefix: str = 'l11_entryfollow') -> dict[int, list[StepSpec]]:
    return {11: xpair_outerhold(10.0, f'{prefix}_xpair_outerhold') + zquad_split(10.0, 0.0, 10.0, 2.0, f'{prefix}_zquad_y10x0back2')}


def l11_ypos10x1yneg8x1(prefix: str = 'l11_entryfollow') -> dict[int, list[StepSpec]]:
    return {11: xpair_outerhold(10.0, f'{prefix}_xpair_outerhold') + zquad_split(10.0, 1.0, 8.0, 1.0, f'{prefix}_zquad_ypos10x1yneg8x1')}


def merge_insertions(*dicts: dict[int, list[StepSpec]]) -> dict[int, list[StepSpec]]:
    out: dict[int, list[StepSpec]] = {}
    for d in dicts:
        for k, v in d.items():
            out.setdefault(k, []).extend(v)
    return out


CANDIDATE_SPECS = [
    {
        'name': 'entryx_l8_xpair_pos1_plus_l11_y10x2',
        'class': 'minimal_entry_mean_tail',
        'rationale': 'Smallest positive entry-boundary x-bookend that still changes the anchor8→9 entry state, paired with the old strongest mean-oriented late11 core. Tests whether mean loss mainly came from over-driving the entry dose.',
        'insertions': merge_insertions(l8_xpair_pos(1.0, 'l8_xpair_pos1'), l11_y10x2('l11_y10x2')),
    },
    {
        'name': 'entryx_l8_xpair_pos1_plus_l11_y10x1',
        'class': 'minimal_entry_balance_tail',
        'rationale': 'Same minimal entry-boundary x-bookend, but relax the late11 x-buffer from y10x2 to y10x1 so the tail stays mean-friendly while remaining closer to the max branch.',
        'insertions': merge_insertions(l8_xpair_pos(1.0, 'l8_xpair_pos1'), l11_y10x1('l11_y10x1')),
    },
    {
        'name': 'entryx_l8_xpair_pos1_plus_l11_y10x0back2',
        'class': 'minimal_entry_resume_tail',
        'rationale': 'Minimal anchor8 entry shaping combined with the resume-side protected late11 tail. This is the cleanest “entry-boundary + softer relay-compatible tail” hybrid.',
        'insertions': merge_insertions(l8_xpair_pos(1.0, 'l8_xpair_pos1'), l11_y10x0back2('l11_y10x0back2')),
    },
    {
        'name': 'entryx_l8_xpair_pos2_plus_l11_y10x2',
        'class': 'medium_entry_mean_tail',
        'rationale': 'Keep the already-validated pos2 entry conditioner, but swap the late11 block all the way to the stronger mean core. Tests whether the new family can approach unified territory without abandoning its entry mechanism.',
        'insertions': merge_insertions(l8_xpair_pos(2.0, 'l8_xpair_pos2'), l11_y10x2('l11_y10x2')),
    },
    {
        'name': 'entryx_l8_xpair_pos2_plus_l11_y10x1',
        'class': 'medium_entry_balance_tail',
        'rationale': 'Pos2 anchor8 entry shaping with a lighter y10x1 late11 core. Intended as the most literal compromise between the new max branch and the old late11 mean branch.',
        'insertions': merge_insertions(l8_xpair_pos(2.0, 'l8_xpair_pos2'), l11_y10x1('l11_y10x1')),
    },
    {
        'name': 'entryx_l8_xpair_pos2_plus_l11_y9x1',
        'class': 'medium_entry_lowmax_tail',
        'rationale': 'Pos2 entry shaping with a slightly lower-dose y9x1 late11 tail. This explicitly tries to keep the entry-boundary max tendency while reducing the heavy mean penalty of y8x0.',
        'insertions': merge_insertions(l8_xpair_pos(2.0, 'l8_xpair_pos2'), l11_y9x1('l11_y9x1')),
    },
    {
        'name': 'entryx_l8_xpair_pos3_plus_l11_y10x0back2',
        'class': 'dose_refine_resume_tail',
        'rationale': 'Local anchor8 dose interpolation between the old pos2 and pos4 branches, while keeping the better-behaved resume-side late11 tail. This probes whether the pos4 max benefit can be partially retained without the full mean tax.',
        'insertions': merge_insertions(l8_xpair_pos(3.0, 'l8_xpair_pos3'), l11_y10x0back2('l11_y10x0back2')),
    },
    {
        'name': 'entryx_l8_xpair_pos2_plus_l11_ypos10x1yneg8x1',
        'class': 'medium_entry_asym_tail',
        'rationale': 'Pos2 entry shaping with an asymmetric late11 tail that softens only the return-side negative-Y exposure. Tests whether the entry-boundary family benefits from asymmetry rather than uniform late-dose reduction.',
        'insertions': merge_insertions(l8_xpair_pos(2.0, 'l8_xpair_pos2'), l11_ypos10x1yneg8x1('l11_ypos10x1yneg8x1')),
    },
]


def load_json_checked(path: Path, noise_scale: float) -> dict[str, Any]:
    payload = _load_json(path)
    expected_cfg = expected_noise_config(noise_scale)
    if not _noise_matches(payload, expected_cfg):
        raise ValueError(f'Noise configuration mismatch: {path}')
    return payload


def load_references(noise_scale: float) -> dict[str, Any]:
    refs = load_reference_payloads(noise_scale)
    return {
        'faithful_markov': refs['faithful_markov'],
        'faithful_kf': refs['faithful_kf'],
        'oldbest_markov': refs['oldbest_markov'],
        'oldbest_kf': refs['oldbest_kf'],
        'relay_markov': load_json_checked(RELAY_WINNER_RESULT, noise_scale),
        'unified_markov': load_json_checked(UNIFIED_WINNER_RESULT, noise_scale),
        'unified_kf': load_json_checked(UNIFIED_WINNER_KF_RESULT, noise_scale),
        'default18_markov': load_json_checked(DEFAULT18_RESULT, noise_scale),
        'default18_kf': load_json_checked(DEFAULT18_KF_RESULT, noise_scale),
        'entry_anchor_markov': load_json_checked(ENTRY_ANCHOR_RESULT, noise_scale),
        'entry_anchor_kf': load_json_checked(ENTRY_ANCHOR_KF_RESULT, noise_scale),
    }


def pareto_frontier(rows: list[dict[str, Any]]) -> set[str]:
    frontier = set()
    for row in rows:
        a = row['metrics']['overall']
        dominated = False
        for other in rows:
            if other['candidate_name'] == row['candidate_name']:
                continue
            b = other['metrics']['overall']
            if (
                b['mean_pct_error'] <= a['mean_pct_error'] + 1e-12
                and b['max_pct_error'] <= a['max_pct_error'] + 1e-12
                and (
                    b['mean_pct_error'] < a['mean_pct_error'] - 1e-12
                    or b['max_pct_error'] < a['max_pct_error'] - 1e-12
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.add(row['candidate_name'])
    return frontier


def select_kf_rechecks(rows: list[dict[str, Any]], refs: dict[str, Any]) -> list[str]:
    names: list[str] = []
    best_mean = min(rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    best_max = min(rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    for row in [best_mean, best_max]:
        if row['candidate_name'] not in names:
            names.append(row['candidate_name'])

    unified_max = compact_result(refs['unified_markov'])['overall']['max_pct_error']
    competitive = [
        row for row in rows
        if row['metrics']['overall']['max_pct_error'] <= unified_max + 0.030
    ]
    if competitive:
        comp = min(competitive, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
        if comp['candidate_name'] not in names:
            names.append(comp['candidate_name'])
    return names


def row_summary(payload: dict[str, Any]) -> str:
    o = payload['overall'] if 'overall' in payload else payload
    return f"{o['mean_pct_error']:.3f} / {o['median_pct_error']:.3f} / {o['max_pct_error']:.3f}"


def render_report(payload: dict[str, Any]) -> str:
    refs = payload['references']
    best = payload['best_followup_candidate']
    best_max = payload['best_max_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 entry-boundary follow-up unification pass')
    lines.append('')
    lines.append('## 1. Search intent')
    lines.append('')
    lines.append('- This pass did **not** try to squeeze the old anchor8 entry branch for max alone.')
    lines.append('- Goal: keep the entry-boundary mechanism alive, but recover enough mean to see whether this family can become a more unified competitor rather than just a max-specialized side branch.')
    lines.append('- Hard constraints held fixed: real dual-axis legality, exact continuity closure before resume, faithful12 base scaffold, 1200–1800 s total time, theory-guided search only.')
    lines.append('')
    lines.append('## 2. Theory-guided directions used')
    lines.append('')
    lines.append('1. **Smaller anchor8 positive x-dose** (`pos1/pos2/pos3`) instead of jumping straight to the old `pos4` max anchor, to test whether mean loss mainly came from over-driving the entry conditioner.')
    lines.append('2. **Softer late11 tail hybrids** (`y10x2`, `y10x1`, `y9x1`, `y10x0back2`, asymmetric `ypos10x1yneg8x1`) so the family stays structurally entry-boundary-driven rather than reverting to the old relay / late10 basin.')
    lines.append('3. **No anchor10-heavy reversion** was allowed in this pass, specifically to avoid renaming the old two-anchor basin as a fake “entry” result.')
    lines.append('')
    lines.append('## 3. Fixed references')
    lines.append('')
    lines.append(f"- current unified winner: **{row_summary(refs['unified_winner']['markov42']['overall'])}** (`{refs['unified_winner']['candidate_name']}`)")
    lines.append(f"- current relay winner: **{row_summary(refs['relay_winner']['markov42']['overall'])}** (`{refs['relay_winner']['candidate_name']}`)")
    lines.append(f"- old best legal: **{row_summary(refs['old_best_legal']['markov42']['overall'])}** (`{refs['old_best_legal']['candidate_name']}`)")
    lines.append(f"- entry-boundary anchor point: **{row_summary(refs['entry_anchor']['markov42']['overall'])}** (`{refs['entry_anchor']['candidate_name']}`)")
    lines.append(f"- default18: **{row_summary(refs['default18']['markov42']['overall'])}**")
    lines.append('')
    lines.append('## 4. Markov42 follow-up results')
    lines.append('')
    lines.append('| rank | candidate | class | frontier? | total_s | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Ka2_z | Δmean vs anchor | Δmax vs anchor | Δmean vs unified | Δmax vs unified |')
    lines.append('|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(payload['markov42_rows'], start=1):
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['candidate_class']} | {'yes' if row['on_pareto_frontier'] else 'no'} | {row['total_time_s']:.0f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {row['delta_vs_entry_anchor']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_entry_anchor']['max_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_unified_winner']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_unified_winner']['max_pct_error']['improvement_pct_points']:+.3f} |"
        )
    lines.append('')
    lines.append('## 5. Best follow-up candidate (unification-oriented)')
    lines.append('')
    lines.append(f"- selected candidate: **{best['candidate_name']}** → **{best['markov42']['overall']['mean_pct_error']:.3f} / {best['markov42']['overall']['median_pct_error']:.3f} / {best['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- vs entry-boundary anchor: Δmean **{best['delta_vs_entry_anchor']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_entry_anchor']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- vs current unified winner: Δmean **{best['delta_vs_unified_winner']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_unified_winner']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- vs current relay winner: Δmean **{best['delta_vs_relay_winner']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_relay_winner']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- targeted residuals: dKa_yy **{best['markov42']['key_param_errors']['dKa_yy']:.3f}**, dKg_zz **{best['markov42']['key_param_errors']['dKg_zz']:.3f}**, Ka2_y **{best['markov42']['key_param_errors']['Ka2_y']:.3f}**, Ka2_z **{best['markov42']['key_param_errors']['Ka2_z']:.3f}**")
    lines.append('')
    lines.append('## 6. Best max-preserving follow-up candidate')
    lines.append('')
    lines.append(f"- max-oriented follow-up: **{best_max['candidate_name']}** → **{best_max['markov42']['overall']['mean_pct_error']:.3f} / {best_max['markov42']['overall']['median_pct_error']:.3f} / {best_max['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- vs entry-boundary anchor: Δmean **{best_max['delta_vs_entry_anchor']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_max['delta_vs_entry_anchor']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append('')
    lines.append('## 7. Continuity proof for the best follow-up candidate')
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
    lines.append('## 8. Exact legal motor / timing table for the best follow-up candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for idx, (row, action, face) in enumerate(zip(best['all_rows'], best['all_actions'], best['all_faces']), start=1):
        lines.append(
            f"| {idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 9. KF36 rechecks for best competitive follow-up candidates')
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
        kk = row['kf36']['overall'] if row['kf36'] is not None else None
        kf_text = 'n/a' if kk is None else f"{kk['mean_pct_error']:.3f} / {kk['median_pct_error']:.3f} / {kk['max_pct_error']:.3f}"
        lines.append(
            f"| {row['label']} | {mm['mean_pct_error']:.3f} / {mm['median_pct_error']:.3f} / {mm['max_pct_error']:.3f} | {kf_text} | {row['note']} |"
        )
    lines.append('')
    lines.append('## 11. Bottom line')
    lines.append('')
    lines.append(f"- verdict: **{payload['bottom_line']['verdict']}**")
    lines.append(f"- entry-boundary follow-up status: **{payload['bottom_line']['status']}**")
    lines.append(f"- scientific conclusion: **{payload['scientific_conclusion']}**")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module('search_ch3_12pos_entry_boundary_followup_src', str(SOURCE_FILE))
    refs = load_references(args.noise_scale)

    faithful = build_candidate(mod, ())
    base_rows = faithful.rows
    base_actions = faithful.action_sequence

    candidates = [build_closedloop_candidate(mod, spec, base_rows, base_actions) for spec in CANDIDATE_SPECS]
    candidate_by_name = {cand.name: cand for cand in candidates}
    spec_by_name = {spec['name']: spec for spec in CANDIDATE_SPECS}

    rows: list[dict[str, Any]] = []
    payload_by_name: dict[str, dict[str, Any]] = {}
    for spec, cand in zip(CANDIDATE_SPECS, candidates):
        payload, status, path = run_candidate_payload(mod, cand, 'markov42_noisy', args.noise_scale, args.force_rerun)
        payload_by_name[cand.name] = payload
        rows.append({
            'candidate_name': cand.name,
            'candidate_class': spec['class'],
            'rationale': spec['rationale'],
            'total_time_s': cand.total_time_s,
            'metrics': compact_result(payload),
            'continuity_checks': cand.continuity_checks,
            'run_json': str(path),
            'status': status,
        })

    for row in rows:
        payload = payload_by_name[row['candidate_name']]
        row['delta_vs_relay_winner'] = delta_vs_ref(refs['relay_markov'], payload)
        row['delta_vs_unified_winner'] = delta_vs_ref(refs['unified_markov'], payload)
        row['delta_vs_old_best'] = delta_vs_ref(refs['oldbest_markov'], payload)
        row['delta_vs_entry_anchor'] = delta_vs_ref(refs['entry_anchor_markov'], payload)
        row['delta_vs_default18'] = delta_vs_ref(refs['default18_markov'], payload)

    rows.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    frontier = pareto_frontier(rows)
    for row in rows:
        row['on_pareto_frontier'] = row['candidate_name'] in frontier

    unified_max = compact_result(refs['unified_markov'])['overall']['max_pct_error']
    competitive_rows = [row for row in rows if row['metrics']['overall']['max_pct_error'] <= unified_max + 0.030]
    if competitive_rows:
        best_followup_row = min(competitive_rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    else:
        best_followup_row = rows[0]
    best_followup_cand = candidate_by_name[best_followup_row['candidate_name']]
    best_followup_payload = payload_by_name[best_followup_cand.name]

    best_max_row = min(rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    best_max_cand = candidate_by_name[best_max_row['candidate_name']]
    best_max_payload = payload_by_name[best_max_cand.name]

    kf36_rows: list[dict[str, Any]] = []
    for name in select_kf_rechecks(rows, refs):
        cand = candidate_by_name[name]
        kf_payload, kf_status, kf_path = run_candidate_payload(mod, cand, 'kf36_noisy', args.noise_scale, args.force_rerun)
        note = 'competitive follow-up'
        if name == best_followup_cand.name:
            note = 'best follow-up candidate'
        elif name == best_max_cand.name:
            note = 'best max-preserving follow-up'
        kf36_rows.append({
            'candidate_name': name,
            'note': note,
            'markov42': compact_result(payload_by_name[name]),
            'kf36': compact_result(kf_payload),
            'kf36_status': kf_status,
            'kf36_run_json': str(kf_path),
        })

    best_followup_summary = {
        'candidate_name': best_followup_cand.name,
        'candidate_class': spec_by_name[best_followup_cand.name]['class'],
        'rationale': spec_by_name[best_followup_cand.name]['rationale'],
        'total_time_s': best_followup_cand.total_time_s,
        'continuity_checks': best_followup_cand.continuity_checks,
        'all_rows': best_followup_cand.all_rows,
        'all_actions': best_followup_cand.all_actions,
        'all_faces': best_followup_cand.all_faces,
        'markov42': compact_result(best_followup_payload),
        'delta_vs_relay_winner': delta_vs_ref(refs['relay_markov'], best_followup_payload),
        'delta_vs_unified_winner': delta_vs_ref(refs['unified_markov'], best_followup_payload),
        'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], best_followup_payload),
        'delta_vs_entry_anchor': delta_vs_ref(refs['entry_anchor_markov'], best_followup_payload),
        'delta_vs_default18': delta_vs_ref(refs['default18_markov'], best_followup_payload),
    }

    best_max_summary = {
        'candidate_name': best_max_cand.name,
        'candidate_class': spec_by_name[best_max_cand.name]['class'],
        'rationale': spec_by_name[best_max_cand.name]['rationale'],
        'total_time_s': best_max_cand.total_time_s,
        'markov42': compact_result(best_max_payload),
        'delta_vs_relay_winner': delta_vs_ref(refs['relay_markov'], best_max_payload),
        'delta_vs_unified_winner': delta_vs_ref(refs['unified_markov'], best_max_payload),
        'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], best_max_payload),
        'delta_vs_entry_anchor': delta_vs_ref(refs['entry_anchor_markov'], best_max_payload),
        'delta_vs_default18': delta_vs_ref(refs['default18_markov'], best_max_payload),
    }

    entry_anchor = compact_result(refs['entry_anchor_markov'])['overall']
    best = best_followup_summary['markov42']['overall']
    improved_mean_vs_anchor = best_followup_summary['delta_vs_entry_anchor']['mean_pct_error']['improvement_pct_points']
    improved_max_vs_anchor = best_followup_summary['delta_vs_entry_anchor']['max_pct_error']['improvement_pct_points']
    improved_mean_vs_unified = best_followup_summary['delta_vs_unified_winner']['mean_pct_error']['improvement_pct_points']
    improved_max_vs_unified = best_followup_summary['delta_vs_unified_winner']['max_pct_error']['improvement_pct_points']

    if improved_mean_vs_anchor > 0.5 and improved_max_vs_unified >= 0:
        status = 'more unified competitive point'
        verdict = (
            f"The follow-up produced a real unification-oriented upgrade inside the entry-boundary family: {best_followup_cand.name} improves the old entry-anchor by {improved_mean_vs_anchor:.3f} mean-points while still keeping max better than the current unified winner."
        )
    elif improved_mean_vs_anchor > 0.5:
        status = 'partial unification, but not enough'
        verdict = (
            f"The follow-up recovered a meaningful amount of mean inside the entry-boundary family ({improved_mean_vs_anchor:.3f} points versus the old anchor), but it did not fully preserve the anchor branch’s max superiority against the unified winner."
        )
    else:
        status = 'still max-specialized only'
        verdict = (
            f"The follow-up did not recover enough mean to turn the entry-boundary family into a true unified challenger. The family still behaves mainly as a max-specialized branch."
        )

    scientific_conclusion = (
        f"The main mechanism insight survived: changing the condition before entering the late block still matters, because all competitive follow-up points retained the anchor8 positive x-bookend. "
        f"But the best unification-oriented follow-up, {best_followup_cand.name}, only moved the family from {entry_anchor['mean_pct_error']:.3f} / {entry_anchor['median_pct_error']:.3f} / {entry_anchor['max_pct_error']:.3f} to {best['mean_pct_error']:.3f} / {best['median_pct_error']:.3f} / {best['max_pct_error']:.3f}. "
        f"That is a clear mean recovery of {improved_mean_vs_anchor:.3f} points relative to the old entry anchor, yet it still trails the current unified winner on mean by {-improved_mean_vs_unified:.3f} points. "
        f"So this pass suggests the entry-boundary family can be bent toward a more balanced frontier, but it did not land a new overall winner under the present legal continuity-safe search constraints."
    )

    comparison_rows = [
        {
            'label': 'current unified winner',
            'note': 'twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2',
            'markov42': compact_result(refs['unified_markov']),
            'kf36': compact_result(refs['unified_kf']),
        },
        {
            'label': 'current relay winner',
            'note': 'relay_y9_z10_x11_y9x0',
            'markov42': compact_result(refs['relay_markov']),
            'kf36': None,
        },
        {
            'label': 'old best legal',
            'note': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
            'markov42': compact_result(refs['oldbest_markov']),
            'kf36': compact_result(refs['oldbest_kf']),
        },
        {
            'label': 'entry-boundary anchor point',
            'note': 'entryx_l8_xpair_pos4_plus_l11_y8x0',
            'markov42': compact_result(refs['entry_anchor_markov']),
            'kf36': compact_result(refs['entry_anchor_kf']),
        },
        {
            'label': 'best follow-up candidate',
            'note': best_followup_cand.name,
            'markov42': best_followup_summary['markov42'],
            'kf36': next((row['kf36'] for row in kf36_rows if row['candidate_name'] == best_followup_cand.name), None),
        },
        {
            'label': 'default18',
            'note': 'non-faithful strong reference',
            'markov42': compact_result(refs['default18_markov']),
            'kf36': compact_result(refs['default18_kf']),
        },
    ]

    out_json = RESULTS_DIR / f'ch3_entry_boundary_followup_{make_suffix(args.noise_scale)}.json'
    out_md = REPORTS_DIR / f'psins_ch3_entry_boundary_followup_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_entry_boundary_followup',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'hard_constraints': {
            'base_skeleton': 'faithful chapter-3 12-position original sequence',
            'continuity_rule': 'exact same mechanism state before resume',
            'time_budget_s': [1200.0, 1800.0],
            'seed': 42,
            'truth_family': 'shared low-noise benchmark',
            'search_style': 'entry-boundary refinement plus late11-tail hybrids only',
        },
        'references': {
            'faithful12': {
                'candidate_name': faithful.name,
                'markov42': compact_result(refs['faithful_markov']),
                'kf36': compact_result(refs['faithful_kf']),
            },
            'relay_winner': {
                'candidate_name': 'relay_y9_z10_x11_y9x0',
                'markov42': compact_result(refs['relay_markov']),
            },
            'unified_winner': {
                'candidate_name': 'twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2',
                'markov42': compact_result(refs['unified_markov']),
                'kf36': compact_result(refs['unified_kf']),
            },
            'old_best_legal': {
                'candidate_name': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
                'markov42': compact_result(refs['oldbest_markov']),
                'kf36': compact_result(refs['oldbest_kf']),
            },
            'entry_anchor': {
                'candidate_name': 'entryx_l8_xpair_pos4_plus_l11_y8x0',
                'markov42': compact_result(refs['entry_anchor_markov']),
                'kf36': compact_result(refs['entry_anchor_kf']),
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
                'class': spec['class'],
                'rationale': spec['rationale'],
                'anchors': sorted(spec['insertions'].keys()),
            }
            for spec in CANDIDATE_SPECS
        ],
        'markov42_rows': rows,
        'pareto_frontier_mean_max': sorted(list(frontier)),
        'best_followup_candidate': best_followup_summary,
        'best_max_candidate': best_max_summary,
        'kf36_rows': kf36_rows,
        'comparison_rows': comparison_rows,
        'bottom_line': {
            'status': status,
            'verdict': verdict,
            'best_followup_mean_gain_vs_entry_anchor': improved_mean_vs_anchor,
            'best_followup_max_gain_vs_entry_anchor': improved_max_vs_anchor,
            'best_followup_mean_gain_vs_unified': improved_mean_vs_unified,
            'best_followup_max_gain_vs_unified': improved_max_vs_unified,
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
    print('BEST_FOLLOWUP', best_followup_cand.name, best_followup_summary['markov42']['overall'], flush=True)
    print('BEST_MAX', best_max_cand.name, best_max_summary['markov42']['overall'], flush=True)
    print('BOTTOM_LINE', verdict, flush=True)


if __name__ == '__main__':
    main()
