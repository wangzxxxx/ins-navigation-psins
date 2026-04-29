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

from common_markov import load_module
from benchmark_ch3_12pos_goalA_repairs import compact_result
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate
from search_ch3_12pos_closedloop_local_insertions import (
    NOISE_SCALE,
    REPORT_DATE,
    StepSpec,
    build_closedloop_candidate,
    delta_vs_ref,
    load_reference_payloads,
    make_suffix,
    render_action,
    run_candidate_payload,
)
from compare_four_methods_shared_noise import _load_json, _noise_matches, expected_noise_config

CURRENT_BEST_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_late11_yneg_xpair_outerhold_then_zquad_y10_x2_shared_noise0p08_param_errors.json'
CURRENT_BEST_KF_RESULT = RESULTS_DIR / 'KF36_ch3closedloop_late11_yneg_xpair_outerhold_then_zquad_y10_x2_shared_noise0p08_param_errors.json'
CURRENT_MAX_BEST_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_late11_yneg_xpair_outerhold_then_zquad_y8_x0_shared_noise0p08_param_errors.json'
CURRENT_MAX_BEST_KF_RESULT = RESULTS_DIR / 'KF36_ch3closedloop_late11_yneg_xpair_outerhold_then_zquad_y8_x0_shared_noise0p08_param_errors.json'
DEFAULT18_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json'
DEFAULT18_KF_RESULT = RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json'


REF_NAMES = {
    'late11_yneg_xpair_outerhold_then_zquad_y10_x2',
    'late11_yneg_xpair_outerhold_then_zquad_y8_x0',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def xpair_outerhold(dwell_s: float = 10.0, prefix: str = 'yneg_xpair_outerhold') -> list[StepSpec]:
    return [
        StepSpec(kind='inner', angle_deg=-90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=0.0, segment_role='motif_inner_open', label=f'{prefix}_inner_open'),
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=dwell_s, segment_role='motif_outer_sweep', label=f'{prefix}_outer_sweep'),
        StepSpec(kind='outer', angle_deg=-90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=dwell_s, segment_role='motif_outer_return', label=f'{prefix}_outer_return'),
        StepSpec(kind='inner', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=0.0, segment_role='motif_inner_close', label=f'{prefix}_inner_close'),
    ]


def zquad_split(y_pos_s: float, x_a_s: float, y_neg_s: float, x_b_s: float, prefix: str = 'zquad') -> list[StepSpec]:
    return [
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=y_pos_s, segment_role='motif_y_pos', label=f'{prefix}_q1'),
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=x_a_s, segment_role='motif_zero_a', label=f'{prefix}_q2'),
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=y_neg_s, segment_role='motif_y_neg', label=f'{prefix}_q3'),
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=x_b_s, segment_role='motif_zero_b', label=f'{prefix}_q4'),
    ]


CANDIDATE_SPECS = [
    {
        'name': 'late11_yneg_xpair_outerhold_then_zquad_y10_x2',
        'rationale': 'Reference overall-best faithful12-base zquad candidate from the previous pass. Kept in-batch as the main mean anchor.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zquad_split(10.0, 2.0, 10.0, 2.0, 'zquad_y10x2')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zquad_y8_x0',
        'rationale': 'Reference max-best faithful12-base zquad sibling from the previous pass. Kept in-batch as the Ka2_y/max anchor.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zquad_split(8.0, 0.0, 8.0, 0.0, 'zquad_y8x0')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zquad_y9_x0',
        'rationale': 'Direct midpoint between the max-oriented y8_x0 and the stronger-mean y10_x0 neighborhood. Tests whether one extra second of ±Y dwell is enough to recover mean without giving back the max benefit.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zquad_split(9.0, 0.0, 9.0, 0.0, 'zquad_y9x0')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zquad_y8_x1',
        'rationale': 'Add only a minimal 1 s intermediate ±X buffer to the max-oriented y8_x0 motif, to see whether mean can recover while keeping the Ka2_y ceiling near the current best.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zquad_split(8.0, 1.0, 8.0, 1.0, 'zquad_y8x1')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zquad_y9_x1',
        'rationale': 'Symmetric midpoint interpolation around the two best anchors: slightly lighter ±Y than y10_x2, but keep a minimal 1 s ±X buffer.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zquad_split(9.0, 1.0, 9.0, 1.0, 'zquad_y9x1')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zquad_y10_x1',
        'rationale': 'Trim the current mean-optimal x-buffer from 2 s to 1 s while preserving the stronger y=10 exposure. Intended to lower Ka2_y/max with minimal damage to the mean.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zquad_split(10.0, 1.0, 10.0, 1.0, 'zquad_y10x1')},
    },
    {
        'name': 'late11_yneg_xpair_d9_then_zquad_y10_x2',
        'rationale': 'Very mild dwell rebalance: shorten the preceding xpair outer holds from 10 s to 9 s while keeping the current overall-best zquad block unchanged. Tests whether slightly less pre-zquad x loading helps Ka2_y/max.',
        'insertions': {11: xpair_outerhold(9.0, 'yneg_xpair_d9') + zquad_split(10.0, 2.0, 10.0, 2.0, 'zquad_y10x2')},
    },
    {
        'name': 'late11_yneg_xpair_d11_then_zquad_y8_x0',
        'rationale': 'Mirror mild rebalance on the max branch: lengthen the preceding xpair outer holds from 10 s to 11 s while keeping the max-oriented zquad_y8_x0 block unchanged. Tests whether stronger pre-zquad outer-state protection can shave Ka2_y/max further.',
        'insertions': {11: xpair_outerhold(11.0, 'yneg_xpair_d11') + zquad_split(8.0, 0.0, 8.0, 0.0, 'zquad_y8x0')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zquad_y10_x0back2',
        'rationale': 'Mild order/balance tweak at fixed zquad family: keep total 2 s of intermediate ±X dwell, but place it only on the closing +X state rather than splitting it symmetrically. This explicitly tries to protect the already-good resume channels near node 12.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zquad_split(10.0, 0.0, 10.0, 2.0, 'zquad_y10x0back2')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zquad_ypos10_x1_yneg8_x1',
        'rationale': 'One closely related three-level Y-exposure tweak integrated into the zquad family: preserve the zero-level ±X buffers, keep the positive-Y hold at 10 s, but lighten the negative-Y hold to 8 s. This tests whether a slightly softer return-side Y exposure can lower Ka2_y/max while holding most of the mean gain.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zquad_split(10.0, 1.0, 8.0, 1.0, 'zquad_ypos10x1yneg8x1')},
    },
]


def load_extra_reference_payloads(noise_scale: float) -> dict[str, dict[str, Any]]:
    expected_cfg = expected_noise_config(noise_scale)
    payloads = {
        'current_best_markov': _load_json(CURRENT_BEST_RESULT),
        'current_best_kf': _load_json(CURRENT_BEST_KF_RESULT),
        'current_max_best_markov': _load_json(CURRENT_MAX_BEST_RESULT),
        'current_max_best_kf': _load_json(CURRENT_MAX_BEST_KF_RESULT),
        'default18_markov': _load_json(DEFAULT18_RESULT),
        'default18_kf': _load_json(DEFAULT18_KF_RESULT),
    }
    for payload in payloads.values():
        if not _noise_matches(payload, expected_cfg):
            raise ValueError('Extra reference noise configuration mismatch')
    return payloads


def is_new_candidate(name: str) -> bool:
    return name not in REF_NAMES


def row_summary(row: dict[str, Any]) -> str:
    m = row['metrics']['overall']
    return f"{m['mean_pct_error']:.3f} / {m['median_pct_error']:.3f} / {m['max_pct_error']:.3f}"


def select_kf_rechecks(new_rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    if not new_rows:
        return names
    best_mean = min(new_rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    best_max = min(new_rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    for row in [best_mean, best_max]:
        if row['candidate_name'] not in names:
            names.append(row['candidate_name'])
    competitive = sorted(new_rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    for row in competitive:
        if row['candidate_name'] in names:
            continue
        if row['metrics']['overall']['mean_pct_error'] <= best_mean['metrics']['overall']['mean_pct_error'] + 0.08:
            names.append(row['candidate_name'])
        if len(names) >= 3:
            break
    return names


def render_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    refs = payload['references']
    best_new = payload['best_new_candidate']
    best_new_max = payload['best_new_max_candidate']

    lines.append('# Chapter-3 faithful12-base zquad follow-up')
    lines.append('')
    lines.append('## 1. Scope of this pass')
    lines.append('')
    lines.append('- This pass stayed **strictly inside the late11 zquad family**. No other repair families were reopened.')
    lines.append('- Hard constraints preserved throughout:')
    lines.append('  - faithful12 base skeleton unchanged')
    lines.append('  - real dual-axis continuity-safe closure before node 12 resumes')
    lines.append('  - same `noise_scale=0.08`, `seed=42`, same shared truth family')
    lines.append('  - candidate total time still kept inside **1200–1800 s**')
    lines.append('- Search directions used here were intentionally narrow:')
    lines.append('  1. local fine-grid only around `zquad_y10_x2` and `zquad_y8_x0`')
    lines.append('  2. one-step dwell rebalance on the preceding xpair outerhold')
    lines.append('  3. one resume-side x-buffer placement tweak inside the zquad itself')
    lines.append('  4. one closely related three-level Y-exposure tweak')
    lines.append('')
    lines.append('## 2. Fixed references')
    lines.append('')
    lines.append(f"- current faithful12-base overall-best: **{row_summary(refs['current_faithful_best_markov_row'])}** (`late11_yneg_xpair_outerhold_then_zquad_y10_x2`)")
    lines.append(f"- current faithful12-base max-best sibling: **{row_summary(refs['current_faithful_max_markov_row'])}** (`late11_yneg_xpair_outerhold_then_zquad_y8_x0`)")
    lines.append(f"- old best legal non-faithful-base: **{refs['old_best_legal']['markov42']['overall']['mean_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['median_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- default18 reference: **{refs['default18']['markov42']['overall']['mean_pct_error']:.3f} / {refs['default18']['markov42']['overall']['median_pct_error']:.3f} / {refs['default18']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('## 3. Markov42 results')
    lines.append('')
    lines.append('| rank | candidate | new? | total_s | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Δmean vs current-best | Δmax vs current-best | Δmax vs current-max-best |')
    lines.append('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(payload['markov42_rows'], start=1):
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {'yes' if row['is_new_candidate'] else 'ref'} | {row['total_time_s']:.0f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {row['delta_vs_current_best']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_current_best']['max_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_current_max_best']['max_pct_error']['improvement_pct_points']:+.3f} |"
        )
    lines.append('')
    lines.append('## 4. Best new-candidate readout')
    lines.append('')
    lines.append(f"- best new candidate by mean: **{best_new['candidate_name']}** → **{best_new['markov42']['overall']['mean_pct_error']:.3f} / {best_new['markov42']['overall']['median_pct_error']:.3f} / {best_new['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"  - vs current faithful overall-best: Δmean **{best_new['delta_vs_current_best']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_new['delta_vs_current_best']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"  - vs current max-best sibling: Δmax **{best_new['delta_vs_current_max_best']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"  - vs old best legal: Δmean **{best_new['delta_vs_old_best']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_new['delta_vs_old_best']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- best new candidate by max: **{best_new_max['candidate_name']}** → **{best_new_max['markov42']['overall']['mean_pct_error']:.3f} / {best_new_max['markov42']['overall']['median_pct_error']:.3f} / {best_new_max['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"  - vs current max-best sibling: Δmax **{best_new_max['delta_vs_current_max_best']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append('')
    lines.append('## 5. Continuity proof for the best new candidate')
    lines.append('')
    for check in best_new['continuity_checks']:
        before = check['state_before_insertion']
        after = check['state_after_insertion']
        lines.append(f"- anchor {check['anchor_id']}: closure_ok = **{'yes' if check['closure_ok'] else 'no'}**")
        lines.append(f"  - before: beta={before['beta_deg']}°, outer_axis={before['outer_axis_body']}, face={before['face_name']}")
        lines.append(f"  - after : beta={after['beta_deg']}°, outer_axis={after['outer_axis_body']}, face={after['face_name']}")
        if check['next_base_action_preview'] is not None:
            preview = check['next_base_action_preview']
            lines.append(f"  - next original action remains legal as `{preview['kind']}` {preview['motor_angle_deg']:+d}° with effective axis {preview['effective_body_axis']}")
    lines.append('')
    lines.append('## 6. Exact legal motor / timing table for the best new candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for seq_idx, (row, action, face) in enumerate(zip(best_new['all_rows'], best_new['all_actions'], best_new['all_faces']), start=1):
        lines.append(
            f"| {seq_idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 7. KF36 rechecks for best competitive new candidates')
    lines.append('')
    lines.append('| candidate | note | Markov42 mean/median/max | KF36 mean/median/max |')
    lines.append('|---|---|---|---|')
    for row in payload['kf36_rows']:
        mm = row['markov42']['overall']
        kk = row['kf36']['overall']
        lines.append(
            f"| {row['candidate_name']} | {row['note']} | {mm['mean_pct_error']:.3f} / {mm['median_pct_error']:.3f} / {mm['max_pct_error']:.3f} | {kk['mean_pct_error']:.3f} / {kk['median_pct_error']:.3f} / {kk['max_pct_error']:.3f} |"
        )
    lines.append('')
    lines.append('## 8. Reference comparison')
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
    lines.append('## 9. Bottom line')
    lines.append('')
    lines.append(f"- current overall-best faithful12-base max: **{refs['current_faithful_best']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- current max-best faithful12-base max   : **{refs['current_faithful_max_best']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- best new follow-up max                 : **{best_new_max['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- default18 max reference                : **{refs['default18']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append(f"- verdict on Ka2_y / max ceiling: **{payload['bottom_line']['ceiling_verdict']}**")
    lines.append(f"- scientific conclusion: **{payload['scientific_conclusion']}**")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_module('search_ch3_12pos_closedloop_zquad_followup_src', str(SOURCE_FILE))

    refs = load_reference_payloads(args.noise_scale)
    extra_refs = load_extra_reference_payloads(args.noise_scale)

    faithful = build_candidate(mod, ())
    base_rows = faithful.rows
    base_actions = faithful.action_sequence

    candidates = [build_closedloop_candidate(mod, spec, base_rows, base_actions) for spec in CANDIDATE_SPECS]
    candidate_by_name = {cand.name: cand for cand in candidates}

    rows: list[dict[str, Any]] = []
    payload_by_name: dict[str, dict[str, Any]] = {}
    for spec, cand in zip(CANDIDATE_SPECS, candidates):
        payload, status, path = run_candidate_payload(mod, cand, 'markov42_noisy', args.noise_scale, args.force_rerun)
        row = {
            'candidate_name': cand.name,
            'is_new_candidate': is_new_candidate(cand.name),
            'family': 'zquad_followup',
            'rationale': spec['rationale'],
            'total_time_s': cand.total_time_s,
            'metrics': compact_result(payload),
            'continuity_checks': cand.continuity_checks,
            'run_json': str(path),
            'status': status,
            'delta_vs_faithful': delta_vs_ref(refs['faithful_markov'], payload),
            'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], payload),
            'delta_vs_current_best': delta_vs_ref(extra_refs['current_best_markov'], payload),
            'delta_vs_current_max_best': delta_vs_ref(extra_refs['current_max_best_markov'], payload),
            'delta_vs_default18': delta_vs_ref(extra_refs['default18_markov'], payload),
        }
        rows.append(row)
        payload_by_name[cand.name] = payload

    rows.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    row_lookup = {row['candidate_name']: row for row in rows}
    new_rows = [row for row in rows if row['is_new_candidate']]
    if not new_rows:
        raise ValueError('No new zquad follow-up candidates were defined')

    best_new_row = min(new_rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    best_new_candidate = candidate_by_name[best_new_row['candidate_name']]
    best_new_payload = payload_by_name[best_new_candidate.name]

    best_new_max_row = min(new_rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    best_new_max_candidate = candidate_by_name[best_new_max_row['candidate_name']]
    best_new_max_payload = payload_by_name[best_new_max_candidate.name]

    kf36_rows = []
    for name in select_kf_rechecks(new_rows):
        cand = candidate_by_name[name]
        kf_payload, kf_status, kf_path = run_candidate_payload(mod, cand, 'kf36_noisy', args.noise_scale, args.force_rerun)
        note = 'competitive new candidate'
        if name == best_new_candidate.name:
            note = 'best new candidate by mean'
        elif name == best_new_max_candidate.name:
            note = 'best new candidate by max'
        kf36_rows.append({
            'candidate_name': name,
            'markov42': compact_result(payload_by_name[name]),
            'kf36': compact_result(kf_payload),
            'kf36_status': kf_status,
            'kf36_run_json': str(kf_path),
            'note': note,
        })

    best_new_summary = {
        'candidate_name': best_new_candidate.name,
        'rationale': next(spec['rationale'] for spec in CANDIDATE_SPECS if spec['name'] == best_new_candidate.name),
        'total_time_s': best_new_candidate.total_time_s,
        'all_rows': best_new_candidate.all_rows,
        'all_actions': best_new_candidate.all_actions,
        'all_faces': best_new_candidate.all_faces,
        'continuity_checks': best_new_candidate.continuity_checks,
        'markov42': compact_result(best_new_payload),
        'markov42_run_json': best_new_row['run_json'],
        'delta_vs_faithful': delta_vs_ref(refs['faithful_markov'], best_new_payload),
        'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], best_new_payload),
        'delta_vs_current_best': delta_vs_ref(extra_refs['current_best_markov'], best_new_payload),
        'delta_vs_current_max_best': delta_vs_ref(extra_refs['current_max_best_markov'], best_new_payload),
        'delta_vs_default18': delta_vs_ref(extra_refs['default18_markov'], best_new_payload),
    }
    best_new_max_summary = {
        'candidate_name': best_new_max_candidate.name,
        'rationale': next(spec['rationale'] for spec in CANDIDATE_SPECS if spec['name'] == best_new_max_candidate.name),
        'total_time_s': best_new_max_candidate.total_time_s,
        'markov42': compact_result(best_new_max_payload),
        'markov42_run_json': best_new_max_row['run_json'],
        'delta_vs_faithful': delta_vs_ref(refs['faithful_markov'], best_new_max_payload),
        'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], best_new_max_payload),
        'delta_vs_current_best': delta_vs_ref(extra_refs['current_best_markov'], best_new_max_payload),
        'delta_vs_current_max_best': delta_vs_ref(extra_refs['current_max_best_markov'], best_new_max_payload),
        'delta_vs_default18': delta_vs_ref(extra_refs['default18_markov'], best_new_max_payload),
    }

    comparison_rows = [
        {
            'label': 'current faithful12-base overall-best',
            'note': 'late11_yneg_xpair_outerhold_then_zquad_y10_x2',
            'markov42': compact_result(extra_refs['current_best_markov']),
            'kf36': compact_result(extra_refs['current_best_kf']),
        },
        {
            'label': 'current faithful12-base max-best sibling',
            'note': 'late11_yneg_xpair_outerhold_then_zquad_y8_x0',
            'markov42': compact_result(extra_refs['current_max_best_markov']),
            'kf36': compact_result(extra_refs['current_max_best_kf']),
        },
        {
            'label': 'best new follow-up candidate',
            'note': best_new_candidate.name,
            'markov42': best_new_summary['markov42'],
            'kf36': next((row['kf36'] for row in kf36_rows if row['candidate_name'] == best_new_candidate.name), None),
        },
        {
            'label': 'best new max-oriented follow-up',
            'note': best_new_max_candidate.name,
            'markov42': best_new_max_summary['markov42'],
            'kf36': next((row['kf36'] for row in kf36_rows if row['candidate_name'] == best_new_max_candidate.name), None),
        },
        {
            'label': 'old best legal non-faithful-base',
            'note': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
            'markov42': compact_result(refs['oldbest_markov']),
            'kf36': compact_result(refs['oldbest_kf']),
        },
        {
            'label': 'default18 reference',
            'note': 'non-faithful strong reference',
            'markov42': compact_result(extra_refs['default18_markov']),
            'kf36': compact_result(extra_refs['default18_kf']),
        },
    ]

    current_best_max = compact_result(extra_refs['current_best_markov'])['overall']['max_pct_error']
    current_max_best_max = compact_result(extra_refs['current_max_best_markov'])['overall']['max_pct_error']
    best_new_max = best_new_summary['markov42']['overall']['max_pct_error']
    best_new_max_opt = best_new_max_summary['markov42']['overall']['max_pct_error']
    default18_max = compact_result(extra_refs['default18_markov'])['overall']['max_pct_error']

    improve_vs_current_best = current_best_max - best_new_max
    improve_vs_current_maxbest = current_max_best_max - best_new_max_opt
    gap_to_default18_after = best_new_max_opt - default18_max

    if improve_vs_current_maxbest >= 0.05:
        ceiling_verdict = (
            f"small but real local reduction: the best new max-oriented candidate pushed the faithful12-base max from {current_max_best_max:.3f} down to {best_new_max_opt:.3f}. "
            f"However this still leaves a gigantic {gap_to_default18_after:.3f}-point gap to default18, so the Ka2_y ceiling is not materially repaired."
        )
    elif improve_vs_current_best > 0 or improve_vs_current_maxbest > 0:
        ceiling_verdict = (
            f"not materially: the follow-up only moved max by {improve_vs_current_maxbest:.3f} points against the current max-best branch "
            f"(and {improve_vs_current_best:.3f} against the current overall-best), far too small to count as a real Ka2_y ceiling repair."
        )
    else:
        ceiling_verdict = (
            'no: none of the focused follow-up candidates lowered the existing zquad Ka2_y/max frontier; the previous zquad anchors remain the true branch frontier.'
        )

    if best_new_summary['delta_vs_current_best']['mean_pct_error']['improvement_pct_points'] > 0:
        scientific_conclusion = (
            f"The focused zquad follow-up did find a better new mean candidate ({best_new_candidate.name}) than the current faithful12-base best, "
            f"but the Ka2_y/max ceiling still did not move materially toward default18."
        )
    else:
        scientific_conclusion = (
            f"The focused zquad follow-up did not beat the current faithful12-base overall-best mean anchor ({best_new_candidate.name} is still weaker on mean), "
            f"and it also failed to lower the Ka2_y/max ceiling materially. This suggests the present late11 zquad family is already near its local limit under the real dual-axis continuity rule."
        )

    out_json = RESULTS_DIR / f'ch3_12pos_closedloop_zquad_followup_{make_suffix(args.noise_scale)}.json'
    out_md = REPORTS_DIR / f'psins_ch3_12pos_closedloop_zquad_followup_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_12pos_closedloop_zquad_followup',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'hard_constraints': {
            'base_skeleton': 'faithful chapter-3 12-position original sequence',
            'continuity_rule': 'exact same mechanism state before resume',
            'search_scope': 'zquad family only around y10_x2 and y8_x0',
            'noise_scale': args.noise_scale,
            'seed': 42,
            'truth_family': 'shared low-noise benchmark',
            'time_budget_s': [1200.0, 1800.0],
        },
        'references': {
            'faithful12': {
                'candidate_name': faithful.name,
                'markov42': compact_result(refs['faithful_markov']),
                'kf36': compact_result(refs['faithful_kf']),
            },
            'current_faithful_best': {
                'candidate_name': 'late11_yneg_xpair_outerhold_then_zquad_y10_x2',
                'markov42': compact_result(extra_refs['current_best_markov']),
                'kf36': compact_result(extra_refs['current_best_kf']),
            },
            'current_faithful_max_best': {
                'candidate_name': 'late11_yneg_xpair_outerhold_then_zquad_y8_x0',
                'markov42': compact_result(extra_refs['current_max_best_markov']),
                'kf36': compact_result(extra_refs['current_max_best_kf']),
            },
            'old_best_legal': {
                'candidate_name': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
                'markov42': compact_result(refs['oldbest_markov']),
                'kf36': compact_result(refs['oldbest_kf']),
            },
            'default18': {
                'candidate_name': 'default18',
                'markov42': compact_result(extra_refs['default18_markov']),
                'kf36': compact_result(extra_refs['default18_kf']),
            },
            'current_faithful_best_markov_row': row_lookup['late11_yneg_xpair_outerhold_then_zquad_y10_x2'],
            'current_faithful_max_markov_row': row_lookup['late11_yneg_xpair_outerhold_then_zquad_y8_x0'],
        },
        'candidate_specs': [
            {
                'name': spec['name'],
                'rationale': spec['rationale'],
                'insertions': sorted(spec['insertions'].keys()),
                'is_new_candidate': is_new_candidate(spec['name']),
            }
            for spec in CANDIDATE_SPECS
        ],
        'markov42_rows': rows,
        'best_new_candidate': best_new_summary,
        'best_new_max_candidate': best_new_max_summary,
        'kf36_rows': kf36_rows,
        'comparison_rows': comparison_rows,
        'bottom_line': {
            'current_overall_best_max': current_best_max,
            'current_max_best_max': current_max_best_max,
            'best_new_candidate_max': best_new_max,
            'best_new_max_candidate_max': best_new_max_opt,
            'default18_max': default18_max,
            'improvement_vs_current_overall_best': improve_vs_current_best,
            'improvement_vs_current_max_best': improve_vs_current_maxbest,
            'gap_to_default18_after_best_new_max': gap_to_default18_after,
            'ceiling_verdict': ceiling_verdict,
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
    print('BEST_NEW_MEAN', best_new_candidate.name, best_new_summary['markov42']['overall'], flush=True)
    print('BEST_NEW_MAX', best_new_max_candidate.name, best_new_max_summary['markov42']['overall'], flush=True)
    print('BOTTOM_LINE', ceiling_verdict, flush=True)
    print('SCIENTIFIC_CONCLUSION', scientific_conclusion, flush=True)


if __name__ == '__main__':
    main()
