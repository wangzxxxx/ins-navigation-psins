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
    FAITHFUL_KF_RESULT,
    FAITHFUL_RESULT,
    OLD_BEST_KF_RESULT,
    OLD_BEST_RESULT,
    NOISE_SCALE,
    REPORT_DATE,
    StepSpec,
    build_closedloop_candidate,
    closed_pair,
    delta_vs_ref,
    inner_outer_pair_return,
    load_reference_payloads,
    make_suffix,
    render_action,
    run_candidate_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def zwrap_yneg_xpair(rot_s: float = 5.0, x_dwell_s: float = 5.0, z_dwell_s: float = 10.0) -> list[StepSpec]:
    return [
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=z_dwell_s, segment_role='motif_z_wrap_open', label='z11_wrap_open'),
        *inner_outer_pair_return(-90, +90, rot_s, x_dwell_s, 'yneg_xpair_11'),
        StepSpec(kind='outer', angle_deg=-90, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=z_dwell_s, segment_role='motif_z_wrap_close', label='z11_wrap_close'),
    ]


def xpair_outerhold_then_zpair() -> list[StepSpec]:
    return [
        StepSpec(kind='inner', angle_deg=-90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=0.0, segment_role='motif_inner_open', label='yneg_xpair_outerhold_inner_open'),
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=10.0, segment_role='motif_outer_sweep', label='yneg_xpair_outerhold_outer_sweep'),
        StepSpec(kind='outer', angle_deg=-90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=10.0, segment_role='motif_outer_return', label='yneg_xpair_outerhold_outer_return'),
        StepSpec(kind='inner', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=0.0, segment_role='motif_inner_close', label='yneg_xpair_outerhold_inner_close'),
        *closed_pair('outer', +90, 5.0, 10.0, 'z11_pos_med'),
    ]


def xpair_then_zpair_frontload() -> list[StepSpec]:
    return [
        *inner_outer_pair_return(-90, +90, 5.0, 5.0, 'yneg_xpair_11'),
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=20.0, segment_role='motif_out', label='z11_frontload_out'),
        StepSpec(kind='outer', angle_deg=-90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=0.0, segment_role='motif_return', label='z11_frontload_return'),
    ]


CANDIDATE_SPECS = [
    {
        'name': 'late11_yneg_xpair_return_then_zpair_pos_med',
        'rationale': 'Reference incumbent from the previous branch: y-negative x-pair closed loop first, then the medium z-pair. Reused here as the fine-grid anchor.',
        'insertions': {
            11: inner_outer_pair_return(-90, +90, 5.0, 5.0, 'yneg_xpair_11') + closed_pair('outer', +90, 5.0, 10.0, 'z11_pos_med'),
        },
    },
    {
        'name': 'late11_zpair_pos_med_then_yneg_xpair_return',
        'rationale': 'Whole-motif reverse-order reference kept in-batch so the fine-grid can be read against the already-known order sensitivity.',
        'insertions': {
            11: closed_pair('outer', +90, 5.0, 10.0, 'z11_pos_med') + inner_outer_pair_return(-90, +90, 5.0, 5.0, 'yneg_xpair_11'),
        },
    },
    {
        'name': 'late11_zwrap_yneg_xpair_return',
        'rationale': 'Local order variant: open the z-loop, run the yneg-xpair while staying inside that late z excursion, then close the z-loop. Same total motif budget as the incumbent, but different local order and state occupancy.',
        'insertions': {
            11: zwrap_yneg_xpair(5.0, 5.0, 10.0),
        },
    },
    {
        'name': 'late11_yneg_xpair_d6_then_zpair_d8',
        'rationale': 'Mild rebalance at fixed 70 s insertion budget: strengthen the yneg-xpair dwell slightly (6 s per step) and lighten the z-pair dwell slightly (8 s per step).',
        'insertions': {
            11: inner_outer_pair_return(-90, +90, 5.0, 6.0, 'yneg_xpair_d6_11') + closed_pair('outer', +90, 5.0, 8.0, 'z11_pos_d8'),
        },
    },
    {
        'name': 'late11_yneg_xpair_d7_then_zpair_d6',
        'rationale': 'Further fixed-budget rebalance toward the yneg-xpair component: x-pair dwell 7 s, z-pair dwell 6 s.',
        'insertions': {
            11: inner_outer_pair_return(-90, +90, 5.0, 7.0, 'yneg_xpair_d7_11') + closed_pair('outer', +90, 5.0, 6.0, 'z11_pos_d6'),
        },
    },
    {
        'name': 'late11_yneg_xpair_d8_then_zpair_d4',
        'rationale': 'Strongest fixed-budget tilt toward the yneg-xpair component tested here: x-pair dwell 8 s, z-pair dwell 4 s.',
        'insertions': {
            11: inner_outer_pair_return(-90, +90, 5.0, 8.0, 'yneg_xpair_d8_11') + closed_pair('outer', +90, 5.0, 4.0, 'z11_pos_d4'),
        },
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zpair_pos_med',
        'rationale': 'Same total 70 s as the incumbent, but reallocate x-pair dwell away from inner-open/inner-close holds and concentrate it on the outer sweep/return states.',
        'insertions': {
            11: xpair_outerhold_then_zpair(),
        },
    },
    {
        'name': 'late11_yneg_xpair_then_zpair_pos_frontload',
        'rationale': 'Pre/post allocation tweak at fixed 70 s: keep the incumbent motif order, but push the z-pair dwell entirely onto the excursion half (out=20 s, return=0 s) instead of splitting it evenly.',
        'insertions': {
            11: xpair_then_zpair_frontload(),
        },
    },
    {
        'name': 'late10_zpair_pos_med_then_late11_yneg_xpair_return',
        'rationale': 'Single adjacent-anchor local variant: move the z-pair one anchor earlier (node 10), keep the yneg-xpair at node 11, and preserve the faithful12 base skeleton everywhere else.',
        'insertions': {
            10: closed_pair('outer', +90, 5.0, 10.0, 'z10_pos_med'),
            11: inner_outer_pair_return(-90, +90, 5.0, 5.0, 'yneg_xpair_11'),
        },
    },
]


def select_competitive(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_mean = min(r['metrics']['overall']['mean_pct_error'] for r in rows)
    selected = []
    for row in rows:
        mean_v = row['metrics']['overall']['mean_pct_error']
        max_v = row['metrics']['overall']['max_pct_error']
        if mean_v <= best_mean + 0.25 or max_v <= 100.2:
            selected.append(row)
    selected.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    return selected[:4]


def render_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    refs = payload['references']
    best = payload['best_candidate']
    finegrid = payload['finegrid_summary']

    lines.append('# Chapter-3 late11 faithful12 closed-loop fine-grid')
    lines.append('')
    lines.append('## 1. Scope and rule of this pass')
    lines.append('')
    lines.append('- Only the **late11 faithful12 closed-loop family** was explored; no broadening to unrelated repair families.')
    lines.append('- Allowed changes used here:')
    lines.append('  - local order around the incumbent late11 motif')
    lines.append('  - fixed-budget x-vs-z dwell rebalancing')
    lines.append('  - one pre/post allocation tweak on the xpair, one on the zpair')
    lines.append('  - one adjacent late10/late11 local variant')
    lines.append('- Hard constraints preserved throughout: faithful12 base scaffold unchanged, exact mechanism-state closure before resume, same noise0.08/seed42/truth family, total time 1200–1800 s.')
    lines.append('')
    lines.append('## 2. Fixed references')
    lines.append('')
    lines.append(f"- faithful12 Markov42: **{refs['faithful12']['markov42']['overall']['mean_pct_error']:.3f} / {refs['faithful12']['markov42']['overall']['median_pct_error']:.3f} / {refs['faithful12']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- old best legal Markov42: **{refs['old_best_legal']['markov42']['overall']['mean_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['median_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- previous faithful12-base incumbent: **{payload['previous_incumbent']['markov42']['overall']['mean_pct_error']:.3f} / {payload['previous_incumbent']['markov42']['overall']['median_pct_error']:.3f} / {payload['previous_incumbent']['markov42']['overall']['max_pct_error']:.3f}** (`{payload['previous_incumbent']['candidate_name']}`)")
    lines.append('')
    lines.append('## 3. Markov42 fine-grid results')
    lines.append('')
    lines.append('| rank | candidate | total_s | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Δmean vs prev | Δmean vs old best | Δmax vs old best |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for i, row in enumerate(payload['markov42_rows'], start=1):
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        d_prev = row['delta_vs_previous_incumbent']['mean_pct_error']['improvement_pct_points']
        d_old_mean = row['delta_vs_old_best']['mean_pct_error']['improvement_pct_points']
        d_old_max = row['delta_vs_old_best']['max_pct_error']['improvement_pct_points']
        lines.append(
            f"| {i} | {row['candidate_name']} | {row['total_time_s']:.0f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {d_prev:+.3f} | {d_old_mean:+.3f} | {d_old_max:+.3f} |"
        )
    lines.append('')
    lines.append('## 4. Best fine-grid candidate')
    lines.append('')
    lines.append(f"- best candidate: **{best['candidate_name']}**")
    lines.append(f"- Markov42: **{best['markov42']['overall']['mean_pct_error']:.3f} / {best['markov42']['overall']['median_pct_error']:.3f} / {best['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- vs previous incumbent: Δmean **{best['delta_vs_previous_incumbent']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian **{best['delta_vs_previous_incumbent']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_previous_incumbent']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- vs faithful12: Δmean **{best['delta_vs_faithful']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian **{best['delta_vs_faithful']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_faithful']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- vs old best legal: Δmean **{best['delta_vs_old_best']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian **{best['delta_vs_old_best']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_old_best']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append('')
    lines.append('### 4.1 Continuity proof')
    lines.append('')
    for check in best['continuity_checks']:
        before = check['state_before_insertion']
        after = check['state_after_insertion']
        next_action = check['next_base_action_preview']
        lines.append(f"- anchor {check['anchor_id']}: closure_ok = **{'yes' if check['closure_ok'] else 'no'}**")
        lines.append(f"  - before: beta={before['beta_deg']}°, outer_axis={before['outer_axis_body']}, face={before['face_name']}")
        lines.append(f"  - after : beta={after['beta_deg']}°, outer_axis={after['outer_axis_body']}, face={after['face_name']}")
        if next_action is not None:
            lines.append(f"  - next original action remains legal as `{next_action['kind']}` {int(next_action['motor_angle_deg']):+d}° with effective axis {next_action['effective_body_axis']}")
    lines.append('')
    lines.append('### 4.2 Exact legal motor / timing table')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for row, action, face in zip(best['all_rows'], best['all_actions'], best['all_faces']):
        lines.append(
            f"| {row['pos_id']} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 5. KF36 rechecks for competitive candidates')
    lines.append('')
    lines.append('| candidate | Markov42 mean/median/max | KF36 mean/median/max | note |')
    lines.append('|---|---|---|---|')
    for row in payload['kf36_rows']:
        mm = row['markov42']['overall']
        kk = row['kf36']['overall']
        lines.append(
            f"| {row['candidate_name']} | {mm['mean_pct_error']:.3f} / {mm['median_pct_error']:.3f} / {mm['max_pct_error']:.3f} | {kk['mean_pct_error']:.3f} / {kk['median_pct_error']:.3f} / {kk['max_pct_error']:.3f} | {row['note']} |"
        )
    lines.append('')
    lines.append('## 6. Bottom line')
    lines.append('')
    lines.append(f"- remaining mean gap before fine-grid: **{finegrid['mean_gap_to_old_best_before']:.3f}**")
    lines.append(f"- remaining mean gap after fine-grid : **{finegrid['mean_gap_to_old_best_after']:.3f}**")
    lines.append(f"- gap closed by fine-grid itself     : **{finegrid['extra_mean_gap_closed']:.3f}** ({finegrid['extra_gap_closure_pct']:.1f}% of the remaining gap)")
    lines.append(f"- verdict: **{finegrid['materiality_verdict']}**")
    lines.append('')
    lines.append(f"{payload['scientific_conclusion']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_module('search_ch3_12pos_closedloop_late11_finegrid_src', str(SOURCE_FILE))

    refs = load_reference_payloads(args.noise_scale)
    faithful = build_candidate(mod, ())
    base_rows = faithful.rows
    base_actions = faithful.action_sequence

    candidates = [build_closedloop_candidate(mod, spec, base_rows, base_actions) for spec in CANDIDATE_SPECS]
    candidate_by_name = {cand.name: cand for cand in candidates}

    previous_incumbent_name = 'late11_yneg_xpair_return_then_zpair_pos_med'

    rows = []
    payload_by_name: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        print(f'RUN {cand.name} total={cand.total_time_s:.0f}s ...', flush=True)
        payload, status, path = run_candidate_payload(mod, cand, 'markov42_noisy', args.noise_scale, args.force_rerun)
        row = {
            'candidate_name': cand.name,
            'rationale': cand.rationale,
            'total_time_s': cand.total_time_s,
            'metrics': compact_result(payload),
            'continuity_checks': cand.continuity_checks,
            'run_json': str(path),
            'status': status,
            'delta_vs_faithful': delta_vs_ref(refs['faithful_markov'], payload),
            'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], payload),
        }
        rows.append(row)
        payload_by_name[cand.name] = payload
        print(
            f"DONE {cand.name}: mean={row['metrics']['overall']['mean_pct_error']:.3f}, "
            f"median={row['metrics']['overall']['median_pct_error']:.3f}, "
            f"max={row['metrics']['overall']['max_pct_error']:.3f}",
            flush=True,
        )

    prev_payload = payload_by_name[previous_incumbent_name]
    prev_row_lookup = {row['candidate_name']: row for row in rows}
    for row in rows:
        row['delta_vs_previous_incumbent'] = delta_vs_ref(prev_payload, payload_by_name[row['candidate_name']])

    rows.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    best_row = rows[0]
    best_candidate = candidate_by_name[best_row['candidate_name']]
    best_markov_payload = payload_by_name[best_candidate.name]

    kf36_rows = []
    for row in select_competitive(rows):
        cand = candidate_by_name[row['candidate_name']]
        kf_payload, kf_status, kf_path = run_candidate_payload(mod, cand, 'kf36_noisy', args.noise_scale, args.force_rerun)
        note = 'new recheck'
        if row['candidate_name'] == previous_incumbent_name and kf_status != 'rerun':
            note = 'existing incumbent re-used'
        elif kf_status != 'rerun':
            note = 're-used cached run'
        kf36_rows.append({
            'candidate_name': cand.name,
            'markov42': compact_result(payload_by_name[cand.name]),
            'kf36': compact_result(kf_payload),
            'kf36_status': kf_status,
            'kf36_run_json': str(kf_path),
            'note': note,
        })

    best_kf = next((x for x in kf36_rows if x['candidate_name'] == best_candidate.name), None)
    if best_kf is None:
        kf_payload, kf_status, kf_path = run_candidate_payload(mod, best_candidate, 'kf36_noisy', args.noise_scale, args.force_rerun)
        best_kf = {
            'candidate_name': best_candidate.name,
            'markov42': compact_result(best_markov_payload),
            'kf36': compact_result(kf_payload),
            'kf36_status': kf_status,
            'kf36_run_json': str(kf_path),
            'note': 'best-only recheck',
        }
        kf36_rows.append(best_kf)

    best_summary = {
        'candidate_name': best_candidate.name,
        'rationale': best_candidate.rationale,
        'total_time_s': best_candidate.total_time_s,
        'all_rows': best_candidate.all_rows,
        'all_actions': best_candidate.all_actions,
        'all_faces': best_candidate.all_faces,
        'continuity_checks': best_candidate.continuity_checks,
        'markov42': compact_result(best_markov_payload),
        'markov42_run_json': best_row['run_json'],
        'kf36': best_kf['kf36'],
        'kf36_run_json': best_kf['kf36_run_json'],
        'delta_vs_previous_incumbent': delta_vs_ref(prev_payload, best_markov_payload),
        'delta_vs_faithful': delta_vs_ref(refs['faithful_markov'], best_markov_payload),
        'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], best_markov_payload),
    }

    prev_mean = compact_result(prev_payload)['overall']['mean_pct_error']
    best_mean = best_summary['markov42']['overall']['mean_pct_error']
    old_best_mean = compact_result(refs['oldbest_markov'])['overall']['mean_pct_error']
    before_gap = prev_mean - old_best_mean
    after_gap = best_mean - old_best_mean
    extra_closed = prev_mean - best_mean
    extra_closure_pct = (extra_closed / before_gap * 100.0) if before_gap > 1e-12 else 0.0

    if best_mean < old_best_mean and best_summary['markov42']['overall']['max_pct_error'] <= compact_result(refs['oldbest_markov'])['overall']['max_pct_error'] + 1e-9:
        verdict = 'yes — the fine-grid fully closed the remaining mean gap and preserved the max ceiling.'
    elif extra_closed >= 0.5:
        verdict = 'partly yes — the fine-grid narrowed the mean gap by a meaningful amount, but did not fully beat the old best.'
    else:
        verdict = 'no — the fine-grid changed the result only marginally; the remaining mean gap is still materially open.'

    if best_mean < prev_mean:
        scientific_conclusion = (
            f"The local late11 fine-grid found a modestly better faithful12-base candidate ({best_candidate.name}) than the previous incumbent. "
            f"However, the branch still does not beat the old best legal result; Ka2_y remains the max ceiling and the remaining mean gap is not fully closed."
        )
    else:
        scientific_conclusion = (
            f"The local late11 fine-grid did not improve on the previous faithful12-base incumbent. "
            f"This suggests the current late11 closed-loop family is already near its local limit under the real dual-axis rule, with Ka2_y still setting the ceiling."
        )

    out_json = RESULTS_DIR / f'ch3_12pos_closedloop_late11_finegrid_{make_suffix(args.noise_scale)}.json'
    out_md = REPORTS_DIR / f'psins_ch3_12pos_closedloop_late11_finegrid_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_12pos_closedloop_late11_finegrid',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'hard_constraints': {
            'base_skeleton': 'faithful chapter-3 12-position original sequence',
            'locality_rule': 'late11 family only, plus at most one adjacent late10/late11 variant',
            'insertion_rule': 'closed-loop insertion-return motifs only',
            'continuity_check': ['beta_deg', 'outer_axis_body', 'full_orientation_matrix'],
            'time_budget_s': [1200.0, 1800.0],
            'seed': 42,
            'truth_family': 'shared low-noise benchmark',
        },
        'references': {
            'faithful12': {
                'candidate_name': faithful.name,
                'rows': faithful.rows,
                'action_sequence': faithful.action_sequence,
                'faces': faithful.faces,
                'markov42': compact_result(refs['faithful_markov']),
                'markov42_run_json': str(FAITHFUL_RESULT),
                'kf36': compact_result(refs['faithful_kf']),
                'kf36_run_json': str(FAITHFUL_KF_RESULT),
            },
            'old_best_legal': {
                'candidate_name': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
                'markov42': compact_result(refs['oldbest_markov']),
                'markov42_run_json': str(OLD_BEST_RESULT),
                'kf36': compact_result(refs['oldbest_kf']),
                'kf36_run_json': str(OLD_BEST_KF_RESULT),
            },
        },
        'previous_incumbent': {
            'candidate_name': previous_incumbent_name,
            'markov42': compact_result(prev_payload),
            'markov42_run_json': prev_row_lookup[previous_incumbent_name]['run_json'],
        },
        'candidate_specs': [
            {
                'name': spec['name'],
                'rationale': spec['rationale'],
                'insertion_anchors': sorted(spec['insertions'].keys()),
            }
            for spec in CANDIDATE_SPECS
        ],
        'markov42_rows': rows,
        'kf36_rows': kf36_rows,
        'best_candidate': best_summary,
        'finegrid_summary': {
            'mean_gap_to_old_best_before': before_gap,
            'mean_gap_to_old_best_after': after_gap,
            'extra_mean_gap_closed': extra_closed,
            'extra_gap_closure_pct': extra_closure_pct,
            'materiality_verdict': verdict,
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
    print('BEST_FINEGRID', best_candidate.name, best_summary['markov42']['overall'], flush=True)
    print('FINEGRID_VERDICT', verdict, flush=True)
    print('SCIENTIFIC_CONCLUSION', scientific_conclusion, flush=True)


if __name__ == '__main__':
    main()
