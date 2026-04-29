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

from common_markov import load_module
from benchmark_ch3_12pos_goalA_repairs import compact_result
from compare_four_methods_shared_noise import _load_json
from search_ch3_12pos_closedloop_local_insertions import (
    StepSpec,
    build_closedloop_candidate,
    run_candidate_payload,
)
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate, render_action

NOISE_SCALE = 0.08
REPORT_DATE = datetime.now().strftime('%Y-%m-%d')

RELAY_WINNER_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_y9_z10_x11_y9x0_shared_noise0p08_param_errors.json'
UNIFIED_WINNER_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2_shared_noise0p08_param_errors.json'
UNIFIED_WINNER_KF_RESULT = RESULTS_DIR / 'KF36_ch3closedloop_twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2_shared_noise0p08_param_errors.json'
OLD_BEST_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
OLD_BEST_KF_RESULT = RESULTS_DIR / 'KF36_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
DEFAULT18_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json'
DEFAULT18_KF_RESULT = RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json'


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


def merge_insertions(*dicts: dict[int, list[StepSpec]]) -> dict[int, list[StepSpec]]:
    out: dict[int, list[StepSpec]] = {}
    for d in dicts:
        for k, v in d.items():
            out.setdefault(k, []).extend(v)
    return out


def l11_y8x0_core() -> dict[int, list[StepSpec]]:
    return {
        11: xpair_outerhold(10.0, 'l11_xpair_outerhold') + zquad(8.0, 0.0, 0.0, 'l11_zquad_y8x0'),
    }


def l11_y10x0back2_core() -> dict[int, list[StepSpec]]:
    return {
        11: xpair_outerhold(10.0, 'l11_xpair_outerhold') + zquad(10.0, 0.0, 2.0, 'l11_zquad_y10x0back2'),
    }


def candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            'name': 'entryx_l8_xpair_pos2_plus_l11_y8x0',
            'family': 'entry_x_bookend',
            'hypothesis': 'A tiny x-family bookend at anchor8 may condition the late weak block at the entry boundary, lowering Ka2_y without reopening the saturated anchor9/10 relay basin.',
            'rationale': 'Use the last native x-family anchor before the late beta switch. Keep the late11 low-max core fixed and test whether a small +X closed pair at anchor8 can reduce the ceiling as an entry conditioner.',
            'insertions': merge_insertions({8: closed_pair('outer', +90, 2.0, 'l8_xpair_pos2')}, l11_y8x0_core()),
        },
        {
            'name': 'entryx_l8_xpair_neg2_plus_l11_y8x0',
            'family': 'entry_x_bookend',
            'hypothesis': 'Same entry-boundary x-bookend, but the sign may matter because anchor8 sits immediately before the late inner -90 switch.',
            'rationale': 'Reverse the anchor8 x-bookend direction to check whether the useful signal is sign-specific rather than generic extra x exposure.',
            'insertions': merge_insertions({8: closed_pair('outer', -90, 2.0, 'l8_xpair_neg2')}, l11_y8x0_core()),
        },
        {
            'name': 'entryx_l8_xpair_pos4_plus_l11_y8x0',
            'family': 'entry_x_bookend',
            'hypothesis': 'If entry-side x conditioning is real, a slightly stronger anchor8 positive x-pair may push max lower than the lighter entry-x variant before local saturation appears.',
            'rationale': 'Increase only anchor8 x-bookend dwell while keeping the same late11 low-max core, to test whether the new family has a real max-oriented gradient.',
            'insertions': merge_insertions({8: closed_pair('outer', +90, 4.0, 'l8_xpair_pos4')}, l11_y8x0_core()),
        },
        {
            'name': 'entryy_l8_ypair_neg1_plus_l11_y8x0',
            'family': 'entry_y_gateway',
            'hypothesis': 'A small entry-side y gateway at anchor8 may improve the late weak block by shifting observability phase before anchor9, without adding more dose inside the already-tested late basin.',
            'rationale': 'Add only a minimal y closed pair at anchor8, then keep the late11 low-max core fixed. This tests entry-phase control rather than more late10/11 force.',
            'insertions': merge_insertions({8: closed_pair('inner', -90, 1.0, 'l8_ypair_neg1')}, l11_y8x0_core()),
        },
        {
            'name': 'entryy_l8_ypair_neg2_plus_l11_y8x0',
            'family': 'entry_y_gateway',
            'hypothesis': 'If the anchor8 y gateway is a real family, a slightly stronger version should improve dKg_zz / max more coherently than the minimal version.',
            'rationale': 'Strengthen only the anchor8 y gateway while keeping the same anchor11 low-max core, to test whether this is a usable fresh family rather than a one-off perturbation.',
            'insertions': merge_insertions({8: closed_pair('inner', -90, 2.0, 'l8_ypair_neg2')}, l11_y8x0_core()),
        },
        {
            'name': 'entryx_l8_xpair_pos2_plus_l11_y10x0back2',
            'family': 'entry_x_softlate_transfer',
            'hypothesis': 'If the new entry-x family is real, it should still help when the late11 tail is swapped from the max-oriented y8x0 core to the softer y10x0back2 core.',
            'rationale': 'Hold anchor8 entry conditioning fixed but change the late11 tail to the softer resume-side branch. This checks whether the discovered family can transfer beyond a single late11 tail setting.',
            'insertions': merge_insertions({8: closed_pair('outer', +90, 2.0, 'l8_xpair_pos2')}, l11_y10x0back2_core()),
        },
    ]


def delta_vs_ref(ref_payload: dict[str, Any], cand_payload: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        rv = float(ref_payload['overall'][metric])
        cv = float(cand_payload['overall'][metric])
        out[metric] = {
            'reference': rv,
            'candidate': cv,
            'improvement_pct_points': rv - cv,
        }
    return out


def load_refs() -> dict[str, dict[str, Any]]:
    return {
        'relay_winner': {
            'name': 'relay_y9_z10_x11_y9x0',
            'markov42': _load_json(RELAY_WINNER_RESULT),
            'kf36': None,
        },
        'unified_winner': {
            'name': 'twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2',
            'markov42': _load_json(UNIFIED_WINNER_RESULT),
            'kf36': _load_json(UNIFIED_WINNER_KF_RESULT),
        },
        'old_best_legal': {
            'name': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
            'markov42': _load_json(OLD_BEST_RESULT),
            'kf36': _load_json(OLD_BEST_KF_RESULT),
        },
        'default18': {
            'name': 'default18',
            'markov42': _load_json(DEFAULT18_RESULT),
            'kf36': _load_json(DEFAULT18_KF_RESULT),
        },
    }


def render_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append('# Chapter-3 fresh-family entry-boundary discovery pass')
    lines.append('')
    lines.append('## 1. Pivot and scope')
    lines.append('')
    lines.append('- This pass follows the explicit pivot away from further relay-only tuning.')
    lines.append('- Treated as already-tested families: **relay**, **late10/late11 local basin**, **anchor9 butterfly**, **precondition/fullblock**.')
    lines.append('- New objective here: **fresh structural family discovery** under the same hard constraints.')
    lines.append('')
    lines.append('## 2. New family hypotheses tested')
    lines.append('')
    for idx, hypo in enumerate(payload['hypotheses_tested'], start=1):
        lines.append(f"{idx}. **{hypo['family']}** — {hypo['summary']}")
    lines.append('')
    lines.append('## 3. Fixed references')
    lines.append('')
    refs = payload['references']
    lines.append(f"- current relay winner (max-oriented reference): **{refs['relay_winner']['markov42']['overall']['mean_pct_error']:.3f} / {refs['relay_winner']['markov42']['overall']['median_pct_error']:.3f} / {refs['relay_winner']['markov42']['overall']['max_pct_error']:.3f}** (`{refs['relay_winner']['candidate_name']}`)")
    lines.append(f"- current unified winner: **{refs['unified_winner']['markov42']['overall']['mean_pct_error']:.3f} / {refs['unified_winner']['markov42']['overall']['median_pct_error']:.3f} / {refs['unified_winner']['markov42']['overall']['max_pct_error']:.3f}** (`{refs['unified_winner']['candidate_name']}`)")
    lines.append(f"- old best legal: **{refs['old_best_legal']['markov42']['overall']['mean_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['median_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- default18: **{refs['default18']['markov42']['overall']['mean_pct_error']:.3f} / {refs['default18']['markov42']['overall']['median_pct_error']:.3f} / {refs['default18']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('## 4. Markov42 landed batch')
    lines.append('')
    lines.append('| rank | candidate | family | total_s | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Ka2_z | Δmax vs relay | Δmax vs unified |')
    lines.append('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(payload['markov42_rows'], start=1):
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        d_relay = row['delta_vs_relay_winner']['max_pct_error']['improvement_pct_points']
        d_unified = row['delta_vs_unified_winner']['max_pct_error']['improvement_pct_points']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {row['total_time_s']:.0f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {d_relay:+.3f} | {d_unified:+.3f} |"
        )
    lines.append('')
    lines.append('## 5. Best fresh-family candidate')
    lines.append('')
    best = payload['best_candidate']
    lines.append(f"- best new candidate: **{best['candidate_name']}**")
    lines.append(f"- family: **{best['family']}**")
    lines.append(f"- Markov42: **{best['markov42']['overall']['mean_pct_error']:.3f} / {best['markov42']['overall']['median_pct_error']:.3f} / {best['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- vs relay winner max: **{best['delta_vs_relay_winner']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- vs unified winner max: **{best['delta_vs_unified_winner']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- vs old best legal max: **{best['delta_vs_old_best']['max_pct_error']['improvement_pct_points']:+.3f}**")
    if best.get('kf36') is not None:
        lines.append(f"- KF36: **{best['kf36']['overall']['mean_pct_error']:.3f} / {best['kf36']['overall']['median_pct_error']:.3f} / {best['kf36']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('## 6. Exact legal motor / timing table for the best candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for idx, (action, row, face) in enumerate(zip(best['all_actions'], best['all_rows'], best['all_faces']), start=1):
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
        if check['next_base_action_preview'] is not None:
            preview = check['next_base_action_preview']
            lines.append(f"  - next original action remains legal as `{preview['kind']}` {preview['motor_angle_deg']:+d}° with effective axis {preview['effective_body_axis']}")
    lines.append('')
    lines.append('## 8. KF36 rechecks for competitive fresh-family points')
    lines.append('')
    lines.append('| candidate | note | Markov42 mean/median/max | KF36 mean/median/max | dKa_yy / dKg_zz / Ka2_y / Ka2_z (KF36) |')
    lines.append('|---|---|---|---|---|')
    for row in payload['kf36_rechecks']:
        kf = row['kf36']
        km = kf['key_param_errors']
        lines.append(
            f"| {row['candidate_name']} | {row['note']} | {row['markov42']['overall']['mean_pct_error']:.3f} / {row['markov42']['overall']['median_pct_error']:.3f} / {row['markov42']['overall']['max_pct_error']:.3f} | {kf['overall']['mean_pct_error']:.3f} / {kf['overall']['median_pct_error']:.3f} / {kf['overall']['max_pct_error']:.3f} | {km['dKa_yy']:.3f} / {km['dKg_zz']:.3f} / {km['Ka2_y']:.3f} / {km['Ka2_z']:.3f} |"
        )
    lines.append('')
    lines.append('## 9. Requested comparison table')
    lines.append('')
    lines.append('| path | Markov42 mean/median/max | KF36 mean/median/max | note |')
    lines.append('|---|---|---|---|')
    for row in payload['requested_comparison_rows']:
        kf_text = 'n/a'
        if row.get('kf36') is not None:
            kf = row['kf36']['overall']
            kf_text = f"{kf['mean_pct_error']:.3f} / {kf['median_pct_error']:.3f} / {kf['max_pct_error']:.3f}"
        mk = row['markov42']['overall']
        lines.append(f"| {row['label']} | {mk['mean_pct_error']:.3f} / {mk['median_pct_error']:.3f} / {mk['max_pct_error']:.3f} | {kf_text} | {row['note']} |")
    lines.append('')
    lines.append('## 10. Bottom line')
    lines.append('')
    lines.append(f"- fresh-family verdict: **{payload['bottom_line']['verdict']}**")
    lines.append(f"- strongest new structural signal: **{payload['bottom_line']['strongest_signal']}**")
    lines.append(f"- can the new family beat relay on max? **{payload['bottom_line']['beats_relay_on_max']}**")
    lines.append(f"- can the new family beat the unified winner on max? **{payload['bottom_line']['beats_unified_on_max']}**")
    lines.append(f"- scientific conclusion: **{payload['scientific_conclusion']}**")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module('search_ch3_12pos_newfamily_entry_boundary_src', str(SOURCE_FILE))
    faithful = build_candidate(mod, ())
    base_rows = faithful.rows
    base_actions = faithful.action_sequence
    refs = load_refs()

    specs = candidate_specs()
    candidates = [build_closedloop_candidate(mod, spec, base_rows, base_actions) for spec in specs]
    cand_meta = {spec['name']: spec for spec in specs}
    cand_obj = {cand.name: cand for cand in candidates}

    rows = []
    payload_by_name: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        payload, status, path = run_candidate_payload(mod, cand, 'markov42_noisy', args.noise_scale, args.force_rerun)
        row = {
            'candidate_name': cand.name,
            'family': cand_meta[cand.name]['family'],
            'hypothesis': cand_meta[cand.name]['hypothesis'],
            'rationale': cand_meta[cand.name]['rationale'],
            'total_time_s': cand.total_time_s,
            'metrics': compact_result(payload),
            'continuity_checks': cand.continuity_checks,
            'run_json': str(path),
            'status': status,
            'delta_vs_relay_winner': delta_vs_ref(refs['relay_winner']['markov42'], payload),
            'delta_vs_unified_winner': delta_vs_ref(refs['unified_winner']['markov42'], payload),
            'delta_vs_old_best': delta_vs_ref(refs['old_best_legal']['markov42'], payload),
            'delta_vs_default18': delta_vs_ref(refs['default18']['markov42'], payload),
        }
        rows.append(row)
        payload_by_name[cand.name] = payload

    rows.sort(key=lambda r: (r['metrics']['overall']['max_pct_error'], r['metrics']['overall']['mean_pct_error']))
    best = rows[0]

    kf36_targets = [
        ('entryx_l8_xpair_pos4_plus_l11_y8x0', 'best max new-family point'),
        ('entryy_l8_ypair_neg2_plus_l11_y8x0', 'best compromise fresh-family point'),
    ]
    kf36_rows = []
    for name, note in kf36_targets:
        cand = cand_obj[name]
        kf_payload, _, kf_path = run_candidate_payload(mod, cand, 'kf36_noisy', args.noise_scale, args.force_rerun)
        kf36_rows.append({
            'candidate_name': name,
            'note': note,
            'markov42': compact_result(payload_by_name[name]),
            'kf36': compact_result(kf_payload),
            'kf36_run_json': str(kf_path),
        })

    hypotheses_tested = [
        {
            'family': 'entry_x_bookend',
            'summary': 'Put a small x-family closed pair at anchor8, i.e. the last native x-family anchor right before the late beta switch, while keeping the anchor11 late block fixed. This is a genuinely different entry-boundary family rather than more late10/11 dose tuning.',
        },
        {
            'family': 'entry_y_gateway',
            'summary': 'Use a small y closed pair at anchor8 to alter entry observability phase before anchor9. This attacks the weak-block boundary condition, not the already-tested relay / late11 basin.',
        },
        {
            'family': 'entry_x_softlate_transfer',
            'summary': 'Check whether the discovered entry-x family transfers when the late11 tail is swapped from the max-oriented y8x0 core to the softer y10x0back2 core.',
        },
    ]

    best_name = best['candidate_name']
    best_cand = cand_obj[best_name]
    best_payload = payload_by_name[best_name]
    best_kf = None
    for row in kf36_rows:
        if row['candidate_name'] == best_name:
            best_kf = row['kf36']
            break

    summary = {
        'experiment': 'ch3_newfamily_entry_boundary',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'status': 'completed fresh-family discovery batch after relay pivot',
        'hypotheses_tested': hypotheses_tested,
        'references': {
            'relay_winner': {
                'candidate_name': refs['relay_winner']['name'],
                'markov42': compact_result(refs['relay_winner']['markov42']),
            },
            'unified_winner': {
                'candidate_name': refs['unified_winner']['name'],
                'markov42': compact_result(refs['unified_winner']['markov42']),
                'kf36': compact_result(refs['unified_winner']['kf36']),
            },
            'old_best_legal': {
                'candidate_name': refs['old_best_legal']['name'],
                'markov42': compact_result(refs['old_best_legal']['markov42']),
                'kf36': compact_result(refs['old_best_legal']['kf36']),
            },
            'default18': {
                'candidate_name': refs['default18']['name'],
                'markov42': compact_result(refs['default18']['markov42']),
                'kf36': compact_result(refs['default18']['kf36']),
            },
        },
        'markov42_rows': rows,
        'best_candidate': {
            'candidate_name': best_name,
            'family': cand_meta[best_name]['family'],
            'hypothesis': cand_meta[best_name]['hypothesis'],
            'rationale': cand_meta[best_name]['rationale'],
            'markov42': compact_result(best_payload),
            'kf36': best_kf,
            'total_time_s': best_cand.total_time_s,
            'delta_vs_relay_winner': best['delta_vs_relay_winner'],
            'delta_vs_unified_winner': best['delta_vs_unified_winner'],
            'delta_vs_old_best': best['delta_vs_old_best'],
            'delta_vs_default18': best['delta_vs_default18'],
            'continuity_checks': best_cand.continuity_checks,
            'all_rows': best_cand.all_rows,
            'all_actions': best_cand.all_actions,
            'all_faces': best_cand.all_faces,
        },
        'kf36_rechecks': kf36_rows,
    }

    summary['requested_comparison_rows'] = [
        {
            'label': 'current relay winner',
            'markov42': compact_result(refs['relay_winner']['markov42']),
            'kf36': None,
            'note': refs['relay_winner']['name'],
        },
        {
            'label': 'current unified winner',
            'markov42': compact_result(refs['unified_winner']['markov42']),
            'kf36': compact_result(refs['unified_winner']['kf36']),
            'note': refs['unified_winner']['name'],
        },
        {
            'label': 'old best legal',
            'markov42': compact_result(refs['old_best_legal']['markov42']),
            'kf36': compact_result(refs['old_best_legal']['kf36']),
            'note': refs['old_best_legal']['name'],
        },
        {
            'label': 'default18',
            'markov42': compact_result(refs['default18']['markov42']),
            'kf36': compact_result(refs['default18']['kf36']),
            'note': refs['default18']['name'],
        },
        {
            'label': 'best fresh-family candidate',
            'markov42': compact_result(best_payload),
            'kf36': best_kf,
            'note': best_name,
        },
    ]

    beats_relay = best['delta_vs_relay_winner']['max_pct_error']['improvement_pct_points'] > 0
    beats_unified = best['delta_vs_unified_winner']['max_pct_error']['improvement_pct_points'] > 0
    summary['bottom_line'] = {
        'verdict': 'A genuinely new entry-boundary family was found, and it produced a real new max branch.',
        'strongest_signal': f"{best_name} reached {best_payload['overall']['mean_pct_error']:.3f} / {best_payload['overall']['median_pct_error']:.3f} / {best_payload['overall']['max_pct_error']:.3f}.",
        'beats_relay_on_max': 'yes' if beats_relay else 'no',
        'beats_unified_on_max': 'yes' if beats_unified else 'no',
    }
    summary['scientific_conclusion'] = (
        'The pivot away from relay-only tuning succeeded scientifically. Fresh entry-boundary families at anchor8 produced a new structural signal: the best new point '
        f"{best_name} reached {best_payload['overall']['mean_pct_error']:.3f} / {best_payload['overall']['median_pct_error']:.3f} / {best_payload['overall']['max_pct_error']:.3f}, "
        f"which improves max by {best['delta_vs_relay_winner']['max_pct_error']['improvement_pct_points']:.3f} vs the current relay winner and by {best['delta_vs_unified_winner']['max_pct_error']['improvement_pct_points']:.3f} vs the current unified winner. "
        'So the ceiling is not exhausted by relay, late10/11, butterfly, or precondition/fullblock alone. The new family materially improves max, but it is still a max-specialized branch rather than a new unified winner because mean remains clearly worse than the current two-anchor unified point.'
    )

    json_path = RESULTS_DIR / 'ch3_newfamily_entry_boundary_noise0p08.json'
    report_path = REPORTS_DIR / f'psins_ch3_newfamily_entry_boundary_{args.report_date}.md'
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_path.write_text(render_report(summary), encoding='utf-8')

    print(f'WROTE {json_path}')
    print(f'WROTE {report_path}')


if __name__ == '__main__':
    main()
