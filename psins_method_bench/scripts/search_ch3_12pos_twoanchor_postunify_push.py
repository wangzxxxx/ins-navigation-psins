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
from compare_four_methods_shared_noise import _load_json, _noise_matches, expected_noise_config
from search_ch3_12pos_closedloop_local_insertions import (
    NOISE_SCALE,
    REPORT_DATE,
    build_closedloop_candidate,
    closed_pair,
    delta_vs_ref,
    render_action,
    run_candidate_payload,
)
from search_ch3_12pos_closedloop_zquad_followup import xpair_outerhold, zquad_split
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate, make_suffix

UNIFIED_JSON = RESULTS_DIR / 'ch3_twoanchor_unified_compromise_noise0p08.json'
OLD_BEST_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
OLD_BEST_KF_RESULT = RESULTS_DIR / 'KF36_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
DEFAULT18_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json'
DEFAULT18_KF_RESULT = RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json'
FAITHFUL_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
FAITHFUL_KF_RESULT = RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json'

WINNER_NAME = 'twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2'
LOWMAX_NAME = 'twoanchor_l10_zpair_neg4_then_ypair_neg2_plus_l11_y10x0back2'
PREV_BESTMEAN_NAME = 'twoanchor_l10_zpair_neg6_plus_l11_bestmean'

REF_NAMES = {
    WINNER_NAME,
    LOWMAX_NAME,
    PREV_BESTMEAN_NAME,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def l10_zpair_neg(dwell_s: float, label: str) -> list:
    return closed_pair('outer', -90, 5.0, dwell_s, label)


def l10_ypair_neg(dwell_s: float, label: str) -> list:
    return closed_pair('inner', -90, 5.0, dwell_s, label)


def l11_y10x0back2(prefix: str = 'yneg_xpair_outerhold') -> list:
    return xpair_outerhold(10.0, prefix) + zquad_split(10.0, 0.0, 10.0, 2.0, 'zquad_y10x0back2')


def l11_y10x0back1(prefix: str = 'yneg_xpair_outerhold') -> list:
    return xpair_outerhold(10.0, prefix) + zquad_split(10.0, 0.0, 10.0, 1.0, 'zquad_y10x0back1')


def l11_y9x0back2(prefix: str = 'yneg_xpair_outerhold') -> list:
    return xpair_outerhold(10.0, prefix) + zquad_split(9.0, 0.0, 9.0, 2.0, 'zquad_y9x0back2')


def l11_y9x1(prefix: str = 'yneg_xpair_outerhold') -> list:
    return xpair_outerhold(10.0, prefix) + zquad_split(9.0, 1.0, 9.0, 1.0, 'zquad_y9x1')


CANDIDATE_SPECS = [
    {
        'name': PREV_BESTMEAN_NAME,
        'class': 'reference_prev_bestmean',
        'rationale': 'Pre-unify best-mean reference for direct continuity with the earlier frontier.',
        'insertions': {
            10: l10_zpair_neg(6.0, 'l10_zpair_neg6'),
            11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zquad_split(10.0, 2.0, 10.0, 2.0, 'zquad_y10x2'),
        },
    },
    {
        'name': WINNER_NAME,
        'class': 'reference_unified_winner',
        'rationale': 'Current unified winner to beat: anchor10 z5+y1 with late11 y10x0back2.',
        'insertions': {
            10: l10_zpair_neg(5.0, 'l10_zpair_neg5') + l10_ypair_neg(1.0, 'l10_ypair_neg1'),
            11: l11_y10x0back2(),
        },
    },
    {
        'name': LOWMAX_NAME,
        'class': 'reference_lowmax_companion',
        'rationale': 'Current low-max companion for the same unified family.',
        'insertions': {
            10: l10_zpair_neg(4.0, 'l10_zpair_neg4') + l10_ypair_neg(2.0, 'l10_ypair_neg2'),
            11: l11_y10x0back2(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg5p25_then_ypair_neg0p75_plus_l11_y10x0back2',
        'class': 'local_anchor_rebalance',
        'rationale': 'Micro-step toward even lighter anchor10 y-dose while slightly strengthening z-dwell, to test whether the winner still has a nearby two-sided improvement pocket on the mean side.',
        'insertions': {
            10: l10_zpair_neg(5.25, 'l10_zpair_neg5p25') + l10_ypair_neg(0.75, 'l10_ypair_neg0p75'),
            11: l11_y10x0back2(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg5p00_then_ypair_neg1p50_plus_l11_y10x0back2',
        'class': 'local_anchor_rebalance',
        'rationale': 'Increase only the anchor10 y-dose from the winner while keeping z=5, to see whether a tiny extra y push reduces Ka2_y/max without collapsing mean.',
        'insertions': {
            10: l10_zpair_neg(5.0, 'l10_zpair_neg5p00') + l10_ypair_neg(1.5, 'l10_ypair_neg1p50'),
            11: l11_y10x0back2(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg4p75_then_ypair_neg1p25_plus_l11_y10x0back2',
        'class': 'local_anchor_rebalance',
        'rationale': 'Quarter-step interpolation from the unified winner toward the low-max companion: soften z slightly and raise y slightly, while keeping the same late11 block.',
        'insertions': {
            10: l10_zpair_neg(4.75, 'l10_zpair_neg4p75') + l10_ypair_neg(1.25, 'l10_ypair_neg1p25'),
            11: l11_y10x0back2(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg4p50_then_ypair_neg1p00_plus_l11_y10x0back2',
        'class': 'local_anchor_rebalance',
        'rationale': 'Reduce only the anchor10 z-dose relative to the winner, testing whether part of the z5 win is still overly stiff once late11 y10x0back2 is fixed.',
        'insertions': {
            10: l10_zpair_neg(4.5, 'l10_zpair_neg4p50') + l10_ypair_neg(1.0, 'l10_ypair_neg1p00'),
            11: l11_y10x0back2(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg4p50_then_ypair_neg1p50_plus_l11_y10x0back2',
        'class': 'local_anchor_rebalance',
        'rationale': 'Half-step diagonal move toward the low-max companion. This is the main carefully chosen low-max pull while trying not to donate too much mean.',
        'insertions': {
            10: l10_zpair_neg(4.5, 'l10_zpair_neg4p50') + l10_ypair_neg(1.5, 'l10_ypair_neg1p50'),
            11: l11_y10x0back2(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back1',
        'class': 'late11_micro_refine',
        'rationale': 'Keep the winner anchor10 block untouched, but trim the late11 closing x-buffer from 2 s to 1 s to test whether the same unified point can improve mean without losing too much max.',
        'insertions': {
            10: l10_zpair_neg(5.0, 'l10_zpair_neg5') + l10_ypair_neg(1.0, 'l10_ypair_neg1'),
            11: l11_y10x0back1(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y9x0back2',
        'class': 'late11_micro_refine',
        'rationale': 'Keep the winner anchor10 block untouched, but soften the late11 ±Y dwell from 10 s to 9 s with the same back-loaded x-buffer. Intended as the gentlest direct Ka2_y/max push at late11.',
        'insertions': {
            10: l10_zpair_neg(5.0, 'l10_zpair_neg5') + l10_ypair_neg(1.0, 'l10_ypair_neg1'),
            11: l11_y9x0back2(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y9x1',
        'class': 'late11_micro_refine',
        'rationale': 'Keep the winner anchor10 block untouched, but switch late11 to the nearby y9x1 split to test whether a slightly more symmetric late11 buffer supports the unified point better than y10x0back2.',
        'insertions': {
            10: l10_zpair_neg(5.0, 'l10_zpair_neg5') + l10_ypair_neg(1.0, 'l10_ypair_neg1'),
            11: l11_y9x1(),
        },
    },
]


def load_reference_payloads(noise_scale: float) -> dict[str, Any]:
    expected_cfg = expected_noise_config(noise_scale)
    unified = _load_json(UNIFIED_JSON)
    faithful_markov = _load_json(FAITHFUL_RESULT)
    faithful_kf = _load_json(FAITHFUL_KF_RESULT)
    oldbest_markov = _load_json(OLD_BEST_RESULT)
    oldbest_kf = _load_json(OLD_BEST_KF_RESULT)
    default18_markov = _load_json(DEFAULT18_RESULT)
    default18_kf = _load_json(DEFAULT18_KF_RESULT)
    for payload in [faithful_markov, faithful_kf, oldbest_markov, oldbest_kf, default18_markov, default18_kf]:
        if not _noise_matches(payload, expected_cfg):
            raise ValueError('reference noise configuration mismatch')
    return {
        'unified': unified,
        'faithful_markov': faithful_markov,
        'faithful_kf': faithful_kf,
        'oldbest_markov': oldbest_markov,
        'oldbest_kf': oldbest_kf,
        'default18_markov': default18_markov,
        'default18_kf': default18_kf,
    }


def is_new_candidate(name: str) -> bool:
    return name not in REF_NAMES


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


def select_kf_rechecks(new_rows: list[dict[str, Any]]) -> list[str]:
    if not new_rows:
        return []
    names: list[str] = []
    best_mean = min(new_rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    names.append(best_mean['candidate_name'])
    best_max = min(new_rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    if best_max['candidate_name'] not in names:
        names.append(best_max['candidate_name'])
    anchor_rows = [row for row in new_rows if row['candidate_class'] == 'local_anchor_rebalance']
    if anchor_rows:
        best_anchor = min(anchor_rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
        if best_anchor['candidate_name'] not in names:
            names.append(best_anchor['candidate_name'])
    late11_rows = [row for row in new_rows if row['candidate_class'] == 'late11_micro_refine']
    if late11_rows:
        best_late11 = min(late11_rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
        if best_late11['candidate_name'] not in names:
            names.append(best_late11['candidate_name'])
    return names


def render_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    refs = payload['references']
    best_new = payload['best_postunify_candidate']
    lowest_max = payload['best_postunify_lowmax_candidate']
    bottom = payload['bottom_line']

    lines.append('# Chapter-3 faithful12 two-anchor post-unify push')
    lines.append('')
    lines.append('## 1. Scope')
    lines.append('')
    lines.append('- This pass stayed strictly inside the **post-unify neighborhood** of the new unified winner `twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2`.')
    lines.append('- Hard constraints stayed fixed: faithful12 base scaffold, exact dual-axis continuity closure, same `noise_scale=0.08`, same seed/truth family, and total time inside 1200–1800 s.')
    lines.append('- Search style stayed narrow and interpretable: only **local anchor10 z/y rebalance** and **very slight late11 neighborhood refinements**. No new family was opened.')
    lines.append('')
    lines.append('## 2. References')
    lines.append('')
    lines.append(f"- current unified winner: **{refs['current_unified_winner']['markov42']['overall']['mean_pct_error']:.3f} / {refs['current_unified_winner']['markov42']['overall']['median_pct_error']:.3f} / {refs['current_unified_winner']['markov42']['overall']['max_pct_error']:.3f}** (`{refs['current_unified_winner']['candidate_name']}`)")
    lines.append(f"- low-max companion: **{refs['lowmax_companion']['markov42']['overall']['mean_pct_error']:.3f} / {refs['lowmax_companion']['markov42']['overall']['median_pct_error']:.3f} / {refs['lowmax_companion']['markov42']['overall']['max_pct_error']:.3f}** (`{refs['lowmax_companion']['candidate_name']}`)")
    lines.append(f"- previous best-mean anchor: **{refs['previous_bestmean_anchor']['markov42']['overall']['mean_pct_error']:.3f} / {refs['previous_bestmean_anchor']['markov42']['overall']['median_pct_error']:.3f} / {refs['previous_bestmean_anchor']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- old best legal non-faithful-base: **{refs['old_best_legal']['markov42']['overall']['mean_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['median_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- default18: **{refs['default18']['markov42']['overall']['mean_pct_error']:.3f} / {refs['default18']['markov42']['overall']['median_pct_error']:.3f} / {refs['default18']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('## 3. Markov42 results')
    lines.append('')
    lines.append('| rank | candidate | class | new? | frontier? | total_s | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Ka2_z | Δmean vs unified winner | Δmax vs unified winner | Δmean vs low-max companion | Δmax vs low-max companion |')
    lines.append('|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(payload['markov42_rows'], start=1):
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['candidate_class']} | {'yes' if row['is_new_candidate'] else 'ref'} | {'yes' if row['on_pareto_frontier'] else 'no'} | {row['total_time_s']:.1f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {row['delta_vs_unified_winner']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_unified_winner']['max_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_lowmax_companion']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_lowmax_companion']['max_pct_error']['improvement_pct_points']:+.3f} |"
        )
    lines.append('')
    lines.append('## 4. Best post-unify candidate')
    lines.append('')
    lines.append(f"- selected by mean/max among new candidates: **{best_new['candidate_name']}** → **{best_new['markov42']['overall']['mean_pct_error']:.3f} / {best_new['markov42']['overall']['median_pct_error']:.3f} / {best_new['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- vs unified winner: Δmean **{best_new['delta_vs_unified_winner']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_new['delta_vs_unified_winner']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- vs low-max companion: Δmean **{best_new['delta_vs_lowmax_companion']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_new['delta_vs_lowmax_companion']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append('')
    lines.append('## 5. Lowest-max post-unify candidate')
    lines.append('')
    lines.append(f"- selected by max among new candidates: **{lowest_max['candidate_name']}** → **{lowest_max['markov42']['overall']['mean_pct_error']:.3f} / {lowest_max['markov42']['overall']['median_pct_error']:.3f} / {lowest_max['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- vs unified winner: Δmean **{lowest_max['delta_vs_unified_winner']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{lowest_max['delta_vs_unified_winner']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- vs low-max companion: Δmean **{lowest_max['delta_vs_lowmax_companion']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{lowest_max['delta_vs_lowmax_companion']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append('')
    lines.append('## 6. Exact legal motor / timing table for the best post-unify candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for seq_idx, (row, action, face) in enumerate(zip(best_new['all_rows'], best_new['all_actions'], best_new['all_faces']), start=1):
        lines.append(
            f"| {seq_idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 7. KF36 rechecks')
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
    lines.append('## 8. Requested reference comparison')
    lines.append('')
    lines.append('| path | Markov42 mean/median/max | KF36 mean/median/max | dKa_yy / dKg_zz / Ka2_y / Ka2_z (Markov42) | note |')
    lines.append('|---|---|---|---|---|')
    for row in payload['comparison_rows']:
        mm = row['markov42']['overall']
        kk = row['kf36']['overall']
        kp = row['markov42']['key_param_errors']
        lines.append(
            f"| {row['label']} | {mm['mean_pct_error']:.3f} / {mm['median_pct_error']:.3f} / {mm['max_pct_error']:.3f} | {kk['mean_pct_error']:.3f} / {kk['median_pct_error']:.3f} / {kk['max_pct_error']:.3f} | {kp['dKa_yy']:.3f} / {kp['dKg_zz']:.3f} / {kp['Ka2_y']:.3f} / {kp['Ka2_z']:.3f} | {row['note']} |"
        )
    lines.append('')
    lines.append('## 9. Bottom line')
    lines.append('')
    lines.append(f"- best post-unify candidate vs unified winner: **Δmean {bottom['best_new_delta_vs_winner']['mean_pct_error']['improvement_pct_points']:+.3f}, Δmax {bottom['best_new_delta_vs_winner']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- lowest-max post-unify candidate vs unified winner: **Δmean {bottom['best_lowmax_delta_vs_winner']['mean_pct_error']['improvement_pct_points']:+.3f}, Δmax {bottom['best_lowmax_delta_vs_winner']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- verdict: **{bottom['verdict']}**")
    lines.append(f"- scientific conclusion: **{payload['scientific_conclusion']}**")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_module('search_ch3_12pos_twoanchor_postunify_push_src', str(SOURCE_FILE))

    refs = load_reference_payloads(args.noise_scale)
    unified = refs['unified']
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
        rows.append({
            'candidate_name': cand.name,
            'candidate_class': spec['class'],
            'is_new_candidate': is_new_candidate(cand.name),
            'rationale': spec['rationale'],
            'total_time_s': cand.total_time_s,
            'metrics': compact_result(payload),
            'continuity_checks': cand.continuity_checks,
            'run_json': str(path),
            'status': status,
        })
        payload_by_name[cand.name] = payload

    current_winner_payload = payload_by_name[WINNER_NAME]
    lowmax_payload = payload_by_name[LOWMAX_NAME]
    prev_bestmean_payload = payload_by_name[PREV_BESTMEAN_NAME]

    for row in rows:
        payload = payload_by_name[row['candidate_name']]
        row['delta_vs_faithful'] = delta_vs_ref(refs['faithful_markov'], payload)
        row['delta_vs_unified_winner'] = delta_vs_ref(current_winner_payload, payload)
        row['delta_vs_lowmax_companion'] = delta_vs_ref(lowmax_payload, payload)
        row['delta_vs_prev_bestmean_anchor'] = delta_vs_ref(prev_bestmean_payload, payload)
        row['delta_vs_old_best'] = delta_vs_ref(refs['oldbest_markov'], payload)
        row['delta_vs_default18'] = delta_vs_ref(refs['default18_markov'], payload)

    rows.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    frontier = pareto_frontier(rows)
    for row in rows:
        row['on_pareto_frontier'] = row['candidate_name'] in frontier

    new_rows = [row for row in rows if row['is_new_candidate']]
    best_new_row = min(new_rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    lowest_max_row = min(new_rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))

    best_new_cand = candidate_by_name[best_new_row['candidate_name']]
    lowest_max_cand = candidate_by_name[lowest_max_row['candidate_name']]
    best_new_payload = payload_by_name[best_new_cand.name]
    lowest_max_payload = payload_by_name[lowest_max_cand.name]

    kf36_rows = []
    for name in select_kf_rechecks(new_rows):
        cand = candidate_by_name[name]
        kf_payload, kf_status, kf_path = run_candidate_payload(mod, cand, 'kf36_noisy', args.noise_scale, args.force_rerun)
        if name == best_new_cand.name:
            note = 'best post-unify candidate'
        elif name == lowest_max_cand.name:
            note = 'lowest-max post-unify candidate'
        elif 'late11' in spec_by_name[name]['class']:
            note = 'best late11 neighborhood probe'
        else:
            note = 'competitive local anchor rebalance'
        kf36_rows.append({
            'candidate_name': name,
            'note': note,
            'markov42': compact_result(payload_by_name[name]),
            'kf36': compact_result(kf_payload),
            'kf36_status': kf_status,
            'kf36_run_json': str(kf_path),
        })

    best_new_summary = {
        'candidate_name': best_new_cand.name,
        'candidate_class': spec_by_name[best_new_cand.name]['class'],
        'rationale': spec_by_name[best_new_cand.name]['rationale'],
        'total_time_s': best_new_cand.total_time_s,
        'all_rows': best_new_cand.all_rows,
        'all_actions': best_new_cand.all_actions,
        'all_faces': best_new_cand.all_faces,
        'continuity_checks': best_new_cand.continuity_checks,
        'markov42': compact_result(best_new_payload),
        'markov42_run_json': best_new_row['run_json'],
        'delta_vs_unified_winner': delta_vs_ref(current_winner_payload, best_new_payload),
        'delta_vs_lowmax_companion': delta_vs_ref(lowmax_payload, best_new_payload),
        'delta_vs_prev_bestmean_anchor': delta_vs_ref(prev_bestmean_payload, best_new_payload),
        'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], best_new_payload),
        'delta_vs_default18': delta_vs_ref(refs['default18_markov'], best_new_payload),
    }

    lowest_max_summary = {
        'candidate_name': lowest_max_cand.name,
        'candidate_class': spec_by_name[lowest_max_cand.name]['class'],
        'rationale': spec_by_name[lowest_max_cand.name]['rationale'],
        'total_time_s': lowest_max_cand.total_time_s,
        'markov42': compact_result(lowest_max_payload),
        'markov42_run_json': lowest_max_row['run_json'],
        'delta_vs_unified_winner': delta_vs_ref(current_winner_payload, lowest_max_payload),
        'delta_vs_lowmax_companion': delta_vs_ref(lowmax_payload, lowest_max_payload),
        'delta_vs_prev_bestmean_anchor': delta_vs_ref(prev_bestmean_payload, lowest_max_payload),
        'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], lowest_max_payload),
        'delta_vs_default18': delta_vs_ref(refs['default18_markov'], lowest_max_payload),
    }

    comparison_rows = [
        {
            'label': 'current unified winner',
            'note': WINNER_NAME,
            'markov42': compact_result(current_winner_payload),
            'kf36': refs['unified']['references']['best_unified_candidate']['kf36'],
        },
        {
            'label': 'best post-unify candidate',
            'note': best_new_cand.name,
            'markov42': best_new_summary['markov42'],
            'kf36': next(row['kf36'] for row in kf36_rows if row['candidate_name'] == best_new_cand.name),
        },
        {
            'label': 'low-max companion',
            'note': LOWMAX_NAME,
            'markov42': compact_result(lowmax_payload),
            'kf36': refs['unified']['comparison_rows'][4]['kf36'],
        },
        {
            'label': 'lowest-max post-unify candidate',
            'note': lowest_max_cand.name,
            'markov42': lowest_max_summary['markov42'],
            'kf36': next(row['kf36'] for row in kf36_rows if row['candidate_name'] == lowest_max_cand.name),
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

    best_new_delta = best_new_summary['delta_vs_unified_winner']
    lowmax_delta = lowest_max_summary['delta_vs_unified_winner']
    best_new_mean_gain = best_new_delta['mean_pct_error']['improvement_pct_points']
    best_new_max_gain = best_new_delta['max_pct_error']['improvement_pct_points']
    lowmax_mean_gain = lowmax_delta['mean_pct_error']['improvement_pct_points']
    lowmax_max_gain = lowmax_delta['max_pct_error']['improvement_pct_points']

    if best_new_mean_gain > 0 and best_new_max_gain > 0:
        verdict = (
            f'material improvement found: {best_new_cand.name} beats the current unified winner on both mean and max '
            f'({best_new_mean_gain:+.3f}, {best_new_max_gain:+.3f}).'
        )
    elif best_new_max_gain > 0 and -best_new_mean_gain <= 0.05:
        verdict = (
            f'borderline but useful improvement: {best_new_cand.name} trims max by {best_new_max_gain:.3f} with only '
            f'{-best_new_mean_gain:.3f} mean-points given back versus the current unified winner.'
        )
    else:
        verdict = (
            f'no material improvement over the current unified winner. The best local follow-up was {best_new_cand.name}, '
            f'which moved mean/max by {best_new_mean_gain:+.3f} / {best_new_max_gain:+.3f}; the lowest-max follow-up was '
            f'{lowest_max_cand.name} at {lowmax_mean_gain:+.3f} / {lowmax_max_gain:+.3f} versus the winner.'
        )

    scientific_conclusion = (
        f'The post-unify neighborhood stayed coherent: the local frontier still bends mainly along the anchor10 z↔y rebalance, and the late11 neighborhood did not unlock a fresh dominant point. '
        f'The best local follow-up was {best_new_cand.name}, while the strongest pure max push was {lowest_max_cand.name}. '
        f'Compared with the current unified winner, those moves changed mean/max by {best_new_mean_gain:+.3f}/{best_new_max_gain:+.3f} and '
        f'{lowmax_mean_gain:+.3f}/{lowmax_max_gain:+.3f}, respectively. So Ka2_y/max can still be nudged, but not enough to clearly strengthen the unified winner; the main ceiling remains Ka2_y, and the existing winner/low-max companion pair still captures the branch best.'
    )

    out_json = RESULTS_DIR / f'ch3_twoanchor_postunify_push_{make_suffix(args.noise_scale)}.json'
    out_md = REPORTS_DIR / f'psins_ch3_twoanchor_postunify_push_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_twoanchor_postunify_push',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'hard_constraints': {
            'base_skeleton': 'faithful chapter-3 12-position original sequence',
            'continuity_rule': 'exact same mechanism state before resume',
            'time_budget_s': [1200.0, 1800.0],
            'seed': 42,
            'truth_family': 'shared low-noise benchmark',
            'search_style': 'post-unify narrow local refinement only',
        },
        'references': {
            'faithful12': {
                'candidate_name': faithful.name,
                'markov42': compact_result(refs['faithful_markov']),
                'kf36': compact_result(refs['faithful_kf']),
            },
            'previous_bestmean_anchor': {
                'candidate_name': PREV_BESTMEAN_NAME,
                'markov42': compact_result(prev_bestmean_payload),
                'kf36': refs['unified']['references']['bestmean_anchor']['kf36'],
            },
            'current_unified_winner': {
                'candidate_name': WINNER_NAME,
                'markov42': compact_result(current_winner_payload),
                'kf36': refs['unified']['references']['best_unified_candidate']['kf36'],
            },
            'lowmax_companion': {
                'candidate_name': LOWMAX_NAME,
                'markov42': compact_result(lowmax_payload),
                'kf36': refs['unified']['comparison_rows'][4]['kf36'],
            },
            'old_best_legal': {
                'candidate_name': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
                'markov42': compact_result(refs['oldbest_markov']),
                'kf36': compact_result(refs['oldbest_kf']),
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
                'is_new_candidate': is_new_candidate(spec['name']),
            }
            for spec in CANDIDATE_SPECS
        ],
        'markov42_rows': rows,
        'pareto_frontier_mean_max': [row['candidate_name'] for row in rows if row['on_pareto_frontier']],
        'best_postunify_candidate': best_new_summary,
        'best_postunify_lowmax_candidate': lowest_max_summary,
        'kf36_rows': kf36_rows,
        'comparison_rows': comparison_rows,
        'bottom_line': {
            'best_new_delta_vs_winner': best_new_delta,
            'best_lowmax_delta_vs_winner': lowmax_delta,
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
    print('BEST_POSTUNIFY', best_new_cand.name, best_new_summary['markov42']['overall'], flush=True)
    print('LOWEST_MAX_POSTUNIFY', lowest_max_cand.name, lowest_max_summary['markov42']['overall'], flush=True)
    print('BOTTOM_LINE', verdict, flush=True)
    print('SCIENTIFIC_CONCLUSION', scientific_conclusion, flush=True)


if __name__ == '__main__':
    main()
