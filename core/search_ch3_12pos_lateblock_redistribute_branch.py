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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def ypair_neg(dwell_s: float, label: str) -> list:
    return closed_pair('inner', -90, 5.0, dwell_s, label)


def zpair_neg(dwell_s: float, label: str) -> list:
    return closed_pair('outer', -90, 5.0, dwell_s, label)


def fullblock(y_s: float, x_s: float, prefix: str) -> list:
    return xpair_outerhold(10.0, f'{prefix}_xpair') + zquad_split(y_s, x_s, y_s, x_s, f'{prefix}_zquad')


CANDIDATE_SPECS = [
    {
        'name': WINNER_NAME,
        'class': 'reference_unified_winner',
        'rationale': 'Current unified winner reference.',
        'insertions': {
            10: zpair_neg(5.0, 'l10_zpair_neg5') + ypair_neg(1.0, 'l10_ypair_neg1'),
            11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zquad_split(10.0, 0.0, 10.0, 2.0, 'zquad_y10x0back2'),
        },
    },
    {
        'name': LOWMAX_NAME,
        'class': 'reference_lowmax_companion',
        'rationale': 'Current low-max companion reference.',
        'insertions': {
            10: zpair_neg(4.0, 'l10_zpair_neg4') + ypair_neg(2.0, 'l10_ypair_neg2'),
            11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zquad_split(10.0, 0.0, 10.0, 2.0, 'zquad_y10x0back2'),
        },
    },
    {
        'name': 'redistribute_l10_l11_fullblock_y8x0',
        'class': 'distributed_double_fullblock',
        'rationale': 'Leave the old local basin and place the same full continuity-safe late-block cycle at both anchors 10 and 11. Theory: repeated sign-separated late Y exposure at two distinct late z-family faces may attack the Ka2_y ceiling more structurally than any single-anchor dose tweak.',
        'insertions': {
            10: fullblock(8.0, 0.0, 'l10_fb_y8x0'),
            11: fullblock(8.0, 0.0, 'l11_fb_y8x0'),
        },
    },
    {
        'name': 'redistribute_l10_l11_fullblock_y10x2',
        'class': 'distributed_double_fullblock',
        'rationale': 'Same distributed double-block idea, but use the stronger mean-oriented full block at both 10 and 11 to test whether multi-anchor redistribution can keep the structure stable while attacking Ka2_y more strongly.',
        'insertions': {
            10: fullblock(10.0, 2.0, 'l10_fb_y10x2'),
            11: fullblock(10.0, 2.0, 'l11_fb_y10x2'),
        },
    },
    {
        'name': 'redistribute_l9_l10_l11_fullblock_y8x0',
        'class': 'distributed_triple_fullblock',
        'rationale': 'Push the structural idea further: distribute the full late-block cycle across anchors 9, 10, and 11 so the weak late segment is excited at three consecutive z-family base states rather than one local pocket.',
        'insertions': {
            9: fullblock(8.0, 0.0, 'l9_fb_y8x0'),
            10: fullblock(8.0, 0.0, 'l10_fb_y8x0'),
            11: fullblock(8.0, 0.0, 'l11_fb_y8x0'),
        },
    },
    {
        'name': 'redistribute_l9_l10_l11_fullblock_y6x0',
        'class': 'distributed_triple_fullblock',
        'rationale': 'Lighter triple-block sibling. Keeps the same three-anchor redistribution but reduces each full-block Y dwell to see whether the ceiling move comes from distribution itself rather than raw dwell size.',
        'insertions': {
            9: fullblock(6.0, 0.0, 'l9_fb_y6x0'),
            10: fullblock(6.0, 0.0, 'l10_fb_y6x0'),
            11: fullblock(6.0, 0.0, 'l11_fb_y6x0'),
        },
    },
    {
        'name': 'redistribute_l9_ypair_neg4_plus_l10_l11_fullblock_y8x0',
        'class': 'preconditioned_triple_branch',
        'rationale': 'Add a dedicated anchor9 inner-y priming loop before the double full-block redistribution on 10/11. Theory: if Ka2_y remains weak because the late segment enters the distributed block with insufficient y-preconditioning, this branch should help.',
        'insertions': {
            9: ypair_neg(4.0, 'l9_ypair_neg4'),
            10: fullblock(8.0, 0.0, 'l10_fb_y8x0'),
            11: fullblock(8.0, 0.0, 'l11_fb_y8x0'),
        },
    },
    {
        'name': 'redistribute_l9_zpair_neg6_plus_l10_l11_fullblock_y8x0',
        'class': 'preconditioned_triple_branch',
        'rationale': 'Alternative anchor9 preconditioning using a strong z-pair instead of a y-pair. This tests whether the redistributed late blocks need a cleaner z-family entrance rather than more local y-dose.',
        'insertions': {
            9: zpair_neg(6.0, 'l9_zpair_neg6'),
            10: fullblock(8.0, 0.0, 'l10_fb_y8x0'),
            11: fullblock(8.0, 0.0, 'l11_fb_y8x0'),
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
    return name not in {WINNER_NAME, LOWMAX_NAME}


def select_kf_rechecks(new_rows: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    if not new_rows:
        return names
    best_mean = min(new_rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    names.append(best_mean['candidate_name'])
    best_max = min(new_rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    if best_max['candidate_name'] not in names:
        names.append(best_max['candidate_name'])
    triple = [row for row in new_rows if row['candidate_class'] == 'distributed_triple_fullblock']
    if triple:
        best_triple = min(triple, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
        if best_triple['candidate_name'] not in names:
            names.append(best_triple['candidate_name'])
    return names


def render_report(payload: dict[str, Any]) -> str:
    refs = payload['references']
    best_new = payload['best_new_candidate']
    best_max = payload['best_new_lowmax_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 late-block structural redistribution branch')
    lines.append('')
    lines.append('## 1. Branch rationale')
    lines.append('')
    lines.append('- The old local post-unify basin only moved max inside the **99.6x** band, so it is not a credible route to `max < 50`.')
    lines.append('- This branch therefore leaves that basin entirely and tests a **structurally different hypothesis**: the Ka2_y ceiling may persist because late excitation is too concentrated at one local spot.')
    lines.append('- New idea: use **distributed, continuity-safe full late-block cycles** across anchors 9/10/11, so the same real-dual-axis-valid sign-separated Y exposure is applied at multiple consecutive late z-family base states.')
    lines.append('- If Ka2_y is limited by local observability saturation rather than total dwell alone, this family is the cleanest way to test that without cheating on body axes or continuity.')
    lines.append('')
    lines.append('## 2. Fixed references')
    lines.append('')
    lines.append(f"- current unified winner: **{refs['current_unified_winner']['markov42']['overall']['mean_pct_error']:.3f} / {refs['current_unified_winner']['markov42']['overall']['median_pct_error']:.3f} / {refs['current_unified_winner']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- low-max companion: **{refs['lowmax_companion']['markov42']['overall']['mean_pct_error']:.3f} / {refs['lowmax_companion']['markov42']['overall']['median_pct_error']:.3f} / {refs['lowmax_companion']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- old best legal non-faithful-base: **{refs['old_best_legal']['markov42']['overall']['mean_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['median_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- default18: **{refs['default18']['markov42']['overall']['mean_pct_error']:.3f} / {refs['default18']['markov42']['overall']['median_pct_error']:.3f} / {refs['default18']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('## 3. Markov42 results')
    lines.append('')
    lines.append('| rank | candidate | class | new? | total_s | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Ka2_z | Δmean vs winner | Δmax vs winner |')
    lines.append('|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(payload['markov42_rows'], start=1):
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['candidate_class']} | {'yes' if row['is_new_candidate'] else 'ref'} | {row['total_time_s']:.1f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {row['delta_vs_unified_winner']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_unified_winner']['max_pct_error']['improvement_pct_points']:+.3f} |"
        )
    lines.append('')
    lines.append('## 4. Best landed candidates')
    lines.append('')
    lines.append(f"- best new candidate by mean/max: **{best_new['candidate_name']}** → **{best_new['markov42']['overall']['mean_pct_error']:.3f} / {best_new['markov42']['overall']['median_pct_error']:.3f} / {best_new['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"  - vs unified winner: Δmean **{best_new['delta_vs_unified_winner']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_new['delta_vs_unified_winner']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- lowest-max new candidate: **{best_max['candidate_name']}** → **{best_max['markov42']['overall']['mean_pct_error']:.3f} / {best_max['markov42']['overall']['median_pct_error']:.3f} / {best_max['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"  - vs unified winner: Δmean **{best_max['delta_vs_unified_winner']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_max['delta_vs_unified_winner']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append('')
    lines.append('## 5. Exact legal motor / timing table for the best new candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for seq_idx, (row, action, face) in enumerate(zip(best_new['all_rows'], best_new['all_actions'], best_new['all_faces']), start=1):
        lines.append(
            f"| {seq_idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 6. KF36 rechecks')
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
    lines.append('## 7. Requested reference comparison')
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
    lines.append('## 8. Verdict')
    lines.append('')
    lines.append(f"- branch verdict: **{payload['bottom_line']['branch_verdict']}**")
    lines.append(f"- sub-50 plausibility: **{payload['bottom_line']['sub50_plausibility']}**")
    lines.append(f"- scientific conclusion: **{payload['scientific_conclusion']}**")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_module('search_ch3_12pos_lateblock_redistribute_branch_src', str(SOURCE_FILE))

    refs = load_reference_payloads(args.noise_scale)

    def unified_kf_for(note: str) -> dict[str, Any]:
        for row in refs['unified'].get('comparison_rows', []):
            if row.get('note') == note:
                return row['kf36']
        raise KeyError(f'unified comparison_rows missing KF36 row for {note}')

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

    for row in rows:
        payload = payload_by_name[row['candidate_name']]
        row['delta_vs_unified_winner'] = delta_vs_ref(current_winner_payload, payload)
        row['delta_vs_lowmax_companion'] = delta_vs_ref(lowmax_payload, payload)
        row['delta_vs_old_best'] = delta_vs_ref(refs['oldbest_markov'], payload)
        row['delta_vs_default18'] = delta_vs_ref(refs['default18_markov'], payload)
        row['delta_vs_faithful'] = delta_vs_ref(refs['faithful_markov'], payload)

    rows.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    new_rows = [row for row in rows if row['is_new_candidate']]
    best_new_row = min(new_rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    lowmax_new_row = min(new_rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))

    best_new_cand = candidate_by_name[best_new_row['candidate_name']]
    lowmax_new_cand = candidate_by_name[lowmax_new_row['candidate_name']]
    best_new_payload = payload_by_name[best_new_cand.name]
    lowmax_new_payload = payload_by_name[lowmax_new_cand.name]

    kf36_rows = []
    for name in select_kf_rechecks(new_rows):
        cand = candidate_by_name[name]
        kf_payload, kf_status, kf_path = run_candidate_payload(mod, cand, 'kf36_noisy', args.noise_scale, args.force_rerun)
        if name == best_new_cand.name:
            note = 'best new distributed branch'
        elif name == lowmax_new_cand.name:
            note = 'lowest-max new distributed branch'
        else:
            note = 'triple-block recheck'
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
        'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], best_new_payload),
        'delta_vs_default18': delta_vs_ref(refs['default18_markov'], best_new_payload),
    }

    lowmax_new_summary = {
        'candidate_name': lowmax_new_cand.name,
        'candidate_class': spec_by_name[lowmax_new_cand.name]['class'],
        'rationale': spec_by_name[lowmax_new_cand.name]['rationale'],
        'total_time_s': lowmax_new_cand.total_time_s,
        'markov42': compact_result(lowmax_new_payload),
        'markov42_run_json': lowmax_new_row['run_json'],
        'delta_vs_unified_winner': delta_vs_ref(current_winner_payload, lowmax_new_payload),
        'delta_vs_lowmax_companion': delta_vs_ref(lowmax_payload, lowmax_new_payload),
        'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], lowmax_new_payload),
        'delta_vs_default18': delta_vs_ref(refs['default18_markov'], lowmax_new_payload),
    }

    def kf_for(name: str) -> dict[str, Any] | None:
        for row in kf36_rows:
            if row['candidate_name'] == name:
                return row['kf36']
        return None

    comparison_rows = [
        {
            'label': 'current unified winner',
            'note': WINNER_NAME,
            'markov42': compact_result(current_winner_payload),
            'kf36': unified_kf_for(WINNER_NAME),
        },
        {
            'label': 'best new distributed branch',
            'note': best_new_cand.name,
            'markov42': best_new_summary['markov42'],
            'kf36': kf_for(best_new_cand.name),
        },
        {
            'label': 'low-max companion',
            'note': LOWMAX_NAME,
            'markov42': compact_result(lowmax_payload),
            'kf36': unified_kf_for(LOWMAX_NAME),
        },
        {
            'label': 'lowest-max new distributed branch',
            'note': lowmax_new_cand.name,
            'markov42': lowmax_new_summary['markov42'],
            'kf36': kf_for(lowmax_new_cand.name),
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
    lowmax_new_delta = lowmax_new_summary['delta_vs_unified_winner']
    best_new_max = best_new_summary['markov42']['overall']['max_pct_error']
    lowmax_new_max = lowmax_new_summary['markov42']['overall']['max_pct_error']

    if lowmax_new_max < 50.0:
        sub50_plausibility = f'YES in this branch: landed candidate already reached max={lowmax_new_max:.3f}.'
    elif lowmax_new_max < 80.0:
        sub50_plausibility = f'weakly plausible: this landed batch reached max={lowmax_new_max:.3f}, so the ceiling moved qualitatively, but sub-50 is still not demonstrated.'
    else:
        sub50_plausibility = f'not plausible from current evidence: even after leaving the old basin, the best landed max is still {lowmax_new_max:.3f}, far from 50.'

    branch_verdict = (
        f'best new distributed branch `{best_new_cand.name}` changed mean/max by '
        f'{best_new_delta["mean_pct_error"]["improvement_pct_points"]:+.3f} / {best_new_delta["max_pct_error"]["improvement_pct_points"]:+.3f} versus the unified winner; '
        f'the strongest max-oriented distributed branch `{lowmax_new_cand.name}` changed them by '
        f'{lowmax_new_delta["mean_pct_error"]["improvement_pct_points"]:+.3f} / {lowmax_new_delta["max_pct_error"]["improvement_pct_points"]:+.3f}.'
    )

    scientific_conclusion = (
        f'This structural redistribution test answered the new question directly. Repeating full continuity-safe late-block cycles across anchors 9/10/11 did create a genuinely different valid family, but the landed batch still reached only max={lowmax_new_max:.3f} at best. '
        f'That means the Ka2_y ceiling is not just a small local-dose problem at 10/11; under the current hard constraints and this distributed closed-loop mechanism-safe design space, the evidence still says sub-50 is not realistically within reach.'
    )

    out_json = RESULTS_DIR / f'ch3_lateblock_redistribute_branch_{make_suffix(args.noise_scale)}.json'
    out_md = REPORTS_DIR / f'psins_ch3_lateblock_redistribute_branch_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_lateblock_redistribute_branch',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'hard_constraints': {
            'base_skeleton': 'faithful chapter-3 12-position original sequence',
            'continuity_rule': 'exact same mechanism state before resume',
            'physical_legality': 'real dual-axis mechanism only',
            'time_budget_s': [1200.0, 1800.0],
            'seed': 42,
            'truth_family': 'shared low-noise benchmark',
            'search_style': 'structural redistribution branch, theory-guided only',
        },
        'branch_design_rationale': {
            'why_old_basin_is_not_enough': 'the old post-unify family only moved max within the 99.6x band, so it is not a credible route to max<50',
            'new_hypothesis': 'Ka2_y remains saturated because late excitation is too concentrated locally; distributing the same legal sign-separated late-block cycle across anchors 9/10/11 may create qualitatively stronger observability',
            'mechanism_safety_rule': 'every insertion is a closed loop at a real reachable anchor state and must reconnect exactly before the next base action',
        },
        'references': {
            'current_unified_winner': {
                'candidate_name': WINNER_NAME,
                'markov42': compact_result(current_winner_payload),
                'kf36': unified_kf_for(WINNER_NAME),
            },
            'lowmax_companion': {
                'candidate_name': LOWMAX_NAME,
                'markov42': compact_result(lowmax_payload),
                'kf36': unified_kf_for(LOWMAX_NAME),
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
        'best_new_candidate': best_new_summary,
        'best_new_lowmax_candidate': lowmax_new_summary,
        'kf36_rows': kf36_rows,
        'comparison_rows': comparison_rows,
        'bottom_line': {
            'branch_verdict': branch_verdict,
            'sub50_plausibility': sub50_plausibility,
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
    print('BEST_NEW', best_new_cand.name, best_new_summary['markov42']['overall'], flush=True)
    print('LOWMAX_NEW', lowmax_new_cand.name, lowmax_new_summary['markov42']['overall'], flush=True)
    print('BOTTOM_LINE', branch_verdict, flush=True)
    print('SUB50', sub50_plausibility, flush=True)


if __name__ == '__main__':
    main()
