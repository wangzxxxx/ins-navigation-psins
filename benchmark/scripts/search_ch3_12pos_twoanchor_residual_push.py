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
    load_reference_payloads,
    make_suffix,
    render_action,
    run_candidate_payload,
)
from search_ch3_12pos_closedloop_zquad_followup import load_extra_reference_payloads as load_late11_refs
from search_ch3_12pos_closedloop_zquad_followup import xpair_outerhold, zquad_split
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate

DEFAULT18_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json'
DEFAULT18_KF_RESULT = RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json'


REF_NAMES = {
    'twoanchor_l10_zpair_neg6_plus_l11_bestmean',
    'twoanchor_l10_zpair_neg4_plus_l11_maxbest',
    'twoanchor_l10_ypair_neg4_plus_l11_maxbest',
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


def l11_bestmean(prefix: str = 'yneg_xpair_outerhold') -> list:
    return xpair_outerhold(10.0, prefix) + zquad_split(10.0, 2.0, 10.0, 2.0, 'zquad_y10x2')


def l11_maxbest(prefix: str = 'yneg_xpair_outerhold') -> list:
    return xpair_outerhold(10.0, prefix) + zquad_split(8.0, 0.0, 8.0, 0.0, 'zquad_y8x0')


def l11_y9x1(prefix: str = 'yneg_xpair_outerhold') -> list:
    return xpair_outerhold(10.0, prefix) + zquad_split(9.0, 1.0, 9.0, 1.0, 'zquad_y9x1')


def l11_ypos10x1yneg8x1(prefix: str = 'yneg_xpair_outerhold') -> list:
    return xpair_outerhold(10.0, prefix) + zquad_split(10.0, 1.0, 8.0, 1.0, 'zquad_ypos10x1yneg8x1')


def l11_y10x0back2(prefix: str = 'yneg_xpair_outerhold') -> list:
    return xpair_outerhold(10.0, prefix) + zquad_split(10.0, 0.0, 10.0, 2.0, 'zquad_y10x0back2')


CANDIDATE_SPECS = [
    {
        'name': 'twoanchor_l10_zpair_neg6_plus_l11_bestmean',
        'class': 'reference_bestmean',
        'rationale': 'Current best-mean two-anchor reference: keep the strong anchor10 negative z pair and the current best-mean late11 zquad motif.',
        'insertions': {
            10: l10_zpair_neg(6.0, 'l10_zpair_neg6'),
            11: l11_bestmean(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg4_plus_l11_maxbest',
        'class': 'reference_compromise',
        'rationale': 'Current cleanest saved compromise point inside the old frontier.',
        'insertions': {
            10: l10_zpair_neg(4.0, 'l10_zpair_neg4'),
            11: l11_maxbest(),
        },
    },
    {
        'name': 'twoanchor_l10_ypair_neg4_plus_l11_maxbest',
        'class': 'reference_bestmax',
        'rationale': 'Current max-oriented two-anchor endpoint.',
        'insertions': {
            10: l10_ypair_neg(4.0, 'l10_ypair_neg4'),
            11: l11_maxbest(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg6_plus_l11_maxbest',
        'class': 'new_bridge',
        'rationale': 'Direct missing bridge between the current best-mean endpoint and the max-oriented late11 sibling: keep the stronger anchor10 neg6 z-pair, but swap the late11 block to maxbest.',
        'insertions': {
            10: l10_zpair_neg(6.0, 'l10_zpair_neg6'),
            11: l11_maxbest(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg5_plus_l11_maxbest',
        'class': 'new_bridge',
        'rationale': 'Midpoint bridge inside the anchor10 z-pair dwell between the saved neg6 and neg4 branches, still using the late11 maxbest motif.',
        'insertions': {
            10: l10_zpair_neg(5.0, 'l10_zpair_neg5'),
            11: l11_maxbest(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg5_plus_l11_bestmean',
        'class': 'new_bridge',
        'rationale': 'Midpoint refinement of the current best-mean family: same late11 best-mean motif, but anchor10 neg-z dwell softened from 6 s to 5 s.',
        'insertions': {
            10: l10_zpair_neg(5.0, 'l10_zpair_neg5'),
            11: l11_bestmean(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg6_plus_l11_y9x1',
        'class': 'new_l11_refine',
        'rationale': 'Keep the current anchor10 neg6 z-pair, but locally soften the late11 zquad from y10/x2 to the y9/x1 split.',
        'insertions': {
            10: l10_zpair_neg(6.0, 'l10_zpair_neg6'),
            11: l11_y9x1(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg6_plus_l11_ypos10x1yneg8x1',
        'class': 'new_l11_refine',
        'rationale': 'Keep the current anchor10 neg6 z-pair, but use the late11 asymmetric positive/negative-Y split that slightly softens the return-side Y exposure.',
        'insertions': {
            10: l10_zpair_neg(6.0, 'l10_zpair_neg6'),
            11: l11_ypos10x1yneg8x1(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg6_plus_l11_y10x0back2',
        'class': 'new_l11_refine',
        'rationale': 'Keep the current anchor10 neg6 z-pair, but move the late11 x-buffer fully to the resume side to test whether resume-side protection can push max down without giving back too much mean.',
        'insertions': {
            10: l10_zpair_neg(6.0, 'l10_zpair_neg6'),
            11: l11_y10x0back2(),
        },
    },
    {
        'name': 'twoanchor_l10_ypair_neg2_plus_l11_maxbest',
        'class': 'new_anchor10_y',
        'rationale': 'Milder anchor10 y-pair test: halve the saved neg4 y dwell while keeping the late11 maxbest motif, to see whether dKg_zz / Ka2_y gains survive with less mean damage.',
        'insertions': {
            10: l10_ypair_neg(2.0, 'l10_ypair_neg2'),
            11: l11_maxbest(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg4_then_ypair_neg2_plus_l11_maxbest',
        'class': 'new_anchor10_hybrid',
        'rationale': 'Residual-focused hybrid at anchor10: first keep the useful neg4 z-pair scaffold, then add a mild neg2 y-pair before the late11 maxbest block. Intended to attack Ka2_y / dKa_yy / dKg_zz together while keeping Ka2_z protected.',
        'insertions': {
            10: l10_zpair_neg(4.0, 'l10_zpair_neg4') + l10_ypair_neg(2.0, 'l10_ypair_neg2'),
            11: l11_maxbest(),
        },
    },
    {
        'name': 'twoanchor_l10_ypair_neg2_then_zpair_neg4_plus_l11_maxbest',
        'class': 'new_anchor10_hybrid',
        'rationale': 'Same anchor10 hybrid ingredients as the previous candidate, but reverse the local order to see whether ending anchor10 with the z-pair better protects the already-good xz-family channels.',
        'insertions': {
            10: l10_ypair_neg(2.0, 'l10_ypair_neg2') + l10_zpair_neg(4.0, 'l10_zpair_neg4'),
            11: l11_maxbest(),
        },
    },
]


def is_new_candidate(name: str) -> bool:
    return name not in REF_NAMES


def load_default18_payloads(noise_scale: float) -> dict[str, Any]:
    expected_cfg = expected_noise_config(noise_scale)
    payloads = {
        'default18_markov': _load_json(DEFAULT18_RESULT),
        'default18_kf': _load_json(DEFAULT18_KF_RESULT),
    }
    for payload in payloads.values():
        if not _noise_matches(payload, expected_cfg):
            raise ValueError('default18 noise configuration mismatch')
    return payloads


def row_summary(row: dict[str, Any]) -> str:
    m = row['metrics']['overall']
    return f"{m['mean_pct_error']:.3f} / {m['median_pct_error']:.3f} / {m['max_pct_error']:.3f}"


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
    names: list[str] = []
    if not new_rows:
        return names

    best_new = min(new_rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    names.append(best_new['candidate_name'])

    best_max = min(new_rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    if best_max['candidate_name'] not in names:
        names.append(best_max['candidate_name'])

    hybrid_rows = [row for row in new_rows if 'hybrid' in row['candidate_class']]
    if hybrid_rows:
        best_hybrid = min(hybrid_rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
        if best_hybrid['candidate_name'] not in names:
            names.append(best_hybrid['candidate_name'])
        alt_hybrid = min(hybrid_rows, key=lambda x: (x['metrics']['key_param_errors']['dKg_zz'], x['metrics']['overall']['mean_pct_error']))
        if alt_hybrid['candidate_name'] not in names:
            names.append(alt_hybrid['candidate_name'])

    return names


def render_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    refs = payload['references']
    best_new = payload['best_new_frontier_point']
    best_hybrid = payload['best_targeted_hybrid']

    lines.append('# Chapter-3 faithful12 two-anchor residual push')
    lines.append('')
    lines.append('## 1. Scope of this pass')
    lines.append('')
    lines.append('- This pass stayed inside the **faithful12-base two-anchor distributed late-block branch**.')
    lines.append('- Hard constraints stayed fixed throughout:')
    lines.append('  - faithful12 scaffold preserved')
    lines.append('  - exact real dual-axis continuity closure at every insertion anchor')
    lines.append('  - same shared low-noise benchmark: `noise_scale=0.08`, `seed=42`, same truth family')
    lines.append('  - total time kept inside **1200–1800 s**')
    lines.append('- The white-noise-only result was used as a design filter: keep the search structural, not drift-overfit. So this batch only refined the already-validated late10/late11 motifs instead of opening arbitrary new families.')
    lines.append('')
    lines.append('## 2. Design directions tested')
    lines.append('')
    lines.append('1. **Direct bridge variants** between the current best-mean and max-oriented endpoints (`neg6/neg5 @ anchor10` with `maxbest/bestmean @ anchor11`).')
    lines.append('2. **Late11 local refinements under the current neg6 anchor10 scaffold**, especially the resume-side x-buffer relocation (`y10x0back2`).')
    lines.append('3. **Residual-focused anchor10 hybrids** that combine a mild z-pair and a mild y-pair before the late11 maxbest block, specifically to attack `Ka2_y / dKa_yy / dKg_zz` together while watching `Ka2_z`.')
    lines.append('')
    lines.append('## 3. Fixed references')
    lines.append('')
    lines.append(f"- faithful12: **{refs['faithful12']['markov42']['overall']['mean_pct_error']:.3f} / {refs['faithful12']['markov42']['overall']['median_pct_error']:.3f} / {refs['faithful12']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- current two-anchor best-mean: **{refs['current_twoanchor_bestmean']['markov42']['overall']['mean_pct_error']:.3f} / {refs['current_twoanchor_bestmean']['markov42']['overall']['median_pct_error']:.3f} / {refs['current_twoanchor_bestmean']['markov42']['overall']['max_pct_error']:.3f}** (`twoanchor_l10_zpair_neg6_plus_l11_bestmean`)")
    lines.append(f"- current two-anchor compromise: **{refs['current_twoanchor_compromise']['markov42']['overall']['mean_pct_error']:.3f} / {refs['current_twoanchor_compromise']['markov42']['overall']['median_pct_error']:.3f} / {refs['current_twoanchor_compromise']['markov42']['overall']['max_pct_error']:.3f}** (`twoanchor_l10_zpair_neg4_plus_l11_maxbest`)")
    lines.append(f"- current two-anchor best-max: **{refs['current_twoanchor_bestmax']['markov42']['overall']['mean_pct_error']:.3f} / {refs['current_twoanchor_bestmax']['markov42']['overall']['median_pct_error']:.3f} / {refs['current_twoanchor_bestmax']['markov42']['overall']['max_pct_error']:.3f}** (`twoanchor_l10_ypair_neg4_plus_l11_maxbest`)")
    lines.append(f"- old best legal non-faithful-base: **{refs['old_best_legal']['markov42']['overall']['mean_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['median_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- default18: **{refs['default18']['markov42']['overall']['mean_pct_error']:.3f} / {refs['default18']['markov42']['overall']['median_pct_error']:.3f} / {refs['default18']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('## 4. Markov42 results')
    lines.append('')
    lines.append('| rank | candidate | class | new? | frontier? | total_s | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Ka2_z | Δmean vs cur-bestmean | Δmax vs cur-bestmean | Δmean vs cur-comp | Δmax vs cur-comp |')
    lines.append('|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(payload['markov42_rows'], start=1):
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['candidate_class']} | {'yes' if row['is_new_candidate'] else 'ref'} | {'yes' if row['on_pareto_frontier'] else 'no'} | {row['total_time_s']:.0f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {row['delta_vs_current_bestmean']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_current_bestmean']['max_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_current_compromise']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_current_compromise']['max_pct_error']['improvement_pct_points']:+.3f} |"
        )
    lines.append('')
    lines.append('## 5. Best new frontier point')
    lines.append('')
    lines.append(f"- selected new frontier point: **{best_new['candidate_name']}** → **{best_new['markov42']['overall']['mean_pct_error']:.3f} / {best_new['markov42']['overall']['median_pct_error']:.3f} / {best_new['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- vs current best-mean: Δmean **{best_new['delta_vs_current_bestmean']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_new['delta_vs_current_bestmean']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- vs current compromise: Δmean **{best_new['delta_vs_current_compromise']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_new['delta_vs_current_compromise']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- targeted residuals: dKa_yy **{best_new['markov42']['key_param_errors']['dKa_yy']:.3f}**, dKg_zz **{best_new['markov42']['key_param_errors']['dKg_zz']:.3f}**, Ka2_y **{best_new['markov42']['key_param_errors']['Ka2_y']:.3f}**, Ka2_z **{best_new['markov42']['key_param_errors']['Ka2_z']:.3f}**")
    lines.append('')
    lines.append('## 6. Best residual-targeted hybrid')
    lines.append('')
    lines.append(f"- selected hybrid: **{best_hybrid['candidate_name']}** → **{best_hybrid['markov42']['overall']['mean_pct_error']:.3f} / {best_hybrid['markov42']['overall']['median_pct_error']:.3f} / {best_hybrid['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- vs current best-mean: Δmean **{best_hybrid['delta_vs_current_bestmean']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_hybrid['delta_vs_current_bestmean']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- vs current best-max: Δmean **{best_hybrid['delta_vs_current_bestmax']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_hybrid['delta_vs_current_bestmax']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- targeted residuals: dKa_yy **{best_hybrid['markov42']['key_param_errors']['dKa_yy']:.3f}**, dKg_zz **{best_hybrid['markov42']['key_param_errors']['dKg_zz']:.3f}**, Ka2_y **{best_hybrid['markov42']['key_param_errors']['Ka2_y']:.3f}**, Ka2_z **{best_hybrid['markov42']['key_param_errors']['Ka2_z']:.3f}**")
    lines.append('')
    lines.append('## 7. Continuity proof for the best new frontier point')
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
    lines.append('## 8. Exact legal motor / timing table for the best new frontier point')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for seq_idx, (row, action, face) in enumerate(zip(best_new['all_rows'], best_new['all_actions'], best_new['all_faces']), start=1):
        lines.append(
            f"| {seq_idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 9. KF36 rechecks for best competitive candidates')
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
    lines.append('## 10. Reference comparison')
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
    lines.append('## 11. Bottom line')
    lines.append('')
    lines.append(f"- best new frontier point vs current best-mean: **Δmean {best_new['delta_vs_current_bestmean']['mean_pct_error']['improvement_pct_points']:+.3f}, Δmax {best_new['delta_vs_current_bestmean']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- best residual-hybrid vs current best-mean: **Δmean {best_hybrid['delta_vs_current_bestmean']['mean_pct_error']['improvement_pct_points']:+.3f}, Δmax {best_hybrid['delta_vs_current_bestmean']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- verdict: **{payload['bottom_line']['verdict']}**")
    lines.append(f"- scientific conclusion: **{payload['scientific_conclusion']}**")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_module('search_ch3_12pos_twoanchor_residual_push_src', str(SOURCE_FILE))

    refs = load_reference_payloads(args.noise_scale)
    late11_refs = load_late11_refs(args.noise_scale)
    default_refs = load_default18_payloads(args.noise_scale)

    faithful = build_candidate(mod, ())
    base_rows = faithful.rows
    base_actions = faithful.action_sequence

    candidates = [build_closedloop_candidate(mod, spec, base_rows, base_actions) for spec in CANDIDATE_SPECS]
    candidate_by_name = {cand.name: cand for cand in candidates}
    spec_by_name = {spec['name']: spec for spec in CANDIDATE_SPECS}

    payload_by_name: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
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

    current_bestmean_payload = payload_by_name['twoanchor_l10_zpair_neg6_plus_l11_bestmean']
    current_compromise_payload = payload_by_name['twoanchor_l10_zpair_neg4_plus_l11_maxbest']
    current_bestmax_payload = payload_by_name['twoanchor_l10_ypair_neg4_plus_l11_maxbest']

    for row in rows:
        payload = payload_by_name[row['candidate_name']]
        row['delta_vs_faithful'] = delta_vs_ref(refs['faithful_markov'], payload)
        row['delta_vs_old_best'] = delta_vs_ref(refs['oldbest_markov'], payload)
        row['delta_vs_default18'] = delta_vs_ref(default_refs['default18_markov'], payload)
        row['delta_vs_current_bestmean'] = delta_vs_ref(current_bestmean_payload, payload)
        row['delta_vs_current_compromise'] = delta_vs_ref(current_compromise_payload, payload)
        row['delta_vs_current_bestmax'] = delta_vs_ref(current_bestmax_payload, payload)

    rows.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    frontier = pareto_frontier(rows)
    for row in rows:
        row['on_pareto_frontier'] = row['candidate_name'] in frontier

    new_rows = [row for row in rows if row['is_new_candidate']]
    new_frontier_rows = [row for row in new_rows if row['on_pareto_frontier']]
    best_new_row = min(new_frontier_rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    best_new_cand = candidate_by_name[best_new_row['candidate_name']]
    best_new_payload = payload_by_name[best_new_cand.name]

    hybrid_rows = [row for row in new_rows if row['candidate_class'] == 'new_anchor10_hybrid']
    best_hybrid_row = min(hybrid_rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    best_hybrid_cand = candidate_by_name[best_hybrid_row['candidate_name']]
    best_hybrid_payload = payload_by_name[best_hybrid_cand.name]

    kf36_rows = []
    for name in select_kf_rechecks(new_rows):
        cand = candidate_by_name[name]
        kf_payload, kf_status, kf_path = run_candidate_payload(mod, cand, 'kf36_noisy', args.noise_scale, args.force_rerun)
        note = 'competitive new candidate'
        if name == best_new_cand.name:
            note = 'best new frontier point'
        elif name == best_hybrid_cand.name:
            note = 'best max-balanced hybrid'
        elif 'ypair_neg2_then_zpair_neg4' in name:
            note = 'best dKg_zz-oriented hybrid'
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
        'delta_vs_faithful': delta_vs_ref(refs['faithful_markov'], best_new_payload),
        'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], best_new_payload),
        'delta_vs_default18': delta_vs_ref(default_refs['default18_markov'], best_new_payload),
        'delta_vs_current_bestmean': delta_vs_ref(current_bestmean_payload, best_new_payload),
        'delta_vs_current_compromise': delta_vs_ref(current_compromise_payload, best_new_payload),
        'delta_vs_current_bestmax': delta_vs_ref(current_bestmax_payload, best_new_payload),
    }

    best_hybrid_summary = {
        'candidate_name': best_hybrid_cand.name,
        'candidate_class': spec_by_name[best_hybrid_cand.name]['class'],
        'rationale': spec_by_name[best_hybrid_cand.name]['rationale'],
        'total_time_s': best_hybrid_cand.total_time_s,
        'markov42': compact_result(best_hybrid_payload),
        'markov42_run_json': best_hybrid_row['run_json'],
        'delta_vs_faithful': delta_vs_ref(refs['faithful_markov'], best_hybrid_payload),
        'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], best_hybrid_payload),
        'delta_vs_default18': delta_vs_ref(default_refs['default18_markov'], best_hybrid_payload),
        'delta_vs_current_bestmean': delta_vs_ref(current_bestmean_payload, best_hybrid_payload),
        'delta_vs_current_compromise': delta_vs_ref(current_compromise_payload, best_hybrid_payload),
        'delta_vs_current_bestmax': delta_vs_ref(current_bestmax_payload, best_hybrid_payload),
    }

    comparison_rows = [
        {
            'label': 'faithful12',
            'note': 'base faithful12 scaffold',
            'markov42': compact_result(refs['faithful_markov']),
            'kf36': compact_result(refs['faithful_kf']),
        },
        {
            'label': 'current two-anchor best-mean',
            'note': 'twoanchor_l10_zpair_neg6_plus_l11_bestmean',
            'markov42': compact_result(current_bestmean_payload),
            'kf36': compact_result(_load_json(RESULTS_DIR / 'KF36_ch3closedloop_twoanchor_l10_zpair_neg6_plus_l11_bestmean_shared_noise0p08_param_errors.json')),
        },
        {
            'label': 'best new frontier point',
            'note': best_new_cand.name,
            'markov42': best_new_summary['markov42'],
            'kf36': next(row['kf36'] for row in kf36_rows if row['candidate_name'] == best_new_cand.name),
        },
        {
            'label': 'best residual-targeted hybrid',
            'note': best_hybrid_cand.name,
            'markov42': best_hybrid_summary['markov42'],
            'kf36': next(row['kf36'] for row in kf36_rows if row['candidate_name'] == best_hybrid_cand.name),
        },
        {
            'label': 'current two-anchor compromise',
            'note': 'twoanchor_l10_zpair_neg4_plus_l11_maxbest',
            'markov42': compact_result(current_compromise_payload),
            'kf36': compact_result(_load_json(RESULTS_DIR / 'KF36_ch3closedloop_twoanchor_l10_zpair_neg4_plus_l11_maxbest_shared_noise0p08_param_errors.json')),
        },
        {
            'label': 'current two-anchor best-max',
            'note': 'twoanchor_l10_ypair_neg4_plus_l11_maxbest',
            'markov42': compact_result(current_bestmax_payload),
            'kf36': compact_result(_load_json(RESULTS_DIR / 'KF36_ch3closedloop_twoanchor_l10_ypair_neg4_plus_l11_maxbest_shared_noise0p08_param_errors.json')),
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
            'markov42': compact_result(default_refs['default18_markov']),
            'kf36': compact_result(default_refs['default18_kf']),
        },
        {
            'label': 'late11 overall-best',
            'note': 'late11_yneg_xpair_outerhold_then_zquad_y10_x2',
            'markov42': compact_result(late11_refs['current_best_markov']),
            'kf36': compact_result(late11_refs['current_best_kf']),
        },
    ]

    best_new_max_gain = best_new_summary['delta_vs_current_bestmean']['max_pct_error']['improvement_pct_points']
    best_new_mean_loss = -best_new_summary['delta_vs_current_bestmean']['mean_pct_error']['improvement_pct_points']
    hybrid_max_gain = best_hybrid_summary['delta_vs_current_bestmean']['max_pct_error']['improvement_pct_points']
    hybrid_mean_loss = -best_hybrid_summary['delta_vs_current_bestmean']['mean_pct_error']['improvement_pct_points']

    if best_new_max_gain >= 0.02:
        verdict = (
            f"small but real interior-frontier progress: {best_new_cand.name} gives back only {best_new_mean_loss:.3f} mean-points versus the current best-mean path while shaving {best_new_max_gain:.3f} max-points. "
            f"However the Ka2_y ceiling is still far above the current best-max endpoint ({compact_result(current_bestmax_payload)['overall']['max_pct_error']:.3f}) and nowhere near default18 ({compact_result(default_refs['default18_markov'])['overall']['max_pct_error']:.3f})."
        )
    else:
        verdict = (
            'no material unified improvement: the best new candidate stayed inside the existing mean/max trade band without moving the Ka2_y ceiling enough to count as a meaningful frontier advance.'
        )

    scientific_conclusion = (
        f"The residual-focused two-anchor push did uncover one extra non-dominated interior point ({best_new_cand.name}) and one genuinely informative anchor10 hybrid ({best_hybrid_cand.name}). "
        f"But the interior point only trades about {best_new_mean_loss:.3f} mean-points for {best_new_max_gain:.3f} max-points, while the hybrid cuts dKa_yy / dKg_zz / Ka2_y much more aggressively only by paying about {hybrid_mean_loss:.3f} mean-points. "
        'So this local family can still be bent, but not enough to produce a clear new unified winner: Ka2_y remains the stubborn ceiling, and the current saved endpoints still define the main branch decision.'
    )

    out_json = RESULTS_DIR / f'ch3_twoanchor_residual_push_{make_suffix(args.noise_scale)}.json'
    out_md = REPORTS_DIR / f'psins_ch3_twoanchor_residual_push_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_twoanchor_residual_push',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'hard_constraints': {
            'base_skeleton': 'faithful chapter-3 12-position original sequence',
            'continuity_rule': 'exact same mechanism state before resume',
            'time_budget_s': [1200.0, 1800.0],
            'seed': 42,
            'truth_family': 'shared low-noise benchmark',
            'search_style': 'two-anchor theory-guided local refinement only',
        },
        'references': {
            'faithful12': {
                'candidate_name': faithful.name,
                'markov42': compact_result(refs['faithful_markov']),
                'kf36': compact_result(refs['faithful_kf']),
            },
            'current_twoanchor_bestmean': {
                'candidate_name': 'twoanchor_l10_zpair_neg6_plus_l11_bestmean',
                'markov42': compact_result(current_bestmean_payload),
                'kf36': compact_result(_load_json(RESULTS_DIR / 'KF36_ch3closedloop_twoanchor_l10_zpair_neg6_plus_l11_bestmean_shared_noise0p08_param_errors.json')),
            },
            'current_twoanchor_compromise': {
                'candidate_name': 'twoanchor_l10_zpair_neg4_plus_l11_maxbest',
                'markov42': compact_result(current_compromise_payload),
                'kf36': compact_result(_load_json(RESULTS_DIR / 'KF36_ch3closedloop_twoanchor_l10_zpair_neg4_plus_l11_maxbest_shared_noise0p08_param_errors.json')),
            },
            'current_twoanchor_bestmax': {
                'candidate_name': 'twoanchor_l10_ypair_neg4_plus_l11_maxbest',
                'markov42': compact_result(current_bestmax_payload),
                'kf36': compact_result(_load_json(RESULTS_DIR / 'KF36_ch3closedloop_twoanchor_l10_ypair_neg4_plus_l11_maxbest_shared_noise0p08_param_errors.json')),
            },
            'old_best_legal': {
                'candidate_name': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
                'markov42': compact_result(refs['oldbest_markov']),
                'kf36': compact_result(refs['oldbest_kf']),
            },
            'default18': {
                'candidate_name': 'default18',
                'markov42': compact_result(default_refs['default18_markov']),
                'kf36': compact_result(default_refs['default18_kf']),
            },
            'late11_overall_best': {
                'candidate_name': 'late11_yneg_xpair_outerhold_then_zquad_y10_x2',
                'markov42': compact_result(late11_refs['current_best_markov']),
                'kf36': compact_result(late11_refs['current_best_kf']),
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
        'best_new_frontier_point': best_new_summary,
        'best_targeted_hybrid': best_hybrid_summary,
        'kf36_rows': kf36_rows,
        'comparison_rows': comparison_rows,
        'bottom_line': {
            'best_new_frontier_point_mean_loss_vs_current_bestmean': best_new_mean_loss,
            'best_new_frontier_point_max_gain_vs_current_bestmean': best_new_max_gain,
            'best_hybrid_mean_loss_vs_current_bestmean': hybrid_mean_loss,
            'best_hybrid_max_gain_vs_current_bestmean': hybrid_max_gain,
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
    print('BEST_NEW_FRONTIER', best_new_cand.name, best_new_summary['markov42']['overall'], flush=True)
    print('BEST_TARGETED_HYBRID', best_hybrid_cand.name, best_hybrid_summary['markov42']['overall'], flush=True)
    print('BOTTOM_LINE', verdict, flush=True)
    print('SCIENTIFIC_CONCLUSION', scientific_conclusion, flush=True)


if __name__ == '__main__':
    main()
