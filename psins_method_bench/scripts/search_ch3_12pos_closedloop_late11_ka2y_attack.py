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

OLD_BEST_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
OLD_BEST_KF_RESULT = RESULTS_DIR / 'KF36_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
CURRENT_BEST_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_late11_yneg_xpair_outerhold_then_zpair_pos_med_shared_noise0p08_param_errors.json'
CURRENT_BEST_KF_RESULT = RESULTS_DIR / 'KF36_ch3closedloop_late11_yneg_xpair_outerhold_then_zpair_pos_med_shared_noise0p08_param_errors.json'
DEFAULT18_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json'
DEFAULT18_KF_RESULT = RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json'


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


def zpair_pos_med(prefix: str = 'z11_pos_med') -> list[StepSpec]:
    return [
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=10.0, segment_role='motif_out', label=f'{prefix}_out'),
        StepSpec(kind='outer', angle_deg=-90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=10.0, segment_role='motif_return', label=f'{prefix}_return'),
    ]


def zbipolar(pos_y_s: float, zero_s: float, neg_y_s: float | None = None, prefix: str = 'zbipolar') -> list[StepSpec]:
    if neg_y_s is None:
        neg_y_s = pos_y_s
    return [
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=pos_y_s, segment_role='motif_y_pos', label=f'{prefix}_pos_out'),
        StepSpec(kind='outer', angle_deg=-90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=zero_s, segment_role='motif_zero_a', label=f'{prefix}_pos_return'),
        StepSpec(kind='outer', angle_deg=-90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=neg_y_s, segment_role='motif_y_neg', label=f'{prefix}_neg_out'),
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=zero_s, segment_role='motif_zero_b', label=f'{prefix}_neg_return'),
    ]


def zquad(y_hold_s: float, x_hold_s: float, prefix: str = 'zquad') -> list[StepSpec]:
    return [
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=y_hold_s, segment_role='motif_y_pos', label=f'{prefix}_q1'),
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=x_hold_s, segment_role='motif_zero_a', label=f'{prefix}_q2'),
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=y_hold_s, segment_role='motif_y_neg', label=f'{prefix}_q3'),
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=x_hold_s, segment_role='motif_zero_b', label=f'{prefix}_q4'),
    ]


CANDIDATE_SPECS = [
    {
        'name': 'late11_yneg_xpair_outerhold_then_zpair_pos_med',
        'rationale': 'Current faithful12-base incumbent from the late11 fine-grid. Kept as the direct in-branch baseline for the Ka2_y attack.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zpair_pos_med('z11_pos_med')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zbipolar_y10_x0',
        'rationale': 'Explicit +Y / 0 / -Y / 0 late z-family exposure using two opposite closed z-pairs, with all dwell placed on ±Y and none on the intervening zero states.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zbipolar(10.0, 0.0, 10.0, 'zbipolar_y10x0')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zbipolar_y10_x5',
        'rationale': 'Same bipolar ±Y pattern, but restore a small 5 s zero-state dwell between the positive and negative Y exposures to protect already-good channels.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zbipolar(10.0, 5.0, 10.0, 'zbipolar_y10x5')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zbipolar_y15_x0',
        'rationale': 'Heavier ±Y hold with zero-state dwell removed. Tests whether stronger sign-separated Y exposure lowers the Ka2_y ceiling further, even at some risk to the mean.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zbipolar(15.0, 0.0, 15.0, 'zbipolar_y15x0')},
    },
    {
        'name': 'late11_yneg_xpair_d8_then_zbipolar_y10_x0',
        'rationale': 'Mildly shorten the xpair outer holds before the zero-dwell bipolar z block. Tests whether more of the late budget should be shifted from xpair dwell into ±Y exposure.',
        'insertions': {11: xpair_outerhold(8.0, 'yneg_xpair_d8') + zbipolar(10.0, 0.0, 10.0, 'zbipolar_y10x0')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zbipolar_p10_x4_n10_x4',
        'rationale': 'Small-grid refinement around the successful bipolar family: reduce zero-state dwell from 5 s to 4 s.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zbipolar(10.0, 4.0, 10.0, 'zbipolar_p10x4n10x4')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zbipolar_p10_x6_n10_x6',
        'rationale': 'Small-grid refinement around the successful bipolar family: increase zero-state dwell from 5 s to 6 s.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zbipolar(10.0, 6.0, 10.0, 'zbipolar_p10x6n10x6')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zbipolar_p10_x5_n12_x5',
        'rationale': 'Asymmetric ±Y refinement: slightly heavier negative-Y dwell to compensate the base family’s existing positive-Y bias.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zbipolar(10.0, 5.0, 12.0, 'zbipolar_p10x5n12x5')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zbipolar_p8_x5_n12_x5',
        'rationale': 'Asymmetric ±Y refinement with a slightly lighter positive-Y hold and a heavier negative-Y hold.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zbipolar(8.0, 5.0, 12.0, 'zbipolar_p8x5n12x5')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zquad_y10_x0',
        'rationale': 'Full four-quarter-turn late z-cycle after the xpair, with hold on ±Y only and no added dwell on the intermediate ±X states. This creates a continuity-safe +Y / -X / -Y / +X sweep while preserving z-family excitation.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zquad(10.0, 0.0, 'zquad_y10x0')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zquad_y8_x0',
        'rationale': 'Same four-quarter-turn z-cycle, but slightly lighter ±Y dwell to see whether the ceiling falls further when the cycle is tightened.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zquad(8.0, 0.0, 'zquad_y8x0')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zquad_y12_x0',
        'rationale': 'Same four-quarter-turn z-cycle, but slightly heavier ±Y dwell to probe a nearby mean/max tradeoff point.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zquad(12.0, 0.0, 'zquad_y12x0')},
    },
    {
        'name': 'late11_yneg_xpair_outerhold_then_zquad_y10_x2',
        'rationale': 'Four-quarter-turn z-cycle with a small 2 s hold on the intermediate ±X states. This tests whether a little zero-state buffering stabilizes the full cycle while keeping the new ±Y structure.',
        'insertions': {11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zquad(10.0, 2.0, 'zquad_y10x2')},
    },
]


def load_extra_reference_payloads(noise_scale: float) -> dict[str, dict[str, Any]]:
    expected_cfg = expected_noise_config(noise_scale)
    payloads = {
        'current_best_markov': _load_json(CURRENT_BEST_RESULT),
        'current_best_kf': _load_json(CURRENT_BEST_KF_RESULT),
        'default18_markov': _load_json(DEFAULT18_RESULT),
        'default18_kf': _load_json(DEFAULT18_KF_RESULT),
    }
    for payload in payloads.values():
        if not _noise_matches(payload, expected_cfg):
            raise ValueError('Extra reference noise configuration mismatch')
    return payloads


def pick_frontier_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = [
        'late11_yneg_xpair_outerhold_then_zquad_y10_x2',
        'late11_yneg_xpair_outerhold_then_zquad_y8_x0',
        'late11_yneg_xpair_outerhold_then_zquad_y12_x0',
        'late11_yneg_xpair_outerhold_then_zquad_y10_x0',
        'late11_yneg_xpair_outerhold_then_zbipolar_y10_x5',
        'late11_yneg_xpair_outerhold_then_zbipolar_p10_x6_n10_x6',
        'late11_yneg_xpair_outerhold_then_zpair_pos_med',
    ]
    lookup = {row['candidate_name']: row for row in rows}
    return [lookup[name] for name in wanted if name in lookup]


def render_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    refs = payload['references']
    best = payload['best_candidate']
    best_max = payload['best_max_candidate']

    lines.append('# Chapter-3 late11 Ka2_y ceiling attack')
    lines.append('')
    lines.append('## 1. Scope and hard rule')
    lines.append('')
    lines.append('- Base scaffold stayed the **faithful chapter-3 12-position sequence**; no base-node sign edits were allowed.')
    lines.append('- All additions were **continuity-safe closed loops at late11** and had to return to the exact same real dual-axis mechanism state before node 12 resumed.')
    lines.append('- Same benchmark condition throughout: **noise_scale=0.08, seed=42, shared truth family**.')
    lines.append('- Search was kept narrow and theory-guided: only late11 outerhold-xpair descendants plus local late z-family loop redesigns were tested.')
    lines.append('')
    lines.append('## 2. Design intent of this branch')
    lines.append('')
    lines.append('- Start from the best faithful12-base candidate: **late11_yneg_xpair_outerhold_then_zpair_pos_med**.')
    lines.append('- Target the remaining **Ka2_y / max ceiling** without giving back the late11 gains already won on the mean.')
    lines.append('- Two narrow motif families were tested:')
    lines.append('  1. **zbipolar**: explicit `+Y / 0 / -Y / 0` exposure using two opposite closed z-pairs.')
    lines.append('  2. **zquad**: a full four-quarter-turn late z-cycle, giving `+Y / -X / -Y / +X` exposure while staying on the same outer-axis family and closing exactly at late11.')
    lines.append('')
    lines.append('## 3. Fixed references')
    lines.append('')
    lines.append(f"- faithful12 baseline: **{refs['faithful12']['markov42']['overall']['mean_pct_error']:.3f} / {refs['faithful12']['markov42']['overall']['median_pct_error']:.3f} / {refs['faithful12']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- previous faithful12-base best: **{refs['current_faithful_best']['markov42']['overall']['mean_pct_error']:.3f} / {refs['current_faithful_best']['markov42']['overall']['median_pct_error']:.3f} / {refs['current_faithful_best']['markov42']['overall']['max_pct_error']:.3f}** (`{refs['current_faithful_best']['candidate_name']}`)")
    lines.append(f"- old best legal non-faithful-base: **{refs['old_best_legal']['markov42']['overall']['mean_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['median_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- default18 reference: **{refs['default18']['markov42']['overall']['mean_pct_error']:.3f} / {refs['default18']['markov42']['overall']['median_pct_error']:.3f} / {refs['default18']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('## 4. Markov42 results (sorted by mean, then max)')
    lines.append('')
    lines.append('| rank | candidate | family | total_s | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Δmean vs prev faithful-best | Δmean vs old best | Δmax vs prev faithful-best |')
    lines.append('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(payload['markov42_rows'], start=1):
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {row['total_time_s']:.0f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {row['delta_vs_current_faithful_best']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_old_best']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_current_faithful_best']['max_pct_error']['improvement_pct_points']:+.3f} |"
        )
    lines.append('')
    lines.append('## 5. Frontier summary')
    lines.append('')
    lines.append(f"- **overall best by mean**: `{best['candidate_name']}` → **{best['markov42']['overall']['mean_pct_error']:.3f} / {best['markov42']['overall']['median_pct_error']:.3f} / {best['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"  - vs previous faithful-best: Δmean **{best['delta_vs_current_faithful_best']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_current_faithful_best']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"  - vs old best legal: Δmean **{best['delta_vs_old_best']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_old_best']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"  - key channels: dKa_yy **{best['markov42']['key_param_errors']['dKa_yy']:.3f}**, dKg_zz **{best['markov42']['key_param_errors']['dKg_zz']:.3f}**, Ka2_y **{best['markov42']['key_param_errors']['Ka2_y']:.3f}**")
    lines.append(f"- **best by max / Ka2_y ceiling**: `{best_max['candidate_name']}` → **{best_max['markov42']['overall']['mean_pct_error']:.3f} / {best_max['markov42']['overall']['median_pct_error']:.3f} / {best_max['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"  - vs previous faithful-best: Δmean **{best_max['delta_vs_current_faithful_best']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_max['delta_vs_current_faithful_best']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append('')
    lines.append('### 5.1 What actually worked')
    lines.append('')
    lines.append('- The **zbipolar split-pair** family did improve the incumbent a little, proving that explicit `+Y / 0 / -Y / 0` exposure was directionally valid.')
    lines.append('- But the real breakthrough came from the **zquad full late z-cycle**. It lowered not only `Ka2_y / max`, but also `dKg_zz` and `dKa_yy` materially, which is why the mean finally crossed below the old best legal result.')
    lines.append('- Among zquad variants, a **small intermediate ±X hold (x=2 s)** gave the best mean, while **no intermediate hold with lighter ±Y dwell (y=8 s, x=0)** gave the lowest max.')
    lines.append('')
    lines.append('## 6. Continuity proof for the overall-best candidate')
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
    lines.append('## 7. Exact legal motor / timing table for the overall-best candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for seq_idx, (row, action, face) in enumerate(zip(best['all_rows'], best['all_actions'], best['all_faces']), start=1):
        lines.append(
            f"| {seq_idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 8. KF36 rechecks for frontier candidates')
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
    lines.append('## 9. Reference comparison table')
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
    lines.append('## 10. Bottom line')
    lines.append('')
    lines.append(f"- previous faithful-best max / Ka2_y: **{refs['current_faithful_best']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- overall-best branch result max     : **{best['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- max-optimized sibling max          : **{best_max['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- default18 max reference            : **{refs['default18']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append(f"- verdict on the Ka2_y ceiling: **{payload['bottom_line']['ceiling_verdict']}**")
    lines.append(f"- scientific conclusion: **{payload['scientific_conclusion']}**")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_module('search_ch3_12pos_closedloop_late11_ka2y_attack_src', str(SOURCE_FILE))

    refs = load_reference_payloads(args.noise_scale)
    extra_refs = load_extra_reference_payloads(args.noise_scale)

    faithful = build_candidate(mod, ())
    base_rows = faithful.rows
    base_actions = faithful.action_sequence

    candidates = [build_closedloop_candidate(mod, spec, base_rows, base_actions) for spec in CANDIDATE_SPECS]
    candidate_by_name = {cand.name: cand for cand in candidates}

    rows = []
    payload_by_name: dict[str, dict[str, Any]] = {}
    for spec, cand in zip(CANDIDATE_SPECS, candidates):
        payload, status, path = run_candidate_payload(mod, cand, 'markov42_noisy', args.noise_scale, args.force_rerun)
        row = {
            'candidate_name': cand.name,
            'family': 'zquad' if 'zquad' in cand.name else ('zbipolar' if 'zbipolar' in cand.name else 'reference'),
            'rationale': spec['rationale'],
            'total_time_s': cand.total_time_s,
            'metrics': compact_result(payload),
            'continuity_checks': cand.continuity_checks,
            'run_json': str(path),
            'status': status,
            'delta_vs_faithful': delta_vs_ref(refs['faithful_markov'], payload),
            'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], payload),
            'delta_vs_current_faithful_best': delta_vs_ref(extra_refs['current_best_markov'], payload),
            'delta_vs_default18': delta_vs_ref(extra_refs['default18_markov'], payload),
        }
        rows.append(row)
        payload_by_name[cand.name] = payload

    rows.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    best_row = rows[0]
    best_candidate = candidate_by_name[best_row['candidate_name']]
    best_payload = payload_by_name[best_candidate.name]

    best_max_row = min(rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    best_max_candidate = candidate_by_name[best_max_row['candidate_name']]
    best_max_payload = payload_by_name[best_max_candidate.name]

    kf36_rows = []
    for row in pick_frontier_rows(rows):
        cand = candidate_by_name[row['candidate_name']]
        kf_payload, kf_status, kf_path = run_candidate_payload(mod, cand, 'kf36_noisy', args.noise_scale, args.force_rerun)
        note = 'frontier recheck'
        if cand.name == best_candidate.name:
            note = 'overall-best mean frontier'
        elif cand.name == best_max_candidate.name:
            note = 'best max / Ka2_y frontier'
        elif cand.name == 'late11_yneg_xpair_outerhold_then_zpair_pos_med':
            note = 'previous faithful12-base best reference'
        kf36_rows.append({
            'candidate_name': cand.name,
            'markov42': compact_result(payload_by_name[cand.name]),
            'kf36': compact_result(kf_payload),
            'kf36_status': kf_status,
            'kf36_run_json': str(kf_path),
            'note': note,
        })

    best_summary = {
        'candidate_name': best_candidate.name,
        'rationale': next(spec['rationale'] for spec in CANDIDATE_SPECS if spec['name'] == best_candidate.name),
        'total_time_s': best_candidate.total_time_s,
        'all_rows': best_candidate.all_rows,
        'all_actions': best_candidate.all_actions,
        'all_faces': best_candidate.all_faces,
        'continuity_checks': best_candidate.continuity_checks,
        'markov42': compact_result(best_payload),
        'markov42_run_json': best_row['run_json'],
        'delta_vs_faithful': delta_vs_ref(refs['faithful_markov'], best_payload),
        'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], best_payload),
        'delta_vs_current_faithful_best': delta_vs_ref(extra_refs['current_best_markov'], best_payload),
        'delta_vs_default18': delta_vs_ref(extra_refs['default18_markov'], best_payload),
    }

    best_max_summary = {
        'candidate_name': best_max_candidate.name,
        'rationale': next(spec['rationale'] for spec in CANDIDATE_SPECS if spec['name'] == best_max_candidate.name),
        'total_time_s': best_max_candidate.total_time_s,
        'markov42': compact_result(best_max_payload),
        'markov42_run_json': best_max_row['run_json'],
        'delta_vs_faithful': delta_vs_ref(refs['faithful_markov'], best_max_payload),
        'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], best_max_payload),
        'delta_vs_current_faithful_best': delta_vs_ref(extra_refs['current_best_markov'], best_max_payload),
        'delta_vs_default18': delta_vs_ref(extra_refs['default18_markov'], best_max_payload),
    }

    comparison_rows = [
        {
            'label': 'faithful12 baseline',
            'note': 'original faithful chapter-3 12-position path',
            'markov42': compact_result(refs['faithful_markov']),
            'kf36': compact_result(refs['faithful_kf']),
        },
        {
            'label': 'previous faithful12-base best',
            'note': 'late11_yneg_xpair_outerhold_then_zpair_pos_med',
            'markov42': compact_result(extra_refs['current_best_markov']),
            'kf36': compact_result(extra_refs['current_best_kf']),
        },
        {
            'label': 'new overall-best faithful12-base',
            'note': best_candidate.name,
            'markov42': best_summary['markov42'],
            'kf36': next(row['kf36'] for row in kf36_rows if row['candidate_name'] == best_candidate.name),
        },
        {
            'label': 'new max-opt faithful12-base sibling',
            'note': best_max_candidate.name,
            'markov42': best_max_summary['markov42'],
            'kf36': next(row['kf36'] for row in kf36_rows if row['candidate_name'] == best_max_candidate.name),
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

    overall_max_prev = compact_result(extra_refs['current_best_markov'])['overall']['max_pct_error']
    overall_max_best = best_summary['markov42']['overall']['max_pct_error']
    overall_max_bestmax = best_max_summary['markov42']['overall']['max_pct_error']
    default18_max = compact_result(extra_refs['default18_markov'])['overall']['max_pct_error']
    gap_prev = overall_max_prev - default18_max
    gap_best = overall_max_best - default18_max
    gap_bestmax = overall_max_bestmax - default18_max

    if overall_max_best < overall_max_prev - 0.05:
        ceiling_verdict = (
            f"yes locally, but not fundamentally: the branch lowered the overall-best Ka2_y/max from {overall_max_prev:.3f} to {overall_max_best:.3f}, "
            f"and the max-optimized sibling reached {overall_max_bestmax:.3f}; however the gap to default18 max remains enormous "
            f"({gap_bestmax:.3f} still open after only {gap_prev - gap_bestmax:.3f} points of closure)."
        )
    else:
        ceiling_verdict = 'no in a meaningful sense: the branch changed Ka2_y/max only marginally and the default18 gap remains essentially intact.'

    scientific_conclusion = (
        f"The focused late11 Ka2_y attack succeeded scientifically in two ways: (1) it found a new faithful12-base continuity-safe winner "
        f"({best_candidate.name}) at {best_summary['markov42']['overall']['mean_pct_error']:.3f} / {best_summary['markov42']['overall']['median_pct_error']:.3f} / {best_summary['markov42']['overall']['max_pct_error']:.3f}, "
        f"which beats the old best legal non-faithful-base mean by {best_summary['delta_vs_old_best']['mean_pct_error']['improvement_pct_points']:.3f} points; "
        f"and (2) it lowered the Ka2_y ceiling modestly, with the max-optimized sibling reaching {best_max_summary['markov42']['overall']['max_pct_error']:.3f}. "
        f"But the ceiling is not truly broken: Ka2_y still remains the max channel and is still far above the default18 regime."
    )

    out_json = RESULTS_DIR / f'ch3_12pos_closedloop_late11_ka2y_attack_{make_suffix(args.noise_scale)}.json'
    out_md = REPORTS_DIR / f'psins_ch3_12pos_closedloop_late11_ka2y_attack_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_12pos_closedloop_late11_ka2y_attack',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'hard_constraints': {
            'base_skeleton': 'faithful chapter-3 12-position original sequence',
            'locality_rule': 'late11 family only',
            'continuity_rule': 'exact same mechanism state before resume',
            'noise_scale': args.noise_scale,
            'seed': 42,
            'truth_family': 'shared low-noise benchmark',
            'time_budget_s': [1200.0, 1800.0],
        },
        'references': {
            'faithful12': {
                'candidate_name': faithful.name,
                'markov42': compact_result(refs['faithful_markov']),
                'markov42_run_json': str(FAITHFUL_RESULT),
                'kf36': compact_result(refs['faithful_kf']),
                'kf36_run_json': str(FAITHFUL_KF_RESULT),
            },
            'current_faithful_best': {
                'candidate_name': 'late11_yneg_xpair_outerhold_then_zpair_pos_med',
                'markov42': compact_result(extra_refs['current_best_markov']),
                'markov42_run_json': str(CURRENT_BEST_RESULT),
                'kf36': compact_result(extra_refs['current_best_kf']),
                'kf36_run_json': str(CURRENT_BEST_KF_RESULT),
            },
            'old_best_legal': {
                'candidate_name': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
                'markov42': compact_result(refs['oldbest_markov']),
                'markov42_run_json': str(OLD_BEST_RESULT),
                'kf36': compact_result(refs['oldbest_kf']),
                'kf36_run_json': str(OLD_BEST_KF_RESULT),
            },
            'default18': {
                'candidate_name': 'default18',
                'markov42': compact_result(extra_refs['default18_markov']),
                'markov42_run_json': str(DEFAULT18_RESULT),
                'kf36': compact_result(extra_refs['default18_kf']),
                'kf36_run_json': str(DEFAULT18_KF_RESULT),
            },
        },
        'candidate_specs': [
            {
                'name': spec['name'],
                'rationale': spec['rationale'],
                'insertions': sorted(spec['insertions'].keys()),
            }
            for spec in CANDIDATE_SPECS
        ],
        'markov42_rows': rows,
        'kf36_rows': kf36_rows,
        'best_candidate': best_summary,
        'best_max_candidate': best_max_summary,
        'comparison_rows': comparison_rows,
        'bottom_line': {
            'previous_faithful_best_max': overall_max_prev,
            'new_overall_best_max': overall_max_best,
            'new_max_optimized_max': overall_max_bestmax,
            'default18_max': default18_max,
            'gap_to_default18_before': gap_prev,
            'gap_to_default18_after_overall_best': gap_best,
            'gap_to_default18_after_max_opt': gap_bestmax,
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
    print('BEST_OVERALL', best_candidate.name, best_summary['markov42']['overall'], flush=True)
    print('BEST_MAX', best_max_candidate.name, best_max_summary['markov42']['overall'], flush=True)
    print('BOTTOM_LINE', ceiling_verdict, flush=True)
    print('SCIENTIFIC_CONCLUSION', scientific_conclusion, flush=True)


if __name__ == '__main__':
    main()
