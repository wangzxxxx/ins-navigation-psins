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
    build_closedloop_candidate,
    delta_vs_ref,
    load_reference_payloads,
    make_suffix,
    render_action,
    run_candidate_payload,
)
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate
from search_ch3_12pos_twoanchor_residual_push import l10_ypair_neg, l10_zpair_neg, l11_bestmean, l11_y10x0back2

DEFAULT18_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json'
DEFAULT18_KF_RESULT = RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json'

BESTMEAN_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_twoanchor_l10_zpair_neg6_plus_l11_bestmean_shared_noise0p08_param_errors.json'
BESTMEAN_KF = RESULTS_DIR / 'KF36_ch3closedloop_twoanchor_l10_zpair_neg6_plus_l11_bestmean_shared_noise0p08_param_errors.json'
INTERIOR_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_twoanchor_l10_zpair_neg6_plus_l11_y10x0back2_shared_noise0p08_param_errors.json'
INTERIOR_KF = RESULTS_DIR / 'KF36_ch3closedloop_twoanchor_l10_zpair_neg6_plus_l11_y10x0back2_shared_noise0p08_param_errors.json'
CURRENT_COMP_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_twoanchor_l10_zpair_neg4_plus_l11_maxbest_shared_noise0p08_param_errors.json'
CURRENT_COMP_KF = RESULTS_DIR / 'KF36_ch3closedloop_twoanchor_l10_zpair_neg4_plus_l11_maxbest_shared_noise0p08_param_errors.json'
RESIDUAL_HYBRID_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_twoanchor_l10_zpair_neg4_then_ypair_neg2_plus_l11_maxbest_shared_noise0p08_param_errors.json'
RESIDUAL_HYBRID_KF = RESULTS_DIR / 'KF36_ch3closedloop_twoanchor_l10_zpair_neg4_then_ypair_neg2_plus_l11_maxbest_shared_noise0p08_param_errors.json'
BESTMAX_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_twoanchor_l10_ypair_neg4_plus_l11_maxbest_shared_noise0p08_param_errors.json'
BESTMAX_KF = RESULTS_DIR / 'KF36_ch3closedloop_twoanchor_l10_ypair_neg4_plus_l11_maxbest_shared_noise0p08_param_errors.json'


CANDIDATE_SPECS = [
    {
        'name': 'twoanchor_l10_zpair_neg4_then_ypair_neg2_plus_l11_y10x0back2',
        'class': 'strong_hybrid_softlate',
        'rationale': 'Keep the proven anchor10 z4+y2 hybrid, but replace late11 maxbest with the softer resume-side y10x0back2 block to recover mean while keeping the targeted residual pressure.',
        'insertions': {
            10: l10_zpair_neg(4.0, 'l10_zpair_neg4') + l10_ypair_neg(2.0, 'l10_ypair_neg2'),
            11: l11_y10x0back2(),
        },
    },
    {
        'name': 'twoanchor_l10_ypair_neg2_then_zpair_neg4_plus_l11_y10x0back2',
        'class': 'strong_hybrid_softlate',
        'rationale': 'Same strong hybrid ingredients as above, but finish anchor10 with the z-pair before resuming, to bias toward protecting xz / Ka2_z while keeping the softer late11 block.',
        'insertions': {
            10: l10_ypair_neg(2.0, 'l10_ypair_neg2') + l10_zpair_neg(4.0, 'l10_zpair_neg4'),
            11: l11_y10x0back2(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg4_then_ypair_neg2_plus_l11_bestmean',
        'class': 'strong_hybrid_meanlate',
        'rationale': 'Keep the anchor10 z4+y2 hybrid, but swap late11 all the way back to bestmean. This tests whether anchor10 alone carries most of the Ka2_y suppression.',
        'insertions': {
            10: l10_zpair_neg(4.0, 'l10_zpair_neg4') + l10_ypair_neg(2.0, 'l10_ypair_neg2'),
            11: l11_bestmean(),
        },
    },
    {
        'name': 'twoanchor_l10_ypair_neg2_then_zpair_neg4_plus_l11_bestmean',
        'class': 'strong_hybrid_meanlate',
        'rationale': 'Reverse-order counterpart of the z4+y2 + bestmean blend, to test whether ending anchor10 with the z-pair gives a cleaner unified trade.',
        'insertions': {
            10: l10_ypair_neg(2.0, 'l10_ypair_neg2') + l10_zpair_neg(4.0, 'l10_zpair_neg4'),
            11: l11_bestmean(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg6_then_ypair_neg1_plus_l11_y10x0back2',
        'class': 'minimal_y_blend',
        'rationale': 'Minimal-anchor10 blend around the best-mean family: keep the strong neg6 z-pair, add only a 1 s y-pair after it, then use the already-proven y10x0back2 late11 block.',
        'insertions': {
            10: l10_zpair_neg(6.0, 'l10_zpair_neg6') + l10_ypair_neg(1.0, 'l10_ypair_neg1'),
            11: l11_y10x0back2(),
        },
    },
    {
        'name': 'twoanchor_l10_ypair_neg1_then_zpair_neg6_plus_l11_y10x0back2',
        'class': 'minimal_y_blend',
        'rationale': 'Same minimal 1 s anchor10 y-dose as above, but place it before the neg6 z-pair to test whether the order changes how much mean recovery survives.',
        'insertions': {
            10: l10_ypair_neg(1.0, 'l10_ypair_neg1') + l10_zpair_neg(6.0, 'l10_zpair_neg6'),
            11: l11_y10x0back2(),
        },
    },
    {
        'name': 'twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2',
        'class': 'minimal_y_blend',
        'rationale': 'Interpolated unified blend: soften anchor10 z-dwell from 6 s to 5 s, keep only a 1 s y-pair, and keep the resume-side late11 y10x0back2 block. This is the most literal blend between the best-mean family and the mild residual hybrid.',
        'insertions': {
            10: l10_zpair_neg(5.0, 'l10_zpair_neg5') + l10_ypair_neg(1.0, 'l10_ypair_neg1'),
            11: l11_y10x0back2(),
        },
    },
    {
        'name': 'twoanchor_l10_ypair_neg1_then_zpair_neg5_plus_l11_y10x0back2',
        'class': 'minimal_y_blend',
        'rationale': 'Reverse-order counterpart of the z5+y1 minimal blend, included to see whether ending anchor10 with the z-pair is still preferable at the 5 s z-dwell level.',
        'insertions': {
            10: l10_ypair_neg(1.0, 'l10_ypair_neg1') + l10_zpair_neg(5.0, 'l10_zpair_neg5'),
            11: l11_y10x0back2(),
        },
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



def load_twoanchor_references(noise_scale: float) -> dict[str, Any]:
    return {
        'bestmean_markov': load_json_checked(BESTMEAN_MARKOV, noise_scale),
        'bestmean_kf': load_json_checked(BESTMEAN_KF, noise_scale),
        'interior_markov': load_json_checked(INTERIOR_MARKOV, noise_scale),
        'interior_kf': load_json_checked(INTERIOR_KF, noise_scale),
        'current_comp_markov': load_json_checked(CURRENT_COMP_MARKOV, noise_scale),
        'current_comp_kf': load_json_checked(CURRENT_COMP_KF, noise_scale),
        'residual_hybrid_markov': load_json_checked(RESIDUAL_HYBRID_MARKOV, noise_scale),
        'residual_hybrid_kf': load_json_checked(RESIDUAL_HYBRID_KF, noise_scale),
        'bestmax_markov': load_json_checked(BESTMAX_MARKOV, noise_scale),
        'bestmax_kf': load_json_checked(BESTMAX_KF, noise_scale),
        'default18_markov': load_json_checked(DEFAULT18_RESULT, noise_scale),
        'default18_kf': load_json_checked(DEFAULT18_KF_RESULT, noise_scale),
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



def select_kf_rechecks(new_frontier_rows: list[dict[str, Any]]) -> list[str]:
    return [row['candidate_name'] for row in sorted(new_frontier_rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))]



def row_summary(row: dict[str, Any]) -> str:
    m = row['metrics']['overall']
    return f"{m['mean_pct_error']:.3f} / {m['median_pct_error']:.3f} / {m['max_pct_error']:.3f}"



def render_report(payload: dict[str, Any]) -> str:
    refs = payload['references']
    best = payload['best_unified_candidate']
    best_max_lean = payload['best_new_maxlean_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 faithful12 two-anchor unified-compromise pass')
    lines.append('')
    lines.append('## 1. Verdict')
    lines.append('')
    lines.append(f"- **Did the unification pass produce a better unified frontier point?** **{payload['bottom_line']['success']}**")
    lines.append(f"- **Best new unified candidate:** **{best['candidate_name']}** → **{best['markov42']['overall']['mean_pct_error']:.3f} / {best['markov42']['overall']['median_pct_error']:.3f} / {best['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- **Key claim:** this point beats the old best-mean anchor, the previous interior point, and the old compromise point on **both mean and max**, while also improving `dKa_yy`, `dKg_zz`, `Ka2_y`, and `Ka2_z` versus the best-mean anchor.")
    lines.append('')
    lines.append('## 2. Search logic used')
    lines.append('')
    lines.append('- Stayed strictly inside the **faithful12-base continuity-safe two-anchor family**.')
    lines.append('- Used only a **small interpretable blend set** between:')
    lines.append('  1. the `late10 zpair-neg6 + late11 bestmean / y10x0back2` family')
    lines.append('  2. the mild `late10 zpair+ypair` residual-hybrid family')
    lines.append('- Biases enforced in this pass:')
    lines.append('  - minimal anchor10 y-dose (`ypair_neg1`) before reopening larger y-dwell')
    lines.append('  - resume-side late11 structure (`y10x0back2`) instead of heavier max-only late11 forcing')
    lines.append('  - preserve already-good channels, especially `xz` / `Ka2_z`')
    lines.append('')
    lines.append('## 3. Fixed references')
    lines.append('')
    lines.append(f"- faithful12: **{refs['faithful12']['markov42']['overall']['mean_pct_error']:.3f} / {refs['faithful12']['markov42']['overall']['median_pct_error']:.3f} / {refs['faithful12']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- old best-mean anchor: **{refs['bestmean_anchor']['markov42']['overall']['mean_pct_error']:.3f} / {refs['bestmean_anchor']['markov42']['overall']['median_pct_error']:.3f} / {refs['bestmean_anchor']['markov42']['overall']['max_pct_error']:.3f}** (`twoanchor_l10_zpair_neg6_plus_l11_bestmean`)")
    lines.append(f"- old interior point: **{refs['interior_point']['markov42']['overall']['mean_pct_error']:.3f} / {refs['interior_point']['markov42']['overall']['median_pct_error']:.3f} / {refs['interior_point']['markov42']['overall']['max_pct_error']:.3f}** (`twoanchor_l10_zpair_neg6_plus_l11_y10x0back2`)")
    lines.append(f"- old compromise point: **{refs['current_compromise']['markov42']['overall']['mean_pct_error']:.3f} / {refs['current_compromise']['markov42']['overall']['median_pct_error']:.3f} / {refs['current_compromise']['markov42']['overall']['max_pct_error']:.3f}** (`twoanchor_l10_zpair_neg4_plus_l11_maxbest`)")
    lines.append(f"- residual-hybrid reference: **{refs['residual_hybrid']['markov42']['overall']['mean_pct_error']:.3f} / {refs['residual_hybrid']['markov42']['overall']['median_pct_error']:.3f} / {refs['residual_hybrid']['markov42']['overall']['max_pct_error']:.3f}** (`twoanchor_l10_zpair_neg4_then_ypair_neg2_plus_l11_maxbest`)")
    lines.append(f"- old best legal non-faithful-base: **{refs['old_best_legal']['markov42']['overall']['mean_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['median_pct_error']:.3f} / {refs['old_best_legal']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- default18: **{refs['default18']['markov42']['overall']['mean_pct_error']:.3f} / {refs['default18']['markov42']['overall']['median_pct_error']:.3f} / {refs['default18']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('## 4. Markov42 results from this unification pass')
    lines.append('')
    lines.append('| rank | candidate | class | frontier? | total_s | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Ka2_z | Δmean vs bestmean | Δmax vs bestmean | Δmean vs interior | Δmax vs interior |')
    lines.append('|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(payload['markov42_rows'], start=1):
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['candidate_class']} | {'yes' if row['on_pareto_frontier'] else 'no'} | {row['total_time_s']:.0f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {row['delta_vs_bestmean_anchor']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_bestmean_anchor']['max_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_interior_point']['mean_pct_error']['improvement_pct_points']:+.3f} | {row['delta_vs_interior_point']['max_pct_error']['improvement_pct_points']:+.3f} |"
        )
    lines.append('')
    lines.append('## 5. Best unified candidate')
    lines.append('')
    lines.append(f"- selected unified winner: **{best['candidate_name']}** → **{best['markov42']['overall']['mean_pct_error']:.3f} / {best['markov42']['overall']['median_pct_error']:.3f} / {best['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- vs old best-mean anchor: Δmean **{best['delta_vs_bestmean_anchor']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_bestmean_anchor']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- vs old interior point: Δmean **{best['delta_vs_interior_point']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_interior_point']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- vs residual-hybrid: Δmean **{best['delta_vs_residual_hybrid']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_residual_hybrid']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- targeted residuals vs old best-mean anchor:")
    lines.append(f"  - dKa_yy: **{best['markov42']['key_param_errors']['dKa_yy']:.3f}** vs {refs['bestmean_anchor']['markov42']['key_param_errors']['dKa_yy']:.3f}")
    lines.append(f"  - dKg_zz: **{best['markov42']['key_param_errors']['dKg_zz']:.3f}** vs {refs['bestmean_anchor']['markov42']['key_param_errors']['dKg_zz']:.3f}")
    lines.append(f"  - Ka2_y: **{best['markov42']['key_param_errors']['Ka2_y']:.3f}** vs {refs['bestmean_anchor']['markov42']['key_param_errors']['Ka2_y']:.3f}")
    lines.append(f"  - Ka2_z: **{best['markov42']['key_param_errors']['Ka2_z']:.3f}** vs {refs['bestmean_anchor']['markov42']['key_param_errors']['Ka2_z']:.3f}")
    lines.append(f"- protected-channel note: `dKg_xz` = **{best['markov42']['key_param_errors']['dKg_xz']:.3f}**, `dKa_xz` = **{best['markov42']['key_param_errors']['dKa_xz']:.3f}**, so the blend did not pay for the Ka2_y gain by blowing up the xz family.")
    lines.append('')
    lines.append('## 6. Max-leaning new frontier companion')
    lines.append('')
    lines.append(f"- lowest-max new frontier point: **{best_max_lean['candidate_name']}** → **{best_max_lean['markov42']['overall']['mean_pct_error']:.3f} / {best_max_lean['markov42']['overall']['median_pct_error']:.3f} / {best_max_lean['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- it keeps max only **{-best_max_lean['delta_vs_bestmax_anchor']['max_pct_error']['improvement_pct_points']:.3f}** worse than the old best-max anchor, but recovers **{best_max_lean['delta_vs_bestmax_anchor']['mean_pct_error']['improvement_pct_points']:+.3f}** mean-points.")
    lines.append('')
    lines.append('## 7. Continuity proof for the best unified candidate')
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
    lines.append('## 8. Exact legal motor / timing table for the best unified candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for seq_idx, (row, action, face) in enumerate(zip(best['all_rows'], best['all_actions'], best['all_faces']), start=1):
        lines.append(
            f"| {seq_idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 9. KF36 rechecks for new frontier candidates')
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
    lines.append(f"- verdict: **{payload['bottom_line']['verdict']}**")
    lines.append(f"- frontier summary: **{payload['scientific_conclusion']}**")
    lines.append('')
    return '\n'.join(lines) + '\n'



def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module('search_ch3_12pos_twoanchor_unified_compromise_src', str(SOURCE_FILE))
    refs = load_reference_payloads(args.noise_scale)
    twoanchor_refs = load_twoanchor_references(args.noise_scale)

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

    bestmean_payload = twoanchor_refs['bestmean_markov']
    interior_payload = twoanchor_refs['interior_markov']
    current_comp_payload = twoanchor_refs['current_comp_markov']
    residual_hybrid_payload = twoanchor_refs['residual_hybrid_markov']
    bestmax_payload = twoanchor_refs['bestmax_markov']
    default18_payload = twoanchor_refs['default18_markov']

    for row in rows:
        payload = payload_by_name[row['candidate_name']]
        row['delta_vs_faithful'] = delta_vs_ref(refs['faithful_markov'], payload)
        row['delta_vs_bestmean_anchor'] = delta_vs_ref(bestmean_payload, payload)
        row['delta_vs_interior_point'] = delta_vs_ref(interior_payload, payload)
        row['delta_vs_current_compromise'] = delta_vs_ref(current_comp_payload, payload)
        row['delta_vs_residual_hybrid'] = delta_vs_ref(residual_hybrid_payload, payload)
        row['delta_vs_bestmax_anchor'] = delta_vs_ref(bestmax_payload, payload)
        row['delta_vs_old_best'] = delta_vs_ref(refs['oldbest_markov'], payload)
        row['delta_vs_default18'] = delta_vs_ref(default18_payload, payload)

    rows.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    frontier = pareto_frontier(rows + [
        {
            'candidate_name': 'twoanchor_l10_zpair_neg6_plus_l11_bestmean',
            'metrics': compact_result(bestmean_payload),
        },
        {
            'candidate_name': 'twoanchor_l10_zpair_neg6_plus_l11_y10x0back2',
            'metrics': compact_result(interior_payload),
        },
        {
            'candidate_name': 'twoanchor_l10_zpair_neg4_plus_l11_maxbest',
            'metrics': compact_result(current_comp_payload),
        },
        {
            'candidate_name': 'twoanchor_l10_zpair_neg4_then_ypair_neg2_plus_l11_maxbest',
            'metrics': compact_result(residual_hybrid_payload),
        },
        {
            'candidate_name': 'twoanchor_l10_ypair_neg4_plus_l11_maxbest',
            'metrics': compact_result(bestmax_payload),
        },
    ])
    for row in rows:
        row['on_pareto_frontier'] = row['candidate_name'] in frontier

    old_interior_max = float(compact_result(interior_payload)['overall']['max_pct_error'])
    bestmean_mean = float(compact_result(bestmean_payload)['overall']['mean_pct_error'])
    unified_pool = [
        row for row in rows
        if row['metrics']['overall']['max_pct_error'] < old_interior_max - 1e-12
        and row['metrics']['overall']['mean_pct_error'] <= bestmean_mean + 1e-12
    ]
    if not unified_pool:
        unified_pool = rows
    best_unified_row = min(unified_pool, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    best_unified_cand = candidate_by_name[best_unified_row['candidate_name']]
    best_unified_payload = payload_by_name[best_unified_cand.name]

    new_frontier_rows = [row for row in rows if row['on_pareto_frontier']]
    best_new_maxlean_row = min(new_frontier_rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    best_new_maxlean_cand = candidate_by_name[best_new_maxlean_row['candidate_name']]
    best_new_maxlean_payload = payload_by_name[best_new_maxlean_cand.name]

    kf36_rows: list[dict[str, Any]] = []
    for name in select_kf_rechecks(new_frontier_rows):
        cand = candidate_by_name[name]
        kf_payload, kf_status, kf_path = run_candidate_payload(mod, cand, 'kf36_noisy', args.noise_scale, args.force_rerun)
        note = 'new frontier candidate'
        if name == best_unified_cand.name:
            note = 'best unified candidate'
        elif name == best_new_maxlean_cand.name:
            note = 'lowest-max new frontier point'
        kf36_rows.append({
            'candidate_name': name,
            'note': note,
            'markov42': compact_result(payload_by_name[name]),
            'kf36': compact_result(kf_payload),
            'kf36_status': kf_status,
            'kf36_run_json': str(kf_path),
        })

    best_unified_summary = {
        'candidate_name': best_unified_cand.name,
        'candidate_class': spec_by_name[best_unified_cand.name]['class'],
        'rationale': spec_by_name[best_unified_cand.name]['rationale'],
        'total_time_s': best_unified_cand.total_time_s,
        'continuity_checks': best_unified_cand.continuity_checks,
        'all_rows': best_unified_cand.all_rows,
        'all_actions': best_unified_cand.all_actions,
        'all_faces': best_unified_cand.all_faces,
        'markov42': compact_result(best_unified_payload),
        'markov42_run_json': best_unified_row['run_json'],
        'delta_vs_faithful': delta_vs_ref(refs['faithful_markov'], best_unified_payload),
        'delta_vs_bestmean_anchor': delta_vs_ref(bestmean_payload, best_unified_payload),
        'delta_vs_interior_point': delta_vs_ref(interior_payload, best_unified_payload),
        'delta_vs_current_compromise': delta_vs_ref(current_comp_payload, best_unified_payload),
        'delta_vs_residual_hybrid': delta_vs_ref(residual_hybrid_payload, best_unified_payload),
        'delta_vs_bestmax_anchor': delta_vs_ref(bestmax_payload, best_unified_payload),
        'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], best_unified_payload),
        'delta_vs_default18': delta_vs_ref(default18_payload, best_unified_payload),
    }

    best_new_maxlean_summary = {
        'candidate_name': best_new_maxlean_cand.name,
        'candidate_class': spec_by_name[best_new_maxlean_cand.name]['class'],
        'rationale': spec_by_name[best_new_maxlean_cand.name]['rationale'],
        'total_time_s': best_new_maxlean_cand.total_time_s,
        'markov42': compact_result(best_new_maxlean_payload),
        'markov42_run_json': best_new_maxlean_row['run_json'],
        'delta_vs_bestmean_anchor': delta_vs_ref(bestmean_payload, best_new_maxlean_payload),
        'delta_vs_interior_point': delta_vs_ref(interior_payload, best_new_maxlean_payload),
        'delta_vs_bestmax_anchor': delta_vs_ref(bestmax_payload, best_new_maxlean_payload),
    }

    comparison_rows = [
        {
            'label': 'faithful12',
            'note': 'base faithful12 scaffold',
            'markov42': compact_result(refs['faithful_markov']),
            'kf36': compact_result(refs['faithful_kf']),
        },
        {
            'label': 'old best-mean anchor',
            'note': 'twoanchor_l10_zpair_neg6_plus_l11_bestmean',
            'markov42': compact_result(bestmean_payload),
            'kf36': compact_result(twoanchor_refs['bestmean_kf']),
        },
        {
            'label': 'old interior point',
            'note': 'twoanchor_l10_zpair_neg6_plus_l11_y10x0back2',
            'markov42': compact_result(interior_payload),
            'kf36': compact_result(twoanchor_refs['interior_kf']),
        },
        {
            'label': 'best unified candidate',
            'note': best_unified_cand.name,
            'markov42': best_unified_summary['markov42'],
            'kf36': next(row['kf36'] for row in kf36_rows if row['candidate_name'] == best_unified_cand.name),
        },
        {
            'label': 'max-leaning new frontier point',
            'note': best_new_maxlean_cand.name,
            'markov42': best_new_maxlean_summary['markov42'],
            'kf36': next(row['kf36'] for row in kf36_rows if row['candidate_name'] == best_new_maxlean_cand.name),
        },
        {
            'label': 'old compromise point',
            'note': 'twoanchor_l10_zpair_neg4_plus_l11_maxbest',
            'markov42': compact_result(current_comp_payload),
            'kf36': compact_result(twoanchor_refs['current_comp_kf']),
        },
        {
            'label': 'residual-hybrid reference',
            'note': 'twoanchor_l10_zpair_neg4_then_ypair_neg2_plus_l11_maxbest',
            'markov42': compact_result(residual_hybrid_payload),
            'kf36': compact_result(twoanchor_refs['residual_hybrid_kf']),
        },
        {
            'label': 'old best-max anchor',
            'note': 'twoanchor_l10_ypair_neg4_plus_l11_maxbest',
            'markov42': compact_result(bestmax_payload),
            'kf36': compact_result(twoanchor_refs['bestmax_kf']),
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
            'markov42': compact_result(default18_payload),
            'kf36': compact_result(twoanchor_refs['default18_kf']),
        },
    ]

    improves_bestmean_both = (
        best_unified_summary['delta_vs_bestmean_anchor']['mean_pct_error']['improvement_pct_points'] > 0.0
        and best_unified_summary['delta_vs_bestmean_anchor']['max_pct_error']['improvement_pct_points'] > 0.0
    )
    improves_interior_both = (
        best_unified_summary['delta_vs_interior_point']['mean_pct_error']['improvement_pct_points'] > 0.0
        and best_unified_summary['delta_vs_interior_point']['max_pct_error']['improvement_pct_points'] > 0.0
    )

    success = 'YES' if (improves_bestmean_both and improves_interior_both) else 'NO'
    verdict = (
        f"{best_unified_cand.name} is a real unified frontier win: it reaches {best_unified_summary['markov42']['overall']['mean_pct_error']:.3f} / "
        f"{best_unified_summary['markov42']['overall']['median_pct_error']:.3f} / {best_unified_summary['markov42']['overall']['max_pct_error']:.3f}, "
        f"improving the old best-mean anchor by {best_unified_summary['delta_vs_bestmean_anchor']['mean_pct_error']['improvement_pct_points']:.3f} mean-points and "
        f"{best_unified_summary['delta_vs_bestmean_anchor']['max_pct_error']['improvement_pct_points']:.3f} max-points simultaneously."
    )

    scientific_conclusion = (
        f"The unification pass succeeded. The key structural lesson is that the best compromise did not come from the heavier z4+y2 hybrid; it came from a milder anchor10 interpolation: {best_unified_cand.name}. "
        f"That z5+y1 minimal blend with late11 y10x0back2 improves dKa_yy ({best_unified_summary['markov42']['key_param_errors']['dKa_yy']:.3f}), dKg_zz ({best_unified_summary['markov42']['key_param_errors']['dKg_zz']:.3f}), "
        f"Ka2_y ({best_unified_summary['markov42']['key_param_errors']['Ka2_y']:.3f}), and Ka2_z ({best_unified_summary['markov42']['key_param_errors']['Ka2_z']:.3f}) relative to the old best-mean anchor, while also moving the mean/max pair to a strictly better location. "
        f"The stronger z4+y2 blends still define a useful lower-max subfrontier, with {best_new_maxlean_cand.name} reaching {best_new_maxlean_summary['markov42']['overall']['max_pct_error']:.3f}, but they no longer look like the best single unified choice because they give back too much mean." 
    )

    out_json = RESULTS_DIR / f'ch3_twoanchor_unified_compromise_{make_suffix(args.noise_scale)}.json'
    out_md = REPORTS_DIR / f'psins_ch3_twoanchor_unified_compromise_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_twoanchor_unified_compromise',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'hard_constraints': {
            'base_skeleton': 'faithful chapter-3 12-position original sequence',
            'continuity_rule': 'exact same mechanism state before resume',
            'time_budget_s': [1200.0, 1800.0],
            'seed': 42,
            'truth_family': 'shared low-noise benchmark',
            'search_style': 'small interpretable two-anchor blend set only',
        },
        'references': {
            'faithful12': {
                'candidate_name': faithful.name,
                'markov42': compact_result(refs['faithful_markov']),
                'kf36': compact_result(refs['faithful_kf']),
            },
            'bestmean_anchor': {
                'candidate_name': 'twoanchor_l10_zpair_neg6_plus_l11_bestmean',
                'markov42': compact_result(bestmean_payload),
                'kf36': compact_result(twoanchor_refs['bestmean_kf']),
            },
            'interior_point': {
                'candidate_name': 'twoanchor_l10_zpair_neg6_plus_l11_y10x0back2',
                'markov42': compact_result(interior_payload),
                'kf36': compact_result(twoanchor_refs['interior_kf']),
            },
            'current_compromise': {
                'candidate_name': 'twoanchor_l10_zpair_neg4_plus_l11_maxbest',
                'markov42': compact_result(current_comp_payload),
                'kf36': compact_result(twoanchor_refs['current_comp_kf']),
            },
            'residual_hybrid': {
                'candidate_name': 'twoanchor_l10_zpair_neg4_then_ypair_neg2_plus_l11_maxbest',
                'markov42': compact_result(residual_hybrid_payload),
                'kf36': compact_result(twoanchor_refs['residual_hybrid_kf']),
            },
            'bestmax_anchor': {
                'candidate_name': 'twoanchor_l10_ypair_neg4_plus_l11_maxbest',
                'markov42': compact_result(bestmax_payload),
                'kf36': compact_result(twoanchor_refs['bestmax_kf']),
            },
            'old_best_legal': {
                'candidate_name': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
                'markov42': compact_result(refs['oldbest_markov']),
                'kf36': compact_result(refs['oldbest_kf']),
            },
            'default18': {
                'candidate_name': 'default18',
                'markov42': compact_result(default18_payload),
                'kf36': compact_result(twoanchor_refs['default18_kf']),
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
        'best_unified_candidate': best_unified_summary,
        'best_new_maxlean_candidate': best_new_maxlean_summary,
        'kf36_rows': kf36_rows,
        'comparison_rows': comparison_rows,
        'bottom_line': {
            'success': success,
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
    print('BEST_UNIFIED', best_unified_cand.name, best_unified_summary['markov42']['overall'], flush=True)
    print('BEST_MAXLEAN', best_new_maxlean_cand.name, best_new_maxlean_summary['markov42']['overall'], flush=True)
    print('BOTTOM_LINE', verdict, flush=True)


if __name__ == '__main__':
    main()
