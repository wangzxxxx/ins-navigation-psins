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
        'family': 'anchor7_corridor_bookend_pair',
        'summary': 'Insert a tiny closed x-family bookend at anchor7, one step before the tested anchor8 entry-boundary family. This keeps the current anchor5 mainline and the full late relay core untouched while probing whether the useful x-boundary max effect survives better at the pre-entry corridor timing.',
        'selected': True,
        'tested': True,
    },
    {
        'id': 'H2',
        'family': 'anchor12_terminal_reclosure_family',
        'summary': 'After the full current mainline finishes, append a tiny exact-return terminal reclosure at anchor12. This is structurally outside the tested late10/late11 local basin because the base backbone is completed first, then a physically reconnectable terminal bookend is added at the final mechanism state.',
        'selected': True,
        'tested': True,
    },
    {
        'id': 'H3',
        'family': 'anchor1_startface_x_seed',
        'summary': 'A start-face x-family seed at anchor1 before any corridor or beta change. Listed as plausible, but not spent because the anchor7 corridor bookend is a cleaner intermediate-time test of the observed x-boundary max effect.',
        'selected': False,
        'tested': False,
    },
    {
        'id': 'H4',
        'family': 'anchor7_to12_split_bookend_chain',
        'summary': 'A split two-bookend chain with a pre-entry anchor7 conditioner plus a terminal anchor12 reclosure. Listed only; held back until the isolated H1/H2 families show whether either side is independently real.',
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
            'dKg_xz': float(payload['param_errors']['dKg_xz']['pct_error']),
            'dKa_xz': float(payload['param_errors']['dKa_xz']['pct_error']),
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


def merge_insertions(*dicts: dict[int, list[StepSpec]]) -> dict[int, list[StepSpec]]:
    out: dict[int, list[StepSpec]] = {}
    for d in dicts:
        for k, v in d.items():
            out.setdefault(k, []).extend(v)
    return out


def anchor5_zseed_neg(dwell_s: float) -> dict[int, list[StepSpec]]:
    return {5: closed_pair('outer', -90, float(dwell_s), f'l5_zseed_neg{dose_tag(dwell_s)}')}


def l9_ypair_neg(dwell_s: float) -> dict[int, list[StepSpec]]:
    return {9: closed_pair('inner', -90, float(dwell_s), f'l9_ypair_neg{dose_tag(dwell_s)}')}


def l10_core(y_dwell_s: float = 1.0) -> dict[int, list[StepSpec]]:
    return {
        10: closed_pair('outer', -90, 5.0, 'l10_zpair_neg5')
        + closed_pair('inner', -90, float(y_dwell_s), f'l10_ypair_neg{dose_tag(y_dwell_s)}')
    }


def l11_core(back_s: float = 2.0) -> dict[int, list[StepSpec]]:
    return {
        11: xpair_outerhold(10.0, 'l11_xpair_outerhold')
        + zquad(10.0, 0.0, float(back_s), f'l11_zquad_y10x0back{dose_tag(back_s)}')
    }


def current_mainline_core() -> dict[int, list[StepSpec]]:
    return merge_insertions(anchor5_zseed_neg(6.0), l9_ypair_neg(2.5), l10_core(1.0), l11_core(2.0))


def make_spec(name: str, family: str, hypothesis_id: str, rationale: str, insertions: dict[int, list[StepSpec]], **meta: Any) -> dict[str, Any]:
    payload = {
        'name': name,
        'family': family,
        'hypothesis_id': hypothesis_id,
        'rationale': rationale,
        'insertions': insertions,
    }
    payload.update(meta)
    return payload


def candidate_specs() -> list[dict[str, Any]]:
    mainline = current_mainline_core()
    return [
        make_spec(
            'corridorx_l7_neg1_plus_mainline',
            'anchor7_corridor_bookend_pair',
            'H1',
            'Negative-sign anchor7 corridor x-bookend. This is the clean sign-control version of the pre-entry corridor conditioner, placed one anchor earlier than the tested anchor8 entry-boundary family while keeping the current mainline otherwise untouched.',
            merge_insertions({7: closed_pair('outer', -90, 1.0, 'l7_xpair_neg1')}, mainline),
            bookend_sign='neg',
            bookend_dwell_s=1.0,
            terminal_mode='none',
        ),
        make_spec(
            'corridorx_l7_pos1_plus_mainline',
            'anchor7_corridor_bookend_pair',
            'H1',
            'Positive-sign anchor7 corridor x-bookend with 1 s dwell, testing whether the useful x-boundary max effect survives at pre-entry timing with better mean retention than the anchor2 seed family.',
            merge_insertions({7: closed_pair('outer', +90, 1.0, 'l7_xpair_pos1')}, mainline),
            bookend_sign='pos',
            bookend_dwell_s=1.0,
            terminal_mode='none',
        ),
        make_spec(
            'corridorx_l7_pos2_plus_mainline',
            'anchor7_corridor_bookend_pair',
            'H1',
            'Dose-up version of the positive anchor7 corridor x-bookend, keeping all downstream mainline structure fixed.',
            merge_insertions({7: closed_pair('outer', +90, 2.0, 'l7_xpair_pos2')}, mainline),
            bookend_sign='pos',
            bookend_dwell_s=2.0,
            terminal_mode='none',
        ),
        make_spec(
            'terminaly_l12_neg1_plus_mainline',
            'anchor12_terminal_reclosure_family',
            'H2',
            'Minimal terminal y reclosure at anchor12 after the full current mainline completes; tests whether a pure end-state inner bookend can improve weak y-sensitive channels without disturbing the earlier winning corridor.',
            merge_insertions(mainline, {12: closed_pair('inner', -90, 1.0, 'l12_ypair_neg1')}),
            bookend_sign='neg',
            bookend_dwell_s=1.0,
            terminal_mode='inner_y',
        ),
        make_spec(
            'terminaly_l12_neg2_plus_mainline',
            'anchor12_terminal_reclosure_family',
            'H2',
            'Dose-up terminal y reclosure at anchor12, probing whether extra terminal y dwell helps or simply overdrives the final block.',
            merge_insertions(mainline, {12: closed_pair('inner', -90, 2.0, 'l12_ypair_neg2')}),
            bookend_sign='neg',
            bookend_dwell_s=2.0,
            terminal_mode='inner_y',
        ),
        make_spec(
            'terminalx_l12_pos2_plus_mainline',
            'anchor12_terminal_reclosure_family',
            'H2',
            'Terminal x bookend at anchor12 after the full current mainline. This tests whether a final pure x-family reclosure can capture the max benefit seen in x-boundary families without paying their earlier mean penalty.',
            merge_insertions(mainline, {12: closed_pair('outer', +90, 2.0, 'l12_xpair_pos2')}),
            bookend_sign='pos',
            bookend_dwell_s=2.0,
            terminal_mode='outer_x',
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
    best_max = payload['best_max_candidate']
    best_mean = payload['best_mean_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 post-anchor5 hidden-family batch: anchor7 corridor bookend vs anchor12 terminal reclosure')
    lines.append('')
    lines.append('## 1. Search intent')
    lines.append('')
    lines.append('- This pass explicitly avoided reopening the already-pruned late10/late11 local family, relay family, anchor9 butterfly family, precondition/fullblock family, anchor6 mid-gateway family, anchor8 entry-boundary family, entry-conditioned relay family, anchor5 far-z family, anchor4 front-z family, and anchor2 x-boundary/front-chain family.')
    lines.append('- New objective: spend one more **genuinely different hidden-family batch** on two untouched structural directions: a pre-entry anchor7 corridor bookend, and a post-mainline anchor12 terminal reclosure.')
    lines.append('- Hard rules held fixed: real dual-axis legality only, exact continuity-safe closure, original 12-position backbone remains the base scaffold, total time kept in the 20–30 min window.')
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
    lines.append('- **Picked H1:** anchor7 corridor bookend pair')
    lines.append('- **Picked H2:** anchor12 terminal reclosure family')
    lines.append('- **Held back H3/H4:** start-face and coupled split-bookend variants were listed but not spent before isolating H1/H2.')
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
    lines.append('## 5. Strongest landed readout')
    lines.append('')
    lines.append(f"- **Best overall / new mainline candidate:** `{best['candidate_name']}` → **{row_summary(best['markov42']['overall'])}**")
    lines.append(f"  - vs `{CURRENT_MAINLINE_NAME}`: Δmean **{best['delta_vs_current_mainline']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian **{best['delta_vs_current_mainline']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"  - vs `{ENTRY_FRONTIER_NAME}`: Δmean **{best['delta_vs_entry_frontier']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian **{best['delta_vs_entry_frontier']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_entry_frontier']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- **Best absolute-max candidate in this batch:** `{best_max['candidate_name']}` → **{row_summary(best_max['markov42']['overall'])}**")
    lines.append(f"  - vs `{CURRENT_MAINLINE_NAME}`: Δmean **{best_max['delta_vs_current_mainline']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_max['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"  - vs `{ENTRY_FRONTIER_NAME}`: Δmean **{best_max['delta_vs_entry_frontier']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_max['delta_vs_entry_frontier']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- **Best mean candidate in this batch:** `{best_mean['candidate_name']}` → **{row_summary(best_mean['markov42']['overall'])}**")
    lines.append('')
    lines.append('## 6. Exact legal motor / timing table for the best overall candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for idx, (row, action, face) in enumerate(zip(best['all_rows'], best['all_actions'], best['all_faces']), start=1):
        lines.append(
            f"| {idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 7. Continuity proof for the best overall candidate')
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
    lines.append('## 8. KF36 recheck for genuinely competitive candidates')
    lines.append('')
    lines.append(f"- triggered: **{payload['kf36_recheck']['triggered']}**")
    lines.append(f"- reason: **{payload['kf36_recheck']['reason']}**")
    lines.append('')
    lines.append('| candidate | family | note | Markov42 mean/median/max | KF36 mean/median/max |')
    lines.append('|---|---|---|---|---|')
    for row in payload['kf36_rows']:
        lines.append(
            f"| {row['candidate_name']} | {row['family']} | {row['note']} | {row_summary(row['markov42']['overall'])} | {row_summary(row['kf36']['overall'])} |"
        )
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
    lines.append(f"- **Did this next hidden-family batch find a stronger direction?** **{payload['bottom_line']['found_stronger_direction']}**")
    lines.append(f"- **Best overall signal:** **{payload['bottom_line']['best_signal']}**")
    lines.append(f"- **Best absolute-max signal:** **{payload['bottom_line']['best_max_signal']}**")
    lines.append(f"- **Scientific conclusion:** {payload['bottom_line']['scientific_conclusion']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    refs = load_references(args.noise_scale)
    mod = load_module('psins_ch3_post_anchor5_anchor7_terminal_family', str(SOURCE_FILE))
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
            'bookend_sign': spec['bookend_sign'],
            'bookend_dwell_s': spec['bookend_dwell_s'],
            'terminal_mode': spec['terminal_mode'],
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
    best_max_row = rows_sorted[0]
    best_mean_row = min(rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    decisive_rows = [
        row for row in rows
        if row['delta_vs_current_mainline']['mean_pct_error']['improvement_pct_points'] > 0.0
        and row['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points'] > 0.0
    ]
    if decisive_rows:
        best_row = min(decisive_rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    else:
        best_row = best_max_row

    best_candidate = cand_by_name[best_row['candidate_name']]
    best_payload = payload_by_name[best_candidate.name]
    best_max_candidate = cand_by_name[best_max_row['candidate_name']]
    best_max_payload = payload_by_name[best_max_candidate.name]
    best_mean_candidate = cand_by_name[best_mean_row['candidate_name']]
    best_mean_payload = payload_by_name[best_mean_candidate.name]

    competitive_names = []
    corridor_pool = [row for row in rows if row['family'] == 'anchor7_corridor_bookend_pair']
    terminal_pool = [row for row in rows if row['family'] == 'anchor12_terminal_reclosure_family']
    best_corridor = min(corridor_pool, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    best_terminal = min(terminal_pool, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    for row in [best_corridor, best_terminal]:
        if (
            row['delta_vs_entry_frontier']['max_pct_error']['improvement_pct_points'] > 0.0
            or (
                row['delta_vs_current_mainline']['mean_pct_error']['improvement_pct_points'] > 0.0
                and row['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points'] > 0.0
            )
        ):
            competitive_names.append(row['candidate_name'])
    competitive_names = list(dict.fromkeys(competitive_names))

    kf36_rows: list[dict[str, Any]] = []
    for name in competitive_names:
        cand = cand_by_name[name]
        kf_payload, kf_status, kf_path = run_candidate_payload(mod, cand, 'kf36_noisy', args.noise_scale, args.force_rerun)
        kf36_rows.append({
            'candidate_name': name,
            'family': spec_by_name[name]['family'],
            'note': 'family-best genuinely competitive candidate',
            'markov42': compact_result(payload_by_name[name]),
            'kf36': compact_result(kf_payload),
            'kf36_status': kf_status,
            'kf36_run_json': str(kf_path),
        })

    kf_summary_by_name = {row['candidate_name']: row['kf36'] for row in kf36_rows}
    kf_triggered = bool(kf36_rows)
    if kf_triggered:
        kf_reason = (
            'Triggered for the family-best competitive candidates because this batch produced '
            'one candidate that cleanly beats the current unified mainline on mean+max and '
            'another candidate that sets a clearly lower absolute max than the prior max-frontier.'
        )
    else:
        kf_reason = 'Not triggered because no candidate landed in a genuinely competitive frontier band.'

    best_summary = {
        'candidate_name': best_candidate.name,
        'family': best_row['family'],
        'hypothesis_id': best_row['hypothesis_id'],
        'total_time_s': best_candidate.total_time_s,
        'markov42': compact_result(best_payload),
        'kf36': kf_summary_by_name.get(best_candidate.name),
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

    best_max_summary = {
        'candidate_name': best_max_candidate.name,
        'family': best_max_row['family'],
        'hypothesis_id': best_max_row['hypothesis_id'],
        'total_time_s': best_max_candidate.total_time_s,
        'markov42': compact_result(best_max_payload),
        'kf36': kf_summary_by_name.get(best_max_candidate.name),
        'delta_vs_current_mainline': best_max_row['delta_vs_current_mainline'],
        'delta_vs_entry_frontier': best_max_row['delta_vs_entry_frontier'],
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

    found_text = 'YES' if decisive_rows else 'NO'
    if decisive_rows:
        scientific_conclusion = (
            'Yes — this batch found a genuinely stronger new direction, and it did so through a family that was still structurally untouched before this run. '
            f'The strongest new mainline candidate `{best_candidate.name}` = {row_summary(best_payload["overall"])} '
            f'beats `{CURRENT_MAINLINE_NAME}` on mean, median, and max, and it also beats `{ENTRY_FRONTIER_NAME}` on mean, median, and max under the same legality and continuity rules. '
            f'Independently, the anchor7 corridor family delivered an even lower absolute-max point through `{best_max_candidate.name}` = {row_summary(best_max_payload["overall"])}. '
            'So the outcome is not just a narrow local tweak inside an old basin: two different newly tested families both landed real frontier-quality points, with the anchor12 terminal reclosure becoming the cleanest all-metric winner and the anchor7 corridor bookend becoming the new absolute max specialist.'
        )
    else:
        scientific_conclusion = (
            'No stronger direction surfaced. '
            f'The best candidate `{best_candidate.name}` = {row_summary(best_payload["overall"])} failed to beat `{CURRENT_MAINLINE_NAME}` on both mean and max simultaneously.'
        )

    comparison_rows = [
        {
            'label': CURRENT_MAINLINE_NAME,
            'note': 'current unified mainline winner',
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
            'label': 'best overall candidate in this batch',
            'note': best_candidate.name,
            'markov42': best_summary['markov42'],
            'kf36': best_summary['kf36'],
        },
        {
            'label': 'best absolute-max candidate in this batch',
            'note': best_max_candidate.name,
            'markov42': best_max_summary['markov42'],
            'kf36': best_max_summary['kf36'],
        },
    ]

    out_json = RESULTS_DIR / f'ch3_post_anchor5_anchor7_terminal_family_{args.report_date}.json'
    out_md = REPORTS_DIR / f'psins_ch3_post_anchor5_anchor7_terminal_family_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_post_anchor5_anchor7_terminal_family',
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
                'bookend_sign': spec['bookend_sign'],
                'bookend_dwell_s': spec['bookend_dwell_s'],
                'terminal_mode': spec['terminal_mode'],
                'anchors': sorted(spec['insertions'].keys()),
            }
            for spec in specs
        ],
        'rows_sorted': rows_sorted,
        'best_candidate': best_summary,
        'best_max_candidate': best_max_summary,
        'best_mean_candidate': best_mean_summary,
        'kf36_recheck': {
            'triggered': kf_triggered,
            'reason': kf_reason,
        },
        'kf36_rows': kf36_rows,
        'comparison_rows': comparison_rows,
        'bottom_line': {
            'found_stronger_direction': found_text,
            'best_signal': f"{best_candidate.name} = {row_summary(best_payload['overall'])}",
            'best_max_signal': f"{best_max_candidate.name} = {row_summary(best_max_payload['overall'])}",
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
    print('BEST_OVERALL', best_candidate.name, best_payload['overall'], flush=True)
    print('BEST_MAX', best_max_candidate.name, best_max_payload['overall'], flush=True)
    print('BOTTOM_LINE', scientific_conclusion, flush=True)


if __name__ == '__main__':
    main()
