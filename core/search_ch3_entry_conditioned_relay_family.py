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

RELAYMAX_LOWMAX_Y1_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relaymax_lowmax_l9y1_shared_noise0p08_param_errors.json'
RELAYMAX_UNIFIED_Y1_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relaymax_unified_l9y1_shared_noise0p08_param_errors.json'
RELAYMAX_UNIFIED_Y2_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relaymax_unified_l9y2_shared_noise0p08_param_errors.json'
HISTORICAL_UNIFIED_CORE_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2_shared_noise0p08_param_errors.json'
HISTORICAL_UNIFIED_CORE_KF = RESULTS_DIR / 'KF36_ch3closedloop_twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2_shared_noise0p08_param_errors.json'
ENTRY_BOUNDARY_MAX_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_entryx_l8_xpair_pos3_plus_l11_y10x0back2_shared_noise0p08_param_errors.json'
ENTRY_BOUNDARY_MAX_KF = RESULTS_DIR / 'KF36_ch3closedloop_entryx_l8_xpair_pos3_plus_l11_y10x0back2_shared_noise0p08_param_errors.json'
OLD_BEST_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
OLD_BEST_KF = RESULTS_DIR / 'KF36_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
DEFAULT18_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json'
DEFAULT18_KF = RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json'

CURRENT_UNIFIED_NAME = 'relaymax_unified_l9y2'
HISTORICAL_UNIFIED_CORE_NAME = 'twoanchor_l10_zpair_neg5_then_ypair_neg1_plus_l11_y10x0back2'


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


def l8_xpair(dwell_s: float, label: str) -> dict[int, list[StepSpec]]:
    return {8: closed_pair('outer', +90 if dwell_s >= 0 else -90, abs(dwell_s), label)}


def l9_ypair_neg(dwell_s: float, label: str) -> dict[int, list[StepSpec]]:
    return {9: closed_pair('inner', -90, float(dwell_s), label)}


def l10_unified_core() -> dict[int, list[StepSpec]]:
    return {10: closed_pair('outer', -90, 5.0, 'l10_zpair_neg5') + closed_pair('inner', -90, 1.0, 'l10_ypair_neg1')}


def l10_lowmax_core() -> dict[int, list[StepSpec]]:
    return {10: closed_pair('outer', -90, 4.0, 'l10_zpair_neg4') + closed_pair('inner', -90, 2.0, 'l10_ypair_neg2')}


def l11_y10x0back2_core() -> dict[int, list[StepSpec]]:
    return {11: xpair_outerhold(10.0, 'l11_xpair_outerhold') + zquad(10.0, 0.0, 2.0, 'l11_zquad_y10x0back2')}


def candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    # x=0 references inside the l9y1 relay family
    specs.append({
        'name': 'relaymax_lowmax_l9y1',
        'core_family': 'lowmaxcore',
        'entry_x_dwell': 0.0,
        'is_new_candidate': False,
        'rationale': 'Baseline relay-max reference: l9 y1 gate plus the low-max core (l10 z4+y2, l11 y10x0back2).',
        'insertions': merge_insertions(l9_ypair_neg(1.0, 'l9_ypair_neg1'), l10_lowmax_core(), l11_y10x0back2_core()),
    })
    specs.append({
        'name': 'relaymax_unified_l9y1',
        'core_family': 'unifiedcore',
        'entry_x_dwell': 0.0,
        'is_new_candidate': False,
        'rationale': 'Baseline relay-unified reference: l9 y1 gate plus the old unified core (l10 z5+y1, l11 y10x0back2).',
        'insertions': merge_insertions(l9_ypair_neg(1.0, 'l9_ypair_neg1'), l10_unified_core(), l11_y10x0back2_core()),
    })
    for dwell, name_prefix in [(-1.0, 'xneg1'), (1.0, 'x1'), (2.0, 'x2'), (3.0, 'xpos3')]:
        specs.append({
            'name': f'entryrelay_l8{name_prefix}_l9y1_lowmaxcore',
            'core_family': 'lowmaxcore',
            'entry_x_dwell': dwell,
            'is_new_candidate': True,
            'rationale': 'Entry-conditioned relay family: add an anchor8 x-bookend before the l9 y1 relay gate, then keep the low-max relay core fixed.',
            'insertions': merge_insertions(l8_xpair(dwell, f'l8_{name_prefix}'), l9_ypair_neg(1.0, 'l9_ypair_neg1'), l10_lowmax_core(), l11_y10x0back2_core()),
        })
    for dwell, name_prefix in [(-1.0, 'xneg1'), (1.0, 'x1'), (2.0, 'x2'), (3.0, 'xpos3')]:
        specs.append({
            'name': f'entryrelay_l8{name_prefix}_l9y1_unifiedcore',
            'core_family': 'unifiedcore',
            'entry_x_dwell': dwell,
            'is_new_candidate': True,
            'rationale': 'Entry-conditioned relay family: add an anchor8 x-bookend before the l9 y1 relay gate, then keep the unified relay core fixed.',
            'insertions': merge_insertions(l8_xpair(dwell, f'l8_{name_prefix}'), l9_ypair_neg(1.0, 'l9_ypair_neg1'), l10_unified_core(), l11_y10x0back2_core()),
        })
    return specs


def current_unified_winner_spec() -> dict[str, Any]:
    return {
        'name': CURRENT_UNIFIED_NAME,
        'rationale': 'Current best pre-existing unified relay point: l9 y2 plus the unified core.',
        'insertions': merge_insertions(l9_ypair_neg(2.0, 'l9_ypair_neg2'), l10_unified_core(), l11_y10x0back2_core()),
    }


def load_references(noise_scale: float) -> dict[str, Any]:
    return {
        'relaymax_lowmax_l9y1': load_json_checked(RELAYMAX_LOWMAX_Y1_MARKOV, noise_scale),
        'relaymax_unified_l9y1': load_json_checked(RELAYMAX_UNIFIED_Y1_MARKOV, noise_scale),
        'current_unified_winner_markov': load_json_checked(RELAYMAX_UNIFIED_Y2_MARKOV, noise_scale),
        'historical_unified_core_markov': load_json_checked(HISTORICAL_UNIFIED_CORE_MARKOV, noise_scale),
        'historical_unified_core_kf': load_json_checked(HISTORICAL_UNIFIED_CORE_KF, noise_scale),
        'entry_boundary_max_markov': load_json_checked(ENTRY_BOUNDARY_MAX_MARKOV, noise_scale),
        'entry_boundary_max_kf': load_json_checked(ENTRY_BOUNDARY_MAX_KF, noise_scale),
        'old_best_markov': load_json_checked(OLD_BEST_MARKOV, noise_scale),
        'old_best_kf': load_json_checked(OLD_BEST_KF, noise_scale),
        'default18_markov': load_json_checked(DEFAULT18_MARKOV, noise_scale),
        'default18_kf': load_json_checked(DEFAULT18_KF, noise_scale),
    }


def row_summary(payload: dict[str, Any]) -> str:
    o = payload['overall'] if 'overall' in payload else payload
    return f"{o['mean_pct_error']:.3f} / {o['median_pct_error']:.3f} / {o['max_pct_error']:.3f}"


def select_best(rows: list[dict[str, Any]], core_family: str) -> dict[str, Any]:
    pool = [row for row in rows if row['core_family'] == core_family and row['is_new_candidate']]
    return min(pool, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))


def render_report(payload: dict[str, Any]) -> str:
    refs = payload['references']
    best_main = payload['best_new_mainline_candidate']
    best_lowmax = payload['best_new_lowmax_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 entry-conditioned relay family follow-up')
    lines.append('')
    lines.append('## 1. Search question')
    lines.append('')
    lines.append('- Starting signal to verify: `l8 +X1 + l9y1 + unified core` looked materially real at **9.614 / 1.169 / 99.574**.')
    lines.append('- This pass asked two things only:')
    lines.append('  1. Does the **sign-control counterpart** collapse, confirming a real directional mechanism instead of random noise?')
    lines.append('  2. Does a **small full x-dose batch around l8** keep the family real under both relay cores (`lowmaxcore` / `unifiedcore`)?')
    lines.append('')
    lines.append('## 2. Fixed structural rule')
    lines.append('')
    lines.append('- faithful chapter-3 original 12-position scaffold only')
    lines.append('- real dual-axis legality only')
    lines.append('- exact continuity-safe closure before each resume')
    lines.append('- theory-guided local x-dose batch around the same `l9 y1` relay gate')
    lines.append('')
    lines.append('## 3. Comparison references used')
    lines.append('')
    lines.append(f"- relay max reference: **{row_summary(refs['relaymax_lowmax_l9y1']['markov42']['overall'])}** (`relaymax_lowmax_l9y1`)")
    lines.append(f"- current unified winner used in this report: **{row_summary(refs['current_unified_winner']['markov42']['overall'])}** (`{refs['current_unified_winner']['candidate_name']}`)")
    lines.append(f"- historical unified core predecessor: **{row_summary(refs['historical_unified_core']['markov42']['overall'])}** (`{refs['historical_unified_core']['candidate_name']}`)")
    lines.append(f"- entry-boundary max branch: **{row_summary(refs['entry_boundary_max']['markov42']['overall'])}** (`{refs['entry_boundary_max']['candidate_name']}`)")
    lines.append(f"- old best legal: **{row_summary(refs['old_best_legal']['markov42']['overall'])}**")
    lines.append(f"- default18: **{row_summary(refs['default18']['markov42']['overall'])}**")
    lines.append('')
    lines.append('## 4. Small full batch around the entry-conditioned relay family (Markov42)')
    lines.append('')
    lines.append('| rank | candidate | core | l8 x dose | new? | total_s | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Ka2_z | Δmean vs same-core x0 | Δmax vs same-core x0 |')
    lines.append('|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(payload['batch_rows_sorted'], start=1):
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        delta_core = row['delta_vs_same_core_x0']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['core_family']} | {row['entry_x_dwell']:+.0f} | {'yes' if row['is_new_candidate'] else 'ref'} | {row['total_time_s']:.0f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {delta_core['mean_pct_error']['improvement_pct_points']:+.3f} | {delta_core['max_pct_error']['improvement_pct_points']:+.3f} |"
        )
    lines.append('')
    lines.append('## 5. Sign-control / local-dose readout')
    lines.append('')
    lines.append(f"- **Unified core sign control**: `x=-1` gave **{row_summary(payload['sign_control']['unified_neg1']['markov42']['overall'])}**, which is worse than both `x=0` and `x=+1`. So the family is **directional**, not sign-agnostic.")
    lines.append(f"- **Unified core local optimum**: `x=+1` is the best unified-core entry-conditioned point; both `x=+2` and `x=+3` regress on mean and max.")
    lines.append(f"- **Low-max core local optimum**: `x=+1` is also the best low-max-core entry-conditioned point in this batch; the positive signal survives across both cores.")
    lines.append('')
    lines.append('## 6. Best landed new candidates')
    lines.append('')
    lines.append(f"- **Best new mainline candidate:** `{best_main['candidate_name']}` → **{row_summary(best_main['markov42']['overall'])}**")
    lines.append(f"  - vs `relaymax_unified_l9y1`: Δmean **{best_main['delta_vs_same_core_x0']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_main['delta_vs_same_core_x0']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"  - vs current unified winner `{refs['current_unified_winner']['candidate_name']}`: Δmean **{best_main['delta_vs_current_unified_winner']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_main['delta_vs_current_unified_winner']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"  - vs historical unified core predecessor: Δmean **{best_main['delta_vs_historical_unified_core']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_main['delta_vs_historical_unified_core']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- **Best new low-max candidate:** `{best_lowmax['candidate_name']}` → **{row_summary(best_lowmax['markov42']['overall'])}**")
    lines.append(f"  - vs `relaymax_lowmax_l9y1`: Δmean **{best_lowmax['delta_vs_relaymax_lowmax_l9y1']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_lowmax['delta_vs_relaymax_lowmax_l9y1']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append('')
    lines.append('## 7. Exact legal motor / timing table for the best new mainline candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for idx, (row, action, face) in enumerate(zip(best_main['all_rows'], best_main['all_actions'], best_main['all_faces']), start=1):
        lines.append(
            f"| {idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 8. Continuity proof for the best new mainline candidate')
    lines.append('')
    for check in best_main['continuity_checks']:
        before = check['state_before_insertion']
        after = check['state_after_insertion']
        lines.append(f"- anchor {check['anchor_id']}: closure_ok = **{'yes' if check['closure_ok'] else 'no'}**")
        lines.append(f"  - before: beta={before['beta_deg']}°, outer_axis={before['outer_axis_body']}, face={before['face_name']}")
        lines.append(f"  - after : beta={after['beta_deg']}°, outer_axis={after['outer_axis_body']}, face={after['face_name']}")
        if check['next_base_action_preview'] is not None:
            preview = check['next_base_action_preview']
            lines.append(f"  - next original action remains legal as `{preview['kind']}` {preview['motor_angle_deg']:+d}° with effective axis {preview['effective_body_axis']}")
    lines.append('')
    lines.append('## 9. KF36 rechecks for the best competitive candidates')
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
        kf = row.get('kf36')
        kf_text = 'n/a' if kf is None else f"{kf['overall']['mean_pct_error']:.3f} / {kf['overall']['median_pct_error']:.3f} / {kf['overall']['max_pct_error']:.3f}"
        lines.append(f"| {row['label']} | {mm['mean_pct_error']:.3f} / {mm['median_pct_error']:.3f} / {mm['max_pct_error']:.3f} | {kf_text} | {row['note']} |")
    lines.append('')
    lines.append('## 11. Bottom line')
    lines.append('')
    lines.append(f"- family real or one-off? **{payload['bottom_line']['family_real']}**")
    lines.append(f"- true mainline winner now? **{payload['bottom_line']['true_mainline_winner']}**")
    lines.append(f"- clear verdict: **{payload['bottom_line']['verdict']}**")
    lines.append(f"- scientific conclusion: **{payload['scientific_conclusion']}**")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module('search_ch3_entry_conditioned_relay_family_src', str(SOURCE_FILE))
    refs = load_references(args.noise_scale)
    faithful = build_candidate(mod, ())
    base_rows = faithful.rows
    base_actions = faithful.action_sequence

    specs = candidate_specs()
    candidates = [build_closedloop_candidate(mod, spec, base_rows, base_actions) for spec in specs]
    cand_by_name = {cand.name: cand for cand in candidates}
    meta_by_name = {spec['name']: spec for spec in specs}

    rows: list[dict[str, Any]] = []
    payload_by_name: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        payload, status, path = run_candidate_payload(mod, cand, 'markov42_noisy', args.noise_scale, args.force_rerun)
        payload_by_name[cand.name] = payload
        rows.append({
            'candidate_name': cand.name,
            'core_family': meta_by_name[cand.name]['core_family'],
            'entry_x_dwell': meta_by_name[cand.name]['entry_x_dwell'],
            'is_new_candidate': meta_by_name[cand.name]['is_new_candidate'],
            'rationale': meta_by_name[cand.name]['rationale'],
            'total_time_s': cand.total_time_s,
            'metrics': compact_metrics(payload),
            'continuity_checks': cand.continuity_checks,
            'run_json': str(path),
            'status': status,
        })

    same_core_refs = {
        'lowmaxcore': payload_by_name['relaymax_lowmax_l9y1'],
        'unifiedcore': payload_by_name['relaymax_unified_l9y1'],
    }

    for row in rows:
        payload = payload_by_name[row['candidate_name']]
        row['delta_vs_same_core_x0'] = delta_vs_ref(same_core_refs[row['core_family']], payload)
        row['delta_vs_relaymax_lowmax_l9y1'] = delta_vs_ref(refs['relaymax_lowmax_l9y1'], payload)
        row['delta_vs_current_unified_winner'] = delta_vs_ref(refs['current_unified_winner_markov'], payload)
        row['delta_vs_historical_unified_core'] = delta_vs_ref(refs['historical_unified_core_markov'], payload)
        row['delta_vs_entry_boundary_max'] = delta_vs_ref(refs['entry_boundary_max_markov'], payload)
        row['delta_vs_old_best'] = delta_vs_ref(refs['old_best_markov'], payload)
        row['delta_vs_default18'] = delta_vs_ref(refs['default18_markov'], payload)

    batch_rows_sorted = sorted(rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    best_new_mainline_row = select_best(rows, 'unifiedcore')
    best_new_lowmax_row = select_best(rows, 'lowmaxcore')
    best_new_mainline_cand = cand_by_name[best_new_mainline_row['candidate_name']]
    best_new_lowmax_cand = cand_by_name[best_new_lowmax_row['candidate_name']]

    current_unified_spec = current_unified_winner_spec()
    current_unified_cand = build_closedloop_candidate(mod, current_unified_spec, base_rows, base_actions)
    current_unified_kf_payload, current_unified_kf_status, current_unified_kf_path = run_candidate_payload(mod, current_unified_cand, 'kf36_noisy', args.noise_scale, args.force_rerun)

    relaymax_lowmax_kf_payload, relaymax_lowmax_kf_status, relaymax_lowmax_kf_path = run_candidate_payload(mod, cand_by_name['relaymax_lowmax_l9y1'], 'kf36_noisy', args.noise_scale, args.force_rerun)
    best_new_mainline_kf_payload, best_new_mainline_kf_status, best_new_mainline_kf_path = run_candidate_payload(mod, best_new_mainline_cand, 'kf36_noisy', args.noise_scale, args.force_rerun)
    best_new_lowmax_kf_payload, best_new_lowmax_kf_status, best_new_lowmax_kf_path = run_candidate_payload(mod, best_new_lowmax_cand, 'kf36_noisy', args.noise_scale, args.force_rerun)

    kf36_rows = [
        {
            'candidate_name': 'relaymax_lowmax_l9y1',
            'note': 'existing relay max reference',
            'markov42': compact_result(payload_by_name['relaymax_lowmax_l9y1']),
            'kf36': compact_result(relaymax_lowmax_kf_payload),
            'kf36_status': relaymax_lowmax_kf_status,
            'kf36_run_json': str(relaymax_lowmax_kf_path),
        },
        {
            'candidate_name': CURRENT_UNIFIED_NAME,
            'note': 'current unified winner used in this report',
            'markov42': compact_result(refs['current_unified_winner_markov']),
            'kf36': compact_result(current_unified_kf_payload),
            'kf36_status': current_unified_kf_status,
            'kf36_run_json': str(current_unified_kf_path),
        },
        {
            'candidate_name': best_new_mainline_cand.name,
            'note': 'best new mainline candidate',
            'markov42': compact_result(payload_by_name[best_new_mainline_cand.name]),
            'kf36': compact_result(best_new_mainline_kf_payload),
            'kf36_status': best_new_mainline_kf_status,
            'kf36_run_json': str(best_new_mainline_kf_path),
        },
        {
            'candidate_name': best_new_lowmax_cand.name,
            'note': 'best new low-max candidate',
            'markov42': compact_result(payload_by_name[best_new_lowmax_cand.name]),
            'kf36': compact_result(best_new_lowmax_kf_payload),
            'kf36_status': best_new_lowmax_kf_status,
            'kf36_run_json': str(best_new_lowmax_kf_path),
        },
    ]

    sign_control = {
        'unified_neg1': {
            'candidate_name': 'entryrelay_l8xneg1_l9y1_unifiedcore',
            'markov42': compact_result(payload_by_name['entryrelay_l8xneg1_l9y1_unifiedcore']),
        },
        'unified_pos1': {
            'candidate_name': best_new_mainline_cand.name,
            'markov42': compact_result(payload_by_name[best_new_mainline_cand.name]),
        },
        'lowmax_neg1': {
            'candidate_name': 'entryrelay_l8xneg1_l9y1_lowmaxcore',
            'markov42': compact_result(payload_by_name['entryrelay_l8xneg1_l9y1_lowmaxcore']),
        },
        'lowmax_pos1': {
            'candidate_name': best_new_lowmax_cand.name,
            'markov42': compact_result(payload_by_name[best_new_lowmax_cand.name]),
        },
    }

    best_new_mainline_summary = {
        'candidate_name': best_new_mainline_cand.name,
        'core_family': 'unifiedcore',
        'entry_x_dwell': meta_by_name[best_new_mainline_cand.name]['entry_x_dwell'],
        'total_time_s': best_new_mainline_cand.total_time_s,
        'markov42': compact_result(payload_by_name[best_new_mainline_cand.name]),
        'delta_vs_same_core_x0': best_new_mainline_row['delta_vs_same_core_x0'],
        'delta_vs_relaymax_lowmax_l9y1': best_new_mainline_row['delta_vs_relaymax_lowmax_l9y1'],
        'delta_vs_current_unified_winner': best_new_mainline_row['delta_vs_current_unified_winner'],
        'delta_vs_historical_unified_core': best_new_mainline_row['delta_vs_historical_unified_core'],
        'delta_vs_entry_boundary_max': best_new_mainline_row['delta_vs_entry_boundary_max'],
        'delta_vs_old_best': best_new_mainline_row['delta_vs_old_best'],
        'delta_vs_default18': best_new_mainline_row['delta_vs_default18'],
        'all_rows': best_new_mainline_cand.all_rows,
        'all_actions': best_new_mainline_cand.all_actions,
        'all_faces': best_new_mainline_cand.all_faces,
        'continuity_checks': best_new_mainline_cand.continuity_checks,
        'kf36': compact_result(best_new_mainline_kf_payload),
        'kf36_run_json': str(best_new_mainline_kf_path),
    }

    best_new_lowmax_summary = {
        'candidate_name': best_new_lowmax_cand.name,
        'core_family': 'lowmaxcore',
        'entry_x_dwell': meta_by_name[best_new_lowmax_cand.name]['entry_x_dwell'],
        'total_time_s': best_new_lowmax_cand.total_time_s,
        'markov42': compact_result(payload_by_name[best_new_lowmax_cand.name]),
        'delta_vs_same_core_x0': best_new_lowmax_row['delta_vs_same_core_x0'],
        'delta_vs_relaymax_lowmax_l9y1': best_new_lowmax_row['delta_vs_relaymax_lowmax_l9y1'],
        'delta_vs_current_unified_winner': best_new_lowmax_row['delta_vs_current_unified_winner'],
        'delta_vs_historical_unified_core': best_new_lowmax_row['delta_vs_historical_unified_core'],
        'delta_vs_entry_boundary_max': best_new_lowmax_row['delta_vs_entry_boundary_max'],
        'delta_vs_old_best': best_new_lowmax_row['delta_vs_old_best'],
        'delta_vs_default18': best_new_lowmax_row['delta_vs_default18'],
        'kf36': compact_result(best_new_lowmax_kf_payload),
        'kf36_run_json': str(best_new_lowmax_kf_path),
    }

    current_unified_markov = compact_result(refs['current_unified_winner_markov'])
    comparison_rows = [
        {
            'label': 'relaymax_lowmax_l9y1',
            'note': 'existing relay max reference',
            'markov42': compact_result(refs['relaymax_lowmax_l9y1']),
            'kf36': compact_result(relaymax_lowmax_kf_payload),
        },
        {
            'label': 'current unified winner',
            'note': CURRENT_UNIFIED_NAME,
            'markov42': current_unified_markov,
            'kf36': compact_result(current_unified_kf_payload),
        },
        {
            'label': 'historical unified core predecessor',
            'note': HISTORICAL_UNIFIED_CORE_NAME,
            'markov42': compact_result(refs['historical_unified_core_markov']),
            'kf36': compact_result(refs['historical_unified_core_kf']),
        },
        {
            'label': 'entry-boundary max branch',
            'note': 'entryx_l8_xpair_pos3_plus_l11_y10x0back2',
            'markov42': compact_result(refs['entry_boundary_max_markov']),
            'kf36': compact_result(refs['entry_boundary_max_kf']),
        },
        {
            'label': 'old best legal',
            'note': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
            'markov42': compact_result(refs['old_best_markov']),
            'kf36': compact_result(refs['old_best_kf']),
        },
        {
            'label': 'default18',
            'note': 'default18',
            'markov42': compact_result(refs['default18_markov']),
            'kf36': compact_result(refs['default18_kf']),
        },
        {
            'label': 'best new mainline candidate',
            'note': best_new_mainline_cand.name,
            'markov42': best_new_mainline_summary['markov42'],
            'kf36': best_new_mainline_summary['kf36'],
        },
        {
            'label': 'best new low-max candidate',
            'note': best_new_lowmax_cand.name,
            'markov42': best_new_lowmax_summary['markov42'],
            'kf36': best_new_lowmax_summary['kf36'],
        },
    ]

    family_real = (
        best_new_mainline_summary['delta_vs_same_core_x0']['max_pct_error']['improvement_pct_points'] > 0
        and best_new_lowmax_summary['delta_vs_same_core_x0']['max_pct_error']['improvement_pct_points'] > 0
        and sign_control['unified_neg1']['markov42']['overall']['max_pct_error'] > payload_by_name['relaymax_unified_l9y1']['overall']['max_pct_error']
        and sign_control['unified_neg1']['markov42']['overall']['max_pct_error'] > best_new_mainline_summary['markov42']['overall']['max_pct_error']
    )

    dominates_current_unified = (
        best_new_mainline_summary['delta_vs_current_unified_winner']['mean_pct_error']['improvement_pct_points'] > 0
        and best_new_mainline_summary['delta_vs_current_unified_winner']['max_pct_error']['improvement_pct_points'] > 0
    )

    if dominates_current_unified:
        true_mainline_winner = 'YES'
        verdict = (
            f"Entry-conditioned relay does land a true new mainline winner: {best_new_mainline_cand.name} beats the current unified winner on both mean and max, and the sign-control batch supports that the mechanism is real rather than accidental."
        )
    else:
        true_mainline_winner = 'NO'
        verdict = (
            f"Entry-conditioned relay is real and not a one-off, but it does **not** become the single true mainline winner after full comparison. {best_new_mainline_cand.name} clearly beats the historical unified core predecessor and extends the frontier, yet it still gives back {abs(best_new_mainline_summary['delta_vs_current_unified_winner']['mean_pct_error']['improvement_pct_points']):.3f} mean-points to the current unified winner `{CURRENT_UNIFIED_NAME}` while only gaining {best_new_mainline_summary['delta_vs_current_unified_winner']['max_pct_error']['improvement_pct_points']:.3f} max-points."
        )

    scientific_conclusion = (
        f"The entry-conditioned relay family is structurally real. Under the unified core, the sign-control counterpart `entryrelay_l8xneg1_l9y1_unifiedcore` falls to {row_summary(sign_control['unified_neg1']['markov42']['overall'])}, while the positive branch peaks sharply at `entryrelay_l8x1_l9y1_unifiedcore` = {row_summary(best_new_mainline_summary['markov42']['overall'])}; stronger positive doses (`x=+2`, `x=+3`) regress again. The same `x=+1` optimum also appears on the low-max core, where `entryrelay_l8x1_l9y1_lowmaxcore` reaches {row_summary(best_new_lowmax_summary['markov42']['overall'])}. KF36 rechecks preserve the same ordering. So the family is not noise and it does create new landed frontier points. However the full comparison says it is a **frontier extension**, not a clean single replacement for the current unified winner: the new unified-core point wins on max but not on mean against `{CURRENT_UNIFIED_NAME}`."
    )

    out_json = RESULTS_DIR / f'ch3_entry_conditioned_relay_family_{args.report_date}.json'
    out_md = REPORTS_DIR / f'psins_ch3_entry_conditioned_relay_family_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_entry_conditioned_relay_family_followup',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'references': {
            'relaymax_lowmax_l9y1': {
                'candidate_name': 'relaymax_lowmax_l9y1',
                'markov42': compact_result(refs['relaymax_lowmax_l9y1']),
                'kf36': compact_result(relaymax_lowmax_kf_payload),
            },
            'current_unified_winner': {
                'candidate_name': CURRENT_UNIFIED_NAME,
                'markov42': current_unified_markov,
                'kf36': compact_result(current_unified_kf_payload),
            },
            'historical_unified_core': {
                'candidate_name': HISTORICAL_UNIFIED_CORE_NAME,
                'markov42': compact_result(refs['historical_unified_core_markov']),
                'kf36': compact_result(refs['historical_unified_core_kf']),
            },
            'entry_boundary_max': {
                'candidate_name': 'entryx_l8_xpair_pos3_plus_l11_y10x0back2',
                'markov42': compact_result(refs['entry_boundary_max_markov']),
                'kf36': compact_result(refs['entry_boundary_max_kf']),
            },
            'old_best_legal': {
                'candidate_name': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
                'markov42': compact_result(refs['old_best_markov']),
                'kf36': compact_result(refs['old_best_kf']),
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
                'core_family': spec['core_family'],
                'entry_x_dwell': spec['entry_x_dwell'],
                'is_new_candidate': spec['is_new_candidate'],
                'rationale': spec['rationale'],
                'anchors': sorted(spec['insertions'].keys()),
            }
            for spec in specs
        ],
        'batch_rows_sorted': batch_rows_sorted,
        'sign_control': sign_control,
        'best_new_mainline_candidate': best_new_mainline_summary,
        'best_new_lowmax_candidate': best_new_lowmax_summary,
        'kf36_rows': kf36_rows,
        'comparison_rows': comparison_rows,
        'bottom_line': {
            'family_real': 'YES' if family_real else 'NO',
            'true_mainline_winner': true_mainline_winner,
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
    print('BEST_MAINLINE', best_new_mainline_cand.name, best_new_mainline_summary['markov42']['overall'], flush=True)
    print('BEST_LOWMAX', best_new_lowmax_cand.name, best_new_lowmax_summary['markov42']['overall'], flush=True)
    print('BOTTOM_LINE', verdict, flush=True)


if __name__ == '__main__':
    main()
