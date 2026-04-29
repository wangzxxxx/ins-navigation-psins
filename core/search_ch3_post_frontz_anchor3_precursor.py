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
HISTORICAL_EARLYGW_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_earlygw_l3_bfly_same_d8_shared_noise0p08_param_errors.json'
OLD_BEST_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
OLD_BEST_KF = RESULTS_DIR / 'KF36_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
FAITHFUL_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
FAITHFUL_KF = RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json'
DEFAULT18_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json'
DEFAULT18_KF = RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json'

CURRENT_MAINLINE_NAME = 'zseed_l5_neg6_plus_relaymax_unified_l9y2p5'
ENTRY_FRONTIER_NAME = 'entryrelay_l8x1_l9y1_unifiedcore'
HISTORICAL_EARLYGW_NAME = 'earlygw_l3_bfly_same_d8'

HYPOTHESES = [
    {
        'id': 'H3',
        'family': 'anchor3_prez_diagonal_precursor',
        'summary': 'Open a mixed-beta diagonal butterfly in the anchor3→4 handoff region before any front-z outer rotation, while keeping the current anchor5 mainline and late relay core untouched.',
        'selected': True,
        'tested': True,
    },
    {
        'id': 'H4',
        'family': 'anchor2_3_bookend_boundary_pair',
        'summary': 'Split a tiny boundary-control pair across anchors2 and3 before the front corridor. Left unspent here because it is a weaker hybrid of the already-weak x-boundary seed and the now-directly tested anchor3 precursor family.',
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


def diag_butterfly(anchor: int, open_angle_deg: int, dwell1: float, dwell2: float, first_sign: int, second_sign: int, label: str, cross_hold_s: float = 3.0, edge_hold_s: float = 3.0) -> dict[int, list[StepSpec]]:
    close_angle_deg = open_angle_deg
    return {
        anchor: [
            StepSpec(kind='inner', angle_deg=open_angle_deg, rotation_time_s=abs(open_angle_deg) / 90.0 * 5.0, pre_static_s=0.0, post_static_s=edge_hold_s, segment_role='motif_diag_open1', label=f'{label}_open1'),
            StepSpec(kind='outer', angle_deg=90 * first_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell1), segment_role='motif_diag1_sweep', label=f'{label}_diag1_sweep'),
            StepSpec(kind='outer', angle_deg=-90 * first_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell1), segment_role='motif_diag1_return', label=f'{label}_diag1_return'),
            StepSpec(kind='inner', angle_deg=-2 * open_angle_deg, rotation_time_s=abs(2 * open_angle_deg) / 90.0 * 5.0, pre_static_s=0.0, post_static_s=cross_hold_s, segment_role='motif_diag_cross', label=f'{label}_cross'),
            StepSpec(kind='outer', angle_deg=90 * second_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell2), segment_role='motif_diag2_sweep', label=f'{label}_diag2_sweep'),
            StepSpec(kind='outer', angle_deg=-90 * second_sign, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell2), segment_role='motif_diag2_return', label=f'{label}_diag2_return'),
            StepSpec(kind='inner', angle_deg=close_angle_deg, rotation_time_s=abs(close_angle_deg) / 90.0 * 5.0, pre_static_s=0.0, post_static_s=edge_hold_s, segment_role='motif_diag_close2', label=f'{label}_close2'),
        ]
    }


def merge_insertions(*dicts: dict[int, list[StepSpec]]) -> dict[int, list[StepSpec]]:
    out: dict[int, list[StepSpec]] = {}
    for d in dicts:
        for k, v in d.items():
            out.setdefault(k, []).extend(v)
    return out


def l9_ypair_neg(dwell_s: float, label: str) -> dict[int, list[StepSpec]]:
    return {9: closed_pair('inner', -90, float(dwell_s), label)}


def l10_core(y_dwell_s: float = 1.0) -> dict[int, list[StepSpec]]:
    return {10: closed_pair('outer', -90, 5.0, 'l10_zpair_neg5') + closed_pair('inner', -90, float(y_dwell_s), f'l10_ypair_neg{dose_tag(y_dwell_s)}')}


def l11_core(back_s: float = 2.0) -> dict[int, list[StepSpec]]:
    return {11: xpair_outerhold(10.0, 'l11_xpair_outerhold') + zquad(10.0, 0.0, float(back_s), f'l11_zquad_y10x0back{dose_tag(back_s)}')}


def relay_core(l9_dwell_s: float = 2.5, l10_y_dwell_s: float = 1.0, l11_back_s: float = 2.0) -> dict[int, list[StepSpec]]:
    return merge_insertions(
        l9_ypair_neg(l9_dwell_s, f'l9_ypair_neg{dose_tag(l9_dwell_s)}'),
        l10_core(l10_y_dwell_s),
        l11_core(l11_back_s),
    )


def anchor5_zseed(dose_abs_s: float, sign: str, label: str) -> dict[int, list[StepSpec]]:
    angle = -90 if sign == 'neg' else +90
    return {5: closed_pair('outer', angle, float(dose_abs_s), label)}


def make_spec(name: str, rationale: str, insertions: dict[int, list[StepSpec]], *, dose_s: float, chirality: str) -> dict[str, Any]:
    return {
        'name': name,
        'family': 'anchor3_prez_diagonal_precursor',
        'hypothesis_id': 'H3',
        'rationale': rationale,
        'dose_s': dose_s,
        'chirality': chirality,
        'insertions': insertions,
    }


def candidate_specs() -> list[dict[str, Any]]:
    mainline = merge_insertions(anchor5_zseed(6.0, 'neg', 'l5_zseed_neg6'), relay_core(2.5))
    return [
        make_spec(
            'prezdiag_l3_bfly_same_d2_plus_zseed_l5_neg6_plus_relaymax_unified_l9y2p5',
            'Minimal anchor3 pre-front mixed-beta butterfly: same-sign sweeps with only 2 s diagonal dwell, added in front of the current anchor5 mainline to test whether the family only fails at large dose.',
            merge_insertions(diag_butterfly(3, -45, 2.0, 2.0, +1, +1, 'l3_prezdiag_same_d2'), mainline),
            dose_s=2.0,
            chirality='same',
        ),
        make_spec(
            'prezdiag_l3_bfly_same_d4_plus_zseed_l5_neg6_plus_relaymax_unified_l9y2p5',
            'Dose-up anchor3 pre-front butterfly: same-sign diagonal sweeps with 4 s dwell before the unchanged current mainline.',
            merge_insertions(diag_butterfly(3, -45, 4.0, 4.0, +1, +1, 'l3_prezdiag_same_d4'), mainline),
            dose_s=4.0,
            chirality='same',
        ),
        make_spec(
            'prezdiag_l3_bfly_flip_d4_plus_zseed_l5_neg6_plus_relaymax_unified_l9y2p5',
            'Chirality control for the anchor3 pre-front butterfly: keep 4 s dwell but flip the second diagonal sweep sign while the current mainline stays fixed.',
            merge_insertions(diag_butterfly(3, -45, 4.0, 4.0, +1, -1, 'l3_prezdiag_flip_d4'), mainline),
            dose_s=4.0,
            chirality='flip',
        ),
    ]


def load_references(noise_scale: float) -> dict[str, Any]:
    return {
        'current_mainline_markov': load_json_checked(CURRENT_MAINLINE_MARKOV, noise_scale),
        'current_mainline_kf': load_json_checked(CURRENT_MAINLINE_KF, noise_scale),
        'entry_frontier_markov': load_json_checked(ENTRY_FRONTIER_MARKOV, noise_scale),
        'entry_frontier_kf': load_json_checked(ENTRY_FRONTIER_KF, noise_scale),
        'historical_earlygw_markov': load_json_checked(HISTORICAL_EARLYGW_MARKOV, noise_scale),
        'old_best_markov': load_json_checked(OLD_BEST_MARKOV, noise_scale),
        'old_best_kf': load_json_checked(OLD_BEST_KF, noise_scale),
        'faithful_markov': load_json_checked(FAITHFUL_MARKOV, noise_scale),
        'faithful_kf': load_json_checked(FAITHFUL_KF, noise_scale),
        'default18_markov': load_json_checked(DEFAULT18_MARKOV, noise_scale),
        'default18_kf': load_json_checked(DEFAULT18_KF, noise_scale),
    }


def render_report(payload: dict[str, Any]) -> str:
    best = payload['best_candidate']
    best_mean = payload['best_mean_candidate']
    hist = payload['historical_reused_control']
    lines: list[str] = []
    lines.append('# Chapter-3 post-frontz anchor3 precursor microbatch')
    lines.append('')
    lines.append('## 1. Search intent')
    lines.append('')
    lines.append('- This microbatch resumes the post-frontz hidden-family task **after** anchor4 front-z and anchor2 x-boundary chain variants were already ruled out as usable new winners.')
    lines.append('- Only **one genuinely fresh family** was spent here: the anchor3 pre-front mixed-beta diagonal precursor (`H3`).')
    lines.append('- Hard rule kept: do **not** reopen anchor5 ridge tuning or anchor2/anchor4 chain variants; keep the current winner `zseed_l5_neg6_plus_relaymax_unified_l9y2p5` untouched downstream and only perturb the early 3→4 handoff.')
    lines.append('')
    lines.append('## 2. Reused prior evidence before spending new budget')
    lines.append('')
    lines.append(f"- Historical early anchor3 control already existed from `psins_ch3_post_entry_midgateway_probe_2026-04-01.md`: `{HISTORICAL_EARLYGW_NAME}` = **{row_summary(hist['markov42']['overall'])}**.")
    lines.append('- That old point was catastrophically bad, but it was measured on an earlier backbone rather than on the current anchor5 mainline. So this pass only spent a **minimal current-mainline microbatch** at much smaller doses to decide whether the family still deserves frontier budget.')
    lines.append('')
    lines.append('## 3. Hypotheses handled in this resume')
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
    lines.append('## 4. Markov42 landed batch')
    lines.append('')
    lines.append('| rank | candidate | dose | chirality | total_s | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Ka2_z | Δmean vs current mainline | Δmax vs current mainline | Δmean vs entry frontier | Δmax vs entry frontier |')
    lines.append('|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(payload['rows_sorted'], start=1):
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        dc = row['delta_vs_current_mainline']
        de = row['delta_vs_entry_frontier']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['dose_s']:.1f} | {row['chirality']} | {row['total_time_s']:.0f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {dc['mean_pct_error']['improvement_pct_points']:+.3f} | {dc['max_pct_error']['improvement_pct_points']:+.3f} | {de['mean_pct_error']['improvement_pct_points']:+.3f} | {de['max_pct_error']['improvement_pct_points']:+.3f} |"
        )
    lines.append('')
    lines.append('## 5. Best landed readout')
    lines.append('')
    lines.append(f"- **Best max-priority candidate:** `{best['candidate_name']}` → **{row_summary(best['markov42']['overall'])}**")
    lines.append(f"  - vs `{CURRENT_MAINLINE_NAME}`: Δmean **{best['delta_vs_current_mainline']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian **{best['delta_vs_current_mainline']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"  - vs `{ENTRY_FRONTIER_NAME}`: Δmean **{best['delta_vs_entry_frontier']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian **{best['delta_vs_entry_frontier']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_entry_frontier']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- **Best mean candidate in this microbatch:** `{best_mean['candidate_name']}` → **{row_summary(best_mean['markov42']['overall'])}**")
    lines.append('')
    lines.append('## 6. Exact legal motor / timing table for the best candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for idx, (row, action, face) in enumerate(zip(best['all_rows'], best['all_actions'], best['all_faces']), start=1):
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
        preview = check['next_base_action_preview']
        if preview is not None:
            lines.append(f"  - next original action remains legal as `{preview['kind']}` {preview['motor_angle_deg']:+d}° with effective axis {preview['effective_body_axis']}")
    lines.append('')
    lines.append('## 8. KF36 recheck gate')
    lines.append('')
    lines.append(f"- triggered: **{payload['kf36_recheck']['triggered']}**")
    lines.append(f"- reason: **{payload['kf36_recheck']['reason']}**")
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
    lines.append(f"- **Did this anchor3 precursor microbatch find a materially stronger direction?** **{payload['bottom_line']['found_stronger_direction']}**")
    lines.append(f"- **Best microbatch landing:** **{payload['bottom_line']['best_signal']}**")
    lines.append(f"- **Can the pending hidden-family task be closed on current evidence?** **{payload['bottom_line']['can_close_pending']}**")
    lines.append(f"- **Scientific conclusion:** {payload['bottom_line']['scientific_conclusion']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    refs = load_references(args.noise_scale)
    mod = load_module('psins_ch3_post_frontz_anchor3_precursor', str(SOURCE_FILE))
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
            'dose_s': spec['dose_s'],
            'chirality': spec['chirality'],
            'total_time_s': cand.total_time_s,
            'metrics': compact_metrics(payload),
            'continuity_checks': cand.continuity_checks,
            'run_json': str(path),
            'status': status,
            'delta_vs_current_mainline': delta_vs_ref(refs['current_mainline_markov'], payload),
            'delta_vs_entry_frontier': delta_vs_ref(refs['entry_frontier_markov'], payload),
            'delta_vs_historical_earlygw': delta_vs_ref(refs['historical_earlygw_markov'], payload),
            'delta_vs_old_best': delta_vs_ref(refs['old_best_markov'], payload),
            'delta_vs_faithful12': delta_vs_ref(refs['faithful_markov'], payload),
            'delta_vs_default18': delta_vs_ref(refs['default18_markov'], payload),
        })

    rows_sorted = sorted(rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    best_row = rows_sorted[0]
    best_mean_row = min(rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))

    best_candidate = cand_by_name[best_row['candidate_name']]
    best_payload = payload_by_name[best_candidate.name]
    best_mean_candidate = cand_by_name[best_mean_row['candidate_name']]
    best_mean_payload = payload_by_name[best_mean_candidate.name]

    frontier_signal = [
        row for row in rows
        if row['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points'] > 0.0
        and row['delta_vs_current_mainline']['mean_pct_error']['improvement_pct_points'] > -0.08
    ]
    decisive_signal = [
        row for row in rows
        if row['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points'] > 0.0
        and row['delta_vs_current_mainline']['mean_pct_error']['improvement_pct_points'] > 0.0
    ]

    competitive = (
        best_payload['overall']['max_pct_error'] < 99.595
        and best_payload['overall']['mean_pct_error'] < 9.45
        and (
            best_row['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points'] > -0.015
            or best_row['delta_vs_entry_frontier']['max_pct_error']['improvement_pct_points'] > -0.015
            or best_row['delta_vs_current_mainline']['mean_pct_error']['improvement_pct_points'] > 0.0
        )
    )

    kf36_rows: list[dict[str, Any]] = []
    if competitive:
        kf_payload, kf_status, kf_path = run_candidate_payload(mod, best_candidate, 'kf36_noisy', args.noise_scale, args.force_rerun)
        kf36_rows.append({
            'candidate_name': best_candidate.name,
            'note': 'best frontier-adjacent candidate in this microbatch',
            'markov42': compact_result(best_payload),
            'kf36': compact_result(kf_payload),
            'kf36_status': kf_status,
            'kf36_run_json': str(kf_path),
        })
        kf_reason = f"Triggered because {best_candidate.name} landed close enough to the current frontiers at {row_summary(best_payload['overall'])}."
        best_kf_summary = compact_result(kf_payload)
    else:
        kf_reason = (
            f"Not triggered because the best landing {best_candidate.name} = {row_summary(best_payload['overall'])} still sits too far from both `{CURRENT_MAINLINE_NAME}` and `{ENTRY_FRONTIER_NAME}` to count as genuinely competitive."
        )
        best_kf_summary = None

    best_summary = {
        'candidate_name': best_candidate.name,
        'family': best_row['family'],
        'hypothesis_id': best_row['hypothesis_id'],
        'total_time_s': best_candidate.total_time_s,
        'markov42': compact_result(best_payload),
        'kf36': best_kf_summary,
        'delta_vs_current_mainline': best_row['delta_vs_current_mainline'],
        'delta_vs_entry_frontier': best_row['delta_vs_entry_frontier'],
        'delta_vs_historical_earlygw': best_row['delta_vs_historical_earlygw'],
        'all_rows': best_candidate.all_rows,
        'all_actions': best_candidate.all_actions,
        'all_faces': best_candidate.all_faces,
        'continuity_checks': best_candidate.continuity_checks,
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

    if decisive_signal:
        found_text = 'YES'
        can_close = 'NO'
        scientific_conclusion = (
            'Yes — the anchor3 precursor family unexpectedly found a clean stronger direction. '
            f"At least one candidate beat `{CURRENT_MAINLINE_NAME}` on both mean and max; the strongest landing was `{best_candidate.name}` = {row_summary(best_payload['overall'])}."
        )
    elif frontier_signal:
        found_text = 'MIXED'
        can_close = 'NO'
        signal_row = sorted(frontier_signal, key=lambda x: (x['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points'] * -1.0, x['metrics']['overall']['mean_pct_error']))[0]
        scientific_conclusion = (
            'Mixed but real signal — the anchor3 precursor family did not take over cleanly, but it did surface a frontier-adjacent point worth another pass. '
            f"The clearest signal was `{signal_row['candidate_name']}` with Δmean {signal_row['delta_vs_current_mainline']['mean_pct_error']['improvement_pct_points']:+.3f} and Δmax {signal_row['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points']:+.3f} versus `{CURRENT_MAINLINE_NAME}`."
        )
    else:
        found_text = 'NO'
        can_close = 'YES'
        scientific_conclusion = (
            'No compelling next family surfaced. The direct current-mainline anchor3 precursor microbatch stayed noncompetitive even after shrinking the old disastrous d8 control down to d2/d4. '
            f"The best landing was `{best_candidate.name}` = {row_summary(best_payload['overall'])}, which moved against `{CURRENT_MAINLINE_NAME}` by Δmean {best_row['delta_vs_current_mainline']['mean_pct_error']['improvement_pct_points']:+.3f}, Δmedian {best_row['delta_vs_current_mainline']['median_pct_error']['improvement_pct_points']:+.3f}, and Δmax {best_row['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points']:+.3f}. "
            f"Relative to the entry-conditioned frontier `{ENTRY_FRONTIER_NAME}`, it moved by Δmean {best_row['delta_vs_entry_frontier']['mean_pct_error']['improvement_pct_points']:+.3f}, Δmedian {best_row['delta_vs_entry_frontier']['median_pct_error']['improvement_pct_points']:+.3f}, and Δmax {best_row['delta_vs_entry_frontier']['max_pct_error']['improvement_pct_points']:+.3f}. "
            f"It does improve sharply over the old coarse control `{HISTORICAL_EARLYGW_NAME}` = {row_summary(refs['historical_earlygw_markov']['overall'])}, but only from catastrophic to merely clearly worse, not into the usable frontier band. "
            'Given that the remaining H4 anchor2/3 bookend idea is just a weaker hybrid of the already-underwhelming anchor2 x-boundary seed and the now-directly negative anchor3 precursor family, current evidence is strong enough to close the pending hidden-family discovery task without spending another reset-window batch.'
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
            'note': 'current max-frontier rival',
            'markov42': compact_result(refs['entry_frontier_markov']),
            'kf36': compact_result(refs['entry_frontier_kf']),
        },
        {
            'label': HISTORICAL_EARLYGW_NAME,
            'note': 'reused old anchor3 control from post-entry mid-gateway probe',
            'markov42': compact_result(refs['historical_earlygw_markov']),
            'kf36': None,
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
            'label': 'best candidate in this microbatch',
            'note': best_candidate.name,
            'markov42': best_summary['markov42'],
            'kf36': best_kf_summary,
        },
    ]

    out_json = RESULTS_DIR / f'ch3_post_frontz_anchor3_precursor_{args.report_date}.json'
    out_md = REPORTS_DIR / f'psins_ch3_post_frontz_anchor3_precursor_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_post_frontz_anchor3_precursor',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'hypotheses': HYPOTHESES,
        'selected_family_focus': ['anchor3_prez_diagonal_precursor'],
        'historical_reused_control': {
            'candidate_name': HISTORICAL_EARLYGW_NAME,
            'markov42': compact_result(refs['historical_earlygw_markov']),
            'source_report': str(REPORTS_DIR / 'psins_ch3_post_entry_midgateway_probe_2026-04-01.md'),
        },
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
            'historical_earlygw': {
                'candidate_name': HISTORICAL_EARLYGW_NAME,
                'markov42': compact_result(refs['historical_earlygw_markov']),
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
                'dose_s': spec['dose_s'],
                'chirality': spec['chirality'],
                'anchors': sorted(spec['insertions'].keys()),
            }
            for spec in specs
        ],
        'rows_sorted': rows_sorted,
        'best_candidate': best_summary,
        'best_mean_candidate': best_mean_summary,
        'kf36_recheck': {
            'triggered': competitive,
            'reason': kf_reason,
        },
        'kf36_rows': kf36_rows,
        'comparison_rows': comparison_rows,
        'bottom_line': {
            'found_stronger_direction': found_text,
            'can_close_pending': can_close,
            'best_signal': (
                f"best-max {best_candidate.name} = {row_summary(best_payload['overall'])}; "
                f"best-mean {best_mean_candidate.name} = {row_summary(best_mean_payload['overall'])}"
            ),
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
    print('BEST_CANDIDATE', best_candidate.name, best_payload['overall'], flush=True)
    print('BEST_MEAN', best_mean_candidate.name, best_mean_payload['overall'], flush=True)
    print('BOTTOM_LINE', scientific_conclusion, flush=True)


if __name__ == '__main__':
    main()
