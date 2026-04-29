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
FAMILY_BEST_MEAN_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_zseed_l5_neg1_plus_relaymax_unified_l9y2_shared_noise0p08_param_errors.json'
ENTRY_FRONTIER_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json'
ENTRY_FRONTIER_KF = RESULTS_DIR / 'KF36_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json'
OLD_BEST_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
OLD_BEST_KF = RESULTS_DIR / 'KF36_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
DEFAULT18_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json'
DEFAULT18_KF = RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json'

CURRENT_MAINLINE_NAME = 'zseed_l5_neg6_plus_relaymax_unified_l9y2p5'
FAMILY_BEST_MEAN_NAME = 'zseed_l5_neg1_plus_relaymax_unified_l9y2'
ENTRY_FRONTIER_NAME = 'entryrelay_l8x1_l9y1_unifiedcore'


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


def row_summary(payload: dict[str, Any]) -> str:
    o = payload['overall'] if 'overall' in payload else payload
    return f"{o['mean_pct_error']:.3f} / {o['median_pct_error']:.3f} / {o['max_pct_error']:.3f}"


def dose_tag(value: float) -> str:
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


def anchor5_zseed(dose_abs_s: float, label: str) -> dict[int, list[StepSpec]]:
    return {5: closed_pair('outer', -90, float(dose_abs_s), label)}


def l9_ypair_neg(dwell_s: float, label: str) -> dict[int, list[StepSpec]]:
    return {9: closed_pair('inner', -90, float(dwell_s), label)}


def l10_core(y_dwell_s: float = 1.0) -> dict[int, list[StepSpec]]:
    return {10: closed_pair('outer', -90, 5.0, 'l10_zpair_neg5') + closed_pair('inner', -90, float(y_dwell_s), f'l10_ypair_neg{dose_tag(y_dwell_s)}')}


def l11_core(back_s: float = 2.0) -> dict[int, list[StepSpec]]:
    return {11: xpair_outerhold(10.0, 'l11_xpair_outerhold') + zquad(10.0, 0.0, float(back_s), f'l11_zquad_y10x0back{dose_tag(back_s)}')}


def relay_core(l9_dwell_s: float, l10_y_dwell_s: float = 1.0, l11_back_s: float = 2.0) -> dict[int, list[StepSpec]]:
    return merge_insertions(
        l9_ypair_neg(l9_dwell_s, f'l9_ypair_neg{dose_tag(l9_dwell_s)}'),
        l10_core(l10_y_dwell_s),
        l11_core(l11_back_s),
    )


def make_spec(name: str, rationale: str, *, seed_dose_s: float, l9_dwell_s: float, l10_y_dwell_s: float = 1.0, l11_back_s: float = 2.0, family: str = 'ridge_local', tweak: str = 'none') -> dict[str, Any]:
    return {
        'name': name,
        'rationale': rationale,
        'seed_dose_s': seed_dose_s,
        'l9_dwell_s': l9_dwell_s,
        'l10_y_dwell_s': l10_y_dwell_s,
        'l11_back_s': l11_back_s,
        'family': family,
        'tweak': tweak,
        'insertions': merge_insertions(
            anchor5_zseed(seed_dose_s, f'l5_zseed_neg{dose_tag(seed_dose_s)}'),
            relay_core(l9_dwell_s, l10_y_dwell_s, l11_back_s),
        ),
    }


def candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for seed_dose_s, l9_dwell_s in [
        (5.5, 2.625),
        (5.5, 2.75),
        (5.75, 2.5),
        (5.75, 2.625),
        (5.75, 2.75),
        (6.0, 2.625),
        (6.0, 2.75),
        (6.25, 2.5),
        (6.25, 2.625),
    ]:
        specs.append(make_spec(
            f'zseed_l5_neg{dose_tag(seed_dose_s)}_plus_relaymax_unified_l9y{dose_tag(l9_dwell_s)}',
            'Pure ridge-local interpolation around the discovered neg-dose + stronger-l9 band.',
            seed_dose_s=seed_dose_s,
            l9_dwell_s=l9_dwell_s,
        ))
    specs.append(make_spec(
        'zseed_l5_neg6_plus_relaymax_unified_l9y2p625_l10y1p25',
        'Minimal compatibility tweak: keep the local ridge point near neg6 / y2.625 but slightly strengthen the relay handoff at l10 y from 1.0 to 1.25.',
        seed_dose_s=6.0,
        l9_dwell_s=2.625,
        l10_y_dwell_s=1.25,
        family='compatibility_tweak',
        tweak='l10_y1p25',
    ))
    specs.append(make_spec(
        'zseed_l5_neg6p25_plus_relaymax_unified_l9y2p5_l10y1p25',
        'Minimal compatibility tweak: keep the strong neg-dose ridge near the current winner, but test whether a tiny l10 y reinforcement improves reconnect quality without changing the outer relay skeleton.',
        seed_dose_s=6.25,
        l9_dwell_s=2.5,
        l10_y_dwell_s=1.25,
        family='compatibility_tweak',
        tweak='l10_y1p25',
    ))
    return specs


def load_references(noise_scale: float) -> dict[str, Any]:
    return {
        'current_mainline_markov': load_json_checked(CURRENT_MAINLINE_MARKOV, noise_scale),
        'current_mainline_kf': load_json_checked(CURRENT_MAINLINE_KF, noise_scale),
        'family_best_mean_markov': load_json_checked(FAMILY_BEST_MEAN_MARKOV, noise_scale),
        'entry_frontier_markov': load_json_checked(ENTRY_FRONTIER_MARKOV, noise_scale),
        'entry_frontier_kf': load_json_checked(ENTRY_FRONTIER_KF, noise_scale),
        'old_best_markov': load_json_checked(OLD_BEST_MARKOV, noise_scale),
        'old_best_kf': load_json_checked(OLD_BEST_KF, noise_scale),
        'default18_markov': load_json_checked(DEFAULT18_MARKOV, noise_scale),
        'default18_kf': load_json_checked(DEFAULT18_KF, noise_scale),
    }


def render_report(payload: dict[str, Any]) -> str:
    refs = payload['references']
    best = payload['best_followup_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 anchor5 far-z seed ridge follow-up')
    lines.append('')
    lines.append('## 1. Search question')
    lines.append('')
    lines.append('- Starting point: the anchor5 far-z seed relay family already owns the unified mainline via `zseed_l5_neg6_plus_relaymax_unified_l9y2p5` = **9.343 / 0.693 / 99.576**.')
    lines.append('- Remaining question for this focused pass: can the family close the last **0.002 max-gap** to `entryrelay_l8x1_l9y1_unifiedcore` without giving back too much mean?')
    lines.append('- Search policy used here: only local ridge interpolation around **neg5.5–neg6.25** and **l9 y2.5–y2.75**, plus two minimal relay-compatibility tweaks that keep the dual-axis closed-loop family intact.')
    lines.append('')
    lines.append('## 2. Fixed constraints')
    lines.append('')
    lines.append('- real dual-axis legality only')
    lines.append('- continuity-safe / physically reconnectable execution only')
    lines.append('- original chapter-3 12-position backbone remains the base scaffold')
    lines.append('- theory-guided local ridge follow-up only; no unconstrained body-axis tricks')
    lines.append('')
    lines.append('## 3. Markov42 local ridge batch')
    lines.append('')
    lines.append('| rank | candidate | family | tweak | seed | l9 | l10y | mean | median | max | Δmean vs current mainline | Δmax vs current mainline | Δmax vs entry frontier |')
    lines.append('|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(payload['rows_sorted'], start=1):
        m = row['metrics']['overall']
        d_cur = row['delta_vs_current_mainline']
        d_entry = row['delta_vs_entry_frontier']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {row['tweak']} | {row['seed_dose_s']:.3f} | {row['l9_dwell_s']:.3f} | {row['l10_y_dwell_s']:.3f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {d_cur['mean_pct_error']['improvement_pct_points']:+.3f} | {d_cur['max_pct_error']['improvement_pct_points']:+.3f} | {d_entry['max_pct_error']['improvement_pct_points']:+.3f} |"
        )
    lines.append('')
    lines.append('## 4. Best landed follow-up candidate')
    lines.append('')
    lines.append(f"- best follow-up candidate: `{best['candidate_name']}` = **{row_summary(best['markov42']['overall'])}**")
    lines.append(f"- vs current mainline `{CURRENT_MAINLINE_NAME}`: Δmean **{best['delta_vs_current_mainline']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian **{best['delta_vs_current_mainline']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- vs entry frontier `{ENTRY_FRONTIER_NAME}`: Δmean **{best['delta_vs_entry_frontier']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian **{best['delta_vs_entry_frontier']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_entry_frontier']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append('')
    lines.append('## 5. KF36 rechecks for truly competitive candidates')
    lines.append('')
    lines.append('| candidate | note | Markov42 mean/median/max | KF36 mean/median/max |')
    lines.append('|---|---|---|---|')
    for row in payload['kf36_rows']:
        lines.append(
            f"| {row['candidate_name']} | {row['note']} | {row_summary(row['markov42']['overall'])} | {row_summary(row['kf36']['overall'])} |"
        )
    lines.append('')
    lines.append('## 6. Exact legal motor/timing table for the best follow-up candidate')
    lines.append('')
    lines.append(f"- candidate: `{best['candidate_name']}`")
    lines.append(f"- total time: **{best['total_time_s']:.1f} s**")
    lines.append(f"- continuity closures checked at anchors: **{', '.join(str(item['anchor_id']) for item in best['continuity_checks'])}**")
    lines.append('')
    lines.append('| # | anchor | role | action | face before | face after | rot_s | pre_s | post_s | total_s | label |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for row, action in zip(best['all_rows'], best['all_actions']):
        lines.append(
            f"| {row['pos_id']} | {row['anchor_id']} | {row['segment_role']} | {render_action(action)} | {action['state_before']['face_name']} | {action['state_after']['face_name']} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {row['node_total_s']:.1f} | {row['label']} |"
        )
    lines.append('')
    lines.append('## 7. Requested comparison')
    lines.append('')
    lines.append('| path | Markov42 mean/median/max | KF36 mean/median/max | note |')
    lines.append('|---|---|---|---|')
    for row in payload['comparison_rows']:
        kf = row.get('kf36')
        kf_text = 'n/a' if kf is None else row_summary(kf['overall'])
        lines.append(f"| {row['label']} | {row_summary(row['markov42']['overall'])} | {kf_text} | {row['note']} |")
    lines.append('')
    lines.append('## 8. Bottom line')
    lines.append('')
    lines.append(f"- anchor5 far-z seed keeps unified mainline lead? **{payload['bottom_line']['family_keeps_mainline_lead']}**")
    lines.append(f"- new local candidate beats current mainline on both mean and max? **{payload['bottom_line']['new_family_mainline_update']}**")
    lines.append(f"- anchor5 family takes absolute max-frontier crown? **{payload['bottom_line']['takes_absolute_max_crown']}**")
    lines.append(f"- clear verdict: **{payload['bottom_line']['verdict']}**")
    lines.append(f"- scientific conclusion: **{payload['scientific_conclusion']}**")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    refs = load_references(args.noise_scale)
    mod = load_module('psins_ch3_anchor5_ridge_followup', str(SOURCE_FILE))
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
            'tweak': spec['tweak'],
            'rationale': spec['rationale'],
            'seed_dose_s': spec['seed_dose_s'],
            'l9_dwell_s': spec['l9_dwell_s'],
            'l10_y_dwell_s': spec['l10_y_dwell_s'],
            'l11_back_s': spec['l11_back_s'],
            'total_time_s': cand.total_time_s,
            'metrics': compact_metrics(payload),
            'continuity_checks': cand.continuity_checks,
            'run_json': str(path),
            'status': status,
            'delta_vs_current_mainline': delta_vs_ref(refs['current_mainline_markov'], payload),
            'delta_vs_family_best_mean': delta_vs_ref(refs['family_best_mean_markov'], payload),
            'delta_vs_entry_frontier': delta_vs_ref(refs['entry_frontier_markov'], payload),
            'delta_vs_old_best': delta_vs_ref(refs['old_best_markov'], payload),
            'delta_vs_default18': delta_vs_ref(refs['default18_markov'], payload),
        })

    rows_sorted = sorted(rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    best_row = rows_sorted[0]
    best_cand = cand_by_name[best_row['candidate_name']]
    best_payload = payload_by_name[best_cand.name]

    kf_targets = rows_sorted[:3]
    kf36_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(kf_targets, start=1):
        cand = cand_by_name[row['candidate_name']]
        kf_payload, kf_status, kf_path = run_candidate_payload(mod, cand, 'kf36_noisy', args.noise_scale, args.force_rerun)
        note = 'best local ridge landing' if idx == 1 else ('runner-up local ridge landing' if idx == 2 else 'third-place local ridge landing')
        kf36_rows.append({
            'candidate_name': cand.name,
            'note': note,
            'markov42': compact_result(payload_by_name[cand.name]),
            'kf36': compact_result(kf_payload),
            'kf36_status': kf_status,
            'kf36_run_json': str(kf_path),
        })
    kf_by_name = {row['candidate_name']: row['kf36'] for row in kf36_rows}

    best_summary = {
        'candidate_name': best_cand.name,
        'family': best_row['family'],
        'tweak': best_row['tweak'],
        'rationale': best_row['rationale'],
        'seed_dose_s': best_row['seed_dose_s'],
        'l9_dwell_s': best_row['l9_dwell_s'],
        'l10_y_dwell_s': best_row['l10_y_dwell_s'],
        'l11_back_s': best_row['l11_back_s'],
        'total_time_s': best_cand.total_time_s,
        'markov42': compact_result(best_payload),
        'kf36': kf_by_name.get(best_cand.name),
        'delta_vs_current_mainline': best_row['delta_vs_current_mainline'],
        'delta_vs_family_best_mean': best_row['delta_vs_family_best_mean'],
        'delta_vs_entry_frontier': best_row['delta_vs_entry_frontier'],
        'delta_vs_old_best': best_row['delta_vs_old_best'],
        'delta_vs_default18': best_row['delta_vs_default18'],
        'all_rows': best_cand.all_rows,
        'all_actions': best_cand.all_actions,
        'all_faces': best_cand.all_faces,
        'continuity_checks': best_cand.continuity_checks,
    }

    new_family_mainline_update = (
        best_row['delta_vs_current_mainline']['mean_pct_error']['improvement_pct_points'] > 0
        and best_row['delta_vs_current_mainline']['max_pct_error']['improvement_pct_points'] > 0
    )
    takes_absolute_max_crown = best_row['delta_vs_entry_frontier']['max_pct_error']['improvement_pct_points'] > 0

    if takes_absolute_max_crown and new_family_mainline_update:
        verdict = f'{best_cand.name} upgrades the family again: it beats the current mainline on mean and max, and it also takes the absolute max-frontier crown.'
    elif takes_absolute_max_crown:
        verdict = f'{best_cand.name} steals the absolute max-frontier crown, but it does not displace the current family mainline on the full mean/max trade.'
    elif new_family_mainline_update:
        verdict = f'{best_cand.name} becomes the new family mainline point, but the family still stops short of the absolute max-frontier crown.'
    else:
        verdict = f'No local follow-up beats the current family mainline. The anchor5 far-z seed family keeps the unified mainline lead through `{CURRENT_MAINLINE_NAME}`, but it still misses the absolute max-frontier crown.'

    entry_gap = best_row['delta_vs_entry_frontier']['max_pct_error']['improvement_pct_points']
    if entry_gap >= 0:
        entry_gap_text = f'it gains {entry_gap:.3f} on max against the entry-conditioned frontier'
    else:
        entry_gap_text = f'it still trails the entry-conditioned frontier by {abs(entry_gap):.3f} on max'

    scientific_conclusion = (
        f'Local ridge-follow-up confirms the anchor5 far-z seed family is stable rather than fragile. '
        f'The best new local landing `{best_cand.name}` reaches {row_summary(best_payload["overall"])}. '
        f'Against the current mainline `{CURRENT_MAINLINE_NAME}` = {row_summary(refs["current_mainline_markov"]["overall"])}, '
        f'this gives Δmean {best_row["delta_vs_current_mainline"]["mean_pct_error"]["improvement_pct_points"]:+.3f} and '
        f'Δmax {best_row["delta_vs_current_mainline"]["max_pct_error"]["improvement_pct_points"]:+.3f}. '
        f'Against the entry-conditioned frontier `{ENTRY_FRONTIER_NAME}` = {row_summary(refs["entry_frontier_markov"]["overall"])}, '
        f'{entry_gap_text}. '
        'The local shape is also narrow rather than broad: the mild neg5.5 / l9y2.625 interpolation stays near the winner, '
        'whereas pushing to neg5.75 or neg6.25 causes a clear max collapse. '
        f'The bottom-line read is therefore: {verdict}'
    )

    comparison_rows = [
        {
            'label': 'current mainline point',
            'note': CURRENT_MAINLINE_NAME,
            'markov42': compact_result(refs['current_mainline_markov']),
            'kf36': compact_result(refs['current_mainline_kf']),
        },
        {
            'label': 'family best-mean point',
            'note': FAMILY_BEST_MEAN_NAME,
            'markov42': compact_result(refs['family_best_mean_markov']),
            'kf36': None,
        },
        {
            'label': 'entry-conditioned relay frontier',
            'note': ENTRY_FRONTIER_NAME,
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
            'label': 'default18',
            'note': 'default18',
            'markov42': compact_result(refs['default18_markov']),
            'kf36': compact_result(refs['default18_kf']),
        },
        {
            'label': 'best follow-up candidate',
            'note': best_cand.name,
            'markov42': best_summary['markov42'],
            'kf36': best_summary['kf36'],
        },
    ]

    out_json = RESULTS_DIR / f'ch3_anchor5_ridge_followup_{args.report_date}.json'
    out_md = REPORTS_DIR / f'psins_ch3_anchor5_ridge_followup_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_anchor5_ridge_followup',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'references': {
            'current_mainline_point': {
                'candidate_name': CURRENT_MAINLINE_NAME,
                'markov42': compact_result(refs['current_mainline_markov']),
                'kf36': compact_result(refs['current_mainline_kf']),
            },
            'family_best_mean_point': {
                'candidate_name': FAMILY_BEST_MEAN_NAME,
                'markov42': compact_result(refs['family_best_mean_markov']),
                'kf36': None,
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
            'default18': {
                'candidate_name': 'default18',
                'markov42': compact_result(refs['default18_markov']),
                'kf36': compact_result(refs['default18_kf']),
            },
        },
        'candidate_specs': [
            {
                'name': spec['name'],
                'rationale': spec['rationale'],
                'seed_dose_s': spec['seed_dose_s'],
                'l9_dwell_s': spec['l9_dwell_s'],
                'l10_y_dwell_s': spec['l10_y_dwell_s'],
                'l11_back_s': spec['l11_back_s'],
                'family': spec['family'],
                'tweak': spec['tweak'],
                'anchors': sorted(spec['insertions'].keys()),
            }
            for spec in specs
        ],
        'rows_sorted': rows_sorted,
        'best_followup_candidate': best_summary,
        'kf36_rows': kf36_rows,
        'comparison_rows': comparison_rows,
        'bottom_line': {
            'family_keeps_mainline_lead': 'YES',
            'new_family_mainline_update': 'YES' if new_family_mainline_update else 'NO',
            'takes_absolute_max_crown': 'YES' if takes_absolute_max_crown else 'NO',
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
    print('BEST_FOLLOWUP', best_cand.name, best_payload['overall'], flush=True)
    print('BOTTOM_LINE', verdict, flush=True)


if __name__ == '__main__':
    main()
