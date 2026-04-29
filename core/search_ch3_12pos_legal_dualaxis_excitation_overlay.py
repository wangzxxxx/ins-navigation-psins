from __future__ import annotations

import argparse
import json
import re
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

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
from compare_ch3_12pos_path_baselines import build_dataset_with_path
from compare_four_methods_shared_noise import _load_json, _noise_matches, compute_payload, expected_noise_config
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs
from diagnose_ch3_12pos_narrow import orientation_faces, paras_to_rows, rows_to_paras
from search_ch3_12pos_legal_dualaxis_repairs import (
    DiscreteDualAxisKinematics,
    OLD_RESULT_MAP,
    VALID_PRIOR_SIGN_ONLY_PATHS,
    build_candidate,
    compare_vs_base,
    make_suffix,
    render_action,
)

NOISE_SCALE = 0.08
REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
ANCHOR_FLIPS = ()
INCUMBENT_FLIPS = (8, 11, 12)
BURST_ROTATION_S = 2.0
BURST_DWELL_S = 2.0
BURST_TOTAL_S = 2 * (BURST_ROTATION_S + BURST_DWELL_S)

CANDIDATE_SPECS = [
    {
        'name': 'overlay_outer_tail10_11_pos',
        'direction': 'overlay',
        'family': 'outer_excitation_return_pair',
        'burst_kind': 'outer',
        'target_anchor_nodes': [10, 11],
        'burst_angle_deg': +90,
        'rationale': 'Keep faithful12 anchors unchanged; add short legal outer +90/-90 burst after the late weak block anchors 10 and 11.',
    },
    {
        'name': 'overlay_outer_tail10_11_neg',
        'direction': 'overlay',
        'family': 'outer_excitation_return_pair',
        'burst_kind': 'outer',
        'target_anchor_nodes': [10, 11],
        'burst_angle_deg': -90,
        'rationale': 'Same as outer tail overlay, but reverse burst direction to test the opposite dynamic sweep while still returning to the same anchors.',
    },
    {
        'name': 'overlay_inner_tail10_11_pos',
        'direction': 'overlay',
        'family': 'inner_excitation_return_pair',
        'burst_kind': 'inner',
        'target_anchor_nodes': [10, 11],
        'burst_angle_deg': +90,
        'rationale': 'Keep faithful12 anchors unchanged; add short legal inner +90/-90 burst after anchors 10 and 11 to inject y-axis dynamic excitation.',
    },
    {
        'name': 'overlay_inner_tail10_11_neg',
        'direction': 'overlay',
        'family': 'inner_excitation_return_pair',
        'burst_kind': 'inner',
        'target_anchor_nodes': [10, 11],
        'burst_angle_deg': -90,
        'rationale': 'Reverse-direction inner burst after anchors 10 and 11, preserving the same 12 anchors and total time.',
    },
    {
        'name': 'overlay_outer_early4_5_pos',
        'direction': 'overlay',
        'family': 'outer_excitation_return_pair',
        'burst_kind': 'outer',
        'target_anchor_nodes': [4, 5],
        'burst_angle_deg': +90,
        'rationale': 'Add short outer excitation-return bursts on the early z-family block (anchors 4 and 5) while preserving the original anchor face sequence.',
    },
    {
        'name': 'overlay_inner_early4_5_pos',
        'direction': 'overlay',
        'family': 'inner_excitation_return_pair',
        'burst_kind': 'inner',
        'target_anchor_nodes': [4, 5],
        'burst_angle_deg': +90,
        'rationale': 'Add short inner excitation-return bursts on the early block (anchors 4 and 5), keeping anchors unchanged and total time at 1200 s.',
    },
]


@dataclass
class OverlayCandidate:
    name: str
    direction: str
    family: str
    rationale: str
    burst_kind: str
    target_anchor_nodes: list[int]
    burst_angle_deg: int
    anchor_rows: list[dict]
    burst_rows: list[dict]
    all_rows: list[dict]
    all_actions: list[dict]
    all_faces: list[dict]
    anchor_face_sequence: list[str]
    method_tag: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def sanitize_tag(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


def candidate_result_path(candidate: OverlayCandidate, method_key: str, noise_scale: float) -> Path:
    suffix = make_suffix(noise_scale)
    if method_key == 'markov42_noisy':
        prefix = 'M_markov_42state_gm1'
    elif method_key == 'kf36_noisy':
        prefix = 'KF36'
    else:
        raise KeyError(method_key)
    return RESULTS_DIR / f'{prefix}_{candidate.method_tag}_shared_{suffix}_param_errors.json'


def load_baselines(noise_scale: float) -> dict:
    faithful_markov = _load_json(RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json')
    faithful_kf = _load_json(RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json')
    default_markov = _load_json(RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json')
    default_kf = _load_json(RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json')
    expected_cfg = expected_noise_config(noise_scale)
    for payload in [faithful_markov, faithful_kf, default_markov, default_kf]:
        if not _noise_matches(payload, expected_cfg):
            raise ValueError('Baseline noise configuration mismatch')
    return {
        'faithful_markov': faithful_markov,
        'faithful_kf': faithful_kf,
        'default_markov': default_markov,
        'default_kf': default_kf,
    }


def previous_old_valid_payload(flips: tuple[int, ...], method_key: str, noise_scale: float):
    expected_cfg = expected_noise_config(noise_scale)
    old_paths = OLD_RESULT_MAP.get(flips, {})
    path = old_paths.get(method_key)
    if not path or not path.exists():
        raise FileNotFoundError(f'No previous valid payload for {flips} / {method_key}')
    payload = _load_json(path)
    if not _noise_matches(payload, expected_cfg):
        raise ValueError(f'Noise mismatch for {path}')
    return payload, path


def run_candidate_payload(mod, candidate: OverlayCandidate, method_key: str, noise_scale: float, force_rerun: bool = False):
    expected_cfg = expected_noise_config(noise_scale)
    out_path = candidate_result_path(candidate, method_key, noise_scale)
    if out_path.exists() and (not force_rerun):
        payload = _load_json(out_path)
        if _noise_matches(payload, expected_cfg) and payload.get('extra', {}).get('candidate_name') == candidate.name:
            return payload, 'reused_verified', out_path

    paras = rows_to_paras(mod, candidate.all_rows)
    dataset = build_dataset_with_path(mod, noise_scale, paras)
    params = _param_specs(mod)

    if method_key == 'markov42_noisy':
        clbt, _, _, _, _ = mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=42,
            bi_g=dataset['bi_g'],
            tau_g=dataset['tau_g'],
            bi_a=dataset['bi_a'],
            tau_a=dataset['tau_a'],
            label=f'{candidate.name}_{method_key}_{make_suffix(noise_scale)}',
        )
    elif method_key == 'kf36_noisy':
        clbt, _, _, _, _ = mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=36,
            label=f'{candidate.name}_{method_key}_{make_suffix(noise_scale)}',
        )
    else:
        raise KeyError(method_key)

    payload = compute_payload(
        mod,
        clbt,
        params,
        variant=f'{candidate.name}_{method_key}_{make_suffix(noise_scale)}',
        method_file='search_ch3_12pos_legal_dualaxis_excitation_overlay.py',
        extra={
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'comparison_mode': 'ch3_legal_dualaxis_excitation_overlay',
            'candidate_name': candidate.name,
            'method_key': method_key,
            'representation': 'same_12_anchor_faces_plus_legal_excitation_return_pairs',
            'burst_kind': candidate.burst_kind,
            'target_anchor_nodes': candidate.target_anchor_nodes,
            'burst_angle_deg': candidate.burst_angle_deg,
            'burst_rotation_s': BURST_ROTATION_S,
            'burst_dwell_s': BURST_DWELL_S,
        },
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', out_path


def delta_vs_incumbent(inc_payload: dict, cand_payload: dict) -> dict:
    out = {}
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        iv = float(inc_payload['overall'][metric])
        cv = float(cand_payload['overall'][metric])
        out[metric] = {
            'incumbent': iv,
            'candidate': cv,
            'improvement_pct_points': iv - cv,
            'relative_improvement_pct': ((iv - cv) / iv * 100.0) if abs(iv) > 1e-12 else None,
        }
    return out


def materially_beats_incumbent(inc_payload: dict, cand_payload: dict) -> bool:
    inc = inc_payload['overall']
    cur = cand_payload['overall']
    mean_gain = float(inc['mean_pct_error']) - float(cur['mean_pct_error'])
    max_gain = float(inc['max_pct_error']) - float(cur['max_pct_error'])
    return mean_gain >= 0.5 and max_gain >= 5.0


def build_overlay_candidate(mod, spec: dict) -> OverlayCandidate:
    anchor = build_candidate(mod, ANCHOR_FLIPS)
    faithful_anchor_faces = [x['face_name'] for x in anchor.faces]
    base_rows = anchor.rows
    base_actions = anchor.action_sequence

    kin = DiscreteDualAxisKinematics()
    all_rows: list[dict] = []
    all_actions: list[dict] = []
    anchor_row_indices: list[int] = []
    anchor_rows: list[dict] = []
    burst_rows: list[dict] = []

    for anchor_idx, (base_action, base_row) in enumerate(zip(base_actions, base_rows), start=1):
        action, row = kin.apply(len(all_rows) + 1, base_action['kind'], base_action['motor_angle_deg'])
        row['rotation_time_s'] = float(base_row['rotation_time_s'])
        row['pre_static_s'] = float(base_row['pre_static_s'])
        row['post_static_s'] = float(base_row['post_static_s'])
        if anchor_idx in spec['target_anchor_nodes']:
            row['post_static_s'] -= BURST_TOTAL_S
        row['node_total_s'] = row['rotation_time_s'] + row['pre_static_s'] + row['post_static_s']
        if row['post_static_s'] < -1e-9:
            raise ValueError(f'Negative anchor post-static for {spec["name"]} at anchor {anchor_idx}')
        row['anchor_id'] = anchor_idx
        row['segment_role'] = 'anchor'
        action['anchor_id'] = anchor_idx
        action['segment_role'] = 'anchor'
        all_rows.append(row)
        all_actions.append(action)
        anchor_row_indices.append(len(all_rows) - 1)
        anchor_rows.append(dict(row))

        if anchor_idx in spec['target_anchor_nodes']:
            for burst_step, angle in enumerate([spec['burst_angle_deg'], -spec['burst_angle_deg']], start=1):
                sub_action, sub_row = kin.apply(len(all_rows) + 1, spec['burst_kind'], angle)
                sub_row['rotation_time_s'] = BURST_ROTATION_S
                sub_row['pre_static_s'] = 0.0
                sub_row['post_static_s'] = BURST_DWELL_S
                sub_row['node_total_s'] = BURST_ROTATION_S + BURST_DWELL_S
                sub_row['anchor_id'] = anchor_idx
                sub_row['segment_role'] = 'burst_out' if burst_step == 1 else 'burst_return'
                sub_action['anchor_id'] = anchor_idx
                sub_action['segment_role'] = sub_row['segment_role']
                all_rows.append(sub_row)
                all_actions.append(sub_action)
                burst_rows.append(dict(sub_row))

    paras = rows_to_paras(mod, all_rows)
    all_faces = orientation_faces(mod, paras)
    anchor_faces = [all_faces[idx]['face_name'] for idx in anchor_row_indices]
    if anchor_faces != faithful_anchor_faces:
        raise ValueError(
            f'Anchor face sequence changed for {spec["name"]}: {anchor_faces} vs {faithful_anchor_faces}'
        )
    total_time_s = sum(row['node_total_s'] for row in all_rows)
    if abs(total_time_s - 1200.0) > 1e-6:
        raise ValueError(f'Total time drift for {spec["name"]}: {total_time_s}')

    return OverlayCandidate(
        name=spec['name'],
        direction=spec['direction'],
        family=spec['family'],
        rationale=spec['rationale'],
        burst_kind=spec['burst_kind'],
        target_anchor_nodes=list(spec['target_anchor_nodes']),
        burst_angle_deg=int(spec['burst_angle_deg']),
        anchor_rows=anchor_rows,
        burst_rows=burst_rows,
        all_rows=all_rows,
        all_actions=all_actions,
        all_faces=all_faces,
        anchor_face_sequence=anchor_faces,
        method_tag=f'ch3overlay_{sanitize_tag(spec["name"])}',
    )


def render_report(payload: dict) -> str:
    lines = []
    lines.append('# Chapter-3 legal dual-axis excitation-overlay relaunch')
    lines.append('')
    lines.append('## 1. Branch constraint actually enforced')
    lines.append('')
    lines.append('- Anchor skeleton is the **original faithful chapter-3 12-position sequence**; anchor face sequence is not changed.')
    lines.append('- Search object is **excitation overlay only**: short legal excitation-return pairs inserted after selected anchors, then returned to the same anchor pose before the next anchor move.')
    lines.append('- Real hardware rule remains strict: inner axis = body y, outer axis = x/z family determined by current inner attitude.')
    lines.append('- Primary branch kept total time at **1200 s exactly** by redistributing anchor post-static time internally; no arbitrary path reordering was allowed.')
    lines.append('')
    lines.append('## 2. Fixed references')
    lines.append('')
    lines.append(f"- faithful12 Markov42: mean **{payload['references']['faithful12']['markov42']['overall']['mean_pct_error']:.3f}**, median **{payload['references']['faithful12']['markov42']['overall']['median_pct_error']:.3f}**, max **{payload['references']['faithful12']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- incumbent legal_flip_8_11_12 Markov42: mean **{payload['references']['incumbent']['markov42']['overall']['mean_pct_error']:.3f}**, median **{payload['references']['incumbent']['markov42']['overall']['median_pct_error']:.3f}**, max **{payload['references']['incumbent']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- default18 Markov42: mean **{payload['references']['default18']['markov42']['overall']['mean_pct_error']:.3f}**, median **{payload['references']['default18']['markov42']['overall']['median_pct_error']:.3f}**, max **{payload['references']['default18']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('## 3. Tested excitation-overlay candidates (Markov42, shared noise0p08/seed42)')
    lines.append('')
    lines.append('| rank | candidate | burst kind | anchors | burst angle | mean | median | max | Δmean vs incumbent | Δmax vs incumbent | note |')
    lines.append('|---:|---|---|---|---:|---:|---:|---:|---:|---:|---|')
    for idx, row in enumerate(payload['markov42_rows'], start=1):
        d = row['delta_vs_incumbent']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['burst_kind']} | {row['target_anchor_nodes']} | {row['burst_angle_deg']:+d} | {row['metrics']['overall']['mean_pct_error']:.3f} | {row['metrics']['overall']['median_pct_error']:.3f} | {row['metrics']['overall']['max_pct_error']:.3f} | {d['mean_pct_error']['improvement_pct_points']:+.3f} | {d['max_pct_error']['improvement_pct_points']:+.3f} | {row['rationale']} |"
        )
    lines.append('')
    lines.append('## 4. Best overlay result and verdict')
    lines.append('')
    best = payload['best_overlay']
    lines.append(f"- best overlay candidate: **{best['candidate_name']}**")
    lines.append(f"- Markov42: mean **{best['markov42']['overall']['mean_pct_error']:.3f}**, median **{best['markov42']['overall']['median_pct_error']:.3f}**, max **{best['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- vs faithful12: Δmean = **{best['delta_vs_faithful_markov42']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian = **{best['delta_vs_faithful_markov42']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax = **{best['delta_vs_faithful_markov42']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- vs incumbent legal_flip_8_11_12: Δmean = **{best['delta_vs_incumbent']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian = **{best['delta_vs_incumbent']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax = **{best['delta_vs_incumbent']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- material beat over incumbent? **{'yes' if payload['material_improvement_over_incumbent'] else 'no'}**")
    lines.append('')
    lines.append('## 5. Exact best overlay legal motor sequence / timing table')
    lines.append('')
    lines.append('| seq | anchor_id | role | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---:|---:|---:|---:|---|')
    for seq_idx, (action, row, face) in enumerate(zip(best['all_actions'], best['all_rows'], best['all_faces']), start=1):
        lines.append(
            f"| {seq_idx} | {row['anchor_id']} | {row['segment_role']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 6. Anchor-sequence preservation check')
    lines.append('')
    lines.append(f"- faithful12 anchor faces: {' → '.join(payload['references']['faithful12']['anchor_face_sequence'])}")
    lines.append(f"- best overlay anchor faces: {' → '.join(best['anchor_face_sequence'])}")
    lines.append(f"- preserved exactly? **{'yes' if best['anchor_face_sequence'] == payload['references']['faithful12']['anchor_face_sequence'] else 'no'}**")
    lines.append('')
    lines.append('## 7. Bottom line')
    lines.append('')
    lines.append(f"- {payload['bottom_line']}")
    if payload['material_improvement_over_incumbent']:
        lines.append('- Because the <=1200 s branch already produced a material beat, no extra-time secondary branch was needed in this pass.')
    else:
        lines.append('- Because the <=1200 s overlay branch did not show a material beat over the incumbent, this pass does **not** expand into a broader extra-time family; that remains only a secondary future option if the user wants it.')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_module('search_ch3_12pos_legal_dualaxis_excitation_overlay_src', str(SOURCE_FILE))

    baselines = load_baselines(args.noise_scale)
    faithful_anchor = build_candidate(mod, ANCHOR_FLIPS)
    incumbent = build_candidate(mod, INCUMBENT_FLIPS)
    incumbent_markov_payload, incumbent_markov_path = previous_old_valid_payload(INCUMBENT_FLIPS, 'markov42_noisy', args.noise_scale)
    incumbent_kf_payload, incumbent_kf_path = previous_old_valid_payload(INCUMBENT_FLIPS, 'kf36_noisy', args.noise_scale)

    candidates = [build_overlay_candidate(mod, spec) for spec in CANDIDATE_SPECS]

    rows = []
    payload_by_name = {}
    candidate_by_name = {cand.name: cand for cand in candidates}
    for cand in candidates:
        print(f'RUN {cand.name} ...', flush=True)
        payload, status, path = run_candidate_payload(mod, cand, 'markov42_noisy', args.noise_scale, args.force_rerun)
        item = {
            'candidate_name': cand.name,
            'direction': cand.direction,
            'family': cand.family,
            'burst_kind': cand.burst_kind,
            'target_anchor_nodes': cand.target_anchor_nodes,
            'burst_angle_deg': cand.burst_angle_deg,
            'rationale': cand.rationale,
            'metrics': compact_result(payload),
            'run_json': str(path),
            'status': status,
            'delta_vs_incumbent': delta_vs_incumbent(incumbent_markov_payload, payload),
        }
        rows.append(item)
        payload_by_name[cand.name] = payload
        print(
            f"DONE {cand.name}: mean={item['metrics']['overall']['mean_pct_error']:.3f}, "
            f"median={item['metrics']['overall']['median_pct_error']:.3f}, "
            f"max={item['metrics']['overall']['max_pct_error']:.3f}",
            flush=True,
        )

    rows.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    best_row = rows[0]
    best_candidate = candidate_by_name[best_row['candidate_name']]
    best_markov_payload = payload_by_name[best_candidate.name]
    best_kf_payload, best_kf_status, best_kf_path = run_candidate_payload(mod, best_candidate, 'kf36_noisy', args.noise_scale, args.force_rerun)

    material = materially_beats_incumbent(incumbent_markov_payload, best_markov_payload)
    if material:
        bottom_line = (
            f"Yes. Under the user-suggested excitation-overlay branch, {best_candidate.name} materially beat legal_flip_8_11_12 "
            'on the primary Markov42 metric while preserving the original 12 anchor faces and keeping total time at 1200 s.'
        )
    else:
        bottom_line = (
            f"No. Under the user-suggested excitation-overlay branch, the best <=1200 s overlay candidate is {best_candidate.name}, "
            'but it does not materially beat legal_flip_8_11_12 on the shared Markov42 metric.'
        )

    best_summary = {
        'candidate_name': best_candidate.name,
        'direction': best_candidate.direction,
        'family': best_candidate.family,
        'rationale': best_candidate.rationale,
        'burst_kind': best_candidate.burst_kind,
        'target_anchor_nodes': best_candidate.target_anchor_nodes,
        'burst_angle_deg': best_candidate.burst_angle_deg,
        'anchor_face_sequence': best_candidate.anchor_face_sequence,
        'all_rows': best_candidate.all_rows,
        'all_actions': best_candidate.all_actions,
        'all_faces': best_candidate.all_faces,
        'markov42': compact_result(best_markov_payload),
        'markov42_run_json': best_row['run_json'],
        'kf36': compact_result(best_kf_payload),
        'kf36_run_json': str(best_kf_path),
        'kf36_status': best_kf_status,
        'delta_vs_faithful_markov42': compare_vs_base(baselines['faithful_markov'], best_markov_payload),
        'delta_vs_incumbent': delta_vs_incumbent(incumbent_markov_payload, best_markov_payload),
        'delta_vs_default18_markov42': compare_vs_base(baselines['default_markov'], best_markov_payload),
    }

    out_json = RESULTS_DIR / f'ch3_12pos_legal_dualaxis_excitation_overlay_{make_suffix(args.noise_scale)}.json'
    out_md = REPORTS_DIR / f'psins_ch3_12pos_legal_dualaxis_excitation_overlay_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_12pos_legal_dualaxis_excitation_overlay',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'hard_constraints': {
            'anchor_skeleton': 'faithful chapter-3 12-position face sequence fixed',
            'anchor_faces_changed': False,
            'overlay_only': True,
            'inner_axis': 'body y only',
            'outer_axis': 'x/z family determined by inner attitude',
            'time_budget_policy': 'primary <=1200 s branch only in this pass',
            'total_time_s': 1200.0,
            'seed': 42,
            'base_family': 'round53_round61_shared',
        },
        'references': {
            'faithful12': {
                'candidate_name': faithful_anchor.name,
                'anchor_face_sequence': [x['face_name'] for x in faithful_anchor.faces],
                'rows': faithful_anchor.rows,
                'action_sequence': faithful_anchor.action_sequence,
                'markov42': compact_result(baselines['faithful_markov']),
                'markov42_run_json': str(RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'),
                'kf36': compact_result(baselines['faithful_kf']),
                'kf36_run_json': str(RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json'),
            },
            'incumbent': {
                'candidate_name': incumbent.name,
                'equivalent_prior_result': VALID_PRIOR_SIGN_ONLY_PATHS[2],
                'rows': incumbent.rows,
                'action_sequence': incumbent.action_sequence,
                'anchor_face_sequence': [x['face_name'] for x in incumbent.faces],
                'markov42': compact_result(incumbent_markov_payload),
                'markov42_run_json': str(incumbent_markov_path),
                'kf36': compact_result(incumbent_kf_payload),
                'kf36_run_json': str(incumbent_kf_path),
            },
            'default18': {
                'candidate_name': 'default18_reference',
                'markov42': compact_result(baselines['default_markov']),
                'markov42_run_json': str(RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json'),
                'kf36': compact_result(baselines['default_kf']),
                'kf36_run_json': str(RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json'),
            },
        },
        'candidate_specs': CANDIDATE_SPECS,
        'markov42_rows': rows,
        'best_overlay': best_summary,
        'material_improvement_over_incumbent': material,
        'bottom_line': bottom_line,
        'files': {
            'json': str(out_json),
            'report': str(out_md),
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    out_md.write_text(render_report(payload), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps(payload['files'], ensure_ascii=False), flush=True)
    print('BEST_OVERLAY', best_summary['candidate_name'], best_summary['markov42']['overall'], flush=True)
    print('MATERIAL_OVER_INCUMBENT', material, flush=True)


if __name__ == '__main__':
    main()
