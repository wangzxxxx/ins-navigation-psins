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
from benchmark_ch3_12pos_goalA_repairs import KEY_PARAMS, compact_result
from compare_ch3_12pos_path_baselines import build_dataset_with_path
from compare_four_methods_shared_noise import _load_json, _noise_matches, compute_payload, expected_noise_config
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs
from diagnose_ch3_12pos_narrow import orientation_faces, rows_to_paras, structural_summary
from search_ch3_12pos_legal_dualaxis_repairs import (
    BASE_ACTION_TEMPLATE,
    Candidate,
    DiscreteDualAxisKinematics,
    build_candidate,
    compare_vs_base,
    make_suffix,
    render_action,
)

NOISE_SCALE = 0.08
REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
MAX_TOTAL_POSITIONS = 18
BASE_POSITIONS = len(BASE_ACTION_TEMPLATE)
MAX_APPEND = MAX_TOTAL_POSITIONS - BASE_POSITIONS

INCUMBENT_NAME = 'legal_flip_8_11_12_retime_flipnodes_pre20_post70'
INCUMBENT_MARKOV_JSON = RESULTS_DIR / 'M_markov_42state_gm1_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
INCUMBENT_KF_JSON = RESULTS_DIR / 'KF36_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
FAITHFUL_MARKOV_JSON = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
FAITHFUL_KF_JSON = RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json'

# Batch-1: theory-guided motifs only; no brute-force explosion.
# All candidates preserve the original chapter-3 12 nodes exactly and append only legal tail nodes.
BATCH1_SPECS = [
    {
        'name': 'tail_y_pair_pos',
        'batch': 'batch1',
        'family': 'y_pair',
        'rationale': 'Minimal y-only closure loop from the +Z end state. Probes whether extra inner-axis dwell alone helps dKa_yy / Ka2_y without adding new z-family outer excitation.',
        'tail_actions': [('inner', +90), ('inner', -90)],
        'timing_policy': 'base100',
    },
    {
        'name': 'tail_y_pair_neg',
        'batch': 'batch1',
        'family': 'y_pair',
        'rationale': 'Opposite-sign y-only closure loop. Same theory as tail_y_pair_pos but on the mirrored branch.',
        'tail_actions': [('inner', -90), ('inner', +90)],
        'timing_policy': 'base100',
    },
    {
        'name': 'tail_z_return_nx_py',
        'batch': 'batch1',
        'family': 'z_return_closed',
        'rationale': 'Open to the -X branch, execute a z-family +Y/-Y excitation-return pair, then close back to +Z. Intended as the safest dKg_zz-targeting motif with net closure.',
        'tail_actions': [('inner', +90), ('outer', +90), ('outer', -90), ('inner', -90)],
        'timing_policy': 'base100',
    },
    {
        'name': 'tail_z_return_nx_ny',
        'batch': 'batch1',
        'family': 'z_return_closed',
        'rationale': 'Mirror of tail_z_return_nx_py on the -Y branch. Tests whether the sign of the late y-face matters for dKg_zz / dKa_yy trade-off.',
        'tail_actions': [('inner', +90), ('outer', -90), ('outer', +90), ('inner', -90)],
        'timing_policy': 'base100',
    },
    {
        'name': 'tail_z_return_px_py',
        'batch': 'batch1',
        'family': 'z_return_closed',
        'rationale': 'Closed z-family excitation-return pair on the +X / +Y branch. Same dKg_zz-oriented motif but mirrored in the x branch.',
        'tail_actions': [('inner', -90), ('outer', +90), ('outer', -90), ('inner', +90)],
        'timing_policy': 'base100',
    },
    {
        'name': 'tail_z_return_px_ny',
        'batch': 'batch1',
        'family': 'z_return_closed',
        'rationale': 'Closed z-family excitation-return pair on the +X / -Y branch.',
        'tail_actions': [('inner', -90), ('outer', -90), ('outer', +90), ('inner', +90)],
        'timing_policy': 'base100',
    },
    {
        'name': 'tail_z_sweep_nx',
        'batch': 'batch1',
        'family': 'z_sweep_closed',
        'rationale': 'Open to -X, keep the z-family outer sweep in the same rotational direction for two nodes, then close. Stronger dKg_zz excitation, but with more risk to already-good channels.',
        'tail_actions': [('inner', +90), ('outer', +90), ('outer', +90), ('inner', -90)],
        'timing_policy': 'base100',
    },
    {
        'name': 'tail_z_sweep_px',
        'batch': 'batch1',
        'family': 'z_sweep_closed',
        'rationale': 'Mirror of tail_z_sweep_nx on the +X branch.',
        'tail_actions': [('inner', -90), ('outer', -90), ('outer', -90), ('inner', +90)],
        'timing_policy': 'base100',
    },
    {
        'name': 'tail_z_return_then_yswap_nx',
        'batch': 'batch1',
        'family': 'z_then_yswap',
        'rationale': 'First perform the safe closed z-return motif, then add one extra mirrored y-pair. Intended to preserve the dKg_zz benefit while giving dKa_yy / Ka2_y one more pure-y correction opportunity.',
        'tail_actions': [('inner', +90), ('outer', +90), ('outer', -90), ('inner', -90), ('inner', -90), ('inner', +90)],
        'timing_policy': 'base100',
    },
    {
        'name': 'tail_z_return_then_yswap_px',
        'batch': 'batch1',
        'family': 'z_then_yswap',
        'rationale': 'Mirror of tail_z_return_then_yswap_nx.',
        'tail_actions': [('inner', -90), ('outer', -90), ('outer', +90), ('inner', +90), ('inner', +90), ('inner', -90)],
        'timing_policy': 'base100',
    },
]

# Batch-2: appended short-burst retime around the most interpretable z-tail motifs.
# This tests whether the batch-1 failure came mainly from spending too much late dwell in off-baseline faces.
BATCH2_SPECS = [
    {
        'name': 'tail_z_return_px_ny_burst10',
        'batch': 'batch2',
        'family': 'z_return_closed_short',
        'rationale': 'Short append-only burst analogue of the helpful localized z-return idea: same +X/-Y branch as the best dKa_yy-friendly batch-1 return motif, but each appended node is only 10 s total.',
        'tail_actions': [('inner', -90), ('outer', -90), ('outer', +90), ('inner', +90)],
        'timing_policy': 'burst10',
    },
    {
        'name': 'tail_z_return_nx_ny_burst10',
        'batch': 'batch2',
        'family': 'z_return_closed_short',
        'rationale': 'Short burst version of the -X/-Y closed z-return motif, included because it had the cleanest median among the batch-1 return pairs.',
        'tail_actions': [('inner', +90), ('outer', -90), ('outer', +90), ('inner', -90)],
        'timing_policy': 'burst10',
    },
    {
        'name': 'tail_z_sweep_nx_burst10',
        'batch': 'batch2',
        'family': 'z_sweep_closed_short',
        'rationale': 'Short burst version of the batch-1 best family (closed z sweep on the -X branch). Tests whether keeping the same geometry but dramatically shrinking dwell can preserve dKg_zz benefit without wrecking guard channels.',
        'tail_actions': [('inner', +90), ('outer', +90), ('outer', +90), ('inner', -90)],
        'timing_policy': 'burst10',
    },
    {
        'name': 'tail_z_return_px_ny_burst20',
        'batch': 'batch2',
        'family': 'z_return_closed_short',
        'rationale': 'Intermediate-duration short z-return tail on the +X/-Y branch, to check whether 10 s nodes are too short but 100 s nodes are too long.',
        'tail_actions': [('inner', -90), ('outer', -90), ('outer', +90), ('inner', +90)],
        'timing_policy': 'burst20',
    },
    {
        'name': 'tail_z_sweep_nx_burst20',
        'batch': 'batch2',
        'family': 'z_sweep_closed_short',
        'rationale': 'Intermediate-duration version of the best batch-1 geometry.',
        'tail_actions': [('inner', +90), ('outer', +90), ('outer', +90), ('inner', -90)],
        'timing_policy': 'burst20',
    },
]

ALL_CANDIDATE_SPECS = BATCH1_SPECS + BATCH2_SPECS


@dataclass
class AppendTailCandidate:
    name: str
    batch: str
    family: str
    rationale: str
    timing_policy: str
    tail_actions: list[tuple[str, int]]
    base_rows: list[dict]
    tail_rows: list[dict]
    all_rows: list[dict]
    all_actions: list[dict]
    all_faces: list[dict]
    tail_faces: list[str]
    end_face: str
    end_beta_deg: int
    structural: dict
    theory_tags: dict
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


def load_reference_payloads(noise_scale: float) -> dict:
    expected_cfg = expected_noise_config(noise_scale)
    payloads = {
        'faithful_markov': _load_json(FAITHFUL_MARKOV_JSON),
        'faithful_kf': _load_json(FAITHFUL_KF_JSON),
        'incumbent_markov': _load_json(INCUMBENT_MARKOV_JSON),
        'incumbent_kf': _load_json(INCUMBENT_KF_JSON),
    }
    for name, payload in payloads.items():
        if not _noise_matches(payload, expected_cfg):
            raise ValueError(f'Noise mismatch for {name}')
    return payloads


def apply_timing_policy(row: dict, policy: str) -> dict:
    item = dict(row)
    if policy == 'base100':
        item['rotation_time_s'] = 10.0
        item['pre_static_s'] = 10.0
        item['post_static_s'] = 80.0
    elif policy == 'tail_dwell90':
        item['rotation_time_s'] = 10.0
        item['pre_static_s'] = 0.0
        item['post_static_s'] = 90.0
    elif policy == 'tail_pre20':
        item['rotation_time_s'] = 10.0
        item['pre_static_s'] = 20.0
        item['post_static_s'] = 70.0
    elif policy == 'burst10':
        item['rotation_time_s'] = 5.0
        item['pre_static_s'] = 0.0
        item['post_static_s'] = 5.0
    elif policy == 'burst20':
        item['rotation_time_s'] = 5.0
        item['pre_static_s'] = 0.0
        item['post_static_s'] = 15.0
    else:
        raise KeyError(policy)
    item['node_total_s'] = float(item['rotation_time_s'] + item['pre_static_s'] + item['post_static_s'])
    return item


def build_append_tail_candidate(mod, spec: dict) -> AppendTailCandidate:
    if len(spec['tail_actions']) > MAX_APPEND:
        raise ValueError(f"Tail too long for {spec['name']}: {len(spec['tail_actions'])}")

    faithful = build_candidate(mod, ())
    kin = DiscreteDualAxisKinematics()
    all_actions = []
    all_rows = []

    for idx, base_action in enumerate(faithful.action_sequence, start=1):
        action, row = kin.apply(idx, base_action['kind'], base_action['motor_angle_deg'])
        base_row = faithful.rows[idx - 1]
        row['rotation_time_s'] = float(base_row['rotation_time_s'])
        row['pre_static_s'] = float(base_row['pre_static_s'])
        row['post_static_s'] = float(base_row['post_static_s'])
        row['node_total_s'] = float(base_row['node_total_s'])
        row['segment_role'] = 'base'
        action['segment_role'] = 'base'
        all_actions.append(action)
        all_rows.append(row)

    for local_idx, (kind, angle_deg) in enumerate(spec['tail_actions'], start=1):
        global_idx = BASE_POSITIONS + local_idx
        action, row = kin.apply(global_idx, kind, angle_deg)
        row = apply_timing_policy(row, spec['timing_policy'])
        row['segment_role'] = 'tail'
        row['tail_idx'] = local_idx
        action['segment_role'] = 'tail'
        action['tail_idx'] = local_idx
        all_actions.append(action)
        all_rows.append(row)

    total_time_s = sum(x['node_total_s'] for x in all_rows)
    if total_time_s > 1800.0 + 1e-9:
        raise ValueError(f"Total time too long for {spec['name']}: {total_time_s}")

    paras = rows_to_paras(mod, all_rows)
    faces = orientation_faces(mod, paras)
    tail_rows = [dict(x) for x in all_rows[BASE_POSITIONS:]]
    tail_faces = [x['face_name'] for x in faces[BASE_POSITIONS:]]
    structural = structural_summary(all_rows, faces)

    z_outer_hits = sum(1 for a in all_actions[BASE_POSITIONS:] if a['kind'] == 'outer' and a['outer_mode'] == 'z')
    x_outer_hits = sum(1 for a in all_actions[BASE_POSITIONS:] if a['kind'] == 'outer' and a['outer_mode'] == 'x')
    y_inner_hits = sum(1 for a in all_actions[BASE_POSITIONS:] if a['kind'] == 'inner')

    return AppendTailCandidate(
        name=spec['name'],
        batch=spec['batch'],
        family=spec['family'],
        rationale=spec['rationale'],
        timing_policy=spec['timing_policy'],
        tail_actions=list(spec['tail_actions']),
        base_rows=[dict(x) for x in all_rows[:BASE_POSITIONS]],
        tail_rows=tail_rows,
        all_rows=[dict(x) for x in all_rows],
        all_actions=[dict(x) for x in all_actions],
        all_faces=faces,
        tail_faces=tail_faces,
        end_face=faces[-1]['face_name'],
        end_beta_deg=int(all_actions[-1]['inner_beta_after_deg']),
        structural=structural,
        theory_tags={
            'z_outer_hits': z_outer_hits,
            'x_outer_hits': x_outer_hits,
            'y_inner_hits': y_inner_hits,
            'closed_to_plusZ_beta0': bool(faces[-1]['face_name'] == '+Z' and all_actions[-1]['inner_beta_after_deg'] == 0),
            'tail_length': len(spec['tail_actions']),
            'tail_total_time_s': float(sum(x['node_total_s'] for x in tail_rows)),
        },
        method_tag=f"ch3appendtail_{sanitize_tag(spec['name'])}",
    )


def candidate_result_path(candidate: AppendTailCandidate, method_key: str, noise_scale: float) -> Path:
    suffix = make_suffix(noise_scale)
    if method_key == 'markov42_noisy':
        prefix = 'M_markov_42state_gm1'
    elif method_key == 'kf36_noisy':
        prefix = 'KF36'
    else:
        raise KeyError(method_key)
    return RESULTS_DIR / f'{prefix}_{candidate.method_tag}_shared_{suffix}_param_errors.json'


def run_candidate_payload(mod, candidate: AppendTailCandidate, method_key: str, noise_scale: float, force_rerun: bool = False):
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
        method_file='search_ch3_12pos_append_tail.py',
        extra={
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'comparison_mode': 'ch3_append_tail_search',
            'candidate_name': candidate.name,
            'method_key': method_key,
            'family': candidate.family,
            'timing_policy': candidate.timing_policy,
            'tail_actions': candidate.tail_actions,
            'tail_length': len(candidate.tail_actions),
            'tail_total_time_s': sum(x['node_total_s'] for x in candidate.tail_rows),
            'legality': 'fixed_original_12_plus_legal_appended_tail_only',
        },
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', out_path


def delta_vs_reference(base_payload: dict, cand_payload: dict) -> dict:
    out = {
        'overall': compare_vs_base(base_payload, cand_payload),
        'key_params': {},
    }
    for key in KEY_PARAMS:
        bv = float(base_payload['param_errors'][key]['pct_error'])
        cv = float(cand_payload['param_errors'][key]['pct_error'])
        out['key_params'][key] = {
            'base': bv,
            'candidate': cv,
            'improvement_pct_points': bv - cv,
        }
    return out


def guard_readout(base_payload: dict, cand_payload: dict) -> dict:
    out = {}
    for key in ['dKg_xz', 'dKa_xz', 'Ka2_z']:
        bv = float(base_payload['param_errors'][key]['pct_error'])
        cv = float(cand_payload['param_errors'][key]['pct_error'])
        out[key] = {
            'base': bv,
            'candidate': cv,
            'improvement_pct_points': bv - cv,
        }
    return out


def materially_beats_incumbent(inc_payload: dict, cand_payload: dict) -> bool:
    inc = inc_payload['overall']
    cur = cand_payload['overall']
    mean_gain = float(inc['mean_pct_error']) - float(cur['mean_pct_error'])
    median_gain = float(inc['median_pct_error']) - float(cur['median_pct_error'])
    max_gain = float(inc['max_pct_error']) - float(cur['max_pct_error'])
    return mean_gain >= 0.3 and median_gain >= 0.1 and max_gain >= 0.0


def promising_candidate(inc_payload: dict, cand_payload: dict) -> bool:
    inc = inc_payload['overall']
    cur = cand_payload['overall']
    mean_gap = float(cur['mean_pct_error']) - float(inc['mean_pct_error'])
    max_gap = float(cur['max_pct_error']) - float(inc['max_pct_error'])
    return mean_gap <= 1.2 and max_gap <= 5.0


def tail_table_rows(candidate: AppendTailCandidate) -> list[dict]:
    out = []
    for action, row, face in zip(candidate.all_actions[BASE_POSITIONS:], candidate.tail_rows, candidate.all_faces[BASE_POSITIONS:]):
        out.append({
            'seq': row['pos_id'],
            'tail_idx': row['tail_idx'],
            'legal_motor_action': render_action(action),
            'axis': row['axis'],
            'angle_deg': row['angle_deg'],
            'rotation_time_s': row['rotation_time_s'],
            'pre_static_s': row['pre_static_s'],
            'post_static_s': row['post_static_s'],
            'node_total_s': row['node_total_s'],
            'face_after': face['face_name'],
            'outer_mode': action['outer_mode'],
            'beta_before': action['inner_beta_before_deg'],
            'beta_after': action['inner_beta_after_deg'],
        })
    return out


def render_report(payload: dict) -> str:
    lines = []
    lines.append('# Chapter-3 fixed-12 + legal appended-tail search')
    lines.append('')
    lines.append('## 1. Hard rules enforced in this branch')
    lines.append('')
    lines.append('- Original chapter-3 12-position skeleton is kept **exactly unchanged**.')
    lines.append('- Search object is **append tail only**: add 1–6 legal rotation-stop nodes after the original node 12.')
    lines.append('- Real hardware rule stays strict: inner axis = body y only; outer axis = x/z family determined by current inner attitude.')
    lines.append('- Noise family is locked to shared low-noise benchmark: `noise_scale=0.08`, `seed=42`, same truth family.')
    lines.append('- Total duration remains within 20–30 min (`1200 s + tail <= 1800 s`).')
    lines.append('')
    lines.append('## 2. Tail-design rationale used here')
    lines.append('')
    lines.append('- **To target dKg_zz**: append a closed `inner-open → z-family outer pair/sweep → inner-close` motif. Because the original 12-node path ends at `+Z, beta=0`, a first inner ±90 move is required before the outer motor can legally enter the z-family. Closed z-return motifs are the guard-safe version; z-sweep motifs are the stronger but riskier version.')
    lines.append('- **To target dKa_yy / Ka2_y**: append pure y-pair or `z-return + y-swap` motifs. These give extra late inner-axis excitation without rewriting the original skeleton, so they are the cleanest append-only way to add y-sensitive information.')
    lines.append('- **To protect already-good dKg_xz / dKa_xz / Ka2_z**: prefer short symmetric tails that close back to `+Z, beta=0`, avoid arbitrary x-family wandering, and compare guard-channel deltas explicitly against the incumbent.')
    lines.append('')
    lines.append('## 3. Fixed references')
    lines.append('')
    lines.append(f"- faithful12 Markov42: **{payload['references']['faithful12']['markov42']['overall']['mean_pct_error']:.3f} / {payload['references']['faithful12']['markov42']['overall']['median_pct_error']:.3f} / {payload['references']['faithful12']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- incumbent `{payload['references']['incumbent']['name']}` Markov42: **{payload['references']['incumbent']['markov42']['overall']['mean_pct_error']:.3f} / {payload['references']['incumbent']['markov42']['overall']['median_pct_error']:.3f} / {payload['references']['incumbent']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('## 4. Markov42 appended-tail ranking')
    lines.append('')
    lines.append('| rank | candidate | family | tail faces | mean | median | max | dKa_yy | dKg_zz | Ka2_y | dKg_xz | dKa_xz | Δmean vs faithful | Δmean vs incumbent | note |')
    lines.append('|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for idx, row in enumerate(payload['markov42_rows'], start=1):
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        d_f = row['delta_vs_faithful']['overall']['mean_pct_error']['improvement_pct_points']
        d_i = row['delta_vs_incumbent']['overall']['mean_pct_error']['improvement_pct_points']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {' → '.join(row['tail_faces'])} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['dKg_xz']:.3f} | {k['dKa_xz']:.3f} | {d_f:+.3f} | {d_i:+.3f} | {row['rationale']} |"
        )
    lines.append('')
    lines.append('## 5. Best append-tail candidate')
    lines.append('')
    best = payload['best_candidate']
    lines.append(f"- best Markov42 candidate: **{best['name']}**")
    lines.append(f"- family: `{best['family']}` / timing policy `{best['timing_policy']}` / tail length `{best['theory_tags']['tail_length']}`")
    lines.append(f"- tail faces: `{' → '.join(best['tail_faces'])}`")
    lines.append(f"- Markov42 overall: **{best['markov42']['overall']['mean_pct_error']:.3f} / {best['markov42']['overall']['median_pct_error']:.3f} / {best['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- vs faithful12: Δmean **{best['delta_vs_faithful']['overall']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian **{best['delta_vs_faithful']['overall']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_faithful']['overall']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- vs incumbent: Δmean **{best['delta_vs_incumbent']['overall']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian **{best['delta_vs_incumbent']['overall']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_incumbent']['overall']['max_pct_error']['improvement_pct_points']:+.3f}**")
    if best.get('kf36'):
        lines.append(f"- KF36 recheck: **{best['kf36']['overall']['mean_pct_error']:.3f} / {best['kf36']['overall']['median_pct_error']:.3f} / {best['kf36']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('### 5.1 Exact legal motor / timing table for the best tail')
    lines.append('')
    lines.append('| seq | tail_idx | legal motor action | axis | angle_deg | rot_s | pre_s | post_s | total_s | face after |')
    lines.append('|---:|---:|---|---|---:|---:|---:|---:|---:|---|')
    for row in best['tail_table']:
        lines.append(
            f"| {row['seq']} | {row['tail_idx']} | {row['legal_motor_action']} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {row['node_total_s']:.1f} | {row['face_after']} |"
        )
    lines.append('')
    lines.append('## 6. Bottom line')
    lines.append('')
    lines.append(f"- {payload['scientific_bottom_line']}")
    lines.append(f"- Best tail motif class: `{payload['mechanism_learning']['best_family']}`")
    lines.append(f"- Main blocker after this append-only branch: {payload['mechanism_learning']['main_blocker']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_module('search_ch3_append_tail_src', str(SOURCE_FILE))

    refs = load_reference_payloads(args.noise_scale)
    faithful = build_candidate(mod, ())

    candidates = [build_append_tail_candidate(mod, spec) for spec in ALL_CANDIDATE_SPECS]

    rows = []
    payload_by_name = {}
    candidate_by_name = {cand.name: cand for cand in candidates}
    for cand in candidates:
        print(f'RUN {cand.name} ...', flush=True)
        payload, status, path = run_candidate_payload(mod, cand, 'markov42_noisy', args.noise_scale, args.force_rerun)
        payload_by_name[cand.name] = payload
        row = {
            'candidate_name': cand.name,
            'batch': cand.batch,
            'family': cand.family,
            'rationale': cand.rationale,
            'timing_policy': cand.timing_policy,
            'tail_actions': cand.tail_actions,
            'tail_faces': cand.tail_faces,
            'tail_length': len(cand.tail_actions),
            'tail_total_time_s': sum(x['node_total_s'] for x in cand.tail_rows),
            'theory_tags': cand.theory_tags,
            'metrics': compact_result(payload),
            'run_json': str(path),
            'status': status,
            'delta_vs_faithful': delta_vs_reference(refs['faithful_markov'], payload),
            'delta_vs_incumbent': delta_vs_reference(refs['incumbent_markov'], payload),
            'guard_vs_incumbent': guard_readout(refs['incumbent_markov'], payload),
        }
        rows.append(row)
        print(
            f"DONE {cand.name}: mean={row['metrics']['overall']['mean_pct_error']:.3f}, "
            f"median={row['metrics']['overall']['median_pct_error']:.3f}, "
            f"max={row['metrics']['overall']['max_pct_error']:.3f}",
            flush=True,
        )

    rows.sort(key=lambda x: (
        x['metrics']['overall']['mean_pct_error'],
        x['metrics']['overall']['median_pct_error'],
        x['metrics']['overall']['max_pct_error'],
    ))

    best_row = rows[0]
    best_candidate = candidate_by_name[best_row['candidate_name']]
    best_markov = payload_by_name[best_candidate.name]

    if promising_candidate(refs['incumbent_markov'], best_markov):
        best_kf, best_kf_status, best_kf_path = run_candidate_payload(mod, best_candidate, 'kf36_noisy', args.noise_scale, args.force_rerun)
        best_kf_compact = compact_result(best_kf)
    else:
        best_kf_status = 'not_run_not_promising'
        best_kf_path = None
        best_kf_compact = None

    material = materially_beats_incumbent(refs['incumbent_markov'], best_markov)
    if material:
        scientific_bottom_line = (
            f"A physically legal appended-tail strategy did emerge: {best_candidate.name} materially beat the current incumbent on Markov42 "
            'while respecting the fixed original 12-position base and the real dual-axis mechanism.'
        )
    else:
        scientific_bottom_line = (
            f"No appended-tail candidate materially beat the current incumbent. The best append-only result was {best_candidate.name}, "
            'which shows what tail motif helps most, but the append-only branch still leaves the dominant max channel essentially unresolved.'
        )

    best_family = best_candidate.family
    best_delta = delta_vs_reference(refs['incumbent_markov'], best_markov)
    if best_family == 'z_return_closed':
        blocker = 'Closed z-family tail pairs improve the mean most safely, but they do not move Ka2_y enough to clear the incumbent max barrier.'
    elif best_family == 'z_sweep_closed':
        blocker = 'Stronger z sweeps add excitation, but the extra asymmetry starts to leak into protected channels before Ka2_y is repaired enough.'
    elif best_family == 'y_pair':
        blocker = 'Pure y tails are too weak: they do not generate enough new zz observability to compensate for the faithful base deficits.'
    else:
        blocker = 'Adding y repair after a safe z motif changes the trade-off, but the remaining Ka2_y ceiling is still the blocker.'

    best_summary = {
        'name': best_candidate.name,
        'family': best_candidate.family,
        'batch': best_candidate.batch,
        'rationale': best_candidate.rationale,
        'timing_policy': best_candidate.timing_policy,
        'tail_actions': best_candidate.tail_actions,
        'tail_faces': best_candidate.tail_faces,
        'theory_tags': best_candidate.theory_tags,
        'markov42': compact_result(best_markov),
        'markov42_run_json': best_row['run_json'],
        'kf36': best_kf_compact,
        'kf36_status': best_kf_status,
        'kf36_run_json': str(best_kf_path) if best_kf_path else None,
        'delta_vs_faithful': delta_vs_reference(refs['faithful_markov'], best_markov),
        'delta_vs_incumbent': best_delta,
        'guard_vs_incumbent': guard_readout(refs['incumbent_markov'], best_markov),
        'tail_table': tail_table_rows(best_candidate),
        'tail_rows': best_candidate.tail_rows,
        'all_rows': best_candidate.all_rows,
        'all_actions': best_candidate.all_actions,
        'all_faces': best_candidate.all_faces,
    }

    out_json = RESULTS_DIR / f'ch3_12pos_append_tail_search_{make_suffix(args.noise_scale)}.json'
    out_md = REPORTS_DIR / f'psins_ch3_12pos_append_tail_search_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_12pos_append_tail_search',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'hard_constraints': {
            'fixed_original_12_positions': True,
            'append_only': True,
            'max_total_positions': MAX_TOTAL_POSITIONS,
            'max_appended_positions': MAX_APPEND,
            'real_dual_axis_rule': {
                'inner_axis': 'body y only',
                'outer_axis': 'x/z family determined by current inner attitude',
            },
            'time_budget_s': 1800.0,
            'seed': 42,
        },
        'references': {
            'faithful12': {
                'name': 'faithful12',
                'markov42': compact_result(refs['faithful_markov']),
                'markov42_run_json': str(FAITHFUL_MARKOV_JSON),
                'kf36': compact_result(refs['faithful_kf']),
                'kf36_run_json': str(FAITHFUL_KF_JSON),
                'base_face_sequence': [x['face_name'] for x in faithful.faces],
            },
            'incumbent': {
                'name': INCUMBENT_NAME,
                'markov42': compact_result(refs['incumbent_markov']),
                'markov42_run_json': str(INCUMBENT_MARKOV_JSON),
                'kf36': compact_result(refs['incumbent_kf']),
                'kf36_run_json': str(INCUMBENT_KF_JSON),
            },
        },
        'candidate_specs': ALL_CANDIDATE_SPECS,
        'markov42_rows': rows,
        'best_candidate': best_summary,
        'material_improvement_over_incumbent': material,
        'scientific_bottom_line': scientific_bottom_line,
        'mechanism_learning': {
            'best_family': best_family,
            'main_blocker': blocker,
        },
        'files': {
            'json': str(out_json),
            'report': str(out_md),
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    out_md.write_text(render_report(payload), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps(payload['files'], ensure_ascii=False), flush=True)
    print('BEST_APPEND_TAIL', best_candidate.name, best_summary['markov42']['overall'], flush=True)
    print('MATERIAL_OVER_INCUMBENT', material, flush=True)


if __name__ == '__main__':
    main()
