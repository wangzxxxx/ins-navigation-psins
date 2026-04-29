from __future__ import annotations

import argparse
import json
import sys
import types
from collections import Counter
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

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / 'tmp_psins_py') not in sys.path:
    sys.path.insert(0, str(ROOT / 'tmp_psins_py'))
if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import load_module
from compare_ch3_12pos_path_baselines import (
    build_ch3_path_paras,
    build_dataset_with_path,
    build_default_path_paras,
    path_method_output_path,
)
from compare_four_methods_shared_noise import _load_json, _noise_matches, compute_payload, expected_noise_config
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs

FACE_NAMES = {
    (1, 0, 0): '+X',
    (-1, 0, 0): '-X',
    (0, 1, 0): '+Y',
    (0, -1, 0): '-Y',
    (0, 0, 1): '+Z',
    (0, 0, -1): '-Z',
}
FACE_AXIS_SIGN = {
    '+X': ('X', +1),
    '-X': ('X', -1),
    '+Y': ('Y', +1),
    '-Y': ('Y', -1),
    '+Z': ('Z', +1),
    '-Z': ('Z', -1),
}
ALL_FACES = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']
KEY_PARAMS = ['dKa_yy', 'dKg_zz', 'Ka2_y', 'Ka2_z', 'dKg_xz', 'dKa_xz']
METHODS = ['markov42_noisy', 'kf36_noisy']
METHOD_LABELS = {
    'markov42_noisy': 'Markov42',
    'kf36_noisy': 'KF36',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=0.08)
    parser.add_argument('--report-date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--methods', nargs='*', default=METHODS)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def make_suffix(noise_scale: float) -> str:
    if abs(noise_scale - 0.08) < 1e-12:
        return 'noise0p08'
    if abs(noise_scale - 1.0) < 1e-12:
        return 'noise1p0'
    if abs(noise_scale - 2.0) < 1e-12:
        return 'noise2p0'
    return f"noise{str(noise_scale).replace('.', 'p')}"


def paras_to_rows(mod, paras):
    rows = []
    for row in paras.tolist():
        rows.append({
            'pos_id': int(row[0]),
            'axis': [int(row[1]), int(row[2]), int(row[3])],
            'angle_deg': float(row[4] / mod.glv.deg),
            'rotation_time_s': float(row[5]),
            'pre_static_s': float(row[6]),
            'post_static_s': float(row[7]),
            'node_total_s': float(row[5] + row[6] + row[7]),
        })
    return rows


def rows_to_paras(mod, rows):
    arr = []
    for idx, row in enumerate(rows, start=1):
        arr.append([
            idx,
            row['axis'][0],
            row['axis'][1],
            row['axis'][2],
            row['angle_deg'],
            row['rotation_time_s'],
            row['pre_static_s'],
            row['post_static_s'],
        ])
    paras = mod.np.array(arr, dtype=float)
    paras[:, 4] = paras[:, 4] * mod.glv.deg
    return paras


def nearest_face(vec):
    rounded = [0, 0, 0]
    idx = int(max(range(3), key=lambda i: abs(vec[i])))
    rounded[idx] = 1 if vec[idx] >= 0 else -1
    key = tuple(rounded)
    return FACE_NAMES[key], key


def orientation_faces(mod, paras):
    from psins_py.math_utils import rv2m

    C = mod.np.eye(3)
    out = []
    for row in paras:
        axis = mod.np.asarray(row[1:4], dtype=float)
        axis = axis / mod.np.linalg.norm(axis)
        C = C @ rv2m(axis * row[4])
        g_body = C.T @ mod.np.array([0.0, 0.0, 1.0])
        face_name, _ = nearest_face(g_body.tolist())
        out.append({
            'pos_id': int(row[0]),
            'axis': [int(row[1]), int(row[2]), int(row[3])],
            'angle_deg': float(row[4] / mod.glv.deg),
            'face_name': face_name,
            'gravity_body': [float(x) for x in g_body.tolist()],
        })
    return out


def structural_summary(rows, face_rows):
    counts = Counter([f['face_name'] for f in face_rows])
    axis_balance = {'X': 0, 'Y': 0, 'Z': 0}
    first_cover_index = {face: None for face in ALL_FACES}
    late_repeat_nodes = []
    seq = []
    seen = Counter()
    full_cover_index = None
    repeat_and_worsen_target_nodes = []

    def target_penalty(balance: dict[str, int]) -> int:
        return 2 * abs(balance['Y']) + 2 * abs(balance['Z']) + abs(balance['X'])

    prev_penalty = 0
    for idx, (row, face_row) in enumerate(zip(rows, face_rows), start=1):
        face = face_row['face_name']
        seq.append(face)
        axis, sign = FACE_AXIS_SIGN[face]
        already_seen = seen[face] > 0
        seen[face] += 1
        axis_balance[axis] += sign
        if first_cover_index[face] is None:
            first_cover_index[face] = idx
        if full_cover_index is None and all(first_cover_index[f] is not None for f in ALL_FACES):
            full_cover_index = idx
        now_penalty = target_penalty(axis_balance)
        if full_cover_index is not None and idx > full_cover_index and already_seen:
            late_repeat_nodes.append(idx)
        if already_seen and now_penalty > prev_penalty:
            repeat_and_worsen_target_nodes.append(idx)
        prev_penalty = now_penalty

    return {
        'face_sequence': seq,
        'face_counts': dict(counts),
        'axis_balance': axis_balance,
        'first_cover_index': first_cover_index,
        'full_cover_index': full_cover_index,
        'late_repeat_nodes': late_repeat_nodes,
        'repeat_and_worsen_target_nodes': repeat_and_worsen_target_nodes,
        'total_time_s': float(sum(r['node_total_s'] for r in rows)),
    }


def summarize_path(mod, name: str, kind: str, rationale: str, paras) -> dict:
    rows = paras_to_rows(mod, paras)
    faces = orientation_faces(mod, paras)
    return {
        'name': name,
        'kind': kind,
        'rationale': rationale,
        'rows': rows,
        'faces': faces,
        'structural_summary': structural_summary(rows, faces),
    }


def build_repair_candidates(mod):
    original_rows = paras_to_rows(mod, build_ch3_path_paras(mod))

    rows_A = [dict(r) for r in original_rows]
    for row in rows_A:
        if row['pos_id'] in [10, 11]:
            row['pre_static_s'] = 30.0
            row['post_static_s'] = 60.0
            row['node_total_s'] = float(row['rotation_time_s'] + row['pre_static_s'] + row['post_static_s'])

    rows_B = [dict(r) for r in original_rows]
    rows_B[10]['angle_deg'] *= -1.0

    rows_C = [dict(r) for r in original_rows]
    rows_C[9]['angle_deg'] *= -1.0

    return [
        summarize_path(
            mod,
            'repair_A_more_pre_static_on_late_Z_block',
            'dwell_only',
            'Keep all 12 nodes unchanged; only move more dwell before nodes 10-11 ([10,30,60]) so the late Z-driven block is less post-heavy.',
            rows_to_paras(mod, rows_A),
        ),
        summarize_path(
            mod,
            'repair_B_break_late_Z_plus_pair_by_flipping_node11',
            'single_sign_flip',
            'Flip node 11 only to break the late same-sign Z-pair without changing node count or total time.',
            rows_to_paras(mod, rows_B),
        ),
        summarize_path(
            mod,
            'repair_C_flip_node10_to_add_late_minusY',
            'single_sign_flip',
            'Flip node 10 only so the late +Y revisit becomes -Y, directly targeting the observed +Y overexposure / -Y underexposure while keeping 12 nodes and 1200 s.',
            rows_to_paras(mod, rows_C),
        ),
    ]


def repair_output_path(candidate_name: str, method_key: str, noise_scale: float) -> Path:
    suffix = make_suffix(noise_scale)
    short = {
        'repair_A_more_pre_static_on_late_Z_block': 'ch3goalA_repairA',
        'repair_B_break_late_Z_plus_pair_by_flipping_node11': 'ch3goalA_repairB',
        'repair_C_flip_node10_to_add_late_minusY': 'ch3goalA_repairC',
    }[candidate_name]
    if method_key == 'markov42_noisy':
        return RESULTS_DIR / f'M_markov_42state_gm1_{short}_shared_{suffix}_param_errors.json'
    if method_key == 'kf36_noisy':
        return RESULTS_DIR / f'KF36_{short}_shared_{suffix}_param_errors.json'
    raise KeyError(method_key)


def load_or_run_payload(mod, candidate_name: str, paras, method_key: str, noise_scale: float, force_rerun: bool = False):
    expected_cfg = expected_noise_config(noise_scale)
    if candidate_name in ('default18_reference', 'chapter3_original'):
        base_key = 'default_path' if candidate_name == 'default18_reference' else 'chapter3_12pos_reconstructed'
        out_path = path_method_output_path(base_key, method_key, noise_scale)
        if out_path.exists():
            payload = _load_json(out_path)
            if _noise_matches(payload, expected_cfg):
                return payload, 'reused_verified', out_path
        # fall through to recompute if needed
    else:
        out_path = repair_output_path(candidate_name, method_key, noise_scale)
        if (not force_rerun) and out_path.exists():
            payload = _load_json(out_path)
            if _noise_matches(payload, expected_cfg) and payload.get('extra', {}).get('candidate_name') == candidate_name:
                return payload, 'reused_verified', out_path

    dataset = build_dataset_with_path(mod, noise_scale, paras)
    params = _param_specs(mod)
    if method_key == 'kf36_noisy':
        clbt, _, _, _, _ = mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=36,
            label=f'{candidate_name}_{method_key}_{make_suffix(noise_scale)}',
        )
    elif method_key == 'markov42_noisy':
        clbt, _, _, _, _ = mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=42,
            bi_g=dataset['bi_g'],
            tau_g=dataset['tau_g'],
            bi_a=dataset['bi_a'],
            tau_a=dataset['tau_a'],
            label=f'{candidate_name}_{method_key}_{make_suffix(noise_scale)}',
        )
    else:
        raise KeyError(method_key)

    payload = compute_payload(
        mod,
        clbt,
        params,
        variant=f'{candidate_name}_{method_key}_{make_suffix(noise_scale)}',
        method_file='benchmark_ch3_12pos_goalA_repairs.py',
        extra={
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'comparison_mode': 'ch3_goalA_repair_benchmark',
            'candidate_name': candidate_name,
            'method_key': method_key,
        },
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', out_path


def compact_result(payload: dict) -> dict:
    return {
        'overall': {
            'mean_pct_error': float(payload['overall']['mean_pct_error']),
            'median_pct_error': float(payload['overall']['median_pct_error']),
            'max_pct_error': float(payload['overall']['max_pct_error']),
        },
        'key_param_errors': {k: float(payload['param_errors'][k]['pct_error']) for k in KEY_PARAMS},
    }


def gap_closed(candidate_value: float, original_value: float, default_value: float) -> float:
    denom = original_value - default_value
    if abs(denom) < 1e-12:
        return 0.0
    return float((original_value - candidate_value) / denom)


def build_method_summary(rows: list[dict], method_key: str) -> dict:
    by_name = {row['candidate_name']: row for row in rows}
    default = by_name['default18_reference']['metrics']['overall']
    original = by_name['chapter3_original']['metrics']['overall']
    ranked = []
    for row in rows:
        overall = row['metrics']['overall']
        extra = {
            'delta_mean_vs_original': float(overall['mean_pct_error'] - original['mean_pct_error']),
            'delta_median_vs_original': float(overall['median_pct_error'] - original['median_pct_error']),
            'delta_max_vs_original': float(overall['max_pct_error'] - original['max_pct_error']),
            'mean_gap_closed_frac': gap_closed(overall['mean_pct_error'], original['mean_pct_error'], default['mean_pct_error']),
            'median_gap_closed_frac': gap_closed(overall['median_pct_error'], original['median_pct_error'], default['median_pct_error']),
            'max_gap_closed_frac': gap_closed(overall['max_pct_error'], original['max_pct_error'], default['max_pct_error']),
        }
        ranked.append(row | {'comparison': extra})
    ranked.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    best_repair = None
    for item in ranked:
        if item['candidate_name'].startswith('repair_'):
            best_repair = item
            break
    return {
        'method_key': method_key,
        'ranked': ranked,
        'best_repair': best_repair,
    }


def render_report(payload: dict) -> str:
    lines = []
    lines.append('# Chapter-3 faithful 12-position Goal-A minimal repair benchmark')
    lines.append('')
    lines.append('## 1. Bottom line')
    lines.append('')
    markov = payload['method_summaries']['markov42_noisy']
    best = markov['best_repair']
    bm = best['metrics']['overall']
    bc = best['comparison']
    lines.append(f"- Markov42 best repair in this narrow batch: **{best['candidate_name']}**.")
    lines.append(f"- On Markov42, best repair overall mean/median/max = **{bm['mean_pct_error']:.3f} / {bm['median_pct_error']:.3f} / {bm['max_pct_error']:.3f}**.")
    lines.append(f"- Gap closed vs original on Markov42: mean **{100.0*bc['mean_gap_closed_frac']:.1f}%**, max **{100.0*bc['max_gap_closed_frac']:.1f}%**.")
    lines.append(f"- Default18 reference remains at **{payload['method_summaries']['markov42_noisy']['ranked'][0]['metrics']['overall']['mean_pct_error']:.3f} / {payload['method_summaries']['markov42_noisy']['ranked'][0]['metrics']['overall']['median_pct_error']:.3f} / {payload['method_summaries']['markov42_noisy']['ranked'][0]['metrics']['overall']['max_pct_error']:.3f}** only if that first row is default; see full table below.")
    lines.append('')

    lines.append('## 2. Candidate structure snapshot')
    lines.append('')
    lines.append('| candidate | kind | total_time_s | face_sequence | face_counts | axis_balance | full_cover | late_repeat_nodes |')
    lines.append('|---|---|---:|---|---|---|---:|---|')
    for cand in payload['candidates']:
        s = cand['structural_summary']
        lines.append(
            f"| {cand['name']} | {cand['kind']} | {s['total_time_s']:.0f} | {' → '.join(s['face_sequence'])} | `{json.dumps(s['face_counts'], ensure_ascii=False)}` | `{json.dumps(s['axis_balance'], ensure_ascii=False)}` | {s['full_cover_index']} | {s['late_repeat_nodes']} |"
        )
    lines.append('')

    lines.append('## 3. Benchmark table')
    lines.append('')
    lines.append('| method | candidate | mean | median | max | Δmean vs original | Δmax vs original | mean gap closed | max gap closed | dKa_yy | dKg_zz | Ka2_y | Ka2_z | source |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for method_key, summary in payload['method_summaries'].items():
        for item in summary['ranked']:
            o = item['metrics']['overall']
            c = item['comparison']
            k = item['metrics']['key_param_errors']
            lines.append(
                f"| {METHOD_LABELS[method_key]} | {item['candidate_name']} | {o['mean_pct_error']:.3f} | {o['median_pct_error']:.3f} | {o['max_pct_error']:.3f} | {c['delta_mean_vs_original']:+.3f} | {c['delta_max_vs_original']:+.3f} | {100.0*c['mean_gap_closed_frac']:.1f}% | {100.0*c['max_gap_closed_frac']:.1f}% | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {item['run_source']} |"
            )
    lines.append('')

    lines.append('## 4. Exact repaired path tables')
    lines.append('')
    for cand in payload['candidates']:
        lines.append(f"### {cand['name']}")
        lines.append('')
        lines.append(f"- rationale: {cand['rationale']}")
        lines.append('')
        lines.append('| node | axis | angle_deg | rot_s | pre_s | post_s | face |')
        lines.append('|---:|---|---:|---:|---:|---:|---|')
        for row, face in zip(cand['rows'], cand['faces']):
            lines.append(
                f"| {row['pos_id']} | {row['axis']} | {row['angle_deg']:.0f} | {row['rotation_time_s']:.0f} | {row['pre_static_s']:.0f} | {row['post_static_s']:.0f} | {face['face_name']} |"
            )
        lines.append('')

    lines.append('## 5. Practical verdict')
    lines.append('')
    lines.append(f"- Markov42 best repair: **{payload['method_summaries']['markov42_noisy']['best_repair']['candidate_name']}**.")
    lines.append(f"- KF36 best repair: **{payload['method_summaries'].get('kf36_noisy', {}).get('best_repair', {}).get('candidate_name', 'not_run')}**.")
    lines.append(f"- Conclusion: **{payload['overall_conclusion']}**")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module('benchmark_ch3_goalA_src', str(SOURCE_FILE))
    noise_cfg = expected_noise_config(args.noise_scale)

    default = summarize_path(
        mod,
        'default18_reference',
        'reference',
        'Current default 18-position shared benchmark path.',
        build_default_path_paras(mod),
    )
    original = summarize_path(
        mod,
        'chapter3_original',
        'baseline',
        'Faithful chapter-3 12-position reconstruction with [10,10,80] split per node.',
        build_ch3_path_paras(mod),
    )
    candidates = build_repair_candidates(mod)
    compare_set = [default, original] + candidates

    all_rows = []
    run_outputs = {}
    for method_key in args.methods:
        for item in compare_set:
            paras = rows_to_paras(mod, item['rows'])
            payload_run, run_source, out_path = load_or_run_payload(
                mod,
                item['name'],
                paras,
                method_key,
                args.noise_scale,
                force_rerun=args.force_rerun,
            )
            metrics = compact_result(payload_run)
            row = {
                'candidate_name': item['name'],
                'method_key': method_key,
                'metrics': metrics,
                'run_source': run_source,
                'run_json': str(out_path),
            }
            all_rows.append(row)
            run_outputs[(item['name'], method_key)] = row

    method_summaries = {}
    for method_key in args.methods:
        rows = [row for row in all_rows if row['method_key'] == method_key]
        method_summaries[method_key] = build_method_summary(rows, method_key)

    markov_best = method_summaries['markov42_noisy']['best_repair']
    meaningful = (
        markov_best['comparison']['mean_gap_closed_frac'] >= 0.30 or
        markov_best['comparison']['max_gap_closed_frac'] >= 0.30
    )
    if meaningful:
        overall_conclusion = 'some minimal repair narrows the gap meaningfully, but default18 still sets the cleaner reference.'
    else:
        overall_conclusion = 'no tested minimal repair narrows the gap meaningfully; the faithful 12-node skeleton remains far worse than default18 under baseline low noise.'

    payload = {
        'experiment': 'benchmark_ch3_12pos_goalA_repairs',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'noise_config': noise_cfg,
        'methods': args.methods,
        'reference_paths': [default, original],
        'candidates': candidates,
        'comparison_rows': all_rows,
        'method_summaries': method_summaries,
        'overall_conclusion': overall_conclusion,
    }

    suffix = make_suffix(args.noise_scale)
    out_json = RESULTS_DIR / f'ch3_12pos_goalA_repair_compare_{suffix}.json'
    out_md = REPORTS_DIR / f'psins_ch3_12pos_goalA_repair_{args.report_date}_{suffix}.md'
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    out_md.write_text(render_report(payload), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps({
        'json': str(out_json),
        'report': str(out_md),
        'overall_conclusion': overall_conclusion,
        'markov_best_repair': markov_best['candidate_name'],
        'markov_best_mean_gap_closed_frac': markov_best['comparison']['mean_gap_closed_frac'],
        'markov_best_max_gap_closed_frac': markov_best['comparison']['max_gap_closed_frac'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
