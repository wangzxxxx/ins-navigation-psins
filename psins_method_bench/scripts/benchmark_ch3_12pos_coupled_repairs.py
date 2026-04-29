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
METHOD_LABELS = {
    'markov42_noisy': 'Markov42',
    'kf36_noisy': 'KF36',
}
CANDIDATE_FILE_TAGS = {
    'coupled_A_flip10_11_retime': 'ch3coupledA_flip1011_retime',
    'coupled_B_move7to4_move12to10': 'ch3coupledB_move7to4_move12to10',
    'coupled_C_move7to4_move12to10_retime': 'ch3coupledC_move7to4_move12to10_retime',
    'coupled_D_move7to4_move10to8': 'ch3coupledD_move7to4_move10to8',
    'coupled_E_move7to4_move10to8_retime': 'ch3coupledE_move7to4_move10to8_retime',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=0.08)
    parser.add_argument('--report-date', default=datetime.now().strftime('%Y-%m-%d'))
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
    for idx, face_row in enumerate(face_rows, start=1):
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



def apply_candidate_recipe(original_rows, *, order=None, flip_old_nodes=(), retime_old_nodes=()):
    if order is None:
        rows = []
        for idx, row in enumerate(original_rows, start=1):
            item = dict(row)
            item['pos_id'] = idx
            item['source_node'] = idx
            rows.append(item)
    else:
        rows = []
        for new_idx, old_idx in enumerate(order, start=1):
            item = dict(original_rows[old_idx - 1])
            item['pos_id'] = new_idx
            item['source_node'] = old_idx
            rows.append(item)

    for row in rows:
        src = row['source_node']
        if src in flip_old_nodes:
            row['angle_deg'] *= -1.0
        if src in retime_old_nodes:
            row['pre_static_s'] = 30.0
            row['post_static_s'] = 60.0
            row['node_total_s'] = float(row['rotation_time_s'] + row['pre_static_s'] + row['post_static_s'])
    return rows



def summarize_path(mod, name: str, kind: str, rationale: str, rows) -> dict:
    paras = rows_to_paras(mod, rows)
    faces = orientation_faces(mod, paras)
    return {
        'name': name,
        'kind': kind,
        'rationale': rationale,
        'rows': rows,
        'faces': faces,
        'structural_summary': structural_summary(rows, faces),
    }



def build_coupled_candidates(mod):
    original_rows = paras_to_rows(mod, build_ch3_path_paras(mod))

    candidates = []
    candidates.append(summarize_path(
        mod,
        'coupled_A_flip10_11_retime',
        'local_coupled_late_block',
        'Repair the late harmful block locally: flip old nodes 10 and 11 together to remove the late +Y/+X revisit, then retime both to [10,30,60].',
        apply_candidate_recipe(original_rows, flip_old_nodes=(10, 11), retime_old_nodes=(10, 11)),
    ))
    candidates.append(summarize_path(
        mod,
        'coupled_B_move7to4_move12to10',
        'two_reorders',
        'Bring old node 7 (-Y source) forward to slot 4 for earlier six-face coverage, and bring old node 12 forward to slot 10 so the tail stops ending with the original 10-11-12 harmful pattern.',
        apply_candidate_recipe(original_rows, order=[1, 2, 3, 7, 4, 5, 6, 8, 9, 12, 10, 11]),
    ))
    candidates.append(summarize_path(
        mod,
        'coupled_C_move7to4_move12to10_retime',
        'two_reorders_plus_retime',
        'Same structure as coupled_B, plus retime old nodes 10 and 11 to [10,30,60] to check whether the residual gap is still dwell-sensitive.',
        apply_candidate_recipe(original_rows, order=[1, 2, 3, 7, 4, 5, 6, 8, 9, 12, 10, 11], retime_old_nodes=(10, 11)),
    ))
    candidates.append(summarize_path(
        mod,
        'coupled_D_move7to4_move10to8',
        'two_reorders',
        'Bring old node 7 (-Y source) forward to slot 4, and move old node 10 forward to slot 8 so the old late harmful block is split before the last quarter of the path.',
        apply_candidate_recipe(original_rows, order=[1, 2, 3, 7, 4, 5, 6, 10, 8, 9, 11, 12]),
    ))
    candidates.append(summarize_path(
        mod,
        'coupled_E_move7to4_move10to8_retime',
        'two_reorders_plus_retime',
        'Same structure as coupled_D, plus retime old nodes 10 and 11 to [10,30,60] so the relocated late block is less post-heavy.',
        apply_candidate_recipe(original_rows, order=[1, 2, 3, 7, 4, 5, 6, 10, 8, 9, 11, 12], retime_old_nodes=(10, 11)),
    ))
    return candidates



def candidate_output_path(candidate_name: str, method_key: str, noise_scale: float) -> Path:
    suffix = make_suffix(noise_scale)
    tag = CANDIDATE_FILE_TAGS[candidate_name]
    if method_key == 'markov42_noisy':
        return RESULTS_DIR / f'M_markov_42state_gm1_{tag}_shared_{suffix}_param_errors.json'
    if method_key == 'kf36_noisy':
        return RESULTS_DIR / f'KF36_{tag}_shared_{suffix}_param_errors.json'
    raise KeyError(method_key)



def load_or_run_payload(mod, candidate_name: str, paras, method_key: str, noise_scale: float, force_rerun: bool = False):
    expected_cfg = expected_noise_config(noise_scale)
    if candidate_name == 'default18_reference':
        out_path = path_method_output_path('default_path', method_key, noise_scale)
    elif candidate_name == 'chapter3_original':
        out_path = path_method_output_path('chapter3_12pos_reconstructed', method_key, noise_scale)
    else:
        out_path = candidate_output_path(candidate_name, method_key, noise_scale)

    if (not force_rerun) and out_path.exists():
        payload = _load_json(out_path)
        if _noise_matches(payload, expected_cfg):
            extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
            if candidate_name in ('default18_reference', 'chapter3_original') or extra.get('candidate_name') == candidate_name:
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
        method_file='benchmark_ch3_12pos_coupled_repairs.py',
        extra={
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'comparison_mode': 'ch3_12pos_coupled_repairs',
            'candidate_name': candidate_name,
            'method_key': method_key,
        },
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', out_path



def compact_result(payload: dict) -> dict:
    top_param_name, top_param_payload = max(payload['param_errors'].items(), key=lambda kv: float(kv[1]['pct_error']))
    return {
        'overall': {
            'mean_pct_error': float(payload['overall']['mean_pct_error']),
            'median_pct_error': float(payload['overall']['median_pct_error']),
            'max_pct_error': float(payload['overall']['max_pct_error']),
        },
        'key_param_errors': {k: float(payload['param_errors'][k]['pct_error']) for k in KEY_PARAMS},
        'top_param': {
            'name': top_param_name,
            'pct_error': float(top_param_payload['pct_error']),
        },
    }



def gap_closed(candidate_value: float, original_value: float, default_value: float) -> float:
    denom = original_value - default_value
    if abs(denom) < 1e-12:
        return 0.0
    return float((original_value - candidate_value) / denom)



def enrich_comparison(row: dict, original: dict, default: dict) -> dict:
    overall = row['metrics']['overall']
    return row | {
        'comparison': {
            'delta_mean_vs_original': float(overall['mean_pct_error'] - original['mean_pct_error']),
            'delta_median_vs_original': float(overall['median_pct_error'] - original['median_pct_error']),
            'delta_max_vs_original': float(overall['max_pct_error'] - original['max_pct_error']),
            'mean_gap_closed_frac': gap_closed(overall['mean_pct_error'], original['mean_pct_error'], default['mean_pct_error']),
            'median_gap_closed_frac': gap_closed(overall['median_pct_error'], original['median_pct_error'], default['median_pct_error']),
            'max_gap_closed_frac': gap_closed(overall['max_pct_error'], original['max_pct_error'], default['max_pct_error']),
            'mean_reduction_vs_original_frac': float((original['mean_pct_error'] - overall['mean_pct_error']) / original['mean_pct_error']),
            'max_reduction_vs_original_frac': float((original['max_pct_error'] - overall['max_pct_error']) / original['max_pct_error']),
        }
    }



def practical_usable(best_repair: dict) -> bool:
    cmp = best_repair['comparison']
    overall = best_repair['metrics']['overall']
    return (
        cmp['mean_gap_closed_frac'] >= 0.70 and
        cmp['max_gap_closed_frac'] >= 0.70 and
        overall['mean_pct_error'] <= 5.0 and
        overall['max_pct_error'] <= 40.0
    )



def render_report(payload: dict) -> str:
    lines = []
    markov = payload['method_summaries']['markov42_noisy']
    best = markov['best_repair']
    b_overall = best['metrics']['overall']
    b_cmp = best['comparison']

    lines.append('# Chapter-3 faithful 12-position coupled repair benchmark')
    lines.append('')
    lines.append('## 1. Bottom line')
    lines.append('')
    lines.append('- Tested a narrow coupled batch only: 5 informed candidates, all still 12 nodes / 1200 s / noise0p08.')
    lines.append(f"- Best Markov42 candidate: **{best['candidate_name']}**.")
    lines.append(f"- Best Markov42 overall mean / median / max = **{b_overall['mean_pct_error']:.3f} / {b_overall['median_pct_error']:.3f} / {b_overall['max_pct_error']:.3f}**.")
    lines.append(f"- Gap closed vs original (relative to default18): mean **{100.0 * b_cmp['mean_gap_closed_frac']:.1f}%**, max **{100.0 * b_cmp['max_gap_closed_frac']:.1f}%**.")
    if payload['practical_verdict']['usable']:
        lines.append('- Practical verdict: **usable in the limited Goal-A sense** — the repaired 12-node path is still worse than default18, but it is no longer catastrophically broken like the original faithful path.')
    else:
        lines.append('- Practical verdict: **not usable** — none of the coupled repairs reduces the original gap enough to justify keeping the chapter-3 12-node skeleton.')
    lines.append('')

    lines.append('## 2. Operational definition of "usable" in this pass')
    lines.append('')
    lines.append('- still exactly 12 nodes and 1200 s;')
    lines.append('- on Markov42 @ noise0p08, closes at least 70% of both the original→default18 mean gap and max gap;')
    lines.append('- and lands at mean <= 5%, max <= 40%.')
    lines.append('')

    lines.append('## 3. Structural snapshot of coupled candidates')
    lines.append('')
    lines.append('| candidate | kind | total_time_s | face_sequence | axis_balance | full_cover | late_repeat_nodes | repeat_worsen_nodes |')
    lines.append('|---|---|---:|---|---|---:|---|---|')
    for cand in payload['candidates']:
        s = cand['structural_summary']
        lines.append(
            f"| {cand['name']} | {cand['kind']} | {s['total_time_s']:.0f} | {' → '.join(s['face_sequence'])} | `{json.dumps(s['axis_balance'], ensure_ascii=False)}` | {s['full_cover_index']} | {s['late_repeat_nodes']} | {s['repeat_and_worsen_target_nodes']} |"
        )
    lines.append('')

    lines.append('## 4. Markov42 benchmark table')
    lines.append('')
    lines.append('| candidate | mean | median | max | Δmean vs orig | Δmax vs orig | mean gap closed | max gap closed | max param | dKa_yy | dKg_zz | Ka2_y | Ka2_z | source |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|')
    for item in markov['ranked']:
        o = item['metrics']['overall']
        c = item['comparison']
        k = item['metrics']['key_param_errors']
        top = item['metrics']['top_param']
        lines.append(
            f"| {item['candidate_name']} | {o['mean_pct_error']:.3f} | {o['median_pct_error']:.3f} | {o['max_pct_error']:.3f} | {c['delta_mean_vs_original']:+.3f} | {c['delta_max_vs_original']:+.3f} | {100.0 * c['mean_gap_closed_frac']:.1f}% | {100.0 * c['max_gap_closed_frac']:.1f}% | {top['name']}={top['pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {item['run_source']} |"
        )
    lines.append('')

    if 'kf36_recheck' in payload:
        lines.append('## 5. KF36 recheck (reference + best coupled candidate only)')
        lines.append('')
        lines.append('| candidate | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Ka2_z | source |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---|')
        for item in payload['kf36_recheck']['rows']:
            o = item['metrics']['overall']
            k = item['metrics']['key_param_errors']
            lines.append(
                f"| {item['candidate_name']} | {o['mean_pct_error']:.3f} | {o['median_pct_error']:.3f} | {o['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {item['run_source']} |"
            )
        lines.append('')

    lines.append('## 6. Exact path tables')
    lines.append('')
    for cand in payload['candidates']:
        lines.append(f"### {cand['name']}")
        lines.append('')
        lines.append(f"- rationale: {cand['rationale']}")
        lines.append('')
        lines.append('| new_node | from_original_node | axis | angle_deg | rot_s | pre_s | post_s | face |')
        lines.append('|---:|---:|---|---:|---:|---:|---:|---|')
        for row, face in zip(cand['rows'], cand['faces']):
            lines.append(
                f"| {row['pos_id']} | {row['source_node']} | {row['axis']} | {row['angle_deg']:.0f} | {row['rotation_time_s']:.0f} | {row['pre_static_s']:.0f} | {row['post_static_s']:.0f} | {face['face_name']} |"
            )
        lines.append('')

    lines.append('## 7. Interpretation')
    lines.append('')
    lines.append('- `coupled_A` proves that only repairing the late 10-11 block locally is not enough: it cuts the original max badly, but the path still stays far from default18 on mean and median.')
    lines.append('- `coupled_B/C` show that balancing Y by itself is also not enough when it is bought by collapsing `-Z/+Z` diversity too much; they remove the original `dKa_yy` disaster, but `Ka2_y` becomes the new tail.')
    lines.append('- The first genuinely practical signal appears only when the repair is **structural and coupled**: move old node 7 earlier **and** split old node 10 out of the late harmful tail (`coupled_D/E`).')
    lines.append('- `coupled_D` vs `coupled_E` are nearly tied, so the main win is structural reordering; retiming helps only slightly once the structure is fixed.')
    lines.append('')
    lines.append(f"**Final conclusion:** {payload['overall_conclusion']}")
    lines.append('')
    return '\n'.join(lines) + '\n'



def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module('benchmark_ch3_12pos_coupled_repairs_src', str(SOURCE_FILE))
    noise_cfg = expected_noise_config(args.noise_scale)

    default = summarize_path(
        mod,
        'default18_reference',
        'reference',
        'Current default 18-position shared benchmark path.',
        paras_to_rows(mod, build_default_path_paras(mod)),
    )
    original = summarize_path(
        mod,
        'chapter3_original',
        'baseline',
        'Faithful chapter-3 12-position reconstruction with [10,10,80] split per node.',
        paras_to_rows(mod, build_ch3_path_paras(mod)),
    )
    candidates = build_coupled_candidates(mod)
    compare_set = [default, original] + candidates

    markov_rows = []
    for item in compare_set:
        paras = rows_to_paras(mod, item['rows'])
        payload_run, run_source, out_path = load_or_run_payload(
            mod,
            item['name'],
            paras,
            'markov42_noisy',
            args.noise_scale,
            force_rerun=args.force_rerun,
        )
        markov_rows.append({
            'candidate_name': item['name'],
            'method_key': 'markov42_noisy',
            'metrics': compact_result(payload_run),
            'run_source': run_source,
            'run_json': str(out_path),
        })

    by_name = {row['candidate_name']: row for row in markov_rows}
    default_overall = by_name['default18_reference']['metrics']['overall']
    original_overall = by_name['chapter3_original']['metrics']['overall']
    markov_ranked = [enrich_comparison(row, original_overall, default_overall) for row in markov_rows]
    markov_ranked.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    markov_best = next(item for item in markov_ranked if item['candidate_name'].startswith('coupled_'))

    # Cheap KF36 recheck only after a promising Markov42 candidate appears.
    kf36_rows = []
    kf36_targets = [default, original, next(c for c in candidates if c['name'] == markov_best['candidate_name'])]
    for item in kf36_targets:
        paras = rows_to_paras(mod, item['rows'])
        payload_run, run_source, out_path = load_or_run_payload(
            mod,
            item['name'],
            paras,
            'kf36_noisy',
            args.noise_scale,
            force_rerun=args.force_rerun,
        )
        kf36_rows.append({
            'candidate_name': item['name'],
            'method_key': 'kf36_noisy',
            'metrics': compact_result(payload_run),
            'run_source': run_source,
            'run_json': str(out_path),
        })

    usable = practical_usable(markov_best)
    if usable:
        overall_conclusion = (
            f"yes — a coupled structural repair can make the faithful 12-node path practically usable. "
            f"The best tested candidate is {markov_best['candidate_name']}, which closes about "
            f"{100.0 * markov_best['comparison']['mean_gap_closed_frac']:.1f}% of the Markov42 mean gap and "
            f"{100.0 * markov_best['comparison']['max_gap_closed_frac']:.1f}% of the max gap versus default18, "
            f"while staying at exactly 12 nodes / 1200 s. It is still worse than default18, but it is no longer in the original broken regime."
        )
    else:
        overall_conclusion = (
            'no — after a reasonable coupled batch, the faithful 12-node skeleton still does not recover enough performance to be called usable; '
            'the current system appears fundamentally mismatched to the original chapter-3 ordering.'
        )

    summary = {
        'method_key': 'markov42_noisy',
        'ranked': markov_ranked,
        'best_repair': markov_best,
    }

    payload = {
        'experiment': 'benchmark_ch3_12pos_coupled_repairs',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'noise_config': noise_cfg,
        'reference_paths': [default, original],
        'candidates': candidates,
        'method_summaries': {
            'markov42_noisy': summary,
        },
        'kf36_recheck': {
            'rows': kf36_rows,
            'best_candidate_name': markov_best['candidate_name'],
        },
        'practical_verdict': {
            'usable': usable,
            'definition': 'Markov42 @ noise0p08 closes >=70% of both mean and max gap vs original relative to default18, with mean <=5% and max <=40%, while preserving 12 nodes / 1200 s.',
        },
        'overall_conclusion': overall_conclusion,
    }

    suffix = make_suffix(args.noise_scale)
    out_json = RESULTS_DIR / f'ch3_12pos_coupled_repairs_{suffix}.json'
    out_md = REPORTS_DIR / f'psins_ch3_12pos_coupled_repairs_{args.report_date}_{suffix}.md'
    payload['files'] = {
        'json': str(out_json),
        'report': str(out_md),
        'script': str(Path(__file__)),
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    out_md.write_text(render_report(payload), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps({
        'json': str(out_json),
        'report': str(out_md),
        'best_markov_candidate': markov_best['candidate_name'],
        'best_markov_overall': markov_best['metrics']['overall'],
        'best_markov_mean_gap_closed_frac': markov_best['comparison']['mean_gap_closed_frac'],
        'best_markov_max_gap_closed_frac': markov_best['comparison']['max_gap_closed_frac'],
        'usable': usable,
        'overall_conclusion': overall_conclusion,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
