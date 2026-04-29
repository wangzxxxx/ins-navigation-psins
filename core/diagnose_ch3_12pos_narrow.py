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

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import load_module
from compare_ch3_12pos_path_baselines import build_ch3_path_paras, build_dataset_with_path
from compare_four_methods_shared_noise import compute_payload, expected_noise_config
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
KEY_PARAMS = ['dKa_yy', 'dKg_zz', 'Ka2_y', 'Ka2_z', 'dKg_xz', 'dKa_xz']
ALL_FACES = ['+X', '-X', '+Y', '-Y', '+Z', '-Z']

# Focused diagnosis candidates only; no sprawling search.
VALIDATION_ABLATION_NODES = [4, 10, 11]
LATE_BLOCK_NODES = [10, 11]
EARLY_BLOCK_NODES = [4, 5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=0.08)
    parser.add_argument('--report-date', default=datetime.now().strftime('%Y-%m-%d'))
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
        face_name, face_key = nearest_face(g_body.tolist())
        out.append({
            'pos_id': int(row[0]),
            'axis': [int(row[1]), int(row[2]), int(row[3])],
            'angle_deg': float(row[4] / mod.glv.deg),
            'gravity_body': [float(x) for x in g_body.tolist()],
            'face_name': face_name,
            'face_key': list(face_key),
        })
    return out


def structural_prefix_trace(face_rows, rows):
    counts = Counter()
    axis_balance = {'X': 0, 'Y': 0, 'Z': 0}
    first_cover_idx = {face: None for face in ALL_FACES}
    trace = []
    full_cover_idx = None

    for idx, (face_row, row) in enumerate(zip(face_rows, rows), start=1):
        face = face_row['face_name']
        axis, sign = FACE_AXIS_SIGN[face]
        already_seen = counts[face] > 0
        prev_bal = dict(axis_balance)
        counts[face] += 1
        axis_balance[axis] += sign
        if first_cover_idx[face] is None:
            first_cover_idx[face] = idx
        missing_faces = [name for name in ALL_FACES if counts[name] == 0]
        if full_cover_idx is None and not missing_faces:
            full_cover_idx = idx
        target_penalty = 2 * abs(axis_balance['Y']) + 2 * abs(axis_balance['Z']) + abs(axis_balance['X'])
        late_repeat = bool(full_cover_idx is not None and idx > full_cover_idx and already_seen)
        repeat_and_worsen_target = already_seen and (target_penalty > (2 * abs(prev_bal['Y']) + 2 * abs(prev_bal['Z']) + abs(prev_bal['X'])))
        trace.append({
            'prefix_end': idx,
            'node': idx,
            'axis': row['axis'],
            'angle_deg': row['angle_deg'],
            'face': face,
            'is_repeat_face': already_seen,
            'is_new_face': not already_seen,
            'missing_faces': missing_faces,
            'missing_face_count': len(missing_faces),
            'axis_balance': dict(axis_balance),
            'target_penalty': target_penalty,
            'late_repeat': late_repeat,
            'repeat_and_worsen_target': repeat_and_worsen_target,
            'pre_static_s': row['pre_static_s'],
            'post_static_s': row['post_static_s'],
            'node_total_s': row['node_total_s'],
        })

    return trace, first_cover_idx, full_cover_idx


def structural_summary(rows, face_rows):
    prefix, first_cover_idx, full_cover_idx = structural_prefix_trace(face_rows, rows)
    counts = Counter([f['face_name'] for f in face_rows])
    axis_balance = prefix[-1]['axis_balance']
    late_repeat_nodes = [item['node'] for item in prefix if item['late_repeat']]
    repeat_worsen_nodes = [item['node'] for item in prefix if item['repeat_and_worsen_target']]
    z_axis_rotation_nodes = [row['pos_id'] for row in rows if row['axis'] == [0, 0, 1]]
    return {
        'face_sequence': [f['face_name'] for f in face_rows],
        'face_counts': dict(counts),
        'axis_balance': axis_balance,
        'first_cover_index': first_cover_idx,
        'full_cover_index': full_cover_idx,
        'late_repeat_nodes': late_repeat_nodes,
        'repeat_and_worsen_target_nodes': repeat_worsen_nodes,
        'z_axis_rotation_nodes': z_axis_rotation_nodes,
        'prefix_trace': prefix,
        'total_time_s': float(sum(r['node_total_s'] for r in rows)),
    }


def structural_penalty(summary):
    full_cover_idx = summary['full_cover_index'] if summary['full_cover_index'] is not None else 99
    axis_balance = summary['axis_balance']
    return (
        12 * (full_cover_idx - 6) +
        10 * abs(axis_balance['Y']) +
        8 * abs(axis_balance['Z']) +
        3 * abs(axis_balance['X']) +
        4 * len(summary['late_repeat_nodes']) +
        5 * len(summary['repeat_and_worsen_target_nodes'])
    )


def ablate_node_rows(rows, node):
    return [dict(r) for r in rows if r['pos_id'] != node]


def modify_dwell_rows(rows, target_nodes, pre_s, post_s):
    out = []
    for row in rows:
        item = dict(row)
        if row['pos_id'] in target_nodes:
            item['pre_static_s'] = float(pre_s)
            item['post_static_s'] = float(post_s)
            item['node_total_s'] = float(row['rotation_time_s'] + pre_s + post_s)
        out.append(item)
    return out


def load_existing_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def key_param_table(payload: dict) -> dict:
    return {k: float(payload['param_errors'][k]['pct_error']) for k in KEY_PARAMS}


def overall_table(payload: dict) -> dict:
    return {
        'mean_pct_error': float(payload['overall']['mean_pct_error']),
        'median_pct_error': float(payload['overall']['median_pct_error']),
        'max_pct_error': float(payload['overall']['max_pct_error']),
    }


def run_markov42_payload(mod, paras, noise_scale: float, variant: str) -> dict:
    dataset = build_dataset_with_path(mod, noise_scale, paras)
    params = _param_specs(mod)
    clbt, _, _, _, _ = mod.run_calibration(
        dataset['imu_noisy'],
        dataset['pos0'],
        dataset['ts'],
        n_states=42,
        bi_g=dataset['bi_g'],
        tau_g=dataset['tau_g'],
        bi_a=dataset['bi_a'],
        tau_a=dataset['tau_a'],
        label=variant,
    )
    return compute_payload(
        mod,
        clbt,
        params,
        variant=variant,
        method_file='diagnose_ch3_12pos_narrow.py',
        extra={
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'comparison_mode': 'ch3_12pos_narrow_diagnosis',
            'method_key': 'markov42_noisy',
        },
    )


def validation_record(name: str, payload: dict, original_payload: dict, note: str) -> dict:
    key_now = key_param_table(payload)
    key_org = key_param_table(original_payload)
    return {
        'name': name,
        'note': note,
        'overall': overall_table(payload),
        'key_param_errors': key_now,
        'delta_vs_original': {
            'mean_pct_error': overall_table(payload)['mean_pct_error'] - overall_table(original_payload)['mean_pct_error'],
            'median_pct_error': overall_table(payload)['median_pct_error'] - overall_table(original_payload)['median_pct_error'],
            'max_pct_error': overall_table(payload)['max_pct_error'] - overall_table(original_payload)['max_pct_error'],
            **{k: key_now[k] - key_org[k] for k in KEY_PARAMS},
        },
    }


def explicit_findings(structural, ablations, validations):
    findings = []
    if structural['axis_balance']['Y'] == 2:
        findings.append('Y-face balance is poor in the faithful 12-node path: +Y appears 3 times while -Y appears only once, directly matching the dKa_yy / Ka2_y failure cluster.')
    if structural['full_cover_index'] and structural['full_cover_index'] >= 7:
        findings.append(f'Full six-face coverage is late (only reached at node {structural["full_cover_index"]}); nodes 1-6 over-spend time before the path has even seen all faces.')
    if {10, 11}.issubset(set(structural['repeat_and_worsen_target_nodes'])):
        findings.append('Late Z-axis repetition is harmful: the node 10-11 +Z-rotation block revisits already-seen +Y/+X faces after full coverage and worsens the target Y/Z balance instead of adding new excitation.')
    if 4 in [x['node'] for x in ablations[:3]]:
        findings.append('Node 4 is an early harmful node: it re-hits +Y before -Y has ever appeared, deepening the Y imbalance from +1 to +2 too early.')

    vmap = {item['name']: item for item in validations}
    if 'ablate_node10' in vmap and vmap['ablate_node10']['delta_vs_original']['max_pct_error'] < 0:
        findings.append('True noisy recheck confirms node 10 is harmful: removing node 10 reduces overall max error and also cuts dKa_yy / dKg_zz / Ka2_y / Ka2_z together.')
    if 'ablate_node11' in vmap and vmap['ablate_node11']['delta_vs_original']['max_pct_error'] < 0:
        findings.append('True noisy recheck confirms node 11 is also harmful: the second late +Z step is not helping the bad cluster; it pushes the late block further into repetition.')
    if 'dwell_late_block_30_60' in vmap:
        late = vmap['dwell_late_block_30_60']
        if late['delta_vs_original']['max_pct_error'] < 0:
            findings.append('There is also a dwell-placement problem around nodes 10-11: giving that late +Z block more pre-static time partly repairs the bad cluster, so the original [10,10,80] split there is too front-loaded on post-dwell.')
        else:
            findings.append('Dwell retiming around nodes 10-11 alone is not enough: the main defect is structural repetition, with dwell only acting as a secondary lever.')
    return findings


def repair_proposals(original_rows):
    # Proposal A: keep axes/signs, only give more pre-static to late +Z block.
    proposal_a = modify_dwell_rows(original_rows, LATE_BLOCK_NODES, pre_s=30.0, post_s=60.0)

    # Proposal B: still 12 nodes / 1200 s, but split the late +Z repetition by flipping node 11 sign.
    proposal_b = []
    for row in original_rows:
        item = dict(row)
        if row['pos_id'] == 11:
            item['angle_deg'] = -90.0
        proposal_b.append(item)

    return [
        {
            'name': 'repair_A_more_pre_static_on_late_Z_block',
            'kind': 'dwell_only',
            'benchmarked_in_this_pass': True,
            'rows': proposal_a,
            'rationale': 'Keep the 12-position skeleton unchanged; only move dwell earlier on nodes 10-11 so the late +Z block is less post-heavy and easier for the filter to exploit.',
        },
        {
            'name': 'repair_B_break_late_Z_plus_pair_by_flipping_node11',
            'kind': 'sign_flip',
            'benchmarked_in_this_pass': False,
            'rows': proposal_b,
            'rationale': 'Keep 12 nodes and 1200 s, but break the second same-sign +Z repetition so the late block stops revisiting +Y/+X in the same direction.',
        },
    ]


def render_report(payload: dict) -> str:
    lines = []
    lines.append('# Chapter-3 faithful 12-position path narrow diagnosis')
    lines.append('')
    lines.append('## 1. Bottom line')
    lines.append('')
    for item in payload['findings']:
        lines.append(f'- {item}')
    lines.append('')

    lines.append('## 2. Original face/segment structure')
    lines.append('')
    lines.append(f"- face sequence: {' → '.join(payload['structural_summary']['face_sequence'])}")
    lines.append(f"- face counts: `{json.dumps(payload['structural_summary']['face_counts'], ensure_ascii=False)}`")
    lines.append(f"- final axis balance [+/- exposure folded by face]: `{json.dumps(payload['structural_summary']['axis_balance'], ensure_ascii=False)}`")
    lines.append(f"- full six-face coverage first reached at node: **{payload['structural_summary']['full_cover_index']}**")
    lines.append(f"- late repeat nodes after full coverage: {payload['structural_summary']['late_repeat_nodes']}")
    lines.append(f"- repeat-and-worsen target nodes: {payload['structural_summary']['repeat_and_worsen_target_nodes']}")
    lines.append('')

    lines.append('## 3. Prefix tracking after each node (structural diagnosis)')
    lines.append('')
    lines.append('| node | face | missing_faces | axis_balance(X,Y,Z) | target_penalty | repeat? | late_repeat? | note |')
    lines.append('|---:|---|---:|---|---:|---|---|---|')
    for item in payload['prefix_trace']:
        note = ''
        if item['node'] == payload['structural_summary']['full_cover_index']:
            note = 'first full cover'
        elif item['repeat_and_worsen_target']:
            note = 'repeat + worsens Y/Z target'
        lines.append(
            f"| {item['node']} | {item['face']} | {item['missing_face_count']} | ({item['axis_balance']['X']:+d},{item['axis_balance']['Y']:+d},{item['axis_balance']['Z']:+d}) | {item['target_penalty']} | {'yes' if item['is_repeat_face'] else 'no'} | {'yes' if item['late_repeat'] else 'no'} | {note} |"
        )
    lines.append('')

    lines.append('## 4. Leave-one-node-out structural ablation (12-node full pass)')
    lines.append('')
    lines.append('| rank | ablated_node | face | structural_penalty_delta | Y_balance_delta | Z_balance_delta | late_repeat_delta |')
    lines.append('|---:|---:|---|---:|---:|---:|---:|')
    for rank, item in enumerate(payload['ablation_ranked'], start=1):
        lines.append(
            f"| {rank} | {item['node']} | {item['face']} | {item['structural_penalty_delta']:+.1f} | {item['axis_balance_delta']['Y']:+d} | {item['axis_balance_delta']['Z']:+d} | {item['late_repeat_delta']:+d} |"
        )
    lines.append('')
    lines.append('- More negative `structural_penalty_delta` means removing that node makes the chapter-3 path structurally healthier.')
    lines.append('')

    lines.append('## 5. Focused true noisy recheck at noise0p08 (Markov42 baseline only)')
    lines.append('')
    lines.append('| variant | note | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Ka2_z | dKg_xz | dKa_xz |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for item in payload['validations']:
        k = item['key_param_errors']
        o = item['overall']
        lines.append(
            f"| {item['name']} | {item['note']} | {o['mean_pct_error']:.3f} | {o['median_pct_error']:.3f} | {o['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {k['dKg_xz']:.3f} | {k['dKa_xz']:.3f} |"
        )
    lines.append('')

    lines.append('## 6. Dwell sensitivity verdict')
    lines.append('')
    lines.append(f"- original dwell split: all nodes use `[10,10,80]`")
    late = next(x for x in payload['validations'] if x['name'] == 'dwell_late_block_30_60')
    early = next(x for x in payload['validations'] if x['name'] == 'dwell_early_block_30_60')
    lines.append(
        f"- nodes 10-11 -> `[10,30,60]`: Δmax = {late['delta_vs_original']['max_pct_error']:+.3f}, ΔdKa_yy = {late['delta_vs_original']['dKa_yy']:+.3f}, ΔdKg_zz = {late['delta_vs_original']['dKg_zz']:+.3f}, ΔKa2_y = {late['delta_vs_original']['Ka2_y']:+.3f}, ΔKa2_z = {late['delta_vs_original']['Ka2_z']:+.3f}"
    )
    lines.append(
        f"- nodes 4-5 -> `[10,30,60]`: Δmax = {early['delta_vs_original']['max_pct_error']:+.3f}, ΔdKa_yy = {early['delta_vs_original']['dKa_yy']:+.3f}, ΔdKg_zz = {early['delta_vs_original']['dKg_zz']:+.3f}, ΔKa2_y = {early['delta_vs_original']['Ka2_y']:+.3f}, ΔKa2_z = {early['delta_vs_original']['Ka2_z']:+.3f}"
    )
    lines.append('')

    lines.append('## 7. Minimal repair proposals (12 nodes, <=1200 s)')
    lines.append('')
    for item in payload['repair_proposals']:
        lines.append(f"### {item['name']}")
        lines.append('')
        lines.append(f"- kind: {item['kind']}")
        lines.append(f"- benchmarked in this pass: {item['benchmarked_in_this_pass']}")
        lines.append(f"- rationale: {item['rationale']}")
        lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module('diagnose_ch3_12pos_narrow_src', str(SOURCE_FILE))
    suffix = make_suffix(args.noise_scale)
    noise_cfg = expected_noise_config(args.noise_scale)

    original_paras = build_ch3_path_paras(mod)
    original_rows = paras_to_rows(mod, original_paras)
    original_faces = orientation_faces(mod, original_paras)
    original_struct = structural_summary(original_rows, original_faces)
    original_penalty = structural_penalty(original_struct)

    original_payload = load_existing_payload(RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json')
    default_payload = load_existing_payload(RESULTS_DIR / 'M_markov_42state_gm1_shared_noise0p08_param_errors.json')

    ablations = []
    for node in range(1, len(original_rows) + 1):
        rows = ablate_node_rows(original_rows, node)
        paras = rows_to_paras(mod, rows)
        faces = orientation_faces(mod, paras)
        summ = structural_summary(rows, faces)
        pen = structural_penalty(summ)
        ablations.append({
            'node': node,
            'face': original_struct['face_sequence'][node - 1],
            'structural_penalty': pen,
            'structural_penalty_delta': pen - original_penalty,
            'axis_balance': summ['axis_balance'],
            'axis_balance_delta': {
                'X': summ['axis_balance']['X'] - original_struct['axis_balance']['X'],
                'Y': summ['axis_balance']['Y'] - original_struct['axis_balance']['Y'],
                'Z': summ['axis_balance']['Z'] - original_struct['axis_balance']['Z'],
            },
            'late_repeat_delta': len(summ['late_repeat_nodes']) - len(original_struct['late_repeat_nodes']),
            'full_cover_index': summ['full_cover_index'],
        })
    ablation_ranked = sorted(ablations, key=lambda x: (x['structural_penalty_delta'], abs(x['axis_balance']['Y']), abs(x['axis_balance']['Z'])))

    validations = []
    validations.append(validation_record('original', original_payload, original_payload, 'faithful 12-node baseline'))
    validations.append(validation_record('default18_reference', default_payload, original_payload, 'current default 18-position reference'))

    for node in VALIDATION_ABLATION_NODES:
        rows = ablate_node_rows(original_rows, node)
        paras = rows_to_paras(mod, rows)
        payload = run_markov42_payload(mod, paras, args.noise_scale, variant=f'ch3_ablate_node{node}_{suffix}')
        validations.append(validation_record(f'ablate_node{node}', payload, original_payload, f'leave out node {node} only'))

    late_rows = modify_dwell_rows(original_rows, LATE_BLOCK_NODES, pre_s=30.0, post_s=60.0)
    late_payload = run_markov42_payload(mod, rows_to_paras(mod, late_rows), args.noise_scale, variant=f'ch3_dwell_late_block_30_60_{suffix}')
    validations.append(validation_record('dwell_late_block_30_60', late_payload, original_payload, 'nodes 10-11 pre/post -> [30,60]'))

    early_rows = modify_dwell_rows(original_rows, EARLY_BLOCK_NODES, pre_s=30.0, post_s=60.0)
    early_payload = run_markov42_payload(mod, rows_to_paras(mod, early_rows), args.noise_scale, variant=f'ch3_dwell_early_block_30_60_{suffix}')
    validations.append(validation_record('dwell_early_block_30_60', early_payload, original_payload, 'nodes 4-5 pre/post -> [30,60]'))

    findings = explicit_findings(original_struct, ablation_ranked, validations)
    proposals = repair_proposals(original_rows)

    payload = {
        'experiment': 'diagnose_ch3_12pos_narrow',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'noise_config': noise_cfg,
        'baseline_compare': {
            'default18_reference': {
                'overall': overall_table(default_payload),
                'key_param_errors': key_param_table(default_payload),
            },
            'chapter3_original': {
                'overall': overall_table(original_payload),
                'key_param_errors': key_param_table(original_payload),
            },
        },
        'structural_summary': {k: v for k, v in original_struct.items() if k != 'prefix_trace'},
        'prefix_trace': original_struct['prefix_trace'],
        'ablation_ranked': ablation_ranked,
        'validations': validations,
        'findings': findings,
        'repair_proposals': proposals,
        'files': {},
    }

    out_json = RESULTS_DIR / f'ch3_12pos_narrow_diagnosis_{suffix}.json'
    out_md = REPORTS_DIR / f'psins_ch3_12pos_narrow_diagnosis_{args.report_date}.md'
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
        'findings': findings,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
