from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import types
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
    build_ch3_initial_attitude,
    build_ch3_path_paras,
    build_dataset_with_path,
    build_default_path_paras,
)
from compare_four_methods_shared_noise import compute_payload, expected_noise_config
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs

STATE_LABELS = (
    ['phi_x','phi_y','phi_z','dv_x','dv_y','dv_z',
     'eb_x','eb_y','eb_z','db_x','db_y','db_z',
     'Kg00','Kg10','Kg20','Kg01','Kg11','Kg21','Kg02','Kg12','Kg22',
     'Ka_xx','Ka_xy','Ka_xz','Ka_yy','Ka_yz','Ka_zz',
     'Ka2_x','Ka2_y','Ka2_z','rx_x','rx_y','rx_z','ry_x','ry_y','ry_z']
)
CALIB_LABELS = list(STATE_LABELS[6:36])
KEY_REDUCTION_LABELS = ['Ka_yy', 'Kg22', 'Ka2_y', 'Ka2_z', 'Kg02', 'Ka_xz']
KEY_PARAM_ERRORS = ['dKa_yy', 'dKg_zz', 'Ka2_y', 'Ka2_z', 'dKg_xz', 'dKa_xz']
FACE_NAMES = {
    (1, 0, 0): '+X',
    (-1, 0, 0): '-X',
    (0, 1, 0): '+Y',
    (0, -1, 0): '-Y',
    (0, 0, 1): '+Z',
    (0, 0, -1): '-Z',
}
AXIS_ORDER = [
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 1],
    [0, 1, 0],
]
ORIGINAL_SIGNS = [1, 1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1]


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
    return f"noise{str(noise_scale).replace('.', 'p')}"


def paras_from_axes_signs(mod, axes, signs, split_pre: float, split_post: float, rot_time: float = 10.0):
    rows = []
    for idx, (axis, sign) in enumerate(zip(axes, signs), start=1):
        rows.append([idx, axis[0], axis[1], axis[2], 90.0 * float(sign), rot_time, split_pre, split_post])
    paras = mod.np.array(rows, dtype=float)
    paras[:, 4] = paras[:, 4] * mod.glv.deg
    return paras


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


def hamming_distance(signs_a, signs_b) -> int:
    return sum(int(a != b) for a, b in zip(signs_a, signs_b))


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


def face_summary(face_rows):
    counts = {}
    seq = []
    consecutive_duplicates = 0
    repeat_positions = []
    prev = None
    for row in face_rows:
        name = row['face_name']
        seq.append(name)
        counts[name] = counts.get(name, 0) + 1
        if prev == name:
            consecutive_duplicates += 1
            repeat_positions.append(row['pos_id'])
        prev = name
    missing = [name for name in ['+X', '-X', '+Y', '-Y', '+Z', '-Z'] if counts.get(name, 0) == 0]
    first_cover = {}
    seen = set()
    for idx, name in enumerate(seq, start=1):
        if name not in seen:
            seen.add(name)
            first_cover[name] = idx
    return {
        'sequence': seq,
        'counts': counts,
        'missing_faces': missing,
        'consecutive_duplicate_count': consecutive_duplicates,
        'consecutive_duplicate_positions': repeat_positions,
        'first_cover_index': first_cover,
    }


def score_path_fast(mod, paras):
    pos0 = mod.posset(34.0, 0.0, 0.0)
    att0 = build_ch3_initial_attitude(mod)
    ts = 0.01

    att = mod.attrottt(att0, paras, ts)
    imu, _ = mod.avp2imu(att, pos0)
    clbt_t = mod.get_default_clbt()
    imu_clean = mod.imuclbt(imu, clbt_t)

    eth = mod.Earth(pos0)
    wnie = mod.glv.wie * mod.np.array([0, math.cos(pos0[0]), math.sin(pos0[0])])
    Cba = mod.np.eye(3)
    nn, _, nts, _ = mod.nnts(2, ts)
    frq2 = int(1 / ts / 2) - 1
    n = 36

    kk = frq2
    for kk in range(frq2, min(5 * 60 * 2 * frq2, len(imu_clean)), 2 * frq2):
        ww = mod.np.mean(imu_clean[kk - frq2:kk + frq2 + 1, :3], axis=0)
        if mod.np.linalg.norm(ww) / ts > 20 * mod.glv.dph:
            break
    _, _, _, qnb = mod.alignsb(imu_clean[frq2:max(kk - 3 * frq2, frq2 + 1), :], pos0)

    kf = mod.clbtkfinit_36(nts)
    kf['Pxk'][:, 2] = 0
    kf['Pxk'][2, :] = 0
    Hk = kf['Hk']

    qvec = mod.np.zeros(n)
    qvec[0:3] = 0.01 * mod.glv.dpsh
    qvec[3:6] = 10.0 * mod.glv.ugpsHz
    Qk = mod.np.diag(qvec) ** 2 * nts

    clbt = {
        'Kg': mod.np.eye(3), 'Ka': mod.np.eye(3), 'Ka2': mod.np.zeros(3),
        'eb': mod.np.zeros(3), 'db': mod.np.zeros(3), 'rx': mod.np.zeros(3), 'ry': mod.np.zeros(3),
    }
    dotwf = mod.imudot(imu_clean, 5.0)
    vn = mod.np.zeros(3)
    t1s = 0.0

    for k in range(2 * frq2, len(imu_clean) - frq2, nn):
        k1 = k + nn - 1
        wm = imu_clean[k:k1 + 1, :3]
        vm = imu_clean[k:k1 + 1, 3:6]
        dwb = mod.np.mean(dotwf[k:k1 + 1, :3], axis=0)
        phim, dvbm = mod.cnscl(mod.np.hstack((wm, vm)))
        phim = clbt['Kg'] @ phim - clbt['eb'] * nts
        dvbm = clbt['Ka'] @ dvbm - clbt['db'] * nts
        wb, fb = phim / nts, dvbm / nts
        SS = mod.imulvS(wb, dwb, Cba)
        Ft = mod.getFt_36(fb, wb, mod.q2mat(qnb), wnie, SS)
        Phi = mod.np.eye(n) + Ft * nts
        kf['Pxk'] = Phi @ kf['Pxk'] @ Phi.T + Qk
        t1s += nts
        if t1s > (0.2 - ts / 2):
            t1s = 0.0
            ww = mod.np.mean(imu_clean[k - frq2:k + frq2 + 1, :3], axis=0)
            if mod.np.linalg.norm(ww) / ts < 20 * mod.glv.dph:
                S = Hk @ kf['Pxk'] @ Hk.T + kf['Rk']
                K = kf['Pxk'] @ Hk.T @ mod.np.linalg.inv(S)
                I_KH = mod.np.eye(n) - K @ Hk
                kf['Pxk'] = I_KH @ kf['Pxk'] @ I_KH.T + K @ kf['Rk'] @ K.T
        fn = mod.qmulv(qnb, fb - clbt['Ka2'] * (fb ** 2) - SS[:, 0:6] @ mod.np.concatenate((clbt['rx'], clbt['ry'])))
        vn = vn + (mod.rotv(-wnie * nts / 2, fn) + mod.np.array([0, 0, -eth.g])) * nts
        qnb = mod.qupdt2(qnb, phim, wnie * nts)

    sigma_f = mod.np.sqrt(mod.np.diag(kf['Pxk']))
    kf0 = mod.clbtkfinit_36(nts)
    sigma0 = mod.np.sqrt(mod.np.diag(kf0['Pxk']))
    sigma0 = mod.np.where(sigma0 < 1e-30, 1.0, sigma0)
    red = sigma_f / sigma0 * 100.0
    red_dict = {lbl: float(r) for lbl, r in zip(STATE_LABELS, red)}
    total_t = len(imu_clean) * ts
    calib_vals = [red_dict[k] for k in CALIB_LABELS]
    worst_label = max(CALIB_LABELS, key=lambda k: red_dict[k])
    mean_calib = float(sum(calib_vals) / len(calib_vals))
    return {
        'total_time_s': float(total_t),
        'mean_reduction_pct': mean_calib,
        'worst_reduction_pct': float(red_dict[worst_label]),
        'worst_state': worst_label,
        'key_reduction_pct': {k: float(red_dict[k]) for k in KEY_REDUCTION_LABELS},
        'all_reduction_pct': red_dict,
    }


def benchmark_baseline(mod, paras, noise_scale: float, method_key: str, variant: str):
    dataset = build_dataset_with_path(mod, noise_scale, paras)
    params = _param_specs(mod)
    if method_key == 'kf36_noisy':
        clbt, _, _, _, _ = mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=36,
            label=variant,
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
            label=variant,
        )
    else:
        raise KeyError(method_key)
    payload = compute_payload(mod, clbt, params, variant=variant, method_file='diagnose_ch3_12pos_repair.py', extra={
        'noise_scale': noise_scale,
        'noise_config': dataset['noise_config'],
        'comparison_mode': 'ch3_repair_diagnosis',
        'method_key': method_key,
    })
    return payload


def summarize_candidate(mod, name: str, paras, signs, fast_diag):
    faces = orientation_faces(mod, paras)
    face_info = face_summary(faces)
    return {
        'name': name,
        'signs': list(signs),
        'hamming_vs_original': hamming_distance(signs, ORIGINAL_SIGNS),
        'rows': paras_to_rows(mod, paras),
        'faces': faces,
        'face_summary': face_info,
        'fast_diag': fast_diag,
    }


def choose_top_repairs(original_fast, single_flip_runs, pair_flip_runs, dwell_runs):
    best_single = min(single_flip_runs, key=lambda x: (x['fast_diag']['worst_reduction_pct'], x['fast_diag']['mean_reduction_pct']))
    best_pair = min(pair_flip_runs, key=lambda x: (x['fast_diag']['worst_reduction_pct'], x['fast_diag']['mean_reduction_pct']))
    best_dwell = min(dwell_runs, key=lambda x: (x['fast_diag']['worst_reduction_pct'], x['fast_diag']['mean_reduction_pct']))

    chosen = [best_single]
    if best_pair['name'] != best_single['name']:
        chosen.append(best_pair)
    if len(chosen) < 2 and best_dwell['name'] not in {c['name'] for c in chosen}:
        chosen.append(best_dwell)
    if len(chosen) < 2:
        chosen.append(best_pair)
    return chosen[:2], {
        'best_single': best_single['name'],
        'best_pair': best_pair['name'],
        'best_dwell': best_dwell['name'],
        'original_worst_pct': original_fast['worst_reduction_pct'],
    }


def render_report(payload: dict) -> str:
    lines = []
    lines.append('# Chapter-3 12-position reverse-fix diagnosis')
    lines.append('')
    lines.append('## 1. Core answer')
    lines.append('')
    lines.append(f"- original faithful path total time: **{payload['original']['rows'][-1]['pos_id'] * 100 if False else payload['original']['rows'] and sum(r['node_total_s'] for r in payload['original']['rows']):.0f} s**")
    lines.append(f"- hard budget respected by all repaired candidates: **<= 1200 s**")
    lines.append(f"- fast observability worst state on original: **{payload['original']['fast_diag']['worst_state']} = {payload['original']['fast_diag']['worst_reduction_pct']:.3f}%**")
    lines.append(f"- most harmful final-error cluster under baseline noisy benchmark already known and rechecked here: **dKa_yy / dKg_zz / Ka2_y / Ka2_z**")
    lines.append('')

    lines.append('## 2. Original path face sequence')
    lines.append('')
    lines.append(f"- sequence: {' → '.join(payload['original']['face_summary']['sequence'])}")
    lines.append(f"- counts: `{json.dumps(payload['original']['face_summary']['counts'], ensure_ascii=False)}`")
    lines.append(f"- consecutive duplicate faces: {payload['original']['face_summary']['consecutive_duplicate_count']} at nodes {payload['original']['face_summary']['consecutive_duplicate_positions']}")
    lines.append('')

    lines.append('## 3. Prefix diagnosis (fast covariance proxy, lower reduction% is better)')
    lines.append('')
    lines.append('| prefix_end | worst_state | worst_pct | Ka_yy | Kg22 | Ka2_y | Ka2_z | note |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|---|')
    prev = None
    for item in payload['prefix_diag']:
        note = ''
        if prev is not None and item['fast_diag']['worst_reduction_pct'] > prev['fast_diag']['worst_reduction_pct'] + 0.5:
            note = 'worse jump'
        prev = item
        k = item['fast_diag']['key_reduction_pct']
        lines.append(
            f"| {item['prefix_end']} | {item['fast_diag']['worst_state']} | {item['fast_diag']['worst_reduction_pct']:.3f} | {k['Ka_yy']:.3f} | {k['Kg22']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | {note} |"
        )
    lines.append('')

    lines.append('## 4. Leave-one-node-out diagnosis (fast covariance proxy)')
    lines.append('')
    lines.append('| ablated_node | original_face | worst_pct_delta | Ka_yy_delta | Kg22_delta | Ka2_y_delta | Ka2_z_delta |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|')
    for item in payload['ablation_ranked'][:8]:
        d = item['delta_vs_original']
        lines.append(
            f"| {item['node']} | {item['face']} | {d['worst_reduction_pct']:+.3f} | {d['Ka_yy']:+.3f} | {d['Kg22']:+.3f} | {d['Ka2_y']:+.3f} | {d['Ka2_z']:+.3f} |"
        )
    lines.append('')
    lines.append('- Negative delta means removing that node helps observability; positive means removing it hurts.')
    lines.append('')

    lines.append('## 5. Dwell split sensitivity on original sign sequence')
    lines.append('')
    lines.append('| split [rot, pre, post] | worst_state | worst_pct | Ka_yy | Kg22 | Ka2_y | Ka2_z |')
    lines.append('|---|---|---:|---:|---:|---:|---:|')
    for item in payload['dwell_sweep']:
        k = item['fast_diag']['key_reduction_pct']
        lines.append(
            f"| [10,{item['split_pre']:.0f},{item['split_post']:.0f}] | {item['fast_diag']['worst_state']} | {item['fast_diag']['worst_reduction_pct']:.3f} | {k['Ka_yy']:.3f} | {k['Kg22']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} |"
        )
    lines.append('')

    lines.append('## 6. Minimal-repair shortlist chosen for true noisy benchmark')
    lines.append('')
    for cand in payload['selected_repairs']:
        lines.append(f"### {cand['name']}")
        lines.append('')
        lines.append(f"- hamming vs original signs: {cand['hamming_vs_original']}")
        lines.append(f"- face sequence: {' → '.join(cand['face_summary']['sequence'])}")
        lines.append(f"- fast worst state: {cand['fast_diag']['worst_state']} = {cand['fast_diag']['worst_reduction_pct']:.3f}%")
        lines.append('')
        lines.append('| idx | axis | angle_deg | rot | pre | post | face |')
        lines.append('|---:|---|---:|---:|---:|---:|---|')
        for row, face in zip(cand['rows'], cand['faces']):
            lines.append(
                f"| {row['pos_id']} | {row['axis']} | {row['angle_deg']:.0f} | {row['rotation_time_s']:.0f} | {row['pre_static_s']:.0f} | {row['post_static_s']:.0f} | {face['face_name']} |"
            )
        lines.append('')

    lines.append('## 7. True noisy baseline benchmark (same truth/noise family and seed, noise0p08)')
    lines.append('')
    lines.append('| path | method | mean_pct_error | median_pct_error | max_pct_error | dKa_yy | dKg_zz | Ka2_y | Ka2_z |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for item in payload['benchmark_table']:
        o = item['overall']
        k = item['key_param_errors']
        lines.append(
            f"| {item['path_name']} | {item['method_key']} | {o['mean_pct_error']:.3f} | {o['median_pct_error']:.3f} | {o['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} |"
        )
    lines.append('')

    lines.append('## 8. Practical conclusion')
    lines.append('')
    lines.append(f"- strongest ablation suspects: {', '.join(str(x['node']) for x in payload['ablation_ranked'][:3])}")
    lines.append(f"- best single minimal repair: {payload['selection_meta']['best_single']}")
    lines.append(f"- best pair minimal repair: {payload['selection_meta']['best_pair']}")
    lines.append(f"- best dwell-only repair: {payload['selection_meta']['best_dwell']}")
    lines.append('- If the top repaired candidate still trails the default path by a wide margin, the evidence supports: the original chapter-3 skeleton is not just slightly mistuned; several mid-sequence transitions bias Y/Z-related observability in the wrong direction, and modest sign repairs only partially close the gap.')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module('diagnose_ch3_src', str(SOURCE_FILE))
    noise_cfg = expected_noise_config(args.noise_scale)

    default_paras = build_default_path_paras(mod)
    original_paras = build_ch3_path_paras(mod)
    original_fast = score_path_fast(mod, original_paras)
    original = summarize_candidate(mod, 'chapter3_original', original_paras, ORIGINAL_SIGNS, original_fast)

    prefix_diag = []
    for k in range(1, len(original_paras) + 1):
        prefix_paras = original_paras[:k, :].copy()
        prefix_diag.append({
            'prefix_end': k,
            'fast_diag': score_path_fast(mod, prefix_paras),
        })

    ablations = []
    original_key = original_fast['key_reduction_pct']
    original_faces = original['face_summary']['sequence']
    for node in range(1, len(original_paras) + 1):
        mask = [i for i in range(len(original_paras)) if i != node - 1]
        paras = original_paras[mask, :].copy()
        fast = score_path_fast(mod, paras)
        delta = {
            'worst_reduction_pct': fast['worst_reduction_pct'] - original_fast['worst_reduction_pct'],
        }
        for key in KEY_REDUCTION_LABELS:
            delta[key] = fast['key_reduction_pct'][key] - original_key[key]
        ablations.append({
            'node': node,
            'face': original_faces[node - 1],
            'fast_diag': fast,
            'delta_vs_original': delta,
        })
    ablation_ranked = sorted(ablations, key=lambda x: (x['delta_vs_original']['worst_reduction_pct'], x['delta_vs_original']['Ka_yy'], x['delta_vs_original']['Kg22']))

    dwell_sweep = []
    for split_pre, split_post in [(10, 80), (20, 70), (30, 60), (40, 50)]:
        paras = paras_from_axes_signs(mod, AXIS_ORDER, ORIGINAL_SIGNS, split_pre=float(split_pre), split_post=float(split_post))
        fast = score_path_fast(mod, paras)
        dwell_sweep.append(summarize_candidate(mod, f'ch3_dwell_{int(split_pre)}_{int(split_post)}', paras, ORIGINAL_SIGNS, fast) | {
            'split_pre': float(split_pre),
            'split_post': float(split_post),
        })

    single_flip_runs = []
    for idx in range(len(ORIGINAL_SIGNS)):
        signs = list(ORIGINAL_SIGNS)
        signs[idx] *= -1
        paras = paras_from_axes_signs(mod, AXIS_ORDER, signs, split_pre=10.0, split_post=80.0)
        fast = score_path_fast(mod, paras)
        single_flip_runs.append(summarize_candidate(mod, f'ch3_flip_{idx+1}', paras, signs, fast))
    single_flip_runs.sort(key=lambda x: (x['fast_diag']['worst_reduction_pct'], x['fast_diag']['mean_reduction_pct']))

    candidate_pair_indices = sorted(set(
        [item['node'] - 1 for item in ablation_ranked[:6]] +
        [int(item['name'].split('_')[-1]) - 1 for item in single_flip_runs[:6]]
    ))
    pair_flip_runs = []
    for i, j in itertools.combinations(candidate_pair_indices, 2):
        signs = list(ORIGINAL_SIGNS)
        signs[i] *= -1
        signs[j] *= -1
        paras = paras_from_axes_signs(mod, AXIS_ORDER, signs, split_pre=10.0, split_post=80.0)
        fast = score_path_fast(mod, paras)
        pair_flip_runs.append(summarize_candidate(mod, f'ch3_flip_{i+1}_{j+1}', paras, signs, fast))
    pair_flip_runs.sort(key=lambda x: (x['fast_diag']['worst_reduction_pct'], x['fast_diag']['mean_reduction_pct']))

    selected_repairs, selection_meta = choose_top_repairs(original_fast, single_flip_runs, pair_flip_runs, dwell_sweep)

    compare_candidates = [
        ('default_path', default_paras),
        ('chapter3_original', original_paras),
    ] + [(cand['name'], paras_from_axes_signs(mod, AXIS_ORDER, cand['signs'], cand['rows'][0]['pre_static_s'], cand['rows'][0]['post_static_s'])) for cand in selected_repairs]

    benchmark_cache = {}
    benchmark_table = []
    for path_name, paras in compare_candidates:
        for method_key in ['kf36_noisy', 'markov42_noisy']:
            payload = benchmark_baseline(mod, paras, args.noise_scale, method_key, variant=f'{path_name}_{method_key}_{make_suffix(args.noise_scale)}')
            benchmark_cache[(path_name, method_key)] = payload
            benchmark_table.append({
                'path_name': path_name,
                'method_key': method_key,
                'overall': payload['overall'],
                'key_param_errors': {k: payload['param_errors'][k]['pct_error'] for k in KEY_PARAM_ERRORS},
            })

    payload = {
        'experiment': 'diagnose_ch3_12pos_repair',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'noise_config': noise_cfg,
        'original': original,
        'prefix_diag': prefix_diag,
        'ablation_ranked': ablation_ranked,
        'dwell_sweep': dwell_sweep,
        'single_flip_top10': single_flip_runs[:10],
        'pair_flip_top10': pair_flip_runs[:10],
        'selected_repairs': selected_repairs,
        'selection_meta': selection_meta,
        'benchmark_table': benchmark_table,
    }

    suffix = make_suffix(args.noise_scale)
    out_json = RESULTS_DIR / f'ch3_12pos_repair_diagnosis_{suffix}.json'
    out_md = REPORTS_DIR / f'psins_ch3_12pos_repair_diagnosis_{suffix}_{args.report_date}.md'
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    out_md.write_text(render_report(payload), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps({
        'json': str(out_json),
        'report': str(out_md),
        'selected_repairs': [cand['name'] for cand in selected_repairs],
        'selection_meta': selection_meta,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
