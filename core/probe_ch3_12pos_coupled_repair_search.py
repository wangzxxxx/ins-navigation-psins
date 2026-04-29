from __future__ import annotations

import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
METHOD_DIR = SCRIPT_DIR.parent / 'methods' / 'markov'
for p in [SCRIPT_DIR, METHOD_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module
from benchmark_ch3_12pos_goalA_repairs import (
    ROOT,
    RESULTS_DIR,
    REPORTS_DIR,
    SOURCE_FILE,
    paras_to_rows,
    rows_to_paras,
    orientation_faces,
    compact_result,
)
from compare_ch3_12pos_path_baselines import build_ch3_path_paras, build_dataset_with_path
from compare_four_methods_shared_noise import _load_json, _noise_matches, compute_payload, expected_noise_config
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs
from diagnose_ch3_12pos_narrow import structural_summary, structural_penalty

TARGET_NODES = [4, 8, 9, 10, 11, 12]
NOISE_SCALE = 0.08
REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
MAX_MARKOV_RUNS = 4
MAX_KF_RUNS = 2


def build_candidates(mod):
    original_rows = paras_to_rows(mod, build_ch3_path_paras(mod))
    original_faces = orientation_faces(mod, rows_to_paras(mod, original_rows))
    original_struct = structural_summary(original_rows, original_faces)
    original_penalty = structural_penalty(original_struct)

    candidates = []
    for r in range(2, 5):
        for subset in itertools.combinations(TARGET_NODES, r):
            rows = [dict(x) for x in original_rows]
            for node in subset:
                rows[node - 1]['angle_deg'] *= -1.0
            paras = rows_to_paras(mod, rows)
            faces = orientation_faces(mod, paras)
            struct = structural_summary(rows, faces)
            penalty = structural_penalty(struct)
            if penalty >= original_penalty:
                continue
            axis = struct['axis_balance']
            score = (
                penalty,
                abs(axis['Y']),
                abs(axis['Z']),
                len(struct['late_repeat_nodes']),
                len(struct['repeat_and_worsen_target_nodes']),
                -sum(1 for node in subset if node in [10, 11, 12]),
                len(subset),
                subset,
            )
            candidates.append({
                'name': 'coupled_flip_' + '_'.join(str(x) for x in subset),
                'subset': list(subset),
                'rows': rows,
                'faces': faces,
                'structural_summary': struct,
                'structural_penalty': penalty,
                'penalty_delta': penalty - original_penalty,
                'score': score,
            })
    candidates.sort(key=lambda x: x['score'])
    return original_rows, original_struct, original_penalty, candidates


def custom_output_path(candidate_name: str, method_key: str) -> Path:
    return RESULTS_DIR / f"{method_key}_{candidate_name}_shared_noise0p08_param_errors.json"


def load_or_run_custom_payload(mod, candidate_name: str, paras, method_key: str):
    out_path = custom_output_path(candidate_name, method_key)
    expected_cfg = expected_noise_config(NOISE_SCALE)
    if out_path.exists():
        payload = _load_json(out_path)
        if _noise_matches(payload, expected_cfg) and payload.get('extra', {}).get('candidate_name') == candidate_name:
            return payload, out_path

    dataset = build_dataset_with_path(mod, NOISE_SCALE, paras)
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
            label=f'{candidate_name}_{method_key}_noise0p08',
        )
    elif method_key == 'kf36_noisy':
        clbt, _, _, _, _ = mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=36,
            label=f'{candidate_name}_{method_key}_noise0p08',
        )
    else:
        raise KeyError(method_key)

    payload = compute_payload(
        mod,
        clbt,
        params,
        variant=f'{candidate_name}_{method_key}_noise0p08',
        method_file='probe_ch3_12pos_coupled_repair_search.py',
        extra={
            'noise_scale': NOISE_SCALE,
            'noise_config': dataset['noise_config'],
            'comparison_mode': 'ch3_coupled_repair_search',
            'candidate_name': candidate_name,
            'method_key': method_key,
        },
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, out_path


def render_report(payload: dict) -> str:
    lines = []
    lines.append('# Chapter-3 12-position coupled structural repair search')
    lines.append('')
    lines.append('## 1. Search setup')
    lines.append('')
    lines.append(f"- target nodes considered for coupled sign flips: {payload['target_nodes']}")
    lines.append('- search space: all 2-flip / 3-flip / 4-flip subsets over the target nodes')
    lines.append(f"- structural baseline penalty: **{payload['original_penalty']}**")
    lines.append(f"- accepted shortlist count after structural filter: **{len(payload['shortlist'])}**")
    lines.append('')
    lines.append('## 2. Top structural shortlist')
    lines.append('')
    lines.append('| rank | candidate | flipped_nodes | penalty | Δpenalty | axis_balance | full_cover | late_repeat | repeat+worsen | face_sequence |')
    lines.append('|---:|---|---|---:|---:|---|---:|---:|---:|---|')
    for idx, item in enumerate(payload['shortlist'], start=1):
        s = item['structural_summary']
        lines.append(
            f"| {idx} | {item['name']} | {item['subset']} | {item['structural_penalty']} | {item['penalty_delta']:+d} | `{json.dumps(s['axis_balance'], ensure_ascii=False)}` | {s['full_cover_index']} | {len(s['late_repeat_nodes'])} | {len(s['repeat_and_worsen_target_nodes'])} | {' → '.join(s['face_sequence'])} |"
        )
    lines.append('')
    lines.append('## 3. True noisy benchmark (Markov42 / KF36)')
    lines.append('')
    lines.append('| method | candidate | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Ka2_z | run_json |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---|')
    for row in payload['benchmarks']:
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        lines.append(
            f"| {row['method']} | {row['candidate']} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | `{row['run_json']}` |"
        )
    lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_module('probe_ch3_coupled_src', str(SOURCE_FILE))

    original_rows, original_struct, original_penalty, candidates = build_candidates(mod)
    shortlist = candidates[:8]

    benchmarks = []
    for item in shortlist[:MAX_MARKOV_RUNS]:
        paras = rows_to_paras(mod, item['rows'])
        payload_run, out_path = load_or_run_custom_payload(mod, item['name'], paras, 'markov42_noisy')
        benchmarks.append({
            'method': 'Markov42',
            'candidate': item['name'],
            'metrics': compact_result(payload_run),
            'run_json': str(out_path),
        })

    ranked_markov = sorted(
        [x for x in benchmarks if x['method'] == 'Markov42'],
        key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error'])
    )
    for item in ranked_markov[:MAX_KF_RUNS]:
        cand = next(x for x in shortlist if x['name'] == item['candidate'])
        paras = rows_to_paras(mod, cand['rows'])
        payload_run, out_path = load_or_run_custom_payload(mod, item['candidate'], paras, 'kf36_noisy')
        benchmarks.append({
            'method': 'KF36',
            'candidate': item['candidate'],
            'metrics': compact_result(payload_run),
            'run_json': str(out_path),
        })

    out_json = RESULTS_DIR / 'ch3_12pos_coupled_repair_search_noise0p08.json'
    out_md = REPORTS_DIR / f'psins_ch3_12pos_coupled_repair_search_{REPORT_DATE}.md'
    payload = {
        'experiment': 'ch3_12pos_coupled_repair_search',
        'report_date': REPORT_DATE,
        'noise_scale': NOISE_SCALE,
        'target_nodes': TARGET_NODES,
        'original_penalty': original_penalty,
        'original_structural_summary': original_struct,
        'shortlist': shortlist,
        'benchmarks': benchmarks,
        'files': {
            'json': str(out_json),
            'report': str(out_md),
        },
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    out_md.write_text(render_report(payload), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps(payload['files'], ensure_ascii=False))
    if benchmarks:
        best = sorted([x for x in benchmarks if x['method'] == 'Markov42'], key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))[0]
        print('BEST_MARKOV', best['candidate'], best['metrics']['overall'])


if __name__ == '__main__':
    main()
