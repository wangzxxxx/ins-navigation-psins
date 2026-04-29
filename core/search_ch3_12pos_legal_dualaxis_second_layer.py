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
from search_ch3_12pos_legal_dualaxis_repairs import (
    Candidate,
    OLD_RESULT_MAP,
    VALID_PRIOR_SIGN_ONLY_PATHS,
    build_candidate,
    compare_vs_base,
    make_suffix,
    render_action,
)

NOISE_SCALE = 0.08
REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
INCUMBENT_FLIPS = (8, 11, 12)
INCUMBENT_NAME = 'legal_flip_8_11_12'
PREVIOUS_BENCHMARKED = {
    (2, 8),
    (1, 2, 8),
    (2, 7, 8),
    (2, 10, 11),
    (4, 5, 8),
    (1, 2, 7, 8),
    (10, 11),
    (10, 12),
    (8, 11, 12),
}
DIRECTION_A_SEEDS = [
    (2, 8),
    (1, 2, 8),
    (2, 7, 8),
    (2, 10, 11),
    (4, 5, 8),
    (1, 2, 7, 8),
    (10, 11),
    (10, 12),
    (8, 11, 12),
]
DIRECTION_A_TOP_K = 6
RETIME_SPECS = [
    {
        'name': 'legal_flip_8_11_12_retime_tail10_12_pre20_post70',
        'target_nodes': [10, 11, 12],
        'pre_static_s': 20.0,
        'post_static_s': 70.0,
        'note': 'same legal motor sequence; move 10 s of dwell from post to pre on tail nodes 10-12',
    },
    {
        'name': 'legal_flip_8_11_12_retime_tail10_12_pre30_post60',
        'target_nodes': [10, 11, 12],
        'pre_static_s': 30.0,
        'post_static_s': 60.0,
        'note': 'same legal motor sequence; stronger pre-static shift on tail nodes 10-12',
    },
    {
        'name': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
        'target_nodes': [8, 11, 12],
        'pre_static_s': 20.0,
        'post_static_s': 70.0,
        'note': 'same legal motor sequence; earlier dwell only on the three flipped nodes 8/11/12',
    },
    {
        'name': 'legal_flip_8_11_12_retime_flipnodes_pre30_post60',
        'target_nodes': [8, 11, 12],
        'pre_static_s': 30.0,
        'post_static_s': 60.0,
        'note': 'same legal motor sequence; stronger earlier dwell on the three flipped nodes 8/11/12',
    },
]


@dataclass
class SecondLayerCandidate:
    name: str
    direction: str
    family: str
    note: str
    base_reference: str
    flips: tuple[int, ...]
    action_sequence: list[dict]
    rows: list[dict]
    faces: list[dict]
    structural: dict
    method_tag: str
    equivalent_prior_result: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def sanitize_tag(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = text.strip('_')
    return text


def candidate_result_path(candidate: SecondLayerCandidate, method_key: str, noise_scale: float) -> Path:
    suffix = make_suffix(noise_scale)
    if method_key == 'markov42_noisy':
        prefix = 'M_markov_42state_gm1'
    elif method_key == 'kf36_noisy':
        prefix = 'KF36'
    else:
        raise KeyError(method_key)
    return RESULTS_DIR / f'{prefix}_{candidate.method_tag}_shared_{suffix}_param_errors.json'


def run_candidate_payload(mod, candidate: SecondLayerCandidate, method_key: str, noise_scale: float, force_rerun: bool = False):
    expected_cfg = expected_noise_config(noise_scale)
    out_path = candidate_result_path(candidate, method_key, noise_scale)
    if out_path.exists() and (not force_rerun):
        payload = _load_json(out_path)
        if _noise_matches(payload, expected_cfg) and payload.get('extra', {}).get('candidate_name') == candidate.name:
            return payload, 'reused_verified', out_path

    paras = mod.np.array([
        [
            row['pos_id'],
            row['axis'][0],
            row['axis'][1],
            row['axis'][2],
            row['angle_deg'],
            row['rotation_time_s'],
            row['pre_static_s'],
            row['post_static_s'],
        ]
        for row in candidate.rows
    ], dtype=float)
    paras[:, 4] = paras[:, 4] * mod.glv.deg
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
        method_file='search_ch3_12pos_legal_dualaxis_second_layer.py',
        extra={
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'comparison_mode': 'ch3_legal_dualaxis_second_layer',
            'candidate_name': candidate.name,
            'candidate_flips': list(candidate.flips),
            'method_key': method_key,
            'direction': candidate.direction,
            'family': candidate.family,
            'base_reference': candidate.base_reference,
            'legality': 'true_dual_axis_motor_sequence',
        },
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', out_path


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


def load_baselines(noise_scale: float):
    faithful_markov = _load_json(RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json')
    faithful_kf = _load_json(RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json')
    default_markov = _load_json(RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json')
    default_kf = _load_json(RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json')
    for payload in [faithful_markov, faithful_kf, default_markov, default_kf]:
        if not _noise_matches(payload, expected_noise_config(noise_scale)):
            raise ValueError('Baseline noise configuration mismatch')
    return {
        'faithful_markov': faithful_markov,
        'faithful_kf': faithful_kf,
        'default_markov': default_markov,
        'default_kf': default_kf,
    }


def make_second_layer_structural_candidates(mod) -> list[SecondLayerCandidate]:
    chosen = {}
    for base_flips in DIRECTION_A_SEEDS:
        base_set = set(base_flips)
        for node in range(1, 13):
            flips = tuple(sorted(base_set ^ {node}))
            if flips in PREVIOUS_BENCHMARKED:
                continue
            cand = build_candidate(mod, flips)
            score = (
                cand.penalty,
                abs(cand.structural['axis_balance']['Y']),
                abs(cand.structural['axis_balance']['Z']),
                len(cand.structural['late_repeat_nodes']),
                len(cand.structural['repeat_and_worsen_target_nodes']),
                len(cand.flips),
                cand.flips,
            )
            if flips not in chosen or score < chosen[flips]['score']:
                chosen[flips] = {
                    'score': score,
                    'candidate': cand,
                    'base_flips': base_flips,
                    'node': node,
                }

    ranked = sorted(chosen.values(), key=lambda x: x['score'])[:DIRECTION_A_TOP_K]
    out = []
    for item in ranked:
        cand: Candidate = item['candidate']
        out.append(
            SecondLayerCandidate(
                name=cand.name,
                direction='A',
                family='one_flip_structural_neighbor',
                note=f'one legal sign toggle at node {item["node"]} from seed {list(item["base_flips"])}',
                base_reference='legal shortlist neighbor',
                flips=cand.flips,
                action_sequence=cand.action_sequence,
                rows=cand.rows,
                faces=cand.faces,
                structural=cand.structural,
                method_tag=f'ch3legaldual2_{sanitize_tag(cand.name)}',
                equivalent_prior_result=cand.equivalent_prior_result,
            )
        )
    return out


def make_retime_candidates(mod) -> list[SecondLayerCandidate]:
    base = build_candidate(mod, INCUMBENT_FLIPS)
    out = []
    for spec in RETIME_SPECS:
        rows = []
        for row in base.rows:
            item = dict(row)
            if row['pos_id'] in spec['target_nodes']:
                item['pre_static_s'] = float(spec['pre_static_s'])
                item['post_static_s'] = float(spec['post_static_s'])
                item['node_total_s'] = float(item['rotation_time_s'] + item['pre_static_s'] + item['post_static_s'])
            rows.append(item)
        total_time_s = sum(x['node_total_s'] for x in rows)
        if total_time_s > 1200.0 + 1e-9:
            raise ValueError(f'Retime family exceeds total time: {spec["name"]}')
        out.append(
            SecondLayerCandidate(
                name=spec['name'],
                direction='B',
                family='incumbent_retime',
                note=spec['note'],
                base_reference=INCUMBENT_NAME,
                flips=base.flips,
                action_sequence=base.action_sequence,
                rows=rows,
                faces=base.faces,
                structural={**base.structural, 'total_time_s': total_time_s},
                method_tag=f'ch3legaldual2_{sanitize_tag(spec["name"])}',
                equivalent_prior_result=base.equivalent_prior_result,
            )
        )
    return out


def markov_row(candidate: SecondLayerCandidate, payload: dict, path: Path, status: str) -> dict:
    return {
        'candidate_name': candidate.name,
        'direction': candidate.direction,
        'family': candidate.family,
        'base_reference': candidate.base_reference,
        'note': candidate.note,
        'flips': list(candidate.flips),
        'metrics': compact_result(payload),
        'run_json': str(path),
        'status': status,
        'equivalent_prior_result': candidate.equivalent_prior_result,
        'target_nodes': [row['pos_id'] for row in candidate.rows if abs(row['pre_static_s'] - 10.0) > 1e-9 or abs(row['post_static_s'] - 80.0) > 1e-9],
    }


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


def render_report(payload: dict) -> str:
    lines = []
    lines.append('# Chapter-3 legal dual-axis repair: narrowed second-layer relaunch')
    lines.append('')
    lines.append('## 1. Scope actually relaunched')
    lines.append('')
    lines.append('- This pass is **not** a broad new legal search.')
    lines.append(f"- Direction A: benchmark the top **{len(payload['direction_a_candidates'])}** unbenchmarked one-flip legal neighbors around the prior shortlist.")
    lines.append(f"- Direction B: benchmark **{len(payload['direction_b_candidates'])}** lightweight retime families around `{payload['incumbent']['candidate_name']}` while preserving the exact same legal motor sequence and total time ≤ 1200 s.")
    lines.append('')
    lines.append('## 2. Fixed references')
    lines.append('')
    lines.append(f"- faithful12 Markov42: mean **{payload['references']['faithful12']['markov42']['overall']['mean_pct_error']:.3f}**, median **{payload['references']['faithful12']['markov42']['overall']['median_pct_error']:.3f}**, max **{payload['references']['faithful12']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- default18 Markov42: mean **{payload['references']['default18']['markov42']['overall']['mean_pct_error']:.3f}**, median **{payload['references']['default18']['markov42']['overall']['median_pct_error']:.3f}**, max **{payload['references']['default18']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- incumbent legal candidate: `{payload['incumbent']['candidate_name']}` = mean **{payload['incumbent']['markov42']['overall']['mean_pct_error']:.3f}**, median **{payload['incumbent']['markov42']['overall']['median_pct_error']:.3f}**, max **{payload['incumbent']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('## 3. Direction A + B benchmark rows (Markov42, shared noise0p08/seed42)')
    lines.append('')
    lines.append('| rank | candidate | dir | family | mean | median | max | Δmean vs incumbent | Δmax vs incumbent | note |')
    lines.append('|---:|---|---|---|---:|---:|---:|---:|---:|---|')
    for idx, row in enumerate(payload['markov42_rows'], start=1):
        d = row['delta_vs_incumbent']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['direction']} | {row['family']} | {row['metrics']['overall']['mean_pct_error']:.3f} | {row['metrics']['overall']['median_pct_error']:.3f} | {row['metrics']['overall']['max_pct_error']:.3f} | {d['mean_pct_error']['improvement_pct_points']:+.3f} | {d['max_pct_error']['improvement_pct_points']:+.3f} | {row['note']} |"
        )
    lines.append('')
    lines.append('## 4. Best narrowed second-layer result')
    lines.append('')
    best = payload['best_of_relaunch']
    lines.append(f"- best relaunched candidate: **{best['candidate_name']}**")
    lines.append(f"- direction/family: `{best['direction']}` / `{best['family']}`")
    lines.append(f"- Markov42: mean **{best['markov42']['overall']['mean_pct_error']:.3f}**, median **{best['markov42']['overall']['median_pct_error']:.3f}**, max **{best['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- vs incumbent `{payload['incumbent']['candidate_name']}`: Δmean = **{best['delta_vs_incumbent']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian = **{best['delta_vs_incumbent']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax = **{best['delta_vs_incumbent']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- material beat over incumbent? **{'yes' if payload['material_improvement_over_incumbent'] else 'no'}**")
    lines.append('')
    lines.append('## 5. Verdict')
    lines.append('')
    lines.append(f"- {payload['verdict']}" )
    lines.append(f"- Gap to faithful12 closed by incumbent already remains: mean +{payload['incumbent_vs_faithful']['mean_pct_error']['improvement_pct_points']:.3f}, max +{payload['incumbent_vs_faithful']['max_pct_error']['improvement_pct_points']:.3f} pct-points.")
    lines.append(f"- Gap to default18 is still huge even after the narrowed relaunch: best relaunch mean-default18 = {best['markov42']['overall']['mean_pct_error'] - payload['references']['default18']['markov42']['overall']['mean_pct_error']:+.3f}, max-default18 = {best['markov42']['overall']['max_pct_error'] - payload['references']['default18']['markov42']['overall']['max_pct_error']:+.3f} pct-points.")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_module('search_ch3_legal_dualaxis_second_layer_src', str(SOURCE_FILE))

    baselines = load_baselines(args.noise_scale)
    incumbent_cand = build_candidate(mod, INCUMBENT_FLIPS)
    incumbent_markov_payload, incumbent_markov_path = previous_old_valid_payload(INCUMBENT_FLIPS, 'markov42_noisy', args.noise_scale)
    incumbent_kf_payload, incumbent_kf_path = previous_old_valid_payload(INCUMBENT_FLIPS, 'kf36_noisy', args.noise_scale)

    direction_a = make_second_layer_structural_candidates(mod)
    direction_b = make_retime_candidates(mod)
    all_candidates = direction_a + direction_b

    rows = []
    payload_by_name = {}
    candidate_by_name = {cand.name: cand for cand in all_candidates}
    for cand in all_candidates:
        payload, status, path = run_candidate_payload(mod, cand, 'markov42_noisy', args.noise_scale, args.force_rerun)
        item = markov_row(cand, payload, path, status)
        item['delta_vs_incumbent'] = delta_vs_incumbent(incumbent_markov_payload, payload)
        rows.append(item)
        payload_by_name[cand.name] = payload

    rows.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    best_row = rows[0]
    best_candidate = candidate_by_name[best_row['candidate_name']]
    best_markov_payload = payload_by_name[best_candidate.name]

    if best_candidate.name == incumbent_cand.name:
        best_kf_payload = incumbent_kf_payload
        best_kf_path = incumbent_kf_path
        best_kf_status = 'reused_verified_oldvalid'
    else:
        best_kf_payload, best_kf_status, best_kf_path = run_candidate_payload(mod, best_candidate, 'kf36_noisy', args.noise_scale, args.force_rerun)

    best_summary = {
        'candidate_name': best_candidate.name,
        'direction': best_candidate.direction,
        'family': best_candidate.family,
        'base_reference': best_candidate.base_reference,
        'note': best_candidate.note,
        'flips': list(best_candidate.flips),
        'action_sequence': best_candidate.action_sequence,
        'rows': best_candidate.rows,
        'faces': best_candidate.faces,
        'markov42': compact_result(best_markov_payload),
        'markov42_run_json': best_row['run_json'],
        'kf36': compact_result(best_kf_payload),
        'kf36_run_json': str(best_kf_path),
        'kf36_status': best_kf_status,
        'delta_vs_incumbent': delta_vs_incumbent(incumbent_markov_payload, best_markov_payload),
        'delta_vs_faithful_markov42': compare_vs_base(baselines['faithful_markov'], best_markov_payload),
        'delta_vs_default18_markov42': compare_vs_base(baselines['default_markov'], best_markov_payload),
    }

    material = materially_beats_incumbent(incumbent_markov_payload, best_markov_payload)
    if best_candidate.name == INCUMBENT_NAME:
        verdict = 'The relaunched narrowed second layer also failed to improve on the incumbent: none of the one-flip neighbors or lightweight retime families beat legal_flip_8_11_12 on Markov42 mean.'
    elif material:
        verdict = f'The relaunched narrowed second layer found a materially better legal candidate: {best_candidate.name}.'
    else:
        verdict = (
            f'The relaunched narrowed second layer found {best_candidate.name} as the best new narrow candidate, '
            'but the gain over legal_flip_8_11_12 is not material under the shared Markov42 metric.'
        )

    out_json = RESULTS_DIR / f'ch3_12pos_legal_dualaxis_second_layer_{make_suffix(args.noise_scale)}.json'
    out_md = REPORTS_DIR / f'psins_ch3_12pos_legal_dualaxis_second_layer_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_12pos_legal_dualaxis_second_layer',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'hard_constraints': {
            'representation': 'legal motor-action sequence only',
            'inner_axis': 'body y only',
            'outer_axis': 'x/z family determined by inner attitude',
            'prefer_exact_12_nodes': True,
            'total_time_s_max': 1200.0,
            'seed': 42,
            'base_family': 'round53_round61_shared',
        },
        'references': {
            'faithful12': {
                'candidate_name': 'legal_base_faithful12',
                'markov42': compact_result(baselines['faithful_markov']),
                'markov42_run_json': str(RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'),
                'kf36': compact_result(baselines['faithful_kf']),
                'kf36_run_json': str(RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json'),
            },
            'default18': {
                'candidate_name': 'default18_reference',
                'markov42': compact_result(baselines['default_markov']),
                'markov42_run_json': str(RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json'),
                'kf36': compact_result(baselines['default_kf']),
                'kf36_run_json': str(RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json'),
            },
        },
        'incumbent': {
            'candidate_name': incumbent_cand.name,
            'flips': list(incumbent_cand.flips),
            'equivalent_prior_result': VALID_PRIOR_SIGN_ONLY_PATHS[2],
            'action_sequence': incumbent_cand.action_sequence,
            'rows': incumbent_cand.rows,
            'faces': incumbent_cand.faces,
            'markov42': compact_result(incumbent_markov_payload),
            'markov42_run_json': str(incumbent_markov_path),
            'kf36': compact_result(incumbent_kf_payload),
            'kf36_run_json': str(incumbent_kf_path),
        },
        'direction_a_candidates': [
            {
                'name': cand.name,
                'family': cand.family,
                'note': cand.note,
                'base_reference': cand.base_reference,
                'flips': list(cand.flips),
                'structural': {
                    'face_sequence': cand.structural['face_sequence'],
                    'axis_balance': cand.structural['axis_balance'],
                    'late_repeat_nodes': cand.structural['late_repeat_nodes'],
                    'repeat_and_worsen_target_nodes': cand.structural['repeat_and_worsen_target_nodes'],
                },
            }
            for cand in direction_a
        ],
        'direction_b_candidates': [
            {
                'name': cand.name,
                'family': cand.family,
                'note': cand.note,
                'base_reference': cand.base_reference,
                'flips': list(cand.flips),
                'retimed_nodes': [row['pos_id'] for row in cand.rows if abs(row['pre_static_s'] - 10.0) > 1e-9 or abs(row['post_static_s'] - 80.0) > 1e-9],
                'timing_rows': [
                    {
                        'pos_id': row['pos_id'],
                        'rotation_time_s': row['rotation_time_s'],
                        'pre_static_s': row['pre_static_s'],
                        'post_static_s': row['post_static_s'],
                        'node_total_s': row['node_total_s'],
                    }
                    for row in cand.rows if abs(row['pre_static_s'] - 10.0) > 1e-9 or abs(row['post_static_s'] - 80.0) > 1e-9
                ],
            }
            for cand in direction_b
        ],
        'markov42_rows': rows,
        'best_of_relaunch': best_summary,
        'material_improvement_over_incumbent': material,
        'incumbent_vs_faithful': compare_vs_base(baselines['faithful_markov'], incumbent_markov_payload),
        'verdict': verdict,
        'files': {
            'json': str(out_json),
            'report': str(out_md),
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    out_md.write_text(render_report(payload), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps(payload['files'], ensure_ascii=False))
    print('BEST_RELAUNCH', best_summary['candidate_name'], best_summary['markov42']['overall'])
    print('MATERIAL_OVER_INCUMBENT', material)


if __name__ == '__main__':
    main()
