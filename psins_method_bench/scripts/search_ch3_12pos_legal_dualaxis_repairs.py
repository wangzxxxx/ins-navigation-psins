from __future__ import annotations

import argparse
import itertools
import json
import math
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
from benchmark_ch3_12pos_goalA_repairs import (
    KEY_PARAMS,
    compact_result,
    orientation_faces,
    paras_to_rows,
    rows_to_paras,
    structural_summary,
)
from compare_ch3_12pos_path_baselines import build_ch3_path_paras, build_dataset_with_path
from compare_four_methods_shared_noise import _load_json, _noise_matches, compute_payload, expected_noise_config
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs
from diagnose_ch3_12pos_narrow import structural_penalty

NOISE_SCALE = 0.08
BASE_NODE_TOTAL_S = 100.0
BASE_ROTATION_S = 10.0
BASE_PRE_S = 10.0
BASE_POST_S = 80.0
TOP_STRUCTURAL_UNIQUE = 6
REPORT_DATE = datetime.now().strftime('%Y-%m-%d')

# Minimal constrained repair search:
# preserve the faithful 12-node / 1200 s O-O-I scaffold,
# but search ONLY over legal motor sign choices.
# This avoids invalid unconstrained body-axis reorders while staying close to chapter-3.
BASE_ACTION_TEMPLATE = [
    ('outer', +90),
    ('outer', +90),
    ('inner', +90),
    ('outer', -90),
    ('outer', -90),
    ('inner', -90),
    ('outer', -90),
    ('outer', -90),
    ('inner', -90),
    ('outer', +90),
    ('outer', +90),
    ('inner', +90),
]

SEEDED_CANDIDATES = [
    (10, 11),
    (10, 12),
    (8, 11, 12),
]

OLD_RESULT_MAP = {
    (10, 11): {
        'markov42_noisy': RESULTS_DIR / 'markov42_noisy_coupled_flip_10_11_shared_noise0p08_param_errors.json',
    },
    (10, 12): {
        'markov42_noisy': RESULTS_DIR / 'markov42_noisy_coupled_flip_10_12_shared_noise0p08_param_errors.json',
        'kf36_noisy': RESULTS_DIR / 'kf36_noisy_coupled_flip_10_12_shared_noise0p08_param_errors.json',
    },
    (8, 11, 12): {
        'markov42_noisy': RESULTS_DIR / 'markov42_noisy_coupled_flip_8_11_12_shared_noise0p08_param_errors.json',
        'kf36_noisy': RESULTS_DIR / 'kf36_noisy_coupled_flip_8_11_12_shared_noise0p08_param_errors.json',
    },
}

INVALID_UNCONSTRAINED_PRIOR_PATHS = [
    'coupled_B_move7to4_move12to10',
    'coupled_C_move7to4_move12to10_retime',
    'coupled_D_move7to4_move10to8',
    'coupled_E_move7to4_move10to8_retime',
]

VALID_PRIOR_SIGN_ONLY_PATHS = [
    'coupled_flip_10_11',
    'coupled_flip_10_12',
    'coupled_flip_8_11_12',
]


@dataclass
class Candidate:
    flips: tuple[int, ...]
    name: str
    action_sequence: list[dict]
    rows: list[dict]
    faces: list[dict]
    structural: dict
    penalty: int
    equivalent_prior_result: str | None


class DiscreteDualAxisKinematics:
    """Legal motor kinematics under the user's real two-axis mechanism.

    - Inner action: always about current body y-axis.
    - Outer action: axis family is determined by current inner angle beta.
      Starting at beta=0 => body x family.
      After |beta| = 90 deg => body z family.
      More generally axis_body = Ry(-beta) * ex, which for 90-deg lattice gives ±x / ±z.
    """

    def __init__(self):
        self.beta_deg = 0

    @staticmethod
    def _wrap_deg(angle_deg: int) -> int:
        x = int(round(angle_deg)) % 360
        if x > 180:
            x -= 360
        return x

    def outer_axis_body(self) -> list[int]:
        beta = math.radians(self.beta_deg)
        c = round(math.cos(beta))
        s = round(math.sin(beta))
        return [int(c), 0, int(s)]

    def apply(self, idx: int, kind: str, angle_deg: int) -> tuple[dict, dict]:
        if kind == 'inner':
            axis = [0, 1, 0]
            mode = 'y'
        elif kind == 'outer':
            axis = self.outer_axis_body()
            mode = 'x' if abs(axis[0]) == 1 else 'z'
        else:
            raise KeyError(kind)

        row = {
            'pos_id': idx,
            'axis': axis,
            'angle_deg': float(angle_deg),
            'rotation_time_s': BASE_ROTATION_S,
            'pre_static_s': BASE_PRE_S,
            'post_static_s': BASE_POST_S,
            'node_total_s': BASE_NODE_TOTAL_S,
        }
        action = {
            'pos_id': idx,
            'kind': kind,
            'motor_angle_deg': int(angle_deg),
            'effective_body_axis': axis,
            'outer_mode': mode if kind == 'outer' else None,
            'inner_beta_before_deg': self.beta_deg,
        }

        if kind == 'inner':
            self.beta_deg = self._wrap_deg(self.beta_deg + int(angle_deg))
        action['inner_beta_after_deg'] = self.beta_deg
        return action, row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def make_suffix(noise_scale: float) -> str:
    if abs(noise_scale - 0.08) < 1e-12:
        return 'noise0p08'
    if abs(noise_scale - 1.0) < 1e-12:
        return 'noise1p0'
    return f"noise{str(noise_scale).replace('.', 'p')}"


def candidate_name(flips: tuple[int, ...]) -> str:
    if not flips:
        return 'legal_base_faithful12'
    return 'legal_flip_' + '_'.join(str(x) for x in flips)


def candidate_short_tag(flips: tuple[int, ...]) -> str:
    if not flips:
        return 'ch3faithful12'
    return 'ch3legaldual_flip_' + '_'.join(str(x) for x in flips)


def build_candidate(mod, flips: tuple[int, ...]) -> Candidate:
    flip_set = set(flips)
    kin = DiscreteDualAxisKinematics()
    actions = []
    rows = []
    for idx, (kind, base_angle_deg) in enumerate(BASE_ACTION_TEMPLATE, start=1):
        angle_deg = -base_angle_deg if idx in flip_set else base_angle_deg
        action, row = kin.apply(idx, kind, angle_deg)
        actions.append(action)
        rows.append(row)

    paras = rows_to_paras(mod, rows)
    faces = orientation_faces(mod, paras)
    structural = structural_summary(rows, faces)
    penalty = structural_penalty(structural)
    prior = None
    if flips in OLD_RESULT_MAP:
        prior = VALID_PRIOR_SIGN_ONLY_PATHS[[tuple(x) for x in SEEDED_CANDIDATES].index(flips)]
    return Candidate(
        flips=flips,
        name=candidate_name(flips),
        action_sequence=actions,
        rows=rows,
        faces=faces,
        structural=structural,
        penalty=penalty,
        equivalent_prior_result=prior,
    )


def select_structural_candidates(mod) -> tuple[Candidate, list[Candidate], list[Candidate]]:
    base = build_candidate(mod, ())
    all_candidates = []
    for mask in range(1 << len(BASE_ACTION_TEMPLATE)):
        flips = tuple(i + 1 for i in range(len(BASE_ACTION_TEMPLATE)) if (mask >> i) & 1)
        cand = build_candidate(mod, flips)
        axis = cand.structural['axis_balance']
        score = (
            cand.penalty,
            abs(axis['Y']),
            abs(axis['Z']),
            len(cand.structural['late_repeat_nodes']),
            len(cand.structural['repeat_and_worsen_target_nodes']),
            len(cand.flips),
            cand.flips,
        )
        all_candidates.append((score, cand))

    all_candidates.sort(key=lambda x: x[0])

    picked = []
    seen_faces = set()
    for _, cand in all_candidates:
        key = tuple(cand.structural['face_sequence'])
        if key in seen_faces:
            continue
        seen_faces.add(key)
        picked.append(cand)
        if len(picked) >= TOP_STRUCTURAL_UNIQUE:
            break

    benchmark_list = list(picked)
    existing = {cand.flips for cand in benchmark_list}
    for flips in SEEDED_CANDIDATES:
        if flips not in existing:
            benchmark_list.append(build_candidate(mod, flips))
            existing.add(flips)

    return base, picked, benchmark_list


def method_output_path(flips: tuple[int, ...], method_key: str, noise_scale: float) -> Path:
    suffix = make_suffix(noise_scale)
    tag = candidate_short_tag(flips)
    if method_key == 'markov42_noisy':
        return RESULTS_DIR / f'M_markov_42state_gm1_{tag}_shared_{suffix}_param_errors.json'
    if method_key == 'kf36_noisy':
        return RESULTS_DIR / f'KF36_{tag}_shared_{suffix}_param_errors.json'
    raise KeyError(method_key)


def load_or_run_payload(mod, cand: Candidate, method_key: str, noise_scale: float, force_rerun: bool = False):
    expected_cfg = expected_noise_config(noise_scale)

    if not cand.flips:
        if method_key == 'markov42_noisy':
            out_path = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
        elif method_key == 'kf36_noisy':
            out_path = RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json'
        else:
            raise KeyError(method_key)
        payload = _load_json(out_path)
        if _noise_matches(payload, expected_cfg):
            return payload, 'reused_verified', out_path

    old_paths = OLD_RESULT_MAP.get(cand.flips, {})
    if method_key in old_paths and old_paths[method_key].exists() and (not force_rerun):
        payload = _load_json(old_paths[method_key])
        if _noise_matches(payload, expected_cfg):
            return payload, 'reused_verified_oldvalid', old_paths[method_key]

    out_path = method_output_path(cand.flips, method_key, noise_scale)
    if out_path.exists() and (not force_rerun):
        payload = _load_json(out_path)
        if _noise_matches(payload, expected_cfg) and payload.get('extra', {}).get('candidate_name') == cand.name:
            return payload, 'reused_verified', out_path

    paras = rows_to_paras(mod, cand.rows)
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
            label=f'{cand.name}_{method_key}_{make_suffix(noise_scale)}',
        )
    elif method_key == 'kf36_noisy':
        clbt, _, _, _, _ = mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=36,
            label=f'{cand.name}_{method_key}_{make_suffix(noise_scale)}',
        )
    else:
        raise KeyError(method_key)

    payload = compute_payload(
        mod,
        clbt,
        params,
        variant=f'{cand.name}_{method_key}_{make_suffix(noise_scale)}',
        method_file='search_ch3_12pos_legal_dualaxis_repairs.py',
        extra={
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'comparison_mode': 'ch3_legal_dualaxis_repair_search',
            'candidate_name': cand.name,
            'candidate_flips': list(cand.flips),
            'method_key': method_key,
            'legality': 'true_dual_axis_motor_sequence',
        },
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', out_path


def compare_vs_base(base_payload: dict, cand_payload: dict) -> dict:
    out = {}
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        bv = float(base_payload['overall'][metric])
        cv = float(cand_payload['overall'][metric])
        out[metric] = {
            'base': bv,
            'candidate': cv,
            'improvement_pct_points': bv - cv,
            'relative_improvement_pct': ((bv - cv) / bv * 100.0) if abs(bv) > 1e-12 else None,
        }
    out['key_params'] = {}
    for key in KEY_PARAMS:
        bv = float(base_payload['param_errors'][key]['pct_error'])
        cv = float(cand_payload['param_errors'][key]['pct_error'])
        out['key_params'][key] = {
            'base': bv,
            'candidate': cv,
            'improvement_pct_points': bv - cv,
        }
    return out


def render_action(action: dict) -> str:
    if action['kind'] == 'inner':
        return f"I({action['inner_beta_before_deg']}→{action['inner_beta_after_deg']}) {action['motor_angle_deg']:+d}°"
    axis = action['effective_body_axis']
    axis_name = 'X' if abs(axis[0]) == 1 else 'Z'
    signed_axis = ('+' if sum(axis) >= 0 else '-') + axis_name
    return f"O[{signed_axis}] {action['motor_angle_deg']:+d}°"


def render_report(payload: dict) -> str:
    lines = []
    lines.append('# Chapter-3 12-position legal dual-axis repair search')
    lines.append('')
    lines.append('## 1. Kinematic rule enforced in this search')
    lines.append('')
    lines.append('- valid search objects are **motor action sequences**, not arbitrary body-axis reordered tables')
    lines.append('- inner motor: always rotate about current body y-axis')
    lines.append('- outer motor: axis family is determined by current inner angle; with 90° lattice it alternates between body x-family and body z-family')
    lines.append('- search family used here: preserve faithful chapter-3 `O,O,I | O,O,I | O,O,I | O,O,I` 12-node scaffold and search only sign-repairs inside this legal motor scaffold')
    lines.append(f"- timing preserved exactly: 12 nodes × 100 s = **{payload['timing']['total_time_s']:.0f} s**")
    lines.append('')
    lines.append('## 2. Invalid vs valid prior families')
    lines.append('')
    lines.append(f"- **invalid under real hardware** (unconstrained body-axis reorders): {', '.join(payload['invalid_unconstrained_prior_paths'])}")
    lines.append(f"- **still valid under real hardware** (pure sign-flip motor-sequence repairs): {', '.join(payload['valid_prior_sign_only_paths'])}")
    lines.append('')
    lines.append('## 3. Structural screening over legal motor sequences')
    lines.append('')
    lines.append(f"- searched legal sign-repair space size: **{payload['search_space']['n_legal_sequences']}**")
    lines.append(f"- top unique structural shortlist: **{len(payload['structural_shortlist'])}**")
    lines.append(f"- additional seeded prior-valid sign-only candidates: **{payload['search_space']['n_seeded_valid_candidates']}**")
    lines.append(f"- total benchmarked legal candidates: **{len(payload['benchmarked_candidates'])}**")
    lines.append('')
    lines.append('| candidate | flips | penalty | face_sequence | prior_equiv |')
    lines.append('|---|---|---:|---|---|')
    for item in payload['structural_shortlist']:
        lines.append(
            f"| {item['name']} | {item['flips']} | {item['penalty']} | {' → '.join(item['face_sequence'])} | {item['equivalent_prior_result'] or '-'} |"
        )
    lines.append('')
    lines.append('## 4. Markov42 benchmark on legal candidates (noise0p08 shared seed/truth family)')
    lines.append('')
    lines.append('| candidate | flips | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Ka2_z | run_json |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|---|')
    for row in payload['markov42_rows']:
        m = row['metrics']['overall']
        k = row['metrics']['key_param_errors']
        lines.append(
            f"| {row['candidate_name']} | {row['flips']} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {k['Ka2_z']:.3f} | `{row['run_json']}` |"
        )
    lines.append('')
    best = payload['best_candidate']
    lines.append('## 5. Selected constrained candidate')
    lines.append('')
    lines.append(f"- selected legal candidate: **{best['candidate_name']}**")
    lines.append(f"- flip set relative to faithful motor scaffold: `{best['flips']}`")
    lines.append(f"- equivalent prior sign-only candidate: `{best['equivalent_prior_result'] or '-'}`")
    lines.append(f"- Markov42 overall: mean **{best['markov42']['overall']['mean_pct_error']:.3f}**, median **{best['markov42']['overall']['median_pct_error']:.3f}**, max **{best['markov42']['overall']['max_pct_error']:.3f}**")
    if best.get('kf36'):
        lines.append(f"- KF36 recheck: mean **{best['kf36']['overall']['mean_pct_error']:.3f}**, median **{best['kf36']['overall']['median_pct_error']:.3f}**, max **{best['kf36']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('### 5.1 vs faithful chapter-3 baseline (Markov42)')
    lines.append('')
    lines.append('| metric | faithful | candidate | improvement | relative |')
    lines.append('|---|---:|---:|---:|---:|')
    for metric, vals in best['delta_vs_faithful_markov42'].items():
        if metric == 'key_params':
            continue
        rel = vals['relative_improvement_pct']
        rel_text = '-' if rel is None else f"{rel:.2f}%"
        lines.append(
            f"| {metric} | {vals['base']:.3f} | {vals['candidate']:.3f} | {vals['improvement_pct_points']:+.3f} | {rel_text} |"
        )
    lines.append('')
    lines.append('### 5.2 key weak-parameter deltas (Markov42)')
    lines.append('')
    lines.append('| param | faithful | candidate | improvement |')
    lines.append('|---|---:|---:|---:|')
    for key, vals in best['delta_vs_faithful_markov42']['key_params'].items():
        lines.append(f"| {key} | {vals['base']:.3f} | {vals['candidate']:.3f} | {vals['improvement_pct_points']:+.3f} |")
    lines.append('')
    lines.append('## 6. Exact legal motor sequence and decoded effective path')
    lines.append('')
    lines.append('| idx | legal motor action | decoded body axis | angle_deg | gravity face after node |')
    lines.append('|---:|---|---|---:|---|')
    for action, row, face in zip(best['action_sequence'], best['rows'], best['faces']):
        lines.append(
            f"| {row['pos_id']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {face['face_name']} |"
        )
    lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_module('search_ch3_legal_dualaxis_src', str(SOURCE_FILE))

    faithful = build_candidate(mod, ())
    base_markov_payload, _, base_markov_path = load_or_run_payload(mod, faithful, 'markov42_noisy', args.noise_scale, args.force_rerun)
    base_kf_payload, _, base_kf_path = load_or_run_payload(mod, faithful, 'kf36_noisy', args.noise_scale, args.force_rerun)

    _, shortlist, benchmark_cands = select_structural_candidates(mod)
    markov_rows = []
    benchmark_payloads = {}

    for cand in benchmark_cands:
        payload, status, path = load_or_run_payload(mod, cand, 'markov42_noisy', args.noise_scale, args.force_rerun)
        benchmark_payloads[cand.flips] = payload
        markov_rows.append({
            'candidate_name': cand.name,
            'flips': list(cand.flips),
            'metrics': compact_result(payload),
            'run_json': str(path),
            'status': status,
            'equivalent_prior_result': cand.equivalent_prior_result,
        })

    markov_rows.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    best_row = markov_rows[0]
    best_cand = next(c for c in benchmark_cands if c.flips == tuple(best_row['flips']))

    best_kf_payload, best_kf_status, best_kf_path = load_or_run_payload(mod, best_cand, 'kf36_noisy', args.noise_scale, args.force_rerun)

    best_summary = {
        'candidate_name': best_cand.name,
        'flips': list(best_cand.flips),
        'equivalent_prior_result': best_cand.equivalent_prior_result,
        'action_sequence': best_cand.action_sequence,
        'rows': best_cand.rows,
        'faces': best_cand.faces,
        'structural': best_cand.structural,
        'markov42': compact_result(benchmark_payloads[best_cand.flips]),
        'markov42_run_json': best_row['run_json'],
        'kf36': compact_result(best_kf_payload),
        'kf36_run_json': str(best_kf_path),
        'delta_vs_faithful_markov42': compare_vs_base(base_markov_payload, benchmark_payloads[best_cand.flips]),
        'delta_vs_faithful_kf36': compare_vs_base(base_kf_payload, best_kf_payload),
    }

    shortlist_payload = []
    for cand in shortlist:
        shortlist_payload.append({
            'name': cand.name,
            'flips': list(cand.flips),
            'penalty': cand.penalty,
            'face_sequence': cand.structural['face_sequence'],
            'equivalent_prior_result': cand.equivalent_prior_result,
        })

    out_json = RESULTS_DIR / f'ch3_12pos_legal_dualaxis_repair_search_{make_suffix(args.noise_scale)}.json'
    out_md = REPORTS_DIR / f'psins_ch3_12pos_legal_dualaxis_repair_search_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_12pos_legal_dualaxis_repair_search',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'timing': {
            'per_node_total_s': BASE_NODE_TOTAL_S,
            'rotation_time_s': BASE_ROTATION_S,
            'pre_static_s': BASE_PRE_S,
            'post_static_s': BASE_POST_S,
            'total_time_s': BASE_NODE_TOTAL_S * len(BASE_ACTION_TEMPLATE),
        },
        'search_space': {
            'description': 'faithful 12-node OOI scaffold with legal sign-repair only',
            'n_legal_sequences': 2 ** len(BASE_ACTION_TEMPLATE),
            'n_seeded_valid_candidates': len(SEEDED_CANDIDATES),
            'template': [{'idx': i + 1, 'kind': kind, 'base_angle_deg': angle} for i, (kind, angle) in enumerate(BASE_ACTION_TEMPLATE)],
        },
        'invalid_unconstrained_prior_paths': INVALID_UNCONSTRAINED_PRIOR_PATHS,
        'valid_prior_sign_only_paths': VALID_PRIOR_SIGN_ONLY_PATHS,
        'faithful_baseline': {
            'candidate_name': faithful.name,
            'flips': [],
            'action_sequence': faithful.action_sequence,
            'rows': faithful.rows,
            'faces': faithful.faces,
            'markov42': compact_result(base_markov_payload),
            'markov42_run_json': str(base_markov_path),
            'kf36': compact_result(base_kf_payload),
            'kf36_run_json': str(base_kf_path),
        },
        'structural_shortlist': shortlist_payload,
        'benchmarked_candidates': [
            {
                'candidate_name': row['candidate_name'],
                'flips': row['flips'],
                'status': row['status'],
                'equivalent_prior_result': row['equivalent_prior_result'],
            }
            for row in markov_rows
        ],
        'markov42_rows': markov_rows,
        'best_candidate': best_summary,
        'files': {
            'json': str(out_json),
            'report': str(out_md),
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    out_md.write_text(render_report(payload), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps(payload['files'], ensure_ascii=False))
    print('BEST_CANDIDATE', best_summary['candidate_name'], best_summary['markov42']['overall'])
    print('BEST_KF36', best_summary['kf36']['overall'])


if __name__ == '__main__':
    main()
