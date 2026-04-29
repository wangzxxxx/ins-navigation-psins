from __future__ import annotations

import argparse
import json
import re
import sys
import types
from dataclasses import dataclass
from datetime import datetime
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

from common_markov import load_module
from benchmark_ch3_12pos_goalA_repairs import compact_result, orientation_faces, rows_to_paras
from compare_ch3_12pos_path_baselines import build_dataset_with_path
from compare_four_methods_shared_noise import _load_json, _noise_matches, compute_payload, expected_noise_config
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate, compare_vs_base, make_suffix, render_action

NOISE_SCALE = 0.08
REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
FAITHFUL_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
FAITHFUL_KF_RESULT = RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json'
OLD_BEST_RESULT = RESULTS_DIR / 'M_markov_42state_gm1_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
OLD_BEST_KF_RESULT = RESULTS_DIR / 'KF36_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'


@dataclass
class StepSpec:
    kind: str
    angle_deg: int
    rotation_time_s: float
    pre_static_s: float
    post_static_s: float
    segment_role: str
    label: str


@dataclass
class ClosedLoopCandidate:
    name: str
    rationale: str
    insertions: dict[int, list[StepSpec]]
    total_time_s: float
    all_rows: list[dict[str, Any]]
    all_actions: list[dict[str, Any]]
    all_faces: list[dict[str, Any]]
    continuity_checks: list[dict[str, Any]]
    method_tag: str


class MechanismTracker:
    """Track real dual-axis mechanism state, not just gravity face.

    State includes:
    - inner-axis angle beta (deg)
    - current outer-axis family in body coordinates (x/z sign)
    - full orientation matrix C (for exact pose closure / next-step legality)
    """

    def __init__(self, mod):
        self.mod = mod
        self.beta_deg = 0
        self.C = mod.np.eye(3)
        from psins_py.math_utils import rv2m

        self._rv2m = rv2m

    @staticmethod
    def _wrap_deg(angle_deg: int) -> int:
        x = int(round(angle_deg)) % 360
        if x > 180:
            x -= 360
        return x

    def clone(self) -> 'MechanismTracker':
        other = MechanismTracker(self.mod)
        other.beta_deg = int(self.beta_deg)
        other.C = self.mod.np.array(self.C, dtype=float)
        return other

    def outer_axis_body(self) -> list[int]:
        beta = self.mod.np.deg2rad(self.beta_deg)
        c = round(float(self.mod.np.cos(beta)))
        s = round(float(self.mod.np.sin(beta)))
        return [int(c), 0, int(s)]

    def state_snapshot(self) -> dict[str, Any]:
        g_body = self.C.T @ self.mod.np.array([0.0, 0.0, 1.0])
        face_name = self._nearest_face(g_body.tolist())[0]
        return {
            'beta_deg': int(self.beta_deg),
            'outer_axis_body': self.outer_axis_body(),
            'outer_family': 'x' if abs(self.outer_axis_body()[0]) == 1 else 'z',
            'gravity_body': [float(x) for x in g_body.tolist()],
            'face_name': face_name,
            'C': [[float(v) for v in row] for row in self.C.tolist()],
        }

    @staticmethod
    def _nearest_face(vec: list[float]) -> tuple[str, tuple[int, int, int]]:
        rounded = [0, 0, 0]
        idx = int(max(range(3), key=lambda i: abs(vec[i])))
        rounded[idx] = 1 if vec[idx] >= 0 else -1
        key = tuple(rounded)
        face_names = {
            (1, 0, 0): '+X',
            (-1, 0, 0): '-X',
            (0, 1, 0): '+Y',
            (0, -1, 0): '-Y',
            (0, 0, 1): '+Z',
            (0, 0, -1): '-Z',
        }
        return face_names[key], key

    def apply(self, pos_id: int, kind: str, angle_deg: int, rotation_time_s: float, pre_static_s: float, post_static_s: float, *, anchor_id: int, segment_role: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
        before = self.state_snapshot()
        if kind == 'inner':
            axis = [0, 1, 0]
            outer_mode = None
        elif kind == 'outer':
            axis = self.outer_axis_body()
            outer_mode = 'x' if abs(axis[0]) == 1 else 'z'
        else:
            raise KeyError(kind)

        row = {
            'pos_id': pos_id,
            'anchor_id': anchor_id,
            'axis': axis,
            'angle_deg': float(angle_deg),
            'rotation_time_s': float(rotation_time_s),
            'pre_static_s': float(pre_static_s),
            'post_static_s': float(post_static_s),
            'node_total_s': float(rotation_time_s + pre_static_s + post_static_s),
            'segment_role': segment_role,
            'label': label,
        }
        action = {
            'pos_id': pos_id,
            'anchor_id': anchor_id,
            'kind': kind,
            'motor_angle_deg': int(angle_deg),
            'effective_body_axis': axis,
            'outer_mode': outer_mode,
            'inner_beta_before_deg': int(self.beta_deg),
            'segment_role': segment_role,
            'label': label,
            'state_before': before,
        }

        if kind == 'inner':
            self.beta_deg = self._wrap_deg(self.beta_deg + int(angle_deg))

        axis_vec = self.mod.np.asarray(axis, dtype=float)
        axis_vec = axis_vec / self.mod.np.linalg.norm(axis_vec)
        self.C = self.C @ self._rv2m(axis_vec * float(angle_deg) * self.mod.glv.deg)

        after = self.state_snapshot()
        action['inner_beta_after_deg'] = int(self.beta_deg)
        action['state_after'] = after
        return row, action


def sanitize_tag(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def closed_pair(kind: str, first_angle_deg: int, rotation_time_s: float, dwell_s: float, label: str) -> list[StepSpec]:
    return [
        StepSpec(kind=kind, angle_deg=first_angle_deg, rotation_time_s=rotation_time_s, pre_static_s=0.0, post_static_s=dwell_s, segment_role='motif_out', label=f'{label}_out'),
        StepSpec(kind=kind, angle_deg=-first_angle_deg, rotation_time_s=rotation_time_s, pre_static_s=0.0, post_static_s=dwell_s, segment_role='motif_return', label=f'{label}_return'),
    ]


def inner_outer_pair_return(inner_angle_deg: int, outer_angle_deg: int, rotation_time_s: float, dwell_s: float, label: str) -> list[StepSpec]:
    return [
        StepSpec(kind='inner', angle_deg=inner_angle_deg, rotation_time_s=rotation_time_s, pre_static_s=0.0, post_static_s=dwell_s, segment_role='motif_inner_open', label=f'{label}_inner_open'),
        StepSpec(kind='outer', angle_deg=outer_angle_deg, rotation_time_s=rotation_time_s, pre_static_s=0.0, post_static_s=dwell_s, segment_role='motif_outer_sweep', label=f'{label}_outer_sweep'),
        StepSpec(kind='outer', angle_deg=-outer_angle_deg, rotation_time_s=rotation_time_s, pre_static_s=0.0, post_static_s=dwell_s, segment_role='motif_outer_return', label=f'{label}_outer_return'),
        StepSpec(kind='inner', angle_deg=-inner_angle_deg, rotation_time_s=rotation_time_s, pre_static_s=0.0, post_static_s=dwell_s, segment_role='motif_inner_close', label=f'{label}_inner_close'),
    ]


CANDIDATE_SPECS = [
    {
        'name': 'late11_zpair_pos_med',
        'rationale': 'Late closed-loop z-family +90/-90 pair at node 11. This is the most direct faithful-base analogue of the previously helpful z11-family signal, but now forced to close exactly to the same node-11 mechanism state before node 12 resumes.',
        'insertions': {
            11: closed_pair('outer', +90, 5.0, 10.0, 'z11_pos_med'),
        },
    },
    {
        'name': 'late11_zpair_neg_med',
        'rationale': 'Same node-11 closed-loop z-family motif, but reverse sweep direction to test whether the useful late signal is direction-specific or mainly comes from the extra late z-family dwell itself.',
        'insertions': {
            11: closed_pair('outer', -90, 5.0, 10.0, 'z11_neg_med'),
        },
    },
    {
        'name': 'late11_zpair_pos_long',
        'rationale': 'Stronger node-11 closed-loop z-family pair with longer dwell. Intended to test whether the weak late block mainly lacks integration time rather than missing sweep direction.',
        'insertions': {
            11: closed_pair('outer', +90, 10.0, 20.0, 'z11_pos_long'),
        },
    },
    {
        'name': 'late11_zpair_pos_double',
        'rationale': 'Two consecutive closed-loop z-family pairs at node 11. Intended to amplify the only clearly helpful late-family signal while still reconnecting to the exact same node-11 state before node 12.',
        'insertions': {
            11: closed_pair('outer', +90, 5.0, 10.0, 'z11_pos_med_a') + closed_pair('outer', +90, 5.0, 10.0, 'z11_pos_med_b'),
        },
    },
    {
        'name': 'late10_11_zpair_pos_med',
        'rationale': 'Distribute one closed-loop z-family pair across both late anchors 10 and 11, instead of concentrating all added excitation at the last anchor. Intended to probe whether the weak block needs breadth across the whole late z-family segment.',
        'insertions': {
            10: closed_pair('outer', +90, 5.0, 10.0, 'z10_pos_med'),
            11: closed_pair('outer', +90, 5.0, 10.0, 'z11_pos_med'),
        },
    },
    {
        'name': 'late11_ypair_neg_med',
        'rationale': 'Minimal closed-loop inner -90/+90 pair at node 11. This is the safest faithful-base way to test whether a late y-sensitive local excursion can cut dKg_zz / dKa_yy without breaking late-block continuity.',
        'insertions': {
            11: closed_pair('inner', -90, 5.0, 10.0, 'y11_neg_med'),
        },
    },
    {
        'name': 'late11_zpair_pos_med_then_ypair_neg_med',
        'rationale': 'Combine the safest helpful late z-family motif with a small late y-sensitive closed-loop excursion at the same anchor. Intended to protect already-good channels with the z-loop, then add one extra y-sensitive probe for dKg_zz / dKa_yy.',
        'insertions': {
            11: closed_pair('outer', +90, 5.0, 10.0, 'z11_pos_med') + closed_pair('inner', -90, 5.0, 10.0, 'y11_neg_med'),
        },
    },
    {
        'name': 'late11_yneg_xpair_return',
        'rationale': 'Closed-loop inner-open / x-family outer pair / inner-close motif launched from node 11. This adds a late y excursion plus a protected x-family pair, then returns exactly to the node-11 state before node 12.',
        'insertions': {
            11: inner_outer_pair_return(-90, +90, 5.0, 5.0, 'yneg_xpair_11'),
        },
    },
    {
        'name': 'late11_yneg_xpair_negouter_return',
        'rationale': 'Same late11 yneg-xpair closed loop, but flip the x-family sweep direction so the local branch visits -Y instead of +Y. This tests whether the mean win of the xpair family is tied to the y-sign of the intermediate branch.',
        'insertions': {
            11: inner_outer_pair_return(-90, -90, 5.0, 5.0, 'yneg_xpair_negouter_11'),
        },
    },
    {
        'name': 'late11_zpair_pos_med_then_yneg_xpair_return',
        'rationale': 'Batch-2 combo: first use the best max-protecting late z-pair, then add the best mean-recovering late yneg-xpair motif at the same anchor. Intended to merge the max benefit of zpair with the mean benefit of the xpair family.',
        'insertions': {
            11: closed_pair('outer', +90, 5.0, 10.0, 'z11_pos_med') + inner_outer_pair_return(-90, +90, 5.0, 5.0, 'yneg_xpair_11'),
        },
    },
    {
        'name': 'late11_yneg_xpair_return_then_zpair_pos_med',
        'rationale': 'Same two motifs as the previous candidate but in reversed order. This checks whether the local time ordering inside the weak block matters even when the mechanism state is closed-loop identical at resume.',
        'insertions': {
            11: inner_outer_pair_return(-90, +90, 5.0, 5.0, 'yneg_xpair_11') + closed_pair('outer', +90, 5.0, 10.0, 'z11_pos_med'),
        },
    },
]


def load_reference_payloads(noise_scale: float) -> dict[str, dict]:
    expected_cfg = expected_noise_config(noise_scale)
    faithful_markov = _load_json(FAITHFUL_RESULT)
    faithful_kf = _load_json(FAITHFUL_KF_RESULT)
    oldbest_markov = _load_json(OLD_BEST_RESULT)
    oldbest_kf = _load_json(OLD_BEST_KF_RESULT)
    for payload in [faithful_markov, faithful_kf, oldbest_markov, oldbest_kf]:
        if not _noise_matches(payload, expected_cfg):
            raise ValueError('Reference noise configuration mismatch')
    return {
        'faithful_markov': faithful_markov,
        'faithful_kf': faithful_kf,
        'oldbest_markov': oldbest_markov,
        'oldbest_kf': oldbest_kf,
    }


def states_match(a: dict[str, Any], b: dict[str, Any], tol: float = 1e-9) -> bool:
    if int(a['beta_deg']) != int(b['beta_deg']):
        return False
    if list(a['outer_axis_body']) != list(b['outer_axis_body']):
        return False
    import numpy as np

    return np.allclose(np.asarray(a['C'], dtype=float), np.asarray(b['C'], dtype=float), atol=tol, rtol=0.0)


def preview_next_base_action(tracker: MechanismTracker, action_template: dict[str, Any], base_row: dict[str, Any], anchor_id: int) -> dict[str, Any]:
    preview = tracker.clone()
    row, action = preview.apply(
        pos_id=9999,
        kind=action_template['kind'],
        angle_deg=int(action_template['motor_angle_deg']),
        rotation_time_s=float(base_row['rotation_time_s']),
        pre_static_s=float(base_row['pre_static_s']),
        post_static_s=float(base_row['post_static_s']),
        anchor_id=anchor_id,
        segment_role='resume_preview',
        label='resume_preview',
    )
    return {
        'kind': action['kind'],
        'motor_angle_deg': action['motor_angle_deg'],
        'effective_body_axis': action['effective_body_axis'],
        'outer_mode': action['outer_mode'],
        'state_before': action['state_before'],
        'state_after': action['state_after'],
        'face_after': row['axis'],
    }


def build_closedloop_candidate(mod, spec: dict[str, Any], base_rows: list[dict[str, Any]], base_actions: list[dict[str, Any]]) -> ClosedLoopCandidate:
    tracker = MechanismTracker(mod)
    all_rows: list[dict[str, Any]] = []
    all_actions: list[dict[str, Any]] = []
    continuity_checks: list[dict[str, Any]] = []

    for anchor_idx, (base_row, base_action) in enumerate(zip(base_rows, base_actions), start=1):
        row, action = tracker.apply(
            pos_id=len(all_rows) + 1,
            kind=base_action['kind'],
            angle_deg=int(base_action['motor_angle_deg']),
            rotation_time_s=float(base_row['rotation_time_s']),
            pre_static_s=float(base_row['pre_static_s']),
            post_static_s=float(base_row['post_static_s']),
            anchor_id=anchor_idx,
            segment_role='anchor',
            label=f'anchor_{anchor_idx}',
        )
        all_rows.append(row)
        all_actions.append(action)

        if anchor_idx in spec['insertions']:
            before_state = tracker.state_snapshot()
            insertion_steps: list[StepSpec] = spec['insertions'][anchor_idx]
            for step in insertion_steps:
                row, action = tracker.apply(
                    pos_id=len(all_rows) + 1,
                    kind=step.kind,
                    angle_deg=step.angle_deg,
                    rotation_time_s=step.rotation_time_s,
                    pre_static_s=step.pre_static_s,
                    post_static_s=step.post_static_s,
                    anchor_id=anchor_idx,
                    segment_role=step.segment_role,
                    label=step.label,
                )
                all_rows.append(row)
                all_actions.append(action)
            after_state = tracker.state_snapshot()
            closure_ok = states_match(before_state, after_state)
            if not closure_ok:
                raise ValueError(f'{spec["name"]} fails exact mechanism-state closure at anchor {anchor_idx}')
            next_action_preview = None
            if anchor_idx < len(base_actions):
                next_action_preview = preview_next_base_action(tracker, base_actions[anchor_idx], base_rows[anchor_idx], anchor_idx + 1)
            continuity_checks.append({
                'anchor_id': anchor_idx,
                'closure_rule': 'exact_same_mechanism_state_before_resume',
                'state_before_insertion': before_state,
                'state_after_insertion': after_state,
                'closure_ok': closure_ok,
                'next_base_action_preview': next_action_preview,
            })

    paras = rows_to_paras(mod, all_rows)
    all_faces = orientation_faces(mod, paras)
    total_time_s = float(sum(row['node_total_s'] for row in all_rows))
    if total_time_s < 1200.0 - 1e-9 or total_time_s > 1800.0 + 1e-9:
        raise ValueError(f'{spec["name"]} violates time budget: {total_time_s}')

    return ClosedLoopCandidate(
        name=spec['name'],
        rationale=spec['rationale'],
        insertions=spec['insertions'],
        total_time_s=total_time_s,
        all_rows=all_rows,
        all_actions=all_actions,
        all_faces=all_faces,
        continuity_checks=continuity_checks,
        method_tag=f'ch3closedloop_{sanitize_tag(spec["name"])}',
    )


def candidate_result_path(candidate: ClosedLoopCandidate, method_key: str, noise_scale: float) -> Path:
    suffix = make_suffix(noise_scale)
    prefix = 'M_markov_42state_gm1' if method_key == 'markov42_noisy' else 'KF36'
    return RESULTS_DIR / f'{prefix}_{candidate.method_tag}_shared_{suffix}_param_errors.json'


def run_candidate_payload(mod, candidate: ClosedLoopCandidate, method_key: str, noise_scale: float, force_rerun: bool = False):
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
        method_file='search_ch3_12pos_closedloop_local_insertions.py',
        extra={
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'comparison_mode': 'ch3_closedloop_local_insertions',
            'candidate_name': candidate.name,
            'method_key': method_key,
            'legality': 'faithful12_base_plus_closed_loop_local_insertions',
            'time_total_s': candidate.total_time_s,
        },
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', out_path


def delta_vs_ref(ref_payload: dict[str, Any], cand_payload: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        rv = float(ref_payload['overall'][metric])
        cv = float(cand_payload['overall'][metric])
        out[metric] = {
            'reference': rv,
            'candidate': cv,
            'improvement_pct_points': rv - cv,
            'relative_improvement_pct': ((rv - cv) / rv * 100.0) if abs(rv) > 1e-12 else None,
        }
    return out


def render_report(payload: dict[str, Any]) -> str:
    lines = []
    lines.append('# Chapter-3 faithful12 closed-loop local-insertion branch')
    lines.append('')
    lines.append('## 1. Design rationale')
    lines.append('')
    lines.append('- **Why local insertion should help more than append-tail:** the late weak block is where the original faithful12 strategy enters its last z-family segment and then closes with the final inner return. A tail appended after node 12 sits outside this observability context and tends to disturb the already-set balance; a local insertion between late base nodes can inject extra excitation while still handing control back to the original strategy before the next base action.')
    lines.append('- **Targeting logic:**')
    lines.append('  - `dKg_zz`: prioritize **late z-family outer closed loops** at node 11, because node 11 is the last native z-family anchor before the base strategy closes the inner axis at node 12.')
    lines.append('  - `dKa_yy` / `Ka2_y`: add only **small late y-sensitive closed loops** near node 11, so any extra y-driven information is injected inside the weak block rather than as a disconnected tail.')
    lines.append('  - **Protected channels rule:** keep all extra motifs closed-loop at the same anchor, avoid changing the faithful12 base actions themselves, and prefer node-11-local motifs over early or cross-skeleton rewrites.')
    lines.append('')
    lines.append('## 2. Hard validity rule enforced')
    lines.append('')
    lines.append('- Base scaffold is the **original faithful chapter-3 12-position sequence**; no base-node sign flips are allowed in this branch.')
    lines.append('- Every candidate uses **closed-loop insertion-return motifs** only.')
    lines.append('- For each insertion, continuity is checked on the true mechanism state, not only the gravity face:')
    lines.append('  - inner angle `beta`')
    lines.append('  - outer-axis family / signed body axis')
    lines.append('  - full orientation matrix `C` (exact reachable pose)')
    lines.append('- A candidate is accepted only if it returns to the **same mechanism state** before the next original base action resumes.')
    lines.append('')
    lines.append('## 3. Fixed references')
    lines.append('')
    lines.append(f"- faithful12 Markov42: mean **{payload['references']['faithful12']['markov42']['overall']['mean_pct_error']:.3f}**, median **{payload['references']['faithful12']['markov42']['overall']['median_pct_error']:.3f}**, max **{payload['references']['faithful12']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- old best legal result Markov42 (`legal_flip_8_11_12_retime_flipnodes_pre20_post70`): mean **{payload['references']['old_best_legal']['markov42']['overall']['mean_pct_error']:.3f}**, median **{payload['references']['old_best_legal']['markov42']['overall']['median_pct_error']:.3f}**, max **{payload['references']['old_best_legal']['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('## 4. Batch-1 faithful-base closed-loop candidates (Markov42, shared noise0p08/seed42)')
    lines.append('')
    lines.append('| rank | candidate | total_s | mean | median | max | dKa_yy | dKg_zz | Ka2_y | Δmean vs faithful | Δmean vs old best | Δmax vs old best |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(payload['markov42_rows'], start=1):
        d_f = row['delta_vs_faithful']
        d_o = row['delta_vs_old_best']
        k = row['metrics']['key_param_errors']
        m = row['metrics']['overall']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['total_time_s']:.0f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKa_yy']:.3f} | {k['dKg_zz']:.3f} | {k['Ka2_y']:.3f} | {d_f['mean_pct_error']['improvement_pct_points']:+.3f} | {d_o['mean_pct_error']['improvement_pct_points']:+.3f} | {d_o['max_pct_error']['improvement_pct_points']:+.3f} |"
        )
    lines.append('')
    lines.append('## 5. Best closed-loop result')
    lines.append('')
    best = payload['best_candidate']
    lines.append(f"- best candidate: **{best['candidate_name']}**")
    lines.append(f"- total time: **{best['total_time_s']:.0f} s**")
    lines.append(f"- Markov42: mean **{best['markov42']['overall']['mean_pct_error']:.3f}**, median **{best['markov42']['overall']['median_pct_error']:.3f}**, max **{best['markov42']['overall']['max_pct_error']:.3f}**")
    lines.append(f"- vs faithful12: Δmean = **{best['delta_vs_faithful']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian = **{best['delta_vs_faithful']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax = **{best['delta_vs_faithful']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- vs old best legal result: Δmean = **{best['delta_vs_old_best']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian = **{best['delta_vs_old_best']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax = **{best['delta_vs_old_best']['max_pct_error']['improvement_pct_points']:+.3f}**")
    if best.get('kf36') is not None:
        lines.append(f"- KF36 recheck: mean **{best['kf36']['overall']['mean_pct_error']:.3f}**, median **{best['kf36']['overall']['median_pct_error']:.3f}**, max **{best['kf36']['overall']['max_pct_error']:.3f}**")
    lines.append('')
    lines.append('## 6. Continuity proof for the best candidate')
    lines.append('')
    for check in best['continuity_checks']:
        before = check['state_before_insertion']
        after = check['state_after_insertion']
        lines.append(f"- anchor {check['anchor_id']}: closure_ok = **{'yes' if check['closure_ok'] else 'no'}**")
        lines.append(
            f"  - before: beta={before['beta_deg']}°, outer_axis={before['outer_axis_body']}, face={before['face_name']}"
        )
        lines.append(
            f"  - after : beta={after['beta_deg']}°, outer_axis={after['outer_axis_body']}, face={after['face_name']}"
        )
        if check['next_base_action_preview'] is not None:
            preview = check['next_base_action_preview']
            lines.append(
                f"  - next original action remains legal as `{preview['kind']}` {preview['motor_angle_deg']:+d}° with effective axis {preview['effective_body_axis']}"
            )
    lines.append('')
    lines.append('## 7. Exact legal motor / timing table for the best candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for seq_idx, (action, row, face) in enumerate(zip(best['all_actions'], best['all_rows'], best['all_faces']), start=1):
        lines.append(
            f"| {seq_idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 8. Scientific conclusion')
    lines.append('')
    lines.append(f"- {payload['scientific_conclusion']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    mod = load_module('search_ch3_12pos_closedloop_local_insertions_src', str(SOURCE_FILE))

    refs = load_reference_payloads(args.noise_scale)
    faithful = build_candidate(mod, ())
    base_rows = faithful.rows
    base_actions = faithful.action_sequence

    candidates = [build_closedloop_candidate(mod, spec, base_rows, base_actions) for spec in CANDIDATE_SPECS]
    candidate_by_name = {cand.name: cand for cand in candidates}

    rows = []
    payload_by_name = {}
    for cand in candidates:
        print(f'RUN {cand.name} total={cand.total_time_s:.0f}s ...', flush=True)
        payload, status, path = run_candidate_payload(mod, cand, 'markov42_noisy', args.noise_scale, args.force_rerun)
        row = {
            'candidate_name': cand.name,
            'rationale': cand.rationale,
            'total_time_s': cand.total_time_s,
            'metrics': compact_result(payload),
            'continuity_checks': cand.continuity_checks,
            'run_json': str(path),
            'status': status,
            'delta_vs_faithful': delta_vs_ref(refs['faithful_markov'], payload),
            'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], payload),
        }
        rows.append(row)
        payload_by_name[cand.name] = payload
        print(
            f"DONE {cand.name}: mean={row['metrics']['overall']['mean_pct_error']:.3f}, "
            f"median={row['metrics']['overall']['median_pct_error']:.3f}, "
            f"max={row['metrics']['overall']['max_pct_error']:.3f}",
            flush=True,
        )

    rows.sort(key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))
    best_row = rows[0]
    best_candidate = candidate_by_name[best_row['candidate_name']]
    best_markov_payload = payload_by_name[best_candidate.name]
    best_kf_payload, best_kf_status, best_kf_path = run_candidate_payload(mod, best_candidate, 'kf36_noisy', args.noise_scale, args.force_rerun)

    best_summary = {
        'candidate_name': best_candidate.name,
        'rationale': best_candidate.rationale,
        'total_time_s': best_candidate.total_time_s,
        'all_rows': best_candidate.all_rows,
        'all_actions': best_candidate.all_actions,
        'all_faces': best_candidate.all_faces,
        'continuity_checks': best_candidate.continuity_checks,
        'markov42': compact_result(best_markov_payload),
        'markov42_run_json': best_row['run_json'],
        'kf36': compact_result(best_kf_payload),
        'kf36_run_json': str(best_kf_path),
        'kf36_status': best_kf_status,
        'delta_vs_faithful': delta_vs_ref(refs['faithful_markov'], best_markov_payload),
        'delta_vs_old_best': delta_vs_ref(refs['oldbest_markov'], best_markov_payload),
    }

    old_best_mean_gap = best_summary['delta_vs_old_best']['mean_pct_error']['improvement_pct_points']
    old_best_max_gap = best_summary['delta_vs_old_best']['max_pct_error']['improvement_pct_points']
    if old_best_mean_gap > 0 and old_best_max_gap > 0:
        scientific_conclusion = (
            f"The faithful-base closed-loop branch found {best_candidate.name} as a fully continuous local-insertion strategy that beats the old best legal result on both mean and max under Markov42."
        )
    else:
        scientific_conclusion = (
            f"The faithful-base closed-loop branch did not beat the old best legal result. The best valid candidate is {best_candidate.name}; it improves over faithful12 by {best_summary['delta_vs_faithful']['mean_pct_error']['improvement_pct_points']:+.3f} mean-points, but still trails the old best by {(-old_best_mean_gap):.3f} mean-points and {(-old_best_max_gap):.3f} max-points. This indicates that late closed-loop insertion can recover part of the original weak block, but the Ka2_y ceiling remains the blocker."
        )

    out_json = RESULTS_DIR / f'ch3_12pos_closedloop_local_insertions_{make_suffix(args.noise_scale)}.json'
    out_md = REPORTS_DIR / f'psins_ch3_12pos_closedloop_local_insertions_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_12pos_closedloop_local_insertions',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'hard_constraints': {
            'base_skeleton': 'faithful chapter-3 12-position original sequence',
            'base_node_signs_changed': False,
            'insertion_rule': 'closed-loop insertion-return motifs only',
            'continuity_check': ['beta_deg', 'outer_axis_body', 'full_orientation_matrix'],
            'time_budget_s': [1200.0, 1800.0],
            'seed': 42,
            'truth_family': 'shared low-noise benchmark',
        },
        'references': {
            'faithful12': {
                'candidate_name': faithful.name,
                'rows': faithful.rows,
                'action_sequence': faithful.action_sequence,
                'faces': faithful.faces,
                'markov42': compact_result(refs['faithful_markov']),
                'markov42_run_json': str(FAITHFUL_RESULT),
                'kf36': compact_result(refs['faithful_kf']),
                'kf36_run_json': str(FAITHFUL_KF_RESULT),
            },
            'old_best_legal': {
                'candidate_name': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
                'markov42': compact_result(refs['oldbest_markov']),
                'markov42_run_json': str(OLD_BEST_RESULT),
                'kf36': compact_result(refs['oldbest_kf']),
                'kf36_run_json': str(OLD_BEST_KF_RESULT),
            },
        },
        'candidate_specs': [
            {
                'name': spec['name'],
                'rationale': spec['rationale'],
                'insertion_anchors': sorted(spec['insertions'].keys()),
            }
            for spec in CANDIDATE_SPECS
        ],
        'markov42_rows': rows,
        'best_candidate': best_summary,
        'scientific_conclusion': scientific_conclusion,
        'files': {
            'json': str(out_json),
            'report': str(out_md),
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    out_md.write_text(render_report(payload), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps(payload['files'], ensure_ascii=False), flush=True)
    print('BEST_CLOSEDLOOP', best_candidate.name, best_summary['markov42']['overall'], flush=True)
    print('SCIENTIFIC_CONCLUSION', scientific_conclusion, flush=True)


if __name__ == '__main__':
    main()
