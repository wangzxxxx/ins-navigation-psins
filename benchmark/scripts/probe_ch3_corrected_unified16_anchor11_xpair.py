from __future__ import annotations

import argparse
import json
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
CURRENT_LEADER_SUMMARY = RESULTS_DIR / 'ch3_corrected_zcadence_handoff_refine_2026-04-03.json'

for p in [ROOT, ROOT / 'tmp_psins_py', METHOD_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module
from benchmark_ch3_12pos_goalA_repairs import compact_result, orientation_faces, rows_to_paras
from compare_ch3_12pos_path_baselines import build_dataset_with_path
from compare_four_methods_shared_noise import _load_json, _noise_matches, compute_payload, expected_noise_config
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate, make_suffix, render_action

NOISE_SCALE = 0.08
REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
ATT0_DEG = [0.0, 0.0, 0.0]
UNIFIED_ROT_S = 7.5
UNIFIED_PRE_S = 7.5
UNIFIED_POST_S = 60.0
UNIFIED_ROW_TOTAL_S = UNIFIED_ROT_S + UNIFIED_PRE_S + UNIFIED_POST_S
TARGET_TOTAL_S = 1200.0
CANDIDATE_NAME = 'anchor11_xpair_outerhold_unified16_75s'
METHOD_TAG = 'ch3corrected_anchor11_xpair_outerhold_unified16_75s'

FAITHFUL_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
FAITHFUL_KF = RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json'
FALLBACK_CURRENT_LEADER_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l11back0p5_l12y0p125_on_entry_shared_noise0p08_param_errors.json'
FALLBACK_CURRENT_LEADER_KF = RESULTS_DIR / 'KF36_ch3closedloop_relay_l11back0p5_l12y0p125_on_entry_shared_noise0p08_param_errors.json'


@dataclass
class InsertStep:
    kind: str
    angle_deg: int
    role: str
    label: str


class MechanismTracker:
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

    def apply(self, *, pos_id: int, kind: str, angle_deg: int, anchor_id: int, source_anchor_id: int, segment_role: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
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
            'source_anchor_id': source_anchor_id,
            'axis': axis,
            'angle_deg': float(angle_deg),
            'rotation_time_s': float(UNIFIED_ROT_S),
            'pre_static_s': float(UNIFIED_PRE_S),
            'post_static_s': float(UNIFIED_POST_S),
            'node_total_s': float(UNIFIED_ROW_TOTAL_S),
            'segment_role': segment_role,
            'label': label,
        }
        action = {
            'pos_id': pos_id,
            'anchor_id': anchor_id,
            'source_anchor_id': source_anchor_id,
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


def states_match(a: dict[str, Any], b: dict[str, Any], tol: float = 1e-9) -> bool:
    if int(a['beta_deg']) != int(b['beta_deg']):
        return False
    if list(a['outer_axis_body']) != list(b['outer_axis_body']):
        return False
    import numpy as np

    return np.allclose(np.asarray(a['C'], dtype=float), np.asarray(b['C'], dtype=float), atol=tol, rtol=0.0)


def preview_next_base_action(tracker: MechanismTracker, action_template: dict[str, Any], anchor_id: int) -> dict[str, Any]:
    preview = tracker.clone()
    row, action = preview.apply(
        pos_id=9999,
        kind=action_template['kind'],
        angle_deg=int(action_template['motor_angle_deg']),
        anchor_id=anchor_id,
        source_anchor_id=anchor_id,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def candidate_output_path(method_key: str, noise_scale: float) -> Path:
    suffix = make_suffix(noise_scale)
    prefix = 'M_markov_42state_gm1' if method_key == 'markov42_noisy' else 'KF36'
    return RESULTS_DIR / f'{prefix}_{METHOD_TAG}_shared_{suffix}_param_errors.json'


def delta_vs_reference(ref_payload: dict[str, Any], cand_payload: dict[str, Any]) -> dict[str, Any]:
    out = {}
    ref_compact = compact_result(ref_payload)
    cand_compact = compact_result(cand_payload)
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        rv = float(ref_compact['overall'][metric])
        cv = float(cand_compact['overall'][metric])
        out[metric] = {
            'reference': rv,
            'candidate': cv,
            'improvement_pct_points': rv - cv,
            'relative_improvement_pct': ((rv - cv) / rv * 100.0) if abs(rv) > 1e-12 else None,
        }
    return out


def load_reference_payloads(noise_scale: float) -> dict[str, Any]:
    expected_cfg = expected_noise_config(noise_scale)
    faithful_markov = _load_json(FAITHFUL_MARKOV)
    faithful_kf = _load_json(FAITHFUL_KF)
    for payload in [faithful_markov, faithful_kf]:
        if not _noise_matches(payload, expected_cfg):
            raise ValueError('faithful12 reference noise mismatch')

    if CURRENT_LEADER_SUMMARY.exists():
        summary = _load_json(CURRENT_LEADER_SUMMARY)
        leader = summary['references']['current_leader']
        leader_markov_path = Path(leader['files']['markov42'])
        leader_kf_path = Path(leader['files']['kf36'])
        leader_name = leader['label'].split('/')[-1].strip()
        leader_source = str(CURRENT_LEADER_SUMMARY)
    else:
        leader_markov_path = FALLBACK_CURRENT_LEADER_MARKOV
        leader_kf_path = FALLBACK_CURRENT_LEADER_KF
        leader_name = 'relay_l11back0p5_l12y0p125_on_entry'
        leader_source = 'fallback_current_leader_paths'

    current_leader_markov = _load_json(leader_markov_path)
    current_leader_kf = _load_json(leader_kf_path)
    for payload in [current_leader_markov, current_leader_kf]:
        if not _noise_matches(payload, expected_cfg):
            raise ValueError('current leader reference noise mismatch')

    return {
        'faithful12': {
            'candidate_name': 'ch3faithful12',
            'markov42': faithful_markov,
            'kf36': faithful_kf,
            'files': {
                'markov42': str(FAITHFUL_MARKOV),
                'kf36': str(FAITHFUL_KF),
            },
        },
        'current_corrected_best': {
            'candidate_name': leader_name,
            'source_summary': leader_source,
            'markov42': current_leader_markov,
            'kf36': current_leader_kf,
            'files': {
                'markov42': str(leader_markov_path),
                'kf36': str(leader_kf_path),
            },
        },
    }


def build_unified16_candidate(mod) -> dict[str, Any]:
    faithful = build_candidate(mod, ())
    base_actions = faithful.action_sequence
    tracker = MechanismTracker(mod)
    all_rows: list[dict[str, Any]] = []
    all_actions: list[dict[str, Any]] = []

    motif = [
        InsertStep(kind='inner', angle_deg=-90, role='motif_inner_open', label='anchor11_xpair_outerhold_inner_open'),
        InsertStep(kind='outer', angle_deg=+90, role='motif_outer_sweep', label='anchor11_xpair_outerhold_outer_sweep'),
        InsertStep(kind='outer', angle_deg=-90, role='motif_outer_return', label='anchor11_xpair_outerhold_outer_return'),
        InsertStep(kind='inner', angle_deg=+90, role='motif_inner_close', label='anchor11_xpair_outerhold_inner_close'),
    ]

    continuity_checks = []

    for anchor_idx, base_action in enumerate(base_actions, start=1):
        row, action = tracker.apply(
            pos_id=len(all_rows) + 1,
            kind=base_action['kind'],
            angle_deg=int(base_action['motor_angle_deg']),
            anchor_id=anchor_idx,
            source_anchor_id=anchor_idx,
            segment_role='anchor',
            label=f'anchor_{anchor_idx}',
        )
        all_rows.append(row)
        all_actions.append(action)

        if anchor_idx == 11:
            before_state = tracker.state_snapshot()
            for step in motif:
                row, action = tracker.apply(
                    pos_id=len(all_rows) + 1,
                    kind=step.kind,
                    angle_deg=step.angle_deg,
                    anchor_id=anchor_idx,
                    source_anchor_id=anchor_idx,
                    segment_role=step.role,
                    label=step.label,
                )
                all_rows.append(row)
                all_actions.append(action)
            after_state = tracker.state_snapshot()
            closure_ok = states_match(before_state, after_state)
            if not closure_ok:
                raise ValueError('anchor11 motif does not close to the same mechanism state before anchor12 resume')
            continuity_checks.append({
                'anchor_id': 11,
                'closure_rule': 'exact_same_mechanism_state_before_resume',
                'state_before_insertion': before_state,
                'state_after_insertion': after_state,
                'closure_ok': closure_ok,
                'next_base_action_preview': preview_next_base_action(tracker, base_actions[11], 12),
            })

    if len(all_rows) != 16:
        raise ValueError(f'expected 16 rows, got {len(all_rows)}')

    total_time_s = float(sum(row['node_total_s'] for row in all_rows))
    if abs(total_time_s - TARGET_TOTAL_S) > 1e-9:
        raise ValueError(f'total time mismatch: got {total_time_s}, want {TARGET_TOTAL_S}')

    paras = rows_to_paras(mod, all_rows)
    faces = orientation_faces(mod, paras)

    return {
        'candidate_name': CANDIDATE_NAME,
        'method_tag': METHOD_TAG,
        'rows': all_rows,
        'actions': all_actions,
        'faces': faces,
        'continuity_checks': continuity_checks,
        'total_time_s': total_time_s,
        'row_timing_s': {
            'rotation_time_s': UNIFIED_ROT_S,
            'pre_static_s': UNIFIED_PRE_S,
            'post_static_s': UNIFIED_POST_S,
            'row_total_s': UNIFIED_ROW_TOTAL_S,
        },
        'n_rows': len(all_rows),
    }


def run_candidate_payload(mod, candidate: dict[str, Any], method_key: str, noise_scale: float, force_rerun: bool = False):
    expected_cfg = expected_noise_config(noise_scale)
    out_path = candidate_output_path(method_key, noise_scale)
    if out_path.exists() and (not force_rerun):
        payload = _load_json(out_path)
        extra = payload.get('extra', {})
        if (
            _noise_matches(payload, expected_cfg)
            and extra.get('candidate_name') == candidate['candidate_name']
            and int(extra.get('n_rows', -1)) == 16
            and abs(float(extra.get('time_total_s', -1.0)) - TARGET_TOTAL_S) < 1e-9
        ):
            return payload, 'reused_verified', out_path

    paras = rows_to_paras(mod, candidate['rows'])
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
            label=f'{candidate["candidate_name"]}_{method_key}_{make_suffix(noise_scale)}',
        )
    elif method_key == 'kf36_noisy':
        clbt, _, _, _, _ = mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=36,
            label=f'{candidate["candidate_name"]}_{method_key}_{make_suffix(noise_scale)}',
        )
    else:
        raise KeyError(method_key)

    payload = compute_payload(
        mod,
        clbt,
        params,
        variant=f'{candidate["candidate_name"]}_{method_key}_{make_suffix(noise_scale)}',
        method_file='probe_ch3_corrected_unified16_anchor11_xpair.py',
        extra={
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'comparison_mode': 'corrected_att0_unified16_anchor11_xpair_probe',
            'candidate_name': candidate['candidate_name'],
            'method_key': method_key,
            'att0_deg': ATT0_DEG,
            'legality': 'true_dual_axis_motor_sequence_with_anchor11_closed_loop_insertion',
            'time_total_s': candidate['total_time_s'],
            'n_rows': candidate['n_rows'],
            'row_timing_s': candidate['row_timing_s'],
            'insertion_anchor_id': 11,
            'motif': 'inner_open_outer_sweep_outer_return_inner_close',
        },
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', out_path


def triplet_text(compact: dict[str, Any]) -> str:
    o = compact['overall']
    return f"{o['mean_pct_error']:.3f} / {o['median_pct_error']:.3f} / {o['max_pct_error']:.3f}"


def render_report(summary: dict[str, Any]) -> str:
    cand = summary['candidate']
    refs = summary['references']
    lines = []
    lines.append('# Chapter-3 corrected 16-step unified-timing anchor11 xpair probe')
    lines.append('')
    lines.append('## 1. What was run')
    lines.append('')
    lines.append('- Base scaffold: original corrected chapter-3 faithful 12-step motor sequence.')
    lines.append('- Local fusion rule: insert the 4-step anchor11 xpair outerhold motif **after original anchor11 and before original anchor12**.')
    lines.append('- Inserted motif order: **inner open → outer sweep → outer return → inner close**.')
    lines.append('- Hard basis enforced: **att0 = (0, 0, 0)**, same shared low-noise setup (`noise_scale=0.08`, shared seed/truth family), real dual-axis legality only, exact continuity-safe resume into anchor12.')
    lines.append(f"- Timing rule used exactly as requested: **16 rows × {UNIFIED_ROW_TOTAL_S:.1f} s = {TARGET_TOTAL_S:.0f} s**, with per-row **rot/pre/post = {UNIFIED_ROT_S:.1f} / {UNIFIED_PRE_S:.1f} / {UNIFIED_POST_S:.1f} s**.")
    lines.append('')
    lines.append('## 2. Headline result')
    lines.append('')
    lines.append(f"- Probe candidate `{cand['candidate_name']}` Markov42: **{triplet_text(cand['markov42'])}**")
    lines.append(f"- Probe candidate `{cand['candidate_name']}` KF36: **{triplet_text(cand['kf36'])}**")
    lines.append(f"- Corrected faithful12 reference Markov42: **{triplet_text(refs['faithful12']['markov42'])}**")
    lines.append(f"- Current corrected best-path reference Markov42 (`{refs['current_corrected_best']['candidate_name']}`): **{triplet_text(refs['current_corrected_best']['markov42'])}**")
    lines.append('')
    lines.append('## 3. Direct comparison')
    lines.append('')
    lines.append('| comparison | method | reference triplet | candidate triplet | Δmean | Δmedian | Δmax |')
    lines.append('|---|---|---|---|---:|---:|---:|')
    for ref_key, ref_label in [('faithful12', 'vs corrected faithful12'), ('current_corrected_best', 'vs current corrected best')]:
        for method_key, method_label in [('markov42', 'Markov42'), ('kf36', 'KF36')]:
            delta = cand[f'delta_vs_{ref_key}'][method_key]
            lines.append(
                f"| {ref_label} | {method_label} | {triplet_text(refs[ref_key][method_key])} | {triplet_text(cand[method_key])} | "
                f"{delta['mean_pct_error']['improvement_pct_points']:+.3f} | {delta['median_pct_error']['improvement_pct_points']:+.3f} | {delta['max_pct_error']['improvement_pct_points']:+.3f} |"
            )
    lines.append('')
    lines.append('## 4. Continuity / legality check')
    lines.append('')
    for check in cand['continuity_checks']:
        before = check['state_before_insertion']
        after = check['state_after_insertion']
        preview = check['next_base_action_preview']
        lines.append(f"- anchor {check['anchor_id']} closed-loop insertion returns to the exact same mechanism state before anchor12 resume: **{'yes' if check['closure_ok'] else 'no'}**")
        lines.append(f"  - before insertion: beta={before['beta_deg']}°, outer_axis={before['outer_axis_body']}, face={before['face_name']}")
        lines.append(f"  - after insertion : beta={after['beta_deg']}°, outer_axis={after['outer_axis_body']}, face={after['face_name']}")
        lines.append(
            f"  - anchor12 resume preview: `{preview['kind']}` {preview['motor_angle_deg']:+d}° on axis {preview['effective_body_axis']} (legal continuation preserved)"
        )
    lines.append('')
    lines.append('## 5. Exact 16-row legal motor / timing table')
    lines.append('')
    lines.append('| seq | source_anchor | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for row, action, face in zip(cand['rows'], cand['actions'], cand['faces']):
        lines.append(
            f"| {row['pos_id']} | {row['source_anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 6. Bottom line')
    lines.append('')
    lines.append(f"- Against corrected faithful12, the unified 16-step probe improves Markov42 by Δmean **{cand['delta_vs_faithful12']['markov42']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{cand['delta_vs_faithful12']['markov42']['max_pct_error']['improvement_pct_points']:+.3f}**.")
    lines.append(f"- Against the current corrected best path `{refs['current_corrected_best']['candidate_name']}`, it changes Markov42 by Δmean **{cand['delta_vs_current_corrected_best']['markov42']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{cand['delta_vs_current_corrected_best']['markov42']['max_pct_error']['improvement_pct_points']:+.3f}**.")
    lines.append(f"- Conclusion: {summary['bottom_line']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module('probe_ch3_corrected_unified16_anchor11_xpair_src', str(SOURCE_FILE))
    refs_raw = load_reference_payloads(args.noise_scale)
    candidate = build_unified16_candidate(mod)

    markov_payload, markov_status, markov_path = run_candidate_payload(mod, candidate, 'markov42_noisy', args.noise_scale, args.force_rerun)
    kf_payload, kf_status, kf_path = run_candidate_payload(mod, candidate, 'kf36_noisy', args.noise_scale, args.force_rerun)

    candidate_summary = {
        'candidate_name': candidate['candidate_name'],
        'method_tag': candidate['method_tag'],
        'n_rows': candidate['n_rows'],
        'total_time_s': candidate['total_time_s'],
        'row_timing_s': candidate['row_timing_s'],
        'rows': candidate['rows'],
        'actions': candidate['actions'],
        'faces': candidate['faces'],
        'continuity_checks': candidate['continuity_checks'],
        'markov42': compact_result(markov_payload),
        'markov42_run_json': str(markov_path),
        'markov42_status': markov_status,
        'kf36': compact_result(kf_payload),
        'kf36_run_json': str(kf_path),
        'kf36_status': kf_status,
    }

    references = {
        'faithful12': {
            'candidate_name': refs_raw['faithful12']['candidate_name'],
            'markov42': compact_result(refs_raw['faithful12']['markov42']),
            'kf36': compact_result(refs_raw['faithful12']['kf36']),
            'files': refs_raw['faithful12']['files'],
        },
        'current_corrected_best': {
            'candidate_name': refs_raw['current_corrected_best']['candidate_name'],
            'source_summary': refs_raw['current_corrected_best']['source_summary'],
            'markov42': compact_result(refs_raw['current_corrected_best']['markov42']),
            'kf36': compact_result(refs_raw['current_corrected_best']['kf36']),
            'files': refs_raw['current_corrected_best']['files'],
        },
    }

    candidate_summary['delta_vs_faithful12'] = {
        'markov42': delta_vs_reference(refs_raw['faithful12']['markov42'], markov_payload),
        'kf36': delta_vs_reference(refs_raw['faithful12']['kf36'], kf_payload),
    }
    candidate_summary['delta_vs_current_corrected_best'] = {
        'markov42': delta_vs_reference(refs_raw['current_corrected_best']['markov42'], markov_payload),
        'kf36': delta_vs_reference(refs_raw['current_corrected_best']['kf36'], kf_payload),
    }

    better_than_faithful = (
        candidate_summary['markov42']['overall']['mean_pct_error'] < references['faithful12']['markov42']['overall']['mean_pct_error']
        and candidate_summary['markov42']['overall']['max_pct_error'] < references['faithful12']['markov42']['overall']['max_pct_error']
    )
    beats_current_best = (
        candidate_summary['markov42']['overall']['mean_pct_error'] < references['current_corrected_best']['markov42']['overall']['mean_pct_error']
        and candidate_summary['markov42']['overall']['max_pct_error'] < references['current_corrected_best']['markov42']['overall']['max_pct_error']
    )

    if beats_current_best:
        bottom_line = (
            'The one-off unified 16-step probe is a genuine corrected-basis frontier improvement: it beats the current corrected best path on both Markov42 mean and max.'
        )
    elif better_than_faithful:
        bottom_line = (
            'The one-off unified 16-step probe is scientifically real but sub-frontier: it improves over corrected faithful12, yet does not beat the current corrected best path on the combined Markov42 frontier.'
        )
    else:
        bottom_line = (
            'The one-off unified 16-step probe does not clear even corrected faithful12 on the combined Markov42 mean/max gate, so the forced equal-time fusion degrades the late-block benefit under the corrected basis.'
        )

    summary = {
        'task': 'chapter-3 corrected one-off unified16 anchor11 xpair probe',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'corrected_att0_deg': ATT0_DEG,
        'hard_constraints': {
            'att0_deg': ATT0_DEG,
            'real_dual_axis_legality_only': True,
            'continuity_safe_execution': True,
            'total_time_s': TARGET_TOTAL_S,
            'same_low_noise_shared_setup': True,
            'uniform_row_timing': candidate['row_timing_s'],
            'fusion_rule': 'original faithful12 + anchor11 xpair outerhold inserted after original anchor11 before original anchor12',
        },
        'references': references,
        'candidate': candidate_summary,
        'bottom_line': bottom_line,
        'comparative_flags': {
            'beats_corrected_faithful12_on_markov42_mean_and_max': better_than_faithful,
            'beats_current_corrected_best_on_markov42_mean_and_max': beats_current_best,
        },
    }

    summary_path = RESULTS_DIR / f'ch3_corrected_unified16_anchor11_xpair_probe_{args.report_date}_summary.json'
    report_path = REPORTS_DIR / f'psins_ch3_corrected_unified16_anchor11_xpair_probe_{args.report_date}.md'
    summary['files'] = {
        'summary_json': str(summary_path),
        'report_md': str(report_path),
        'markov42_run_json': str(markov_path),
        'kf36_run_json': str(kf_path),
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_path.write_text(render_report(summary), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps(summary['files'], ensure_ascii=False), flush=True)
    print('CANDIDATE_MARKOV42', triplet_text(candidate_summary['markov42']), flush=True)
    print('CANDIDATE_KF36', triplet_text(candidate_summary['kf36']), flush=True)
    print('BOTTOM_LINE', bottom_line, flush=True)


if __name__ == '__main__':
    main()
