from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import types
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
TMP_PSINS_DIR = ROOT / 'tmp_psins_py'
SOURCE_FILE = TMP_PSINS_DIR / 'psins_py' / 'test_calibration_markov_pruned.py'
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'

COMPARE_CH3_FILE = SCRIPTS_DIR / 'compare_ch3_corrected_symmetric20_vs_legacy19pos_1200s.py'
G3_BASELINE_JSON = RESULTS_DIR / 'M_markov_42state_gm1_ch3corrected_symmetric20_att0zero_1200s_shared_noise0p12_param_errors.json'
CURRENT_BEST_JSON = RESULTS_DIR / 'G4_sym20_native_scd_round61_stageA_sel_once_b0_a099990_td0p8_shared_noise0p12_param_errors.json'

COMPARISON_MODE = 'sym20_native_scd_ultrafine_local_refine_2026_04_05'

STAGE_C1_ALPHAS = [0.99992, 0.99991, 0.999905, 0.99990, 0.999895, 0.99989, 0.99988]
STAGE_C1_TDS = [0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.00]

for p in [ROOT, TMP_PSINS_DIR, METHOD_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module
from compare_four_methods_shared_noise import (
    _build_neutral_scd_candidate,
    _load_json,
    _noise_matches,
    compute_payload,
    make_suffix,
)
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs
from probe_round55_newline import _build_patched_method


METRIC_KEYS = ['mean_pct_error', 'median_pct_error', 'max_pct_error']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=0.12)
    parser.add_argument('--report-date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def _sanitize_name(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]+', '_', text)


def _alpha_tag(alpha: float) -> str:
    return f"a{int(round(alpha * 100000)):06d}"


def _td_tag(td: float) -> str:
    if abs(td - round(td)) < 1e-12:
        return f"td{int(round(td))}"
    return f"td{str(td).replace('.', 'p')}"


def _overall_triplet(payload: dict[str, Any]) -> dict[str, float]:
    ov = payload['overall']
    return {
        'mean_pct_error': float(ov['mean_pct_error']),
        'median_pct_error': float(ov['median_pct_error']),
        'max_pct_error': float(ov['max_pct_error']),
    }


def _triplet_text(overall: dict[str, float]) -> str:
    return f"{overall['mean_pct_error']:.6f} / {overall['median_pct_error']:.6f} / {overall['max_pct_error']:.6f}"


def _candidate_signature(candidate: dict[str, Any]) -> str:
    keep = {
        'name': candidate['name'],
        'stage': candidate['stage'],
        'stage_index': candidate.get('stage_index'),
        'parent': candidate.get('parent'),
        'scd': candidate['scd'],
        'subset': candidate.get('subset'),
    }
    return json.dumps(keep, ensure_ascii=False, sort_keys=True)


def _resolve_target_indices(method_mod, scd_cfg: dict[str, Any]) -> list[int]:
    if scd_cfg.get('target_state_indices') is not None:
        return [int(x) for x in scd_cfg['target_state_indices']]

    target_name = scd_cfg.get('target')
    if target_name == 'selected':
        return [int(idx) for idx in method_mod.SELECTED_SCALE_STATES]
    if target_name == 'scale_block':
        return list(range(12, 27))
    raise KeyError(f'Unknown SCD target: {target_name}')


def _resolve_target_labels(method_mod, target_indices: list[int]) -> dict[str, str]:
    labels = {}
    selected_labels = {
        int(k): str(v)
        for k, v in getattr(method_mod, 'SELECTED_STATE_LABELS', {}).items()
    }
    for idx in target_indices:
        labels[str(int(idx))] = selected_labels.get(int(idx), f'state_{int(idx)}')
    return labels


def _apply_hybrid_scd(kf, scd_cfg: dict[str, Any], target_indices: list[int]) -> None:
    alpha = float(scd_cfg['alpha'])
    P = kf['Pxk']

    P[0:6, target_indices] *= alpha
    P[target_indices, 0:6] *= alpha

    if scd_cfg.get('bias_to_target', True):
        P[6:12, target_indices] *= alpha
        P[target_indices, 6:12] *= alpha


def _run_internalized_hybrid_scd(
    method_mod,
    mod,
    imu1,
    pos0,
    ts,
    bi_g,
    bi_a,
    tau_g,
    tau_a,
    label,
    scd_cfg,
):
    eth = mod.Earth(pos0)
    wnie = mod.glv.wie * mod.np.array([0, mod.math.cos(pos0[0]), mod.math.sin(pos0[0])])
    gn = mod.np.array([0, 0, -eth.g])
    Cba = mod.np.eye(3)
    nn, _, nts, _ = mod.nnts(2, ts)
    frq2 = int(1 / ts / 2) - 1

    k = frq2
    for k in range(frq2, min(5 * 60 * 2 * frq2, len(imu1)), 2 * frq2):
        ww = mod.np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
        if mod.np.linalg.norm(ww) / ts > 20 * mod.glv.dph:
            break
    kstatic = k - 3 * frq2

    clbt = {
        'Kg': mod.np.eye(3),
        'Ka': mod.np.eye(3),
        'Ka2': mod.np.zeros(3),
        'eb': mod.np.zeros(3),
        'db': mod.np.zeros(3),
        'rx': mod.np.zeros(3),
        'ry': mod.np.zeros(3),
    }

    length = len(imu1)
    dotwf = mod.imudot(imu1, 5.0)
    P_trace, X_trace, iter_bounds = [], [], []
    feedback_log = []
    schedule_log = []
    scd_log = []

    target_indices = _resolve_target_indices(method_mod, scd_cfg)
    target_labels = _resolve_target_labels(method_mod, target_indices)
    apply_policy_names = set(scd_cfg.get('apply_policy_names', []))

    def apply_clbt(imu_s, c):
        res = mod.np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    for it, policy in enumerate(method_mod.ITERATION_POLICIES):
        print(f'  [{label}] {policy["name"]} ({it+1}/{len(method_mod.ITERATION_POLICIES)})')
        kf = mod.clbtkfinit_42(nts, bi_g, tau_g, bi_a, tau_a)
        prior_diag = method_mod._configure_iteration_prior(mod, kf, policy)
        base_q = mod.np.array(kf['Qk'], dtype=float).copy()
        base_r = mod.np.array(kf['Rk'], dtype=float).copy()

        if policy.get('readout_only'):
            kf['Pxk'] = kf['Pxk'] * 100
            kf['Pxk'][:, 2] = 0
            kf['Pxk'][2, :] = 0
            kf['xk'] = mod.np.zeros(42)

        imu_align = apply_clbt(imu1[frq2:kstatic, :], clbt)
        _, _, _, qnb = mod.alignsb(imu_align, pos0)
        vn = mod.np.zeros(3)
        t1s = 0.0
        n_static_meas = 0
        n_dynamic_sched = 0
        n_late_sched = 0
        n_scd = 0
        n_transition_eligible = 0
        was_rotating = False
        time_since_rot_stop = 999.0
        scd_applied_this_phase = False
        scd_enabled_here = (policy['name'] in apply_policy_names) and (not policy.get('readout_only'))

        for k in range(2 * frq2, length - frq2, nn):
            k1 = k + nn - 1
            wm = imu1[k:k1+1, 0:3]
            vm = imu1[k:k1+1, 3:6]
            dwb = mod.np.mean(dotwf[k:k1+1, 0:3], axis=0)

            phim, dvbm = mod.cnscl(mod.np.hstack((wm, vm)))
            phim = clbt['Kg'] @ phim - clbt['eb'] * nts
            dvbm = clbt['Ka'] @ dvbm - clbt['db'] * nts
            wb = phim / nts
            fb = dvbm / nts

            SS = mod.imulvS(wb, dwb, Cba)
            fL = SS[:, 0:6] @ mod.np.concatenate((clbt['rx'], clbt['ry']))
            fn = mod.qmulv(qnb, fb - clbt['Ka2'] * (fb**2) - fL)
            vn = vn + (mod.rotv(-wnie * nts / 2, fn) + gn) * nts
            qnb = mod.qupdt2(qnb, phim, wnie * nts)

            t1s += nts
            Ft = mod.getFt_42(fb, wb, mod.q2mat(qnb), wnie, SS, tau_g, tau_a)
            kf['Phikk_1'] = mod.np.eye(42) + Ft * nts
            kf = mod.kfupdate(kf, TimeMeasBoth='T')

            if t1s > (0.2 - ts / 2):
                t1s = 0.0
                ww = mod.np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
                is_static = bool(mod.np.linalg.norm(ww) / ts < 20 * mod.glv.dph)
                progress = float(k) / float(length)
                method_mod._set_cov_schedule(kf, base_q, base_r, policy, progress, is_static)

                if not is_static:
                    was_rotating = True
                    time_since_rot_stop = 0.0
                    scd_applied_this_phase = False
                else:
                    if was_rotating:
                        was_rotating = False
                        time_since_rot_stop = 0.0
                    else:
                        time_since_rot_stop += 0.2

                if is_static:
                    n_static_meas += 1
                    kf = mod.kfupdate(kf, yk=vn, TimeMeasBoth='M')

                    if scd_enabled_here and time_since_rot_stop >= float(scd_cfg['transition_duration']):
                        n_transition_eligible += 1
                        if scd_cfg['mode'] == 'once_per_phase':
                            if not scd_applied_this_phase:
                                _apply_hybrid_scd(kf, scd_cfg, target_indices)
                                scd_applied_this_phase = True
                                n_scd += 1
                        elif scd_cfg['mode'] == 'repeat_after_transition':
                            _apply_hybrid_scd(kf, scd_cfg, target_indices)
                            scd_applied_this_phase = True
                            n_scd += 1
                        else:
                            raise KeyError(f"Unknown SCD mode: {scd_cfg['mode']}")
                else:
                    n_dynamic_sched += 1

                if progress >= policy.get('late_release_frac', 2.0):
                    n_late_sched += 1
                P_trace.append(mod.np.diag(kf['Pxk']))
                X_trace.append(mod.np.copy(kf['xk']))

        if not policy.get('readout_only'):
            clbt, meta = method_mod._apply_trust_internalized_feedback(mod, clbt, kf, prior_diag, policy)
            feedback_log.append(meta)

        schedule_log.append({
            'policy_name': policy['name'],
            'n_static_meas': int(n_static_meas),
            'n_dynamic_sched': int(n_dynamic_sched),
            'n_late_sched': int(n_late_sched),
            'late_release_frac': float(policy.get('late_release_frac', 1.0)),
        })
        scd_log.append({
            'policy_name': policy['name'],
            'enabled': bool(scd_enabled_here),
            'mode': scd_cfg['mode'],
            'alpha': float(scd_cfg['alpha']),
            'transition_duration': float(scd_cfg['transition_duration']),
            'target': scd_cfg.get('target'),
            'target_indices': [int(x) for x in target_indices],
            'target_labels': copy.deepcopy(target_labels),
            'bias_to_target': bool(scd_cfg.get('bias_to_target', True)),
            'n_transition_eligible': int(n_transition_eligible),
            'n_scd': int(n_scd),
        })
        iter_bounds.append(len(P_trace))

    return clbt, kf, mod.np.array(P_trace), mod.np.array(X_trace), {
        'iter_bounds': iter_bounds,
        'selected_state_labels': method_mod.SELECTED_STATE_LABELS,
        'iteration_policies': method_mod.ITERATION_POLICIES,
        'feedback_log': feedback_log,
        'schedule_log': schedule_log,
        'scd_log': scd_log,
    }


def _result_json_path(candidate_name: str, noise_scale: float) -> Path:
    suffix = make_suffix(noise_scale)
    return RESULTS_DIR / f'G4_sym20_native_scd_ultrafine_{candidate_name}_shared_{suffix}_param_errors.json'


def _build_stage_c1_candidates() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    idx = 0
    for alpha in STAGE_C1_ALPHAS:
        for td in STAGE_C1_TDS:
            idx += 1
            name = f"c1_sel_once_b0_{_alpha_tag(alpha)}_{_td_tag(td)}"
            out.append({
                'name': name,
                'description': (
                    'Stage C1 ultra-fine local refinement around current best: '
                    f'alpha={alpha}, transition_duration={td}, '
                    "target=selected, mode=once_per_phase, bias_to_target=False, apply_policy_names=['iter2_commit']"
                ),
                'stage': 'C1',
                'stage_index': idx,
                'parent': None,
                'scd': {
                    'mode': 'once_per_phase',
                    'alpha': float(alpha),
                    'transition_duration': float(td),
                    'target': 'selected',
                    'bias_to_target': False,
                    'apply_policy_names': ['iter2_commit'],
                },
                'subset': {
                    'type': 'full_selected',
                    'variant': 'full_selected',
                    'notes': 'No subset shrink in Stage C1.',
                },
            })
    return out


def _selected_pct_errors_from_payload(payload: dict[str, Any], selected_labels: dict[int, str]) -> dict[int, float]:
    out: dict[int, float] = {}
    param_errors = payload.get('param_errors', {}) if isinstance(payload, dict) else {}
    for idx, label in selected_labels.items():
        try:
            out[idx] = float(param_errors[label]['pct_error'])
        except Exception:
            out[idx] = float('inf')
    return out


def _normalize_subset_order(subset: list[int], full_order: list[int]) -> list[int]:
    s = {int(x) for x in subset}
    return [int(x) for x in full_order if int(x) in s]


def _build_stage_c2_candidates(
    stage_c1_top3: list[dict[str, Any]],
    method_mod,
) -> list[dict[str, Any]]:
    selected_full = [int(x) for x in method_mod.SELECTED_SCALE_STATES]
    selected_labels = {int(k): str(v) for k, v in method_mod.SELECTED_STATE_LABELS.items()}

    diag_like = [idx for idx, label in selected_labels.items() if label in {'dKg_xx', 'dKg_yy', 'dKg_zz'}]
    diag_like = _normalize_subset_order(diag_like, selected_full)

    out: list[dict[str, Any]] = []
    idx = 0

    for parent_row in stage_c1_top3:
        parent_candidate = parent_row['candidate']
        parent_payload = _load_json(parent_row['result_json'])
        selected_pct = _selected_pct_errors_from_payload(parent_payload, selected_labels)

        weakest_idx = sorted(selected_pct.items(), key=lambda kv: (float(kv[1]), kv[0]))[0][0]
        drop_low1 = [x for x in selected_full if x != weakest_idx]

        top3_focus = [
            kv[0]
            for kv in sorted(selected_pct.items(), key=lambda kv: (-float(kv[1]), kv[0]))[:3]
        ]
        top3_focus = _normalize_subset_order(top3_focus, selected_full)

        variant_specs = [
            {
                'variant': 'drop_low1',
                'subset_indices': drop_low1,
                'rationale': (
                    'Conservative shrink: remove the currently lowest-error selected state '
                    'to reduce potential over-coupling on already-stable terms.'
                ),
                'heuristic': {
                    'kind': 'remove_lowest_selected_pct_error',
                    'removed_idx': int(weakest_idx),
                    'removed_label': selected_labels.get(int(weakest_idx), f'state_{int(weakest_idx)}'),
                    'selected_pct_errors': {str(k): float(v) for k, v in selected_pct.items()},
                },
            },
            {
                'variant': 'diag_xx_yy_zz',
                'subset_indices': diag_like,
                'rationale': (
                    'Axis-oriented conservative subset: keep the natural selected diagonal '
                    'gyro block (xx/yy/zz) only.'
                ),
                'heuristic': {
                    'kind': 'axis_oriented_diag_subset',
                    'selected_pct_errors': {str(k): float(v) for k, v in selected_pct.items()},
                },
            },
            {
                'variant': 'focus_top3',
                'subset_indices': top3_focus,
                'rationale': (
                    'Ultra-conservative shrink preserving the strongest-error selected terms '
                    '(top-3 selected-state pct errors from parent result).'
                ),
                'heuristic': {
                    'kind': 'keep_top3_selected_pct_error',
                    'selected_pct_errors': {str(k): float(v) for k, v in selected_pct.items()},
                },
            },
        ]

        for spec in variant_specs:
            idx += 1
            subset_indices = [int(x) for x in spec['subset_indices']]
            subset_labels = {
                str(i): selected_labels.get(int(i), f'state_{int(i)}')
                for i in subset_indices
            }
            name = f"c2_{_sanitize_name(parent_candidate['name'])}_{spec['variant']}"
            out.append({
                'name': name,
                'description': (
                    f"Stage C2 tiny subset shrink on {parent_candidate['name']}: "
                    f"variant={spec['variant']}, subset_indices={subset_indices}"
                ),
                'stage': 'C2',
                'stage_index': idx,
                'parent': parent_candidate['name'],
                'scd': {
                    **copy.deepcopy(parent_candidate['scd']),
                    'target_state_indices': subset_indices,
                },
                'subset': {
                    'type': 'selected_subset_shrink',
                    'variant': spec['variant'],
                    'subset_indices': subset_indices,
                    'subset_labels': subset_labels,
                    'full_selected_indices': selected_full,
                    'full_selected_labels': {str(k): v for k, v in selected_labels.items()},
                    'parent_result_json': parent_row['result_json'],
                    'rationale': spec['rationale'],
                    'heuristic': spec['heuristic'],
                },
            })

    return out[:9]


def _run_one_candidate(
    *,
    idx: int,
    candidate: dict[str, Any],
    args: argparse.Namespace,
    source_mod,
    case: dict[str, Any],
    dataset: dict[str, Any],
    params,
) -> dict[str, Any]:
    result_path = _result_json_path(candidate['name'], args.noise_scale)
    expected_cfg = dataset['noise_config']
    signature = _candidate_signature(candidate)

    if (not args.force_rerun) and result_path.exists():
        old = _load_json(result_path)
        extra = old.get('extra', {}) if isinstance(old, dict) else {}
        if (
            _noise_matches(old, expected_cfg)
            and extra.get('comparison_mode') == COMPARISON_MODE
            and extra.get('path_case_tag') == case['case_tag']
            and extra.get('candidate_signature') == signature
        ):
            return {
                'candidate': copy.deepcopy(candidate),
                'overall': _overall_triplet(old),
                'result_json': str(result_path),
                'status': 'reused_verified',
            }

    method_mod = load_module(
        f"sym20_native_ultrafine_r53_{idx}_{_sanitize_name(candidate['name'])}",
        str(R53_METHOD_FILE),
    )

    neutral_candidate = _build_neutral_scd_candidate()
    neutral_candidate['name'] = candidate['name']
    neutral_candidate['description'] = candidate['description']
    neutral_candidate['scd'] = copy.deepcopy(candidate['scd'])
    method_mod = _build_patched_method(method_mod, neutral_candidate)

    scd_result = list(_run_internalized_hybrid_scd(
        method_mod,
        source_mod,
        dataset['imu_noisy'],
        dataset['pos0'],
        dataset['ts'],
        bi_g=dataset['bi_g'],
        bi_a=dataset['bi_a'],
        tau_g=dataset['tau_g'],
        tau_a=dataset['tau_a'],
        label=f"SYM20-ULTRAFINE-{idx:03d}-{_sanitize_name(candidate['name']).upper()}",
        scd_cfg=candidate['scd'],
    ))

    runtime = scd_result[4] if len(scd_result) >= 5 and isinstance(scd_result[4], dict) else {}
    target_indices = _resolve_target_indices(method_mod, candidate['scd'])
    target_labels = _resolve_target_labels(method_mod, target_indices)

    payload = compute_payload(
        source_mod,
        scd_result[0],
        params,
        variant=f"g4_sym20_native_scd_ultrafine_{candidate['name']}_{make_suffix(args.noise_scale)}",
        method_file='round53_base + _build_patched_method(neutral) + local _run_internalized_hybrid_scd',
        extra={
            'comparison_mode': COMPARISON_MODE,
            'path_case_key': case['case_key'],
            'path_case_tag': case['case_tag'],
            'path_case_display_name': case['display_name'],
            'att0_deg': case['att0_deg'],
            'n_motion_rows': case['n_motion_rows'],
            'claimed_position_count': case['claimed_position_count'],
            'total_time_s': case['total_time_s'],
            'timing_note': case['timing_note'],
            'source_builder': case['source_builder'],
            'source_reference': case['source_reference'],
            'builder_method_tag': case.get('builder_method_tag'),
            'rationale': candidate.get('description'),
            'noise_scale': args.noise_scale,
            'noise_config': dataset['noise_config'],
            'candidate_name': candidate['name'],
            'candidate_stage': candidate['stage'],
            'candidate_parent': candidate.get('parent'),
            'candidate_stage_index': candidate.get('stage_index'),
            'candidate_signature': signature,
            'scd_cfg': copy.deepcopy(candidate['scd']),
            'subset': copy.deepcopy(candidate.get('subset')),
            'target_indices_effective': [int(x) for x in target_indices],
            'target_labels_effective': copy.deepcopy(target_labels),
            'runtime_log': {
                'schedule_log': runtime.get('schedule_log'),
                'feedback_log': runtime.get('feedback_log'),
                'scd_log': runtime.get('scd_log'),
            },
            'baseline_g3_json': str(G3_BASELINE_JSON),
            'current_best_before_ultrafine_json': str(CURRENT_BEST_JSON),
        },
    )
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    return {
        'candidate': copy.deepcopy(candidate),
        'overall': _overall_triplet(payload),
        'result_json': str(result_path),
        'status': 'rerun',
    }


def _metric_deltas(candidate_overall: dict[str, float], reference_overall: dict[str, float]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in METRIC_KEYS:
        cand_v = float(candidate_overall[key])
        ref_v = float(reference_overall[key])
        improve = ref_v - cand_v
        metrics[key] = {
            'candidate_value': cand_v,
            'reference_value': ref_v,
            'improvement_pct_points': improve,
            'candidate_better': bool(cand_v < ref_v),
            'remaining_gap_pct_points': float(max(0.0, cand_v - ref_v)),
        }

    beats_mean = bool(metrics['mean_pct_error']['candidate_better'])
    beats_median = bool(metrics['median_pct_error']['candidate_better'])
    beats_max = bool(metrics['max_pct_error']['candidate_better'])
    return {
        'beats_mean': beats_mean,
        'beats_median': beats_median,
        'beats_max': beats_max,
        'beats_mean_median': bool(beats_mean and beats_median),
        'beats_all_three': bool(beats_mean and beats_median and beats_max),
        'metrics': metrics,
    }


def _rank_rows(rows: list[dict[str, Any]], g3_overall: dict[str, float]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    g3_max = float(g3_overall['max_pct_error'])

    for row in rows:
        r = copy.deepcopy(row)
        r['delta_vs_g3'] = _metric_deltas(r['overall'], g3_overall)

        gaps = {
            key: float(max(0.0, float(r['overall'][key]) - float(g3_overall[key])))
            for key in METRIC_KEYS
        }
        r['remaining_gaps_to_g3'] = gaps
        r['mean_median_gap_sum'] = float(gaps['mean_pct_error'] + gaps['median_pct_error'])
        r['max_loss_vs_g3'] = float(max(0.0, float(r['overall']['max_pct_error']) - g3_max))
        r['max_not_worse_than_g3'] = bool(float(r['overall']['max_pct_error']) <= g3_max)
        prepared.append(r)

    winners = [r for r in prepared if r['delta_vs_g3']['beats_mean_median']]
    others = [r for r in prepared if not r['delta_vs_g3']['beats_mean_median']]

    winners = sorted(
        winners,
        key=lambda x: (
            float(x['max_loss_vs_g3']),
            float(x['overall']['max_pct_error']),
            float(x['overall']['mean_pct_error']),
            float(x['overall']['median_pct_error']),
        ),
    )
    others = sorted(
        others,
        key=lambda x: (
            float(x['mean_median_gap_sum']),
            float(x['remaining_gaps_to_g3']['mean_pct_error']),
            float(x['remaining_gaps_to_g3']['median_pct_error']),
            float(x['max_loss_vs_g3']),
            float(x['overall']['max_pct_error']),
            float(x['overall']['mean_pct_error']),
            float(x['overall']['median_pct_error']),
        ),
    )

    ranked = winners + others
    for i, row in enumerate(ranked, start=1):
        row['rank'] = i
        row['ranking_group'] = 'beats_mean_median' if row['delta_vs_g3']['beats_mean_median'] else 'closest_mean_median_gap'
    return ranked


def _nearest_mm_gap_row(ranked: list[dict[str, Any]]) -> dict[str, Any]:
    return sorted(
        ranked,
        key=lambda r: (
            float(r['mean_median_gap_sum']),
            float(r['remaining_gaps_to_g3']['mean_pct_error']),
            float(r['remaining_gaps_to_g3']['median_pct_error']),
            float(r['max_loss_vs_g3']),
        ),
    )[0]


def _render_report(summary: dict[str, Any]) -> str:
    g3 = summary['baselines']['g3']['overall']
    current = summary['baselines']['current_best_before_ultrafine']['overall']
    best = summary['best_candidate']

    lines: list[str] = []
    lines.append('# Sym20-native SCD ultrafine local refinement (2026-04-05)')
    lines.append('')
    lines.append('## Setup')
    lines.append('')
    lines.append(f"- Path: `{summary['setup']['path_case_tag']}`")
    lines.append(f"- att0: `{summary['setup']['att0_deg']}`")
    lines.append(f"- noise_scale: `{summary['noise_scale']}`")
    lines.append('- Fixed SCD constraints: target=selected (or conservative selected subset in C2), mode=once_per_phase, bias_to_target=False, apply_policy_names=[iter2_commit]')
    lines.append('')
    lines.append('## Baselines (mean / median / max, % error)')
    lines.append('')
    lines.append(f"- G3: **{_triplet_text(g3)}**")
    lines.append(f"- current best before ultrafine: **{_triplet_text(current)}**")
    lines.append('')

    lines.append('## Stage C1 (ultra-fine hyper-refinement)')
    lines.append('')
    lines.append('| rankC1 | name | alpha | td | mean | median | max | beats(mean/median/max) | mm_gap_sum | max_loss_vs_g3 |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|---|---:|---:|')
    for row in summary['stageC1']['ranking']:
        c = row['candidate']
        b = row['delta_vs_g3']
        lines.append(
            f"| {row['rank']} | {c['name']} | {c['scd']['alpha']:.6f} | {c['scd']['transition_duration']:.2f} | "
            f"{row['overall']['mean_pct_error']:.6f} | {row['overall']['median_pct_error']:.6f} | {row['overall']['max_pct_error']:.6f} | "
            f"{int(b['beats_mean'])}/{int(b['beats_median'])}/{int(b['beats_max'])} | {row['mean_median_gap_sum']:.6f} | {row['max_loss_vs_g3']:.6f} |"
        )
    lines.append('')

    lines.append('## Stage C2 (tiny selected-subset shrink, <=9)')
    lines.append('')
    lines.append('| rankC2 | name | parent | variant | subset_indices | mean | median | max | beats(mean/median/max) | mm_gap_sum | max_loss_vs_g3 |')
    lines.append('|---:|---|---|---|---|---:|---:|---:|---|---:|---:|')
    for row in summary['stageC2']['ranking']:
        c = row['candidate']
        b = row['delta_vs_g3']
        subset = c.get('subset', {})
        subset_indices = subset.get('subset_indices', [])
        lines.append(
            f"| {row['rank']} | {c['name']} | {c.get('parent')} | {subset.get('variant')} | {subset_indices} | "
            f"{row['overall']['mean_pct_error']:.6f} | {row['overall']['median_pct_error']:.6f} | {row['overall']['max_pct_error']:.6f} | "
            f"{int(b['beats_mean'])}/{int(b['beats_median'])}/{int(b['beats_max'])} | {row['mean_median_gap_sum']:.6f} | {row['max_loss_vs_g3']:.6f} |"
        )
    lines.append('')

    lines.append('## Final decision')
    lines.append('')
    lines.append(f"- Best candidate: **{best['candidate']['name']}** ({best['candidate']['stage']})")
    lines.append(f"  - overall: **{_triplet_text(best['overall'])}**")
    lines.append(
        f"  - vs G3: Δmean={best['delta_vs_g3']['metrics']['mean_pct_error']['improvement_pct_points']:+.6f}, "
        f"Δmedian={best['delta_vs_g3']['metrics']['median_pct_error']['improvement_pct_points']:+.6f}, "
        f"Δmax={best['delta_vs_g3']['metrics']['max_pct_error']['improvement_pct_points']:+.6f}"
    )
    lines.append(
        f"- Beats G3? mean={summary['beats_g3']['mean']}, median={summary['beats_g3']['median']}, "
        f"max={summary['beats_g3']['max']}, mean+median={summary['beats_g3']['mean_median']}, all_three={summary['beats_g3']['all_three']}"
    )

    subset_help = summary['subset_shrink_assessment']
    lines.append(
        f"- Subset shrink helped median vs best C1? **{subset_help['helped_median_vs_best_c1']}** "
        f"(bestC1={subset_help['best_c1_median']:.6f}, bestC2={subset_help['best_c2_median']:.6f})"
    )
    lines.append(
        f"- Subset shrink helped mean vs best C1? **{subset_help['helped_mean_vs_best_c1']}** "
        f"(bestC1={subset_help['best_c1_mean']:.6f}, bestC2={subset_help['best_c2_mean']:.6f})"
    )

    if not summary['beats_g3']['mean_median']:
        gap = summary['nearest_remaining_gap_to_mean_median_win']
        lines.append(
            f"- Nearest remaining gap to mean+median win: `{gap['candidate_name']}` "
            f"needs mean_gap={gap['remaining_gap_pct_points']['mean_pct_error']:.6f}, "
            f"median_gap={gap['remaining_gap_pct_points']['median_pct_error']:.6f}; "
            f"max_loss={gap['max_loss_vs_g3']:.6f}."
        )
    lines.append('')

    lines.append('## Artifacts')
    lines.append('')
    lines.append(f"- script: `{summary['files']['script']}`")
    lines.append(f"- summary json: `{summary['files']['summary_json']}`")
    lines.append(f"- report md: `{summary['files']['report_md']}`")
    lines.append(f"- best result json: `{summary['files']['best_result_json']}`")
    lines.append('')

    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    compare_ch3_mod = load_module('sym20_ultrafine_compare_ch3_mod', str(COMPARE_CH3_FILE))
    source_mod = load_module(
        f"sym20_ultrafine_source_{make_suffix(args.noise_scale)}",
        str(SOURCE_FILE),
    )

    case = compare_ch3_mod.build_symmetric20_case(source_mod)
    dataset = compare_ch3_mod.build_dataset(source_mod, case['paras'], case['att0_deg'], args.noise_scale)
    params = _param_specs(source_mod)

    baseline_g3 = _load_json(G3_BASELINE_JSON)
    baseline_current = _load_json(CURRENT_BEST_JSON)
    g3_overall = _overall_triplet(baseline_g3)
    current_overall = _overall_triplet(baseline_current)

    inspect_method_mod = load_module('sym20_ultrafine_inspect_method', str(R53_METHOD_FILE))
    selected_indices_full = [int(x) for x in inspect_method_mod.SELECTED_SCALE_STATES]
    selected_labels_full = {
        str(k): str(v)
        for k, v in inspect_method_mod.SELECTED_STATE_LABELS.items()
    }

    # Stage C1
    stage_c1_candidates = _build_stage_c1_candidates()
    stage_c1_rows: list[dict[str, Any]] = []
    for i, cand in enumerate(stage_c1_candidates, start=1):
        stage_c1_rows.append(_run_one_candidate(
            idx=i,
            candidate=cand,
            args=args,
            source_mod=source_mod,
            case=case,
            dataset=dataset,
            params=params,
        ))

    stage_c1_ranked = _rank_rows(stage_c1_rows, g3_overall)
    stage_c1_top3 = [copy.deepcopy(r) for r in stage_c1_ranked[:3]]

    # Stage C2
    stage_c2_candidates = _build_stage_c2_candidates(stage_c1_top3, inspect_method_mod)
    stage_c2_rows: list[dict[str, Any]] = []
    for j, cand in enumerate(stage_c2_candidates, start=1):
        stage_c2_rows.append(_run_one_candidate(
            idx=1000 + j,
            candidate=cand,
            args=args,
            source_mod=source_mod,
            case=case,
            dataset=dataset,
            params=params,
        ))

    stage_c2_ranked = _rank_rows(stage_c2_rows, g3_overall)

    all_rows = stage_c1_rows + stage_c2_rows
    ranked = _rank_rows(all_rows, g3_overall)
    best = copy.deepcopy(ranked[0])

    beats_g3 = {
        'mean': any(r['delta_vs_g3']['beats_mean'] for r in ranked),
        'median': any(r['delta_vs_g3']['beats_median'] for r in ranked),
        'max': any(r['delta_vs_g3']['beats_max'] for r in ranked),
        'mean_median': any(r['delta_vs_g3']['beats_mean_median'] for r in ranked),
        'all_three': any(r['delta_vs_g3']['beats_all_three'] for r in ranked),
    }

    nearest_mm_gap = _nearest_mm_gap_row(ranked)
    nearest_remaining_gap_to_mean_median_win = {
        'candidate_name': nearest_mm_gap['candidate']['name'],
        'stage': nearest_mm_gap['candidate']['stage'],
        'rank': nearest_mm_gap['rank'],
        'remaining_gap_pct_points': copy.deepcopy(nearest_mm_gap['remaining_gaps_to_g3']),
        'mean_median_gap_sum': float(nearest_mm_gap['mean_median_gap_sum']),
        'max_loss_vs_g3': float(nearest_mm_gap['max_loss_vs_g3']),
        'delta_vs_g3': copy.deepcopy(nearest_mm_gap['delta_vs_g3']),
    }

    best_c1 = copy.deepcopy(stage_c1_ranked[0])
    best_c2 = copy.deepcopy(stage_c2_ranked[0]) if stage_c2_ranked else None

    subset_shrink_assessment = {
        'best_c1_candidate': best_c1['candidate']['name'],
        'best_c1_mean': float(best_c1['overall']['mean_pct_error']),
        'best_c1_median': float(best_c1['overall']['median_pct_error']),
        'best_c1_max': float(best_c1['overall']['max_pct_error']),
        'best_c2_candidate': best_c2['candidate']['name'] if best_c2 else None,
        'best_c2_mean': float(best_c2['overall']['mean_pct_error']) if best_c2 else float('nan'),
        'best_c2_median': float(best_c2['overall']['median_pct_error']) if best_c2 else float('nan'),
        'best_c2_max': float(best_c2['overall']['max_pct_error']) if best_c2 else float('nan'),
        'helped_median_vs_best_c1': bool(best_c2 and float(best_c2['overall']['median_pct_error']) < float(best_c1['overall']['median_pct_error'])),
        'helped_mean_vs_best_c1': bool(best_c2 and float(best_c2['overall']['mean_pct_error']) < float(best_c1['overall']['mean_pct_error'])),
        'helped_max_vs_best_c1': bool(best_c2 and float(best_c2['overall']['max_pct_error']) < float(best_c1['overall']['max_pct_error'])),
        'best_overall_from_subset_stage': bool(best['candidate']['stage'] == 'C2'),
    }

    if beats_g3['mean_median']:
        verdict = {
            'success': True,
            'label': 'reproduced',
            'reason': 'At least one candidate beats G3 on both mean and median under ultrafine local refinement.',
        }
    else:
        verdict = {
            'success': False,
            'label': 'not_reproduced_yet',
            'reason': (
                'No candidate beats G3 on mean+median simultaneously; nearest remaining '
                f"mean/median gap is {nearest_remaining_gap_to_mean_median_win['remaining_gap_pct_points']['mean_pct_error']:.6f} / "
                f"{nearest_remaining_gap_to_mean_median_win['remaining_gap_pct_points']['median_pct_error']:.6f}."
            ),
        }

    suffix = make_suffix(args.noise_scale)
    summary_json = RESULTS_DIR / f'g4_sym20_native_scd_ultrafine_{args.report_date}_{suffix}_summary.json'
    report_md = REPORTS_DIR / f'psins_sym20_native_scd_ultrafine_{args.report_date}_{suffix}.md'

    summary = {
        'experiment': 'sym20_native_scd_ultrafine_local_refine_2026_04_05',
        'comparison_mode': COMPARISON_MODE,
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'noise_config': dataset['noise_config'],
        'ranking_rule': {
            'primary_target': 'beat G3 on mean and median',
            'secondary_target': 'keep max <= G3 if possible; otherwise smallest max loss',
            'fallback': 'smallest mean+median remaining gap, then max loss',
        },
        'setup': {
            'path_case_key': case['case_key'],
            'path_case_tag': case['case_tag'],
            'path_case_display_name': case['display_name'],
            'att0_deg': case['att0_deg'],
            'n_motion_rows': case['n_motion_rows'],
            'claimed_position_count': case['claimed_position_count'],
            'total_time_s': case['total_time_s'],
            'timing_note': case['timing_note'],
            'source_builder': case['source_builder'],
            'source_reference': case['source_reference'],
            'builder_method_tag': case.get('builder_method_tag'),
            'selected_full_indices': selected_indices_full,
            'selected_full_labels': selected_labels_full,
            'fixed_scd_constraints': {
                'mode': 'once_per_phase',
                'bias_to_target': False,
                'apply_policy_names': ['iter2_commit'],
            },
        },
        'baselines': {
            'g3': {
                'json_path': str(G3_BASELINE_JSON),
                'overall': g3_overall,
            },
            'current_best_before_ultrafine': {
                'json_path': str(CURRENT_BEST_JSON),
                'overall': current_overall,
            },
        },
        'stageC1': {
            'count': len(stage_c1_rows),
            'grid': {
                'alpha': STAGE_C1_ALPHAS,
                'transition_duration': STAGE_C1_TDS,
                'target': ['selected'],
                'mode': ['once_per_phase'],
                'bias_to_target': [False],
                'apply_policy_names': [['iter2_commit']],
            },
            'top3_for_stageC2': [copy.deepcopy(r['candidate']) for r in stage_c1_top3],
            'ranking': stage_c1_ranked,
        },
        'stageC2': {
            'count': len(stage_c2_rows),
            'candidate_cap': 9,
            'parent_top3_from_stageC1': [copy.deepcopy(r['candidate']['name']) for r in stage_c1_top3],
            'variants_per_parent': ['drop_low1', 'diag_xx_yy_zz', 'focus_top3'],
            'ranking': stage_c2_ranked,
        },
        'total_tried_candidates': len(ranked),
        'ranking': ranked,
        'best_candidate': best,
        'beats_g3': beats_g3,
        'subset_shrink_assessment': subset_shrink_assessment,
        'nearest_remaining_gap_to_mean_median_win': nearest_remaining_gap_to_mean_median_win,
        'verdict': verdict,
        'code_fixes_needed': 'none',
        'files': {
            'script': str(Path(__file__)),
            'summary_json': str(summary_json),
            'report_md': str(report_md),
            'best_result_json': best['result_json'],
            'all_result_jsons': [r['result_json'] for r in ranked],
            'stageC1_result_jsons': [r['result_json'] for r in stage_c1_ranked],
            'stageC2_result_jsons': [r['result_json'] for r in stage_c2_ranked],
        },
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_md.write_text(_render_report(summary), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps({
        'summary_json': str(summary_json),
        'report_md': str(report_md),
        'best_candidate': {
            'name': best['candidate']['name'],
            'stage': best['candidate']['stage'],
            'overall': best['overall'],
            'delta_vs_g3': best['delta_vs_g3'],
            'result_json': best['result_json'],
        },
        'beats_g3': beats_g3,
        'subset_shrink_assessment': subset_shrink_assessment,
        'nearest_remaining_gap_to_mean_median_win': nearest_remaining_gap_to_mean_median_win,
        'verdict': verdict,
        'code_fixes_needed': summary['code_fixes_needed'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
