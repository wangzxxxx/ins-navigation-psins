from __future__ import annotations

import argparse
import copy
import json
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
METHOD_DIR = ROOT / 'psins_method_bench' / 'methods' / 'markov'
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
REPORTS_DIR = ROOT / 'reports'
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'
COMPARE_SCRIPT = SCRIPTS_DIR / 'compare_ch3_corrected_symmetric20_vs_legacy19pos_1200s.py'
BASELINE_G3_JSON = RESULTS_DIR / 'M_markov_42state_gm1_ch3corrected_symmetric20_att0zero_1200s_shared_noise0p12_param_errors.json'
PURE_SCD_JSON = RESULTS_DIR / 'G4_pure_scd_neutral_ch3corrected_symmetric20_att0zero_1200s_shared_noise0p12_param_errors.json'

COMPARISON_MODE = 'g4_sym20_pure_followup_on_corrected_symmetric20_att0zero_1200s'
DEFAULT_NOISE_SCALE = 0.12

for p in [ROOT, ROOT / 'tmp_psins_py', METHOD_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module
from compare_four_methods_shared_noise import _build_neutral_scd_candidate, _load_json, _noise_matches, compute_payload, make_suffix
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs
from probe_round55_newline import _build_patched_method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=DEFAULT_NOISE_SCALE)
    parser.add_argument('--report-date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def _safe_float(v: Any) -> float:
    return float(v)


def _delta_vs_ref(overall: dict[str, Any], ref: dict[str, Any]) -> dict[str, float]:
    return {
        k: _safe_float(overall[k]) - _safe_float(ref[k])
        for k in ['mean_pct_error', 'median_pct_error', 'max_pct_error']
    }


def _improve_vs_ref(overall: dict[str, Any], ref: dict[str, Any]) -> dict[str, float]:
    return {
        k: _safe_float(ref[k]) - _safe_float(overall[k])
        for k in ['mean_pct_error', 'median_pct_error', 'max_pct_error']
    }


def _beats_flags(overall: dict[str, Any], ref: dict[str, Any]) -> dict[str, bool]:
    mean_b = _safe_float(overall['mean_pct_error']) < _safe_float(ref['mean_pct_error'])
    med_b = _safe_float(overall['median_pct_error']) < _safe_float(ref['median_pct_error'])
    max_b = _safe_float(overall['max_pct_error']) < _safe_float(ref['max_pct_error'])
    return {
        'mean': mean_b,
        'median': med_b,
        'max': max_b,
        'all_three': bool(mean_b and med_b and max_b),
    }


def _triplet(overall: dict[str, Any]) -> str:
    return f"{overall['mean_pct_error']:.6f} / {overall['median_pct_error']:.6f} / {overall['max_pct_error']:.6f}"


def _candidate_output_path(candidate_name: str, noise_scale: float) -> Path:
    suffix = make_suffix(noise_scale)
    return RESULTS_DIR / f'G4_sym20_pure_followup_{candidate_name}_shared_{suffix}_param_errors.json'


def _resolve_target_indices(method_mod, target_name: str):
    if target_name == 'xxzz_pair':
        return [12, 20]
    if target_name == 'selected':
        return [int(idx) for idx in method_mod.SELECTED_SCALE_STATES]
    if target_name == 'scale_block':
        return list(range(12, 27))
    raise KeyError(f'Unknown SCD target: {target_name}')


def _apply_hybrid_scd(method_mod, kf, scd_cfg, target_indices):
    alpha = float(scd_cfg['alpha'])
    P = kf['Pxk']

    P[0:6, target_indices] *= alpha
    P[target_indices, 0:6] *= alpha

    if scd_cfg.get('bias_to_target', True):
        P[6:12, target_indices] *= alpha
        P[target_indices, 6:12] *= alpha


def _run_internalized_hybrid_scd(method_mod, mod, imu1, pos0, ts, bi_g, bi_a, tau_g, tau_a, label, scd_cfg):
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
        'Kg': mod.np.eye(3), 'Ka': mod.np.eye(3), 'Ka2': mod.np.zeros(3),
        'eb': mod.np.zeros(3), 'db': mod.np.zeros(3),
        'rx': mod.np.zeros(3), 'ry': mod.np.zeros(3),
    }

    length = len(imu1)
    dotwf = mod.imudot(imu1, 5.0)
    P_trace, X_trace, iter_bounds = [], [], []
    feedback_log = []
    schedule_log = []
    scd_log = []

    target_indices = _resolve_target_indices(method_mod, scd_cfg['target'])
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
                                _apply_hybrid_scd(method_mod, kf, scd_cfg, target_indices)
                                scd_applied_this_phase = True
                                n_scd += 1
                        elif scd_cfg['mode'] == 'repeat_after_transition':
                            _apply_hybrid_scd(method_mod, kf, scd_cfg, target_indices)
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
            'target': scd_cfg['target'],
            'target_indices': [int(x) for x in target_indices],
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


def _merge_candidate(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    merged['name'] = extra['name']
    merged['description'] = extra['description']
    merged['rationale'] = extra['rationale']

    merged['scd'] = copy.deepcopy(merged.get('scd', {}))
    merged['scd'].update(copy.deepcopy(extra.get('scd_patch', {})))

    merged_patches = copy.deepcopy(merged.get('iter_patches', {}))
    for iter_idx, patch in extra.get('iter_patches', {}).items():
        dst = merged_patches.setdefault(iter_idx, {})
        for key, value in patch.items():
            if isinstance(value, dict):
                current = copy.deepcopy(dst.get(key, {}))
                current.update(copy.deepcopy(value))
                dst[key] = current
            else:
                dst[key] = copy.deepcopy(value)
    merged['iter_patches'] = merged_patches

    for key in ['post_rx_y_mult', 'post_ry_z_mult']:
        if extra.get(key) is not None:
            merged[key] = float(extra[key])

    merged['search_patch'] = copy.deepcopy(extra)
    return merged


def _candidate_registry() -> list[dict[str, Any]]:
    base = _build_neutral_scd_candidate()
    cands = [
        {
            'name': 'neutral_scale_ref',
            'description': 'Reference pure neutral SCD baseline (same as focused pure-SCD eval).',
            'rationale': 'Anchor for all follow-up micro-retunes.',
            'reference_json': str(PURE_SCD_JSON),
        },
        {
            'name': 'neutral_scale_os098',
            'description': 'Keep pure scale-block SCD, add very mild iter2 other-scale damping.',
            'rationale': 'Target dKa_yy / dKa_yz / dKg_yy neighborhood without reopening selected-state trust surgery.',
            'iter_patches': {1: {'other_scale_alpha': 0.98}},
        },
        {
            'name': 'neutral_scale_os096',
            'description': 'Same as os098, slightly stronger iter2 other-scale damping.',
            'rationale': 'Stress-test whether the pure route is simply over-feeding the non-selected scale block.',
            'iter_patches': {1: {'other_scale_alpha': 0.96}},
        },
        {
            'name': 'neutral_scale_ka2095',
            'description': 'Keep pure scale-block SCD, only damp iter2 Ka2 feedback.',
            'rationale': 'Pure route is close on mean/max but loses Ka2_x/y/z; trimming Ka2 may recover mean cheaply.',
            'iter_patches': {1: {'ka2_alpha': 0.95}},
        },
        {
            'name': 'neutral_scale_os098_ka2095',
            'description': 'Pure scale-block SCD with mild iter2 other-scale + Ka2 damping.',
            'rationale': 'Joint fix for dKa_yy/dKa_yz neighborhood and Ka2 mean drag, while leaving selected states neutral.',
            'iter_patches': {1: {'other_scale_alpha': 0.98, 'ka2_alpha': 0.95}},
        },
        {
            'name': 'neutral_scale_os098_ka2095_mark098',
            'description': 'Add mild iter2 Markov-state damping on top of os098+ka2095.',
            'rationale': 'Probe whether db_y / eb_z central-bulk errors are slightly over-committed in the pure route.',
            'iter_patches': {1: {'other_scale_alpha': 0.98, 'ka2_alpha': 0.95, 'markov_alpha': 0.98}},
        },
        {
            'name': 'neutral_xxzz_a0992_os098_ka2095',
            'description': 'Switch SCD target to xx/zz only, use slightly milder alpha, retain mild iter2 other-scale + Ka2 damping.',
            'rationale': 'Try to keep pure route bulk shape while pushing only the xx/zz subpath that already showed good direction.',
            'scd_patch': {'target': 'xxzz_pair', 'alpha': 0.9992},
            'iter_patches': {1: {'other_scale_alpha': 0.98, 'ka2_alpha': 0.95}},
        },
        {
            'name': 'neutral_xxzz_a0990_os098_ka2095',
            'description': 'xx/zz-only SCD with slightly stronger alpha plus mild iter2 other-scale + Ka2 damping.',
            'rationale': 'Check whether a touch more xx/zz suppression improves mean/max without the full scale-block collateral movement.',
            'scd_patch': {'target': 'xxzz_pair', 'alpha': 0.9990},
            'iter_patches': {1: {'other_scale_alpha': 0.98, 'ka2_alpha': 0.95}},
        },
        {
            'name': 'neutral_scale_nav_a0999_os098_ka2095',
            'description': 'Keep full scale-block target but cut nav-only cross-cov, plus mild iter2 other-scale + Ka2 damping.',
            'rationale': 'Tests whether preserving bias↔target routes helps the central bulk while still benefiting from the pure route.',
            'scd_patch': {'target': 'scale_block', 'alpha': 0.9990, 'bias_to_target': False},
            'iter_patches': {1: {'other_scale_alpha': 0.98, 'ka2_alpha': 0.95}},
        },
        {
            'name': 'neutral_scale_td4_os098_ka2095',
            'description': 'Delay pure scale-block SCD to 4 s after transition, plus mild iter2 other-scale + Ka2 damping.',
            'rationale': 'A later one-shot cut may preserve median bulk while micro-damping still repairs mean/max tail terms.',
            'scd_patch': {'transition_duration': 4.0},
            'iter_patches': {1: {'other_scale_alpha': 0.98, 'ka2_alpha': 0.95}},
        },
        {
            'name': 'neutral_scale_a0992_os098_ka2095',
            'description': 'Keep scale-block target but weaken pure SCD slightly, plus mild iter2 other-scale + Ka2 damping.',
            'rationale': 'Checks whether the pure route mainly needs a lighter cut once the other-scale / Ka2 over-commit is trimmed.',
            'scd_patch': {'alpha': 0.9992},
            'iter_patches': {1: {'other_scale_alpha': 0.98, 'ka2_alpha': 0.95}},
        },
    ]
    rows = []
    for cand in cands:
        if cand['name'] == 'neutral_scale_ref':
            rows.append({
                'family': 'pure_reference',
                'source_name': cand['name'],
                'candidate': cand,
                'is_reference': True,
            })
        else:
            rows.append({
                'family': 'pure_followup',
                'source_name': cand['name'],
                'candidate': _merge_candidate(base, cand),
                'is_reference': False,
            })
    return rows


def _load_compare_module(noise_scale: float):
    suffix = make_suffix(noise_scale)
    return load_module(f'compare_sym20_pure_followup_mod_{suffix}', str(COMPARE_SCRIPT))


def _load_reference_row(reg: dict[str, Any], noise_scale: float, baseline_g3: dict[str, Any], pure_ref: dict[str, Any]) -> dict[str, Any]:
    ref_path = Path(reg['candidate']['reference_json'])
    payload = _load_json(ref_path)
    overall = payload['overall']
    return {
        'family': reg['family'],
        'source_name': reg['source_name'],
        'candidate_name': reg['candidate']['name'],
        'overall': overall,
        'delta_vs_g3': _delta_vs_ref(overall, baseline_g3),
        'delta_vs_pure_ref': _delta_vs_ref(overall, pure_ref),
        'improvement_vs_g3': _improve_vs_ref(overall, baseline_g3),
        'beats_g3': _beats_flags(overall, baseline_g3),
        'beats_pure_ref': _beats_flags(overall, pure_ref),
        'result_json': str(ref_path),
        'status': 'reference_reused',
        'is_reference': True,
    }


def _run_one_candidate(*, source_mod, params, dataset, case, noise_scale: float, baseline_g3: dict[str, Any], pure_ref: dict[str, Any], reg: dict[str, Any], idx: int, force_rerun: bool) -> dict[str, Any]:
    if reg.get('is_reference'):
        return _load_reference_row(reg, noise_scale, baseline_g3, pure_ref)

    merged = copy.deepcopy(reg['candidate'])
    candidate_name = merged['name']
    out_path = _candidate_output_path(candidate_name, noise_scale)

    if (not force_rerun) and out_path.exists():
        payload = _load_json(out_path)
        extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
        if (
            _noise_matches(payload, dataset['noise_config'])
            and extra.get('comparison_mode') == COMPARISON_MODE
            and extra.get('candidate_name') == candidate_name
            and extra.get('path_case_tag') == case['case_tag']
        ):
            overall = payload['overall']
            return {
                'family': reg['family'],
                'source_name': reg['source_name'],
                'candidate_name': candidate_name,
                'overall': overall,
                'delta_vs_g3': _delta_vs_ref(overall, baseline_g3),
                'delta_vs_pure_ref': _delta_vs_ref(overall, pure_ref),
                'improvement_vs_g3': _improve_vs_ref(overall, baseline_g3),
                'beats_g3': _beats_flags(overall, baseline_g3),
                'beats_pure_ref': _beats_flags(overall, pure_ref),
                'result_json': str(out_path),
                'status': 'reused_verified',
                'is_reference': False,
            }

    method_mod = load_module(f'g4_sym20_pure_followup_method_{idx}_{candidate_name}', str(R53_METHOD_FILE))
    method_mod = _build_patched_method(method_mod, merged)
    result = _run_internalized_hybrid_scd(
        method_mod,
        source_mod,
        dataset['imu_noisy'],
        dataset['pos0'],
        dataset['ts'],
        bi_g=dataset['bi_g'],
        bi_a=dataset['bi_a'],
        tau_g=dataset['tau_g'],
        tau_a=dataset['tau_a'],
        label=f'G4-PURE-FOLLOWUP-{idx:02d}-{candidate_name}',
        scd_cfg=merged['scd'],
    )
    runtime_extra = result[4] if len(result) >= 5 and isinstance(result[4], dict) else {}

    payload = compute_payload(
        source_mod,
        result[0],
        params,
        variant=f'g4_sym20_pure_followup_{candidate_name}_{make_suffix(noise_scale)}',
        method_file='round53_base + _build_patched_method + local _run_internalized_hybrid_scd',
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
            'rationale': merged.get('rationale'),
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'candidate_family': reg['family'],
            'source_name': reg['source_name'],
            'candidate_name': candidate_name,
            'candidate_description': merged.get('description'),
            'iter_patches': copy.deepcopy(merged.get('iter_patches', {})),
            'scd': copy.deepcopy(merged.get('scd', {})),
            'post_rx_y_mult': float(merged.get('post_rx_y_mult', 1.0)),
            'post_ry_z_mult': float(merged.get('post_ry_z_mult', 1.0)),
            'runtime_log': {
                'schedule_log': runtime_extra.get('schedule_log'),
                'feedback_log': runtime_extra.get('feedback_log'),
                'scd_log': runtime_extra.get('scd_log'),
            },
            'baseline_g3_json': str(BASELINE_G3_JSON),
            'pure_reference_json': str(PURE_SCD_JSON),
        },
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    overall = payload['overall']
    return {
        'family': reg['family'],
        'source_name': reg['source_name'],
        'candidate_name': candidate_name,
        'overall': overall,
        'delta_vs_g3': _delta_vs_ref(overall, baseline_g3),
        'delta_vs_pure_ref': _delta_vs_ref(overall, pure_ref),
        'improvement_vs_g3': _improve_vs_ref(overall, baseline_g3),
        'beats_g3': _beats_flags(overall, baseline_g3),
        'beats_pure_ref': _beats_flags(overall, pure_ref),
        'result_json': str(out_path),
        'status': 'rerun',
        'is_reference': False,
    }


def _render_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append('# G4 pure-neighborhood follow-up on corrected symmetric20 (noise0p12, att0=(0,0,0))')
    lines.append('')
    lines.append('## 1) Fixed setup')
    lines.append('')
    lines.append(f"- Path: **{summary['path_case']['case_tag']}**")
    lines.append(f"- Att0: **{summary['path_case']['att0_deg']}**")
    lines.append(f"- Noise scale: **{summary['noise_scale']}**")
    lines.append(f"- G3 baseline overall (mean / median / max): **{_triplet(summary['baseline_g3']['overall'])}**")
    lines.append(f"- Pure reference overall (mean / median / max): **{_triplet(summary['pure_reference']['overall'])}**")
    lines.append('')

    lines.append('## 2) Candidate ranking (sort: mean → max → median)')
    lines.append('')
    lines.append('| rank | family | candidate | mean | median | max | Δmean vs G3 | Δmedian vs G3 | Δmax vs G3 | Δmean vs pure | Δmedian vs pure | Δmax vs pure | beats(mean/med/max) |')
    lines.append('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for row in summary['ranking']:
        dg = row['delta_vs_g3']
        dp = row['delta_vs_pure_ref']
        b = row['beats_g3']
        beat_txt = f"{int(b['mean'])}/{int(b['median'])}/{int(b['max'])}"
        lines.append(
            f"| {row['rank']} | {row['family']} | {row['candidate_name']} | "
            f"{row['overall']['mean_pct_error']:.6f} | {row['overall']['median_pct_error']:.6f} | {row['overall']['max_pct_error']:.6f} | "
            f"{dg['mean_pct_error']:+.6f} | {dg['median_pct_error']:+.6f} | {dg['max_pct_error']:+.6f} | "
            f"{dp['mean_pct_error']:+.6f} | {dp['median_pct_error']:+.6f} | {dp['max_pct_error']:+.6f} | {beat_txt} |"
        )
    lines.append('')

    lines.append('## 3) Headline verdict')
    lines.append('')
    best = summary['best_candidate']
    lines.append(f"- Best candidate: **{best['candidate_name']}** ({best['family']})")
    lines.append(f"- Best overall: **{_triplet(best['overall'])}**")
    lines.append(f"- Beats G3 on all three? **{best['beats_g3']['all_three']}**")
    lines.append(f"- Any candidate beats G3 on all three? **{summary['any_beats_g3_all_three']}**")
    if summary.get('clear_all_three_winner'):
        lines.append(f"- ✅ Clear all-three winner: **{summary['clear_all_three_winner']['candidate_name']}**")
    else:
        gap = summary['nearest_gap_best_vs_g3']
        lines.append(
            '- ❌ No candidate beats G3 on all three metrics. '
            f"Nearest gap (best vs G3): Δmean={gap['mean_pct_error']:+.6f}, "
            f"Δmedian={gap['median_pct_error']:+.6f}, Δmax={gap['max_pct_error']:+.6f}."
        )
    lines.append('')

    lines.append('## 4) Best new (non-reference) candidate')
    lines.append('')
    best_new = summary['best_new_candidate']
    lines.append(f"- Candidate: **{best_new['candidate_name']}**")
    lines.append(f"- Overall (mean / median / max): **{_triplet(best_new['overall'])}**")
    lines.append(
        f"- vs G3: Δmean={best_new['delta_vs_g3']['mean_pct_error']:+.6f}, "
        f"Δmedian={best_new['delta_vs_g3']['median_pct_error']:+.6f}, "
        f"Δmax={best_new['delta_vs_g3']['max_pct_error']:+.6f}"
    )
    lines.append(
        f"- vs pure reference: Δmean={best_new['delta_vs_pure_ref']['mean_pct_error']:+.6f}, "
        f"Δmedian={best_new['delta_vs_pure_ref']['median_pct_error']:+.6f}, "
        f"Δmax={best_new['delta_vs_pure_ref']['max_pct_error']:+.6f}"
    )
    lines.append('')

    lines.append('## 5) Artifacts')
    lines.append('')
    lines.append(f"- summary_json: `{summary['artifacts']['summary_json']}`")
    lines.append(f"- report_md: `{summary['artifacts']['report_md']}`")
    lines.append(f"- best_candidate_json: `{summary['artifacts']['best_candidate_json']}`")
    lines.append(f"- best_new_candidate_json: `{summary['artifacts']['best_new_candidate_json']}`")
    lines.append('')

    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_payload = _load_json(BASELINE_G3_JSON)
    pure_payload = _load_json(PURE_SCD_JSON)
    baseline_g3 = baseline_payload['overall']
    pure_ref = pure_payload['overall']

    compare_mod = _load_compare_module(args.noise_scale)
    source_mod = load_module(f'g4_sym20_pure_followup_source_{make_suffix(args.noise_scale)}', str(SOURCE_FILE))
    case = compare_mod.build_symmetric20_case(source_mod)
    dataset = compare_mod.build_dataset(source_mod, case['paras'], case['att0_deg'], args.noise_scale)
    params = _param_specs(source_mod)

    rows: list[dict[str, Any]] = []
    for idx, reg in enumerate(_candidate_registry(), start=1):
        row = _run_one_candidate(
            source_mod=source_mod,
            params=params,
            dataset=dataset,
            case=case,
            noise_scale=args.noise_scale,
            baseline_g3=baseline_g3,
            pure_ref=pure_ref,
            reg=reg,
            idx=idx,
            force_rerun=args.force_rerun,
        )
        rows.append(row)

    rows_sorted = sorted(
        rows,
        key=lambda x: (
            float(x['overall']['mean_pct_error']),
            float(x['overall']['max_pct_error']),
            float(x['overall']['median_pct_error']),
        ),
    )
    for rank, row in enumerate(rows_sorted, start=1):
        row['rank'] = rank

    best = copy.deepcopy(rows_sorted[0])
    nonref_rows = [r for r in rows_sorted if not r.get('is_reference')]
    best_new = copy.deepcopy(nonref_rows[0])

    clear_all_three_winner = None
    for r in rows_sorted:
        if r['beats_g3']['all_three']:
            clear_all_three_winner = {
                'candidate_name': r['candidate_name'],
                'family': r['family'],
                'rank': r['rank'],
                'overall': r['overall'],
                'result_json': r['result_json'],
            }
            break

    suffix = make_suffix(args.noise_scale)
    summary_json = RESULTS_DIR / f'g4_sym20_pure_followup_{args.report_date}_{suffix}_summary.json'
    report_md = REPORTS_DIR / f'psins_g4_sym20_pure_followup_{args.report_date}_{suffix}.md'

    summary = {
        'task': 'g4_pure_neighborhood_followup_on_corrected_symmetric20',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'noise_config': dataset['noise_config'],
        'comparison_mode': COMPARISON_MODE,
        'path_case': {
            'case_key': case['case_key'],
            'case_tag': case['case_tag'],
            'display_name': case['display_name'],
            'att0_deg': case['att0_deg'],
            'n_motion_rows': case['n_motion_rows'],
            'claimed_position_count': case['claimed_position_count'],
            'total_time_s': case['total_time_s'],
            'timing_note': case['timing_note'],
            'source_builder': case['source_builder'],
            'source_reference': case['source_reference'],
            'builder_method_tag': case.get('builder_method_tag'),
        },
        'baseline_g3': {
            'json_path': str(BASELINE_G3_JSON),
            'overall': baseline_g3,
        },
        'pure_reference': {
            'json_path': str(PURE_SCD_JSON),
            'overall': pure_ref,
        },
        'candidate_count': len(rows_sorted),
        'ranking_rule': 'sort by mean_pct_error, then max_pct_error, then median_pct_error (ascending)',
        'ranking': rows_sorted,
        'best_candidate': best,
        'best_new_candidate': best_new,
        'any_beats_g3': {
            'mean': any(r['beats_g3']['mean'] for r in rows_sorted),
            'median': any(r['beats_g3']['median'] for r in rows_sorted),
            'max': any(r['beats_g3']['max'] for r in rows_sorted),
        },
        'any_beats_g3_all_three': any(r['beats_g3']['all_three'] for r in rows_sorted),
        'clear_all_three_winner': clear_all_three_winner,
        'nearest_gap_best_vs_g3': best['delta_vs_g3'],
        'nearest_gap_best_new_vs_g3': best_new['delta_vs_g3'],
        'artifacts': {
            'summary_json': str(summary_json),
            'report_md': str(report_md),
            'best_candidate_json': best['result_json'],
            'best_new_candidate_json': best_new['result_json'],
            'all_candidate_jsons': [r['result_json'] for r in rows_sorted],
        },
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_md.write_text(_render_report(summary), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps({
        'summary_json': str(summary_json),
        'report_md': str(report_md),
        'best_candidate': {
            'name': best['candidate_name'],
            'family': best['family'],
            'overall': best['overall'],
            'beats_g3': best['beats_g3'],
            'result_json': best['result_json'],
        },
        'best_new_candidate': {
            'name': best_new['candidate_name'],
            'family': best_new['family'],
            'overall': best_new['overall'],
            'beats_g3': best_new['beats_g3'],
            'result_json': best_new['result_json'],
        },
        'any_beats_g3_all_three': summary['any_beats_g3_all_three'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
