from __future__ import annotations

import copy
import json
import sys
import types
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
METHOD_DIR = ROOT / 'psins_method_bench' / 'methods' / 'markov'
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
REPORTS_DIR = ROOT / 'reports'
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'
COMPUTE_R61_FILE = SCRIPTS_DIR / 'compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors.py'
ROUND62_SUMMARY_JSON = RESULTS_DIR / 'round62_alpha_guard_probe_summary.json'
OUTPUT_JSON = RESULTS_DIR / 'round63a_scd_gating_probe_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'round63a_scd_gating_candidates.json'
REPORT_MD = REPORTS_DIR / 'psins_round63a_scd_gating_probe_2026-03-28.md'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import load_module
from probe_round55_newline import _build_patched_method
from probe_round61_hybrid_micro import ROUND61_CANDIDATES, _merge_round61_candidate

BASE_R61_CANDIDATE_NAME = 'r61_s20_08988_ryz00116'
PROBE_SCALES = [1.0, 0.10, 0.08, 0.05, 0.03]

ROUND63A_CANDIDATES = [
    {
        'name': 'r63a_ul_scd_alpha09995',
        'description': 'Below 0.08x only weaken the once-per-phase SCD alpha from 0.999 to 0.9995; keep cadence/target unchanged.',
        'rationale': 'Pure “too strong” test: if the ultra-low distortion comes from the SCD cut being slightly over-aggressive, the mildest repair is a smaller cross-covariance shrink.',
        'round63a_scd_guard': {
            'active_below_noise_scale': 0.08,
            'scd_override': {
                'alpha': 0.9995,
            },
        },
    },
    {
        'name': 'r63a_ul_scd_delay6p0',
        'description': 'Below 0.08x only delay the once-per-phase SCD trigger from 2.0 s to 6.0 s after each rotation stop.',
        'rationale': 'Direct “too early” test: preserve the same alpha and target block, but push the one-shot cut later so the static update settles first.',
        'round63a_scd_guard': {
            'active_below_noise_scale': 0.08,
            'scd_override': {
                'transition_duration': 6.0,
            },
        },
    },
    {
        'name': 'r63a_ul_scd_core3_only',
        'description': 'Below 0.08x only shrink the SCD target from the full scale block to the Round61 backbone trio: dKg_xx / dKg_xy / dKg_zz.',
        'rationale': 'Interpretable subset test: keep the ultra-low SCD idea only on the three states that still look structurally helpful, and stop suppressing yy / Ka_xx-side cross-covariances.',
        'round63a_scd_guard': {
            'active_below_noise_scale': 0.08,
            'scd_override': {
                'target': 'custom',
                'target_name': 'selected_core3',
                'target_indices': [12, 15, 20],
            },
        },
    },
    {
        'name': 'r63a_ul_scd_floor_off',
        'description': 'Below 0.08x disable the iter2 once-per-phase SCD entirely and keep the rest of Round61 unchanged.',
        'rationale': 'Floor ablation test: checks whether the remaining ultra-low distortion is largely coming from the SCD branch itself, not just from the feedback alpha side.',
        'round63a_scd_guard': {
            'active_below_noise_scale': 0.08,
            'scd_override': {
                'apply_policy_names': [],
                'target_name': 'disabled',
            },
        },
    },
]

BASE_ARW = 0.005
BASE_VRW = 5.0
BASE_BI_G = 0.002
BASE_BI_A = 5.0
TAU_G = 300.0
TAU_A = 300.0


def make_suffix(noise_scale: float) -> str:
    mapping = {
        1.0: 'noise1x',
        1.0 / 3.0: 'noise1over3',
        0.2: 'noise1over5',
        0.5: 'noise0p5',
        2.0: 'noise2p0',
    }
    for key, value in mapping.items():
        if abs(noise_scale - key) < 1e-12:
            return value
    return f"noise{str(noise_scale).replace('.', 'p')}"


def _baseline_payload_path(noise_scale: float) -> Path:
    return RESULTS_DIR / f'M_markov_42state_gm1_shared_{make_suffix(noise_scale)}_param_errors.json'


def _round61_payload_path(noise_scale: float) -> Path:
    return RESULTS_DIR / f'R61_42state_gm1_round61_h_scd_state20_microtight_commit_shared_{make_suffix(noise_scale)}_param_errors.json'


def _compare_payload_path(noise_scale: float) -> Path:
    return RESULTS_DIR / f'compare_baseline_vs_round61_shared_{make_suffix(noise_scale)}.json'


def _load_r61_base_candidate():
    for candidate in ROUND61_CANDIDATES:
        if candidate['name'] == BASE_R61_CANDIDATE_NAME:
            return _merge_round61_candidate(candidate)
    raise KeyError(BASE_R61_CANDIDATE_NAME)


def build_shared_dataset(mod, noise_scale: float):
    ts = 0.01
    att0 = mod.np.array([1.0, -91.0, -91.0]) * mod.glv.deg
    pos0 = mod.posset(34.0, 0.0, 0.0)
    paras = mod.np.array([
        [1, 0, 1, 0, 90, 9, 70, 70], [2, 0, 1, 0, 90, 9, 20, 20], [3, 0, 1, 0, 90, 9, 20, 20],
        [4, 0, 1, 0, -90, 9, 20, 20], [5, 0, 1, 0, -90, 9, 20, 20], [6, 0, 1, 0, -90, 9, 20, 20],
        [7, 0, 0, 1, 90, 9, 20, 20], [8, 1, 0, 0, 90, 9, 20, 20], [9, 1, 0, 0, 90, 9, 20, 20],
        [10, 1, 0, 0, 90, 9, 20, 20], [11, -1, 0, 0, 90, 9, 20, 20], [12, -1, 0, 0, 90, 9, 20, 20],
        [13, -1, 0, 0, 90, 9, 20, 20], [14, 0, 0, 1, 90, 9, 20, 20], [15, 0, 0, 1, 90, 9, 20, 20],
        [16, 0, 0, -1, 90, 9, 20, 20], [17, 0, 0, -1, 90, 9, 20, 20], [18, 0, 0, -1, 90, 9, 20, 20],
    ], dtype=float)
    paras[:, 4] = paras[:, 4] * mod.glv.deg
    att = mod.attrottt(att0, paras, ts)
    imu, _ = mod.avp2imu(att, pos0)
    clbt_truth = mod.get_default_clbt()
    imu_clean = mod.imuclbt(imu, clbt_truth)

    arw = BASE_ARW * noise_scale * mod.glv.dpsh
    vrw = BASE_VRW * noise_scale * mod.glv.ugpsHz
    bi_g = BASE_BI_G * noise_scale * mod.glv.dph
    bi_a = BASE_BI_A * noise_scale * mod.glv.ug
    imu_noisy = mod.imuadderr_full(
        imu_clean, ts,
        arw=arw, vrw=vrw,
        bi_g=bi_g, tau_g=TAU_G,
        bi_a=bi_a, tau_a=TAU_A, seed=42,
    )
    return {
        'ts': ts,
        'pos0': pos0,
        'imu_noisy': imu_noisy,
        'bi_g': bi_g,
        'bi_a': bi_a,
        'tau_g': TAU_G,
        'tau_a': TAU_A,
        'noise_scale': noise_scale,
        'noise_config': {
            'arw_dpsh': BASE_ARW * noise_scale,
            'vrw_ugpsHz': BASE_VRW * noise_scale,
            'bi_g_dph': BASE_BI_G * noise_scale,
            'bi_a_ug': BASE_BI_A * noise_scale,
            'tau_g': TAU_G,
            'tau_a': TAU_A,
            'seed': 42,
            'base_family': 'round63a_shared_probe',
        },
    }


def _merge_round63a_candidate(extra_candidate: dict):
    merged = copy.deepcopy(_load_r61_base_candidate())
    merged['name'] = extra_candidate['name']
    merged['description'] = extra_candidate['description']
    merged['rationale'] = extra_candidate['rationale']
    merged['round63a_scd_guard'] = copy.deepcopy(extra_candidate.get('round63a_scd_guard', {}))
    merged['round63a_extra_patch'] = copy.deepcopy(extra_candidate)
    return merged


def _resolve_round63a_scd(merged_candidate: dict, noise_scale: float):
    base_scd = copy.deepcopy(merged_candidate.get('scd', {}))
    guard_cfg = copy.deepcopy(merged_candidate.get('round63a_scd_guard', {}))
    active_below = float(guard_cfg.get('active_below_noise_scale', -1.0))
    triggered = noise_scale < active_below - 1e-12
    resolved = copy.deepcopy(base_scd)
    if triggered:
        override = copy.deepcopy(guard_cfg.get('scd_override', {}))
        resolved.update(override)
    return resolved, {
        'active_below_noise_scale': active_below,
        'triggered': bool(triggered),
        'base_scd': base_scd,
        'resolved_scd': copy.deepcopy(resolved),
    }


def _resolve_target_indices(method_mod, scd_cfg):
    target_name = scd_cfg.get('target')
    if target_name == 'selected':
        return [int(idx) for idx in method_mod.SELECTED_SCALE_STATES], 'selected'
    if target_name == 'scale_block':
        return list(range(12, 27)), 'scale_block'
    if target_name == 'custom':
        indices = [int(idx) for idx in scd_cfg.get('target_indices', [])]
        if not indices:
            raise ValueError('custom SCD target requires target_indices')
        return indices, str(scd_cfg.get('target_name', 'custom'))
    raise KeyError(f'Unknown SCD target: {target_name}')


def _apply_hybrid_scd(method_mod, kf, scd_cfg, target_indices):
    alpha = float(scd_cfg['alpha'])
    P = kf['Pxk']

    P[0:6, target_indices] *= alpha
    P[target_indices, 0:6] *= alpha

    if scd_cfg.get('bias_to_target', True):
        P[6:12, target_indices] *= alpha
        P[target_indices, 6:12] *= alpha


# Round63-A reuses the Round61 hybrid route exactly, and only swaps the SCD
# config per shared-noise scale for a tiny deterministic ultra-low batch.
def _run_internalized_hybrid_scd_round63a(method_mod, mod, imu1, pos0, ts, bi_g, bi_a, tau_g, tau_a, label, scd_cfg):
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

    target_indices, target_label = _resolve_target_indices(method_mod, scd_cfg)
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
            'target_name': target_label,
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
        'policy': 'Round63-A keeps the exact Round61 feedback stack and probes only ultra-low-noise SCD gating changes.',
    }


def _compute_payload(source_mod, clbt, params, variant: str, method_file: str, extra=None):
    param_errors = {}
    pct_values = []
    for name, true_v, get_est in params:
        true_f = float(true_v)
        est_f = float(get_est(clbt))
        abs_err = abs(true_f - est_f)
        pct_err = abs_err / abs(true_f) * 100.0 if abs(true_f) > 1e-15 else 0.0
        param_errors[name] = {
            'true': true_f,
            'est': est_f,
            'abs_error': abs_err,
            'pct_error': pct_err,
        }
        pct_values.append(pct_err)

    pct_arr = source_mod.np.asarray(pct_values, dtype=float)
    focus_scale_pct = {
        'dKg_xx': param_errors['dKg_xx']['pct_error'],
        'dKg_xy': param_errors['dKg_xy']['pct_error'],
        'dKg_yy': param_errors['dKg_yy']['pct_error'],
        'dKg_zz': param_errors['dKg_zz']['pct_error'],
        'dKa_xx': param_errors['dKa_xx']['pct_error'],
    }
    lever_guard_pct = {
        'rx_y': param_errors['rx_y']['pct_error'],
        'ry_z': param_errors['ry_z']['pct_error'],
    }
    overall = {
        'mean_pct_error': float(source_mod.np.mean(pct_arr)),
        'median_pct_error': float(source_mod.np.median(pct_arr)),
        'max_pct_error': float(source_mod.np.max(pct_arr)),
    }
    return {
        'variant': variant,
        'method_file': method_file,
        'source_file': str(SOURCE_FILE),
        'param_order': [name for name, _, _ in params],
        'param_errors': param_errors,
        'focus_scale_pct': focus_scale_pct,
        'lever_guard_pct': lever_guard_pct,
        'overall': overall,
        'extra': extra or {},
    }


def _relative_improvement_block(baseline_payload: dict, candidate_payload: dict, keys: list[str]):
    out = {}
    for key in keys:
        if key in candidate_payload['param_errors']:
            b = float(baseline_payload['param_errors'][key]['pct_error'])
            c = float(candidate_payload['param_errors'][key]['pct_error'])
        else:
            b = float(baseline_payload['overall'][key])
            c = float(candidate_payload['overall'][key])
        out[key] = {
            'baseline_pct_error': b,
            'candidate_pct_error': c,
            'delta_pct_points': b - c,
            'relative_improvement_pct': ((b - c) / b * 100.0) if abs(b) > 1e-15 else None,
        }
    return out


def _sorted_policy_patch(iter_patches: dict):
    out = {}
    for iter_idx, patch in sorted(iter_patches.items()):
        out[str(iter_idx + 1)] = {
            key: {str(k): float(v) for k, v in value.items()} if isinstance(value, dict) else value
            for key, value in patch.items()
        }
    return out


def _run_candidate_at_scale(candidate: dict, noise_scale: float, idx: int):
    suffix = make_suffix(noise_scale)
    source_mod = load_module(f'markov_pruned_source_r63a_probe_{candidate["name"]}_{suffix}', str(SOURCE_FILE))
    compute_r61_mod = load_module(f'compute_r61_for_r63a_probe_{suffix}_{idx}', str(COMPUTE_R61_FILE))
    params = compute_r61_mod._param_specs(source_mod)

    dataset = build_shared_dataset(source_mod, noise_scale)
    merged_candidate = _merge_round63a_candidate(candidate)
    resolved_scd, scd_meta = _resolve_round63a_scd(merged_candidate, noise_scale)
    base_method_mod = load_module(f'markov_method_r63a_probe_base_{candidate["name"]}_{suffix}', str(R53_METHOD_FILE))
    method_mod = _build_patched_method(base_method_mod, merged_candidate)
    result = list(_run_internalized_hybrid_scd_round63a(
        method_mod,
        source_mod,
        dataset['imu_noisy'],
        dataset['pos0'],
        dataset['ts'],
        bi_g=dataset['bi_g'],
        bi_a=dataset['bi_a'],
        tau_g=dataset['tau_g'],
        tau_a=dataset['tau_a'],
        label=f'42-GM1-R63A-{idx}-{suffix.upper()}',
        scd_cfg=resolved_scd,
    ))
    extra = result[4] if len(result) >= 5 and isinstance(result[4], dict) else {}
    extra = dict(extra)
    extra.update({
        'noise_scale': float(noise_scale),
        'noise_config': dataset['noise_config'],
        'comparison_mode': 'shared_dataset_apples_to_apples',
        'round63a_selected_candidate': merged_candidate['name'],
        'round63a_base_round61_candidate': BASE_R61_CANDIDATE_NAME,
        'round63a_scd_guard': copy.deepcopy(merged_candidate.get('round63a_scd_guard', {})),
        'round63a_resolved_scd': copy.deepcopy(resolved_scd),
        'round63a_scd_guard_meta': copy.deepcopy(scd_meta),
        'round63a_note': 'Round63-A leaves the Round61 feedback route intact and changes only the ultra-low SCD gate/cadence/target.',
    })
    payload = _compute_payload(
        source_mod,
        result[0],
        params,
        variant=f'42state_gm1_round63a_scd_gating_{merged_candidate["name"]}_shared_{suffix}',
        method_file=f'probe_round63a_scd_gating::{merged_candidate["name"]}',
        extra=extra,
    )
    return merged_candidate, payload


def _score_candidate(candidate_scales: dict):
    score = 0.0
    penalties = []
    protect_scale_keys = ['noise1x', 'noise0p1', 'noise0p08']
    protect_metrics = [
        ('focus', 'dKg_xx'), ('focus', 'dKg_xy'), ('focus', 'dKg_yy'), ('focus', 'dKg_zz'), ('focus', 'dKa_xx'),
        ('lever', 'rx_y'), ('lever', 'ry_z'),
        ('overall', 'mean_pct_error'), ('overall', 'median_pct_error'), ('overall', 'max_pct_error'),
    ]
    for scale_key in protect_scale_keys:
        cand = candidate_scales[scale_key]
        for block_name, metric_name in protect_metrics:
            value = float(cand['delta_vs_round61'][metric_name])
            if value > 1e-9:
                penalties.append({'scale': scale_key, 'metric': metric_name, 'delta': value})
                score -= 1000.0 * value

    weights = {
        'noise0p05': {'dKg_yy': 1.5, 'dKa_xx': 1.4, 'rx_y': 1.2, 'mean_pct_error': 0.9, 'median_pct_error': 0.8},
        'noise0p03': {'dKg_yy': 2.4, 'dKa_xx': 2.0, 'rx_y': 1.6, 'mean_pct_error': 1.1, 'median_pct_error': 1.0},
    }
    for scale_key, scale_weights in weights.items():
        cand = candidate_scales[scale_key]['delta_vs_round61']
        for metric_name, weight in scale_weights.items():
            score += weight * (-float(cand[metric_name]))
    return float(score), penalties


def _score_vs_round62(candidate_scales: dict):
    score = 0.0
    weights = {
        'noise0p05': {'dKg_yy': 1.5, 'dKa_xx': 1.4, 'rx_y': 1.2, 'mean_pct_error': 0.9, 'median_pct_error': 0.8},
        'noise0p03': {'dKg_yy': 2.4, 'dKa_xx': 2.0, 'rx_y': 1.6, 'mean_pct_error': 1.1, 'median_pct_error': 1.0},
    }
    for scale_key, scale_weights in weights.items():
        cand = candidate_scales[scale_key]['delta_vs_round62_winner']
        for metric_name, weight in scale_weights.items():
            score += weight * (-float(cand[metric_name]))
    return float(score)


def _build_scale_delta(cand_focus, cand_lever, cand_overall, ref_focus, ref_lever, ref_overall):
    return {
        'dKg_xx': float(cand_focus['dKg_xx'] - ref_focus['dKg_xx']),
        'dKg_xy': float(cand_focus['dKg_xy'] - ref_focus['dKg_xy']),
        'dKg_yy': float(cand_focus['dKg_yy'] - ref_focus['dKg_yy']),
        'dKg_zz': float(cand_focus['dKg_zz'] - ref_focus['dKg_zz']),
        'dKa_xx': float(cand_focus['dKa_xx'] - ref_focus['dKa_xx']),
        'rx_y': float(cand_lever['rx_y'] - ref_lever['rx_y']),
        'ry_z': float(cand_lever['ry_z'] - ref_lever['ry_z']),
        'mean_pct_error': float(cand_overall['mean_pct_error'] - ref_overall['mean_pct_error']),
        'median_pct_error': float(cand_overall['median_pct_error'] - ref_overall['median_pct_error']),
        'max_pct_error': float(cand_overall['max_pct_error'] - ref_overall['max_pct_error']),
    }


def _selection_note(candidate_scales: dict, penalties: list[dict], score_vs_round62: float):
    if penalties:
        return f'Protected-scale regression detected: {penalties}'

    d03_r61 = candidate_scales['noise0p03']['delta_vs_round61']
    d05_r61 = candidate_scales['noise0p05']['delta_vs_round61']
    d03_r62 = candidate_scales['noise0p03']['delta_vs_round62_winner']
    d05_r62 = candidate_scales['noise0p05']['delta_vs_round62_winner']

    if (
        d03_r61['dKg_yy'] < 0 and d03_r61['dKa_xx'] < 0 and d03_r61['rx_y'] < 0
        and d03_r61['mean_pct_error'] <= 0 and d03_r61['median_pct_error'] <= 0
    ):
        if score_vs_round62 > 0.0:
            return 'Strong branch win: improves the full ultra-low target set versus Round61 and also edges past Round62 on the weighted 0.03x/0.05x score.'
        return 'Strong branch win versus Round61 on the full ultra-low target set; still a complementary branch rather than a clear Round62 replacement.'

    if d03_r61['dKg_yy'] < 0 and d03_r61['dKa_xx'] < 0 and d03_r61['mean_pct_error'] <= 0:
        if d03_r62['dKa_xx'] < 0 or d05_r62['dKa_xx'] < 0:
            return 'Complementary branch signal: SCD gating repairs dKg_yy / dKa_xx / overall versus Round61 and may trade against Round62 mainly on rx_y.'
        return 'Clean Round61 improvement signal, but not obviously better than Round62 on the weighted ultra-low tradeoff.'

    if d03_r61['dKg_yy'] < 0 or d03_r61['dKa_xx'] < 0 or d03_r61['rx_y'] < 0:
        return 'Partial ultra-low repair: at least one target improves, but the full dKg_yy / dKa_xx / rx_y / mean / median set does not move together.'

    return 'No useful ultra-low repair signal: the SCD-side change is too weak or moves the wrong target set.'


def _render_report(summary: dict):
    lines = []
    lines.append('<callout emoji="🧪" background-color="light-blue">')
    lines.append('Round63-A：从 Round61 主干分出 **ultra-low SCD gating** 小分叉，只改 SCD 侧（强度 / 时机 / 目标 / 是否关闭），不再继续做 alpha-guard 扫描。')
    lines.append('</callout>')
    lines.append('')
    lines.append('## 1. Probe 设置')
    lines.append('')
    lines.append(f'- Base candidate: `{summary["base_round61_candidate"]}`')
    lines.append(f'- Round62 reference winner: `{summary["round62_reference"]["winner_name"]}`')
    lines.append('- Shared-noise probe scales: `1.00x, 0.10x, 0.08x, 0.05x, 0.03x`')
    lines.append('- Batch size: **4 deterministic ultra-low SCD-gating candidates**')
    lines.append('')
    lines.append('## 2. 候选摘要')
    lines.append('')
    lines.append('| candidate | 0.03x dKg_yy ΔvsR61 | 0.03x dKa_xx ΔvsR61 | 0.03x rx_y ΔvsR61 | 0.03x mean ΔvsR61 | 0.03x median ΔvsR61 | score vs R61 | score vs R62 | note |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---|')
    for name in summary['candidate_order']:
        cand = summary['candidates'][name]
        s03 = cand['scales']['noise0p03']
        lines.append(
            f"| `{name}` | {s03['delta_vs_round61']['dKg_yy']:.4f} | {s03['delta_vs_round61']['dKa_xx']:.4f} | {s03['delta_vs_round61']['rx_y']:.4f} | {s03['delta_vs_round61']['mean_pct_error']:.4f} | {s03['delta_vs_round61']['median_pct_error']:.4f} | {cand['selection']['score_vs_round61']:.4f} | {cand['selection']['score_vs_round62_winner']:.4f} | {cand['selection']['note']} |"
        )
    lines.append('')
    lines.append('## 3. Winner / status')
    lines.append('')
    branch_winner = summary['branch_winner']
    replacement = summary['mainline_replacement']
    if branch_winner:
        lines.append(f'- Branch winner: `{branch_winner["name"]}`')
        lines.append(f'- Branch winner reason: {branch_winner["reason"]}')
    else:
        lines.append('- Branch winner: **none**')
        lines.append(f'- Reason: {summary["no_branch_winner_reason"]}')
    if replacement:
        lines.append(f'- Round62 replacement status: **yes** → `{replacement["name"]}`')
        lines.append(f'- Replacement reason: {replacement["reason"]}')
    else:
        lines.append('- Round62 replacement status: **no clear replacement yet**')
        lines.append(f'- Reason: {summary["no_replacement_reason"]}')
    lines.append('')
    lines.append('## 4. Candidate notes')
    lines.append('')
    for name in summary['candidate_order']:
        cand = summary['candidates'][name]
        lines.append(f'### `{name}`')
        lines.append(f'- Description: {cand["description"]}')
        lines.append(f'- Rationale: {cand["rationale"]}')
        lines.append(f'- Selection note: {cand["selection"]["note"]}')
        lines.append(f'- Ultra-low SCD guard: `{json.dumps(cand["round63a_scd_guard"], ensure_ascii=False)}`')
        lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    round62_summary = json.loads(ROUND62_SUMMARY_JSON.read_text(encoding='utf-8'))
    round62_winner = round62_summary.get('winner')
    round62_winner_name = round62_winner['name'] if round62_winner else None
    if not round62_winner_name:
        raise RuntimeError('Round62 winner not found in summary JSON')
    round62_winner_scales = round62_summary['candidates'][round62_winner_name]['scales']

    candidate_dump = {
        'base_round61_candidate': BASE_R61_CANDIDATE_NAME,
        'round62_reference_winner': round62_winner_name,
        'probe_scales': PROBE_SCALES,
        'round63a_candidates': ROUND63A_CANDIDATES,
    }
    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    baseline_payloads = {}
    round61_payloads = {}
    round61_compare = {}
    for scale in PROBE_SCALES:
        suffix = make_suffix(scale)
        baseline_payloads[suffix] = json.loads(_baseline_payload_path(scale).read_text(encoding='utf-8'))
        round61_payloads[suffix] = json.loads(_round61_payload_path(scale).read_text(encoding='utf-8'))
        round61_compare[suffix] = json.loads(_compare_payload_path(scale).read_text(encoding='utf-8'))

    out = {
        'base_round61_candidate': BASE_R61_CANDIDATE_NAME,
        'probe_scales': PROBE_SCALES,
        'candidate_order': [c['name'] for c in ROUND63A_CANDIDATES],
        'baseline_paths': {make_suffix(s): str(_baseline_payload_path(s)) for s in PROBE_SCALES},
        'round61_paths': {make_suffix(s): str(_round61_payload_path(s)) for s in PROBE_SCALES},
        'round62_reference': {
            'summary_json': str(ROUND62_SUMMARY_JSON),
            'winner_name': round62_winner_name,
            'winner_reason': round62_winner.get('reason'),
        },
        'candidate_json': str(CANDIDATE_JSON),
        'candidates': {},
        'branch_winner': None,
        'mainline_replacement': None,
        'no_branch_winner_reason': None,
        'no_replacement_reason': None,
    }

    for idx, candidate in enumerate(ROUND63A_CANDIDATES, start=1):
        merged_candidate = _merge_round63a_candidate(candidate)
        candidate_scales = {}
        for scale in PROBE_SCALES:
            suffix = make_suffix(scale)
            merged_candidate, payload = _run_candidate_at_scale(candidate, scale, idx)
            baseline_payload = baseline_payloads[suffix]
            round61_payload = round61_payloads[suffix]
            compare_payload = round61_compare[suffix]
            round62_payload = round62_winner_scales[suffix]
            focus = payload['focus_scale_pct']
            lever = payload['lever_guard_pct']
            overall = payload['overall']
            delta_vs_round61 = _build_scale_delta(
                focus,
                lever,
                overall,
                round61_payload['focus_scale_pct'],
                round61_payload['lever_guard_pct'],
                round61_payload['overall'],
            )
            delta_vs_round62 = _build_scale_delta(
                focus,
                lever,
                overall,
                round62_payload['focus'],
                round62_payload['lever'],
                round62_payload['overall'],
            )
            candidate_vs_baseline = _relative_improvement_block(
                baseline_payload,
                payload,
                ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'mean_pct_error', 'median_pct_error', 'max_pct_error'],
            )
            candidate_scales[suffix] = {
                'noise_scale': float(scale),
                'focus': focus,
                'lever': lever,
                'overall': overall,
                'delta_vs_round61': delta_vs_round61,
                'delta_vs_round62_winner': delta_vs_round62,
                'candidate_vs_baseline': candidate_vs_baseline,
                'round61_vs_baseline_reference': {
                    'dKg_xx': compare_payload['all_params']['dKg_xx'],
                    'dKg_xy': compare_payload['all_params']['dKg_xy'],
                    'dKg_yy': compare_payload['all_params']['dKg_yy'],
                    'dKg_zz': compare_payload['all_params']['dKg_zz'],
                    'dKa_xx': compare_payload['all_params']['dKa_xx'],
                    'rx_y': compare_payload['all_params']['rx_y'],
                    'ry_z': compare_payload['all_params']['ry_z'],
                    'mean_pct_error': compare_payload['overall']['mean_pct_error'],
                    'median_pct_error': compare_payload['overall']['median_pct_error'],
                    'max_pct_error': compare_payload['overall']['max_pct_error'],
                },
                'round62_winner_reference': {
                    'focus': round62_payload['focus'],
                    'lever': round62_payload['lever'],
                    'overall': round62_payload['overall'],
                },
                'extra': {
                    'resolved_scd': payload['extra'].get('round63a_resolved_scd'),
                    'scd_guard_meta': payload['extra'].get('round63a_scd_guard_meta'),
                    'feedback_log': payload['extra'].get('feedback_log'),
                    'scd_log': payload['extra'].get('scd_log'),
                    'schedule_log': payload['extra'].get('schedule_log'),
                },
            }

        score_vs_round61, penalties = _score_candidate(candidate_scales)
        score_vs_round62 = _score_vs_round62(candidate_scales)
        note = _selection_note(candidate_scales, penalties, score_vs_round62)

        out['candidates'][candidate['name']] = {
            'description': candidate['description'],
            'rationale': candidate['rationale'],
            'base_round61_candidate': BASE_R61_CANDIDATE_NAME,
            'policy_patch': _sorted_policy_patch(merged_candidate.get('iter_patches', {})),
            'round63a_scd_guard': copy.deepcopy(candidate.get('round63a_scd_guard', {})),
            'round63a_extra_patch': copy.deepcopy(candidate),
            'scd': copy.deepcopy(merged_candidate['scd']),
            'post_rx_y_mult': float(merged_candidate.get('post_rx_y_mult', 1.0)),
            'post_ry_z_mult': float(merged_candidate.get('post_ry_z_mult', 1.0)),
            'scales': candidate_scales,
            'selection': {
                'score_vs_round61': float(score_vs_round61),
                'score_vs_round62_winner': float(score_vs_round62),
                'penalties': penalties,
                'note': note,
            },
        }

        print(candidate['name'], json.dumps({
            'score_vs_round61': score_vs_round61,
            'score_vs_round62_winner': score_vs_round62,
            'note': note,
            'noise0p03_vs_round61': candidate_scales['noise0p03']['delta_vs_round61'],
            'noise0p03_vs_round62': candidate_scales['noise0p03']['delta_vs_round62_winner'],
        }, ensure_ascii=False))

    ordered = sorted(
        [(name, out['candidates'][name]['selection']['score_vs_round61']) for name in out['candidate_order']],
        key=lambda x: x[1],
        reverse=True,
    )
    best_name, best_score = ordered[0]
    best_candidate = out['candidates'][best_name]
    best_penalties = best_candidate['selection']['penalties']
    best_r62_score = best_candidate['selection']['score_vs_round62_winner']
    best_d03_r61 = best_candidate['scales']['noise0p03']['delta_vs_round61']
    best_d05_r61 = best_candidate['scales']['noise0p05']['delta_vs_round61']

    if (not best_penalties) and best_score > 0.01 and (
        best_d03_r61['dKg_yy'] < 0 or best_d03_r61['dKa_xx'] < 0 or best_d03_r61['rx_y'] < 0 or best_d03_r61['mean_pct_error'] < 0
    ):
        out['branch_winner'] = {
            'name': best_name,
            'score_vs_round61': float(best_score),
            'score_vs_round62_winner': float(best_r62_score),
            'reason': 'Best protected no-regression SCD-only ultra-low branch score versus Round61 across the 0.05x / 0.03x target set.',
        }
    else:
        out['branch_winner'] = None
        out['no_branch_winner_reason'] = 'No Round63-A candidate produced a clean protected no-regression ultra-low improvement signal versus Round61.'

    if out['branch_winner'] and best_r62_score > 0.01 and (
        best_d03_r61['dKg_yy'] < 0 and best_d03_r61['dKa_xx'] < 0 and best_d03_r61['mean_pct_error'] <= 0 and best_d05_r61['mean_pct_error'] <= 0
    ):
        out['mainline_replacement'] = {
            'name': best_name,
            'score_vs_round62_winner': float(best_r62_score),
            'reason': 'Branch winner also edges past the Round62 winner on the weighted ultra-low target score while preserving the protected 1x / 0.10x / 0.08x region.',
        }
    else:
        out['mainline_replacement'] = None
        out['no_replacement_reason'] = 'Round63-A did not show a clear weighted ultra-low replacement over the existing Round62 winner; treat the best candidate as a branch/complement unless later evidence says otherwise.'

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    REPORT_MD.write_text(_render_report(out), encoding='utf-8')
    print(f'Wrote {OUTPUT_JSON}')
    print(f'Wrote {REPORT_MD}')
    print('__RESULT_JSON__=' + json.dumps({
        'output_json': str(OUTPUT_JSON),
        'candidate_json': str(CANDIDATE_JSON),
        'report_md': str(REPORT_MD),
        'branch_winner': out['branch_winner'],
        'mainline_replacement': out['mainline_replacement'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
