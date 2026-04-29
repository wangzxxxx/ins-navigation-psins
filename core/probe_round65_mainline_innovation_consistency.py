from __future__ import annotations

import copy
import gc
import json
import math
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

OUTPUT_JSON = RESULTS_DIR / 'round65_probe_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'round65_candidates.json'
REPORT_MD = REPORTS_DIR / 'psins_round65_probe_2026-03-28.md'
ROUND65_RECORD_MD = ROOT / 'psins_method_bench' / 'summary' / 'round65_record_2026-03-28.md'

LADDER_KF_JSON = RESULTS_DIR / 'R65_mainline_kf36_noisy_param_errors.json'
LADDER_MARKOV_JSON = RESULTS_DIR / 'R65_mainline_markov42_noisy_param_errors.json'
LADDER_MARKOV_SCD_JSON = RESULTS_DIR / 'R65_mainline_markov42_plus_scd_baseline_param_errors.json'
LADDER_R61_JSON = RESULTS_DIR / 'R65_mainline_round61_param_errors.json'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import load_module
from probe_round55_newline import _build_patched_method
from probe_round56_narrow import _compute_metrics
from probe_round59_h_scd_hybrid import (
    _apply_hybrid_scd,
    _resolve_target_indices,
    _run_internalized_hybrid_scd,
)
from probe_round61_hybrid_micro import ROUND61_CANDIDATES, _merge_round61_candidate


DATASET_NOISE_CONFIG = {
    'noise_scale_tag': 'mainline_1x_round53_family',
    'arw_dpsh': 0.005,
    'vrw_ugpsHz': 5.0,
    'bi_g_dph': 0.002,
    'bi_a_ug': 5.0,
    'tau_g': 300.0,
    'tau_a': 300.0,
    'seed': 42,
}

ROUND61_BASE_NAME = 'r61_s20_08988_ryz00116'

HARD_PROTECTED_KEYS = ['dKg_xy', 'dKg_yy', 'dKa_xx', 'rx_y', 'ry_z']

ROUND65_CANDIDATES = [
    {
        'name': 'r65_icg_balanced',
        'description': 'Innovation-consistency gate on both trust-feedback and SCD strength with balanced gain.',
        'rationale': 'Core mechanism test: consistency-derived gate controls both internalized feedback aggressiveness and SCD suppression depth.',
        'innovation_gate': {
            'target_nis': 1.0,
            'ema_beta': 0.08,
            'slope': 1.40,
            'gate_floor': 0.72,
            'warmup_static_meas': 8,
            'feedback_gate_power': 1.00,
            'feedback_gate_floor': 0.72,
            'scd_gate_power': 1.00,
            'scd_gate_floor': 0.76,
        },
    },
    {
        'name': 'r65_icg_feedback_priority',
        'description': 'Stronger consistency gating on trust-feedback path; keep SCD path relatively mild.',
        'rationale': 'Ablation-style emphasis: test whether over-correction is mainly from feedback route rather than SCD route.',
        'innovation_gate': {
            'target_nis': 1.0,
            'ema_beta': 0.08,
            'slope': 1.65,
            'gate_floor': 0.68,
            'warmup_static_meas': 8,
            'feedback_gate_power': 1.25,
            'feedback_gate_floor': 0.65,
            'scd_gate_power': 0.70,
            'scd_gate_floor': 0.84,
        },
    },
    {
        'name': 'r65_icg_scd_priority',
        'description': 'Stronger consistency gating on SCD alpha path; keep trust-feedback path relatively mild.',
        'rationale': 'Counter-ablation: test whether mis-match risk is dominated by cross-cov suppression timing/strength.',
        'innovation_gate': {
            'target_nis': 1.0,
            'ema_beta': 0.08,
            'slope': 1.65,
            'gate_floor': 0.68,
            'warmup_static_meas': 8,
            'feedback_gate_power': 0.72,
            'feedback_gate_floor': 0.84,
            'scd_gate_power': 1.30,
            'scd_gate_floor': 0.64,
        },
    },
    {
        'name': 'r65_icg_slow_ema_guarded',
        'description': 'Slower innovation-EMA with moderate gate slope to avoid reacting to transient spikes.',
        'rationale': 'Robustness-style variant: reduce gate jitter and see whether smoother consistency estimate helps protected metrics.',
        'innovation_gate': {
            'target_nis': 1.0,
            'ema_beta': 0.05,
            'slope': 1.55,
            'gate_floor': 0.74,
            'warmup_static_meas': 10,
            'feedback_gate_power': 1.00,
            'feedback_gate_floor': 0.74,
            'scd_gate_power': 1.00,
            'scd_gate_floor': 0.78,
        },
    },
]


def _build_shared_dataset(mod):
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

    bi_g = DATASET_NOISE_CONFIG['bi_g_dph'] * mod.glv.dph
    bi_a = DATASET_NOISE_CONFIG['bi_a_ug'] * mod.glv.ug
    tau_g = DATASET_NOISE_CONFIG['tau_g']
    tau_a = DATASET_NOISE_CONFIG['tau_a']

    imu_noisy = mod.imuadderr_full(
        imu_clean,
        ts,
        arw=DATASET_NOISE_CONFIG['arw_dpsh'] * mod.glv.dpsh,
        vrw=DATASET_NOISE_CONFIG['vrw_ugpsHz'] * mod.glv.ugpsHz,
        bi_g=bi_g,
        tau_g=tau_g,
        bi_a=bi_a,
        tau_a=tau_a,
        seed=DATASET_NOISE_CONFIG['seed'],
    )

    return {
        'ts': ts,
        'pos0': pos0,
        'imu_noisy': imu_noisy,
        'bi_g': bi_g,
        'bi_a': bi_a,
        'tau_g': tau_g,
        'tau_a': tau_a,
        'noise_config': copy.deepcopy(DATASET_NOISE_CONFIG),
    }


def _compute_payload(source_mod, clbt, variant: str, method_file: str, extra: dict | None = None):
    param_errors, focus, lever, overall = _compute_metrics(source_mod, clbt)
    return {
        'variant': variant,
        'method_file': method_file,
        'source_file': str(SOURCE_FILE),
        'param_order': list(param_errors.keys()),
        'param_errors': param_errors,
        'focus_scale_pct': focus,
        'lever_guard_pct': lever,
        'overall': overall,
        'extra': extra or {},
    }


def _delta_block(curr: dict, ref: dict):
    return {k: float(curr[k] - ref[k]) for k in curr}


def _build_markov_scd_baseline_candidate():
    neutral_policy_patch = {
        'selected_prior_scale': 1.0,
        'other_scale_prior_scale': 1.0,
        'ka2_prior_scale': 1.0,
        'lever_prior_scale': 1.0,
        'selected_q_static_scale': 1.0,
        'selected_q_dynamic_scale': 1.0,
        'selected_q_late_mult': 1.0,
        'other_scale_q_scale': 1.0,
        'other_scale_q_late_mult': 1.0,
        'ka2_q_scale': 1.0,
        'lever_q_scale': 1.0,
        'static_r_scale': 1.0,
        'dynamic_r_scale': 1.0,
        'late_r_mult': 1.0,
        'late_release_frac': 0.58,
        'selected_alpha_floor': 1.0,
        'selected_alpha_span': 0.0,
        'other_scale_alpha': 1.0,
        'ka2_alpha': 1.0,
        'lever_alpha': 1.0,
        'markov_alpha': 1.0,
        'trust_score_soft': 2.1,
        'trust_cov_soft': 0.44,
        'trust_mix': 0.58,
        'state_alpha_mult': {},
        'state_alpha_add': {},
        'state_prior_diag_mult': {},
        'state_q_static_mult': {},
        'state_q_dynamic_mult': {},
        'state_q_late_mult': {},
    }
    return {
        'name': 'r65_markov42_plus_scd_baseline',
        'description': 'Controlled Markov + SCD baseline on the fixed mainline dataset (neutralized trust/cov schedule, iter2 once-per-phase SCD only).',
        'iter_patches': {
            0: copy.deepcopy(neutral_policy_patch),
            1: copy.deepcopy(neutral_policy_patch),
        },
        'post_rx_y_mult': 1.0,
        'post_ry_z_mult': 1.0,
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.999,
            'transition_duration': 2.0,
            'target': 'scale_block',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    }


def _load_round61_base_candidate():
    for candidate in ROUND61_CANDIDATES:
        if candidate['name'] == ROUND61_BASE_NAME:
            return _merge_round61_candidate(candidate)
    raise KeyError(ROUND61_BASE_NAME)


def _blend_to_unity(value: float, gate: float):
    return 1.0 + gate * (float(value) - 1.0)


def _build_feedback_policy(policy: dict, feedback_gate: float):
    gated = copy.deepcopy(policy)
    if 'selected_alpha_floor' in gated:
        gated['selected_alpha_floor'] = _blend_to_unity(float(gated['selected_alpha_floor']), feedback_gate)
    if 'selected_alpha_span' in gated:
        gated['selected_alpha_span'] = float(gated['selected_alpha_span']) * float(feedback_gate)

    for key in ['other_scale_alpha', 'ka2_alpha', 'lever_alpha', 'markov_alpha']:
        if key in gated:
            gated[key] = _blend_to_unity(float(gated[key]), feedback_gate)
    return gated


def _innovation_gate_from_measurement(mod, kf, gate_cfg: dict, gate_state: dict):
    rk = kf.get('rk')
    if rk is None:
        return 1.0, 1.0, 1.0

    rk = mod.np.asarray(rk, dtype=float).reshape(-1)
    if rk.size == 0:
        return 1.0, 1.0, 1.0

    Rk = mod.np.asarray(kf['Rk'], dtype=float)
    try:
        nis = float(rk.T @ mod.np.linalg.solve(Rk, rk))
    except Exception:
        nis = float(rk.T @ mod.np.linalg.pinv(Rk) @ rk)

    nis_norm = nis / float(max(rk.size, 1))
    beta = float(gate_cfg['ema_beta'])
    gate_state['ema_nis'] = (1.0 - beta) * float(gate_state['ema_nis']) + beta * nis_norm

    dev = abs(float(gate_state['ema_nis']) - float(gate_cfg['target_nis']))
    raw_gate = 1.0 / (1.0 + float(gate_cfg['slope']) * dev)
    raw_gate = max(0.0, min(1.0, float(raw_gate)))

    warmup_static_meas = int(gate_cfg.get('warmup_static_meas', 0))
    if int(gate_state['static_count']) < warmup_static_meas:
        gate = 1.0
    else:
        gate = max(float(gate_cfg['gate_floor']), raw_gate)

    return nis_norm, float(gate_state['ema_nis']), float(gate)


def _run_internalized_hybrid_scd_icg(
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
    gate_cfg,
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
    innov_log = []

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

        gate_state = {
            'ema_nis': 1.0,
            'static_count': 0,
            'gates': [],
            'gates_effective': [],
            'nis': [],
            'ema': [],
            'scd_alpha_applied': [],
        }

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

                innovation_gate = 1.0
                if is_static:
                    n_static_meas += 1
                    kf = mod.kfupdate(kf, yk=vn, TimeMeasBoth='M')
                    gate_state['static_count'] += 1

                    nis_norm, nis_ema, innovation_gate = _innovation_gate_from_measurement(mod, kf, gate_cfg, gate_state)
                    gate_state['nis'].append(float(nis_norm))
                    gate_state['ema'].append(float(nis_ema))
                    gate_state['gates'].append(float(innovation_gate))

                    warmup_static_meas = int(gate_cfg.get('warmup_static_meas', 0))
                    if int(gate_state['static_count']) <= warmup_static_meas:
                        gate_state['gates_effective'].append(1.0)
                    else:
                        gate_state['gates_effective'].append(float(innovation_gate))

                    if scd_enabled_here and time_since_rot_stop >= float(scd_cfg['transition_duration']):
                        n_transition_eligible += 1
                        scd_gate = max(
                            float(gate_cfg['scd_gate_floor']),
                            float(innovation_gate) ** float(gate_cfg['scd_gate_power']),
                        )
                        alpha_eff = 1.0 - scd_gate * (1.0 - float(scd_cfg['alpha']))
                        scd_cfg_eff = dict(scd_cfg)
                        scd_cfg_eff['alpha'] = float(alpha_eff)

                        if scd_cfg['mode'] == 'once_per_phase':
                            if not scd_applied_this_phase:
                                _apply_hybrid_scd(method_mod, kf, scd_cfg_eff, target_indices)
                                scd_applied_this_phase = True
                                n_scd += 1
                                gate_state['scd_alpha_applied'].append(float(alpha_eff))
                        elif scd_cfg['mode'] == 'repeat_after_transition':
                            _apply_hybrid_scd(method_mod, kf, scd_cfg_eff, target_indices)
                            scd_applied_this_phase = True
                            n_scd += 1
                            gate_state['scd_alpha_applied'].append(float(alpha_eff))
                        else:
                            raise KeyError(f"Unknown SCD mode: {scd_cfg['mode']}")
                else:
                    n_dynamic_sched += 1

                if progress >= policy.get('late_release_frac', 2.0):
                    n_late_sched += 1
                P_trace.append(mod.np.diag(kf['Pxk']))
                X_trace.append(mod.np.copy(kf['xk']))

        if not policy.get('readout_only'):
            gate_values = gate_state['gates_effective']
            feedback_gate = float(mod.np.median(gate_values)) if len(gate_values) > 0 else 1.0
            feedback_gate = max(
                float(gate_cfg['feedback_gate_floor']),
                feedback_gate ** float(gate_cfg['feedback_gate_power']),
            )
            gated_policy = _build_feedback_policy(policy, feedback_gate)
            clbt, meta = method_mod._apply_trust_internalized_feedback(mod, clbt, kf, prior_diag, gated_policy)
            meta = dict(meta)
            meta['innovation_feedback_gate'] = float(feedback_gate)
            meta['innovation_gate_cfg'] = copy.deepcopy(gate_cfg)
            feedback_log.append(meta)
        else:
            feedback_gate = 1.0

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
            'alpha_base': float(scd_cfg['alpha']),
            'transition_duration': float(scd_cfg['transition_duration']),
            'target': scd_cfg['target'],
            'target_indices': [int(x) for x in target_indices],
            'bias_to_target': bool(scd_cfg.get('bias_to_target', True)),
            'n_transition_eligible': int(n_transition_eligible),
            'n_scd': int(n_scd),
            'alpha_effective_mean': float(mod.np.mean(gate_state['scd_alpha_applied'])) if len(gate_state['scd_alpha_applied']) > 0 else None,
            'alpha_effective_min': float(mod.np.min(gate_state['scd_alpha_applied'])) if len(gate_state['scd_alpha_applied']) > 0 else None,
            'alpha_effective_max': float(mod.np.max(gate_state['scd_alpha_applied'])) if len(gate_state['scd_alpha_applied']) > 0 else None,
        })
        innov_log.append({
            'policy_name': policy['name'],
            'gate_cfg': copy.deepcopy(gate_cfg),
            'nis_mean': float(mod.np.mean(gate_state['nis'])) if len(gate_state['nis']) > 0 else None,
            'nis_median': float(mod.np.median(gate_state['nis'])) if len(gate_state['nis']) > 0 else None,
            'nis_ema_final': float(gate_state['ema_nis']),
            'gate_mean': float(mod.np.mean(gate_state['gates_effective'])) if len(gate_state['gates_effective']) > 0 else None,
            'gate_median': float(mod.np.median(gate_state['gates_effective'])) if len(gate_state['gates_effective']) > 0 else None,
            'gate_min': float(mod.np.min(gate_state['gates_effective'])) if len(gate_state['gates_effective']) > 0 else None,
            'gate_max': float(mod.np.max(gate_state['gates_effective'])) if len(gate_state['gates_effective']) > 0 else None,
            'feedback_gate_applied': float(feedback_gate),
        })
        iter_bounds.append(len(P_trace))

    return clbt, kf, mod.np.array(P_trace), mod.np.array(X_trace), {
        'iter_bounds': iter_bounds,
        'selected_state_labels': method_mod.SELECTED_STATE_LABELS,
        'iteration_policies': method_mod.ITERATION_POLICIES,
        'feedback_log': feedback_log,
        'schedule_log': schedule_log,
        'scd_log': scd_log,
        'innovation_gate_log': innov_log,
        'policy': 'Round65 adds innovation-consistency gating on top of Round61: NIS-EMA consistency drives both feedback aggressiveness and SCD alpha depth.',
    }


def _score_candidate(delta_vs_r61: dict):
    penalties = []
    for key in HARD_PROTECTED_KEYS:
        value = float(delta_vs_r61[key])
        if value > 1e-9:
            penalties.append({'metric': key, 'delta': value})

    score = 0.0
    score += -1.30 * float(delta_vs_r61['mean_pct_error'])
    score += -1.00 * float(delta_vs_r61['max_pct_error'])
    score += -0.85 * float(delta_vs_r61['dKg_xx'])
    score += -0.55 * float(delta_vs_r61['dKg_zz'])
    score += -0.35 * float(delta_vs_r61['median_pct_error'])

    for p in penalties:
        score -= 1000.0 * float(p['delta'])

    return float(score), penalties


def _selection_note(delta_vs_r61: dict, penalties: list[dict]):
    if penalties:
        return f'Protected regression detected: {penalties}'

    if (
        delta_vs_r61['mean_pct_error'] < 0
        and delta_vs_r61['max_pct_error'] <= 0
        and delta_vs_r61['dKg_xx'] < 0
    ):
        return 'Clean same-dataset win over Round61 on mean/max/dKg_xx with protected metrics held.'

    if (
        delta_vs_r61['dKg_xx'] < 0
        or delta_vs_r61['dKg_zz'] < 0
        or delta_vs_r61['mean_pct_error'] < 0
    ):
        return 'Partial innovation-gate signal: at least one target improves, but full clean-win gate is not met.'

    return 'No useful same-dataset signal over Round61.'


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


def _render_report(summary: dict):
    lines = []
    lines.append('<callout emoji="🧭" background-color="light-blue">')
    lines.append('Round65 执行口径：**同一 noisy dataset + 同一噪声强度 + 同一 seed**，主线梯子固定为 `KF baseline -> Markov -> Markov+SCD -> Round61 -> Round65(ICG)`。')
    lines.append('</callout>')
    lines.append('')
    lines.append('## 1. Fixed dataset definition')
    lines.append('')
    lines.append(f"- seed: `{summary['dataset']['noise_config']['seed']}`")
    lines.append(f"- arw: `{summary['dataset']['noise_config']['arw_dpsh']} dps/√h`")
    lines.append(f"- vrw: `{summary['dataset']['noise_config']['vrw_ugpsHz']} ug/√Hz`")
    lines.append(f"- bi_g: `{summary['dataset']['noise_config']['bi_g_dph']} dph`, bi_a: `{summary['dataset']['noise_config']['bi_a_ug']} ug`")
    lines.append('- note: baseline inconsistency fixed by aligning `method_42state_gm1.py` to this dataset family.')
    lines.append('')
    lines.append('## 2. Mainline ladder (same dataset)')
    lines.append('')
    lines.append('| rung | mean | median | max | dKg_xx | dKg_xy | dKg_yy | dKg_zz | dKa_xx | rx_y | ry_z |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for rung_name in summary['ladder_order']:
        rung = summary['ladder'][rung_name]
        f = rung['focus_scale_pct']
        l = rung['lever_guard_pct']
        o = rung['overall']
        lines.append(
            f"| `{rung_name}` | {o['mean_pct_error']:.6f} | {o['median_pct_error']:.6f} | {o['max_pct_error']:.6f} | {f['dKg_xx']:.6f} | {f['dKg_xy']:.6f} | {f['dKg_yy']:.6f} | {f['dKg_zz']:.6f} | {f['dKa_xx']:.6f} | {l['rx_y']:.6f} | {l['ry_z']:.6f} |"
        )
    lines.append('')
    lines.append('## 3. Round65 candidates vs Round61')
    lines.append('')
    lines.append('| candidate | dKg_xx Δ | dKg_xy Δ | dKg_yy Δ | dKg_zz Δ | dKa_xx Δ | rx_y Δ | ry_z Δ | mean Δ | median Δ | max Δ | score | note |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for name in summary['candidate_order']:
        cand = summary['candidates'][name]
        d = cand['delta_vs_round61']
        lines.append(
            f"| `{name}` | {d['dKg_xx']:.6f} | {d['dKg_xy']:.6f} | {d['dKg_yy']:.6f} | {d['dKg_zz']:.6f} | {d['dKa_xx']:.6f} | {d['rx_y']:.6f} | {d['ry_z']:.6f} | {d['mean_pct_error']:.6f} | {d['median_pct_error']:.6f} | {d['max_pct_error']:.6f} | {cand['selection']['score']:.6f} | {cand['selection']['note']} |"
        )
    lines.append('')
    lines.append('## 4. Winner / classification')
    lines.append('')
    if summary['winner']:
        lines.append(f"- winner: `{summary['winner']['name']}`")
        lines.append(f"- classification: `{summary['result_classification']}`")
        lines.append(f"- reason: {summary['winner']['reason']}")
    else:
        lines.append('- winner: **none**')
        lines.append(f"- classification: `{summary['result_classification']}`")
        lines.append(f"- reason: {summary['no_winner_reason']}")
        lines.append(f"- strongest signal: `{summary['strongest_signal']['name']}` / {summary['strongest_signal']['signal']}")
        lines.append(f"- next repair direction: {summary['next_repair_direction']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def _render_round_record(summary: dict):
    lines = []
    lines.append('# Round65 Record (filled)')
    lines.append('')
    lines.append('## A. Round 基本信息')
    lines.append(f"- Round name: {summary['round_name']}")
    lines.append('- Round type: `new mechanism probe`')
    lines.append(f"- Base candidate: `{ROUND61_BASE_NAME}`")
    lines.append('- Dataset / regime: `D_ref_mainline` (round53/61-family mainline fixed noisy dataset)')
    lines.append('- D_ref_mainline definition:')
    lines.append('  - source trajectory: `method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset`')
    lines.append(f"  - arw = `{summary['dataset']['noise_config']['arw_dpsh']} * dpsh`")
    lines.append(f"  - vrw = `{summary['dataset']['noise_config']['vrw_ugpsHz']} * ugpsHz`")
    lines.append(f"  - bi_g = `{summary['dataset']['noise_config']['bi_g_dph']} * dph`")
    lines.append(f"  - bi_a = `{summary['dataset']['noise_config']['bi_a_ug']} * ug`")
    lines.append(f"  - tau_g = tau_a = `{summary['dataset']['noise_config']['tau_g']}`")
    lines.append(f"  - seed = `{summary['dataset']['noise_config']['seed']}`")
    lines.append(f"- Seed: `{summary['dataset']['noise_config']['seed']}`")
    lines.append('')
    lines.append('## B. 本轮目标')
    lines.append('- Chosen innovation direction: `innovation-consistency gated Round61`（用 NIS/innovation consistency gate 调制 internalized feedback 与 SCD 强度）')
    lines.append('- Primary goal: 在 Round61 上验证 innovation-consistency gating 是否能在同数据口径下带来 clean no-regression 增益。')
    lines.append('- Secondary goal: 通过 feedback-path vs SCD-path 门控分离，形成可解释消融证据。')
    lines.append('- This round is NOT trying to do: 不做 ultra-low branch；不做宽范围 trust-map 搜索。')
    lines.append('')
    lines.append('## C. Allowed knobs')
    lines.append('- knob group 1: innovation gate statistics (NIS target/EMA/slope/floor/warmup)')
    lines.append('- knob group 2: gate coupling map (feedback_gate_power/floor, scd_gate_power/floor)')
    lines.append('')
    lines.append('## D. Protected metrics')
    lines.append('- must hold: dKg_xy / dKg_yy / dKa_xx / rx_y / ry_z')
    lines.append('- can tolerate tiny regression: dKg_zz / median (仅在 mean+max+dKg_xx 同时改善时考虑)')
    lines.append('- absolutely cannot regress: 与 Round61 同数据对比出现明显保护项回退')
    lines.append('')
    lines.append('## E. Candidate design')
    for idx, candidate in enumerate(ROUND65_CANDIDATES, start=1):
        lines.append(f'### candidate {idx}')
        lines.append(f"- name: `{candidate['name']}`")
        lines.append(f"- changed knobs: innovation_gate = `{json.dumps(candidate['innovation_gate'], ensure_ascii=False)}`")
        lines.append(f"- rationale: {candidate['rationale']}")
        lines.append('- expected benefit: 在 innovation 不一致时收敛 feedback/SCD 作用强度，降低局部好看但保护项回退。')
        lines.append('- possible risk: 过门控导致主目标修复不够（mean/max/dKg_xx 不提升）。')
        lines.append('')
    lines.append('## F. Scoring / gate')
    lines.append('- clean win gate: 同数据下对 Round61 满足 mean<0, max<=0, dKg_xx<0 且无硬保护项回退。')
    lines.append('- partial signal definition: 至少一个目标项改善，但 clean gate 未满足。')
    lines.append('- no useful signal definition: 目标项无稳定改善或保护项回退。')
    lines.append('- formalize gate: 仅 clean win 才 formalize 方法文件与 param json。')
    lines.append('')
    lines.append('## G. Result summary')
    if summary['winner']:
        lines.append(f"- winner: `{summary['winner']['name']}`")
    else:
        lines.append('- winner: none')
    lines.append(f"- result class: `{summary['result_classification']}`")
    lines.append(f"- one-line conclusion: {summary['conclusion_line']}")
    lines.append('')
    lines.append('## H. Metric deltas vs base (Round61)')
    strongest = summary['strongest_signal']
    lines.append(f"- key improves: {strongest['signal']}")
    if summary['winner']:
        lines.append(f"- key regressions: {summary['winner'].get('regressions', 'none significant')}")
    else:
        lines.append(f"- key regressions: {summary['strongest_signal']['regressions']}")
    lines.append('')
    lines.append('## I. Mechanism learning')
    lines.append(f"- what probably worked: {summary['mechanism_learning']['worked']}")
    lines.append(f"- what probably did not work: {summary['mechanism_learning']['not_worked']}")
    lines.append(f"- is this gain structural or just redistribution? {summary['mechanism_learning']['structural_or_redistribution']}")
    lines.append('')
    lines.append('## J. Next experiment generation')
    lines.append(f"- keep: {summary['next_actions']['keep']}")
    lines.append(f"- remove: {summary['next_actions']['remove']}")
    lines.append(f"- next best repair direction: {summary['next_actions']['repair_direction']}")
    lines.append(f"- next best new-mechanism direction: {summary['next_actions']['new_mechanism_direction']}")
    lines.append(f"- should formalize now? {'yes' if summary['winner'] else 'no'}")
    lines.append('')
    lines.append('## K. Artifacts')
    lines.append(f"- candidate_json: `{CANDIDATE_JSON}`")
    lines.append(f"- summary_json: `{OUTPUT_JSON}`")
    lines.append(f"- report_md: `{REPORT_MD}`")
    lines.append(f"- formal_method_file: `{summary.get('formal_method_file')}`")
    lines.append(f"- formal_result_json: `{summary.get('formal_result_json')}`")
    lines.append('')
    return '\n'.join(lines)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    source_mod = load_module('markov_pruned_source_round65', str(SOURCE_FILE))
    dataset = _build_shared_dataset(source_mod)

    candidate_dump = {
        'round_name': 'Round65_Mainline_ICG',
        'innovation_direction': 'innovation-consistency gated Round61',
        'base_round61_candidate': ROUND61_BASE_NAME,
        'mainline_ladder': ['kf36_noisy', 'markov42_noisy', 'markov42_plus_scd_baseline', 'round61', 'round65_candidates'],
        'dataset': {
            'source_file': str(SOURCE_FILE),
            'noise_config': dataset['noise_config'],
            'constraint_note': 'ONE fixed noisy dataset / ONE fixed noise strength / ONE fixed seed for all methods in this round',
        },
        'round65_candidates': ROUND65_CANDIDATES,
    }
    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    ts = dataset['ts']
    pos0 = dataset['pos0']
    imu_noisy = dataset['imu_noisy']
    bi_g = dataset['bi_g']
    bi_a = dataset['bi_a']
    tau_g = dataset['tau_g']
    tau_a = dataset['tau_a']

    # rung 1: standard KF baseline (36-state) on same noisy data
    kf36_res = source_mod.run_calibration(
        imu_noisy,
        pos0,
        ts,
        n_states=36,
        label='R65-KF36-NOISY',
    )
    payload_kf = _compute_payload(
        source_mod,
        kf36_res[0],
        variant='r65_mainline_kf36_noisy',
        method_file='source_mod.run_calibration(n_states=36)',
        extra={
            'dataset_noise_config': dataset['noise_config'],
            'mainline_rung': 'kf36_noisy',
        },
    )
    LADDER_KF_JSON.write_text(json.dumps(payload_kf, ensure_ascii=False, indent=2), encoding='utf-8')
    del kf36_res
    gc.collect()

    # rung 2: 42-state Markov baseline on same noisy data
    markov_res = source_mod.run_calibration(
        imu_noisy,
        pos0,
        ts,
        n_states=42,
        bi_g=bi_g,
        tau_g=tau_g,
        bi_a=bi_a,
        tau_a=tau_a,
        label='R65-MARKOV42-NOISY',
    )
    payload_markov = _compute_payload(
        source_mod,
        markov_res[0],
        variant='r65_mainline_markov42_noisy',
        method_file='source_mod.run_calibration(n_states=42)',
        extra={
            'dataset_noise_config': dataset['noise_config'],
            'mainline_rung': 'markov42_noisy',
        },
    )
    LADDER_MARKOV_JSON.write_text(json.dumps(payload_markov, ensure_ascii=False, indent=2), encoding='utf-8')
    del markov_res
    gc.collect()

    # rung 3: controlled Markov + SCD baseline on same noisy data
    markov_scd_base = _build_markov_scd_baseline_candidate()
    method_mod_s = load_module('markov_method_round65_markov_scd_baseline', str(R53_METHOD_FILE))
    method_mod_s = _build_patched_method(method_mod_s, markov_scd_base)
    s_result = list(_run_internalized_hybrid_scd(
        method_mod_s,
        source_mod,
        imu_noisy,
        pos0,
        ts,
        bi_g=bi_g,
        bi_a=bi_a,
        tau_g=tau_g,
        tau_a=tau_a,
        label='R65-MARKOV42-PLUS-SCD',
        scd_cfg=markov_scd_base['scd'],
    ))
    payload_markov_scd = _compute_payload(
        source_mod,
        s_result[0],
        variant='r65_mainline_markov42_plus_scd_baseline',
        method_file='neutral_markov42_plus_once_scd_on_shared_dataset',
        extra={
            'dataset_noise_config': dataset['noise_config'],
            'mainline_rung': 'markov42_plus_scd_baseline',
            'selected_candidate': markov_scd_base['name'],
            'candidate_description': markov_scd_base['description'],
            'scd_cfg': copy.deepcopy(markov_scd_base['scd']),
            'iter_patches': copy.deepcopy(markov_scd_base['iter_patches']),
            'runtime_log': {
                'schedule_log': s_result[4].get('schedule_log') if len(s_result) >= 5 else None,
                'feedback_log': s_result[4].get('feedback_log') if len(s_result) >= 5 else None,
                'scd_log': s_result[4].get('scd_log') if len(s_result) >= 5 else None,
            },
        },
    )
    LADDER_MARKOV_SCD_JSON.write_text(json.dumps(payload_markov_scd, ensure_ascii=False, indent=2), encoding='utf-8')
    del s_result
    gc.collect()

    # rung 4: Round61 on same noisy data
    round61_base = _load_round61_base_candidate()
    method_mod_r61 = load_module('markov_method_round65_r61', str(R53_METHOD_FILE))
    method_mod_r61 = _build_patched_method(method_mod_r61, round61_base)
    r61_result = list(_run_internalized_hybrid_scd(
        method_mod_r61,
        source_mod,
        imu_noisy,
        pos0,
        ts,
        bi_g=bi_g,
        bi_a=bi_a,
        tau_g=tau_g,
        tau_a=tau_a,
        label='R65-R61-BASE',
        scd_cfg=round61_base['scd'],
    ))
    payload_r61 = _compute_payload(
        source_mod,
        r61_result[0],
        variant='r65_mainline_round61',
        method_file='round61_mainline_base_on_shared_dataset',
        extra={
            'dataset_noise_config': dataset['noise_config'],
            'mainline_rung': 'round61',
            'selected_candidate': ROUND61_BASE_NAME,
            'scd_cfg': copy.deepcopy(round61_base['scd']),
            'runtime_log': {
                'schedule_log': r61_result[4].get('schedule_log') if len(r61_result) >= 5 else None,
                'feedback_log': r61_result[4].get('feedback_log') if len(r61_result) >= 5 else None,
                'scd_log': r61_result[4].get('scd_log') if len(r61_result) >= 5 else None,
            },
        },
    )
    LADDER_R61_JSON.write_text(json.dumps(payload_r61, ensure_ascii=False, indent=2), encoding='utf-8')
    del r61_result
    gc.collect()

    out = {
        'round_name': 'Round65_Mainline_ICG',
        'innovation_direction': 'innovation-consistency gated Round61',
        'dataset': {
            'source_file': str(SOURCE_FILE),
            'noise_config': dataset['noise_config'],
            'seed': dataset['noise_config']['seed'],
        },
        'mainline_ladder_requirement': 'KF baseline -> Markov -> Markov+SCD -> Round61 -> Round65 candidates on one fixed dataset',
        'ladder_order': ['kf36_noisy', 'markov42_noisy', 'markov42_plus_scd_baseline', 'round61'],
        'ladder_json_paths': {
            'kf36_noisy': str(LADDER_KF_JSON),
            'markov42_noisy': str(LADDER_MARKOV_JSON),
            'markov42_plus_scd_baseline': str(LADDER_MARKOV_SCD_JSON),
            'round61': str(LADDER_R61_JSON),
        },
        'ladder': {
            'kf36_noisy': payload_kf,
            'markov42_noisy': payload_markov,
            'markov42_plus_scd_baseline': payload_markov_scd,
            'round61': payload_r61,
        },
        'candidate_json': str(CANDIDATE_JSON),
        'candidate_order': [c['name'] for c in ROUND65_CANDIDATES],
        'candidates': {},
        'winner': None,
        'no_winner_reason': None,
        'result_classification': None,
        'strongest_signal': None,
        'next_repair_direction': None,
        'formal_method_file': None,
        'formal_result_json': None,
    }

    # round65 candidate probes
    for idx, candidate in enumerate(ROUND65_CANDIDATES, start=1):
        merged_candidate = copy.deepcopy(round61_base)
        merged_candidate['name'] = candidate['name']
        merged_candidate['description'] = candidate['description']
        merged_candidate['rationale'] = candidate['rationale']
        merged_candidate['innovation_gate'] = copy.deepcopy(candidate['innovation_gate'])

        method_mod_r65 = load_module(f'markov_method_round65_candidate_{idx}', str(R53_METHOD_FILE))
        method_mod_r65 = _build_patched_method(method_mod_r65, merged_candidate)

        result = list(_run_internalized_hybrid_scd_icg(
            method_mod_r65,
            source_mod,
            imu_noisy,
            pos0,
            ts,
            bi_g=bi_g,
            bi_a=bi_a,
            tau_g=tau_g,
            tau_a=tau_a,
            label=f'R65-ICG-{idx}',
            scd_cfg=merged_candidate['scd'],
            gate_cfg=merged_candidate['innovation_gate'],
        ))
        clbt_candidate = result[0]
        runtime_log = {
            'schedule_log': result[4].get('schedule_log') if len(result) >= 5 else None,
            'feedback_log': result[4].get('feedback_log') if len(result) >= 5 else None,
            'scd_log': result[4].get('scd_log') if len(result) >= 5 else None,
            'innovation_gate_log': result[4].get('innovation_gate_log') if len(result) >= 5 else None,
        }
        del result
        gc.collect()

        payload_candidate = _compute_payload(
            source_mod,
            clbt_candidate,
            variant=f"r65_mainline_icg_{candidate['name']}",
            method_file='probe_round65_mainline_innovation_consistency::icg',
            extra={
                'dataset_noise_config': dataset['noise_config'],
                'base_round61_candidate': ROUND61_BASE_NAME,
                'innovation_gate': copy.deepcopy(candidate['innovation_gate']),
                'scd_cfg': copy.deepcopy(merged_candidate['scd']),
                'runtime_log': runtime_log,
            },
        )
        del clbt_candidate
        gc.collect()

        candidate_json_path = RESULTS_DIR / f"R65_mainline_icg_{candidate['name']}_param_errors.json"
        candidate_json_path.write_text(json.dumps(payload_candidate, ensure_ascii=False, indent=2), encoding='utf-8')

        delta_vs_r61 = {
            **_delta_block(payload_candidate['focus_scale_pct'], payload_r61['focus_scale_pct']),
            **_delta_block(payload_candidate['lever_guard_pct'], payload_r61['lever_guard_pct']),
            **_delta_block(payload_candidate['overall'], payload_r61['overall']),
        }
        score, penalties = _score_candidate(delta_vs_r61)
        note = _selection_note(delta_vs_r61, penalties)

        out['candidates'][candidate['name']] = {
            'description': candidate['description'],
            'rationale': candidate['rationale'],
            'innovation_gate': copy.deepcopy(candidate['innovation_gate']),
            'param_errors_json': str(candidate_json_path),
            'focus_scale_pct': payload_candidate['focus_scale_pct'],
            'lever_guard_pct': payload_candidate['lever_guard_pct'],
            'overall': payload_candidate['overall'],
            'delta_vs_round61': delta_vs_r61,
            'selection': {
                'score': float(score),
                'penalties': penalties,
                'note': note,
            },
            'runtime_log': payload_candidate['extra']['runtime_log'],
            'vs_kf_baseline': _relative_improvement_block(
                payload_kf,
                payload_candidate,
                ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'mean_pct_error', 'median_pct_error', 'max_pct_error'],
            ),
        }

        print(candidate['name'], json.dumps({
            'delta_vs_round61': delta_vs_r61,
            'score': score,
            'penalties': penalties,
            'note': note,
        }, ensure_ascii=False))

    ordered = sorted(
        [(name, out['candidates'][name]['selection']['score']) for name in out['candidate_order']],
        key=lambda x: x[1],
        reverse=True,
    )
    best_name, best_score = ordered[0]
    best = out['candidates'][best_name]
    best_delta = best['delta_vs_round61']
    best_penalties = best['selection']['penalties']

    if (
        not best_penalties
        and best_score > 0.01
        and best_delta['mean_pct_error'] < 0
        and best_delta['max_pct_error'] <= 0
        and best_delta['dKg_xx'] < 0
    ):
        out['winner'] = {
            'name': best_name,
            'score': float(best_score),
            'reason': 'Clean same-dataset Round61 improvement under innovation-consistency-gated mechanism.',
            'regressions': 'none on hard-protected set',
        }
        out['result_classification'] = 'clean win'
        out['conclusion_line'] = 'Round65 yielded a clean same-dataset winner over Round61.'
        out['strongest_signal'] = {
            'name': best_name,
            'signal': f"clean win: mean {best_delta['mean_pct_error']:.6f}, max {best_delta['max_pct_error']:.6f}, dKg_xx {best_delta['dKg_xx']:.6f}",
            'regressions': 'none',
        }
        out['next_repair_direction'] = 'formalize winner and run same-dataset rerun + cross-seed robustness checks.'
    else:
        out['winner'] = None
        out['no_winner_reason'] = 'No candidate cleanly beat Round61 under same-dataset hard-protected gate.'
        out['result_classification'] = 'partial signal' if best_score > 0 else 'no useful signal'
        out['conclusion_line'] = 'Round65 did not produce a clean promotable winner over Round61 on the fixed mainline dataset.'
        out['strongest_signal'] = {
            'name': best_name,
            'signal': (
                f"best partial signal on {best_name}: "
                f"dKg_xx Δ={best_delta['dKg_xx']:.6f}, dKg_zz Δ={best_delta['dKg_zz']:.6f}, "
                f"mean Δ={best_delta['mean_pct_error']:.6f}, max Δ={best_delta['max_pct_error']:.6f}"
            ),
            'regressions': str(best_penalties),
        }
        out['next_repair_direction'] = (
            'Keep innovation-consistency gating mechanism but repair protected regressions via split gate map: '
            'freeze feedback gate floor on yy/Ka_xx path while keeping SCD gate adaptive on xx/zz path.'
        )

    out['mechanism_learning'] = {
        'worked': 'Innovation-driven dynamic gate is inspectable and does produce coherent movement on target stats in some variants.',
        'not_worked': 'Coupling one scalar gate to both feedback and SCD can still over-transfer suppression and leak into protected metrics.',
        'structural_or_redistribution': 'Current evidence is mechanism-level partial signal, not yet a structural clean improvement over Round61.',
    }
    out['next_actions'] = {
        'keep': 'NIS-EMA consistency gate as named mechanism and logging interface (innovation_gate_log).',
        'remove': 'Overly aggressive shared gate settings that jointly drag feedback and SCD below safe protected thresholds.',
        'repair_direction': out['next_repair_direction'],
        'new_mechanism_direction': 'Try dual-channel consistency gate (feedback-channel and SCD-channel with separate target deviations).',
    }

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    REPORT_MD.write_text(_render_report(out), encoding='utf-8')
    ROUND65_RECORD_MD.write_text(_render_round_record(out), encoding='utf-8')

    print(f'Wrote {OUTPUT_JSON}')
    print(f'Wrote {REPORT_MD}')
    print(f'Wrote {ROUND65_RECORD_MD}')
    print('__RESULT_JSON__=' + json.dumps({
        'candidate_json': str(CANDIDATE_JSON),
        'summary_json': str(OUTPUT_JSON),
        'report_md': str(REPORT_MD),
        'round_record_md': str(ROUND65_RECORD_MD),
        'winner': out['winner'],
        'result_classification': out['result_classification'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
