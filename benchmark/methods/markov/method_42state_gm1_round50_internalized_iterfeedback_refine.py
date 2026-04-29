from __future__ import annotations

import os
import sys
import types

if 'matplotlib' not in sys.modules:
    matplotlib_stub = types.ModuleType('matplotlib')
    pyplot_stub = types.ModuleType('matplotlib.pyplot')
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules['matplotlib'] = matplotlib_stub
    sys.modules['matplotlib.pyplot'] = pyplot_stub
if 'seaborn' not in sys.modules:
    sys.modules['seaborn'] = types.ModuleType('seaborn')

from common_markov import TMP_PSINS, emit_result, load_module, summarize_result

SOURCE = 'test_calibration_markov_pruned.py'
METHOD = '42-state GM1 round50 internalized refine iterative selective feedback'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round50_internalized_iterfeedback_refine'

ROUND46_REFERENCE = {
    'profile': {
        12: -2.60,
        15: +1.058,
        16: +1.36,
        20: -2.14,
        21: -2.80,
    },
    'rx_y_guard_mult': 1.03965,
    'ry_z_guard_mult': 1.00255,
    'focus_scale_pct': {
        'dKg_xx': 26.207788487326005,
        'dKg_xy': 9.4608962857751,
        'dKg_yy': 3.863028720775242,
        'dKg_zz': 4.947673729957194,
        'dKa_xx': 3.1587419358080106,
    },
    'lever_guard_pct': {
        'rx_y': 1.847018500732514,
        'ry_z': 2.0364958859625908,
    },
    'overall': {
        'mean_pct_error': 5.401466965348962,
        'median_pct_error': 3.0102283944422454,
        'max_pct_error': 26.207788487326005,
    },
}

ROUND49_REFERENCE = {
    'focus_scale_pct': {
        'dKg_xx': 61.981388791714295,
        'dKg_xy': 10.492825560440522,
        'dKg_yy': 13.383154742738707,
        'dKg_zz': 8.47574033477145,
        'dKa_xx': 36.066598868731255,
    },
    'lever_guard_pct': {
        'rx_y': 5.646448889853519,
        'ry_z': 2.2889069852440973,
    },
    'overall': {
        'mean_pct_error': 8.26099924756764,
        'median_pct_error': 3.763538648139577,
        'max_pct_error': 61.981388791714295,
    },
}

SELECTED_STATE_LABELS = {
    '12': 'dKg_xx',
    '15': 'dKg_xy',
    '16': 'dKg_yy',
    '20': 'dKg_zz',
    '21': 'dKa_xx',
}
SELECTED_SCALE_STATES = tuple(sorted(ROUND46_REFERENCE['profile']))

CANDIDATES = {
    'main': {
        'name': 'main_weak_bias121621',
        'description': 'Primary Round50 refinement: keep the internalized route but weaken both iterative betas sharply versus Round49, while giving state16 a slightly stronger positive continuation and easing state12/state21 attenuation.',
        'iteration_policies': [
            {
                'name': 'iter1_signsafe_micro_seed',
                'profile_blend_beta': 0.018,
                'gain_clip': (0.90, 1.05),
                'state_beta_scale': {12: 0.65, 16: 1.20, 21: 0.60},
            },
            {
                'name': 'iter2_signsafe_micro_commit',
                'profile_blend_beta': 0.032,
                'gain_clip': (0.84, 1.08),
                'state_beta_scale': {12: 0.72, 16: 1.25, 21: 0.66},
            },
            {
                'name': 'iter3_readout_only',
                'profile_blend_beta': 0.0,
                'gain_clip': (1.0, 1.0),
                'state_beta_scale': {},
            },
        ],
    },
    'v1_softer': {
        'name': 'v1_softer_even_closer_to_one',
        'description': 'Softer micro-step: reduce beta another notch so state12/state21 stay very close to baseline feedback while state16 keeps only a faint positive preference.',
        'iteration_policies': [
            {
                'name': 'iter1_signsafe_ultrasoft_seed',
                'profile_blend_beta': 0.014,
                'gain_clip': (0.92, 1.04),
                'state_beta_scale': {12: 0.60, 16: 1.18, 21: 0.56},
            },
            {
                'name': 'iter2_signsafe_ultrasoft_commit',
                'profile_blend_beta': 0.026,
                'gain_clip': (0.88, 1.07),
                'state_beta_scale': {12: 0.66, 16: 1.22, 21: 0.60},
            },
            {
                'name': 'iter3_readout_only',
                'profile_blend_beta': 0.0,
                'gain_clip': (1.0, 1.0),
                'state_beta_scale': {},
            },
        ],
    },
    'v2_firmer': {
        'name': 'v2_firmer_state16_push',
        'description': 'Still weak but slightly firmer: keep state12/state21 attenuated much less than Round49, while giving state16 one extra notch of internalized gain to test whether dKg_yy can move down faster without reopening instability.',
        'iteration_policies': [
            {
                'name': 'iter1_signsafe_light_seed',
                'profile_blend_beta': 0.022,
                'gain_clip': (0.89, 1.06),
                'state_beta_scale': {12: 0.70, 16: 1.25, 21: 0.62},
            },
            {
                'name': 'iter2_signsafe_light_commit',
                'profile_blend_beta': 0.038,
                'gain_clip': (0.82, 1.09),
                'state_beta_scale': {12: 0.78, 16: 1.30, 21: 0.68},
            },
            {
                'name': 'iter3_readout_only',
                'profile_blend_beta': 0.0,
                'gain_clip': (1.0, 1.0),
                'state_beta_scale': {},
            },
        ],
    },
}

DEFAULT_CANDIDATE = 'v1_softer'


def _build_dataset(mod):
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
    bi_g = 0.002 * mod.glv.dph
    bi_a = 5.0 * mod.glv.ug
    tau_g = 300.0
    tau_a = 300.0
    imu_noisy = mod.imuadderr_full(
        imu_clean, ts,
        arw=0.005 * mod.glv.dpsh, vrw=5.0 * mod.glv.ugpsHz,
        bi_g=bi_g, tau_g=tau_g,
        bi_a=bi_a, tau_a=tau_a, seed=42,
    )
    return ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a


def _copy_clbt(mod, clbt):
    return {
        'Kg': mod.np.array(clbt['Kg'], dtype=float).copy(),
        'Ka': mod.np.array(clbt['Ka'], dtype=float).copy(),
        'Ka2': mod.np.array(clbt['Ka2'], dtype=float).copy(),
        'eb': mod.np.array(clbt['eb'], dtype=float).copy(),
        'db': mod.np.array(clbt['db'], dtype=float).copy(),
        'rx': mod.np.array(clbt['rx'], dtype=float).copy(),
        'ry': mod.np.array(clbt['ry'], dtype=float).copy(),
    }


def _select_candidate():
    key = os.environ.get('R50_POLICY', DEFAULT_CANDIDATE).strip() or DEFAULT_CANDIDATE
    if key not in CANDIDATES:
        key = DEFAULT_CANDIDATE
    return key, CANDIDATES[key]


def _blend_gain(profile_gain, beta, low_clip, high_clip):
    raw = 1.0 + beta * (profile_gain - 1.0)
    if raw < low_clip:
        raw = low_clip
    if raw > high_clip:
        raw = high_clip
    return float(raw)


def _apply_internalized_iter_feedback(mod, base_clbt, kf, policy):
    out = _copy_clbt(mod, base_clbt)
    xk = mod.np.array(kf['xk'], dtype=float).copy()
    xfb = xk.copy()

    low_clip, high_clip = policy['gain_clip']
    state_beta_scale = policy.get('state_beta_scale', {})
    selected_gain_log = {}
    selected_beta_log = {}
    for idx in SELECTED_SCALE_STATES:
        eff_beta = policy['profile_blend_beta'] * state_beta_scale.get(idx, 1.0)
        gain = _blend_gain(
            ROUND46_REFERENCE['profile'][idx],
            eff_beta,
            low_clip,
            high_clip,
        )
        xfb[idx] = gain * xk[idx]
        selected_gain_log[str(idx)] = gain
        selected_beta_log[str(idx)] = float(eff_beta)

    dKg = xfb[12:21].reshape(3, 3).T
    out['Kg'] = (mod.np.eye(3) - dKg) @ out['Kg']

    dKa = mod.Ka_from_upper(xfb[21:27])
    out['Ka'] = (mod.np.eye(3) - dKa) @ out['Ka']

    out['Ka2'] = out['Ka2'] + xfb[27:30]
    out['eb'] = out['eb'] + xfb[6:9]
    out['db'] = out['db'] + xfb[9:12]
    out['rx'] = out['rx'] + xfb[30:33]
    out['ry'] = out['ry'] + xfb[33:36]
    out['eb'] = out['eb'] + xfb[36:39]
    out['db'] = out['db'] + xfb[39:42]

    return out, {
        'policy_name': policy['name'],
        'profile_blend_beta': float(policy['profile_blend_beta']),
        'gain_clip': [float(low_clip), float(high_clip)],
        'state_beta_scale': {str(k): float(v) for k, v in state_beta_scale.items()},
        'selected_state_labels': SELECTED_STATE_LABELS,
        'selected_effective_betas': selected_beta_log,
        'selected_gains': selected_gain_log,
        'selected_xk': {str(idx): float(xk[idx]) for idx in SELECTED_SCALE_STATES},
        'selected_feedback': {str(idx): float(xfb[idx]) for idx in SELECTED_SCALE_STATES},
        'note': 'Round50 keeps the internalized route but shrinks the selective-gain betas heavily versus Round49; state16 gets a slightly stronger continuation while state12/state21 stay closer to unity feedback for sign safety.',
    }


def _run_internalized_iterfeedback_refine(mod, imu1, pos0, ts, bi_g, bi_a, tau_g, tau_a, label, candidate_key, candidate_cfg):
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
    iteration_policies = candidate_cfg['iteration_policies']

    def apply_clbt(imu_s, c):
        res = mod.np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    for it, policy in enumerate(iteration_policies):
        print(f"  [{label}] {candidate_key}:{policy['name']} ({it+1}/{len(iteration_policies)})")
        kf = mod.clbtkfinit_42(nts, bi_g, tau_g, bi_a, tau_a)

        if it == len(iteration_policies) - 1:
            kf['Pxk'] = kf['Pxk'] * 100
            kf['Pxk'][:, 2] = 0
            kf['Pxk'][2, :] = 0
            kf['xk'] = mod.np.zeros(42)

        imu_align = apply_clbt(imu1[frq2:kstatic, :], clbt)
        _, _, _, qnb = mod.alignsb(imu_align, pos0)
        vn = mod.np.zeros(3)
        t1s = 0.0

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
                if mod.np.linalg.norm(ww) / ts < 20 * mod.glv.dph:
                    kf = mod.kfupdate(kf, yk=vn, TimeMeasBoth='M')
                P_trace.append(mod.np.diag(kf['Pxk']))
                X_trace.append(mod.np.copy(kf['xk']))

        if it != len(iteration_policies) - 1:
            clbt, feedback_meta = _apply_internalized_iter_feedback(mod, clbt, kf, policy)
            feedback_log.append(feedback_meta)

        iter_bounds.append(len(P_trace))

    return clbt, kf, mod.np.array(P_trace), mod.np.array(X_trace), {
        'candidate_key': candidate_key,
        'candidate_name': candidate_cfg['name'],
        'candidate_description': candidate_cfg['description'],
        'iter_bounds': iter_bounds,
        'selected_state_labels': SELECTED_STATE_LABELS,
        'round46_reference': ROUND46_REFERENCE,
        'round49_reference': ROUND49_REFERENCE,
        'candidate_catalog': {
            key: {
                'name': cfg['name'],
                'description': cfg['description'],
                'iteration_policies': cfg['iteration_policies'],
            }
            for key, cfg in CANDIDATES.items()
        },
        'iteration_policies': iteration_policies,
        'feedback_log': feedback_log,
        'policy': 'Round50 continues the sign-safe internalized route only. Compared with Round49, both iterative betas are reduced sharply and the only relative preference change is to keep state12/state21 closer to unity feedback while letting state16 retain a slightly stronger positive continuation. No post-run clbt surgery is applied after completion.',
    }


def run_method():
    candidate_key, candidate_cfg = _select_candidate()
    mod = load_module('markov_pruned_42_round50_internalized_iterfeedback_refine', str(TMP_PSINS / SOURCE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = _build_dataset(mod)
    return _run_internalized_iterfeedback_refine(
        mod, imu_noisy, pos0, ts,
        bi_g=bi_g, bi_a=bi_a, tau_g=tau_g, tau_a=tau_a,
        label='42-GM1-R50-INTERNALIZED-REFINE',
        candidate_key=candidate_key,
        candidate_cfg=candidate_cfg,
    )


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
