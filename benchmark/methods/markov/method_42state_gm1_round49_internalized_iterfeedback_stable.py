from __future__ import annotations

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
METHOD = '42-state GM1 round49 internalized stable iterative selective feedback'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round49_internalized_iterfeedback_stable'

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

SELECTED_STATE_LABELS = {
    '12': 'dKg_xx',
    '15': 'dKg_xy',
    '16': 'dKg_yy',
    '20': 'dKg_zz',
    '21': 'dKa_xx',
}
SELECTED_SCALE_STATES = tuple(sorted(ROUND46_REFERENCE['profile']))

ITERATION_POLICIES = [
    {
        'name': 'iter1_signsafe_blend_seed',
        'profile_blend_beta': 0.10,
        'gain_clip': (0.55, 1.12),
    },
    {
        'name': 'iter2_signsafe_blend_commit',
        'profile_blend_beta': 0.16,
        'gain_clip': (0.40, 1.18),
    },
    {
        'name': 'iter3_readout_only',
        'profile_blend_beta': 0.0,
        'gain_clip': (1.0, 1.0),
    },
]


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
    selected_gain_log = {}
    for idx in SELECTED_SCALE_STATES:
        gain = _blend_gain(
            ROUND46_REFERENCE['profile'][idx],
            policy['profile_blend_beta'],
            low_clip,
            high_clip,
        )
        xfb[idx] = gain * xk[idx]
        selected_gain_log[str(idx)] = gain

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
        'selected_state_labels': SELECTED_STATE_LABELS,
        'selected_gains': selected_gain_log,
        'selected_xk': {str(idx): float(xk[idx]) for idx in SELECTED_SCALE_STATES},
        'selected_feedback': {str(idx): float(xfb[idx]) for idx in SELECTED_SCALE_STATES},
        'note': 'Only state12/15/16/20/21 use structured selective gains inside the iteration feedback. All other states keep the baseline clbtkffeedback_pruned-equivalent feedback path.',
    }


def _run_internalized_iterfeedback_stable(mod, imu1, pos0, ts, bi_g, bi_a, tau_g, tau_a, label):
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

    def apply_clbt(imu_s, c):
        res = mod.np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    for it, policy in enumerate(ITERATION_POLICIES):
        print(f"  [{label}] {policy['name']} ({it+1}/{len(ITERATION_POLICIES)})")
        kf = mod.clbtkfinit_42(nts, bi_g, tau_g, bi_a, tau_a)

        if it == len(ITERATION_POLICIES) - 1:
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

        if it != len(ITERATION_POLICIES) - 1:
            clbt, feedback_meta = _apply_internalized_iter_feedback(mod, clbt, kf, policy)
            feedback_log.append(feedback_meta)

        iter_bounds.append(len(P_trace))

    return clbt, kf, mod.np.array(P_trace), mod.np.array(X_trace), {
        'iter_bounds': iter_bounds,
        'selected_state_labels': SELECTED_STATE_LABELS,
        'round46_reference': ROUND46_REFERENCE,
        'iteration_policies': ITERATION_POLICIES,
        'feedback_log': feedback_log,
        'policy': 'Internalize the scale-block preference by replacing the end-of-iteration feedback operator itself: state12/15/16/20/21 use a sign-safe blended version of the Round46 profile inside iterations 1-2, while all other states keep the baseline feedback path. No post-run clbt surgery is applied after run completion.',
    }


def run_method():
    mod = load_module('markov_pruned_42_round49_internalized_iterfeedback_stable', str(TMP_PSINS / SOURCE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = _build_dataset(mod)
    return _run_internalized_iterfeedback_stable(
        mod, imu_noisy, pos0, ts,
        bi_g=bi_g, bi_a=bi_a, tau_g=tau_g, tau_a=tau_a,
        label='42-GM1-R49-INTERNALIZED-STABLE',
    )


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
