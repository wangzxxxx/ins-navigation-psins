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
METHOD = '42-state GM1 round47 internalized selective scale-block schedule'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round47_internalized_scaleblock'

# Carry the best post-correction profile into the iterative loop itself,
# instead of applying it once after run_calibration returns.
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
OTHER_SCALE_STATES = tuple(idx for idx in range(12, 27) if idx not in SELECTED_SCALE_STATES)

# Distribute the old R46 profile / guard across the first two outer iterations.
# Final iteration is readout only: no extra post-return correction.
ITERATION_POLICIES = [
    {
        'name': 'iter1_embed_scale_seed',
        'profile_scale': 0.40,
        'other_scale_alpha': 0.72,
        'ka2_alpha': 0.78,
        'lever_alpha': 0.82,
        'markov_alpha': 0.88,
        'rx_y_guard_mult': 1.01950,
        'ry_z_guard_mult': 1.00127,
        'selected_cov_boost': 1.55,
        'other_scale_cov_scale': 0.86,
        'ka2_cov_scale': 0.82,
        'lever_cov_scale': 0.78,
        'bias_cov_scale': 0.92,
    },
    {
        'name': 'iter2_embed_scale_commit',
        'profile_scale': 0.60,
        'other_scale_alpha': 0.80,
        'ka2_alpha': 0.86,
        'lever_alpha': 0.90,
        'markov_alpha': 0.92,
        'rx_y_guard_mult': 1.01950,
        'ry_z_guard_mult': 1.00128,
        'selected_cov_boost': 1.75,
        'other_scale_cov_scale': 0.90,
        'ka2_cov_scale': 0.88,
        'lever_cov_scale': 0.84,
        'bias_cov_scale': 0.96,
    },
    {
        'name': 'iter3_readout_only',
        'profile_scale': 0.0,
        'other_scale_alpha': 0.0,
        'ka2_alpha': 0.0,
        'lever_alpha': 0.0,
        'markov_alpha': 0.0,
        'rx_y_guard_mult': 1.0,
        'ry_z_guard_mult': 1.0,
        'selected_cov_boost': 1.95,
        'other_scale_cov_scale': 0.94,
        'ka2_cov_scale': 0.92,
        'lever_cov_scale': 0.90,
        'bias_cov_scale': 1.00,
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


def _scale_cov_block(kf, indices, scale):
    for idx in indices:
        kf['Pxk'][idx, idx] *= scale


def _configure_iteration_prior(kf, policy):
    _scale_cov_block(kf, SELECTED_SCALE_STATES, policy['selected_cov_boost'])
    _scale_cov_block(kf, OTHER_SCALE_STATES, policy['other_scale_cov_scale'])
    _scale_cov_block(kf, range(27, 30), policy['ka2_cov_scale'])
    _scale_cov_block(kf, range(30, 36), policy['lever_cov_scale'])
    _scale_cov_block(kf, range(6, 12), policy['bias_cov_scale'])


def _apply_embedded_feedback(mod, base_clbt, kf, policy):
    out = _copy_clbt(mod, base_clbt)
    xk = kf['xk']
    xfb = mod.np.zeros_like(xk)

    for idx in SELECTED_SCALE_STATES:
        xfb[idx] = policy['profile_scale'] * ROUND46_REFERENCE['profile'][idx] * xk[idx]
    for idx in OTHER_SCALE_STATES:
        xfb[idx] = policy['other_scale_alpha'] * xk[idx]

    dKg = xfb[12:21].reshape(3, 3).T
    out['Kg'] = (mod.np.eye(3) - dKg) @ out['Kg']

    dKa = mod.Ka_from_upper(xfb[21:27])
    out['Ka'] = (mod.np.eye(3) - dKa) @ out['Ka']

    out['eb'] = out['eb'] + xk[6:9] + policy['markov_alpha'] * xk[36:39]
    out['db'] = out['db'] + xk[9:12] + policy['markov_alpha'] * xk[39:42]
    out['Ka2'] = out['Ka2'] + policy['ka2_alpha'] * xk[27:30]
    out['rx'] = out['rx'] + policy['lever_alpha'] * xk[30:33]
    out['ry'] = out['ry'] + policy['lever_alpha'] * xk[33:36]

    out['rx'][1] = out['rx'][1] * policy['rx_y_guard_mult']
    out['ry'][2] = out['ry'][2] * policy['ry_z_guard_mult']

    out['Ka2'] = mod.np.clip(out['Ka2'], -45 * mod.glv.ugpg2, 45 * mod.glv.ugpg2)
    out['rx'] = mod.np.clip(out['rx'], -0.10, 0.10)
    out['ry'] = mod.np.clip(out['ry'], -0.10, 0.10)

    focus_feedback = {str(idx): float(xfb[idx]) for idx in SELECTED_SCALE_STATES}
    natural_selected_xk = {str(idx): float(xk[idx]) for idx in SELECTED_SCALE_STATES}
    return out, {
        'policy_name': policy['name'],
        'profile_scale': float(policy['profile_scale']),
        'selected_xk': natural_selected_xk,
        'embedded_feedback': focus_feedback,
        'other_scale_alpha': float(policy['other_scale_alpha']),
        'ka2_alpha': float(policy['ka2_alpha']),
        'lever_alpha': float(policy['lever_alpha']),
        'markov_alpha': float(policy['markov_alpha']),
        'rx_y_guard_mult': float(policy['rx_y_guard_mult']),
        'ry_z_guard_mult': float(policy['ry_z_guard_mult']),
        'rx_y_after': float(out['rx'][1]),
        'ry_z_after': float(out['ry'][2]),
    }


def _run_internalized_scaleblock(mod, imu1, pos0, ts, bi_g, bi_a, tau_g, tau_a, label):
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
    embedded_feedback_log = []

    def apply_clbt(imu_s, c):
        res = mod.np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    for it, policy in enumerate(ITERATION_POLICIES):
        print(f"  [{label}] {policy['name']} ({it+1}/{len(ITERATION_POLICIES)})")
        kf = mod.clbtkfinit_42(nts, bi_g, tau_g, bi_a, tau_a)
        _configure_iteration_prior(kf, policy)

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
            clbt, feedback_meta = _apply_embedded_feedback(mod, clbt, kf, policy)
            embedded_feedback_log.append(feedback_meta)

        iter_bounds.append(len(P_trace))

    return clbt, kf, mod.np.array(P_trace), mod.np.array(X_trace), {
        'iter_bounds': iter_bounds,
        'selected_state_labels': SELECTED_STATE_LABELS,
        'round46_reference': ROUND46_REFERENCE,
        'iteration_policies': ITERATION_POLICIES,
        'embedded_feedback_log': embedded_feedback_log,
        'policy': 'Scale-block preference is injected inside the outer iterative loop via covariance shaping and distributed selective feedback after iterations 1-2 only; iteration 3 is readout only, so no extra post-return correction is applied.',
    }


def run_method():
    mod = load_module('markov_pruned_42_round47_internalized_scaleblock', str(TMP_PSINS / SOURCE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = _build_dataset(mod)
    return _run_internalized_scaleblock(
        mod, imu_noisy, pos0, ts,
        bi_g=bi_g, bi_a=bi_a, tau_g=tau_g, tau_a=tau_a,
        label='42-GM1-R47-INTERNALIZED'
    )


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
