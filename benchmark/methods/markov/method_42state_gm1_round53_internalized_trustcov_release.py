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
METHOD = '42-state GM1 round53 internalized trust-shaped covariance release'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round53_internalized_trustcov_release'

SELECTED_SCALE_STATES = (12, 15, 16, 20, 21)
OTHER_SCALE_STATES = tuple(idx for idx in range(12, 27) if idx not in SELECTED_SCALE_STATES)
SELECTED_STATE_LABELS = {
    '12': 'dKg_xx',
    '15': 'dKg_xy',
    '16': 'dKg_yy',
    '20': 'dKg_zz',
    '21': 'dKa_xx',
}

ITERATION_POLICIES = [
    {
        'name': 'iter1_seed',
        'selected_prior_scale': 1.38,
        'other_scale_prior_scale': 0.94,
        'ka2_prior_scale': 0.93,
        'lever_prior_scale': 0.91,
        'selected_q_static_scale': 0.62,
        'selected_q_dynamic_scale': 0.88,
        'selected_q_late_mult': 1.42,
        'other_scale_q_scale': 0.92,
        'other_scale_q_late_mult': 1.05,
        'ka2_q_scale': 0.95,
        'lever_q_scale': 0.93,
        'static_r_scale': 1.06,
        'dynamic_r_scale': 1.00,
        'late_r_mult': 0.99,
        'late_release_frac': 0.55,
        'selected_alpha_floor': 0.88,
        'selected_alpha_span': 0.16,
        'other_scale_alpha': 0.98,
        'ka2_alpha': 1.00,
        'lever_alpha': 1.00,
        'markov_alpha': 1.00,
        'trust_score_soft': 2.4,
        'trust_cov_soft': 0.48,
        'trust_mix': 0.56,
    },
    {
        'name': 'iter2_commit',
        'selected_prior_scale': 1.24,
        'other_scale_prior_scale': 0.97,
        'ka2_prior_scale': 0.96,
        'lever_prior_scale': 0.94,
        'selected_q_static_scale': 0.78,
        'selected_q_dynamic_scale': 1.00,
        'selected_q_late_mult': 1.28,
        'other_scale_q_scale': 0.96,
        'other_scale_q_late_mult': 1.03,
        'ka2_q_scale': 0.98,
        'lever_q_scale': 0.98,
        'static_r_scale': 1.03,
        'dynamic_r_scale': 1.00,
        'late_r_mult': 0.995,
        'late_release_frac': 0.58,
        'selected_alpha_floor': 0.96,
        'selected_alpha_span': 0.12,
        'other_scale_alpha': 0.99,
        'ka2_alpha': 1.00,
        'lever_alpha': 1.00,
        'markov_alpha': 1.00,
        'trust_score_soft': 2.1,
        'trust_cov_soft': 0.44,
        'trust_mix': 0.58,
    },
    {
        'name': 'iter3_readout_only',
        'readout_only': True,
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


def _scale_diag_block(mat, indices, scale):
    for idx in indices:
        mat[idx, idx] *= scale


def _configure_iteration_prior(mod, kf, policy):
    if policy.get('readout_only'):
        return mod.np.diag(kf['Pxk']).copy()
    _scale_diag_block(kf['Pxk'], SELECTED_SCALE_STATES, policy['selected_prior_scale'])
    _scale_diag_block(kf['Pxk'], OTHER_SCALE_STATES, policy['other_scale_prior_scale'])
    _scale_diag_block(kf['Pxk'], range(27, 30), policy['ka2_prior_scale'])
    _scale_diag_block(kf['Pxk'], range(30, 36), policy['lever_prior_scale'])
    return mod.np.diag(kf['Pxk']).copy()


def _set_cov_schedule(kf, base_q, base_r, policy, progress, is_static):
    if policy.get('readout_only'):
        return
    kf['Qk'][:, :] = base_q
    kf['Rk'][:, :] = base_r

    selected_q_scale = policy['selected_q_static_scale'] if is_static else policy['selected_q_dynamic_scale']
    other_scale_q_scale = policy['other_scale_q_scale']
    ka2_q_scale = policy['ka2_q_scale']
    lever_q_scale = policy['lever_q_scale']
    r_scale = policy['static_r_scale'] if is_static else policy['dynamic_r_scale']

    if progress >= policy['late_release_frac']:
        selected_q_scale *= policy['selected_q_late_mult']
        other_scale_q_scale *= policy['other_scale_q_late_mult']
        r_scale *= policy['late_r_mult']

    _scale_diag_block(kf['Qk'], SELECTED_SCALE_STATES, selected_q_scale)
    _scale_diag_block(kf['Qk'], OTHER_SCALE_STATES, other_scale_q_scale)
    _scale_diag_block(kf['Qk'], range(27, 30), ka2_q_scale)
    _scale_diag_block(kf['Qk'], range(30, 36), lever_q_scale)
    kf['Rk'] *= r_scale


def _trust_components(mod, kf, prior_diag, idx, policy):
    p = float(max(kf['Pxk'][idx, idx], 1e-24))
    x_abs = float(abs(kf['xk'][idx]))
    score = x_abs / mod.np.sqrt(p)
    score_n = score / (score + policy['trust_score_soft'])
    cov_drop = mod.np.sqrt(float(max(prior_diag[idx], 1e-24)) / p)
    cov_n = max(cov_drop - 1.0, 0.0)
    cov_n = cov_n / (cov_n + policy['trust_cov_soft']) if cov_n > 0 else 0.0
    trust = policy['trust_mix'] * score_n + (1.0 - policy['trust_mix']) * cov_n
    alpha = policy['selected_alpha_floor'] + policy['selected_alpha_span'] * trust
    return {
        'p_diag': p,
        'x_abs': x_abs,
        'snr_like': float(score),
        'score_n': float(score_n),
        'cov_drop': float(cov_drop),
        'cov_n': float(cov_n),
        'trust': float(trust),
        'alpha': float(alpha),
    }


def _apply_trust_internalized_feedback(mod, base_clbt, kf, prior_diag, policy):
    out = _copy_clbt(mod, base_clbt)
    xk = mod.np.array(kf['xk'], dtype=float).copy()
    xfb = xk.copy()
    trust_log = {}

    for idx in SELECTED_SCALE_STATES:
        tc = _trust_components(mod, kf, prior_diag, idx, policy)
        xfb[idx] = tc['alpha'] * xk[idx]
        tc['label'] = SELECTED_STATE_LABELS[str(idx)]
        tc['x_feedback'] = float(xfb[idx])
        trust_log[str(idx)] = tc

    for idx in OTHER_SCALE_STATES:
        xfb[idx] = policy['other_scale_alpha'] * xk[idx]

    dKg = xfb[12:21].reshape(3, 3).T
    out['Kg'] = (mod.np.eye(3) - dKg) @ out['Kg']

    dKa = mod.Ka_from_upper(xfb[21:27])
    out['Ka'] = (mod.np.eye(3) - dKa) @ out['Ka']

    out['Ka2'] = out['Ka2'] + policy['ka2_alpha'] * xfb[27:30]
    out['eb'] = out['eb'] + xfb[6:9]
    out['db'] = out['db'] + xfb[9:12]
    out['rx'] = out['rx'] + policy['lever_alpha'] * xfb[30:33]
    out['ry'] = out['ry'] + policy['lever_alpha'] * xfb[33:36]
    out['eb'] = out['eb'] + policy['markov_alpha'] * xfb[36:39]
    out['db'] = out['db'] + policy['markov_alpha'] * xfb[39:42]

    return out, {
        'policy_name': policy['name'],
        'selected_state_labels': SELECTED_STATE_LABELS,
        'selected_trust': trust_log,
        'other_scale_alpha': float(policy['other_scale_alpha']),
        'ka2_alpha': float(policy['ka2_alpha']),
        'lever_alpha': float(policy['lever_alpha']),
        'markov_alpha': float(policy['markov_alpha']),
    }


def _run_internalized_trustcov_release(mod, imu1, pos0, ts, bi_g, bi_a, tau_g, tau_a, label):
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

    def apply_clbt(imu_s, c):
        res = mod.np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    for it, policy in enumerate(ITERATION_POLICIES):
        print(f'  [{label}] {policy["name"]} ({it+1}/{len(ITERATION_POLICIES)})')
        kf = mod.clbtkfinit_42(nts, bi_g, tau_g, bi_a, tau_a)
        prior_diag = _configure_iteration_prior(mod, kf, policy)
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
                _set_cov_schedule(kf, base_q, base_r, policy, progress, is_static)
                if is_static:
                    n_static_meas += 1
                    kf = mod.kfupdate(kf, yk=vn, TimeMeasBoth='M')
                else:
                    n_dynamic_sched += 1
                if progress >= policy.get('late_release_frac', 2.0):
                    n_late_sched += 1
                P_trace.append(mod.np.diag(kf['Pxk']))
                X_trace.append(mod.np.copy(kf['xk']))

        if not policy.get('readout_only'):
            clbt, meta = _apply_trust_internalized_feedback(mod, clbt, kf, prior_diag, policy)
            feedback_log.append(meta)

        schedule_log.append({
            'policy_name': policy['name'],
            'n_static_meas': int(n_static_meas),
            'n_dynamic_sched': int(n_dynamic_sched),
            'n_late_sched': int(n_late_sched),
            'late_release_frac': float(policy.get('late_release_frac', 1.0)),
        })
        iter_bounds.append(len(P_trace))

    return clbt, kf, mod.np.array(P_trace), mod.np.array(X_trace), {
        'iter_bounds': iter_bounds,
        'selected_state_labels': SELECTED_STATE_LABELS,
        'iteration_policies': ITERATION_POLICIES,
        'feedback_log': feedback_log,
        'schedule_log': schedule_log,
        'policy': 'Round53 does not inject the Round46 gain profile into iteration feedback. Instead it uses selected-state prior inflation, static-vs-dynamic covariance shaping, a late release schedule, and trust-capped selective feedback that only moves slightly below/around unity.',
    }


def run_method():
    mod = load_module('markov_pruned_42_round53_internalized_trustcov_release', str(TMP_PSINS / SOURCE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = _build_dataset(mod)
    return _run_internalized_trustcov_release(
        mod, imu_noisy, pos0, ts,
        bi_g=bi_g, bi_a=bi_a, tau_g=tau_g, tau_a=tau_a,
        label='42-GM1-R53-TRUSTCOV-RELEASE',
    )


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
