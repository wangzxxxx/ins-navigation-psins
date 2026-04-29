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
METHOD = '42-state GM1 round21 conditioned anisotropic weak-state solve'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round21_conditioned_aniso_weaksolve'


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


def _static_window_bounds(mod, imu1, ts):
    frq2 = int(1 / ts / 2) - 1
    windows = []
    in_static = False
    start = None
    for k in range(frq2, len(imu1) - frq2, 2 * frq2):
        ww = mod.np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
        is_static = mod.np.linalg.norm(ww) / ts < 20 * mod.glv.dph
        if is_static and not in_static:
            in_static = True
            start = max(0, k - frq2)
        elif (not is_static) and in_static:
            in_static = False
            windows.append((start, min(len(imu1), k + frq2)))
            start = None
    if in_static and start is not None:
        windows.append((start, len(imu1)))
    return windows


def _apply_strong_only(mod, imu_s, clbt, ts):
    res = mod.np.copy(imu_s)
    for i in range(len(res)):
        res[i, 0:3] = clbt['Kg'] @ res[i, 0:3] - clbt['eb'] * ts
        res[i, 3:6] = clbt['Ka'] @ res[i, 3:6] - clbt['db'] * ts
    return res


def _conditioned_aniso_weaksolve(mod, imu1, pos0, ts, strong_clbt):
    eth = mod.Earth(pos0)
    wnie = mod.glv.wie * mod.np.array([0, mod.math.cos(pos0[0]), mod.math.sin(pos0[0])])
    gn = mod.np.array([0, 0, -eth.g])
    Cba = mod.np.eye(3)
    windows = _static_window_bounds(mod, imu1, ts)
    refined = _copy_clbt(mod, strong_clbt)
    if not windows:
        return refined

    imu_corr = _apply_strong_only(mod, imu1, strong_clbt, ts)
    rows, ys, weights = [], [], []
    for s, e in windows:
        seg = imu_corr[s:e, :]
        if len(seg) < 80:
            continue
        qnb = mod.alignsb(seg, pos0)[3]
        vn = mod.np.zeros(3)
        fb_sq_sum = mod.np.zeros(3)
        ss_sum = mod.np.zeros((3, 6))
        dotwf = mod.imudot(seg, 5.0)
        for k in range(len(seg)):
            wm = seg[k, 0:3]
            vm = seg[k, 3:6]
            dwb = dotwf[k, 0:3]
            fb = vm / ts
            SS = mod.imulvS(wm / ts, dwb, Cba)
            fb_sq_sum += fb**2
            ss_sum += SS[:, 0:6]
            qnb = mod.qupdt2(qnb, wm, wnie * ts)
            fn0 = mod.qmulv(qnb, fb)
            vn = vn + (mod.rotv(-wnie * ts / 2, fn0) + gn) * ts
        mean_fb_sq = fb_sq_sum / max(len(seg), 1)
        mean_ss = ss_sum / max(len(seg), 1)
        Awin = mod.np.hstack((mod.np.diag(mean_fb_sq), mean_ss))
        ywin = -vn / max(len(seg) * ts, 1e-12)
        info = mod.np.linalg.norm(mean_fb_sq) + 0.35 * mod.np.linalg.norm(mean_ss, ord='fro') + 1e-9
        rows.append(Awin)
        ys.append(ywin)
        weights.append(len(seg) * info)

    if not rows:
        return refined

    A = mod.np.vstack(rows)
    b = mod.np.hstack(ys)
    row_w = mod.np.repeat(mod.np.sqrt(mod.np.asarray(weights, dtype=float)), 3)
    Aw = A * row_w[:, None]
    bw = b * row_w

    col_scale = mod.np.sqrt(mod.np.sum(Aw * Aw, axis=0) + 1e-18)
    col_scale = mod.np.maximum(col_scale, mod.np.array([
        4.0e-8, 4.0e-8, 4.0e-8,
        0.12, 0.12, 0.12,
        0.12, 0.12, 0.12,
    ]))
    As = Aw / col_scale[None, :]

    prior = mod.np.hstack((-strong_clbt['Ka2'], -strong_clbt['rx'], -strong_clbt['ry']))
    prior_s = prior * col_scale
    reg_sigma = mod.np.array([
        24.0 * mod.glv.ugpg2, 16.0 * mod.glv.ugpg2, 16.0 * mod.glv.ugpg2,
        0.032, 0.028, 0.028,
        0.032, 0.028, 0.028,
    ])
    reg_s = mod.np.diag((col_scale / reg_sigma) ** 2)
    lam = 0.22
    lhs = As.T @ As + lam * reg_s
    rhs = As.T @ bw + lam * reg_s @ prior_s
    try:
        theta_s = mod.np.linalg.solve(lhs, rhs)
    except mod.np.linalg.LinAlgError:
        theta_s, *_ = mod.np.linalg.lstsq(lhs, rhs, rcond=None)
    theta = theta_s / col_scale
    theta = mod.np.asarray(theta).reshape(-1)

    proposal_ka2 = mod.np.clip(-theta[0:3], -35 * mod.glv.ugpg2, 35 * mod.glv.ugpg2)
    proposal_lever = mod.np.clip(-theta[3:9], -0.08, 0.08)

    ka2_prior = strong_clbt['Ka2']
    lever_prior = mod.np.concatenate((strong_clbt['rx'], strong_clbt['ry']))
    ka2_delta = mod.np.clip(proposal_ka2 - ka2_prior, -2.5 * mod.glv.ugpg2, 2.5 * mod.glv.ugpg2)
    lever_delta = mod.np.clip(proposal_lever - lever_prior, -0.0035, 0.0035)

    refined['Ka2'] = ka2_prior + ka2_delta
    lever = lever_prior + lever_delta
    refined['rx'] = lever[0:3]
    refined['ry'] = lever[3:6]
    return refined


def run_method():
    mod = load_module('markov_pruned_42_round21_conditioned_aniso_weaksolve', str(TMP_PSINS / SOURCE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = _build_dataset(mod)
    baseline_res = mod.run_calibration(
        imu_noisy, pos0, ts, n_states=42,
        bi_g=bi_g, tau_g=tau_g, bi_a=bi_a, tau_a=tau_a,
        label='42-GM1-R21-BASE'
    )
    refined_clbt = _conditioned_aniso_weaksolve(mod, imu_noisy, pos0, ts, baseline_res[0])
    return refined_clbt, baseline_res[1], baseline_res[2], baseline_res[3], {
        'iter_bounds': baseline_res[4],
    }


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
