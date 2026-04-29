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
METHOD = '42-state GM1 round25 pair-gated ranked-trim lever refine'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round25_pairgate_rankedtrim'


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


def _pairgate_rankedtrim_lever_refine(mod, imu1, pos0, ts, strong_clbt):
    eth = mod.Earth(pos0)
    wnie = mod.glv.wie * mod.np.array([0, mod.math.cos(pos0[0]), mod.math.sin(pos0[0])])
    gn = mod.np.array([0, 0, -eth.g])
    Cba = mod.np.eye(3)
    refined = _copy_clbt(mod, strong_clbt)
    windows = _static_window_bounds(mod, imu1, ts)
    if not windows:
        return refined, {'n_windows': 0, 'used_windows': 0}

    imu_corr = _apply_strong_only(mod, imu1, strong_clbt, ts)
    lever_rows, lever_targets, weights = [], [], []
    cond_before = []
    quality_vals = []
    ka2_theta = -mod.np.asarray(strong_clbt['Ka2'], dtype=float).reshape(3)

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
        A_lev = mean_ss
        b = -vn / max(len(seg) * ts, 1e-12) - mod.np.diag(mean_fb_sq) @ ka2_theta

        coln = mod.np.linalg.norm(A_lev, axis=0) + 1e-18
        A_scaled = A_lev / coln[None, :]
        sv = mod.np.linalg.svd(A_scaled, compute_uv=False)
        smax = float(mod.np.max(sv)) if len(sv) else 0.0
        smin = float(mod.np.min(sv)) if len(sv) else 0.0
        cond = smax / max(smin, 1e-15) if smax > 0 else 1e15
        cond_before.append(cond)

        quality = smin / max(smax, 1e-15)
        quality_vals.append(quality)
        motion = mod.np.linalg.norm(mean_ss, ord='fro')
        cond_soft = 1.0 / (1.0 + (cond / 2.5e11) ** 0.65)
        quality_soft = 0.10 + 2.20 * quality
        motion_soft = 0.35 + 0.06 * motion
        weight = max(len(seg) * cond_soft * quality_soft * motion_soft, 1e-6)
        lever_rows.append(A_lev)
        lever_targets.append(b)
        weights.append(weight)

    if not lever_rows:
        return refined, {'n_windows': len(windows), 'used_windows': 0}

    A = mod.np.vstack(lever_rows)
    b = mod.np.hstack(lever_targets)
    row_w = mod.np.repeat(mod.np.sqrt(mod.np.asarray(weights, dtype=float)), 3)
    Aw = A * row_w[:, None]
    bw = b * row_w

    col_scale = mod.np.sqrt(mod.np.sum(Aw * Aw, axis=0) + 1e-18)
    col_scale = mod.np.maximum(col_scale, mod.np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.08]))
    As = Aw / col_scale[None, :]

    prior = -mod.np.concatenate((strong_clbt['rx'], strong_clbt['ry']))
    prior_s = prior * col_scale
    delta_sigma = mod.np.array([0.0037, 0.0025, 0.0025, 0.0037, 0.0025, 0.0025])
    ridge = mod.np.diag((1.0 / delta_sigma) ** 2)

    U, S, Vt = mod.np.linalg.svd(As, full_matrices=False)
    smax = float(S[0]) if len(S) else 0.0
    trunc_floor = max(0.038 * smax, 1e-9)
    Sinv = mod.np.array([1.0 / s if s >= trunc_floor else 0.0 for s in S])
    theta_pinv = Vt.T @ (Sinv * (U.T @ bw)) if len(S) else prior_s.copy()

    lhs = As.T @ As + 1.7 * ridge
    rhs = As.T @ bw + 1.7 * ridge @ prior_s
    try:
        theta_map = mod.np.linalg.solve(lhs, rhs)
    except mod.np.linalg.LinAlgError:
        theta_map, *_ = mod.np.linalg.lstsq(lhs, rhs, rcond=None)

    theta_s = 0.80 * theta_map + 0.20 * theta_pinv
    theta = theta_s / col_scale
    proposal = -mod.np.asarray(theta).reshape(-1)
    prior_lever = mod.np.concatenate((strong_clbt['rx'], strong_clbt['ry']))
    raw_delta = mod.np.clip(proposal - prior_lever, -0.0024, 0.0024)

    gram = As.T @ As
    diag_gram = mod.np.diag(gram)
    coupling = mod.np.sum(mod.np.abs(gram), axis=1) + 1e-18
    diag_ratio = diag_gram / coupling
    grad = mod.np.abs(As.T @ bw)
    score = diag_ratio * grad / mod.np.sqrt(diag_gram + 1e-18)
    order = mod.np.argsort(score)[::-1]

    trust = mod.np.zeros(6)
    if len(order) >= 1:
        trust[order[0]] = 0.47
    if len(order) >= 2 and score[order[1]] / max(score[order[0]], 1e-18) > 0.15:
        trust[order[1]] = 0.14
    if len(order) >= 3:
        trust[order[2]] = 0.02

    delta = trust * raw_delta
    lever = mod.np.clip(prior_lever + delta, -0.08, 0.08)
    refined['rx'] = lever[0:3]
    refined['ry'] = lever[3:6]
    return refined, {
        'n_windows': len(windows),
        'used_windows': len(lever_rows),
        'cond_median': float(mod.np.median(cond_before)) if cond_before else None,
        'cond_worst': float(mod.np.max(cond_before)) if cond_before else None,
        'quality_median': float(mod.np.median(quality_vals)) if quality_vals else None,
        'trunc_floor': trunc_floor,
        'weight_sum': float(mod.np.sum(weights)),
        'score_order': [int(i) for i in order],
        'score': [float(x) for x in score],
        'diag_ratio': [float(x) for x in diag_ratio],
        'trust': [float(x) for x in trust],
    }


def run_method():
    mod = load_module('markov_pruned_42_round25_pairgate_rankedtrim', str(TMP_PSINS / SOURCE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = _build_dataset(mod)
    base_res = mod.run_calibration(
        imu_noisy, pos0, ts, n_states=42,
        bi_g=bi_g, tau_g=tau_g, bi_a=bi_a, tau_a=tau_a,
        label='42-GM1-R25-RTRIM'
    )
    refined_clbt, extra = _pairgate_rankedtrim_lever_refine(mod, imu_noisy, pos0, ts, base_res[0])
    return refined_clbt, base_res[1], base_res[2], base_res[3], {
        'iter_bounds': base_res[4],
        **extra,
    }


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
