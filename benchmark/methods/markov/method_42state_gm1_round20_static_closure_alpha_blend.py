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
METHOD = '42-state GM1 round20 static-closure alpha-blend weak solve'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round20_static_closure_alpha_blend'


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


def _solve_weak_states(mod, imu1, pos0, ts, strong_clbt):
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
        rows.append(mod.np.hstack((mod.np.diag(mean_fb_sq), mean_ss)))
        ys.append(-vn / max(len(seg) * ts, 1e-12))
        weights.append(len(seg) * float(mod.np.linalg.norm(mean_fb_sq) + 1e-6))

    if not rows or not ys:
        return refined

    W = mod.np.repeat(mod.np.sqrt(mod.np.asarray(weights, dtype=float)), 3)
    A = mod.np.vstack(rows)
    b = mod.np.hstack(ys)
    Aw = A * W[:, None]
    bw = b * W
    prior = mod.np.hstack((
        -strong_clbt['Ka2'],
        -strong_clbt['rx'],
        -strong_clbt['ry'],
    ))
    reg = mod.np.diag(mod.np.array([
        1.0 / (15.0 * mod.glv.ugpg2),
        1.0 / (15.0 * mod.glv.ugpg2),
        1.0 / (15.0 * mod.glv.ugpg2),
        1.0 / 0.028,
        1.0 / 0.028,
        1.0 / 0.028,
        1.0 / 0.028,
        1.0 / 0.028,
        1.0 / 0.028,
    ])**2)
    try:
        theta = mod.np.linalg.solve(Aw.T @ Aw + 0.28 * reg, Aw.T @ bw + 0.28 * reg @ prior)
    except mod.np.linalg.LinAlgError:
        theta, *_ = mod.np.linalg.lstsq(Aw.T @ Aw + 0.28 * reg, Aw.T @ bw + 0.28 * reg @ prior, rcond=None)
    theta = mod.np.asarray(theta).reshape(-1)

    refined['Ka2'] = mod.np.clip(-theta[0:3], -30 * mod.glv.ugpg2, 30 * mod.glv.ugpg2)
    lever = mod.np.clip(-theta[3:9], -0.06, 0.06)
    refined['rx'] = lever[0:3]
    refined['ry'] = lever[3:6]
    return refined


def _blend_weak(mod, base, proposal, alpha):
    out = _copy_clbt(mod, base)
    out['Ka2'] = (1 - alpha) * base['Ka2'] + alpha * proposal['Ka2']
    out['rx'] = (1 - alpha) * base['rx'] + alpha * proposal['rx']
    out['ry'] = (1 - alpha) * base['ry'] + alpha * proposal['ry']
    return out


def _static_closure_score(mod, imu1, pos0, ts, clbt):
    eth = mod.Earth(pos0)
    wnie = mod.glv.wie * mod.np.array([0, mod.math.cos(pos0[0]), mod.math.sin(pos0[0])])
    gn = mod.np.array([0, 0, -eth.g])
    Cba = mod.np.eye(3)
    windows = _static_window_bounds(mod, imu1, ts)
    if not windows:
        return float('inf')
    score = 0.0
    used = 0
    imu_corr = _apply_strong_only(mod, imu1, clbt, ts)
    for s, e in windows:
        seg = imu_corr[s:e, :]
        if len(seg) < 80:
            continue
        qnb = mod.alignsb(seg, pos0)[3]
        vn = mod.np.zeros(3)
        dotwf = mod.imudot(seg, 5.0)
        for k in range(len(seg)):
            wm = seg[k, 0:3]
            vm = seg[k, 3:6]
            dwb = dotwf[k, 0:3]
            fb = vm / ts
            SS = mod.imulvS(wm / ts, dwb, Cba)
            fL = SS[:, 0:6] @ mod.np.concatenate((clbt['rx'], clbt['ry']))
            fn = mod.qmulv(qnb, fb - clbt['Ka2'] * (fb**2) - fL)
            vn = vn + (mod.rotv(-wnie * ts / 2, fn) + gn) * ts
            qnb = mod.qupdt2(qnb, wm, wnie * ts)
        score += float(mod.np.linalg.norm(vn)) / max(len(seg) * ts, 1e-12)
        used += 1
    return score / max(used, 1)


def run_method():
    mod = load_module('markov_pruned_42_round20_static_closure_alpha_blend', str(TMP_PSINS / SOURCE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = _build_dataset(mod)
    baseline_res = mod.run_calibration(
        imu_noisy, pos0, ts, n_states=42,
        bi_g=bi_g, tau_g=tau_g, bi_a=bi_a, tau_a=tau_a,
        label='42-GM1-R20-BASE'
    )
    baseline_clbt = baseline_res[0]
    solved_clbt = _solve_weak_states(mod, imu_noisy, pos0, ts, baseline_clbt)

    alpha_grid = [0.0, 0.2, 0.4, 0.6, 0.8]
    best_alpha = 0.0
    best_score = float('inf')
    best_clbt = _copy_clbt(mod, baseline_clbt)
    for alpha in alpha_grid:
        cand = _blend_weak(mod, baseline_clbt, solved_clbt, alpha)
        score = _static_closure_score(mod, imu_noisy, pos0, ts, cand)
        print(f'  [42-GM1-R20-ALPHA] alpha={alpha:.2f} score={score:.6e}')
        if score < best_score:
            best_score = score
            best_alpha = alpha
            best_clbt = cand
    print(f'  [42-GM1-R20-SELECT] best_alpha={best_alpha:.2f} score={best_score:.6e}')
    return best_clbt, baseline_res[1], baseline_res[2], baseline_res[3], {
        'iter_bounds': baseline_res[4],
        'selected_alpha': best_alpha,
        'selected_score': best_score,
    }


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
