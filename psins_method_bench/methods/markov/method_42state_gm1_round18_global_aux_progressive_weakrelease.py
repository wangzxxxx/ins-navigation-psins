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
METHOD = '42-state GM1 round18 global auxiliary progressive weak-release'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round18_global_aux_progressive_weakrelease'


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


def _global_aux_weaksolve(mod, imu1, pos0, ts, strong_clbt):
    eth = mod.Earth(pos0)
    wnie = mod.glv.wie * mod.np.array([0, mod.math.cos(pos0[0]), mod.math.sin(pos0[0])])
    gn = mod.np.array([0, 0, -eth.g])
    Cba = mod.np.eye(3)
    windows = _static_window_bounds(mod, imu1, ts)
    refined = _copy_clbt(mod, strong_clbt)
    if not windows:
        return refined

    imu_corr = _apply_strong_only(mod, imu1, strong_clbt, ts)
    rows, ys = [], []
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

    if not rows:
        return refined

    A = mod.np.vstack(rows)
    b = mod.np.hstack(ys)
    prior = mod.np.hstack((
        -strong_clbt['Ka2'],
        -strong_clbt['rx'],
        -strong_clbt['ry'],
    ))
    reg = mod.np.diag(mod.np.array([
        1.0 / (16.0 * mod.glv.ugpg2),
        1.0 / (16.0 * mod.glv.ugpg2),
        1.0 / (16.0 * mod.glv.ugpg2),
        1.0 / 0.030,
        1.0 / 0.030,
        1.0 / 0.030,
        1.0 / 0.030,
        1.0 / 0.030,
        1.0 / 0.030,
    ])**2)
    lhs = A.T @ A + 0.32 * reg
    rhs = A.T @ b + 0.32 * reg @ prior
    try:
        theta = mod.np.linalg.solve(lhs, rhs)
    except mod.np.linalg.LinAlgError:
        theta, *_ = mod.np.linalg.lstsq(lhs, rhs, rcond=None)
    theta = mod.np.asarray(theta).reshape(-1)

    aux = _copy_clbt(mod, strong_clbt)
    aux['Ka2'] = mod.np.clip(-theta[0:3], -26 * mod.glv.ugpg2, 26 * mod.glv.ugpg2)
    lever = mod.np.clip(-theta[3:9], -0.06, 0.06)
    aux['rx'] = lever[0:3]
    aux['ry'] = lever[3:6]

    refined['Ka2'] = 0.65 * strong_clbt['Ka2'] + 0.35 * aux['Ka2']
    refined['rx'] = 0.70 * strong_clbt['rx'] + 0.30 * aux['rx']
    refined['ry'] = 0.70 * strong_clbt['ry'] + 0.30 * aux['ry']
    return refined


def _run_progressive_release(mod, imu1, pos0, ts, bi_g, bi_a, tau_g, tau_a, init_clbt, label):
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

    clbt = _copy_clbt(mod, init_clbt)
    dotwf = mod.imudot(imu1, 5.0)
    P_trace, X_trace, iter_bounds = [], [], []
    weak_diag_scales = [0.18, 0.40, 0.75]
    coupling_scales = [0.85, 1.00, 1.10]
    weak_blends = [0.20, 0.45]

    def apply_clbt(imu_s, c):
        res = mod.np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    for it in range(3):
        print(f'  [{label}] Iter {it+1}/3')
        kf = mod.clbtkfinit_42(nts, bi_g, tau_g, bi_a, tau_a)
        if it == 2:
            kf['Pxk'] = kf['Pxk'] * 100
            kf['Pxk'][:, 2] = 0
            kf['Pxk'][2, :] = 0
            kf['xk'] = mod.np.zeros(42)
        kf['Pxk'][27:30, 27:30] *= weak_diag_scales[it]
        kf['Pxk'][30:36, 30:36] *= weak_diag_scales[it]

        imu_align = apply_clbt(imu1[frq2:kstatic, :], clbt)
        _, _, _, qnb = mod.alignsb(imu_align, pos0)
        vn = mod.np.zeros(3)
        t1s = 0.0

        for k in range(2 * frq2, len(imu1) - frq2, nn):
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
            Ft[27:36, 12:27] *= coupling_scales[it]
            Ft[12:27, 27:36] *= coupling_scales[it]
            kf['Phikk_1'] = mod.np.eye(42) + Ft * nts
            kf = mod.kfupdate(kf, TimeMeasBoth='T')

            if t1s > (0.2 - ts / 2):
                t1s = 0.0
                ww = mod.np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
                if mod.np.linalg.norm(ww) / ts < 20 * mod.glv.dph:
                    kf = mod.kfupdate(kf, yk=vn, TimeMeasBoth='M')
                P_trace.append(mod.np.diag(kf['Pxk']))
                X_trace.append(mod.np.copy(kf['xk']))

        if it != 2:
            proposal = mod.clbtkffeedback_pruned(kf, _copy_clbt(mod, clbt), 42)
            alpha = weak_blends[it]
            clbt['Kg'] = proposal['Kg']
            clbt['Ka'] = proposal['Ka']
            clbt['eb'] = proposal['eb']
            clbt['db'] = proposal['db']
            clbt['Ka2'] = (1 - alpha) * clbt['Ka2'] + alpha * proposal['Ka2']
            clbt['rx'] = (1 - alpha) * clbt['rx'] + alpha * proposal['rx']
            clbt['ry'] = (1 - alpha) * clbt['ry'] + alpha * proposal['ry']
        else:
            clbt = mod.clbtkffeedback_pruned(kf, clbt, 42)
        iter_bounds.append(len(P_trace))

    return clbt, kf, mod.np.array(P_trace), mod.np.array(X_trace), iter_bounds


def run_method():
    mod = load_module('markov_pruned_42_round18_global_aux_progressive_weakrelease', str(TMP_PSINS / SOURCE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = _build_dataset(mod)
    base_res = mod.run_calibration(
        imu_noisy, pos0, ts, n_states=42,
        bi_g=bi_g, tau_g=tau_g, bi_a=bi_a, tau_a=tau_a,
        label='42-GM1-R18-BASE'
    )
    aux_clbt = _global_aux_weaksolve(mod, imu_noisy, pos0, ts, base_res[0])
    return _run_progressive_release(
        mod, imu_noisy, pos0, ts, bi_g, bi_a, tau_g, tau_a,
        init_clbt=aux_clbt, label='42-GM1-R18-PROG-WEAK'
    )


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
