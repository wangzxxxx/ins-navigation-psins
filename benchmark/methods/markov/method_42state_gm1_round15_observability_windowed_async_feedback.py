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
METHOD = '42-state GM1 round15 observability-windowed async feedback'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round15_observability_windowed_async_feedback'


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


def _strong_only_apply(mod, imu_s, clbt, ts):
    res = mod.np.copy(imu_s)
    for i in range(len(res)):
        res[i, 0:3] = clbt['Kg'] @ res[i, 0:3] - clbt['eb'] * ts
        res[i, 3:6] = clbt['Ka'] @ res[i, 3:6] - clbt['db'] * ts
    return res


def _estimate_aux_weak_states(mod, imu1, pos0, ts, coarse_clbt):
    eth = mod.Earth(pos0)
    wnie = mod.glv.wie * mod.np.array([0, mod.math.cos(pos0[0]), mod.math.sin(pos0[0])])
    gn = mod.np.array([0, 0, -eth.g])
    Cba = mod.np.eye(3)
    windows = _static_window_bounds(mod, imu1, ts)
    refined = _copy_clbt(mod, coarse_clbt)
    if not windows:
        return refined

    imu_corr = _strong_only_apply(mod, imu1, coarse_clbt, ts)
    rows, ys = [], []
    for s, e in windows:
        seg = imu_corr[s:e, :]
        if len(seg) < 50:
            continue
        qnb = mod.alignsb(seg, pos0)[3]
        vn = mod.np.zeros(3)
        dotwf = mod.imudot(seg, 5.0)
        ss_sum = mod.np.zeros((3, 6))
        for k in range(len(seg)):
            wm = seg[k, 0:3]
            vm = seg[k, 3:6]
            dwb = dotwf[k, 0:3]
            fb = vm / ts
            SS = mod.imulvS(wm / ts, dwb, Cba)
            fn_nominal = mod.qmulv(qnb, fb)
            vn = vn + (mod.rotv(-wnie * ts / 2, fn_nominal) + gn) * ts
            qnb = mod.qupdt2(qnb, wm, wnie * ts)
            rows.append(mod.np.hstack((mod.np.tile((fb**2).reshape(3, 1), (1, 3)), SS[:, 0:6])))
            ys.append(fn_nominal)
            ss_sum += SS[:, 0:6]
        mean_fb = mod.np.mean(seg[:, 3:6] / ts, axis=0)
        rows.append(mod.np.hstack((mod.np.tile((mean_fb**2).reshape(3, 1), (1, 3)), ss_sum / max(len(seg), 1))))
        ys.append(-vn / max(len(seg) * ts, 1e-12))

    if not rows:
        return refined
    A = mod.np.vstack(rows)
    b = mod.np.hstack(ys)
    try:
        theta, *_ = mod.np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return refined
    theta = mod.np.asarray(theta).reshape(-1)
    refined['Ka2'] = mod.np.clip(theta[0:3], -45 * mod.glv.ugpg2, 45 * mod.glv.ugpg2)
    lever = mod.np.clip(theta[3:9], -0.10, 0.10)
    refined['rx'] = lever[0:3]
    refined['ry'] = lever[3:6]
    return refined


def _select_measurement_windows(mod, imu1, pos0, ts, seed_clbt):
    eth = mod.Earth(pos0)
    wnie = mod.glv.wie * mod.np.array([0, mod.math.cos(pos0[0]), mod.math.sin(pos0[0])])
    gn = mod.np.array([0, 0, -eth.g])
    Cba = mod.np.eye(3)
    dotwf = mod.imudot(imu1, 5.0)
    windows = _static_window_bounds(mod, imu1, ts)
    scored = []
    for s, e in windows:
        seg = imu1[s:e, :]
        if len(seg) < 100:
            continue
        qnb = mod.alignsb(_strong_only_apply(mod, seg, seed_clbt, ts), pos0)[3]
        vn = mod.np.zeros(3)
        s_energy = 0.0
        fb_sq = mod.np.zeros(3)
        for k in range(len(seg)):
            wm = seg[k, 0:3]
            vm = seg[k, 3:6]
            dwb = dotwf[s + k, 0:3]
            wb = wm / ts
            fb = vm / ts
            SS = mod.imulvS(wb, dwb, Cba)
            fL = SS[:, 0:6] @ mod.np.concatenate((seed_clbt['rx'], seed_clbt['ry']))
            fn = mod.qmulv(qnb, fb - seed_clbt['Ka2'] * (fb**2) - fL)
            vn = vn + (mod.rotv(-wnie * ts / 2, fn) + gn) * ts
            qnb = mod.qupdt2(qnb, wm, wnie * ts)
            s_energy += float(mod.np.linalg.norm(SS[:, 0:6], ord='fro'))
            fb_sq += fb**2
        obs_score = float(mod.np.linalg.norm(vn)) + 0.02 * s_energy + 0.2 * float(mod.np.linalg.norm(fb_sq / len(seg)))
        scored.append((obs_score, s, e))
    if not scored:
        return []
    scored.sort(reverse=True)
    keep = max(1, int(round(len(scored) * 0.60)))
    selected = [(s, e) for _, s, e in scored[:keep]]
    selected.sort()
    return selected


def _feedback_strong_only(mod, base_clbt, xk):
    out = _copy_clbt(mod, base_clbt)
    dKg = xk[12:21].reshape(3, 3).T
    out['Kg'] = (mod.np.eye(3) - dKg) @ out['Kg']
    dKa = mod.Ka_from_upper(xk[21:27])
    out['Ka'] = (mod.np.eye(3) - dKa) @ out['Ka']
    out['eb'] = out['eb'] + xk[6:9] + 0.85 * xk[36:39]
    out['db'] = out['db'] + xk[9:12] + 0.85 * xk[39:42]
    return out


def _feedback_weak_only(mod, base_clbt, xk, alpha=0.65):
    out = _copy_clbt(mod, base_clbt)
    out['Ka2'] = out['Ka2'] + alpha * xk[27:30]
    out['rx'] = out['rx'] + alpha * xk[30:33]
    out['ry'] = out['ry'] + alpha * xk[33:36]
    out['Ka2'] = mod.np.clip(out['Ka2'], -45 * mod.glv.ugpg2, 45 * mod.glv.ugpg2)
    out['rx'] = mod.np.clip(out['rx'], -0.10, 0.10)
    out['ry'] = mod.np.clip(out['ry'], -0.10, 0.10)
    return out


def _run_42_windowed_async(mod, imu1, pos0, ts, bi_g, bi_a, tau_g, tau_a, coarse_clbt, weak_seed_clbt, label):
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

    measurement_windows = _select_measurement_windows(mod, imu1, pos0, ts, weak_seed_clbt)
    clbt = _copy_clbt(mod, weak_seed_clbt)
    length = len(imu1)
    dotwf = mod.imudot(imu1, 5.0)
    iterations = 3
    P_trace, X_trace, iter_bounds = [], [], []

    def in_selected_window(idx):
        for s, e in measurement_windows:
            if s <= idx < e:
                return True
        return False

    def apply_clbt(imu_s, c):
        res = mod.np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    for it in range(iterations):
        print(f'  [{label}] Iter {it+1}/{iterations}')
        kf = mod.clbtkfinit_42(nts, bi_g, tau_g, bi_a, tau_a)
        if it == 0:
            kf['Pxk'][27:30, 27:30] *= 0.08
            kf['Pxk'][30:36, 30:36] *= 0.08
            kf['Pxk'][12:27, 12:27] *= 0.65
            kf['Pxk'][6:12, 6:12] *= 0.70
        elif it == 1:
            kf['Pxk'][27:30, 27:30] *= 0.12
            kf['Pxk'][30:36, 30:36] *= 0.12
            kf['Pxk'][12:27, 12:27] *= 0.80
            kf['Pxk'][6:12, 6:12] *= 0.85
        else:
            kf['Pxk'] = kf['Pxk'] * 100
            kf['Pxk'][:, 2] = 0
            kf['Pxk'][2, :] = 0
            kf['Pxk'][27:30, 27:30] *= 0.20
            kf['Pxk'][30:36, 30:36] *= 0.20
            kf['Pxk'][12:27, 12:27] *= 0.90
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
                if mod.np.linalg.norm(ww) / ts < 20 * mod.glv.dph and in_selected_window(k):
                    kf = mod.kfupdate(kf, yk=vn, TimeMeasBoth='M')
                P_trace.append(mod.np.diag(kf['Pxk']))
                X_trace.append(mod.np.copy(kf['xk']))
        if it != iterations - 1:
            strong_updated = _feedback_strong_only(mod, clbt, kf['xk'])
            strong_updated['Ka2'] = clbt['Ka2']
            strong_updated['rx'] = clbt['rx']
            strong_updated['ry'] = clbt['ry']
            if it == 0:
                strong_updated['Ka2'] = 0.85 * weak_seed_clbt['Ka2'] + 0.15 * coarse_clbt['Ka2']
                strong_updated['rx'] = 0.85 * weak_seed_clbt['rx'] + 0.15 * coarse_clbt['rx']
                strong_updated['ry'] = 0.85 * weak_seed_clbt['ry'] + 0.15 * coarse_clbt['ry']
            clbt = strong_updated
        else:
            clbt = _feedback_strong_only(mod, clbt, kf['xk'])
            clbt = _feedback_weak_only(mod, clbt, kf['xk'], alpha=0.55)
        iter_bounds.append(len(P_trace))

    return clbt, kf, mod.np.array(P_trace), mod.np.array(X_trace), iter_bounds


def run_method():
    mod = load_module('markov_pruned_42_round15_observability_windowed_async_feedback', str(TMP_PSINS / SOURCE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = _build_dataset(mod)
    coarse_res = mod.run_calibration(
        imu_noisy, pos0, ts, n_states=42,
        bi_g=bi_g, tau_g=tau_g, bi_a=bi_a, tau_a=tau_a,
        label='42-GM1-R15-COARSE'
    )
    coarse_clbt = coarse_res[0]
    weak_seed = _estimate_aux_weak_states(mod, imu_noisy, pos0, ts, coarse_clbt)
    final_res = _run_42_windowed_async(
        mod, imu_noisy, pos0, ts, bi_g, bi_a, tau_g, tau_a,
        coarse_clbt=coarse_clbt, weak_seed_clbt=weak_seed,
        label='42-GM1-R15-OBS-WIN-ASYNC'
    )
    return final_res


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
