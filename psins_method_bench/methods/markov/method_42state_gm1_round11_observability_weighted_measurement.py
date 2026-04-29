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
METHOD = '42-state GM1 round11 observability-weighted measurement'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round11_observability_weighted_measurement'

WEAK_IDXS = [12, 21, 27, 28, 29, 30, 31, 32, 33, 34, 35]


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


def _weak_observability_score(mod, kf):
    H = kf['Hk']
    P = kf['Pxk']
    weak_cols = H @ P[:, WEAK_IDXS]
    full_proj = H @ P @ H.T
    weak_energy = float(mod.np.linalg.norm(weak_cols, ord='fro'))
    full_energy = float(mod.np.sqrt(max(mod.np.trace(full_proj), 1e-30)))
    ratio = weak_energy / max(full_energy, 1e-12)
    return float(mod.np.clip(ratio, 0.05, 1.50))


def run_method():
    mod = load_module('markov_pruned_42_round11_observability_weighted_measurement', str(TMP_PSINS / SOURCE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = _build_dataset(mod)

    eth = mod.Earth(pos0)
    wnie = mod.glv.wie * mod.np.array([0, mod.math.cos(pos0[0]), mod.math.sin(pos0[0])])
    gn = mod.np.array([0, 0, -eth.g])
    Cba = mod.np.eye(3)
    nn, _, nts, _ = mod.nnts(2, ts)
    frq2 = int(1 / ts / 2) - 1
    k = frq2
    for k in range(frq2, min(5 * 60 * 2 * frq2, len(imu_noisy)), 2 * frq2):
        ww = mod.np.mean(imu_noisy[k-frq2:k+frq2+1, 0:3], axis=0)
        if mod.np.linalg.norm(ww) / ts > 20 * mod.glv.dph:
            break
    kstatic = k - 3 * frq2

    clbt = {
        'Kg': mod.np.eye(3), 'Ka': mod.np.eye(3), 'Ka2': mod.np.zeros(3),
        'eb': mod.np.zeros(3), 'db': mod.np.zeros(3), 'rx': mod.np.zeros(3), 'ry': mod.np.zeros(3)
    }
    length = len(imu_noisy)
    dotwf = mod.imudot(imu_noisy, 5.0)
    iterations = 3
    P_trace, X_trace, iter_bounds = [], [], []

    def apply_clbt(imu_s, c):
        res = mod.np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    base_r_diag = mod.np.array([0.001, 0.001, 0.001]) ** 2

    for it in range(iterations):
        print(f'  [42-GM1-R11-OBS-WEIGHTED] Iter {it+1}/{iterations}')
        kf = mod.clbtkfinit_42(nts, bi_g, tau_g, bi_a, tau_a)
        if it == iterations - 1:
            kf['Pxk'] = kf['Pxk'] * 100
            kf['Pxk'][:, 2] = 0
            kf['Pxk'][2, :] = 0
            kf['xk'] = mod.np.zeros(42)

        imu_align = apply_clbt(imu_noisy[frq2:kstatic, :], clbt)
        _, _, _, qnb = mod.alignsb(imu_align, pos0)
        vn = mod.np.zeros(3)
        t1s = 0.0

        for k in range(2 * frq2, length - frq2, nn):
            k1 = k + nn - 1
            wm = imu_noisy[k:k1+1, 0:3]
            vm = imu_noisy[k:k1+1, 3:6]
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
                ww = mod.np.mean(imu_noisy[k-frq2:k+frq2+1, 0:3], axis=0)
                if mod.np.linalg.norm(ww) / ts < 20 * mod.glv.dph:
                    obs_ratio = _weak_observability_score(mod, kf)
                    # More observable weak-state projection => trust this static velocity-zero update more.
                    # Poorly observable window => relax update to avoid injecting unhelpful corrections.
                    r_scale = 1.55 - 0.95 * obs_ratio
                    if it == 0:
                        r_scale *= 1.08
                    elif it == 2:
                        r_scale *= 0.95
                    r_scale = float(mod.np.clip(r_scale, 0.42, 1.65))
                    kf['Rk'] = mod.np.diag(base_r_diag * r_scale)
                    kf = mod.kfupdate(kf, yk=vn, TimeMeasBoth='M')
                    kf['Rk'] = mod.np.diag(base_r_diag)

                P_trace.append(mod.np.diag(kf['Pxk']))
                X_trace.append(mod.np.copy(kf['xk']))

        clbt = mod.clbtkffeedback_pruned(kf, clbt, 42)
        iter_bounds.append(len(P_trace))

    return clbt, kf, mod.np.array(P_trace), mod.np.array(X_trace), iter_bounds


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
