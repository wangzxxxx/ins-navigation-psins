"""
test_system_calibration_19pos_baseline.py
双轨噪声对比：无噪声 vs 加入 ARW/VRW 白噪声
生成各状态的方差收敛曲线对比 SVG 图
"""
import numpy as np
import sys
import os
import math
import copy
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psins_py.nav_utils import glv, posset, Earth
from psins_py.imu_utils import attrottt, avp2imu, imuclbt, imudot, cnscl, imulvS, imuadderr
from psins_py.kf_utils import clbtkfinit, kfupdate, clbtkffeedback, getFt, alignsb, nnts
from psins_py.math_utils import q2mat, qmulv, qupdt2, rotv


def get_default_clbt():
    Kg_mat = np.eye(3) - np.diag([10., 20., 30.]) * glv.ppm + \
             np.array([[0., 10., 20.], [30., 0., 40.], [50., 60., 0.]]) * glv.sec
    Ka_mat = np.eye(3) - np.diag([10., 20., 30.]) * glv.ppm + \
             np.array([[0., 0., 0.], [10., 0., 0.], [20., 30., 0.]]) * glv.sec
    return {
        'sf': np.ones(6), 'Kg': Kg_mat, 'Ka': Ka_mat,
        'eb': np.array([0.1, 0.2, 0.3]) * glv.dph,
        'db': np.array([100.0, 200.0, 300.0]) * glv.ug,
        'Ka2': np.array([10.0, 20.0, 30.0]) * glv.ugpg2,
        'rx': np.array([1.0, 2.0, 3.0]) / 100.0,
        'ry': np.array([4.0, 5.0, 6.0]) / 100.0,
        'rz': np.zeros(3), 'tGA': 0.005
    }


def run_calibration(imu1, pos0, ts, label=""):
    """Run iterative Kalman calibration and return P_trace, X_trace, iteration_boundaries."""
    eth = Earth(pos0)
    wnie = glv.wie * np.array([0, math.cos(pos0[0]), math.sin(pos0[0])])
    gn = np.array([0, 0, -eth.g])
    Cba = np.eye(3)
    nn, _, nts, _ = nnts(2, ts)
    frq2 = int(1 / ts / 2) - 1

    # Find kstatic: first instant where rotation starts
    k = frq2
    for k in range(frq2, min(5*60*2*frq2, len(imu1)), 2*frq2):
        ww = np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
        if np.linalg.norm(ww) / ts > 20 * glv.dph:
            break
    kstatic = k - 3 * frq2

    clbt = {'Kg': np.eye(3), 'Ka': np.eye(3), 'Ka2': np.zeros(3),
            'eb': np.zeros(3), 'db': np.zeros(3),
            'rx': np.zeros(3), 'ry': np.zeros(3), 'rz': np.zeros(3), 'tGA': 0.0}

    length = len(imu1)
    dotwf = imudot(imu1, 5.0)
    iterations = 3

    def apply_clbt(imu_s, c):
        res = np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    P_trace, X_trace, iter_bounds = [], [], []

    for it in range(iterations):
        print(f"  [{label}] Iteration {it+1}/{iterations}...")
        kf = clbtkfinit(nts)
        if it == iterations - 1:
            kf['Pxk'] = kf['Pxk'] * 100
            kf['Pxk'][:, 2] = 0; kf['Pxk'][2, :] = 0
            kf['xk'] = np.zeros(43)

        imu_align = apply_clbt(imu1[frq2:kstatic, :], clbt)
        _, _, _, qnb = alignsb(imu_align, pos0)

        vn = np.zeros(3)
        t1s = 0.0

        for k in range(2 * frq2, length - frq2, nn):
            k1 = k + nn - 1
            wm = imu1[k:k1+1, 0:3]
            vm = imu1[k:k1+1, 3:6]
            dwb = np.mean(dotwf[k:k1+1, 0:3], axis=0)

            phim, dvbm = cnscl(np.hstack((wm, vm)))
            phim = clbt['Kg'] @ phim - clbt['eb'] * nts
            dvbm = clbt['Ka'] @ dvbm - clbt['db'] * nts
            wb = phim / nts
            fb = dvbm / nts

            SS = imulvS(wb, dwb, Cba)
            fL = SS @ np.concatenate((clbt['rx'], clbt['ry'], clbt['rz']))
            fn = qmulv(qnb, fb - clbt['Ka2'] * (fb**2) - fL - clbt['tGA'] * np.cross(wb, fb))
            vn = vn + (rotv(-wnie * nts / 2, fn) + gn) * nts
            qnb = qupdt2(qnb, phim, wnie * nts)

            t1s += nts
            Ft = getFt(fb, wb, q2mat(qnb), wnie, SS)
            kf['Phikk_1'] = np.eye(43) + Ft * nts
            kf = kfupdate(kf, TimeMeasBoth='T')

            if t1s > (0.2 - ts / 2):
                t1s = 0.0
                ww = np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
                if np.linalg.norm(ww) / ts < 20 * glv.dph:
                    kf = kfupdate(kf, yk=vn, TimeMeasBoth='M')

                P_trace.append(np.diag(kf['Pxk']))
                X_trace.append(np.copy(kf['xk']))

        if it != iterations - 1:
            clbt = clbtkffeedback(kf, clbt)

        iter_bounds.append(len(P_trace))

    return np.array(P_trace), np.array(X_trace), iter_bounds, clbt


def main():
    ts = 0.01
    att0 = np.array([1.0, -91.0, -91.0]) * glv.deg
    pos0 = posset(34.0, 0.0, 0.0)

    paras = np.array([
        [1,    0, 1, 0,  90, 9, 70, 70],
        [2,    0, 1, 0,  90, 9, 20, 20],
        [3,    0, 1, 0,  90, 9, 20, 20],
        [4,    0, 1, 0, -90, 9, 20, 20],
        [5,    0, 1, 0, -90, 9, 20, 20],
        [6,    0, 1, 0, -90, 9, 20, 20],
        [7,    0, 0, 1,  90, 9, 20, 20],
        [8,    1, 0, 0,  90, 9, 20, 20],
        [9,    1, 0, 0,  90, 9, 20, 20],
        [10,   1, 0, 0,  90, 9, 20, 20],
        [11,  -1, 0, 0,  90, 9, 20, 20],
        [12,  -1, 0, 0,  90, 9, 20, 20],
        [13,  -1, 0, 0,  90, 9, 20, 20],
        [14,   0, 0, 1,  90, 9, 20, 20],
        [15,   0, 0, 1,  90, 9, 20, 20],
        [16,   0, 0,-1,  90, 9, 20, 20],
        [17,   0, 0,-1,  90, 9, 20, 20],
        [18,   0, 0,-1,  90, 9, 20, 20]
    ], dtype=float)
    paras[:, 4] = paras[:, 4] * glv.deg

    print("Generating attitude and IMU data...")
    att = attrottt(att0, paras, ts)
    imu, _ = avp2imu(att, pos0)
    clbt_truth = get_default_clbt()
    imu_clean = imuclbt(imu, clbt_truth)

    # 加入 ARW/VRW 白噪声（固定随机种子保证可复现）
    np.random.seed(42)
    imuerr_wn = {
        'web': np.array([0.01, 0.01, 0.01]) * glv.dpsh,
        'wdb': np.array([10.0, 10.0, 10.0]) * glv.ugpsHz
    }
    imu_noisy = imuadderr(imu_clean, imuerr_wn)

    print("\nRunning calibration WITHOUT noise...")
    P_clean, X_clean, bounds_clean, clbt_clean = run_calibration(imu_clean, pos0, ts, "No Noise")

    print("\nRunning calibration WITH noise (ARW=0.01 dpsh, VRW=10 ug/√Hz)...")
    P_noisy, X_noisy, bounds_noisy, clbt_noisy = run_calibration(imu_noisy, pos0, ts, "With Noise")

    # ============================================================
    # Plot convergence comparison
    # ============================================================
    state_names = [
        "phi_E", "phi_N", "phi_U",
        "dVE", "dVN", "dVU",
        "eb_x", "eb_y", "eb_z",
        "db_x", "db_y", "db_z",
        "dKg_xx", "dKg_yx", "dKg_zx",
        "dKg_xy", "dKg_yy", "dKg_zy",
        "dKg_xz", "dKg_yz", "dKg_zz",
        "dKa_xx", "dKa_yx", "dKa_zx",
        "dKa_xy", "dKa_yy", "dKa_zy",
        "dKa_xz", "dKa_yz", "dKa_zz",
        "dKa2_x", "dKa2_y", "dKa2_z",
        "rx", "ry", "rz",
        "ry_ext1", "ry_ext2", "ry_ext3",
        "rz_ext1", "rz_ext2", "rz_ext3",
        "tGA"
    ]

    base_dir = os.path.dirname(os.path.abspath(__file__))
    nn_ts = 2 * ts  # actual plotting dt
    time_arr_c = np.arange(len(P_clean)) * nn_ts
    time_arr_n = np.arange(len(P_noisy)) * nn_ts

    # --- Uncertainty (std) comparison ---
    out_dir = os.path.join(base_dir, 'plots_noise_compare_svg')
    os.makedirs(out_dir, exist_ok=True)
    print(f"\nGenerating uncertainty convergence comparison plots -> '{out_dir}/'")

    for i in range(43):
        name = state_names[i] if i < len(state_names) else f"State_{i}"
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

        # Top: standard deviation
        ax = axes[0]
        ax.plot(time_arr_c, np.sqrt(P_clean[:, i]), color='blue', label='No Noise')
        ax.plot(time_arr_n, np.sqrt(P_noisy[:, i]), color='red',  label='With ARW/VRW', linestyle='--')
        for b in bounds_clean[:-1]:
            ax.axvline(x=b * nn_ts, color='gray', linestyle=':', alpha=0.6)
        ax.set_title(f'[{i:02d}] {name}  —  Uncertainty (std dev)')
        ax.set_ylabel('Std Dev')
        ax.legend(); ax.grid(True)

        # Bottom: estimated state value
        ax2 = axes[1]
        ax2.plot(time_arr_c, X_clean[:, i], color='blue', label='No Noise')
        ax2.plot(time_arr_n, X_noisy[:, i], color='red', label='With ARW/VRW', linestyle='--')
        for b in bounds_clean[:-1]:
            ax2.axvline(x=b * nn_ts, color='gray', linestyle=':', alpha=0.6)
        ax2.set_title(f'[{i:02d}] {name}  —  State Estimate')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Estimated Value')
        ax2.legend(); ax2.grid(True)

        plt.tight_layout()
        safe = name.replace('/', '_').replace('\\', '_')
        plt.savefig(os.path.join(out_dir, f"{i:02d}_{safe}.svg"), format='svg')
        plt.close()

    print(f"Saved {43} comparison plots to '{out_dir}/'")

    # ============================================================
    # Final accuracy summary
    # ============================================================
    print("\n" + "="*100)
    print("ACCURACY COMPARISON: No Noise vs With Noise")
    print("="*100)
    print(f"{'Param':<12} | {'True Val':>15} | {'No-Noise Est':>15} | {'N Err%':>7} | {'Noisy Est':>15} | {'Noisy Err%':>9}")
    print("-"*100)

    def cmp(name, true_v, c_v, n_v):
        c_err = abs(true_v + c_v) / abs(true_v) * 100 if abs(true_v) > 1e-15 else 0
        n_err = abs(true_v + n_v) / abs(true_v) * 100 if abs(true_v) > 1e-15 else 0
        print(f"{name:<12} | {true_v:>15.4e} | {-c_v:>15.4e} | {c_err:>6.2f}% | {-n_v:>15.4e} | {n_err:>8.2f}%")

    cmp("eb_x", clbt_truth['eb'][0], clbt_clean['eb'][0], clbt_noisy['eb'][0])
    cmp("eb_y", clbt_truth['eb'][1], clbt_clean['eb'][1], clbt_noisy['eb'][1])
    cmp("eb_z", clbt_truth['eb'][2], clbt_clean['eb'][2], clbt_noisy['eb'][2])
    cmp("db_x", clbt_truth['db'][0], clbt_clean['db'][0], clbt_noisy['db'][0])
    cmp("db_y", clbt_truth['db'][1], clbt_clean['db'][1], clbt_noisy['db'][1])
    cmp("db_z", clbt_truth['db'][2], clbt_clean['db'][2], clbt_noisy['db'][2])
    cmp("rx",   clbt_truth['rx'][0], clbt_clean['rx'][0], clbt_noisy['rx'][0])
    cmp("ry",   clbt_truth['ry'][0], clbt_clean['ry'][0], clbt_noisy['ry'][0])
    cmp("tGA",  clbt_truth['tGA'],   clbt_clean['tGA'],   clbt_noisy['tGA'])
    print("="*100)


if __name__ == "__main__":
    main()
