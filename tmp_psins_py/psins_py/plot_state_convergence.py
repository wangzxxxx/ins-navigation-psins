"""
plot_state_convergence.py
--------------------------
运行 49 状态一阶 Markov 标定滤波器（正确 tau），
记录全部状态量 xk 和 P 对角线随时间的变化，生成收敛变化图。
"""
import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psins_py.nav_utils import glv, posset, Earth
from psins_py.imu_utils import attrottt, avp2imu, imuclbt, imudot, cnscl, imulvS
from psins_py.kf_utils import kfupdate, clbtkffeedback, alignsb, nnts
from psins_py.math_utils import q2mat, qmulv, qupdt2, rotv

from test_calibration_markov_noise import (
    imuadderr_full, get_default_clbt,
    clbtkfinit_49, getFt_49, clbtkffeedback_49,
)


def run_with_history(imu1, pos0, ts, bi_g, tau_g, bi_a, tau_a):
    """运行 49 态滤波器，记录每步的 xk 和 sqrt(Pxk) 对角线"""
    n_states = 49
    eth  = Earth(pos0)
    wnie = glv.wie * np.array([0, math.cos(pos0[0]), math.sin(pos0[0])])
    gn   = np.array([0, 0, -eth.g])
    Cba  = np.eye(3)
    nn, _, nts, _ = nnts(2, ts)
    frq2 = int(1 / ts / 2) - 1

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
    dotwf  = imudot(imu1, 5.0)

    def apply_clbt(imu_s, c):
        res = np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    # 只跑第 1 次迭代来记录状态演化
    print("  Running 49-state filter (iter 1), recording state history...")
    kf = clbtkfinit_49(nts, bi_g, tau_g, bi_a, tau_a)

    imu_align = apply_clbt(imu1[frq2:kstatic, :], clbt)
    _, _, _, qnb = alignsb(imu_align, pos0)
    vn  = np.zeros(3)
    t1s = 0.0

    # 记录历史
    t_hist = []
    xk_hist = []
    sigma_hist = []
    elapsed = 0.0

    for k in range(2 * frq2, length - frq2, nn):
        k1  = k + nn - 1
        wm  = imu1[k:k1+1, 0:3]
        vm  = imu1[k:k1+1, 3:6]
        dwb = np.mean(dotwf[k:k1+1, 0:3], axis=0)

        phim, dvbm = cnscl(np.hstack((wm, vm)))
        phim  = clbt['Kg'] @ phim - clbt['eb'] * nts
        dvbm  = clbt['Ka'] @ dvbm - clbt['db'] * nts
        wb    = phim / nts
        fb    = dvbm / nts

        SS  = imulvS(wb, dwb, Cba)
        fL  = SS @ np.concatenate((clbt['rx'], clbt['ry'], clbt['rz']))
        fn  = qmulv(qnb, fb - clbt['Ka2']*(fb**2) - fL - clbt['tGA']*np.cross(wb, fb))
        vn  = vn + (rotv(-wnie*nts/2, fn) + gn) * nts
        qnb = qupdt2(qnb, phim, wnie * nts)

        t1s += nts
        elapsed += nts
        Ft = getFt_49(fb, wb, q2mat(qnb), wnie, SS, tau_g, tau_a)

        kf['Phikk_1'] = np.eye(n_states) + Ft * nts
        kf = kfupdate(kf, TimeMeasBoth='T')

        if t1s > (0.2 - ts / 2):
            t1s = 0.0
            ww = np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
            if np.linalg.norm(ww) / ts < 20 * glv.dph:
                kf = kfupdate(kf, yk=vn, TimeMeasBoth='M')

        # 每秒记录一次
        if len(t_hist) == 0 or elapsed - t_hist[-1] >= 1.0:
            t_hist.append(elapsed)
            xk_hist.append(kf['xk'].copy())
            sigma_hist.append(np.sqrt(np.abs(np.diag(kf['Pxk']))))

    t_hist = np.array(t_hist)
    xk_hist = np.array(xk_hist)
    sigma_hist = np.array(sigma_hist)
    print(f"  Recorded {len(t_hist)} time points over {elapsed:.0f}s")

    return t_hist, xk_hist, sigma_hist


def plot_convergence(t, xk, sigma, out_dir):
    """绘制 49 个状态的收敛图，按分组画子图"""
    os.makedirs(out_dir, exist_ok=True)

    # 状态分组
    groups = [
        ("Attitude Error (φ)", [0, 1, 2],
         ["φ_E", "φ_N", "φ_U"], 1/glv.deg, "deg"),
        ("Velocity Error (δv)", [3, 4, 5],
         ["δv_E", "δv_N", "δv_U"], 1.0, "m/s"),
        ("Gyro Bias (eb)", [6, 7, 8],
         ["eb_x", "eb_y", "eb_z"], 1/glv.dph, "deg/h"),
        ("Accel Bias (db)", [9, 10, 11],
         ["db_x", "db_y", "db_z"], 1/glv.ug, "µg"),
        ("Gyro Scale Factor (dKg diag)", [12, 16, 20],
         ["dKg_xx", "dKg_yy", "dKg_zz"], 1/glv.ppm, "ppm"),
        ("Gyro Misalignment (dKg off-diag)", [13, 14, 15, 17, 18, 19],
         ["dKg_yx", "dKg_zx", "dKg_xy", "dKg_zy", "dKg_xz", "dKg_yz"],
         1/glv.sec, "arcsec"),
        ("Accel Scale Factor (dKa diag)", [21, 25, 29],
         ["dKa_xx", "dKa_yy", "dKa_zz"], 1/glv.ppm, "ppm"),
        ("Accel Misalignment (dKa off-diag)", [22, 23, 24, 26, 27, 28],
         ["dKa_yx", "dKa_zx", "dKa_xy", "dKa_zy", "dKa_xz", "dKa_yz"],
         1/glv.sec, "arcsec"),
        ("Accel Quadratic (Ka2)", [30, 31, 32],
         ["Ka2_x", "Ka2_y", "Ka2_z"], 1/glv.ugpg2, "µg/g²"),
        ("Lever Arm rx", [33, 34, 35],
         ["rx_x", "rx_y", "rx_z"], 1.0, "m"),
        ("Lever Arm ry", [36, 37, 38],
         ["ry_x", "ry_y", "ry_z"], 1.0, "m"),
        ("tGA (Gyro-Accel delay)", [42],
         ["tGA"], 1.0, "s"),
        ("Markov Gyro Bias (bm_g)", [43, 44, 45],
         ["bm_g_x", "bm_g_y", "bm_g_z"], 1/glv.dph, "deg/h"),
        ("Markov Accel Bias (bm_a)", [46, 47, 48],
         ["bm_a_x", "bm_a_y", "bm_a_z"], 1/glv.ug, "µg"),
    ]

    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 1, figsize=(16, 3.5 * n_groups), sharex=True)
    if n_groups == 1:
        axes = [axes]

    colors_pool = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for gi, (title, indices, labels, scale, unit) in enumerate(groups):
        ax = axes[gi]
        for ci, (idx, lbl) in enumerate(zip(indices, labels)):
            color = colors_pool[ci % len(colors_pool)]
            x_scaled = xk[:, idx] * scale
            s_scaled = sigma[:, idx] * scale
            ax.plot(t, x_scaled, color=color, linewidth=1.0, label=lbl)
            ax.fill_between(t, x_scaled - s_scaled, x_scaled + s_scaled,
                            color=color, alpha=0.15)
        ax.set_ylabel(f"{unit}")
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8, ncol=min(len(indices), 3))
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()

    svg_path = os.path.join(out_dir, 'state_convergence_49.svg')
    plt.savefig(svg_path, format='svg')
    plt.close()
    print(f"Saved convergence plot to '{svg_path}'")

    # 额外：只画 Markov 状态的放大版
    fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ci, (idx, lbl) in enumerate(zip([43, 44, 45], ["bm_g_x", "bm_g_y", "bm_g_z"])):
        color = colors_pool[ci]
        x_s = xk[:, idx] / glv.dph
        s_s = sigma[:, idx] / glv.dph
        ax1.plot(t, x_s, color=color, linewidth=1.2, label=lbl)
        ax1.fill_between(t, x_s - s_s, x_s + s_s, color=color, alpha=0.2)
    ax1.set_ylabel("deg/h")
    ax1.set_title("Markov Gyro Bias bm_g (estimate ± σ)", fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')

    for ci, (idx, lbl) in enumerate(zip([46, 47, 48], ["bm_a_x", "bm_a_y", "bm_a_z"])):
        color = colors_pool[ci]
        x_s = xk[:, idx] / glv.ug
        s_s = sigma[:, idx] / glv.ug
        ax2.plot(t, x_s, color=color, linewidth=1.2, label=lbl)
        ax2.fill_between(t, x_s - s_s, x_s + s_s, color=color, alpha=0.2)
    ax2.set_ylabel("µg")
    ax2.set_title("Markov Accel Bias bm_a (estimate ± σ)", fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='--')
    ax2.set_xlabel("Time (s)")

    plt.tight_layout()
    svg_path2 = os.path.join(out_dir, 'markov_state_convergence.svg')
    plt.savefig(svg_path2, format='svg')
    plt.close()
    print(f"Saved Markov state convergence to '{svg_path2}'")


def main():
    ts   = 0.01
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

    ARW   = 0.01  * glv.dpsh
    VRW   = 10.0  * glv.ugpsHz
    BI_G  = 0.025 * glv.dph
    BI_A  = 50.0  * glv.ug
    TAU_G = 300.0
    TAU_A = 300.0

    print("Generating IMU trajectory...")
    att  = attrottt(att0, paras, ts)
    imu, _ = avp2imu(att, pos0)
    clbt_truth = get_default_clbt()
    imu_clean  = imuclbt(imu, clbt_truth)
    imu_noisy = imuadderr_full(imu_clean, ts,
                                arw=ARW, vrw=VRW,
                                bi_g=BI_G, tau_g=TAU_G,
                                bi_a=BI_A, tau_a=TAU_A, seed=42)

    print("\n49-state filter with correct tau...")
    t, xk, sigma = run_with_history(imu_noisy, pos0, ts, BI_G, TAU_G, BI_A, TAU_A)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots_markov_svg')
    plot_convergence(t, xk, sigma, out_dir)


if __name__ == "__main__":
    main()
