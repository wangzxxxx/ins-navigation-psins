"""
test_calibration_markov_pruned.py
-----------------------------------
裁剪版标定滤波器：强制 Ka 上三角 + 移除 tGA + 移除 rz

  状态裁剪方案（在标准 49 维基础上）：
    1. Ka 强制上三角：消去 dKa_yx, dKa_zx, dKa_zy（49→46）
    2. 移除 tGA：（46→45）
    3. 移除 rz：（45→42）—— rz 不可观（对 19 位置法激励不足），固定为零

  36 状态基础布局（无 Markov）：
  [0:3]   姿态误差 φ
  [3:6]   速度误差 δv
  [6:9]   陀螺常值零偏 eb
  [9:12]  加计常值零偏 db
  [12:21] dKg (3×3 = 9)
  [21:27] dKa 上三角 (6): [xx, xy, xz, yy, yz, zz]
  [27:30] Ka2
  [30:33] rx
  [33:36] ry
  共 36 维

  42 状态：36 + bm_g(3) + bm_a(3)
  48 状态：42 + dbm_g(3) + dbm_a(3)  [二阶 GM]

  rz 作为已知量（真值 = 0），在补偿计算中固定为 zeros(3)，不参与估计。

对比四种条件：
  A: 无噪声（理想） + 36 状态
  B: 有全噪声 + 36 状态（不建模 Markov）
  C: 有全噪声 + 42 状态（一阶 Markov，正确 tau）
  D: 有全噪声 + 48 状态（二阶 Gauss-Markov，正确 tau）
以及各自的相关系数热力图
"""
import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt
import seaborn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psins_py.nav_utils import glv, posset, Earth
from psins_py.imu_utils import attrottt, avp2imu, imuclbt, imudot, cnscl, imulvS
from psins_py.kf_utils import kfupdate, alignsb, nnts
from psins_py.math_utils import q2mat, qmulv, qupdt2, rotv, askew


# ═══════════════════════════════════════════════════════════════
#  Gauss-Markov 噪声注入
# ═══════════════════════════════════════════════════════════════
def imuadderr_full(imu_in, ts,
                   arw=0.0, vrw=0.0,
                   bi_g=0.0, tau_g=3600.0,
                   bi_a=0.0, tau_a=3600.0,
                   seed=42):
    np.random.seed(seed)
    imu = np.copy(imu_in)
    m = imu.shape[0]
    sts = math.sqrt(ts)
    if arw > 0:
        imu[:, 0:3] += arw * sts * np.random.randn(m, 3)
    if vrw > 0:
        imu[:, 3:6] += vrw * sts * np.random.randn(m, 3)
    if bi_g > 0 and tau_g > 0:
        coeff = math.exp(-ts / tau_g)
        sigma_w = bi_g * math.sqrt(2.0 * ts / tau_g)
        b = np.zeros(3)
        for k in range(m):
            b = coeff * b + sigma_w * np.random.randn(3)
            imu[k, 0:3] += b * ts
    if bi_a > 0 and tau_a > 0:
        coeff = math.exp(-ts / tau_a)
        sigma_w = bi_a * math.sqrt(2.0 * ts / tau_a)
        b = np.zeros(3)
        for k in range(m):
            b = coeff * b + sigma_w * np.random.randn(3)
            imu[k, 3:6] += b * ts
    return imu


# ═══════════════════════════════════════════════════════════════
#  标定参数真值（Ka 上三角，rz = 0 已知）
# ═══════════════════════════════════════════════════════════════
def get_default_clbt():
    Kg_mat = np.eye(3) - np.diag([10., 20., 30.]) * glv.ppm + \
             np.array([[0., 10., 20.], [30., 0., 40.], [50., 60., 0.]]) * glv.sec
    Ka_mat = np.eye(3) - np.diag([10., 20., 30.]) * glv.ppm + \
             np.array([[0., 10., 20.], [0., 0., 40.], [0., 0., 0.]]) * glv.sec
    return {
        'sf': np.ones(6), 'Kg': Kg_mat, 'Ka': Ka_mat,
        'eb': np.array([0.1, 0.2, 0.3]) * glv.dph,
        'db': np.array([100.0, 200.0, 300.0]) * glv.ug,
        'Ka2': np.array([10.0, 20.0, 30.0]) * glv.ugpg2,
        'rx': np.array([1.0, 2.0, 3.0]) / 100.0,
        'ry': np.array([4.0, 5.0, 6.0]) / 100.0,
        # rz = 0（已知，不需要估计，标定时刻意令 Ka 对齐 B 系）
    }


# ═══════════════════════════════════════════════════════════════
#  Ka 上三角辅助函数
# ═══════════════════════════════════════════════════════════════
def Ka_from_upper(x_dKa_6):
    dKa = np.zeros((3, 3))
    dKa[0, 0] = x_dKa_6[0]
    dKa[0, 1] = x_dKa_6[1]
    dKa[0, 2] = x_dKa_6[2]
    dKa[1, 1] = x_dKa_6[3]
    dKa[1, 2] = x_dKa_6[4]
    dKa[2, 2] = x_dKa_6[5]
    return dKa


# ═══════════════════════════════════════════════════════════════
#  36 状态滤波器初始化（基础，无 tGA，无 rz，无 Markov）
#
#  状态布局：
#  [0:3]   φ     [3:6]  δv   [6:9]  eb   [9:12]  db
#  [12:21] dKg(9) [21:27] dKa_upper(6) [27:30] Ka2
#  [30:33] rx    [33:36] ry
# ═══════════════════════════════════════════════════════════════
def clbtkfinit_36(nts):
    n = 36
    kf = {'nts': nts, 'n': n, 'm': 3}

    qvec = np.zeros(n)
    qvec[0:3] = 0.01 * glv.dpsh
    qvec[3:6] = 100 * glv.ugpsHz
    kf['Qk'] = np.diag(qvec)**2 * nts

    kf['Rk'] = np.diag([0.001, 0.001, 0.001])**2

    pvec = np.zeros(n)
    pvec[0:3]   = np.array([0.1, 0.1, 1.0]) * glv.deg
    pvec[3:6]   = 1.0
    pvec[6:9]   = 0.1 * glv.dph
    pvec[9:12]  = 1.0 * glv.mg
    pvec[12:15] = [100*glv.ppm, 100*glv.sec, 100*glv.sec]
    pvec[15:18] = [100*glv.sec, 100*glv.ppm, 100*glv.sec]
    pvec[18:21] = [100*glv.sec, 100*glv.sec, 100*glv.ppm]
    pvec[21]    = 100 * glv.ppm   # dKa_xx
    pvec[22]    = 100 * glv.sec   # dKa_xy
    pvec[23]    = 100 * glv.sec   # dKa_xz
    pvec[24]    = 100 * glv.ppm   # dKa_yy
    pvec[25]    = 100 * glv.sec   # dKa_yz
    pvec[26]    = 100 * glv.ppm   # dKa_zz
    pvec[27:30] = 100 * glv.ugpg2
    pvec[30:33] = 0.1             # rx
    pvec[33:36] = 0.1             # ry

    kf['Pxk'] = np.diag(pvec)**2
    Hk = np.zeros((3, n))
    Hk[:, 3:6] = np.eye(3)
    kf['Hk'] = Hk
    kf['xk'] = np.zeros(n)
    kf['I'] = np.eye(n)
    return kf


# ═══════════════════════════════════════════════════════════════
#  42 状态滤波器初始化（一阶 Gauss-Markov）
#  [36:39] bm_g,  [39:42] bm_a
# ═══════════════════════════════════════════════════════════════
def clbtkfinit_42(nts, bi_g, tau_g, bi_a, tau_a):
    n = 42
    kf = {'nts': nts, 'n': n, 'm': 3}

    qvec = np.zeros(n)
    qvec[0:3] = 0.01 * glv.dpsh
    qvec[3:6] = 100 * glv.ugpsHz
    q_markov_g = bi_g * math.sqrt(1 - math.exp(-2*nts/tau_g)) if tau_g > 0 else 0.0
    q_markov_a = bi_a * math.sqrt(1 - math.exp(-2*nts/tau_a)) if tau_a > 0 else 0.0
    qvec[36:39] = q_markov_g
    qvec[39:42] = q_markov_a
    kf['Qk'] = np.diag(qvec)**2
    kf['Qk'][0:6, 0:6] = np.diag(qvec[0:6])**2 * nts

    kf['Rk'] = np.diag([0.001, 0.001, 0.001])**2

    pvec = np.zeros(n)
    pvec[0:3]   = np.array([0.1, 0.1, 1.0]) * glv.deg
    pvec[3:6]   = 1.0
    pvec[6:9]   = 0.1 * glv.dph
    pvec[9:12]  = 1.0 * glv.mg
    pvec[12:15] = [100*glv.ppm, 100*glv.sec, 100*glv.sec]
    pvec[15:18] = [100*glv.sec, 100*glv.ppm, 100*glv.sec]
    pvec[18:21] = [100*glv.sec, 100*glv.sec, 100*glv.ppm]
    pvec[21]    = 100 * glv.ppm
    pvec[22]    = 100 * glv.sec
    pvec[23]    = 100 * glv.sec
    pvec[24]    = 100 * glv.ppm
    pvec[25]    = 100 * glv.sec
    pvec[26]    = 100 * glv.ppm
    pvec[27:30] = 100 * glv.ugpg2
    pvec[30:33] = 0.1
    pvec[33:36] = 0.1
    pvec[36:39] = bi_g   # bm_g
    pvec[39:42] = bi_a   # bm_a

    kf['Pxk'] = np.diag(pvec)**2
    Hk = np.zeros((3, n))
    Hk[:, 3:6] = np.eye(3)
    kf['Hk'] = Hk
    kf['xk'] = np.zeros(n)
    kf['I'] = np.eye(n)
    kf['tau_g'] = tau_g
    kf['tau_a'] = tau_a
    return kf


# ═══════════════════════════════════════════════════════════════
#  48 状态滤波器初始化（二阶 Gauss-Markov）
#  [36:39] bm_g, [39:42] bm_a, [42:45] dbm_g, [45:48] dbm_a
# ═══════════════════════════════════════════════════════════════
def clbtkfinit_48(nts, bi_g, tau_g, bi_a, tau_a):
    n = 48
    kf = {'nts': nts, 'n': n, 'm': 3}
    beta_g = 1.0 / tau_g if tau_g > 0 else 0.0
    beta_a = 1.0 / tau_a if tau_a > 0 else 0.0

    qvec = np.zeros(n)
    qvec[0:3] = 0.01 * glv.dpsh
    qvec[3:6] = 100 * glv.ugpsHz
    q_rate_g = bi_g * (2*beta_g)**1.5 * math.sqrt(nts) if beta_g > 0 else 0.0
    q_rate_a = bi_a * (2*beta_a)**1.5 * math.sqrt(nts) if beta_a > 0 else 0.0
    qvec[42:45] = q_rate_g
    qvec[45:48] = q_rate_a
    kf['Qk'] = np.diag(qvec)**2
    kf['Qk'][0:6, 0:6] = np.diag(qvec[0:6])**2 * nts

    kf['Rk'] = np.diag([0.001, 0.001, 0.001])**2

    pvec = np.zeros(n)
    pvec[0:3]   = np.array([0.1, 0.1, 1.0]) * glv.deg
    pvec[3:6]   = 1.0
    pvec[6:9]   = 0.1 * glv.dph
    pvec[9:12]  = 1.0 * glv.mg
    pvec[12:15] = [100*glv.ppm, 100*glv.sec, 100*glv.sec]
    pvec[15:18] = [100*glv.sec, 100*glv.ppm, 100*glv.sec]
    pvec[18:21] = [100*glv.sec, 100*glv.sec, 100*glv.ppm]
    pvec[21]    = 100 * glv.ppm
    pvec[22]    = 100 * glv.sec
    pvec[23]    = 100 * glv.sec
    pvec[24]    = 100 * glv.ppm
    pvec[25]    = 100 * glv.sec
    pvec[26]    = 100 * glv.ppm
    pvec[27:30] = 100 * glv.ugpg2
    pvec[30:33] = 0.1
    pvec[33:36] = 0.1
    pvec[36:39] = bi_g
    pvec[39:42] = bi_a
    pvec[42:45] = bi_g * beta_g
    pvec[45:48] = bi_a * beta_a

    kf['Pxk'] = np.diag(pvec)**2
    Hk = np.zeros((3, n))
    Hk[:, 3:6] = np.eye(3)
    kf['Hk'] = Hk
    kf['xk'] = np.zeros(n)
    kf['I'] = np.eye(n)
    kf['tau_g'] = tau_g
    kf['tau_a'] = tau_a
    return kf


# ═══════════════════════════════════════════════════════════════
#  Ft 矩阵：36 状态（无 tGA，无 rz，Ka 上三角）
#  SS 为 3×9，只取前 6 列（对应 rx 和 ry），rz 固定为零
# ═══════════════════════════════════════════════════════════════
def getFt_36(fb, wb, Cnb, wnie, SS):
    n = 36
    o33 = np.zeros((3, 3))
    wX  = askew(wnie)
    fX  = askew(Cnb @ fb)
    fx, fy, fz = fb[0], fb[1], fb[2]
    wx, wy, wz = wb[0], wb[1], wb[2]
    CDf2 = Cnb @ np.diag(fb**2)

    # dKa 上三角 6 列
    Ca_upper = np.zeros((3, 6))
    Ca_upper[:, 0] = Cnb[:, 0] * fx   # (r=0,c=0)
    Ca_upper[:, 1] = Cnb[:, 0] * fy   # (r=0,c=1)
    Ca_upper[:, 2] = Cnb[:, 0] * fz   # (r=0,c=2)
    Ca_upper[:, 3] = Cnb[:, 1] * fy   # (r=1,c=1)
    Ca_upper[:, 4] = Cnb[:, 1] * fz   # (r=1,c=2)
    Ca_upper[:, 5] = Cnb[:, 2] * fz   # (r=2,c=2)

    Ft = np.zeros((n, n))
    # dφ/dt
    Ft[0:3, 0:3]   = -wX
    Ft[0:3, 6:9]   = -Cnb
    Ft[0:3, 12:15] = -wx * Cnb
    Ft[0:3, 15:18] = -wy * Cnb
    Ft[0:3, 18:21] = -wz * Cnb
    # dδv/dt
    Ft[3:6, 0:3]   = fX
    Ft[3:6, 9:12]  = Cnb
    Ft[3:6, 21:27] = Ca_upper
    Ft[3:6, 27:30] = CDf2
    # 只取 SS 前 6 列 (rx 和 ry 各 3 列，rz 固定为 0 故不估计)
    Ft[3:6, 30:36] = Cnb @ SS[:, 0:6]
    return Ft


# ═══════════════════════════════════════════════════════════════
#  Ft 矩阵：42 状态（一阶 Markov）
# ═══════════════════════════════════════════════════════════════
def getFt_42(fb, wb, Cnb, wnie, SS, tau_g, tau_a):
    n = 42
    I33 = np.eye(3)
    Ft = np.zeros((n, n))
    Ft[0:36, 0:36] = getFt_36(fb, wb, Cnb, wnie, SS)
    Ft[0:3, 36:39] = -Cnb       # bm_g → φ
    Ft[3:6, 39:42] = Cnb        # bm_a → δv
    Ft[36:39, 36:39] = -I33 / tau_g
    Ft[39:42, 39:42] = -I33 / tau_a
    return Ft


# ═══════════════════════════════════════════════════════════════
#  Ft 矩阵：48 状态（二阶 Gauss-Markov）
# ═══════════════════════════════════════════════════════════════
def getFt_48(fb, wb, Cnb, wnie, SS, tau_g, tau_a):
    n = 48
    I33 = np.eye(3)
    beta_g = 1.0 / tau_g
    beta_a = 1.0 / tau_a
    Ft = np.zeros((n, n))
    Ft[0:36, 0:36] = getFt_36(fb, wb, Cnb, wnie, SS)
    Ft[0:3, 36:39]   = -Cnb     # bm_g → φ
    Ft[3:6, 39:42]   = Cnb      # bm_a → δv
    # 二阶 GM 动力学
    Ft[36:39, 42:45] = I33                      # d(bm_g)/dt = dbm_g
    Ft[39:42, 45:48] = I33                      # d(bm_a)/dt = dbm_a
    Ft[42:45, 36:39] = -beta_g**2 * I33         # d(dbm_g)/dt = -β² bm_g
    Ft[42:45, 42:45] = -2 * beta_g * I33        # - 2β dbm_g
    Ft[45:48, 39:42] = -beta_a**2 * I33
    Ft[45:48, 45:48] = -2 * beta_a * I33
    return Ft


# ═══════════════════════════════════════════════════════════════
#  反馈函数（通用，支持 36/42/48）
# ═══════════════════════════════════════════════════════════════
def clbtkffeedback_pruned(kf, clbt, n_states):
    xk = kf['xk']
    dKg = xk[12:21].reshape(3, 3).T
    clbt['Kg'] = (np.eye(3) - dKg) @ clbt['Kg']
    dKa = Ka_from_upper(xk[21:27])
    clbt['Ka'] = (np.eye(3) - dKa) @ clbt['Ka']
    clbt['Ka2'] = clbt['Ka2'] + xk[27:30]
    clbt['eb']  = clbt['eb']  + xk[6:9]
    clbt['db']  = clbt['db']  + xk[9:12]
    clbt['rx']  = clbt['rx']  + xk[30:33]
    clbt['ry']  = clbt['ry']  + xk[33:36]
    if n_states >= 42:
        clbt['eb'] += xk[36:39]   # bm_g
        clbt['db'] += xk[39:42]   # bm_a
    return clbt


# ═══════════════════════════════════════════════════════════════
#  通用标定主循环（支持 36/42/48 状态）
# ═══════════════════════════════════════════════════════════════
def run_calibration(imu1, pos0, ts, n_states=36,
                    bi_g=0.0, tau_g=300.0, bi_a=0.0, tau_a=300.0,
                    label=""):
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

    # rz 固定为零（已知，不估计）
    clbt = {'Kg': np.eye(3), 'Ka': np.eye(3), 'Ka2': np.zeros(3),
            'eb': np.zeros(3), 'db': np.zeros(3),
            'rx': np.zeros(3), 'ry': np.zeros(3)}

    length = len(imu1)
    dotwf  = imudot(imu1, 5.0)
    iterations = 3
    P_trace, X_trace, iter_bounds = [], [], []

    def apply_clbt(imu_s, c):
        res = np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    for it in range(iterations):
        print(f"  [{label}] Iter {it+1}/{iterations}")
        if n_states == 48:
            kf = clbtkfinit_48(nts, bi_g, tau_g, bi_a, tau_a)
        elif n_states == 42:
            kf = clbtkfinit_42(nts, bi_g, tau_g, bi_a, tau_a)
        else:  # 36
            kf = clbtkfinit_36(nts)

        if it == iterations - 1:
            kf['Pxk'] = kf['Pxk'] * 100
            kf['Pxk'][:, 2] = 0; kf['Pxk'][2, :] = 0
            kf['xk'] = np.zeros(n_states)

        imu_align = apply_clbt(imu1[frq2:kstatic, :], clbt)
        _, _, _, qnb = alignsb(imu_align, pos0)
        vn  = np.zeros(3)
        t1s = 0.0

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
            # rz 固定为零，只使用 rx 和 ry 的补偿
            fL  = SS[:, 0:6] @ np.concatenate((clbt['rx'], clbt['ry']))
            fn  = qmulv(qnb, fb - clbt['Ka2']*(fb**2) - fL)
            vn  = vn + (rotv(-wnie*nts/2, fn) + gn) * nts
            qnb = qupdt2(qnb, phim, wnie * nts)

            t1s += nts

            if n_states == 48:
                Ft = getFt_48(fb, wb, q2mat(qnb), wnie, SS, tau_g, tau_a)
            elif n_states == 42:
                Ft = getFt_42(fb, wb, q2mat(qnb), wnie, SS, tau_g, tau_a)
            else:
                Ft = getFt_36(fb, wb, q2mat(qnb), wnie, SS)

            kf['Phikk_1'] = np.eye(n_states) + Ft * nts
            kf = kfupdate(kf, TimeMeasBoth='T')

            if t1s > (0.2 - ts / 2):
                t1s = 0.0
                ww = np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
                if np.linalg.norm(ww) / ts < 20 * glv.dph:
                    kf = kfupdate(kf, yk=vn, TimeMeasBoth='M')

                P_trace.append(np.diag(kf['Pxk']))
                X_trace.append(np.copy(kf['xk']))

        if it != iterations - 1:
            clbt = clbtkffeedback_pruned(kf, clbt, n_states)

        iter_bounds.append(len(P_trace))

    return clbt, kf, np.array(P_trace), np.array(X_trace), iter_bounds


# ═══════════════════════════════════════════════════════════════
#  标签（36 维基础，42/48 在此基础上追加）
# ═══════════════════════════════════════════════════════════════
STATE_LABELS_36 = (
    ['φ_x', 'φ_y', 'φ_z'] +
    ['δv_x', 'δv_y', 'δv_z'] +
    ['eb_x', 'eb_y', 'eb_z'] +
    ['db_x', 'db_y', 'db_z'] +
    ['Kg_xx', 'Kg_yx', 'Kg_zx', 'Kg_xy', 'Kg_yy', 'Kg_zy', 'Kg_xz', 'Kg_yz', 'Kg_zz'] +
    ['Ka_xx', 'Ka_xy', 'Ka_xz', 'Ka_yy', 'Ka_yz', 'Ka_zz'] +
    ['Ka2_x', 'Ka2_y', 'Ka2_z'] +
    ['rx_x', 'rx_y', 'rx_z'] +
    ['ry_x', 'ry_y', 'ry_z']
)

STATE_LABELS_42 = STATE_LABELS_36 + ['bm_gx', 'bm_gy', 'bm_gz', 'bm_ax', 'bm_ay', 'bm_az']

STATE_LABELS_48 = STATE_LABELS_42 + ['dbm_gx', 'dbm_gy', 'dbm_gz', 'dbm_ax', 'dbm_ay', 'dbm_az']


# ═══════════════════════════════════════════════════════════════
#  相关系数热力图
# ═══════════════════════════════════════════════════════════════
def plot_correlation_heatmap(Pxk, title, save_path=None, labels=None):
    n = Pxk.shape[0]
    std = np.sqrt(np.diag(Pxk))
    std = np.where(std < 1e-30, 1e-30, std)
    corr = Pxk / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    fig_size = max(10, n * 0.30)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.85))
    if labels is None:
        labels = [str(i) for i in range(n)]
    seaborn.heatmap(corr, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.3, linecolor='#333',
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
    ax.set_title(title, fontsize=12, pad=10)
    plt.xticks(fontsize=6.5, rotation=90)
    plt.yticks(fontsize=6.5, rotation=0)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='svg', dpi=150, bbox_inches='tight')
        print(f"  Saved heatmap: {save_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════════
#  对比输出
# ═══════════════════════════════════════════════════════════════
def compare_results(clbt_truth, results_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    dKg_truth = clbt_truth['Kg'] - np.eye(3)
    dKa_truth = clbt_truth['Ka'] - np.eye(3)

    params = [
        ("eb_x",   clbt_truth['eb'][0],  lambda c: -c['eb'][0]),
        ("eb_y",   clbt_truth['eb'][1],  lambda c: -c['eb'][1]),
        ("eb_z",   clbt_truth['eb'][2],  lambda c: -c['eb'][2]),
        ("db_x",   clbt_truth['db'][0],  lambda c: -c['db'][0]),
        ("db_y",   clbt_truth['db'][1],  lambda c: -c['db'][1]),
        ("db_z",   clbt_truth['db'][2],  lambda c: -c['db'][2]),
        ("dKg_xx", dKg_truth[0,0], lambda c: -(c['Kg']-np.eye(3))[0,0]),
        ("dKg_yx", dKg_truth[1,0], lambda c: -(c['Kg']-np.eye(3))[1,0]),
        ("dKg_zx", dKg_truth[2,0], lambda c: -(c['Kg']-np.eye(3))[2,0]),
        ("dKg_xy", dKg_truth[0,1], lambda c: -(c['Kg']-np.eye(3))[0,1]),
        ("dKg_yy", dKg_truth[1,1], lambda c: -(c['Kg']-np.eye(3))[1,1]),
        ("dKg_zy", dKg_truth[2,1], lambda c: -(c['Kg']-np.eye(3))[2,1]),
        ("dKg_xz", dKg_truth[0,2], lambda c: -(c['Kg']-np.eye(3))[0,2]),
        ("dKg_yz", dKg_truth[1,2], lambda c: -(c['Kg']-np.eye(3))[1,2]),
        ("dKg_zz", dKg_truth[2,2], lambda c: -(c['Kg']-np.eye(3))[2,2]),
        ("dKa_xx", dKa_truth[0,0], lambda c: -(c['Ka']-np.eye(3))[0,0]),
        ("dKa_xy", dKa_truth[0,1], lambda c: -(c['Ka']-np.eye(3))[0,1]),
        ("dKa_xz", dKa_truth[0,2], lambda c: -(c['Ka']-np.eye(3))[0,2]),
        ("dKa_yy", dKa_truth[1,1], lambda c: -(c['Ka']-np.eye(3))[1,1]),
        ("dKa_yz", dKa_truth[1,2], lambda c: -(c['Ka']-np.eye(3))[1,2]),
        ("dKa_zz", dKa_truth[2,2], lambda c: -(c['Ka']-np.eye(3))[2,2]),
        ("Ka2_x",  clbt_truth['Ka2'][0], lambda c: -c['Ka2'][0]),
        ("Ka2_y",  clbt_truth['Ka2'][1], lambda c: -c['Ka2'][1]),
        ("Ka2_z",  clbt_truth['Ka2'][2], lambda c: -c['Ka2'][2]),
        ("rx_x",   clbt_truth['rx'][0],  lambda c: -c['rx'][0]),
        ("rx_y",   clbt_truth['rx'][1],  lambda c: -c['rx'][1]),
        ("rx_z",   clbt_truth['rx'][2],  lambda c: -c['rx'][2]),
        ("ry_x",   clbt_truth['ry'][0],  lambda c: -c['ry'][0]),
        ("ry_y",   clbt_truth['ry'][1],  lambda c: -c['ry'][1]),
        ("ry_z",   clbt_truth['ry'][2],  lambda c: -c['ry'][2]),
    ]

    labels = list(results_dict.keys())
    n_params = len(params)
    n_labels = len(labels)

    err_pct = np.zeros((n_params, n_labels))
    for pi, (name, true_v, get_est) in enumerate(params):
        for li, label in enumerate(labels):
            est_v = get_est(results_dict[label][0])
            if abs(true_v) > 1e-15:
                err_pct[pi, li] = abs(true_v - est_v) / abs(true_v) * 100
            else:
                err_pct[pi, li] = 0.0

    print("\n" + "=" * 120)
    print("CALIBRATION ACCURACY COMPARISON (Error %)")
    print("=" * 120)
    header = f"{'Param':<12}" + "".join([f"{lb:>22}" for lb in labels])
    print(header); print("-" * 120)
    for pi, (name, _, _) in enumerate(params):
        row = f"{name:<12}"
        for li in range(n_labels):
            row += f"{err_pct[pi, li]:>20.2f}%"
        print(row)
    print("=" * 120)

    x = np.arange(n_params)
    width = 0.75 / n_labels
    colors = ['steelblue', 'tomato', 'forestgreen', 'darkorange']
    fig, ax = plt.subplots(figsize=(22, 8))
    for li, label in enumerate(labels):
        offset = (li - n_labels/2 + 0.5) * width
        ax.bar(x + offset, err_pct[:, li], width, label=label,
               color=colors[li % len(colors)], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in params], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Calibration Error (%)')
    ax.set_title('Pruned (36/42/48 state, Upper Ka, No tGA, No rz) Calibration Accuracy')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    svg_path = os.path.join(out_dir, 'markov_model_comparison_36_42_48.svg')
    plt.savefig(svg_path, format='svg')
    plt.close()
    print(f"\nSaved chart to '{svg_path}'")
    return svg_path


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════
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
        [18,   0, 0,-1,  90, 9, 20, 20],
    ], dtype=float)
    paras[:, 4] = paras[:, 4] * glv.deg

    ARW   = 0.01  * glv.dpsh
    VRW   = 10.0  * glv.ugpsHz
    BI_G  = 0.005 * glv.dph
    BI_A  = 10.0  * glv.ug
    TAU_G = 300.0
    TAU_A = 300.0

    print("=" * 60)
    print("Pruned Model: 36/42/48 State (Upper Ka, No tGA, No rz)")
    print(f"  ARW  = {ARW/glv.dpsh:.4f} deg/sqrt(h)")
    print(f"  VRW  = {VRW/glv.ugpsHz:.2f} ug/sqrt(Hz)")
    print(f"  BI_g = {BI_G/glv.dph:.4f} deg/h  (tau={TAU_G}s)")
    print(f"  BI_a = {BI_A/glv.ug:.2f} ug       (tau={TAU_A}s)")
    print("=" * 60)

    print("\nGenerating IMU trajectory...")
    att  = attrottt(att0, paras, ts)
    imu, _ = avp2imu(att, pos0)
    clbt_truth = get_default_clbt()
    imu_clean  = imuclbt(imu, clbt_truth)

    imu_noisy = imuadderr_full(imu_clean, ts,
                                arw=ARW, vrw=VRW,
                                bi_g=BI_G, tau_g=TAU_G,
                                bi_a=BI_A, tau_a=TAU_A, seed=42)

    # ── A: 无噪声 (理想) + 36 状态 ──
    print("\n[A] Clean (no noise, 36-state)...")
    res_A = run_calibration(imu_clean, pos0, ts, n_states=36, label="Clean-36")

    # ── B: 全噪声 + 36 状态（不建模 Markov）──
    print("\n[B] Full noise, 36-state (no Markov model)...")
    res_B = run_calibration(imu_noisy, pos0, ts, n_states=36, label="36-state")

    # ── C: 全噪声 + 42 状态（一阶 Markov，正确 tau）──
    print("\n[C] Full noise, 42-state (1st-order GM, correct tau)...")
    res_C = run_calibration(imu_noisy, pos0, ts, n_states=42,
                             bi_g=BI_G, tau_g=TAU_G, bi_a=BI_A, tau_a=TAU_A,
                             label="42-GM1")

    # ── D: 全噪声 + 48 状态（二阶 Gauss-Markov，正确 tau）──
    print("\n[D] Full noise, 48-state (2nd-order GM, correct tau)...")
    res_D = run_calibration(imu_noisy, pos0, ts, n_states=48,
                             bi_g=BI_G, tau_g=TAU_G, bi_a=BI_A, tau_a=TAU_A,
                             label="48-GM2")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots_markov_pruned')
    results = {
        "A: Clean-36": res_A[0],
        "B: 36-state": res_B[0],
        "C: 42-GM1":   res_C[0],
        "D: 48-GM2":   res_D[0],
    }
    # Pass clbt_truth and the dict of calibration results to compare_results.
    # Note: compare_results gets `(clbt)` because we index `[0]` below! Wait, currently compare_results expects a dict where values are pairs, so I'll wrap them back.
    results_compat = {
        "A: Clean-36": (res_A[0], res_A[1]),
        "B: 36-state": (res_B[0], res_B[1]),
        "C: 42-GM1":   (res_C[0], res_C[1]),
        "D: 48-GM2":   (res_D[0], res_D[1]),
    }
    compare_results(clbt_truth, results_compat, out_dir)

    # ── Convergence Plots for A vs B ──
    print("\nGenerating convergence comparison plots (A vs B) -> 'plots_noise_compare_svg/'")
    svg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots_noise_compare_svg')
    os.makedirs(svg_dir, exist_ok=True)
    
    P_A, X_A, bounds_A = res_A[2], res_A[3], res_A[4]
    P_B, X_B, bounds_B = res_B[2], res_B[3], res_B[4]
    
    # Use 2 * ts as plotting dt
    # Ensure length matches
    min_len = min(len(P_A), len(P_B))
    time_arr = np.arange(min_len) * (2 * ts)
    
    # For A vs B (both 36 states)
    for i in range(36):
        name = STATE_LABELS_36[i]
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

        # Std Dev
        ax = axes[0]
        ax.plot(time_arr, np.sqrt(P_A[:min_len, i]), color='blue', label='Ideal (No Noise)')
        ax.plot(time_arr, np.sqrt(P_B[:min_len, i]), color='red',  label='With Full Noise', linestyle='--')
        for b in bounds_A[:-1]:
            b_val = min(b, min_len-1)
            ax.axvline(x=b_val * 2 * ts, color='gray', linestyle=':', alpha=0.6)
        ax.set_title(f'[{i:02d}] {name}  —  Uncertainty (std dev)')
        ax.set_ylabel('Std Dev')
        ax.legend(); ax.grid(True)

        # State Estimate
        ax2 = axes[1]
        ax2.plot(time_arr, X_A[:min_len, i], color='blue', label='Ideal (No Noise)')
        ax2.plot(time_arr, X_B[:min_len, i], color='red', label='With Full Noise', linestyle='--')
        for b in bounds_A[:-1]:
            b_val = min(b, min_len-1)
            ax2.axvline(x=b_val * 2 * ts, color='gray', linestyle=':', alpha=0.6)
        ax2.set_title(f'[{i:02d}] {name}  —  State Estimate')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Estimated Value')
        ax2.legend(); ax2.grid(True)

        plt.tight_layout()
        safe_name = name.replace('/', '_').replace('\\', '_')
        plt.savefig(os.path.join(svg_dir, f"{i:02d}_{safe_name}.svg"), format='svg')
        plt.close()
    print(f"Saved 36 comparison plots to '{svg_dir}/'")

    # ── Correlation Heatmaps ──
    print("\nGenerating correlation heatmaps...")
    os.makedirs(out_dir, exist_ok=True)
    plot_correlation_heatmap(res_A[1]['Pxk'],
                             "36-State Clean (ideal, no noise)",
                             save_path=os.path.join(out_dir, 'correlation_heatmap_36_State_Clean.svg'),
                             labels=STATE_LABELS_36)
    plot_correlation_heatmap(res_B[1]['Pxk'],
                             "36-State Noisy (full noise, no Markov model)",
                             save_path=os.path.join(out_dir, 'correlation_heatmap_36_State_Noisy.svg'),
                             labels=STATE_LABELS_36)
    plot_correlation_heatmap(res_C[1]['Pxk'],
                             "42-State GM1 (1st-order Gauss-Markov)",
                             save_path=os.path.join(out_dir, 'correlation_heatmap_42_State_GM1.svg'),
                             labels=STATE_LABELS_42)
    plot_correlation_heatmap(res_D[1]['Pxk'],
                             "48-State GM2 (2nd-order Gauss-Markov)",
                             save_path=os.path.join(out_dir, 'correlation_heatmap_48_State_GM2.svg'),
                             labels=STATE_LABELS_48)
    print("\nDone! All outputs in:", out_dir)


if __name__ == "__main__":
    main()
