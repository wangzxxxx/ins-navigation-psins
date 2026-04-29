"""
test_calibration_markov_noise.py
---------------------------------
扩维 Kalman 滤波器：43 → 49 状态
  新增状态 [43:46] = 陀螺 Gauss-Markov 零偏 bm_g
  新增状态 [46:49] = 加计 Gauss-Markov 零偏 bm_a

对比五种条件：
  A: 无噪声（理想）
  B: 有全噪声 + 43 状态模型（不建模 Markov）
  C: 有全噪声 + 49 状态模型（一阶 Markov，正确 tau）
  D: 有全噪声 + 49 状态模型（一阶 Markov，错误 tau）
  E: 有全噪声 + 55 状态模型（二阶 Gauss-Markov，正确 tau）
"""
import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psins_py.nav_utils import glv, posset, Earth
from psins_py.imu_utils import attrottt, avp2imu, imuclbt, imudot, cnscl, imulvS
from psins_py.kf_utils import kfupdate, clbtkffeedback, alignsb, nnts
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
#  标定参数真值
# ═══════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════
#  43 状态滤波器初始化（原始，不建模 Markov）
# ═══════════════════════════════════════════════════════════════
def clbtkfinit_43(nts):
    kf = {}
    kf['nts'] = nts
    kf['n'] = 43
    kf['m'] = 3

    qvec = np.zeros(43)
    qvec[0:3] = 0.01 * glv.dpsh
    qvec[3:6] = 100 * glv.ugpsHz
    kf['Qk'] = np.diag(qvec)**2 * nts

    kf['Rk'] = np.diag([0.001, 0.001, 0.001])**2

    pvec = np.zeros(43)
    pvec[0:3] = np.array([0.1, 0.1, 1.0]) * glv.deg
    pvec[3:6] = 1.0
    pvec[6:9] = 0.1 * glv.dph
    pvec[9:12] = 1.0 * glv.mg
    pvec[12:15] = [100*glv.ppm, 100*glv.sec, 100*glv.sec]
    pvec[15:18] = [100*glv.sec, 100*glv.ppm, 100*glv.sec]
    pvec[18:21] = [100*glv.sec, 100*glv.sec, 100*glv.ppm]
    pvec[21:24] = [100*glv.ppm, 100*glv.sec, 100*glv.sec]
    pvec[24:27] = [0, 100*glv.ppm, 100*glv.sec]
    pvec[27:30] = [0, 0, 100*glv.ppm]
    pvec[30:33] = 100 * glv.ugpg2
    pvec[33:36] = 0.1
    pvec[36:39] = 0.1
    pvec[39:42] = 0.0
    pvec[42] = 0.01
    kf['Pxk'] = np.diag(pvec)**2

    Hk = np.zeros((3, 43))
    Hk[:, 3:6] = np.eye(3)
    kf['Hk'] = Hk
    kf['xk'] = np.zeros(43)
    kf['I'] = np.eye(43)
    return kf


# ═══════════════════════════════════════════════════════════════
#  49 状态滤波器初始化（扩维，建模 Markov 零偏不稳定性）
#
#  状态布局：
#  [0:3]   姿态误差 φ
#  [3:6]   速度误差 δv
#  [6:9]   陀螺常值零偏 eb
#  [9:12]  加计常值零偏 db
#  [12:21] dKg (3×3)
#  [21:30] dKa (3×3)
#  [30:33] Ka2
#  [33:36] rx
#  [36:39] ry
#  [39:42] rz
#  [42]    tGA
#  [43:46] 陀螺 Markov 零偏 bm_g  ← NEW
#  [46:49] 加计 Markov 零偏 bm_a  ← NEW
# ═══════════════════════════════════════════════════════════════
def clbtkfinit_49(nts, bi_g, tau_g, bi_a, tau_a):
    kf = {}
    kf['nts'] = nts
    kf['n'] = 49
    kf['m'] = 3

    # Q 矩阵：前 43 维同原始，后 6 维是 Markov 驱动噪声
    qvec = np.zeros(49)
    qvec[0:3] = 0.01 * glv.dpsh
    qvec[3:6] = 100 * glv.ugpsHz
    # Markov 驱动噪声：sigma_w = sigma_bi * sqrt(2/(tau*nts))
    # 但在离散形式里 Qk_markov = sigma_bi^2 * (1 - exp(-2*nts/tau))
    q_markov_g = bi_g * math.sqrt(1 - math.exp(-2*nts/tau_g)) if tau_g > 0 else 0.0
    q_markov_a = bi_a * math.sqrt(1 - math.exp(-2*nts/tau_a)) if tau_a > 0 else 0.0
    qvec[43:46] = q_markov_g
    qvec[46:49] = q_markov_a
    kf['Qk'] = np.diag(qvec)**2  # 注意：不乘 nts，因为 Markov Q 已经是离散形式

    # 但 ARW/VRW 的 Q 需要乘 nts（连续转离散）
    kf['Qk'][0:6, 0:6] = np.diag(qvec[0:6])**2 * nts

    kf['Rk'] = np.diag([0.001, 0.001, 0.001])**2

    pvec = np.zeros(49)
    pvec[0:3] = np.array([0.1, 0.1, 1.0]) * glv.deg
    pvec[3:6] = 1.0
    pvec[6:9] = 0.1 * glv.dph
    pvec[9:12] = 1.0 * glv.mg
    pvec[12:15] = [100*glv.ppm, 100*glv.sec, 100*glv.sec]
    pvec[15:18] = [100*glv.sec, 100*glv.ppm, 100*glv.sec]
    pvec[18:21] = [100*glv.sec, 100*glv.sec, 100*glv.ppm]
    pvec[21:24] = [100*glv.ppm, 100*glv.sec, 100*glv.sec]
    pvec[24:27] = [0, 100*glv.ppm, 100*glv.sec]
    pvec[27:30] = [0, 0, 100*glv.ppm]
    pvec[30:33] = 100 * glv.ugpg2
    pvec[33:36] = 0.1
    pvec[36:39] = 0.1
    pvec[39:42] = 0.0
    pvec[42] = 0.01
    # Markov 零偏初始不确定度 = 稳态标准差
    pvec[43:46] = bi_g   # 陀螺 Markov
    pvec[46:49] = bi_a   # 加计 Markov
    kf['Pxk'] = np.diag(pvec)**2

    Hk = np.zeros((3, 49))
    Hk[:, 3:6] = np.eye(3)
    kf['Hk'] = Hk
    kf['xk'] = np.zeros(49)
    kf['I'] = np.eye(49)

    # 保存 Markov 参数供 Ft 计算使用
    kf['tau_g'] = tau_g
    kf['tau_a'] = tau_a
    return kf


# ═══════════════════════════════════════════════════════════════
#  43 状态 Ft 矩阵（原始）
# ═══════════════════════════════════════════════════════════════
def getFt_43(fb, wb, Cnb, wnie, SS):
    o33 = np.zeros((3, 3))
    o31 = np.zeros((3, 1))
    wX = askew(wnie)
    fX = askew(Cnb @ fb)
    wx, wy, wz = wb[0], wb[1], wb[2]
    fx, fy, fz = fb[0], fb[1], fb[2]
    CDf2 = Cnb @ np.diag(fb**2)
    CwXf = Cnb @ np.cross(wb, fb)
    CwXf = CwXf.reshape(3, 1)

    row1 = np.hstack([-wX, o33, -Cnb, o33, -wx*Cnb, -wy*Cnb, -wz*Cnb, o33, o33, o33, o33, o33, o33, o33, o31])
    row2 = np.hstack([fX, o33, o33, Cnb, o33, o33, o33, fx*Cnb, fy*Cnb, fz*Cnb, CDf2, Cnb@SS, CwXf])
    row3 = np.zeros((37, 43))
    return np.vstack([row1, row2, row3])


# ═══════════════════════════════════════════════════════════════
#  49 状态 Ft 矩阵（扩维）
#
#  Markov 零偏影响：
#    dφ/dt = ... - Cnb * (eb + bm_g)   → bm_g 与 eb 具有相同的耦合项
#    dv/dt = ... + Cnb * (db + bm_a)   → bm_a 与 db 具有相同的耦合项
#    dbm_g/dt = -1/tau_g * bm_g + w_g  → 自衰减
#    dbm_a/dt = -1/tau_a * bm_a + w_a  → 自衰减
# ═══════════════════════════════════════════════════════════════
def getFt_49(fb, wb, Cnb, wnie, SS, tau_g, tau_a):
    o33 = np.zeros((3, 3))
    o31 = np.zeros((3, 1))
    o36 = np.zeros((3, 6))  # 新增列
    wX = askew(wnie)
    fX = askew(Cnb @ fb)
    wx, wy, wz = wb[0], wb[1], wb[2]
    fx, fy, fz = fb[0], fb[1], fb[2]
    CDf2 = Cnb @ np.diag(fb**2)
    CwXf = Cnb @ np.cross(wb, fb)
    CwXf = CwXf.reshape(3, 1)

    # Row 1: dφ/dt  (1×49)
    # 原来43列 + 新增6列: [-Cnb(bm_g对φ), 0(bm_a对φ)]
    row1_43 = np.hstack([-wX, o33, -Cnb, o33, -wx*Cnb, -wy*Cnb, -wz*Cnb, o33, o33, o33, o33, o33, o33, o33, o31])
    row1_ext = np.hstack([-Cnb, o33])  # bm_g 和 eb 具有相同耦合
    row1 = np.hstack([row1_43, row1_ext])

    # Row 2: dv/dt  (1×49)
    # 原来43列 + 新增6列: [0(bm_g对v), Cnb(bm_a对v)]
    row2_43 = np.hstack([fX, o33, o33, Cnb, o33, o33, o33, fx*Cnb, fy*Cnb, fz*Cnb, CDf2, Cnb@SS, CwXf])
    row2_ext = np.hstack([o33, Cnb])  # bm_a 和 db 具有相同耦合
    row2 = np.hstack([row2_43, row2_ext])

    # Row 3: 原来的 37 个参数状态（常值）+ 6 新列全零
    row3_43 = np.zeros((37, 43))
    row3_ext = np.zeros((37, 6))
    row3 = np.hstack([row3_43, row3_ext])

    # Row 4: Markov 状态的自衰减  (6×49)
    # dbm_g/dt = -1/tau_g * bm_g
    # dbm_a/dt = -1/tau_a * bm_a
    row4 = np.zeros((6, 49))
    row4[0:3, 43:46] = -np.eye(3) / tau_g  # bm_g 自衰减
    row4[3:6, 46:49] = -np.eye(3) / tau_a  # bm_a 自衰减

    return np.vstack([row1, row2, row3, row4])


# ═══════════════════════════════════════════════════════════════
#  扩维 feedback：迭代时也回收 Markov 零偏估计
# ═══════════════════════════════════════════════════════════════
def clbtkffeedback_49(kf, clbt):
    """同 43 状态 feedback，额外把 bm_g/bm_a 累加到 eb/db 里"""
    clbt['Kg']  = (np.eye(3) - kf['xk'][12:21].reshape(3,3).T) @ clbt['Kg']
    clbt['Ka']  = (np.eye(3) - kf['xk'][21:30].reshape(3,3).T) @ clbt['Ka']
    clbt['Ka2'] = clbt['Ka2'] + kf['xk'][30:33]
    # eb/db = 常值零偏 + Markov 零偏估计
    clbt['eb']  = clbt['eb']  + kf['xk'][6:9]  + kf['xk'][43:46]
    clbt['db']  = clbt['db']  + kf['xk'][9:12] + kf['xk'][46:49]
    clbt['rx']  = clbt['rx']  + kf['xk'][33:36]
    clbt['ry']  = clbt['ry']  + kf['xk'][36:39]
    clbt['rz']  = clbt['rz']  + kf['xk'][39:42]
    clbt['tGA'] = clbt['tGA'] + kf['xk'][42]
    return clbt


# ═══════════════════════════════════════════════════════════════
#  55 状态二阶 Gauss-Markov 模型
#
#  二阶 GM 状态空间：
#    d(bm)/dt   = dbm            (零偏的变化率)
#    d(dbm)/dt  = -β² bm - 2β dbm + w    (β = 1/τ)
#
#  与一阶的区别：二阶模型的零偏漂移是“平滑”的，不会突变，
#  而是通过加速/减速过程来变化，更符合真实物理过程。
#
#  状态布局：
#    [0:43]   原始 43 维
#    [43:46]  陀螺 Markov 零偏 bm_g
#    [46:49]  加计 Markov 零偏 bm_a
#    [49:52]  陀螺 Markov 零偏变化率 dbm_g  ← NEW
#    [52:55]  加计 Markov 零偏变化率 dbm_a  ← NEW
# ═══════════════════════════════════════════════════════════════
def clbtkfinit_55(nts, bi_g, tau_g, bi_a, tau_a):
    kf = {}
    kf['nts'] = nts
    kf['n'] = 55
    kf['m'] = 3
    beta_g = 1.0 / tau_g if tau_g > 0 else 0.0
    beta_a = 1.0 / tau_a if tau_a > 0 else 0.0

    # Q 矩阵：前 43 维 ARW/VRW，[43:49]零偏无驱动，[49:55]速率有驱动噪声
    qvec = np.zeros(55)
    qvec[0:3] = 0.01 * glv.dpsh
    qvec[3:6] = 100 * glv.ugpsHz
    # 二阶 GM 的驱动噪声只作用在速率状态 [49:55]
    # 离散形式 Q_rate ≈ sigma_bi^2 * (2*beta)^3 * nts  (二阶 GM 的功率谱密度)
    q_rate_g = bi_g * (2*beta_g)**1.5 * math.sqrt(nts) if beta_g > 0 else 0.0
    q_rate_a = bi_a * (2*beta_a)**1.5 * math.sqrt(nts) if beta_a > 0 else 0.0
    qvec[49:52] = q_rate_g
    qvec[52:55] = q_rate_a
    kf['Qk'] = np.diag(qvec)**2
    # ARW/VRW 需要乘 nts
    kf['Qk'][0:6, 0:6] = np.diag(qvec[0:6])**2 * nts

    kf['Rk'] = np.diag([0.001, 0.001, 0.001])**2

    pvec = np.zeros(55)
    pvec[0:3] = np.array([0.1, 0.1, 1.0]) * glv.deg
    pvec[3:6] = 1.0
    pvec[6:9] = 0.1 * glv.dph
    pvec[9:12] = 1.0 * glv.mg
    pvec[12:15] = [100*glv.ppm, 100*glv.sec, 100*glv.sec]
    pvec[15:18] = [100*glv.sec, 100*glv.ppm, 100*glv.sec]
    pvec[18:21] = [100*glv.sec, 100*glv.sec, 100*glv.ppm]
    pvec[21:24] = [100*glv.ppm, 100*glv.sec, 100*glv.sec]
    pvec[24:27] = [0, 100*glv.ppm, 100*glv.sec]
    pvec[27:30] = [0, 0, 100*glv.ppm]
    pvec[30:33] = 100 * glv.ugpg2
    pvec[33:36] = 0.1
    pvec[36:39] = 0.1
    pvec[39:42] = 0.0
    pvec[42] = 0.01
    pvec[43:46] = bi_g      # Markov 零偏的稳态标准差
    pvec[46:49] = bi_a
    pvec[49:52] = bi_g * beta_g  # 零偏变化率的初始不确定度
    pvec[52:55] = bi_a * beta_a
    kf['Pxk'] = np.diag(pvec)**2

    Hk = np.zeros((3, 55))
    Hk[:, 3:6] = np.eye(3)
    kf['Hk'] = Hk
    kf['xk'] = np.zeros(55)
    kf['I'] = np.eye(55)
    kf['tau_g'] = tau_g
    kf['tau_a'] = tau_a
    return kf


def getFt_55(fb, wb, Cnb, wnie, SS, tau_g, tau_a):
    """55 状态 Ft：二阶 Gauss-Markov 动力学"""
    o33 = np.zeros((3, 3))
    o31 = np.zeros((3, 1))
    I33 = np.eye(3)
    wX = askew(wnie)
    fX = askew(Cnb @ fb)
    wx, wy, wz = wb[0], wb[1], wb[2]
    fx, fy, fz = fb[0], fb[1], fb[2]
    CDf2 = Cnb @ np.diag(fb**2)
    CwXf = Cnb @ np.cross(wb, fb)
    CwXf = CwXf.reshape(3, 1)
    beta_g = 1.0 / tau_g
    beta_a = 1.0 / tau_a

    # Row 1: dφ/dt  (1×55)
    row1_43 = np.hstack([-wX, o33, -Cnb, o33, -wx*Cnb, -wy*Cnb, -wz*Cnb,
                         o33, o33, o33, o33, o33, o33, o33, o31])
    row1_ext = np.hstack([-Cnb, o33, o33, o33])  # bm_g 耦合，dbm 不直接耦合
    row1 = np.hstack([row1_43, row1_ext])

    # Row 2: dv/dt  (1×55)
    row2_43 = np.hstack([fX, o33, o33, Cnb, o33, o33, o33,
                         fx*Cnb, fy*Cnb, fz*Cnb, CDf2, Cnb@SS, CwXf])
    row2_ext = np.hstack([o33, Cnb, o33, o33])   # bm_a 耦合
    row2 = np.hstack([row2_43, row2_ext])

    # Row 3: 原始 37 个参数状态（常值）
    row3 = np.zeros((37, 55))

    # Row 4: 二阶 GM 状态 (12×55)
    # d(bm_g)/dt = dbm_g
    # d(bm_a)/dt = dbm_a
    # d(dbm_g)/dt = -β_g² bm_g - 2β_g dbm_g
    # d(dbm_a)/dt = -β_a² bm_a - 2β_a dbm_a
    row4 = np.zeros((12, 55))
    # bm_g' = dbm_g
    row4[0:3, 49:52] = I33               # bm_g ← dbm_g
    # bm_a' = dbm_a
    row4[3:6, 52:55] = I33               # bm_a ← dbm_a
    # dbm_g' = -β² bm_g - 2β dbm_g
    row4[6:9, 43:46] = -beta_g**2 * I33  # dbm_g ← bm_g
    row4[6:9, 49:52] = -2*beta_g * I33   # dbm_g ← dbm_g
    # dbm_a' = -β² bm_a - 2β dbm_a
    row4[9:12, 46:49] = -beta_a**2 * I33
    row4[9:12, 52:55] = -2*beta_a * I33

    return np.vstack([row1, row2, row3, row4])


def clbtkffeedback_55(kf, clbt):
    """同 49 状态 feedback，Markov 零偏累加到 eb/db"""
    clbt['Kg']  = (np.eye(3) - kf['xk'][12:21].reshape(3,3).T) @ clbt['Kg']
    clbt['Ka']  = (np.eye(3) - kf['xk'][21:30].reshape(3,3).T) @ clbt['Ka']
    clbt['Ka2'] = clbt['Ka2'] + kf['xk'][30:33]
    clbt['eb']  = clbt['eb']  + kf['xk'][6:9]  + kf['xk'][43:46]
    clbt['db']  = clbt['db']  + kf['xk'][9:12] + kf['xk'][46:49]
    clbt['rx']  = clbt['rx']  + kf['xk'][33:36]
    clbt['ry']  = clbt['ry']  + kf['xk'][36:39]
    clbt['rz']  = clbt['rz']  + kf['xk'][39:42]
    clbt['tGA'] = clbt['tGA'] + kf['xk'][42]
    return clbt


# ═══════════════════════════════════════════════════════════════
#  通用标定主循环，支持 43、49、55 状态以及 RTS 附加选项
# ═══════════════════════════════════════════════════════════════
def run_calibration(imu1, pos0, ts, n_states=43,
                    bi_g=0.0, tau_g=3600.0, bi_a=0.0, tau_a=3600.0,
                    use_rts=False, label=""):
    """Support 43 / 49(first-order GM) / 55(second-order GM) states"""
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

    length     = len(imu1)
    dotwf      = imudot(imu1, 5.0)
    iterations = 3

    def apply_clbt(imu_s, c):
        res = np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    for it in range(iterations):
        print(f"  [{label}] Iter {it+1}/{iterations}")
        if n_states == 55:
            kf = clbtkfinit_55(nts, bi_g, tau_g, bi_a, tau_a)
        elif n_states == 49:
            kf = clbtkfinit_49(nts, bi_g, tau_g, bi_a, tau_a)
        else:
            kf = clbtkfinit_43(nts)

        if it == iterations - 1:
            kf['Pxk'] = kf['Pxk'] * 100
            kf['Pxk'][:, 2] = 0; kf['Pxk'][2, :] = 0
            kf['xk'] = np.zeros(n_states)
            
            if use_rts:
                num_steps = (length - frq2 - 2 * frq2) // nn + 2
                history = {
                    'Phi': np.zeros((num_steps, n_states, n_states)),
                    'P_prior': np.zeros((num_steps, n_states, n_states)),
                    'X_prior': np.zeros((num_steps, n_states)),
                    'P_post': np.zeros((num_steps, n_states, n_states)),
                    'X_post': np.zeros((num_steps, n_states)),
                }
                step = 0

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
            fL  = SS @ np.concatenate((clbt['rx'], clbt['ry'], clbt['rz']))
            fn  = qmulv(qnb, fb - clbt['Ka2']*(fb**2) - fL - clbt['tGA']*np.cross(wb, fb))
            vn  = vn + (rotv(-wnie*nts/2, fn) + gn) * nts
            qnb = qupdt2(qnb, phim, wnie * nts)

            t1s += nts
            if n_states == 55:
                Ft = getFt_55(fb, wb, q2mat(qnb), wnie, SS, tau_g, tau_a)
            elif n_states == 49:
                Ft = getFt_49(fb, wb, q2mat(qnb), wnie, SS, tau_g, tau_a)
            else:
                Ft = getFt_43(fb, wb, q2mat(qnb), wnie, SS)

            kf['Phikk_1'] = np.eye(n_states) + Ft * nts
            
            # --- RTS: 记录先验 ---
            if it == iterations - 1 and use_rts:
                Phi = kf['Phikk_1']
                X_prior = Phi @ kf['xk']
                P_prior = Phi @ kf['Pxk'] @ Phi.T + kf['Qk']
                
            kf = kfupdate(kf, TimeMeasBoth='T')

            if t1s > (0.2 - ts / 2):
                t1s = 0.0
                ww  = np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
                if np.linalg.norm(ww) / ts < 20 * glv.dph:
                    kf = kfupdate(kf, yk=vn, TimeMeasBoth='M')
            
            # --- RTS: 记录后验 ---
            if it == iterations - 1 and use_rts:
                history['Phi'][step]     = Phi
                history['P_prior'][step] = P_prior
                history['X_prior'][step] = X_prior
                history['P_post'][step]  = np.copy(kf['Pxk'])
                history['X_post'][step]  = np.copy(kf['xk'])
                step += 1

        if it != iterations - 1:
            if n_states == 55:
                clbt = clbtkffeedback_55(kf, clbt)
            elif n_states == 49:
                clbt = clbtkffeedback_49(kf, clbt)
            else:
                clbt = clbtkffeedback(kf, clbt)

    return clbt, kf['Pxk']


# ═══════════════════════════════════════════════════════════════
#  画图：相关系数矩阵透视热力图
# ═══════════════════════════════════════════════════════════════
def plot_correlation_matrix(P, n_states, out_dir, label):
    # Calculate Correlation Matrix: Corr(i,j) = P(i,j) / sqrt(P(i,i)*P(j,j))
    std_devs = np.sqrt(np.diag(P))
    # Avoid division by zero for perfectly known/zero-variance states
    std_devs[std_devs == 0] = 1e-12 
    
    Outer_std = np.outer(std_devs, std_devs)
    Corr = P / Outer_std
    
    # 截取全部可能的状态（最长 49 或者 55）
    limit = n_states
    Corr_sub = Corr[:limit, :limit]
    
    state_names = [
        "phi_x", "phi_y", "phi_z", "dv_x", "dv_y", "dv_z", 
        "eb_x", "eb_y", "eb_z", "db_x", "db_y", "db_z",
        "dKg_xx", "dKg_yx", "dKg_zx", "dKg_xy", "dKg_yy", "dKg_zy", "dKg_xz", "dKg_yz", "dKg_zz",
        "dKa_xx", "dKa_yx", "dKa_zx", "dKa_xy", "dKa_yy", "dKa_zy", "dKa_xz", "dKa_yz", "dKa_zz",
        "Ka2_x", "Ka2_y", "Ka2_z",
        "rx_x", "rx_y", "rx_z", "ry_x", "ry_y", "ry_z",
        "rz_x", "rz_y", "rz_z", "tGA"
    ]
    # 如果系统是 49 或 55，加上马尔可夫噪声一阶状态
    if n_states >= 49:
        state_names.extend(["bm_g_x", "bm_g_y", "bm_g_z", "bm_a_x", "bm_a_y", "bm_a_z"])
    # 如果系统是 55，再加上二阶变化率状态
    if n_states >= 55:
        state_names.extend(["dbm_g_x", "dbm_g_y", "dbm_g_z", "dbm_a_x", "dbm_a_y", "dbm_a_z"])
        
    names_sub = state_names[:limit]
    
    plt.figure(figsize=(20, 18))
    sns.heatmap(Corr_sub, annot=False, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                xticklabels=names_sub, yticklabels=names_sub, square=True,
                linewidths=.5, cbar_kws={"shrink": .7})
                
    title = f"Correlation Matrix Heatmap ({label})"
    plt.title(title, fontsize=16, pad=20)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    
    os.makedirs(out_dir, exist_ok=True)
    filename = f"correlation_heatmap_{label.replace(' ', '_')}.svg"
    plt.savefig(os.path.join(out_dir, filename), format='svg')
    plt.close()
    print(f"Saved correlation heatmap to '{out_dir}/{filename}'")


# ═══════════════════════════════════════════════════════════════
#  对比输出
# ═══════════════════════════════════════════════════════════════
def compare_results(clbt_truth, results_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    dKg_truth = clbt_truth['Kg'] - np.eye(3)
    dKa_truth = clbt_truth['Ka'] - np.eye(3)

    params = [
        # eb (陀螺零偏)
        ("eb_x",    clbt_truth['eb'][0],   lambda c: -c['eb'][0]),
        ("eb_y",    clbt_truth['eb'][1],   lambda c: -c['eb'][1]),
        ("eb_z",    clbt_truth['eb'][2],   lambda c: -c['eb'][2]),
        # db (加计零偏)
        ("db_x",    clbt_truth['db'][0],   lambda c: -c['db'][0]),
        ("db_y",    clbt_truth['db'][1],   lambda c: -c['db'][1]),
        ("db_z",    clbt_truth['db'][2],   lambda c: -c['db'][2]),
        # dKg (陀螺标度因数及安装角)
        ("dKg_xx",  dKg_truth[0,0],  lambda c: -(c['Kg']-np.eye(3))[0,0]),
        ("dKg_yx",  dKg_truth[1,0],  lambda c: -(c['Kg']-np.eye(3))[1,0]),
        ("dKg_zx",  dKg_truth[2,0],  lambda c: -(c['Kg']-np.eye(3))[2,0]),
        ("dKg_xy",  dKg_truth[0,1],  lambda c: -(c['Kg']-np.eye(3))[0,1]),
        ("dKg_yy",  dKg_truth[1,1],  lambda c: -(c['Kg']-np.eye(3))[1,1]),
        ("dKg_zy",  dKg_truth[2,1],  lambda c: -(c['Kg']-np.eye(3))[2,1]),
        ("dKg_xz",  dKg_truth[0,2],  lambda c: -(c['Kg']-np.eye(3))[0,2]),
        ("dKg_yz",  dKg_truth[1,2],  lambda c: -(c['Kg']-np.eye(3))[1,2]),
        ("dKg_zz",  dKg_truth[2,2],  lambda c: -(c['Kg']-np.eye(3))[2,2]),
        # dKa (加计标度因数及安装角)
        ("dKa_xx",  dKa_truth[0,0],  lambda c: -(c['Ka']-np.eye(3))[0,0]),
        ("dKa_yx",  dKa_truth[1,0],  lambda c: -(c['Ka']-np.eye(3))[1,0]),
        ("dKa_zx",  dKa_truth[2,0],  lambda c: -(c['Ka']-np.eye(3))[2,0]),
        ("dKa_yy",  dKa_truth[1,1],  lambda c: -(c['Ka']-np.eye(3))[1,1]),
        ("dKa_zy",  dKa_truth[2,1],  lambda c: -(c['Ka']-np.eye(3))[2,1]),
        ("dKa_zz",  dKa_truth[2,2],  lambda c: -(c['Ka']-np.eye(3))[2,2]),
        # Ka2 (加计二次项)
        ("Ka2_x",   clbt_truth['Ka2'][0],  lambda c: -c['Ka2'][0]),
        ("Ka2_y",   clbt_truth['Ka2'][1],  lambda c: -c['Ka2'][1]),
        ("Ka2_z",   clbt_truth['Ka2'][2],  lambda c: -c['Ka2'][2]),
        # rx, ry (内臂误差)
        ("rx_x",    clbt_truth['rx'][0],   lambda c: -c['rx'][0]),
        ("rx_y",    clbt_truth['rx'][1],   lambda c: -c['rx'][1]),
        ("rx_z",    clbt_truth['rx'][2],   lambda c: -c['rx'][2]),
        ("ry_x",    clbt_truth['ry'][0],   lambda c: -c['ry'][0]),
        ("ry_y",    clbt_truth['ry'][1],   lambda c: -c['ry'][1]),
        ("ry_z",    clbt_truth['ry'][2],   lambda c: -c['ry'][2]),
        # tGA
        ("tGA",     clbt_truth['tGA'],     lambda c: -c['tGA']),
    ]

    labels = list(results_dict.keys())
    n_params = len(params)
    n_labels = len(labels)

    err_pct = np.zeros((n_params, n_labels))
    for pi, (name, true_v, get_est) in enumerate(params):
        for li, label in enumerate(labels):
            est_v = get_est(results_dict[label])
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

    # 柱状图
    x = np.arange(n_params)
    width = 0.75 / n_labels
    colors = ['steelblue', 'tomato', 'forestgreen', 'darkorange', 'mediumpurple']
    fig, ax = plt.subplots(figsize=(20, 8))
    for li, label in enumerate(labels):
        offset = (li - n_labels/2 + 0.5) * width
        ax.bar(x + offset, err_pct[:, li], width, label=label, color=colors[li % len(colors)], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in params], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Calibration Error (%)')
    ax.set_title('Calibration Accuracy Comparison (All Parameters)')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'markov_model_comparison.svg'), format='svg')
    plt.close()
    print(f"\nSaved chart to '{out_dir}/markov_model_comparison.svg'")


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
        [18,   0, 0,-1,  90, 9, 20, 20]
    ], dtype=float)
    paras[:, 4] = paras[:, 4] * glv.deg

    # 噪声参数（使用能显著激发影响的 5 倍大噪声）
    ARW   = 0.01  * glv.dpsh       # 角度随机游走: 0.01 deg/sqrt(h)
    VRW   = 10.0  * glv.ugpsHz     # 速度随机游走: 10 ug/sqrt(Hz)
    BI_G  = 0.025 * glv.dph        # 陀螺零偏不稳定性: 0.025 deg/h (增大 5 倍)
    BI_A  = 50.0  * glv.ug         # 加计零偏不稳定性: 50 ug       (增大 5 倍)
    TAU_G = 300.0                   # 相关时间: 300s
    TAU_A = 300.0                   # 相关时间: 300s

    print("=" * 60)
    print("IMU Noise (High Markov Bias Instability + White Noise):")
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

    # 注入全噪声
    imu_noisy = imuadderr_full(imu_clean, ts,
                                arw=ARW, vrw=VRW,
                                bi_g=BI_G, tau_g=TAU_G,
                                bi_a=BI_A, tau_a=TAU_A, seed=42)

    # ── B: 全噪声 + 49 状态（仅 ZUPT）──
    print("\n[B] Full noise, 49-state (ZUPT only)...")
    clbt_B, P_final = run_calibration(imu_noisy, pos0, ts, n_states=49,
                              bi_g=BI_G, tau_g=TAU_G, bi_a=BI_A, tau_a=TAU_A,
                              label="49-GM1 (ZUPT)")
                              
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots_observability_svg')
    
    print("\nDrafting Correlation Heatmap...")
    plot_correlation_matrix(P_final, 49, out_dir, "19_Position_Trajectory")


if __name__ == "__main__":
    main()
