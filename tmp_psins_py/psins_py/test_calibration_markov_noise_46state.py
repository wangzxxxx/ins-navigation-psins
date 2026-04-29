"""
test_calibration_markov_noise_46state.py
-----------------------------------------
46 状态强制上三角 Ka 滤波器（49 → 46）

  强制 Ka 为上三角形式（加速度计对齐 B 系），消去下三角 3 个自由度：
    dKa_yx ≡ 0   (原 [22] 位置)
    dKa_zx ≡ 0   (原 [23] 位置)
    dKa_zy ≡ 0   (原 [27] 位置)

  保留状态（共 46 维）：
  [0:3]   姿态误差 φ
  [3:6]   速度误差 δv
  [6:9]   陀螺常值零偏 eb
  [9:12]  加计常值零偏 db
  [12:21] dKg (3×3 = 9)
  [21:27] dKa 上三角 (6): [dKa_xx, dKa_xy, dKa_xz, dKa_yy, dKa_yz, dKa_zz]
  [27:30] Ka2
  [30:33] rx
  [33:36] ry
  [36:39] rz
  [39]    tGA
  [40:43] 陀螺 Markov 零偏 bm_g
  [43:46] 加计 Markov 零偏 bm_a

对比四种条件：
  A: 无噪声（理想）
  B: 有全噪声 + 39 状态模型（不建模 Markov，无 rz/tGA 暂时保留去掉rz时的43→39之过渡版）
  C: 有全噪声 + 46 状态模型（一阶 Markov，正确 tau）
  D: 有全噪声 + 46 状态模型（一阶 Markov，错误 tau）
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
    # Ka 为上三角（消除了 yx, zx, zy 三个耦合项）
    Ka_mat = np.eye(3) - np.diag([10., 20., 30.]) * glv.ppm + \
             np.array([[0., 10., 20.], [0., 0., 40.], [0., 0., 0.]]) * glv.sec
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
#  Ka 上三角辅助函数
#  46 状态 dKa 布局（6 个，上三角）：
#    [21]: dKa_xx, [22]: dKa_xy, [23]: dKa_xz
#    [24]: dKa_yy, [25]: dKa_yz
#    [26]: dKa_zz
# ═══════════════════════════════════════════════════════════════
def Ka_from_upper(x_dKa_6):
    """从 6 个上三角分量重建 3×3 的 dKa 矩阵（注：行优先，上三角填入）"""
    dKa = np.zeros((3, 3))
    dKa[0, 0] = x_dKa_6[0]
    dKa[0, 1] = x_dKa_6[1]
    dKa[0, 2] = x_dKa_6[2]
    dKa[1, 1] = x_dKa_6[3]
    dKa[1, 2] = x_dKa_6[4]
    dKa[2, 2] = x_dKa_6[5]
    return dKa


def upper_Cnb_blocks(Cnb):
    """
    对于全 9 维 dKa，观测矩阵中 dv/dt 中 dKa 相关列为 diag(f) ⊗ Cnb（在row_2）。
    但 46 状态中 dKa 只有上三角 6 个，对应列为：
      fx*Cnb[:,0], fy*Cnb[:,0], fz*Cnb[:,0]  → 只取第 0, 1, 2 列（第 0 行）
      fy*Cnb[:,1], fz*Cnb[:,1]               → 只取第 1, 2 列（第 1 行，跳过第 0 列）
      fz*Cnb[:,2]                             → 只取第 2 列（第 2 行）
    即：
      col_xx = fx * Cnb   (3×1, fb[0] * Cnb 的列0)
    等价地，dKa 块选择：[fx*Cnb, fy*Cnb, fz*Cnb] 中的上三角部分
    """
    return None  # 见 getFt_46 实现细节


# ═══════════════════════════════════════════════════════════════
#  46 状态滤波器初始化（上三角 Ka，一阶 Markov）
# ═══════════════════════════════════════════════════════════════
def clbtkfinit_46(nts, bi_g=0.0, tau_g=3600.0, bi_a=0.0, tau_a=3600.0):
    """
    状态：[φ(3), δv(3), eb(3), db(3), dKg(9), dKa_upper(6), Ka2(3), rx(3), ry(3), rz(3), tGA(1), bm_g(3), bm_a(3)]
    共 46 维
    """
    n = 46
    use_markov = (bi_g > 0 or bi_a > 0)
    kf = {}
    kf['nts'] = nts
    kf['n'] = n
    kf['m'] = 3

    qvec = np.zeros(n)
    qvec[0:3] = 0.01 * glv.dpsh
    qvec[3:6] = 100 * glv.ugpsHz
    if use_markov:
        q_markov_g = bi_g * math.sqrt(1 - math.exp(-2*nts/tau_g)) if tau_g > 0 else 0.0
        q_markov_a = bi_a * math.sqrt(1 - math.exp(-2*nts/tau_a)) if tau_a > 0 else 0.0
        qvec[40:43] = q_markov_g
        qvec[43:46] = q_markov_a
    kf['Qk'] = np.diag(qvec)**2
    kf['Qk'][0:6, 0:6] = np.diag(qvec[0:6])**2 * nts

    kf['Rk'] = np.diag([0.001, 0.001, 0.001])**2

    pvec = np.zeros(n)
    pvec[0:3]  = np.array([0.1, 0.1, 1.0]) * glv.deg
    pvec[3:6]  = 1.0
    pvec[6:9]  = 0.1 * glv.dph
    pvec[9:12] = 1.0 * glv.mg
    # dKg 9 个
    pvec[12:15] = [100*glv.ppm, 100*glv.sec, 100*glv.sec]
    pvec[15:18] = [100*glv.sec, 100*glv.ppm, 100*glv.sec]
    pvec[18:21] = [100*glv.sec, 100*glv.sec, 100*glv.ppm]
    # dKa 上三角 6 个: [xx, xy, xz, yy, yz, zz]
    pvec[21] = 100 * glv.ppm   # dKa_xx
    pvec[22] = 100 * glv.sec   # dKa_xy
    pvec[23] = 100 * glv.sec   # dKa_xz
    pvec[24] = 100 * glv.ppm   # dKa_yy
    pvec[25] = 100 * glv.sec   # dKa_yz
    pvec[26] = 100 * glv.ppm   # dKa_zz
    # Ka2, rx, ry, rz, tGA
    pvec[27:30] = 100 * glv.ugpg2
    pvec[30:33] = 0.1
    pvec[33:36] = 0.1
    pvec[36:39] = 0.1          # rz: 非零初始化，防止 NaN 相关系数
    pvec[39]    = 0.01         # tGA
    # Markov 零偏
    if use_markov:
        pvec[40:43] = bi_g
        pvec[43:46] = bi_a
    else:
        pvec[40:43] = 0.01 * glv.dph
        pvec[43:46] = 10.0 * glv.ug

    kf['Pxk'] = np.diag(pvec)**2

    Hk = np.zeros((3, n))
    Hk[:, 3:6] = np.eye(3)
    kf['Hk'] = Hk
    kf['xk'] = np.zeros(n)
    kf['I'] = np.eye(n)
    kf['tau_g'] = tau_g
    kf['tau_a'] = tau_a
    kf['use_markov'] = use_markov
    return kf


# ═══════════════════════════════════════════════════════════════
#  46 状态 Ft 矩阵
# ═══════════════════════════════════════════════════════════════
def getFt_46(fb, wb, Cnb, wnie, SS, tau_g, tau_a, use_markov=True):
    """
    46 状态 Ft，dKa 仅上三角 6 个分量
    状态顺序：
    [0:3] φ, [3:6] δv, [6:9] eb, [9:12] db,
    [12:21] dKg(9), [21:27] dKa_upper(6),
    [27:30] Ka2, [30:33] rx, [33:36] ry, [36:39] rz,
    [39] tGA, [40:43] bm_g, [43:46] bm_a
    """
    n = 46
    o33 = np.zeros((3, 3))
    o31 = np.zeros((3, 1))
    I33 = np.eye(3)
    wX = askew(wnie)
    fX = askew(Cnb @ fb)
    wx, wy, wz = wb[0], wb[1], wb[2]
    fx, fy, fz = fb[0], fb[1], fb[2]
    CDf2 = Cnb @ np.diag(fb**2)
    CwXf = (Cnb @ np.cross(wb, fb)).reshape(3, 1)

    # dKa 上三角 6 列（对应 dv/dt = Cnb * dKa * fb，按列提取）
    # 全 9 维时：[fx*Cnb, fy*Cnb, fz*Cnb] = 每列 Cnb 乘以对应 fb 分量
    # 上三角 6 维：行 0 全取 (xx,xy,xz)，行 1 取 (yy,yz)，行 2 取 (zz)
    # 对应 dKa 矩阵：行 i, 列 j 的效果 = Cnb_col_i * fb[j]
    # 上三角 mapping: (0,0)→xx, (0,1)→xy, (0,2)→xz, (1,1)→yy, (1,2)→yz, (2,2)→zz
    # dv/dt 相关：δv̇ += Cnb * (dKa * fb) 其中 dKa 行 i, 列 j 只在上三角有效
    # 列对应 state: [21:27] -> [(i=0,j=0),(i=0,j=1),(i=0,j=2),(i=1,j=1),(i=1,j=2),(i=2,j=2)]
    # 对应关系：δv += Cnb @ [[dKa_xx*fx+dKa_xy*fy+dKa_xz*fz], [dKa_yy*fy+dKa_yz*fz], [dKa_zz*fz]]
    # 每个分量的偏导：
    # d(δv)/d(dKa_xx) = Cnb[:,0]*fx = Cnb*e0*fx → col 0 of Cnb * fx
    # 简洁写法：
    Ca_upper = np.zeros((3, 6))
    # Ka_upper[0]=(0,0): effect Cnb @ [fx, 0, 0]^T = fx * Cnb[:,0]
    Ca_upper[:, 0] = Cnb[:, 0] * fx
    # Ka_upper[1]=(0,1): effect Cnb @ [fy, 0, 0]^T * (from row 0) ..
    # 实际更精确地：dv_i += Cnb[i,0]*dKa_00*fb[0] + Cnb[i,0]*dKa_01*fb[1] + ...
    # 向量化：d(δv)/d(dKa_rc) = Cnb[:,r] * fb[c]
    # (0,0)→Cnb[:,0]*fb[0], (0,1)→Cnb[:,0]*fb[1], (0,2)→Cnb[:,0]*fb[2]
    Ca_upper[:, 0] = Cnb[:, 0] * fx   # (r=0,c=0)
    Ca_upper[:, 1] = Cnb[:, 0] * fy   # (r=0,c=1)
    Ca_upper[:, 2] = Cnb[:, 0] * fz   # (r=0,c=2)
    Ca_upper[:, 3] = Cnb[:, 1] * fy   # (r=1,c=1)
    Ca_upper[:, 4] = Cnb[:, 1] * fz   # (r=1,c=2)
    Ca_upper[:, 5] = Cnb[:, 2] * fz   # (r=2,c=2)

    # 构建完整 Ft（46×46）
    Ft = np.zeros((n, n))

    # dφ/dt
    Ft[0:3, 0:3]  = -wX
    Ft[0:3, 6:9]  = -Cnb                          # eb
    Ft[0:3, 12:15] = -wx * Cnb                    # dKg 列 0
    Ft[0:3, 15:18] = -wy * Cnb                    # dKg 列 1
    Ft[0:3, 18:21] = -wz * Cnb                    # dKg 列 2
    if use_markov:
        Ft[0:3, 40:43] = -Cnb                     # bm_g

    # dδv/dt
    Ft[3:6, 0:3]  = fX
    Ft[3:6, 9:12] = Cnb                           # db
    Ft[3:6, 12:15] = o33                          # dKg 对 δv 无直接影响（陀螺不参与加速度方程）
    # dKa 上三角 6 列
    Ft[3:6, 21:27] = Ca_upper
    # Ka2
    Ft[3:6, 27:30] = CDf2
    # 内杆臂 SS
    Ft[3:6, 30:39] = Cnb @ SS                     # rx(30:33), ry(33:36), rz(36:39) 合并
    # tGA
    Ft[3:6, 39]   = (Cnb @ np.cross(wb, fb)).reshape(3)
    if use_markov:
        Ft[3:6, 43:46] = Cnb                      # bm_a

    # Markov 衰减
    if use_markov:
        Ft[40:43, 40:43] = -I33 / tau_g
        Ft[43:46, 43:46] = -I33 / tau_a

    return Ft


# ═══════════════════════════════════════════════════════════════
#  46 状态反馈函数
# ═══════════════════════════════════════════════════════════════
def clbtkffeedback_46(kf, clbt):
    """从 46 维状态向量提取估计并更新 clbt"""
    xk = kf['xk']
    # dKg (9): 存于 [12:21]，行主序
    dKg = xk[12:21].reshape(3, 3).T
    clbt['Kg'] = (np.eye(3) - dKg) @ clbt['Kg']

    # dKa 上三角（6）：重建 3×3 并更新
    dKa = Ka_from_upper(xk[21:27])
    clbt['Ka'] = (np.eye(3) - dKa) @ clbt['Ka']

    clbt['Ka2'] = clbt['Ka2'] + xk[27:30]
    clbt['eb']  = clbt['eb']  + xk[6:9]
    clbt['db']  = clbt['db']  + xk[9:12]
    clbt['rx']  = clbt['rx']  + xk[30:33]
    clbt['ry']  = clbt['ry']  + xk[33:36]
    clbt['rz']  = clbt['rz']  + xk[36:39]
    clbt['tGA'] = clbt['tGA'] + xk[39]

    if kf.get('use_markov', False):
        clbt['eb'] += xk[40:43]
        clbt['db'] += xk[43:46]

    return clbt


# ═══════════════════════════════════════════════════════════════
#  imulvS 46 版本适配（SS 矩阵需同时包含 rz 列）
#  imulvS 生成的 SS 对应 [rx, ry, rz] 全 9 列
# ═══════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════
#  通用标定主循环（46 状态）
# ═══════════════════════════════════════════════════════════════
def run_calibration_46(imu1, pos0, ts, bi_g=0.0, tau_g=3600.0, bi_a=0.0, tau_a=3600.0, label=""):
    use_markov = (bi_g > 0 or bi_a > 0)
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
    iterations = 3

    def apply_clbt(imu_s, c):
        res = np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    for it in range(iterations):
        print(f"  [{label}] Iter {it+1}/{iterations}")
        kf = clbtkfinit_46(nts, bi_g, tau_g, bi_a, tau_a)

        if it == iterations - 1:
            kf['Pxk'] = kf['Pxk'] * 100
            kf['Pxk'][:, 2] = 0; kf['Pxk'][2, :] = 0
            kf['xk'] = np.zeros(46)

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
            Ft = getFt_46(fb, wb, q2mat(qnb), wnie, SS, tau_g, tau_a, use_markov)
            kf['Phikk_1'] = np.eye(46) + Ft * nts
            kf = kfupdate(kf, TimeMeasBoth='T')

            if t1s > (0.2 - ts / 2):
                t1s = 0.0
                ww = np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
                if np.linalg.norm(ww) / ts < 20 * glv.dph:
                    kf = kfupdate(kf, yk=vn, TimeMeasBoth='M')

        if it != iterations - 1:
            clbt = clbtkffeedback_46(kf, clbt)

    return clbt, kf


# ═══════════════════════════════════════════════════════════════
#  相关系数热力图
# ═══════════════════════════════════════════════════════════════
STATE_LABELS_46 = (
    ['φ_x', 'φ_y', 'φ_z'] +
    ['δv_x', 'δv_y', 'δv_z'] +
    ['eb_x', 'eb_y', 'eb_z'] +
    ['db_x', 'db_y', 'db_z'] +
    ['Kg_xx', 'Kg_yx', 'Kg_zx', 'Kg_xy', 'Kg_yy', 'Kg_zy', 'Kg_xz', 'Kg_yz', 'Kg_zz'] +
    ['Ka_xx', 'Ka_xy', 'Ka_xz', 'Ka_yy', 'Ka_yz', 'Ka_zz'] +
    ['Ka2_x', 'Ka2_y', 'Ka2_z'] +
    ['rx_x', 'rx_y', 'rx_z'] +
    ['ry_x', 'ry_y', 'ry_z'] +
    ['rz_x', 'rz_y', 'rz_z'] +
    ['tGA'] +
    ['bm_gx', 'bm_gy', 'bm_gz'] +
    ['bm_ax', 'bm_ay', 'bm_az']
)


def plot_correlation_heatmap(Pxk, title, save_path=None, labels=None):
    n = Pxk.shape[0]
    std = np.sqrt(np.diag(Pxk))
    # 防止除零
    std = np.where(std < 1e-30, 1e-30, std)
    corr = Pxk / np.outer(std, std)
    np.fill_diagonal(corr, 1.0)
    corr = np.clip(corr, -1.0, 1.0)

    fig, ax = plt.subplots(figsize=(14, 12))
    if labels is None:
        labels = [str(i) for i in range(n)]
    seaborn.heatmap(corr, ax=ax, cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                    square=True, linewidths=0.3, linecolor='#333',
                    xticklabels=labels, yticklabels=labels,
                    cbar_kws={'shrink': 0.8, 'label': 'Correlation'})
    ax.set_title(title, fontsize=13, pad=12)
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6, rotation=0)
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
        ("tGA",    clbt_truth['tGA'],    lambda c: -c['tGA']),
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
    colors = ['steelblue', 'tomato', 'forestgreen', 'darkorange', 'mediumpurple']
    fig, ax = plt.subplots(figsize=(22, 8))
    for li, label in enumerate(labels):
        offset = (li - n_labels/2 + 0.5) * width
        ax.bar(x + offset, err_pct[:, li], width, label=label,
               color=colors[li % len(colors)], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in params], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Calibration Error (%)')
    ax.set_title('46-State (Upper-Triangular Ka) Calibration Accuracy Comparison')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, axis='y', alpha=0.4)
    plt.tight_layout()
    svg_path = os.path.join(out_dir, 'markov_model_comparison_46state.svg')
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
    print("46-State Upper-Triangular Ka IMU Calibration")
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

    # ── A: 无噪声 (理想) ──
    print("\n[A] Clean (no noise, 46-state, 无 Markov)...")
    res_A = run_calibration_46(imu_clean, pos0, ts, label="Clean")

    # ── B: 全噪声 + 46 状态（正确 Markov 参数）──
    print("\n[B] Full noise, 46-state (correct tau/sigma, 一阶 Markov)...")
    res_B = run_calibration_46(imu_noisy, pos0, ts,
                                bi_g=BI_G, tau_g=TAU_G, bi_a=BI_A, tau_a=TAU_A,
                                label="46-GM1-correct")

    # ── C: 全噪声 + 46 状态（错误 tau）──
    TAU_WRONG = 3600.0
    print(f"\n[C] Full noise, 46-state (WRONG tau={TAU_WRONG}s, truth={TAU_G}s)...")
    res_C = run_calibration_46(imu_noisy, pos0, ts,
                                bi_g=BI_G, tau_g=TAU_WRONG, bi_a=BI_A, tau_a=TAU_WRONG,
                                label="46-GM1-wrong-tau")

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots_markov_46state')
    results = {
        "A: Clean"         : (res_A[0], res_A[1]),
        "B: 46-GM1-correct": (res_B[0], res_B[1]),
        "C: 46-GM1-wrong"  : (res_C[0], res_C[1]),
    }
    compare_results(clbt_truth, results, out_dir)

    # ── Correlation Heatmaps ──
    print("\nGenerating correlation heatmaps...")
    plot_correlation_heatmap(res_A[1]['Pxk'],
                             "46-State Correlation (Clean, no noise)",
                             save_path=os.path.join(out_dir, 'correlation_heatmap_46state_clean.svg'),
                             labels=STATE_LABELS_46)
    plot_correlation_heatmap(res_B[1]['Pxk'],
                             "46-State Correlation (Full Noise, 一阶 Markov, Correct tau)",
                             save_path=os.path.join(out_dir, 'correlation_heatmap_46state_noisy.svg'),
                             labels=STATE_LABELS_46)
    print("\nDone!")


if __name__ == "__main__":
    main()
