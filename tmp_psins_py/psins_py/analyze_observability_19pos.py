"""
analyze_observability_19pos.py
-------------------------------
19 位置法 - 可观性定量分析（36 态模型）

对比两种情况的协方差收敛：
  Case 1: 无过程噪声（Q=0）——理论 CRLB 下界，反映轨迹几何可观性
  Case 2: 有过程噪声（Q=ARW/VRW）——真实噪声下的滤波器性能上界

决定性指标：
  sigma_final / sigma_init → 0 : 该状态可以被有效标定
  sigma_final / sigma_init → 1 : 该状态不可观（无信息增益）
"""
import numpy as np
import sys, os, math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psins_py.nav_utils import glv, posset, Earth
from psins_py.imu_utils import attrottt, avp2imu, imuclbt, imudot, cnscl, imulvS
from psins_py.kf_utils import alignsb, nnts
from psins_py.math_utils import q2mat, qmulv, qupdt2, rotv, askew
from test_calibration_markov_pruned import get_default_clbt, getFt_36, clbtkfinit_36

# ──────────────────────────────────────────────────────────────
# 轨迹参数
# ──────────────────────────────────────────────────────────────
ts   = 0.01
pos0 = posset(34.0, 0.0, 0.0)
att0 = np.array([1.0, -91.0, -91.0]) * glv.deg

paras = np.array([
    [1,    0, 1, 0,  90, 9, 70, 700],
    [2,    0, 1, 0,  90, 9, 20, 200],
    [3,    0, 1, 0,  90, 9, 20, 200],
    [4,    0, 1, 0, -90, 9, 20, 200],
    [5,    0, 1, 0, -90, 9, 20, 200],
    [6,    0, 1, 0, -90, 9, 20, 200],
    [7,    0, 0, 1,  90, 9, 20, 200],
    [8,    1, 0, 0,  90, 9, 20, 200],
    [9,    1, 0, 0,  90, 9, 20, 200],
    [10,   1, 0, 0,  90, 9, 20, 200],
    [11,  -1, 0, 0,  90, 9, 20, 200],
    [12,  -1, 0, 0,  90, 9, 20, 200],
    [13,  -1, 0, 0,  90, 9, 20, 200],
    [14,   0, 0, 1,  90, 9, 20, 200],
    [15,   0, 0, 1,  90, 9, 20, 200],
    [16,   0, 0,-1,  90, 9, 20, 200],
    [17,   0, 0,-1,  90, 9, 20, 200],
    [18,   0, 0,-1,  90, 9, 20, 200],
], dtype=float)
paras[:, 4] = paras[:, 4] * glv.deg

att        = attrottt(att0, paras, ts)
imu, _     = avp2imu(att, pos0)
clbt_truth = get_default_clbt()
imu_clean  = imuclbt(imu, clbt_truth)

# ──────────────────────────────────────────────────────────────
# 噪声参数（与 test_calibration_markov_pruned 保持一致）
# ──────────────────────────────────────────────────────────────
ARW = 0.01  * glv.dpsh
VRW = 10.0  * glv.ugpsHz

# ──────────────────────────────────────────────────────────────
# 公共初始化（对准 + 轨迹导航）
# ──────────────────────────────────────────────────────────────
eth  = Earth(pos0)
wnie = glv.wie * np.array([0, math.cos(pos0[0]), math.sin(pos0[0])])
Cba  = np.eye(3)
nn, _, nts, _ = nnts(2, ts)
frq2 = int(1 / ts / 2) - 1

k = frq2
for k in range(frq2, min(5*60*2*frq2, len(imu_clean)), 2*frq2):
    ww = np.mean(imu_clean[k-frq2:k+frq2+1, 0:3], axis=0)
    if np.linalg.norm(ww)/ts > 20*glv.dph: break
kstatic = k - 3*frq2
_, _, _, qnb_init = alignsb(imu_clean[frq2:kstatic, :], pos0)

clbt  = {'Kg': np.eye(3), 'Ka': np.eye(3), 'Ka2': np.zeros(3),
         'eb': np.zeros(3), 'db': np.zeros(3), 'rx': np.zeros(3), 'ry': np.zeros(3)}
dotwf = imudot(imu_clean, 5.0)
n     = 36

labels = (
    ['phi_x','phi_y','phi_z','dv_x','dv_y','dv_z',
     'eb_x','eb_y','eb_z','db_x','db_y','db_z',
     'Kg00','Kg10','Kg20','Kg01','Kg11','Kg21','Kg02','Kg12','Kg22',
     'Ka_xx','Ka_xy','Ka_xz','Ka_yy','Ka_yz','Ka_zz',
     'Ka2_x','Ka2_y','Ka2_z','rx_x','rx_y','rx_z','ry_x','ry_y','ry_z']
)


# ──────────────────────────────────────────────────────────────
# 通用协方差传播函数
# ──────────────────────────────────────────────────────────────
def run_cov_propagation(use_process_noise=False, label_str=''):
    """
    只传播协方差 P，不做状态估计。
    use_process_noise: 若 True，把 ARW/VRW 加入 Qk，体现量测噪声导致的不确定度下界
    """
    kf = clbtkfinit_36(nts)
    kf['Pxk'][:, 2] = 0
    kf['Pxk'][2, :] = 0
    Hk = kf['Hk']

    # 构造过程噪声 Q（若启用）
    if use_process_noise:
        qvec = np.zeros(n)
        qvec[0:3] = ARW       # 姿态对应角度随机游走
        qvec[3:6] = VRW       # 速度对应速度随机游走
        Qk_noise = np.diag(qvec)**2 * nts
    else:
        Qk_noise = np.zeros((n, n))

    qnb = qnb_init.copy()
    vn  = np.zeros(3)
    t1s = 0.0
    P_hist = []
    P_t    = []

    for k in range(2*frq2, len(imu_clean)-frq2, nn):
        k1  = k + nn - 1
        wm  = imu_clean[k:k1+1, 0:3]
        vm  = imu_clean[k:k1+1, 3:6]
        dwb = np.mean(dotwf[k:k1+1, 0:3], axis=0)

        phim, dvbm = cnscl(np.hstack((wm, vm)))
        phim  = clbt['Kg'] @ phim - clbt['eb'] * nts
        dvbm  = clbt['Ka'] @ dvbm - clbt['db'] * nts
        wb    = phim / nts
        fb    = dvbm / nts

        SS  = imulvS(wb, dwb, Cba)
        Cnb = q2mat(qnb)
        Ft  = getFt_36(fb, wb, Cnb, wnie, SS)
        Phi = np.eye(n) + Ft * nts

        # 时间传播：P = Φ P Φ^T + Q
        kf['Pxk'] = Phi @ kf['Pxk'] @ Phi.T + Qk_noise

        t1s += nts
        if t1s > (0.2 - ts/2):
            t1s = 0.0
            ww = np.mean(imu_clean[k-frq2:k+frq2+1, 0:3], axis=0)
            if np.linalg.norm(ww)/ts < 20*glv.dph:
                # 量测更新（Joseph form）
                S    = Hk @ kf['Pxk'] @ Hk.T + kf['Rk']
                K    = kf['Pxk'] @ Hk.T @ np.linalg.inv(S)
                I_KH = np.eye(n) - K @ Hk
                kf['Pxk'] = I_KH @ kf['Pxk'] @ I_KH.T + K @ kf['Rk'] @ K.T

        P_hist.append(np.sqrt(np.diag(kf['Pxk'])).copy())
        P_t.append(k * ts)

        fn  = qmulv(qnb, fb - clbt['Ka2']*(fb**2) - SS[:,0:6] @ np.concatenate((clbt['rx'], clbt['ry'])))
        vn  = vn + (rotv(-wnie*nts/2, fn) + np.array([0, 0, -eth.g])) * nts
        qnb = qupdt2(qnb, phim, wnie * nts)

    P_hist = np.array(P_hist)
    P_t    = np.array(P_t)
    sigma_final = np.sqrt(np.diag(kf['Pxk']))
    print(f'\n--- {label_str} ---')
    print(f'  {"State":<10}  {"sigma_final":>14}  {"reduction%":>12}')
    kf0 = clbtkfinit_36(nts)
    sigma0 = np.sqrt(np.diag(kf0['Pxk']))
    sigma0 = np.where(sigma0 < 1e-30, 1.0, sigma0)
    reduction = sigma_final / sigma0
    for lbl, sf, r in zip(labels, sigma_final, reduction):
        status = 'GOOD' if r < 0.1 else ('OK' if r < 0.5 else 'POOR')
        print(f'  {lbl:<10}  {sf:>14.4e}  {r*100:>11.2f}%  {status}')
    return P_hist, P_t, sigma_final, reduction


# ──────────────────────────────────────────────────────────────
# 运行两种情况
# ──────────────────────────────────────────────────────────────
print('='*60)
print('  19-Position Observability Analysis: Q=0 vs Q=ARW/VRW')
print('='*60)

P_hist_0, P_t_0, sigma0_final, red0 = run_cov_propagation(
    use_process_noise=False, label_str='Case 1: No Process Noise (Q=0, Theoretical CRLB)')

P_hist_Q, P_t_Q, sigmaQ_final, redQ = run_cov_propagation(
    use_process_noise=True,  label_str='Case 2: With Process Noise (Q=ARW/VRW)')

kf0 = clbtkfinit_36(nts)
sigma_init = np.sqrt(np.diag(kf0['Pxk']))
sigma_init = np.where(sigma_init < 1e-30, 1.0, sigma_init)

# ──────────────────────────────────────────────────────────────
# 绘图
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('19-Position Calibration: Observability Analysis\nQ=0 (CRLB Lower Bound) vs Q=ARW/VRW (Real Noise Floor)',
             fontsize=13, fontweight='bold')

groups = {
    'Gyro bias eb' : (slice(6,  9),  'steelblue'),
    'Acc  bias db' : (slice(9,  12), 'tomato'),
    'dKa_upper'    : (slice(21, 27), 'forestgreen'),
    'Ka2'          : (slice(27, 30), 'darkorange'),
    'rx'           : (slice(30, 33), 'mediumpurple'),
    'ry'           : (slice(33, 36), 'gold'),
}

# ── Row 1: Q=0 ──────────────────────────────────────────────
for row, (P_hist, P_t, sigma_fin, reduction, case_label, Q_label) in enumerate([
    (P_hist_0, P_t_0, sigma0_final, red0, 'Case 1: Q = 0 (Theoretical CRLB)', 'No ARW/VRW'),
    (P_hist_Q, P_t_Q, sigmaQ_final, redQ, 'Case 2: Q = ARW/VRW (Real Noise Floor)', 'With ARW/VRW'),
]):
    # 子图1：P 收敛曲线
    ax = axes[row, 0]
    for name, (sl, color) in groups.items():
        mean_sig = P_hist[:, sl].mean(axis=1)
        ax.semilogy(P_t, mean_sig, color=color, lw=1.8, label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean 1-sigma per group')
    ax.set_title(f'{case_label}\nUncertainty Convergence')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # 子图2：单个状态收敛（eb 组）
    ax = axes[row, 1]
    colors_eb = ['steelblue', 'tomato', 'forestgreen']
    names_eb  = ['eb_x', 'eb_y', 'eb_z']
    for ci, (i, nm) in enumerate(zip([6, 7, 8], names_eb)):
        ax.semilogy(P_t, P_hist[:, i] / glv.dph, color=colors_eb[ci], lw=1.8, label=nm)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Gyro bias sigma (deg/h)')
    title_extra = '\n(Noise floor visible: sigma does not -> 0)' if row == 1 else '\n(Clean convergence to CRLB)'
    ax.set_title(f'Gyro Bias eb convergence{title_extra}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 子图3：不确定度压缩比柱状图
    ax = axes[row, 2]
    colors_bar = []
    for r in reduction:
        if r < 0.1:   colors_bar.append('#27ae60')
        elif r < 0.5: colors_bar.append('#f39c12')
        else:         colors_bar.append('#e74c3c')
    x = np.arange(len(labels))
    ax.bar(x, reduction * 100, color=colors_bar, alpha=0.9)
    ax.axhline(10, color='g', ls='--', lw=1.5, label='10% (GOOD)')
    ax.axhline(50, color='r', ls='--', lw=1.5, label='50% (POOR)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=5.5)
    ax.set_ylabel('sigma_final / sigma_init (%)')
    ax.set_title(f'Reduction Ratio [{Q_label}]')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)
    for text, xi_range, c in [
        ('phi/dv',(0,5),'#aaaaaa'),('eb/db',(6,11),'#3498db'),
        ('Kg',(12,20),'#9b59b6'),('Ka',(21,26),'#2ecc71'),
        ('Ka2',(27,29),'#e67e22'),('rx/ry',(30,35),'#e74c3c'),
    ]:
        x_mid = (xi_range[0] + xi_range[1]) / 2
        ymax  = ax.get_ylim()[1]
        ax.text(x_mid, ymax*0.88, text, ha='center',
                fontsize=6, color=c, fontweight='bold')

# 在 eb subplot 上标注"噪声下界"
ax_eb_noisy = axes[1, 1]
# 用水平虚线标注理论 noise floor: sigma_eb ≈ ARW / sqrt(T)
T_total = P_t_Q[-1]
noise_floor_eb = ARW / math.sqrt(T_total)
ax_eb_noisy.axhline(noise_floor_eb / glv.dph, color='navy', ls=':', lw=2,
                    label=f'ARW floor = {noise_floor_eb/glv.dph:.5f} deg/h')
ax_eb_noisy.legend(fontsize=8)

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots_observability')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'observability_analysis_19pos.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\n[SAVED] {out_path}')
