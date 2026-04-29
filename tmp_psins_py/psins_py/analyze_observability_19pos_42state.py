"""
analyze_observability_19pos_42state.py
---------------------------------------
19 位置法 - 可观性定量分析（引入一阶 Gauss-Markov 模型）

从 36 态扩展到 42 态模型:
  新增状态 [36:39] bm_g, [39:42] bm_a
  包含真实噪声参数的影响。

对比两种情况：
  Case 1: Q=0 (纯理论几何下界，无系统噪声激发)
  Case 2: Q=真实参数 (带有 ARW/VRW 及 GM1 的驱动白噪声，反映真实标定极限下界)
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
from test_calibration_markov_pruned import get_default_clbt, getFt_42, clbtkfinit_42

# ──────────────────────────────────────────────────────────────
# 轨迹参数
# ──────────────────────────────────────────────────────────────
ts   = 0.01
pos0 = posset(34.0, 0.0, 0.0)
att0 = np.array([1.0, -91.0, -91.0]) * glv.deg

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

att        = attrottt(att0, paras, ts)
imu, _     = avp2imu(att, pos0)
clbt_truth = get_default_clbt()
imu_clean  = imuclbt(imu, clbt_truth)

# ──────────────────────────────────────────────────────────────
# 噪声与马尔可夫参数
# ──────────────────────────────────────────────────────────────
ARW   = 0.01  * glv.dpsh
VRW   = 10.0  * glv.ugpsHz
BI_G  = 0.005 * glv.dph
BI_A  = 10.0  * glv.ug
TAU_G = 300.0
TAU_A = 300.0

# ──────────────────────────────────────────────────────────────
# 公共初始化
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
n     = 42

labels = (
    ['phi_x','phi_y','phi_z','dv_x','dv_y','dv_z',
     'eb_x','eb_y','eb_z','db_x','db_y','db_z',
     'Kg00','Kg10','Kg20','Kg01','Kg11','Kg21','Kg02','Kg12','Kg22',
     'Ka_xx','Ka_xy','Ka_xz','Ka_yy','Ka_yz','Ka_zz',
     'Ka2_x','Ka2_y','Ka2_z','rx_x','rx_y','rx_z','ry_x','ry_y','ry_z',
     'bm_gx','bm_gy','bm_gz','bm_ax','bm_ay','bm_az']
)


# ──────────────────────────────────────────────────────────────
# 协方差传播主函数 (支持 42 态)
# ──────────────────────────────────────────────────────────────
def run_cov_propagation_42(use_process_noise=False, label_str=''):
    kf = clbtkfinit_42(nts, BI_G, TAU_G, BI_A, TAU_A)
    # 对于无噪声理想情况，如果想要看绝对纯净的下界，需关闭 Qk，但我们保留了转移矩阵 Ft (包含 -1/tau)，
    # 这样可以看出系统本身的收敛性。
    kf['Pxk'][:, 2] = 0
    kf['Pxk'][2, :] = 0
    Hk = kf['Hk']

    if use_process_noise:
        # 使用完整的真实 Qk 矩阵（包含了 ARW、VRW 以及 GM 驱动白噪声）
        Qk_noise = kf['Qk'].copy()
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
        Ft  = getFt_42(fb, wb, Cnb, wnie, SS, TAU_G, TAU_A)
        Phi = np.eye(n) + Ft * nts

        # 时间传播
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
    
    kf0 = clbtkfinit_42(nts, BI_G, TAU_G, BI_A, TAU_A)
    sigma0 = np.sqrt(np.diag(kf0['Pxk']))
    sigma0 = np.where(sigma0 < 1e-30, 1.0, sigma0)
    reduction = sigma_final / sigma0

    print(f'\n--- {label_str} ---')
    print(f'  {"State":<10}  {"sigma_final":>14}  {"reduction%":>12}')
    for lbl, sf, r in zip(labels, sigma_final, reduction):
        status = 'GOOD' if r < 0.1 else ('OK' if r < 0.5 else 'POOR')
        print(f'  {lbl:<10}  {sf:>14.4e}  {r*100:>11.2f}%  {status}')

    return P_hist, P_t, sigma_final, reduction


# ──────────────────────────────────────────────────────────────
# 运行两种情况
# ──────────────────────────────────────────────────────────────
print('='*70)
print('  19-Pos Observability Analysis: 42-State (1st-Order Markov)')
print('='*70)

P_hist_0, P_t_0, sigma0_final, red0 = run_cov_propagation_42(
    use_process_noise=False, label_str='Case 1: No Process Noise (Q=0, Theo CRLB)')

P_hist_Q, P_t_Q, sigmaQ_final, redQ = run_cov_propagation_42(
    use_process_noise=True,  label_str='Case 2: With Full Process Noise (ARW+VRW+GM)')


# ──────────────────────────────────────────────────────────────
# 绘图
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('19-Pos Calibration: Observability Analysis (42-State 1st-Order GM)\nQ=0 (CRLB) vs Q=Full Noise (Real Floor)',
             fontsize=14, fontweight='bold')

groups = {
    'Gyro bias (eb)' : (slice(6,  9),  'steelblue'),
    'Acc bias (db)'  : (slice(9,  12), 'tomato'),
    'dKg'            : (slice(12, 21), 'mediumpurple'),
    'dKa_upper'      : (slice(21, 27), 'forestgreen'),
    'Ka2'            : (slice(27, 30), 'darkorange'),
    'rx/ry'          : (slice(30, 36), 'gold'),
    'Markov bm_g/a'  : (slice(36, 42), 'brown'),
}

for row, (P_hist, P_t, sigma_fin, reduction, case_label, Q_label) in enumerate([
    (P_hist_0, P_t_0, sigma0_final, red0, 'Case 1: Q = 0 (Theoretical CRLB)', 'No Noise'),
    (P_hist_Q, P_t_Q, sigmaQ_final, redQ, 'Case 2: Q = Full Noise (ARW+VRW+GM)', 'With Full Q'),
]):
    # 子图1：整体趋势分组
    ax = axes[row, 0]
    for name, (sl, color) in groups.items():
        mean_sig = P_hist[:, sl].mean(axis=1)
        ax.semilogy(P_t, mean_sig, color=color, lw=1.8, label=name)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mean 1-sigma per group')
    ax.set_title(f'{case_label}\nState Groups Uncertainty')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(True, alpha=0.3)

    # 子图2：重点关注零偏 eb 和 变动项 bm_g
    ax = axes[row, 1]
    nm_eb, c_eb     = ['eb_x','eb_y','eb_z'], ['#3498db', '#e74c3c', '#2ecc71']
    nm_bmg, c_bmg   = ['bm_gx','bm_gy','bm_gz'], ['#2980b9', '#c0392b', '#27ae60']
    
    for ci, i in enumerate([6, 7, 8]):     # eb
        ax.semilogy(P_t, P_hist[:, i] / glv.dph, color=c_eb[ci], lw=1.8, label=nm_eb[ci])
    for ci, i in enumerate([36, 37, 38]):  # bm_g
        ax.semilogy(P_t, P_hist[:, i] / glv.dph, color=c_bmg[ci], lw=1.5, ls='--', label=nm_bmg[ci])
        
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Gyro bias components sigma (deg/h)')
    title_extra = 'Noise Floor Limits eb/bm_g Split' if row == 1 else 'CRLB for eb & bm_g'
    ax.set_title(title_extra)
    ax.legend(fontsize=8, loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)

    # 子图3：最终压缩比
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
        ('Ka2/rx/ry',(27,35),'#e67e22'), ('Markov_g/a',(36,41),'brown')
    ]:
        x_mid = (xi_range[0] + xi_range[1]) / 2
        ymax  = ax.get_ylim()[1]
        ax.text(x_mid, ymax*0.88, text, ha='center',
                fontsize=7, color=c, fontweight='bold')

# 添加理论下界参考线
ax_eb_noisy = axes[1, 1]
T_total = P_t_Q[-1]
noise_floor_eb = ARW / math.sqrt(T_total)
ax_eb_noisy.axhline(noise_floor_eb / glv.dph, color='k', ls=':', lw=2,
                    label=f'ARW floor = {noise_floor_eb/glv.dph:.4f} deg/h')
# 真正的常值零偏估计极限通常还会受到 Markov tau 的影响，这里只画基准ARW下界作参考
ax_eb_noisy.legend(fontsize=8, loc='upper right', ncol=2)

plt.tight_layout()
out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots_observability')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, 'observability_analysis_19pos_42state.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\n[SAVED] {out_path}')
