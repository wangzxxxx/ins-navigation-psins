import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt

# Add the parent directory to sys.path so 'psins_py' can be resolved when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psins_py.nav_utils import glv, posset, Earth
from psins_py.imu_utils import attrottt, avp2imu, imuclbt, imudot, cnscl, imulvS
from psins_py.kf_utils import clbtkfinit, kfupdate, clbtkffeedback, getFt, alignsb, nnts
from psins_py.math_utils import q2mat, q2att, qmulv, qupdt2, rotv
from psins_py.plot_utils import clbtfile

class EnhancedKalmanFilter(dict):
    """
    继承自 dict 的卡尔曼滤波容器（适配 PSINS 的字典风格基类）
    提供标准的高频物理 update 和低频异步的 shadow_update
    """
    def __init__(self, d):
        super().__init__(d)
        
    def kf_update(self, yk=None, TimeMeasBoth=None):
        """传统的 15/43 维物理闭环观测更新"""
        out = kfupdate(self, yk, TimeMeasBoth)
        for k, v in out.items():
            self[k] = v
        
    def shadow_update(self, z_llm, H_llm, R_llm):
        """异步影子注入，直接映射标量观测"""
        z_pred = H_llm @ self['xk']
        innovation = z_llm - z_pred[0]
        
        S = H_llm @ self['Pxk'] @ H_llm.T + R_llm
        S_scalar = float(S[0, 0])
        
        K = self['Pxk'] @ H_llm.T * (1.0 / S_scalar)
        self['xk'] = self['xk'] + K.flatten() * innovation
        self['Pxk'] = (np.eye(len(self['xk'])) - K @ H_llm) @ self['Pxk']
        self['Pxk'] = (self['Pxk'] + self['Pxk'].T) * 0.5


def mock_llm_inference(current_P, current_residual):
    """
    模拟大模型 (LLM) 获取残差和协方差后返回的异步观测注入诊断。
    """
    # 模拟 LLM 洞察到 X 轴失准角/标度因数 (例如 index 13 代表 dKg_xy 项) 陷入了死锁
    # 返回其应当具备的理论真实误差和极高的置信度
    z_llm = -30 * glv.sec + np.random.randn() * (1 * glv.sec) 
    target_idx = 13
    
    H_llm = np.zeros((1, current_P.shape[0]))
    H_llm[0, target_idx] = 1.0
    R_llm = 1e-8 # High Confidence
    
    return z_llm, H_llm, R_llm


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

    print("Generating attitude data for 19 positions...")
    att = attrottt(att0, paras, ts)
    
    print("Generating IMU simulation from trajectory...")
    imu, _ = avp2imu(att, pos0)
    
    print("Applying synthetic errors (imuclbt)...")
    imu1 = imuclbt(imu)
    
    # 展开的手动滤波主循环
    eth = Earth(pos0)
    g0 = eth.g
    wnie = glv.wie * np.array([0, math.cos(pos0[0]), math.sin(pos0[0])])
    gn = np.array([0, 0, -g0])
    
    Cba = np.eye(3)
    nn, _, nts, _ = nnts(2, ts)
    frq2 = int(1 / ts / 2) - 1
    
    k = frq2
    for k in range(frq2, min(5*60*2*frq2, len(imu1)), 2*frq2):
        ww = np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
        if np.linalg.norm(ww) / ts > 20 * glv.dph:
            break
    kstatic = k - 3 * frq2

    def init_clbt_state():
        return {'Kg': np.eye(3), 'Ka': np.eye(3), 'Ka2': np.zeros(3), 
                'eb': np.zeros(3), 'db': np.zeros(3),
                'rx': np.zeros(3), 'ry': np.zeros(3), 'rz': np.zeros(3), 'tGA': 0.0}

    clbt_base = init_clbt_state()
    clbt_shadow = init_clbt_state()
    
    length = len(imu1)
    dotwf = imudot(imu1, 5.0)
    
    # 我们只执行一次大迭代 (iter=1) 来直观对比收敛过程
    imu_base = np.copy(imu1)
    imu_shadow = np.copy(imu1)
    
    kf_base = EnhancedKalmanFilter(clbtkfinit(nts))
    kf_shadow = EnhancedKalmanFilter(clbtkfinit(nts))
    
    att_align, _, _, qnb_base = alignsb(imu_base[frq2:kstatic, :], pos0)
    qnb_shadow = qnb_base.copy()
    
    vn_base = np.zeros(3)
    vn_shadow = np.zeros(3)
    
    t1s_base = 0.0
    t1s_shadow = 0.0
    
    P_trace_base = []
    P_trace_shadow = []

    
    print("Starting Expanded Calibration Filter (Baseline vs Shadow)...")
    
    shadow_trigger_count = 0
    
    for k in range(2 * frq2, length - frq2, nn):
        k1 = k + nn - 1
        wm = imu1[k:k1+1, 0:3]
        vm = imu1[k:k1+1, 3:6]
        t = imu1[k1, -1]
        dwb = np.mean(dotwf[k:k1+1, 0:3], axis=0)
        
        # === BASELINE EKF ===
        phim_b, dvbm_b = cnscl(np.hstack((wm, vm)))
        phim_b = clbt_base['Kg'] @ phim_b - clbt_base['eb'] * nts
        dvbm_b = clbt_base['Ka'] @ dvbm_b - clbt_base['db'] * nts
        wb_b = phim_b / nts
        fb_b = dvbm_b / nts
        
        SS_b = imulvS(wb_b, dwb, Cba)
        fL_b = SS_b @ np.concatenate((clbt_base['rx'], clbt_base['ry'], clbt_base['rz']))
        fn_b = qmulv(qnb_base, fb_b - clbt_base['Ka2'] * (fb_b**2) - fL_b - clbt_base['tGA'] * np.cross(wb_b, fb_b))
        vn_base = vn_base + (rotv(-wnie * nts / 2, fn_b) + gn) * nts
        qnb_base = qupdt2(qnb_base, phim_b, wnie * nts)
        
        t1s_base += nts
        Ft_b = getFt(fb_b, wb_b, q2mat(qnb_base), wnie, SS_b)
        kf_base['Phikk_1'] = np.eye(43) + Ft_b * nts
        kf_base.kf_update(TimeMeasBoth='T')
        
        if t1s_base > (0.2 - ts / 2):
            t1s_base = 0.0
            ww = np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0) 
            if np.linalg.norm(ww) / ts < 20 * glv.dph:  # Zero-velocity physical stationary update
                kf_base.kf_update(yk=vn_base, TimeMeasBoth='M')
        
        # === SHADOW EKF ===
        phim_s, dvbm_s = cnscl(np.hstack((wm, vm)))
        phim_s = clbt_shadow['Kg'] @ phim_s - clbt_shadow['eb'] * nts
        dvbm_s = clbt_shadow['Ka'] @ dvbm_s - clbt_shadow['db'] * nts
        wb_s = phim_s / nts
        fb_s = dvbm_s / nts
        
        SS_s = imulvS(wb_s, dwb, Cba)
        fL_s = SS_s @ np.concatenate((clbt_shadow['rx'], clbt_shadow['ry'], clbt_shadow['rz']))
        fn_s = qmulv(qnb_shadow, fb_s - clbt_shadow['Ka2'] * (fb_s**2) - fL_s - clbt_shadow['tGA'] * np.cross(wb_s, fb_s))
        vn_shadow = vn_shadow + (rotv(-wnie * nts / 2, fn_s) + gn) * nts
        qnb_shadow = qupdt2(qnb_shadow, phim_s, wnie * nts)
        
        t1s_shadow += nts
        Ft_s = getFt(fb_s, wb_s, q2mat(qnb_shadow), wnie, SS_s)
        kf_shadow['Phikk_1'] = np.eye(43) + Ft_s * nts
        kf_shadow.kf_update(TimeMeasBoth='T')
        
        if t1s_shadow > (0.2 - ts / 2):
            t1s_shadow = 0.0
            ww = np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0) 
            if np.linalg.norm(ww) / ts < 20 * glv.dph:
                kf_shadow.kf_update(yk=vn_shadow, TimeMeasBoth='M')
                
                shadow_trigger_count += 1
                
                # ------ ASYNCHRONOUS SHADOW OBSERVATION ------
                # 每发生 20 次有效静止物理更新 (相当于 4 秒)，强制触发一次 LLM 代理观测注入
                if shadow_trigger_count % 20 == 0:
                    z_llm, H_llm, R_llm = mock_llm_inference(kf_shadow['Pxk'], kf_shadow['rk'])
                    var_pre = kf_shadow['Pxk'][13, 13]
                    print(f"[{t:.2f}s] [Shadow Oracle] Triggered! Target: dKg_xy (idx 13), z_llm: {z_llm:.2e}, R: {R_llm:.1e}")
                    print(f"      -> Pre-inject Variance (idx 13): {var_pre:.2e}")
                    kf_shadow.shadow_update(z_llm, H_llm, R_llm)
                    var_post = kf_shadow['Pxk'][13, 13]
                    print(f"      -> Post-inject Variance (idx 13): {var_post:.2e} (Reduction: {var_pre - var_post:.2e})")
                    
        # Log Diagonals for target index 13
        P_trace_base.append(kf_base['Pxk'][13, 13])
        P_trace_shadow.append(kf_shadow['Pxk'][13, 13])
        
    print("Calibration Core Loop Finished.")
    
    # Plotting Comparative Results
    plt.figure(figsize=(10, 6))
    time_arr = np.arange(len(P_trace_base)) * nn * ts
    plt.plot(time_arr, np.sqrt(P_trace_base) / glv.sec, label="Baseline KF (Only Velocity Obs)", color='red')
    plt.plot(time_arr, np.sqrt(P_trace_shadow) / glv.sec, label="Shadow KF (LLM Assisted)", color='blue', linestyle='--')
    plt.title("Convergence Comparison for Parameter index 13 (dKg_xy)")
    plt.xlabel("Wait Time / Maneuver Time (s)")
    plt.ylabel("Variance std (arcsec)")
    plt.ylim([0, 150])
    plt.grid(True)
    plt.legend()
    plt.savefig('comparative_convergence.png')
    print("Saved comparative plot to comparative_convergence.png")
    
if __name__ == "__main__":
    main()
