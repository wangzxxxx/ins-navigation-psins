import numpy as np
import sys
import os
import math
import matplotlib.pyplot as plt
import copy

# Add the parent directory to sys.path so 'psins_py' can be resolved when running directly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psins_py.nav_utils import glv, posset, Earth
from psins_py.imu_utils import attrottt, avp2imu, imuclbt, imudot, cnscl, imulvS, imuadderr
from psins_py.kf_utils import clbtkfinit, kfupdate, clbtkffeedback, getFt, alignsb, nnts
from psins_py.math_utils import q2mat, q2att, qmulv, qupdt2, rotv
from psins_py.plot_utils import clbtfile
from psins_py.shadow_manager_hybrid import ShadowSimulationManager

def get_default_clbt():
    """
    Standard PSINS synthetic IMU error profile defaults.
    These encapsulate the uncalibrated states typically simulated in imuclbt.m.
    """
    Kg_mat = np.eye(3) - np.diag([10., 20., 30.]) * glv.ppm + \
             np.array([[0., 10., 20.], [30., 0., 40.], [50., 60., 0.]]) * glv.sec
             
    Ka_mat = np.eye(3) - np.diag([10., 20., 30.]) * glv.ppm + \
             np.array([[0., 0., 0.], [10., 0., 0.], [20., 30., 0.]]) * glv.sec
             
    return {
        'sf': np.ones(6),
        'Kg': Kg_mat,
        'Ka': Ka_mat,
        'eb': np.array([0.1, 0.2, 0.3]) * glv.dph,
        'db': np.array([100.0, 200.0, 300.0]) * glv.ug,
        'Ka2': np.array([10.0, 20.0, 30.0]) * glv.ugpg2,
        'rx': np.array([1.0, 2.0, 3.0]) / 100.0, 
        'ry': np.array([4.0, 5.0, 6.0]) / 100.0, 
        'rz': np.zeros(3),
        'tGA': 0.005
    }

class EnhancedKalmanFilter(dict):
    """
    继承自 dict 的卡尔曼滤波容器（适配 PSINS 的字典风格基类）
    提供标准的高频物理 update 和低频异步的 shadow_update
    """
    def __init__(self, d):
        super().__init__(copy.deepcopy(d))
        
    def kf_update(self, yk=None, TimeMeasBoth=None):
        """传统的 15/43 维物理闭环观测更新"""
        out = kfupdate(self, yk, TimeMeasBoth)
        for k, v in out.items():
            self[k] = v
        
    def hybrid_update(self, z_llm, target_idx, conf_ratio, inflation_factor, enable_pseudo_meas=True, enable_inflation=True):
        """
        混合智能突围注入 (Hybrid Intelligent Overdrive):
        1. Break Deadlock: 若开启(enable_inflation=True)，强制拉大目标状态 P_xx，使其对即将到来的观测敞开大门
        2. Provide Direction: 若开启(enable_pseudo_meas=True)，使用 z_llm 建立伪观测，用打开的 K 矩阵把方向灌进去
        """
        # Step 1: Covariance Inflation (Activated)
        if enable_inflation:
            # Note: inflation_factor here is expected to be pre-processed by an
            # activation function (e.g. from the ShadowManager) so it acts as 
            # a safe "awaking probability" (e.g., 1.0x to 20.0x) rather than a raw destructive scalar.
            self['Pxk'][target_idx, target_idx] *= float(inflation_factor)
        
        if enable_pseudo_meas:
            # 构建稀疏观测降维矩阵 H
            H_llm = np.zeros((1, len(self['xk'])))
            H_llm[0, target_idx] = 1.0
            
            # Step 2: Pseudo Measurement Update
            z_pred = H_llm @ self['xk']
            innovation = z_llm - z_pred[0]
            
            # 获取膨胀后的最新方差
            current_variance = float((H_llm @ self['Pxk'] @ H_llm.T)[0, 0])
            
            # === 方案一：物理 R 下界保护 ===
            # 给 R_llm 设定一个基于物理精度的绝对下界，防止滤波器已高度收敛(P_xx极小)时
            # LLM 的 conf_ratio 联动压低 R，导致增益 K 趋近于 1，造成暴力拉偏状态。
            # 即：不管 LLM 多自信，其等效测量精度也不能超过物理先验噪声极限。
            idx = target_idx
            if   0  <= idx <= 2:   r_floor = (0.01  * glv.deg)**2   # phi 姿态误差: 最小 0.01 deg 精度
            elif 3  <= idx <= 5:   r_floor = (0.01 )**2              # 速度误差: 最小 0.01 m/s 精度
            elif 6  <= idx <= 8:   r_floor = (0.1   * glv.dph)**2   # 陀螺零偏: 最小 0.1 dph 精度
            elif 9  <= idx <= 11:  r_floor = (200   * glv.ug)**2    # 加计零偏: 最小 200 ug 精度
            elif 12 <= idx <= 29:  r_floor = (200   * glv.ppm)**2   # Kg/Ka 分量: 最小 200 ppm 精度
            elif 30 <= idx <= 32:  r_floor = (50    * glv.ugpg2)**2 # Ka2: 最小 50 ug/g^2 精度
            elif 33 <= idx <= 35:  r_floor = (0.005)**2             # rx/ry/rz 内臂: 最小 5 mm 精度
            else:                  r_floor = (1e-5)**2              # 其他保守默认
            
            R_llm = max(current_variance * conf_ratio, r_floor)
            
            S_scalar = current_variance + R_llm
            if S_scalar < 1e-30:
                return False
            
            K = self['Pxk'] @ H_llm.T * (1.0 / S_scalar)
            
            # 限定每次最大修改步长，利用水滴石穿避免直接拉爆
            delta_x = K.flatten() * innovation
            max_step = np.abs(self['xk']) * 0.10 + 1e-6 # 最多跨越当前值的 10%
            delta_x = np.clip(delta_x, -max_step, max_step)
            
            self['xk'] = self['xk'] + delta_x
            self['Pxk'] = (np.eye(len(self['xk'])) - K @ H_llm) @ self['Pxk']
            self['Pxk'] = (self['Pxk'] + self['Pxk'].T) * 0.5
            
        return True


def main():
    ts = 0.01
    att0 = np.array([1.0, -91.0, -91.0]) * glv.deg
    pos0 = posset(34.0, 0.0, 0.0)
    
    paras = np.array([
        [1,    0, 1, 0,  90, 9, 70, 20],
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
    
    # --- Use Standard Authentic PSINS Truth Settings ---
    clbt_truth = get_default_clbt()
    
    # In PSINS, `imuclbt` uses `db`, `eb`, `Kg` etc to inject errors.
    imu1 = imuclbt(imu, clbt_truth)
    
    # --- 注入随机噪声 (White Noise: ARW & VRW) ---
    # 对于导航级(Navigation-grade) IMU，典型参数如下：
    # ARW (Angular Random Walk / 角度随机游走): 0.01 deg/sqrt(h) 
    # VRW (Velocity Random Walk / 速度随机游走): 10.0 ug/sqrt(Hz)
    imuerr_wn = {
        'web': np.array([0.01, 0.01, 0.01]) * 0.2 * glv.dpsh,
        'wdb': np.array([10.0, 10.0, 10.0]) * 0.2 * glv.ugpsHz
    }
    imu1 = imuadderr(imu1, imuerr_wn)
    
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
        if np.linalg.norm(ww) / ts > 30 * glv.dph:
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
    
    P_trace_base = []
    P_trace_shadow = []
    X_trace_base = []
    X_trace_shadow = []
    
    # Store indices where new iterations start for plotting vertical lines
    iteration_boundaries = []

    print("Starting Expanded Calibration Filter (Baseline vs Hybrid)...")
    
    # === 模式控制开关 ===
    # 两者皆为 False 则是标准无辅助滤波（等同于 test_system_calibration_19pos.py）
    enable_llm_pseudo_meas = True  # 控制是否开启 LLM 伪观测注入方向 (如 test_system_calibration_19pos_llm.py)
    enable_llm_inflation = False    # 控制是否开启 LLM 协方差破死锁膨胀 (如 test_system_calibration_19pos_inflation.py)
    # 两者皆为 True 即为当前的完整混合模式 (test_system_calibration_19pos_hybrid.py)
    
    iterations = 3
    
    def apply_clbt(imu_slice, clbt):
        res = np.copy(imu_slice)
        for i in range(len(res)):
            res[i, 0:3] = clbt['Kg'] @ res[i, 0:3] - clbt['eb'] * ts
            res[i, 3:6] = clbt['Ka'] @ res[i, 3:6] - clbt['db'] * ts
        return res
    
    for it in range(iterations):
        print(f"System Calibration of SIMU (iter={it+1})")
        
        if it != iterations - 1:
            kf_base = EnhancedKalmanFilter(clbtkfinit(nts))
            kf_shadow = EnhancedKalmanFilter(clbtkfinit(nts))
        else:
            kf_base['Pxk'] = kf_base['Pxk'] * 100
            kf_base['Pxk'][:, 2] = 0; kf_base['Pxk'][2, :] = 0
            kf_base['xk'] = np.zeros(43)
            
            kf_shadow['Pxk'] = kf_shadow['Pxk'] * 100
            kf_shadow['Pxk'][:, 2] = 0; kf_shadow['Pxk'][2, :] = 0
            kf_shadow['xk'] = np.zeros(43)
            
        shadow_manager = ShadowSimulationManager(kf_shadow, window_size=200, print_debug=False)
        
        # Determine alignment qnb based on Current Calibrated State
        imu_base_align = apply_clbt(imu1[frq2:kstatic, :], clbt_base)
        att_align_b, _, _, qnb_base = alignsb(imu_base_align, pos0)
        
        imu_shadow_align = apply_clbt(imu1[frq2:kstatic, :], clbt_shadow)
        att_align_s, _, _, qnb_shadow = alignsb(imu_shadow_align, pos0)
        
        vn_base = np.zeros(3)
        vn_shadow = np.zeros(3)
        t1s_base = 0.0
        t1s_shadow = 0.0
        
        shadow_trigger_count = 0
        is_rotating = False
        current_rot_axis = None
        dyn_peak_max = np.zeros(3)
        past_maneuvers = []

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
                if np.linalg.norm(ww) / ts < 30 * glv.dph:  # Zero-velocity physical stationary update
                    kf_base.kf_update(yk=vn_base, TimeMeasBoth='M')
                
                # Unconditional logging across all loops for continuous multi-iteration tracking
                P_trace_base.append(np.diag(kf_base['Pxk']))
                X_trace_base.append(np.copy(kf_base['xk']))
        
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
                if np.linalg.norm(ww) / ts < 30 * glv.dph:
                    kf_shadow.kf_update(yk=vn_shadow, TimeMeasBoth='M')
                    
                P_trace_shadow.append(np.diag(kf_shadow['Pxk']))
                X_trace_shadow.append(np.copy(kf_shadow['xk']))
                
                # We also need the buffer trigger block which is inside this 1Hz tracking update logic.
                if np.linalg.norm(ww) / ts < 30 * glv.dph:
                    # Push the latest physics innovation (residual) into the Manager's sliding buffer
                    # rk exists inside the dict after a physical update
                    shadow_manager.innovation_buffer.append(kf_shadow['rk'])
                    shadow_trigger_count += 1
                
                    # Check for state transition: Rotating -> Stationary
                    if is_rotating:
                        is_rotating = False
                        action_summary = f"Rotation around {current_rot_axis}-axis completed. Peak Vel Resid: [{dyn_peak_max[0]:.2e}, {dyn_peak_max[1]:.2e}, {dyn_peak_max[2]:.2e}]"
                        past_maneuvers.append(action_summary)
                        # Keep full history of maneuvers for cumulative observability analysis
                        # Reset peaks
                        dyn_peak_max = np.zeros(3)
                
                    # ------ ASYNCHRONOUS SHADOW OBSERVATION ------
                    # 每当物理时钟在静止段累计 100 次更新 (相当于 20 秒) 时，触发 LLM 前置观察
                    if (enable_llm_pseudo_meas or enable_llm_inflation) and it == 0 and shadow_trigger_count > 0 and shadow_trigger_count % 100 == 0:
                        print(f"\n[{t:.2f}s] Engaging Actual LLM Shadow Expert...")
                    
                        # 1. 抽取基于窗口缓存的残差斜率特征和当前的P阵
                        action_ctx = f"Stationary Calibration Frame at t={t:.1f}s"
                        dynamic_peaks_dict = {"vx_peak": dyn_peak_max[0], "vy_peak": dyn_peak_max[1], "vz_peak": dyn_peak_max[2]}
                    
                        features = shadow_manager._extract_semantic_features(
                            current_action=action_ctx,
                            past_maneuvers=past_maneuvers, 
                            dynamic_peaks=dynamic_peaks_dict
                        )
                    
                        # 2. 调用真实的 LLM (这会阻塞直到模型回复完成)
                        decisions = shadow_manager._call_llm_expert(features)
                    
                        if decisions:
                            if isinstance(decisions, dict):
                                decisions = [decisions]
                            
                            for decision in decisions:
                                if decision.get("confidence") in shadow_manager.confidence_map:
                                    conf = decision["confidence"]
                                    r_llm = shadow_manager.confidence_map[conf]
                                
                                    if r_llm is not None:
                                        idx = decision.get("target_index", 0)
                                        val = decision.get("predicted_value", 0.0)
                                        inflation = decision.get("deadlock_inflation", 1.0)
                                        
                                        # --- SAFETY CLAMPING TO PREVENT DIVERGENCE ---
                                        if 0 <= idx <= 2:     # phi (rad)
                                            val = np.clip(val, -0.01, 0.01)
                                        elif 3 <= idx <= 5:   # velocity (m/s)
                                            val = np.clip(val, -1.0, 1.0)
                                        elif 6 <= idx <= 8:   # gyro bias (rad/s)
                                            val = np.clip(val, -1e-4, 1e-4) # ~20 dph max
                                        elif 9 <= idx <= 11:  # acc bias (m/s2)
                                            val = np.clip(val, -0.05, 0.05) # ~5 mg max
                                        elif 12 <= idx <= 29: # dKg, dKa (ppm) -> scale factor & misalignment
                                            val = np.clip(val, -0.005, 0.005) # ~5000 ppm max
                                        else:
                                            val = np.clip(val, -0.05, 0.05)
                                            
                                        # Cap extreme inputs, then pass through Sigmoid activation
                                        x_infl = np.clip(inflation, 1.0, 1000.0)
                                        M_max = 20.0
                                        k_steep = 0.05
                                        x_0 = 50.0
                                        activated_infl = 1.0 + (M_max - 1.0) / (1.0 + np.exp(-k_steep * (x_infl - x_0)))
                                    
                                        var_pre = kf_shadow['Pxk'][idx, idx]
                                        print(f"      -> Hybrid Authorized! Target {idx}, z_llm: {val:.2e}, Raw Inflate: {inflation:.1f}x, Act Inflate: {activated_infl:.1f}x, R: {r_llm:.1e}")
                                        print(f"      -> Pre-inject Variance (idx {idx}): {var_pre:.2e}")
                                    
                                        # 调用全新的二段式破局注入
                                        kf_shadow.hybrid_update(val, idx, r_llm, activated_infl, enable_llm_pseudo_meas, enable_llm_inflation)
                                    
                                        var_post = kf_shadow['Pxk'][idx, idx]
                                        print(f"      -> Post-inject Variance (idx {idx}): {var_post:.2e} (Reduction: {var_pre - var_post:.2e})")
                                    else:
                                        print("      -> LLM Rejected Injection (Low Confidence).")
                        else:
                            print("      -> No valid JSON array parsed from LLM.")
                            
                            
                        # Wipe buffer so next interval is fresh independent data
                        shadow_manager.innovation_buffer.clear()
                else:
                    # We are currently in a dynamic/rotational phase. 
                    # Track the motion properties and max residuals.
                    shadow_trigger_count = 0
                    if not is_rotating:
                        is_rotating = True
                        # Determine dominant axis of rotation based on norm of components
                        axes = ['X', 'Y', 'Z']
                        dom_axis = np.argmax(np.abs(ww))
                        current_rot_axis = axes[dom_axis]
                    
            # Store the absolute maximum peak of the velocity residual (yk - z_pred)
            # `kf_shadow['rk']` holds the innovation vector if it was calculated, 
            # but during pure rotation physics update might not run (or runs with poor GPS).
            # We manually evaluate the INS velocity residual from vn_shadow - 0 assuming ZUPT is expected
            v_err = np.abs(vn_shadow)
            dyn_peak_max = np.maximum(dyn_peak_max, v_err)
            
        # Feedback the KF residual estimation into the absolute configuration map
        if it != iterations - 1:
            clbt_base = clbtkffeedback(kf_base, clbt_base)
            clbt_shadow = clbtkffeedback(kf_shadow, clbt_shadow)
            
        iteration_boundaries.append(len(P_trace_base))
        
    print("Calibration Core Loop Finished.")
    
    P_trace_base = np.array(P_trace_base)
    P_trace_shadow = np.array(P_trace_shadow)
    X_trace_base = np.array(X_trace_base)
    X_trace_shadow = np.array(X_trace_shadow)
    
    # Define descriptive names for all 43 states
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
    out_dir = os.path.join(base_dir, 'plots_svg')
    os.makedirs(out_dir, exist_ok=True)
    actual_dt = nn * ts 
    time_arr = np.arange(len(P_trace_base)) * actual_dt
    
    print(f"Generating 43 SVG convergence plots in '{out_dir}/'...")
    for i in range(43):
        plt.figure(figsize=(10, 6))
        std_base = np.sqrt(P_trace_base[:, i])
        std_shadow = np.sqrt(P_trace_shadow[:, i])
        
        plt.plot(time_arr, std_base, label="Baseline KF", color='red')
        plt.plot(time_arr, std_shadow, label="Shadow KF (LLM Assisted)", color='blue', linestyle='--')
        
        for b_idx in iteration_boundaries[:-1]:  # Exclude the very end
            plt.axvline(x=b_idx * actual_dt, color='gray', linestyle=':', label='Iteration Boundary' if b_idx == iteration_boundaries[0] else "")
            
        name = state_names[i] if i < len(state_names) else f"State_{i}"
        plt.title(f"Uncertainty Convergence for {name} (Index {i})")
        plt.xlabel("Maneuver Time (s) [Continuous across cycles]")
        plt.ylabel("Standard Deviation")
        plt.grid(True)
        plt.legend()
        
        safe_name = name.replace('/', '_').replace('\\', '_')
        filepath = os.path.join(out_dir, f"{i:02d}_{safe_name}.svg")
        plt.savefig(filepath, format='svg')
        plt.close()
        
    print(f"Saved all 43 variance comparative plots to {out_dir}/")
    
    out_dir_states = os.path.join(base_dir, 'plots_state_svg')
    os.makedirs(out_dir_states, exist_ok=True)
    print(f"Generating 43 SVG actual value state plots in '{out_dir_states}/'...")
    for i in range(43):
        plt.figure(figsize=(10, 6))
        
        plt.plot(time_arr, X_trace_base[:, i], label="Baseline KF State", color='red')
        plt.plot(time_arr, X_trace_shadow[:, i], label="Shadow KF State", color='blue', linestyle='--')
        
        for b_idx in iteration_boundaries[:-1]:
            plt.axvline(x=b_idx * actual_dt, color='gray', linestyle=':')
            
        name = state_names[i] if i < len(state_names) else f"State_{i}"
        plt.title(f"State Estimation Trajectory for {name} (Index {i})")
        plt.xlabel("Maneuver Time (s) [Continuous across cycles]")
        plt.ylabel("Estimated State Value")
        plt.grid(True)
        plt.legend()
        
        safe_name = name.replace('/', '_').replace('\\', '_')
        filepath = os.path.join(out_dir_states, f"{i:02d}_{safe_name}.svg")
        plt.savefig(filepath, format='svg')
        plt.close()
        
    print(f"Saved all 43 state value comparative plots to {out_dir_states}/")
    
    print("\n" + "="*120)
    print("FINAL PARAMETER ESTIMATION COMPARISON (TRUE vs BASELINE vs SHADOW)")
    print("="*120)
    print(f"{'State Name':<12} | {'True Val':<15} | {'Base Est':<15} | {'Shadow Est':<15} | {'Base Err':<12} | {'Base Err%':<10} | {'Shadow Err':<12} | {'Shadow Err%':<10}")
    print("-" * 120)
    
    def print_comp(name, true_val, base_val, shadow_val):
        b_actual = -base_val
        s_actual = -shadow_val
        err_b = abs(true_val - b_actual)
        err_s = abs(true_val - s_actual)
        
        # Calculate error percentage, avoid division by zero
        err_b_pct = (err_b / abs(true_val)) * 100 if abs(true_val) > 1e-15 else 0.0
        err_s_pct = (err_s / abs(true_val)) * 100 if abs(true_val) > 1e-15 else 0.0
        
        print(f"{name:<12} | {true_val:>15.6e} | {b_actual:>15.6e} | {s_actual:>15.6e} | {err_b:>12.6e} | {err_b_pct:>9.2f}% | {err_s:>12.6e} | {err_s_pct:>9.2f}%")

    print_comp("eb_x", clbt_truth['eb'][0], clbt_base['eb'][0], clbt_shadow['eb'][0])
    print_comp("eb_y", clbt_truth['eb'][1], clbt_base['eb'][1], clbt_shadow['eb'][1])
    print_comp("eb_z", clbt_truth['eb'][2], clbt_base['eb'][2], clbt_shadow['eb'][2])
    print_comp("db_x", clbt_truth['db'][0], clbt_base['db'][0], clbt_shadow['db'][0])
    print_comp("db_y", clbt_truth['db'][1], clbt_base['db'][1], clbt_shadow['db'][1])
    print_comp("db_z", clbt_truth['db'][2], clbt_base['db'][2], clbt_shadow['db'][2])
    
    dKg_t = clbt_truth['Kg'] - np.eye(3)
    dKg_b = clbt_base['Kg'] - np.eye(3)
    dKg_s = clbt_shadow['Kg'] - np.eye(3)
    
    dKa_t = clbt_truth['Ka'] - np.eye(3)
    dKa_b = clbt_base['Ka'] - np.eye(3)
    dKa_s = clbt_shadow['Ka'] - np.eye(3)
    
    kg_names = ["dKg_xx", "dKg_yx", "dKg_zx", "dKg_xy", "dKg_yy", "dKg_zy", "dKg_xz", "dKg_yz", "dKg_zz"]
    ka_names = ["dKa_xx", "dKa_yx", "dKa_zx", "dKa_xy", "dKa_yy", "dKa_zy", "dKa_xz", "dKa_yz", "dKa_zz"]
    
    for i in range(3):
        for j in range(3):
            idx = j*3 + i
            print_comp(kg_names[idx], dKg_t[i,j], dKg_b[i,j], dKg_s[i,j])
            
    for i in range(3):
        for j in range(3):
            idx = j*3 + i
            print_comp(ka_names[idx], dKa_t[i,j], dKa_b[i,j], dKa_s[i,j])
            
    print_comp("dKa2_x", clbt_truth['Ka2'][0], clbt_base['Ka2'][0], clbt_shadow['Ka2'][0])
    print_comp("dKa2_y", clbt_truth['Ka2'][1], clbt_base['Ka2'][1], clbt_shadow['Ka2'][1])
    print_comp("dKa2_z", clbt_truth['Ka2'][2], clbt_base['Ka2'][2], clbt_shadow['Ka2'][2])
    
    print_comp("rx", clbt_truth['rx'][0], clbt_base['rx'][0], clbt_shadow['rx'][0])
    print_comp("ry", clbt_truth['ry'][0], clbt_base['ry'][0], clbt_shadow['ry'][0])
    print_comp("rz", clbt_truth['rz'][0], clbt_base['rz'][0], clbt_shadow['rz'][0])
    
    print_comp("tGA", clbt_truth['tGA'], clbt_base['tGA'], clbt_shadow['tGA'])
    print("="*85 + "\n")
    
if __name__ == "__main__":
    main()
