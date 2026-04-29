"""
test_calibration_markov_pruned_llm.py
-----------------------------------
加入 LLM API 动态裁剪更新步长（Scheme A）实验版。
在有噪声的情境下，通过 LLM 给出的先验物理判断，对不可观测的状态强制压缩卡尔曼更新步长。
"""
import numpy as np
import sys
import os
import math
import re
import matplotlib.pyplot as plt
import seaborn
from dotenv import load_dotenv
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from psins_py.nav_utils import glv, posset, Earth
from psins_py.imu_utils import attrottt, avp2imu, imuclbt, imudot, cnscl, imulvS
from psins_py.kf_utils import kfupdate, alignsb, nnts
from psins_py.math_utils import q2mat, qmulv, qupdt2, rotv, askew

# 读取环境变量
load_dotenv()
try:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")
    provider_id = os.environ.get("MODEL_PROVIDER_ID", "azure_openai")

    if api_key and base_url:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={"X-Model-Provider-Id": provider_id}
        )
        print(f"[LLM] Client initialized: model={model_name}, provider={provider_id}")
    else:
        client = None
        print("[LLM] WARNING: No API key/base_url found, LLM disabled.")
except Exception as e:
    print(f"Warning: OpenAI API init failed: {e}")
    client = None
    model_name = None

# =====================================================================
# LLM 动态决策核心
# =====================================================================
is_first_llm_call = True

def get_llm_delta_mask(history_context, kf_state, kf_diag_p, n_states=36):
    """
    调用 LLM，不仅根据运动历史，还根据当前的滤波器状态和方差，
    自主评判哪些状态被噪声严重污染且不可观测。
    """
    global is_first_llm_call
    if client is None:
        return np.ones(n_states)
    
    # 提取 12-35 维度的状态和标准差信息
    state_info_lines = []
    for i in range(12, 36):
        state_val = kf_state[i]
        std_val = math.sqrt(kf_diag_p[i]) if kf_diag_p[i] > 0 else 0.0
        state_info_lines.append(f"Index {i:02d}: Value={state_val:+.4e}, StdDev={std_val:.4e}")
    state_info_str = "\n".join(state_info_lines)

    system_rules = """
    You are an autonomous expert Kalman Filter parameter scheduler.
    The IMU calibration filter has a 36-dimensional state vector.
    Indices 12-20: Gyro scale/misalignment dKg (xx, yx, zx, xy, yy, zy, xz, yz, zz)
    Indices 21-27: Accel scale/misalignment dKa upper-triangle (xx, xy, xz, yy, yz, zz)
    Indices 27-35: Other parameters (Ka2, rx, ry)

    CRITICAL RULE 1: ONLY evaluate indices 12 to 35. 
    CRITICAL RULE 2: NEVER output indices 0 to 11. Biases (6-11) are ALWAYS partially observable and must remain unrestricted.
    CRITICAL RULE 3: You must select AT MOST 5 INDICES representing the parameters that are currently the most contaminated by noise or undeniably unobservable based on the filter's numerical state.

    *YOUR AUTONOMOUS TASK*:
    Instead of following rigid prior rules, you will be given the CURRENT ACTUAL NUMERICAL STATE of the filter (Value and Standard Deviation for indices 12-35), along with the preceding motion history.
    1. Look at the preceding rotation history to understand what physical channels SHOULD have been excited.
    2. Look at the `StdDev` (Standard Deviation from the P matrix). If an unexcited parameter's StdDev relies solely on noise, it might stagnate or grow.
    3. Look at the `Value` (State Vector). If a theoretically unobservable parameter is ballooning to an unreasonable magnitude, it is absorbing pure noise.
    
    Based on your physical intuition AND the live numerical filter state, pick the TOP 3 to 5 worst-behaving/unobservable parameters (indices 12-35) to suppress.
    
    Please output your response in TWO sections:
    1. REASONING: Explain your judgment based on both the motion history AND the numerical state (Value/StdDev) you were provided.
    2. INDICES: Only output a comma-separated list of EXACT indices (12-35, MAX 5 ITEMS) that should NOT be updated.

    Example format:
    REASONING: The last rotation was Y-axis. Look at Index 12 (Kg_xx) and 21 (Ka_xx), they were not excited but their state Values are drifting wildly due to noise, and their StdDev is stagnating. I will suppress them.
    INDICES: 12,20,21,29
    """

    user_prompt = f"""
    Current and historical physical context of the calibration turntable:
    =============
{history_context}
    =============

    Live Kalman Filter Numerical State (Indices 12-35):
    =============
{state_info_str}
    =============

    Based on the physics logic AND the numerical filter state provided above, tell me which scale factor/misalignment parameters should be FORBIDDEN from updating AT THIS VERY MOMENT.
    (Output your response in the requested REASONING and INDICES format).
    """

    if is_first_llm_call:
        messages = [
            {"role": "system", "content": "You are a precise Kalman filter parameter scheduler."},
            {"role": "user", "content": system_rules.strip() + "\n\n" + user_prompt.strip()}
        ]
        is_first_llm_call = False
    else:
        messages = [
            {"role": "user", "content": user_prompt.strip()}
        ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
        )
        content = response.choices[0].message.content.strip()
        
        # Log the full response so the user can read the reasoning in the terminal
        print("\n" + "="*60)
        print("[LLM Insight]")
        safe_content = content.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
        print(safe_content)
        print("="*60 + "\n")
        
        # Parse indices
        indices_match = re.search(r'INDICES:\s*([0-9,\s]+)', content)
        mask = np.ones(n_states)
        if indices_match:
            indices_str = indices_match.group(1)
            indices = [int(x.strip()) for x in indices_str.split(',') if x.strip().isdigit()]
            # 强化安全限制：如果 LLM 不听话输出了超过 5 个，强行截断至前 5 个
            if len(indices) > 5:
                indices = indices[:5]
            
            # Soft Suppression: 0.3 discount factor instead of 0.1 for higher fault tolerance
            BETA_SUPPRESS = 0.3
            for idx in indices:
                if 12 <= idx <= 35: # Double safety check
                    mask[idx] = BETA_SUPPRESS
        return mask
    except Exception as e:
        print(f"    [LLM API Error] {e}")
        return np.ones(n_states)


# ═══════════════════════════════════════════════════════════════
#  代码复用 test_calibration_markov_pruned.py 部分
# ═══════════════════════════════════════════════════════════════
def imuadderr_full(imu_in, ts, arw=0.0, vrw=0.0, bi_g=0.0, tau_g=3600.0, bi_a=0.0, tau_a=3600.0):
    np.random.seed(42)
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
    }

def Ka_from_upper(x_dKa_6):
    dKa = np.zeros((3, 3))
    dKa[0, 0] = x_dKa_6[0]; dKa[0, 1] = x_dKa_6[1]; dKa[0, 2] = x_dKa_6[2]
    dKa[1, 1] = x_dKa_6[3]; dKa[1, 2] = x_dKa_6[4]; dKa[2, 2] = x_dKa_6[5]
    return dKa

def clbtkfinit_36(nts):
    n = 36
    kf = {'nts': nts, 'n': n, 'm': 3}
    qvec = np.zeros(n)
    qvec[0:3] = 0.01 * glv.dpsh
    qvec[3:6] = 100 * glv.ugpsHz
    kf['Qk'] = np.diag(qvec)**2 * nts
    kf['Rk'] = np.diag([0.001, 0.001, 0.001])**2
    pvec = np.zeros(n)
    pvec[0:3] = np.array([0.1, 0.1, 1.0]) * glv.deg
    pvec[3:6] = 1.0
    pvec[6:9] = 0.1 * glv.dph
    pvec[9:12] = 1.0 * glv.mg
    pvec[12:15] = [100*glv.ppm, 100*glv.sec, 100*glv.sec]
    pvec[15:18] = [100*glv.sec, 100*glv.ppm, 100*glv.sec]
    pvec[18:21] = [100*glv.sec, 100*glv.sec, 100*glv.ppm]
    pvec[21] = 100 * glv.ppm; pvec[22] = 100 * glv.sec; pvec[23] = 100 * glv.sec
    pvec[24] = 100 * glv.ppm; pvec[25] = 100 * glv.sec; pvec[26] = 100 * glv.ppm
    pvec[27:30] = 100 * glv.ugpg2
    pvec[30:33] = 0.1; pvec[33:36] = 0.1
    kf['Pxk'] = np.diag(pvec)**2
    Hk = np.zeros((3, n)); Hk[:, 3:6] = np.eye(3)
    kf['Hk'] = Hk; kf['xk'] = np.zeros(n); kf['I'] = np.eye(n)
    return kf

def getFt_36(fb, wb, Cnb, wnie, SS):
    n = 36
    wX = askew(wnie); fX = askew(Cnb @ fb)
    fx, fy, fz = fb[0], fb[1], fb[2]; wx, wy, wz = wb[0], wb[1], wb[2]
    CDf2 = Cnb @ np.diag(fb**2)
    Ca_upper = np.zeros((3, 6))
    Ca_upper[:, 0] = Cnb[:, 0] * fx; Ca_upper[:, 1] = Cnb[:, 0] * fy; Ca_upper[:, 2] = Cnb[:, 0] * fz
    Ca_upper[:, 3] = Cnb[:, 1] * fy; Ca_upper[:, 4] = Cnb[:, 1] * fz; Ca_upper[:, 5] = Cnb[:, 2] * fz
    Ft = np.zeros((n, n))
    Ft[0:3, 0:3] = -wX; Ft[0:3, 6:9] = -Cnb
    Ft[0:3, 12:15] = -wx * Cnb; Ft[0:3, 15:18] = -wy * Cnb; Ft[0:3, 18:21] = -wz * Cnb
    Ft[3:6, 0:3] = fX; Ft[3:6, 9:12] = Cnb
    Ft[3:6, 21:27] = Ca_upper; Ft[3:6, 27:30] = CDf2
    Ft[3:6, 30:36] = Cnb @ SS[:, 0:6]
    return Ft

def clbtkffeedback_pruned(kf, clbt):
    xk = kf['xk']
    dKg = xk[12:21].reshape(3, 3).T
    clbt['Kg'] = (np.eye(3) - dKg) @ clbt['Kg']
    dKa = Ka_from_upper(xk[21:27])
    clbt['Ka'] = (np.eye(3) - dKa) @ clbt['Ka']
    clbt['Ka2'] = clbt['Ka2'] + xk[27:30]
    clbt['eb'] = clbt['eb'] + xk[6:9]
    clbt['db'] = clbt['db'] + xk[9:12]
    clbt['rx'] = clbt['rx'] + xk[30:33]
    clbt['ry'] = clbt['ry'] + xk[33:36]
    return clbt


# ═══════════════════════════════════════════════════════════════
#  标定主引擎：带有 LLM 动态增益裁剪
# ═══════════════════════════════════════════════════════════════
def run_calibration(imu1, pos0, ts, llm_mode=False, label=""):
    eth  = Earth(pos0)
    wnie = glv.wie * np.array([0, math.cos(pos0[0]), math.sin(pos0[0])])
    gn   = np.array([0, 0, -eth.g])
    Cba  = np.eye(3)
    nn, _, nts, _ = nnts(2, ts)
    frq2 = int(1 / ts / 2) - 1

    k = frq2
    for k in range(frq2, min(5*60*2*frq2, len(imu1)), 2*frq2):
        ww = np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
        if np.linalg.norm(ww) / ts > 20 * glv.dph: break
    kstatic = k - 3 * frq2

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
        kf = clbtkfinit_36(nts)

        if it == iterations - 1:
            kf['Pxk'][:, 2] = 0; kf['Pxk'][2, :] = 0
            kf['xk'] = np.zeros(36)

        imu_align = apply_clbt(imu1[frq2:kstatic, :], clbt)
        _, _, _, qnb = alignsb(imu_align, pos0)
        vn  = np.zeros(3)
        t1s = 0.0
        
        # 为了减少 API 调用频率，我们只有在特定时刻才去要掩码
        current_mask = np.ones(36)
        
        # 记录绝对旋转状态，用来识别物理阶段，并保存历史
        last_motion_status = "Unknown"
        motion_history = []
        time_elapsed = 0.0
        
        # 统计静止时间，以便在静止中期再向 LLM 要 mask
        stationary_time = 0.0
        llm_called_this_phase = False
        
        # 统计掩码被实际应用的次数（调试用）
        mask_applied_count = 0


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

            time_elapsed += nts

            # LLM 情境判断逻辑
            vel_rot = np.linalg.norm(wb)
            if vel_rot < 2.0 * glv.dps: 
                current_motion_status = "STATIONARY"
            else:
                axis_idx = np.argmax(np.abs(wb))
                axis_name = ['X', 'Y', 'Z'][axis_idx]
                current_motion_status = f"ROTATING around {axis_name}-axis"
            
            # --- 动作切换检测 ---
            if current_motion_status != last_motion_status:
                motion_history.append(f"T={time_elapsed:.1f}s -> {current_motion_status}")
                
                if current_motion_status == "STATIONARY":
                    stationary_time = 0.0
                    llm_called_this_phase = False
                    current_mask = np.ones(36) # 刚停下时不急着砍状态，让 KF 自由收敛一会儿跑掉随机白噪声
                else: # ROTATING
                    current_mask = np.ones(36) # 开始旋转了，解放所有状态
                
                last_motion_status = current_motion_status
            
            # --- 延迟询问 LLM 与掩码实施 ---
            if current_motion_status == "STATIONARY":
                stationary_time += nts
                # 在静止了 5 秒钟后，系统状态比较稳定了，向 LLM 询问专家分析
                if stationary_time > 5.0 and llm_mode and not llm_called_this_phase:
                    if it == 0:
                        recent_history = "\n".join(motion_history[-5:])
                        # 此时提供给 LLM 的 xk 和 Pxk 是已经在静止期平滑融合过 5 秒的优质数值
                        current_mask = get_llm_delta_mask(recent_history, kf['xk'], np.diag(kf['Pxk']), n_states=36)
                        # 为了避免缓存 key 重复（因为 STATIONARY string 每次都一样），我们将它和它发生的时间段锚定，
                        # 但因为 iter0/1/2 的时长是完美一致的，我们可以用 motion_history 里的动作串作为 key
                        state_key = "_".join([h.split("->")[1].strip() for h in motion_history[-4:]])
                        global_llm_mask_cache[state_key] = current_mask
                    else:
                        state_key = "_".join([h.split("->")[1].strip() for h in motion_history[-4:]])
                        if state_key in global_llm_mask_cache:
                            current_mask = global_llm_mask_cache[state_key]
                        else:
                            current_mask = np.ones(36)
                    llm_called_this_phase = True
            
            SS  = imulvS(wb, dwb, Cba)
            fL  = SS[:, 0:6] @ np.concatenate((clbt['rx'], clbt['ry']))
            fn  = qmulv(qnb, fb - clbt['Ka2']*(fb**2) - fL)
            vn  = vn + (rotv(-wnie*nts/2, fn) + gn) * nts
            qnb = qupdt2(qnb, phim, wnie * nts)

            t1s += nts
            Ft = getFt_36(fb, wb, q2mat(qnb), wnie, SS)
            kf['Phikk_1'] = np.eye(36) + Ft * nts
            kf = kfupdate(kf, TimeMeasBoth='T')

            if t1s > (0.2 - ts / 2):
                t1s = 0.0
                ww = np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
                if np.linalg.norm(ww) / ts < 20 * glv.dph:
                    if llm_mode:
                        # 核心修复：修改观测矩阵 Hk，掐断不可观状态与当前测量的联系
                        # 原本 Hk 只在 [:, 3:6] 有非零项（速度误差），但状态经过 Phi 转移后存在相关性
                        # 如果要把某个参数冻结，最稳妥的工程做法是强行让它的更新增量 (K * innov) 为 0
                        # 比较安全的做法其实是恢复到第一版直接砍状态，但要在 Feedback 阶段保留 P 阵
                        x_old = np.copy(kf['xk'])
                        kf = kfupdate(kf, yk=vn, TimeMeasBoth='M')
                        delta_x = kf['xk'] - x_old
                        # 仅限制不可观参数的数值更新，但不破坏 Pxk
                        kf['xk'] = x_old + delta_x * current_mask
                        mask_applied_count += 1
                    else:
                        kf = kfupdate(kf, yk=vn, TimeMeasBoth='M')
                
                # 记录所有迭代的连续轨迹
                P_trace.append(np.diag(kf['Pxk']))
                X_trace.append(np.copy(kf['xk']))

        if it != iterations - 1:
            clbt = clbtkffeedback_pruned(kf, clbt)
        iter_bounds.append(len(P_trace))
        if llm_mode:
            print(f"    [Debug] Over this iteration, delta clipping was applied {mask_applied_count} times.")

    return clbt, kf, np.array(P_trace), np.array(X_trace), iter_bounds


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

    # 噪声参数（加大 10 倍进行抗强噪压力测试）
    ARW   = 0.01  * glv.dpsh
    VRW   = 10.0  * glv.ugpsHz
    BI_G  = 0.05  * glv.dph
    BI_A  = 100.0 * glv.ug
    TAU_G = 3000.0
    TAU_A = 3000.0

    print("Generating IMU trajectory...")
    att  = attrottt(att0, paras, ts)
    imu, _ = avp2imu(att, pos0)
    clbt_truth = get_default_clbt()
    imu_clean  = imuclbt(imu, clbt_truth)
    imu_noisy = imuadderr_full(imu_clean, ts, arw=ARW, vrw=VRW, bi_g=BI_G, tau_g=TAU_G, bi_a=BI_A, tau_a=TAU_A)

    print("\n[A] Clean (No noise)...")
    res_clean = run_calibration(imu_clean, pos0, ts, llm_mode=False, label="Clean")
    
    print("\n[B] Noisy (Standard KF)...")
    res_noisy = run_calibration(imu_noisy, pos0, ts, llm_mode=False, label="Noisy KF")
    
    print("\n[C] Noisy w/ LLM Masking (Scheme A)...")
    global is_first_llm_call
    global global_llm_mask_cache
    is_first_llm_call = True
    global_llm_mask_cache = {}
    res_llm = run_calibration(imu_noisy, pos0, ts, llm_mode=True, label="Noisy LLM")

    # 绘制对比图
    P_C, X_C = res_clean[2], res_clean[3]
    P_N, X_N = res_noisy[2], res_noisy[3]
    P_L, X_L = res_llm[2], res_llm[3]
    
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots_llm_clipping')
    os.makedirs(out_dir, exist_ok=True)
    
    labels_36 = (
        ['phi_x', 'phi_y', 'phi_z', 'dv_x', 'dv_y', 'dv_z', 'eb_x', 'eb_y', 'eb_z', 'db_x', 'db_y', 'db_z'] +
        ['Kg_xx', 'Kg_yx', 'Kg_zx', 'Kg_xy', 'Kg_yy', 'Kg_zy', 'Kg_xz', 'Kg_yz', 'Kg_zz'] +
        ['Ka_xx', 'Ka_xy', 'Ka_xz', 'Ka_yy', 'Ka_yz', 'Ka_zz', 'Ka2_x', 'Ka2_y', 'Ka2_z', 'rx_x', 'rx_y', 'rx_z', 'ry_x', 'ry_y', 'ry_z']
    )
    
    min_len = min(len(P_C), len(P_N), len(P_L))
    time_arr = np.arange(min_len) * (2 * ts)
    
    for i in range(6, 36):
        name = labels_36[i]
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
        
        # Uncertainty
        ax1 = axes[0]
        ax1.plot(time_arr, np.sqrt(P_C[:min_len, i]), color='blue', label='Clean', alpha=0.8)
        ax1.plot(time_arr, np.sqrt(P_N[:min_len, i]), color='red',  label='Noisy (No LLM)', linestyle='--', linewidth=2)
        ax1.plot(time_arr, np.sqrt(P_L[:min_len, i]), color='green', label='Noisy w/ LLM', linewidth=2, alpha=0.6)
        
        # 绘制迭代分界线
        for b_idx in res_llm[4][:-1]: # Exclude the last bound which is the end
            if b_idx < min_len:
                t_bound = b_idx * (2 * ts)
                ax1.axvline(x=t_bound, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
                
        ax1.set_title(f'[{i:02d}] {name}  —  Uncertainty (std dev)')
        ax1.legend(); ax1.grid(True)
        
        # Value
        ax2 = axes[1]
        ax2.plot(time_arr, X_C[:min_len, i], color='blue', label='Clean', alpha=0.8)
        ax2.plot(time_arr, X_N[:min_len, i], color='red', label='Noisy (No LLM)', linestyle='--', linewidth=2)
        ax2.plot(time_arr, X_L[:min_len, i], color='green', label='Noisy w/ LLM', linewidth=2, alpha=0.6)
        
        # 绘制迭代分界线
        for b_idx in res_llm[4][:-1]:
            if b_idx < min_len:
                t_bound = b_idx * (2 * ts)
                ax2.axvline(x=t_bound, color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
                
        ax2.set_title(f'[{i:02d}] {name}  —  State Estimate')
        ax2.legend(); ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{i:02d}_{name}.svg"), format='svg')
        plt.close()
        
    print(f"\nDone! Plots saved to {out_dir}")

if __name__ == "__main__":
    main()
