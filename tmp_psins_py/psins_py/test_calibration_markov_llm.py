"""
test_calibration_markov_llm.py
-------------------------------
在 49 状态 Markov 标定滤波器基础上，加入真实 LLM API 调用，
从错误的 tau 出发，通过 LLM 分析每次迭代的滤波器状态来在线修正 Markov 参数。

对比三种条件：
  C: 49 状态 + 正确 tau（上限基准）
  D: 49 状态 + 错误 tau（下限基准）
  E: 49 状态 + 错误 tau + LLM 在线修正
"""
import numpy as np
import sys
import os
import math
import json
import re
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from openai import OpenAI

from psins_py.nav_utils import glv, posset, Earth
from psins_py.imu_utils import attrottt, avp2imu, imuclbt, imudot, cnscl, imulvS
from psins_py.kf_utils import kfupdate, clbtkffeedback, alignsb, nnts
from psins_py.math_utils import q2mat, qmulv, qupdt2, rotv, askew

# 导入 base script 中的共享函数
from test_calibration_markov_noise import (
    imuadderr_full, get_default_clbt,
    clbtkfinit_43, clbtkfinit_49,
    getFt_43, getFt_49,
    clbtkffeedback_49,
    compare_results,
    run_calibration,
)


# ═══════════════════════════════════════════════════════════════
#  LLM 客户端初始化
# ═══════════════════════════════════════════════════════════════
def init_llm_client():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY", "")
    base_url = os.getenv("OPENAI_BASE_URL", "")
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
    provider_id = os.getenv("MODEL_PROVIDER_ID", "azure_openai")

    if api_key and base_url:
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers={"X-Model-Provider-Id": provider_id}
        )
        print(f"[LLM] Client initialized: model={model_name}, base_url={base_url}")
    else:
        client = None
        print("[LLM] WARNING: No API key/base_url found, LLM disabled.")

    return client, model_name


# ═══════════════════════════════════════════════════════════════
#  System Prompt：告诉 LLM 它的角色和返回格式
# ═══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = """角色设定：
您是一位顶尖的惯性导航系统（INS）标定算法专家，精通卡尔曼滤波器中 Gauss-Markov 过程建模理论。

背景任务：
当前系统正在运行一个 49 维扩维系统级标定 Kalman 滤波器。其中：
  - 状态 [0:43] 是标准 43 维标定状态（姿态误差、速度误差、陀螺/加计零偏、标度因数、安装角等）
  - 状态 [43:46] 是陀螺 Gauss-Markov 零偏 bm_g（3维）
  - 状态 [46:49] 是加计 Gauss-Markov 零偏 bm_a（3维）

Gauss-Markov 过程模型：
  dbm/dt = -1/tau * bm + w,  w ~ N(0, sigma_bi^2 * 2/tau)
  
关键参数：
  - tau（相关时间）：控制零偏漂移的速度。tau 太大 → 模型认为零偏几乎不变（等效常值）；tau 太小 → 模型过度追踪噪声
  - sigma_bi（零偏不稳定性幅度）：Markov 过程的稳态标准差

当前问题：
滤波器的 tau 参数可能设置不正确。您需要根据提供的滤波器状态信息，判断 tau 是否合理，并建议调整。

判断依据：
1. bm_g/bm_a 估计值分析：
   - 如果 |bm_est| / sigma_bi ≈ 0 → tau 设得太大，Markov 状态没有被充分利用，应减小 tau
   - 如果 |bm_est| / sigma_bi > 2 → tau 设得太小，Markov 状态在追踪白噪声，应增大 tau
   - 如果 |bm_est| / sigma_bi 在 0.3~1.0 之间 → tau 设置较合理

2. Markov 状态的协方差 P[43:49] 对角线：
   - 如果 P_bm 没有从初值显著下降 → tau 太大，观测无法有效更新 Markov 状态
   - 如果 P_bm 快速下降到接近零 → tau 可能太小

3. eb/db 估计值的迭代变化量：
   - 如果常值零偏 eb/db 在迭代之间变化剧烈 → 可能是因为 tau 设错，Markov 零偏的真实效果被错误地吸收进了 eb/db

响应格式要求：
首先输出一段简要的物理分析（2-3句），然后必须输出严格合法的 JSON 对象：
```json
{
    "tau_g_scale": float,
    "tau_a_scale": float,
    "reasoning": "简要说明调整原因"
}
```

其中 tau_g_scale 和 tau_a_scale 是对当前 tau 的乘数因子：
  - 1.0 = 不调整
  - 0.3 = 大幅减小 tau（当 Markov 状态完全不活跃时）
  - 0.5~0.7 = 适度减小
  - 1.3~2.0 = 适度增大（当 Markov 估计值过大时）

约束：scale 必须在 [0.1, 5.0] 范围内。"""


# ═══════════════════════════════════════════════════════════════
#  提取滤波器语义特征 → 构造 User Prompt
# ═══════════════════════════════════════════════════════════════
def extract_kf_features(kf, iteration, tau_g, tau_a, bi_g, bi_a,
                        prev_eb=None, prev_db=None, clbt=None):
    """提取当前滤波器状态信息供 LLM 分析"""
    xk = kf['xk']
    Pxk = kf['Pxk']

    # Markov 零偏估计值
    bm_g_est = xk[43:46].tolist()
    bm_a_est = xk[46:49].tolist()

    # Markov 零偏协方差对角线
    P_bm_g = [Pxk[43+i, 43+i] for i in range(3)]
    P_bm_a = [Pxk[46+i, 46+i] for i in range(3)]

    # 常值零偏估计值
    eb_est = xk[6:9].tolist()
    db_est = xk[9:12].tolist()

    # eb/db 协方差
    P_eb = [Pxk[6+i, 6+i] for i in range(3)]
    P_db = [Pxk[9+i, 9+i] for i in range(3)]

    # 累积标定的 eb/db（含 feedback）
    cum_eb = clbt['eb'].tolist() if clbt else [0, 0, 0]
    cum_db = clbt['db'].tolist() if clbt else [0, 0, 0]

    # bm/sigma 比值
    bm_g_ratio = float(np.linalg.norm(bm_g_est) / (bi_g + 1e-20))
    bm_a_ratio = float(np.linalg.norm(bm_a_est) / (bi_a + 1e-20))

    features = {
        "iteration": iteration,
        "current_tau_g": tau_g,
        "current_tau_a": tau_a,
        "sigma_bi_g": float(bi_g),
        "sigma_bi_a": float(bi_a),
        "bm_g_estimate": bm_g_est,
        "bm_a_estimate": bm_a_est,
        "bm_g_over_sigma_ratio": bm_g_ratio,
        "bm_a_over_sigma_ratio": bm_a_ratio,
        "P_bm_g_diagonal": P_bm_g,
        "P_bm_a_diagonal": P_bm_a,
        "eb_estimate_current_iter": eb_est,
        "db_estimate_current_iter": db_est,
        "P_eb_diagonal": P_eb,
        "P_db_diagonal": P_db,
        "cumulative_eb_from_feedback": cum_eb,
        "cumulative_db_from_feedback": cum_db,
    }

    if prev_eb is not None:
        features["eb_change_from_last_iter"] = (np.array(eb_est) - np.array(prev_eb)).tolist()
    if prev_db is not None:
        features["db_change_from_last_iter"] = (np.array(db_est) - np.array(prev_db)).tolist()

    return features


# ═══════════════════════════════════════════════════════════════
#  调用 LLM API
# ═══════════════════════════════════════════════════════════════
def call_llm_for_tau(client, model_name, features):
    """调用 LLM 获取 tau 调整建议"""
    user_prompt = f"当前 49 维标定滤波器迭代结束，以下是 Markov 状态诊断数据：\n{json.dumps(features, indent=2, ensure_ascii=False)}"

    print("\n" + "=" * 60)
    print("[LLM] Sending request...")
    print(f"[LLM] User Prompt (truncated):\n{user_prompt[:500]}...")
    print("=" * 60)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ]
        )

        content = response.choices[0].message.content.strip()

        print("\n" + "=" * 60)
        print("[LLM] Response received:")
        safe_content = content.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
        print(safe_content)
        print("=" * 60 + "\n")

        # 解析 JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            content_to_parse = json_match.group(1).strip()
        else:
            content_to_parse = content

        parsed = json.loads(content_to_parse)

        # 安全约束
        tau_g_scale = float(parsed.get("tau_g_scale", 1.0))
        tau_a_scale = float(parsed.get("tau_a_scale", 1.0))
        tau_g_scale = np.clip(tau_g_scale, 0.1, 5.0)
        tau_a_scale = np.clip(tau_a_scale, 0.1, 5.0)

        reasoning = parsed.get("reasoning", "No reasoning provided")
        print(f"[LLM] Parsed: tau_g_scale={tau_g_scale:.2f}, tau_a_scale={tau_a_scale:.2f}")
        print(f"[LLM] Reasoning: {reasoning}")

        return tau_g_scale, tau_a_scale

    except Exception as e:
        print(f"[LLM] API call failed: {e}")
        return 1.0, 1.0  # 失败时不调整


# ═══════════════════════════════════════════════════════════════
#  更新 Q 矩阵中 Markov 项（tau 变化后需要重算）
# ═══════════════════════════════════════════════════════════════
def update_kf_q_markov(kf, nts, tau_g, tau_a, bi_g, bi_a):
    """根据新 tau 更新 Q 矩阵中 Markov 驱动噪声项"""
    q_g = bi_g * math.sqrt(1 - math.exp(-2*nts/tau_g)) if tau_g > 0 else 0.0
    q_a = bi_a * math.sqrt(1 - math.exp(-2*nts/tau_a)) if tau_a > 0 else 0.0
    for i in range(3):
        kf['Qk'][43+i, 43+i] = q_g**2
        kf['Qk'][46+i, 46+i] = q_a**2
    return kf


# ═══════════════════════════════════════════════════════════════
#  带 LLM 自适应的 49 状态标定主循环
#    LLM 在第 1 次迭代的每个静止→旋转边界调用
# ═══════════════════════════════════════════════════════════════
def run_calibration_llm(imu1, pos0, ts,
                        bi_g=0.0, tau_g=3600.0, bi_a=0.0, tau_a=3600.0,
                        client=None, model_name="gpt-4o", label=""):
    """49 状态标定 + LLM 在每个 ZUPT 位置切换时在线调整 tau（仅第 1 次迭代）"""
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

    length     = len(imu1)
    dotwf      = imudot(imu1, 5.0)
    iterations = 3

    # 实时 Markov 参数（LLM 在第 1 次迭代中逐步修改）
    live_tau_g = tau_g
    live_tau_a = tau_a

    def apply_clbt(imu_s, c):
        res = np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    for it in range(iterations):
        print(f"  [{label}] Iter {it+1}/{iterations} (tau_g={live_tau_g:.1f}s)")
        kf = clbtkfinit_49(nts, bi_g, live_tau_g, bi_a, live_tau_a)

        if it == iterations - 1:
            kf['Pxk'] = kf['Pxk'] * 100
            kf['Pxk'][:, 2] = 0; kf['Pxk'][2, :] = 0
            kf['xk'] = np.zeros(n_states)

        imu_align = apply_clbt(imu1[frq2:kstatic, :], clbt)
        _, _, _, qnb = alignsb(imu_align, pos0)
        vn  = np.zeros(3)
        t1s = 0.0

        # 第 1 次迭代专用：位置切换检测 + LLM 调用
        was_static = False
        position_count = 0
        innov_list = []  # 收集当前静止段的创新

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
            Ft = getFt_49(fb, wb, q2mat(qnb), wnie, SS, live_tau_g, live_tau_a)

            kf['Phikk_1'] = np.eye(n_states) + Ft * nts
            kf = kfupdate(kf, TimeMeasBoth='T')

            if t1s > (0.2 - ts / 2):
                t1s = 0.0
                ww  = np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
                is_static = np.linalg.norm(ww) / ts < 20 * glv.dph

                if is_static:
                    kf = kfupdate(kf, yk=vn, TimeMeasBoth='M')
                    # 收集创新
                    if it == 0 and 'rk' in kf:
                        innov_list.append(kf['rk'].copy())
                    was_static = True
                else:
                    # 静止→旋转边界：第 1 次迭代时调用 LLM
                    if was_static and it == 0 and client is not None and len(innov_list) > 3:
                        position_count += 1
                        # 构造特征
                        innov_arr = np.array(innov_list)
                        features = extract_kf_features(
                            kf, iteration=1, tau_g=live_tau_g, tau_a=live_tau_a,
                            bi_g=bi_g, bi_a=bi_a, clbt=clbt
                        )
                        features["position_index"] = position_count
                        features["total_positions"] = 18
                        features["innovation_mean"] = innov_arr.mean(axis=0).tolist()
                        features["innovation_std"] = innov_arr.std(axis=0).tolist()
                        features["n_zupt_samples"] = len(innov_list)

                        # 调用 LLM
                        tau_g_scale, tau_a_scale = call_llm_for_tau(client, model_name, features)

                        old_tau_g = live_tau_g
                        live_tau_g = float(np.clip(live_tau_g * tau_g_scale, 30.0, 10000.0))
                        live_tau_a = float(np.clip(live_tau_a * tau_a_scale, 30.0, 10000.0))
                        print(f"    [{label}] Pos {position_count}: "
                              f"tau_g {old_tau_g:.1f} → {live_tau_g:.1f}s")

                        # 实时更新 Q 矩阵
                        kf = update_kf_q_markov(kf, nts, live_tau_g, live_tau_a, bi_g, bi_a)

                        innov_list = []
                    was_static = False

        # 最后一段静止期（第 1 次迭代）
        if it == 0 and client is not None and len(innov_list) > 3:
            position_count += 1
            innov_arr = np.array(innov_list)
            features = extract_kf_features(
                kf, iteration=1, tau_g=live_tau_g, tau_a=live_tau_a,
                bi_g=bi_g, bi_a=bi_a, clbt=clbt
            )
            features["position_index"] = position_count
            features["total_positions"] = 18
            features["innovation_mean"] = innov_arr.mean(axis=0).tolist()
            features["innovation_std"] = innov_arr.std(axis=0).tolist()
            features["n_zupt_samples"] = len(innov_list)

            tau_g_scale, tau_a_scale = call_llm_for_tau(client, model_name, features)
            live_tau_g = float(np.clip(live_tau_g * tau_g_scale, 30.0, 10000.0))
            live_tau_a = float(np.clip(live_tau_a * tau_a_scale, 30.0, 10000.0))

        if it != iterations - 1:
            clbt = clbtkffeedback_49(kf, clbt)

    print(f"  [{label}] Final tau_g={live_tau_g:.1f}s (started at {tau_g:.1f}s)")
    return clbt


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

    # 噪声参数
    ARW   = 0.01  * glv.dpsh
    VRW   = 10.0  * glv.ugpsHz
    BI_G  = 0.025 * glv.dph
    BI_A  = 50.0  * glv.ug
    TAU_G = 300.0
    TAU_A = 300.0
    TAU_WRONG = 3600.0  # 故意设错 12 倍

    print("=" * 60)
    print("IMU Noise (Markov Bias Instability + White Noise):")
    print(f"  ARW  = 0.01 deg/sqrt(h)")
    print(f"  VRW  = 10 ug/sqrt(Hz)")
    print(f"  BI_g = 0.025 deg/h  (true tau={TAU_G}s)")
    print(f"  BI_a = 50 ug        (true tau={TAU_A}s)")
    print(f"  WRONG tau = {TAU_WRONG}s (12x error)")
    print("=" * 60)

    # 初始化 LLM
    client, model_name = init_llm_client()

    print("\nGenerating IMU trajectory...")
    att  = attrottt(att0, paras, ts)
    imu, _ = avp2imu(att, pos0)
    clbt_truth = get_default_clbt()
    imu_clean  = imuclbt(imu, clbt_truth)
    imu_noisy = imuadderr_full(imu_clean, ts,
                                arw=ARW, vrw=VRW,
                                bi_g=BI_G, tau_g=TAU_G,
                                bi_a=BI_A, tau_a=TAU_A, seed=42)

    # ── C: 49 状态 + 正确 tau（上限基准）──
    print("\n[C] 49-state with CORRECT tau (upper bound)...")
    clbt_C = run_calibration(imu_noisy, pos0, ts, n_states=49,
                              bi_g=BI_G, tau_g=TAU_G, bi_a=BI_A, tau_a=TAU_A,
                              label="49-correct")

    # ── D: 49 状态 + 错误 tau（下限基准）──
    print(f"\n[D] 49-state with WRONG tau={TAU_WRONG}s (lower bound)...")
    clbt_D = run_calibration(imu_noisy, pos0, ts, n_states=49,
                              bi_g=BI_G, tau_g=TAU_WRONG, bi_a=BI_A, tau_a=TAU_WRONG,
                              label="49-wrong")

    # ── E: 49 状态 + LLM 在线修正（从错误 tau 出发）──
    print(f"\n[E] 49-state + LLM adaptive (start tau={TAU_WRONG}s)...")
    clbt_E = run_calibration_llm(imu_noisy, pos0, ts,
                                  bi_g=BI_G, tau_g=TAU_WRONG,
                                  bi_a=BI_A, tau_a=TAU_WRONG,
                                  client=client, model_name=model_name,
                                  label="49+LLM")

    # ── 对比输出 ──
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots_markov_svg')
    results = {
        "C: 49-correct"    : clbt_C,
        "D: 49-wrong-tau"  : clbt_D,
        "E: 49+LLM"        : clbt_E,
    }
    compare_results(clbt_truth, results, out_dir)


if __name__ == "__main__":
    main()
