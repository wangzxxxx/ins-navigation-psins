"""
calibration_path_optimizer_llm.py
------------------------------------
LLM-in-the-Loop 标定路径自动优化器

流程：
  1. 以 19 位置标定路径为初始猜测
  2. 运行协方差传播评分（36 态 + ARW/VRW 过程噪声）
  3. 把当前路径 + 各状态压缩比 → 构造 Prompt 发给 LLM
  4. LLM 返回修改建议（新 paras），解析后代入评分
  5. 循环直到：总时间 ≤ 20min 且所有 reduction% < 5%

目标：
  - 历史积累时间 ≤ 1200 s（20 分钟）
  - eb_x/y/z reduction < 5%，其余 reduction < 5%
"""
import numpy as np
import sys, os, math, json, re, time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from psins_py.nav_utils import glv, posset, Earth
from psins_py.imu_utils import attrottt, avp2imu, imuclbt, imudot, cnscl, imulvS
from psins_py.kf_utils import alignsb, nnts
from psins_py.math_utils import q2mat, qmulv, qupdt2, rotv, askew
from test_calibration_markov_pruned import get_default_clbt, getFt_36, clbtkfinit_36

# ──────────────────────────────────────────────────────────────
# 常量
# ──────────────────────────────────────────────────────────────
ARW = 0.01  * glv.dpsh
VRW = 10.0  * glv.ugpsHz
MAX_TOT_TIME_S = 1200.0   # 20 分钟
CONVERGE_THRESH = 5.0     # 所有状态 reduction% < 5%
MAX_ITER = 20             # 最大 LLM 迭代次数

STATE_LABELS = (
    ['phi_x','phi_y','phi_z','dv_x','dv_y','dv_z',
     'eb_x','eb_y','eb_z','db_x','db_y','db_z',
     'Kg00','Kg10','Kg20','Kg01','Kg11','Kg21','Kg02','Kg12','Kg22',
     'Ka_xx','Ka_xy','Ka_xz','Ka_yy','Ka_yz','Ka_zz',
     'Ka2_x','Ka2_y','Ka2_z','rx_x','rx_y','rx_z','ry_x','ry_y','ry_z']
)
# 标定需要精确估计的状态（去掉 phi/dv 这些辅助状态）
CALIB_STATES = slice(6, 36)

# ──────────────────────────────────────────────────────────────
# 初始路径（19 位置，单位：秒）
# ──────────────────────────────────────────────────────────────
# 初始路径的静止时间设为 50s（18 个位置 × (9+20+50) = 1422s；总时间需 <= 1200s，先由主循环自适应压缩）
INIT_PARAS = np.array([
    [1,    0, 1, 0,  90, 9, 70,  50],
    [2,    0, 1, 0,  90, 9, 20,  50],
    [3,    0, 1, 0,  90, 9, 20,  50],
    [4,    0, 1, 0, -90, 9, 20,  50],
    [5,    0, 1, 0, -90, 9, 20,  50],
    [6,    0, 1, 0, -90, 9, 20,  50],
    [7,    0, 0, 1,  90, 9, 20,  50],
    [8,    1, 0, 0,  90, 9, 20,  50],
    [9,    1, 0, 0,  90, 9, 20,  50],
    [10,   1, 0, 0,  90, 9, 20,  50],
    [11,  -1, 0, 0,  90, 9, 20,  50],
    [12,  -1, 0, 0,  90, 9, 20,  50],
    [13,  -1, 0, 0,  90, 9, 20,  50],
    [14,   0, 0, 1,  90, 9, 20,  50],
    [15,   0, 0, 1,  90, 9, 20,  50],
    [16,   0, 0,-1,  90, 9, 20,  50],
    [17,   0, 0,-1,  90, 9, 20,  50],
    [18,   0, 0,-1,  90, 9, 20,  50],
], dtype=float)
INIT_PARAS[:, 4] = INIT_PARAS[:, 4] * glv.deg


# ──────────────────────────────────────────────────────────────
# 核心评分函数
# ──────────────────────────────────────────────────────────────
def score_path(paras_rad, verbose=False):
    """
    返回：
      reduction_dict: {状态名: reduction_%}
      total_time_s:   路径总时间（秒）
      worst_%:        最差的标定状态压缩比
    """
    pos0 = posset(34.0, 0.0, 0.0)
    att0 = np.array([1.0, -91.0, -91.0]) * glv.deg
    ts   = 0.01

    att       = attrottt(att0, paras_rad, ts)
    imu, _    = avp2imu(att, pos0)
    clbt_t    = get_default_clbt()
    imu_clean = imuclbt(imu, clbt_t)

    eth  = Earth(pos0)
    wnie = glv.wie * np.array([0, math.cos(pos0[0]), math.sin(pos0[0])])
    Cba  = np.eye(3)
    nn, _, nts, _ = nnts(2, ts)
    frq2 = int(1 / ts / 2) - 1
    n    = 36

    # 粗对准
    kk = frq2
    for kk in range(frq2, min(5*60*2*frq2, len(imu_clean)), 2*frq2):
        ww = np.mean(imu_clean[kk-frq2:kk+frq2+1, :3], axis=0)
        if np.linalg.norm(ww)/ts > 20*glv.dph: break
    _, _, _, qnb = alignsb(imu_clean[frq2:max(kk-3*frq2, frq2+1), :], pos0)

    kf = clbtkfinit_36(nts)
    kf['Pxk'][:, 2] = 0; kf['Pxk'][2, :] = 0
    Hk    = kf['Hk']
    # 过程噪声 Q
    qvec = np.zeros(n)
    qvec[0:3] = ARW; qvec[3:6] = VRW
    Qk = np.diag(qvec)**2 * nts

    clbt = {'Kg': np.eye(3), 'Ka': np.eye(3), 'Ka2': np.zeros(3),
            'eb': np.zeros(3), 'db': np.zeros(3), 'rx': np.zeros(3), 'ry': np.zeros(3)}
    dotwf = imudot(imu_clean, 5.0)
    vn    = np.zeros(3)
    t1s   = 0.0

    for k in range(2*frq2, len(imu_clean)-frq2, nn):
        k1  = k + nn - 1
        wm  = imu_clean[k:k1+1, :3]
        vm  = imu_clean[k:k1+1, 3:6]
        dwb = np.mean(dotwf[k:k1+1, :3], axis=0)
        phim, dvbm = cnscl(np.hstack((wm, vm)))
        phim = clbt['Kg'] @ phim - clbt['eb'] * nts
        dvbm = clbt['Ka'] @ dvbm - clbt['db'] * nts
        wb, fb = phim/nts, dvbm/nts
        SS  = imulvS(wb, dwb, Cba)
        Ft  = getFt_36(fb, wb, q2mat(qnb), wnie, SS)
        Phi = np.eye(n) + Ft * nts
        kf['Pxk'] = Phi @ kf['Pxk'] @ Phi.T + Qk
        t1s += nts
        if t1s > (0.2 - ts/2):
            t1s = 0.0
            ww = np.mean(imu_clean[k-frq2:k+frq2+1, :3], axis=0)
            if np.linalg.norm(ww)/ts < 20*glv.dph:
                S    = Hk @ kf['Pxk'] @ Hk.T + kf['Rk']
                K    = kf['Pxk'] @ Hk.T @ np.linalg.inv(S)
                I_KH = np.eye(n) - K @ Hk
                kf['Pxk'] = I_KH @ kf['Pxk'] @ I_KH.T + K @ kf['Rk'] @ K.T
        fn = qmulv(qnb, fb - clbt['Ka2']*(fb**2) - SS[:,0:6] @ np.concatenate((clbt['rx'], clbt['ry'])))
        vn = vn + (rotv(-wnie*nts/2, fn) + np.array([0,0,-eth.g])) * nts
        qnb = qupdt2(qnb, phim, wnie*nts)

    sigma_f = np.sqrt(np.diag(kf['Pxk']))
    kf0     = clbtkfinit_36(nts)
    sigma0  = np.sqrt(np.diag(kf0['Pxk']))
    sigma0  = np.where(sigma0 < 1e-30, 1.0, sigma0)
    red     = sigma_f / sigma0 * 100.0

    red_dict = {lbl: float(r) for lbl, r in zip(STATE_LABELS, red)}
    total_t  = len(imu_clean) * ts
    worst_c  = float(max(red[6:36]))  # 只看标定参数

    if verbose:
        print(f"  Total time: {total_t:.1f} s ({total_t/60:.1f} min)")
        print(f"  Worst calib-state reduction: {worst_c:.2f}%")
        for lbl, r in zip(STATE_LABELS[6:], red[6:]):
            tag = 'GOOD' if r < 5 else ('OK' if r < 10 else 'POOR')
            print(f"    {lbl:<10} {r:7.2f}%  {tag}")

    return red_dict, total_t, worst_c


# ──────────────────────────────────────────────────────────────
# LLM 客户端初始化
# ──────────────────────────────────────────────────────────────
def init_llm():
    load_dotenv()
    api_key    = os.getenv("OPENAI_API_KEY", "")
    base_url   = os.getenv("OPENAI_BASE_URL", "")
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-4o")
    prov_id    = os.getenv("MODEL_PROVIDER_ID", "")
    if not api_key or not base_url:
        raise RuntimeError("[LLM] 未找到 OPENAI_API_KEY 或 OPENAI_BASE_URL，请检查 .env 文件")
    headers = {"X-Model-Provider-Id": prov_id} if prov_id else {}
    client = OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)
    print(f"[LLM] Initialized: model={model_name}")
    return client, model_name


# ──────────────────────────────────────────────────────────────
# Prompt 构造
# ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert in IMU (Inertial Measurement Unit) calibration and Kalman filter design.
Your task is to iteratively improve a multi-position static calibration trajectory for an IMU.

BACKGROUND:
The calibration uses a 36-state Kalman filter with velocity measurement updates during static positions.
Calibration parameters include: gyro bias (eb), accel bias (db), gyro scale factor matrix (Kg),
accel scale factor upper-triangular (Ka), second-order nonlinearity (Ka2), and lever arm (rx, ry).

TRAJECTORY FORMAT:
Each position is defined by a row: [pos_id, axis_x, axis_y, axis_z, angle_deg, rotation_time_s, align_time_s, static_time_s]
- axis (x,y,z): unit rotation axis (one of the standard vectors like [0,1,0], [1,0,0], [0,0,1], or [-1,0,0] etc.)
- angle_deg: rotation angle in degrees (typically 90 or -90 for standard positions)
- rotation_time_s: time to complete the rotation (typically 9 s)
- align_time_s: settling time after rotation (typically 20 s for short, or 70 s for first position)
- static_time_s: how long to stay static and collect data for this position (key parameter!)

CONSTRAINTS:
- Total time budget: <= 1200 seconds (20 minutes) including all rotations and static times
- Number of positions: 10 to 25 (keep it reasonable for a physical lab setup)
- Each position must use a standard 90-degree rotation along one of the 6 face directions
- rotation_time_s should be 9 s, align_time_s should be 20 s (or 70 s for the first position)
- static_time_s should be between 10 and 600 s per position

SCORING METRIC:
After each iteration, each calibration state gets a "reduction%" score:
- reduction% = (final sigma / initial sigma) * 100%
- Lower is better. Target: ALL calibration states < 5%
- Key states to watch: eb_x, eb_y, eb_z (gyro biases) — hardest to observe under noise
- If a state has reduction% > 10%, it needs more excitation or longer static time

HOW TO IMPROVE:
1. To improve eb (gyro bias): increase static times (especially after face-up/face-down positions)
2. To improve Ka2 (second-order nonlinearity): need high-g and diverse acceleration directions
3. To improve Kg (gyro scale factor): need diverse rotation axes and long rotation durations
4. To improve rx/ry (lever arm): need angular velocity changes (dynamic positions)
5. Removing redundant positions (same face, similar excitation) and redistributing time can help

RESPONSE FORMAT:
First write 2-3 sentences of analysis, then output EXACTLY this JSON block:
```json
{
  "paras": [
    [pos_id, axis_x, axis_y, axis_z, angle_deg, rot_time_s, align_time_s, static_time_s],
    ...
  ],
  "reasoning": "brief explanation of key changes"
}
```
CRITICAL REQUIREMENTS (NEVER VIOLATE):
1. ALL SIX face directions MUST be covered: +Z (axis [0,0,1]), -Z (axis [0,0,-1]),
   +X (axis [1,0,0]), -X (axis [-1,0,0]), +Y (axis [0,1,0]), -Y (axis [0,-1,0]).
   If ANY face is missing, Ka_xx/Ka_zz/Ka2 will collapse to POOR (>60%)!
2. Total time budget: <= 1200 seconds (20 minutes) including all rotations and static times
3. Number of positions: 10 to 25
4. rotation_time_s = 9 s, align_time_s = 20 s (or 70 s for first position)
5. static_time_s between 10 and 600 s per position
6. The JSON must be valid. No trailing commas. axis values must be integers (-1, 0, or 1).
7. The first position's align_time_s must be 70.
"""

def build_user_prompt(paras_raw, red_dict, total_t, worst_pct, iteration):
    # paras_raw: list of lists (angle still in degrees)
    failing = {k: v for k, v in red_dict.items() if STATE_LABELS.index(k) >= 6 and v >= 5.0}
    passing = {k: v for k, v in red_dict.items() if STATE_LABELS.index(k) >= 6 and v < 5.0}

    prompt = f"""=== Iteration {iteration} ===
Current total time: {total_t:.1f} s / 1200 s budget ({total_t/60:.1f} / 20.0 min)
Current worst calibration state reduction%: {worst_pct:.2f}%
Target: all calibration states < 5.0%

STATES FAILING (reduction% >= 5%, must improve):
{json.dumps({k: round(v,2) for k,v in sorted(failing.items(), key=lambda x:-x[1])}, indent=2)}

STATES PASSING (reduction% < 5%, keep these covered):
{json.dumps({k: round(v,2) for k,v in sorted(passing.items(), key=lambda x:-x[1])}, indent=2)}

CURRENT TRAJECTORY (paras, angle in degrees):
{json.dumps(paras_raw, indent=2)}

Please analyze the failing states and propose an improved trajectory.
Remember: total time of ALL positions (rot_time_s + align_time_s + static_time_s) must be <= 1200 s.
"""
    return prompt


# ──────────────────────────────────────────────────────────────
# JSON 解析
# ──────────────────────────────────────────────────────────────
def normalize_unicode(text):
    """把 LLM 常见的 Unicode 特殊字符替换为 ASCII 等价，防止 GBK 编码崩溃"""
    replacements = {
        '\u2212': '-',    # 数学减号
        '\u2011': '-',    # non-breaking hyphen
        '\u2013': '-',    # en-dash
        '\u2014': '--',   # em-dash
        '\u202f': ' ',    # narrow no-break space
        '\u00b1': '+/-',  # plus-minus
        '\u2019': "'",    # right single quote
        '\u2018': "'",    # left single quote
        '\u201c': '"',    # left double quote
        '\u201d': '"',    # right double quote
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def parse_llm_paras(response_text):

    """从 LLM 返回文本中提取 paras 列表，并把角度转为 rad。多重 fallback 策略"""
    json_str = None

    # 策略1: 优先匹配 ```json ... ``` 代码块
    m = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if m:
        json_str = m.group(1)
    else:
        # 策略2: 匹配 ``` ... ``` 任意代码块
        m = re.search(r'```\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if m:
            json_str = m.group(1)
        else:
            # 策略3: 找最外层 { ... }
            m = re.search(r'\{[\s\S]*\}', response_text)
            if m:
                json_str = m.group(0)

    if not json_str:
        raise ValueError(f'No JSON found in LLM response (len={len(response_text)})')

    # 清理常见格式问题：trailing comma before } or ]
    json_str = re.sub(r',\s*(\])', r'\1', json_str)
    json_str = re.sub(r',\s*(\})', r'\}', json_str)

    data = json.loads(json_str)
    paras_raw = data['paras']
    reasoning = data.get('reasoning', '')

    paras = np.array(paras_raw, dtype=float)
    paras[:, 4] = paras[:, 4] * glv.deg   # angle_deg → rad
    return paras, paras_raw, reasoning


def paras_to_raw(paras_rad):
    """把 numpy paras（角度 rad）转 list of lists（角度 degrees）"""
    p = paras_rad.copy()
    p[:, 4] = np.round(p[:, 4] / glv.deg, 1)
    return p.tolist()


def estimate_total_time(paras_raw):
    """估算轨迹总时间（秒），用于 prompt 中显示"""
    return sum(row[5] + row[6] + row[7] for row in paras_raw)


# ──────────────────────────────────────────────────────────────
# 主循环
# ──────────────────────────────────────────────────────────────
def main():
    print('='*65)
    print('  LLM-in-the-Loop Calibration Path Optimizer')
    print('='*65)

    client, model_name = init_llm()

    paras_rad = INIT_PARAS.copy()
    history   = []   # 每轮的得分历史

    for iteration in range(1, MAX_ITER+1):
        print(f"\n{'─'*65}")
        print(f"  [Iter {iteration:2d}] Scoring current path...")
        t0 = time.time()
        red_dict, total_t, worst_c = score_path(paras_rad, verbose=True)
        elapsed = time.time() - t0
        print(f"  Scored in {elapsed:.1f} s")

        history.append({
            'iter': iteration,
            'reduction': {k: v for k, v in red_dict.items() if STATE_LABELS.index(k) >= 6},
            'total_t': total_t,
            'worst%': worst_c,
        })

        # ── 收敛检查 ───────────────────────────────────────────
        if worst_c < CONVERGE_THRESH and total_t <= MAX_TOT_TIME_S:
            print(f"\n[✓] CONVERGED at iteration {iteration}!")
            print(f"    Worst reduction = {worst_c:.2f}% < {CONVERGE_THRESH}%")
            print(f"    Total time = {total_t:.1f} s <= {MAX_TOT_TIME_S} s")
            break

        if iteration == MAX_ITER:
            print(f"\n[!] Max iterations reached ({MAX_ITER}). Best worst% = {worst_c:.2f}%")
            break

        # ── LLM 调用 ──────────────────────────────────────────
        paras_raw = paras_to_raw(paras_rad)
        user_msg  = build_user_prompt(paras_raw, red_dict, total_t, worst_c, iteration)

        print(f"\n  [LLM] Sending to {model_name}...")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg}
            ],
            max_tokens=3000,
        )
        llm_text = response.choices[0].message.content
        print(f"  [LLM] Response received ({len(llm_text)} chars)")
        preview = llm_text[:400].encode('gbk', errors='replace').decode('gbk')
        print(f"  LLM preview: {preview}...")


        # ── 解析 + 时间检查 ────────────────────────────────────
        try:
            llm_clean = normalize_unicode(llm_text)
            new_paras_rad, new_paras_raw, reasoning = parse_llm_paras(llm_clean)
            est_t = estimate_total_time(new_paras_raw)
            print(f"  [Parse] Reasoning: {reasoning}")
            print(f"  [Parse] Estimated total time: {est_t:.1f} s")

            if est_t > MAX_TOT_TIME_S * 1.05:
                print(f"  [!] Proposed path too long ({est_t:.1f}s > {MAX_TOT_TIME_S}s). Scaling down static times.")
                # 等比缩放 static_time 使总时间满足约束
                overhead = sum(row[5] + row[6] for row in new_paras_raw)
                budget_static = MAX_TOT_TIME_S - overhead
                total_static  = sum(row[7] for row in new_paras_raw)
                scale = budget_static / max(total_static, 1)
                new_paras_rad[:, 7] = new_paras_rad[:, 7] * scale
                new_paras_raw = paras_to_raw(new_paras_rad)
                print(f"  [!] Scaled static times by {scale:.3f}")

            paras_rad = new_paras_rad
        except Exception as e:
            print(f"  [!] Parse error: {e}. Keeping previous path.")

    # ──────────────────────────────────────────────────────────
    # 绘制历史曲线
    # ──────────────────────────────────────────────────────────
    iters  = [h['iter'] for h in history]
    eb_avg = [np.mean([h['reduction'][k] for k in ['eb_x','eb_y','eb_z']]) for h in history]
    worst  = [h['worst%'] for h in history]
    ttimes = [h['total_t'] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('LLM-in-the-Loop Calibration Path Optimizer', fontsize=13, fontweight='bold')

    ax = axes[0]
    ax.plot(iters, worst,  'r-o', lw=2, label='Worst calib-state reduction%')
    ax.plot(iters, eb_avg, 'b-s', lw=2, label='eb (gyro bias) avg reduction%')
    ax.axhline(CONVERGE_THRESH, color='g', ls='--', lw=1.5, label=f'Target {CONVERGE_THRESH}%')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Reduction %')
    ax.set_title('Convergence History')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(iters, [t/60 for t in ttimes], 'purple', lw=2, marker='D')
    ax.axhline(MAX_TOT_TIME_S/60, color='r', ls='--', lw=1.5, label='20 min budget')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Path Time (min)')
    ax.set_title('Time Budget per Iteration')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plots_observability')
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, 'llm_optimizer_convergence.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'\n[SAVED] Convergence plot: {fig_path}')

    # 输出最终路径
    final_raw = paras_to_raw(paras_rad)
    print('\n[FINAL TRAJECTORY]:')
    print('  pos  ax  ay  az  angle  rot_t  align_t  static_t')
    for row in final_raw:
        print(f'  [{int(row[0]):2d}]  {int(row[1]):2d}  {int(row[2]):2d}  {int(row[3]):2d}  {row[4]:6.1f}  {row[5]:5.1f}  {row[6]:7.1f}  {row[7]:8.1f}')
    print(f'\n  Estimated total time: {estimate_total_time(final_raw):.1f} s ({estimate_total_time(final_raw)/60:.1f} min)')


if __name__ == '__main__':
    main()
