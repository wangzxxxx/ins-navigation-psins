"""
test_calibration_staged_llm.py
------------------------------
LLM-Guided Sequential Staged Calibration (State Graduation).
After each complete 19-position iteration, the LLM reviews convergence
metrics and "graduates" well-converged parameters by freezing them.
Subsequent iterations run with fewer effective states, improving
observability for remaining parameters.

All three comparison methods use unified 5 iterations.
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from psins_py.nav_utils import glv, posset, Earth
from psins_py.imu_utils import attrottt, avp2imu, imuclbt, imudot, cnscl, imulvS
from psins_py.kf_utils import kfupdate, alignsb, nnts
from psins_py.math_utils import q2mat, qmulv, qupdt2, rotv, askew

# =====================================================================
# LLM Init
# =====================================================================
load_dotenv()
try:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")
    provider_id = os.environ.get("MODEL_PROVIDER_ID", "azure_openai")
    if api_key and base_url:
        client = OpenAI(api_key=api_key, base_url=base_url,
                        default_headers={"X-Model-Provider-Id": provider_id})
        print(f"[LLM] Client initialized: model={model_name}")
    else:
        client = None
except Exception as e:
    client = None

STATE_LABELS_36 = (
    ['phi_x', 'phi_y', 'phi_z'] +
    ['dv_x', 'dv_y', 'dv_z'] +
    ['eb_x', 'eb_y', 'eb_z'] +
    ['db_x', 'db_y', 'db_z'] +
    ['Kg_xx', 'Kg_yx', 'Kg_zx', 'Kg_xy', 'Kg_yy', 'Kg_zy', 'Kg_xz', 'Kg_yz', 'Kg_zz'] +
    ['Ka_xx', 'Ka_xy', 'Ka_xz', 'Ka_yy', 'Ka_yz', 'Ka_zz'] +
    ['Ka2_x', 'Ka2_y', 'Ka2_z'] +
    ['rx_x', 'rx_y', 'rx_z'] +
    ['ry_x', 'ry_y', 'ry_z']
)

PHYSICAL_UNITS = {
    range(6, 9): 'dph (deg/h)',
    range(9, 12): 'ug',
    range(12, 21): 'ppm or arcsec',
    range(21, 27): 'ppm or arcsec',
    range(27, 30): 'ug/g^2',
    range(30, 36): 'dimensionless (lever arm ratio)',
}


# =====================================================================
# LLM Graduation Review
# =====================================================================
def get_llm_graduation(P_initial_diag, P_final_diag, X_trace_iter, kf_Pxk, already_graduated):
    """
    LLM reviews the convergence of each candidate parameter (indices 6-35)
    and decides which to graduate (freeze) for the next iteration.
    """
    if client is None:
        return []

    # Build report for each candidate
    report_lines = []
    candidates = [i for i in range(6, 36) if i not in already_graduated]
    
    X_arr = np.array(X_trace_iter)
    tail_len = max(50, len(X_arr) // 5)
    
    for idx in candidates:
        p_init = math.sqrt(P_initial_diag[idx]) if P_initial_diag[idx] > 0 else 1e-30
        p_final = math.sqrt(P_final_diag[idx]) if P_final_diag[idx] > 0 else 0.0
        p_ratio = p_final / p_init if p_init > 1e-30 else 0.0
        x_final = X_trace_iter[-1][idx] if len(X_trace_iter) > 0 else 0.0
        x_volatility = np.std(X_arr[-tail_len:, idx]) if len(X_arr) >= tail_len else 0.0
        
        # Cross-correlation with other active states
        max_cross_corr = 0.0
        std_i = math.sqrt(kf_Pxk[idx, idx]) if kf_Pxk[idx, idx] > 0 else 1e-30
        for j in candidates:
            if j != idx:
                std_j = math.sqrt(kf_Pxk[j, j]) if kf_Pxk[j, j] > 0 else 1e-30
                cross = abs(kf_Pxk[idx, j] / (std_i * std_j)) if (std_i > 1e-25 and std_j > 1e-25) else 0.0
                max_cross_corr = max(max_cross_corr, min(cross, 1.0))
        
        # Get unit info
        unit = 'unknown'
        for r, u in PHYSICAL_UNITS.items():
            if idx in r:
                unit = u
                break
        
        report_lines.append(
            f"  [{idx:02d}] {STATE_LABELS_36[idx]:>8s} ({unit}): "
            f"P_ratio={p_ratio:.4f}, X_final={x_final:+.4e}, "
            f"X_volatility={x_volatility:.4e}, max_cross_corr={max_cross_corr:.3f}"
        )
    
    report = "\n".join(report_lines)
    already_grad_str = ", ".join([f"{i}({STATE_LABELS_36[i]})" for i in sorted(already_graduated)]) if already_graduated else "None"
    
    system_prompt = """You are a Kalman Filter convergence expert reviewing IMU calibration results.
Your task is to decide which parameters have sufficiently converged and can be "graduated" (frozen).

GRADUATION CRITERIA (ALL must be met):
1. P_ratio < 0.25 (uncertainty shrunk by >75%)
2. X_volatility is small relative to X_final magnitude
3. max_cross_corr < 0.95 (only reject if nearly perfectly entangled)
4. X_final is within physically reasonable bounds for its type

RULES:
- Indices 0-5 (navigation states) can NEVER graduate
- Be MODERATELY AGGRESSIVE: graduate parameters that show reasonable convergence. The benefit of graduation is that remaining states get better observability.
- Graduate parameters in batches of 2-12 per round
- If cross_corr > 0.95 between two parameters, either graduate BOTH or NEITHER
- Parameters with P_ratio < 0.10 should almost always graduate

OUTPUT FORMAT:
REASONING: [Your analysis]
GRADUATE: [comma-separated indices, e.g. 6,7,8,9,10,11]
If no parameters should graduate, output: GRADUATE: NONE
"""

    user_prompt = f"""
Already graduated (frozen) parameters: {already_grad_str}

Performance report for remaining active parameters:
{report}

Which parameters should graduate? Be conservative.
"""

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]

    try:
        response = client.chat.completions.create(model=model_name, messages=messages)
        content = response.choices[0].message.content.strip()
        
        print("\n" + "="*60)
        print("[LLM Graduation Review]")
        safe = content.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
        print(safe)
        print("="*60 + "\n")
        
        match = re.search(r'GRADUATE:\s*(.+)', content)
        if match:
            grad_str = match.group(1).strip()
            if grad_str.upper() == 'NONE':
                return []
            indices = [int(x.strip()) for x in grad_str.split(',') if x.strip().isdigit()]
            # Safety: only allow 6-35, exclude already graduated
            indices = [i for i in indices if 6 <= i <= 35 and i not in already_graduated]
            return indices
        return []
    except Exception as e:
        print(f"  [LLM] Error: {e}")
        return []


# =====================================================================
# Standard Building Blocks
# =====================================================================
def imuadderr_full(imu_in, ts, arw=0.0, vrw=0.0, bi_g=0.0, tau_g=3600.0, bi_a=0.0, tau_a=3600.0):
    np.random.seed(42)
    imu = np.copy(imu_in)
    m = imu.shape[0]
    sts = math.sqrt(ts)
    if arw > 0: imu[:, 0:3] += arw * sts * np.random.randn(m, 3)
    if vrw > 0: imu[:, 3:6] += vrw * sts * np.random.randn(m, 3)
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

def clbtkfinit_36(nts, graduated_indices=None):
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
    
    # Freeze graduated parameters
    if graduated_indices:
        for idx in graduated_indices:
            kf['Pxk'][idx, :] = 0.0
            kf['Pxk'][:, idx] = 0.0
            kf['Pxk'][idx, idx] = 1e-30  # Tiny but nonzero for numerical safety
            kf['Qk'][idx, idx] = 0.0
    
    return kf

def getFt_36(fb, wb, Cnb, wnie, SS, graduated_indices=None):
    n = 36
    wX = askew(wnie); fX = askew(Cnb @ fb)
    fx, fy, fz = fb[0], fb[1], fb[2]
    wx, wy, wz = wb[0], wb[1], wb[2]
    CDf2 = Cnb @ np.diag(fb**2)
    Ca_upper = np.zeros((3, 6))
    Ca_upper[:, 0] = Cnb[:, 0] * fx; Ca_upper[:, 1] = Cnb[:, 0] * fy
    Ca_upper[:, 2] = Cnb[:, 0] * fz; Ca_upper[:, 3] = Cnb[:, 1] * fy
    Ca_upper[:, 4] = Cnb[:, 1] * fz; Ca_upper[:, 5] = Cnb[:, 2] * fz
    Ft = np.zeros((n, n))
    Ft[0:3, 0:3] = -wX; Ft[0:3, 6:9] = -Cnb
    Ft[0:3, 12:15] = -wx * Cnb; Ft[0:3, 15:18] = -wy * Cnb; Ft[0:3, 18:21] = -wz * Cnb
    Ft[3:6, 0:3] = fX; Ft[3:6, 9:12] = Cnb
    Ft[3:6, 21:27] = Ca_upper; Ft[3:6, 27:30] = CDf2
    Ft[3:6, 30:36] = Cnb @ SS[:, 0:6]
    
    # Zero out rows AND columns for graduated states
    if graduated_indices:
        for idx in graduated_indices:
            Ft[idx, :] = 0.0
            Ft[:, idx] = 0.0
    
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


# =====================================================================
# Main Calibration Engine
# =====================================================================
def run_calibration(imu1, pos0, ts, staged_mode=False, label=""):
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
    iterations = 5  # Unified for all methods
    P_trace, X_trace, iter_bounds = [], [], []
    
    graduated_indices = set()
    llm_cache = {}  # Cache LLM decisions for iterations 2+ if needed

    def apply_clbt(imu_s, c):
        res = np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    for it in range(iterations):
        n_active = 36 - len(graduated_indices)
        grad_str = f" | Graduated: {len(graduated_indices)}, Active: {n_active}" if staged_mode else ""
        print(f"  [{label}] Iter {it+1}/{iterations}{grad_str}")
        
        kf = clbtkfinit_36(nts, graduated_indices if staged_mode else None)
        P_initial_diag = np.diag(kf['Pxk']).copy()

        if it == iterations - 1:
            kf['Pxk'] = kf['Pxk'] * 100
            kf['Pxk'][:, 2] = 0; kf['Pxk'][2, :] = 0
            # Re-freeze graduated states after inflation
            if staged_mode and graduated_indices:
                for idx in graduated_indices:
                    kf['Pxk'][idx, :] = 0.0
                    kf['Pxk'][:, idx] = 0.0
                    kf['Pxk'][idx, idx] = 1e-30
            kf['xk'] = np.zeros(36)

        imu_align = apply_clbt(imu1[frq2:kstatic, :], clbt)
        _, _, _, qnb = alignsb(imu_align, pos0)
        vn  = np.zeros(3)
        t1s = 0.0
        iter_X_trace = []

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
            fL  = SS[:, 0:6] @ np.concatenate((clbt['rx'], clbt['ry']))
            fn  = qmulv(qnb, fb - clbt['Ka2']*(fb**2) - fL)
            vn  = vn + (rotv(-wnie*nts/2, fn) + gn) * nts
            qnb = qupdt2(qnb, phim, wnie * nts)

            t1s += nts
            Ft = getFt_36(fb, wb, q2mat(qnb), wnie, SS,
                          graduated_indices if staged_mode else None)
            kf['Phikk_1'] = np.eye(36) + Ft * nts
            kf = kfupdate(kf, TimeMeasBoth='T')

            if t1s > (0.2 - ts / 2):
                t1s = 0.0
                ww = np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
                if np.linalg.norm(ww) / ts < 20 * glv.dph:
                    kf = kfupdate(kf, yk=vn, TimeMeasBoth='M')
                
                P_trace.append(np.diag(kf['Pxk']))
                X_trace.append(np.copy(kf['xk']))
                iter_X_trace.append(np.copy(kf['xk']))

        if it != iterations - 1:
            clbt = clbtkffeedback_pruned(kf, clbt)
            
            # LLM Graduation Review (only in staged_mode)
            if staged_mode:
                if it == 0:
                    new_grads = get_llm_graduation(
                        P_initial_diag, np.diag(kf['Pxk']),
                        iter_X_trace, kf['Pxk'], graduated_indices
                    )
                    llm_cache[it] = new_grads
                else:
                    # For subsequent iterations, re-call LLM with updated state
                    new_grads = get_llm_graduation(
                        P_initial_diag, np.diag(kf['Pxk']),
                        iter_X_trace, kf['Pxk'], graduated_indices
                    )
                    llm_cache[it] = new_grads
                
                if new_grads:
                    graduated_indices.update(new_grads)
                    grad_names = [f"{i}({STATE_LABELS_36[i]})" for i in sorted(new_grads)]
                    print(f"    >> Graduated {len(new_grads)} params: {', '.join(grad_names)}")
                    print(f"    >> Total graduated: {len(graduated_indices)}/30 candidate states")
                else:
                    print(f"    >> No new graduations this round.")

        iter_bounds.append(len(P_trace))

    return clbt, kf, np.array(P_trace), np.array(X_trace), iter_bounds


# =====================================================================
# Main
# =====================================================================
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

    ARW   = 0.005 * glv.dpsh       # halved
    VRW   = 5.0   * glv.ugpsHz     # halved
    BI_G  = 0.002 * glv.dph        # halved
    BI_A  = 5.0   * glv.ug         # halved
    TAU_G = 300.0
    TAU_A = 300.0

    print("Generating IMU trajectory...")
    att  = attrottt(att0, paras, ts)
    imu, _ = avp2imu(att, pos0)
    clbt_truth = get_default_clbt()
    imu_clean  = imuclbt(imu, clbt_truth)
    imu_noisy = imuadderr_full(imu_clean, ts, arw=ARW, vrw=VRW,
                                bi_g=BI_G, tau_g=TAU_G, bi_a=BI_A, tau_a=TAU_A)

    print("\n[A] Clean (No noise, 5 iter)...")
    res_clean = run_calibration(imu_clean, pos0, ts, staged_mode=False, label="Clean")

    print("\n[B] Noisy (Standard KF, 5 iter)...")
    res_noisy = run_calibration(imu_noisy, pos0, ts, staged_mode=False, label="Noisy KF")
    
    print("\n[C] Noisy + LLM Staged Graduation (5 iter)...")
    res_staged = run_calibration(imu_noisy, pos0, ts, staged_mode=True, label="LLM Staged")

    # ── Calibration Accuracy Comparison ──
    clbt_A, clbt_B, clbt_S = res_clean[0], res_noisy[0], res_staged[0]
    
    dKg_truth = clbt_truth['Kg'] - np.eye(3)
    dKa_truth = clbt_truth['Ka'] - np.eye(3)
    
    params = [
        ("eb_x",   clbt_truth['eb'][0],  lambda c: -c['eb'][0]),
        ("eb_y",   clbt_truth['eb'][1],  lambda c: -c['eb'][1]),
        ("eb_z",   clbt_truth['eb'][2],  lambda c: -c['eb'][2]),
        ("db_x",   clbt_truth['db'][0],  lambda c: -c['db'][0]),
        ("db_y",   clbt_truth['db'][1],  lambda c: -c['db'][1]),
        ("db_z",   clbt_truth['db'][2],  lambda c: -c['db'][2]),
        ("Kg_xx",  dKg_truth[0,0], lambda c: -(c['Kg']-np.eye(3))[0,0]),
        ("Kg_yx",  dKg_truth[1,0], lambda c: -(c['Kg']-np.eye(3))[1,0]),
        ("Kg_zx",  dKg_truth[2,0], lambda c: -(c['Kg']-np.eye(3))[2,0]),
        ("Kg_xy",  dKg_truth[0,1], lambda c: -(c['Kg']-np.eye(3))[0,1]),
        ("Kg_yy",  dKg_truth[1,1], lambda c: -(c['Kg']-np.eye(3))[1,1]),
        ("Kg_zy",  dKg_truth[2,1], lambda c: -(c['Kg']-np.eye(3))[2,1]),
        ("Kg_xz",  dKg_truth[0,2], lambda c: -(c['Kg']-np.eye(3))[0,2]),
        ("Kg_yz",  dKg_truth[1,2], lambda c: -(c['Kg']-np.eye(3))[1,2]),
        ("Kg_zz",  dKg_truth[2,2], lambda c: -(c['Kg']-np.eye(3))[2,2]),
        ("Ka_xx",  dKa_truth[0,0], lambda c: -(c['Ka']-np.eye(3))[0,0]),
        ("Ka_xy",  dKa_truth[0,1], lambda c: -(c['Ka']-np.eye(3))[0,1]),
        ("Ka_xz",  dKa_truth[0,2], lambda c: -(c['Ka']-np.eye(3))[0,2]),
        ("Ka_yy",  dKa_truth[1,1], lambda c: -(c['Ka']-np.eye(3))[1,1]),
        ("Ka_yz",  dKa_truth[1,2], lambda c: -(c['Ka']-np.eye(3))[1,2]),
        ("Ka_zz",  dKa_truth[2,2], lambda c: -(c['Ka']-np.eye(3))[2,2]),
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
    
    print("\n" + "=" * 150)
    print("CALIBRATION ACCURACY COMPARISON (5 iterations each)")
    print("=" * 150)
    header = f"{'Param':<10}{'Truth':>14}" \
             f"{'Clean Est':>14}{'Err%':>8}" \
             f"{'Noisy Est':>14}{'Err%':>8}" \
             f"{'Staged Est':>14}{'Err%':>8}" \
             f"  {'Winner':>10}"
    print(header)
    print("-" * 150)
    
    wins = {'Clean': 0, 'Noisy': 0, 'Staged': 0}
    
    for name, truth, get_est in params:
        est_A = get_est(clbt_A)
        est_B = get_est(clbt_B)
        est_S = get_est(clbt_S)
        
        if abs(truth) > 1e-15:
            err_A = abs(truth - est_A) / abs(truth) * 100
            err_B = abs(truth - est_B) / abs(truth) * 100
            err_S = abs(truth - est_S) / abs(truth) * 100
        else:
            err_A = abs(est_A) * 1e6
            err_B = abs(est_B) * 1e6
            err_S = abs(est_S) * 1e6
        
        # Determine winner between Noisy and Staged (skip Clean as it's the ideal)
        if err_S < err_B:
            winner = "Staged"
            wins['Staged'] += 1
        elif err_B < err_S:
            winner = "Noisy"
            wins['Noisy'] += 1
        else:
            winner = "Tie"
        
        print(f"{name:<10}{truth:>+14.6e}"
              f"{est_A:>+14.6e}{err_A:>7.2f}%"
              f"{est_B:>+14.6e}{err_B:>7.2f}%"
              f"{est_S:>+14.6e}{err_S:>7.2f}%"
              f"  {winner:>10}")
    
    print("=" * 150)
    print(f"\nSCOREBOARD (Noisy KF vs LLM Staged):")
    print(f"  LLM Staged wins: {wins['Staged']}/30 parameters")
    print(f"  Noisy KF   wins: {wins['Noisy']}/30 parameters")
    print(f"  Ties:            {30 - wins['Staged'] - wins['Noisy']}/30 parameters")
    print("=" * 150)

    # ── Plots ──
    P_C, X_C = res_clean[2], res_clean[3]
    P_N, X_N = res_noisy[2], res_noisy[3]
    P_S, X_S = res_staged[2], res_staged[3]
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    
    min_len = min(len(P_C), len(P_N), len(P_S))
    time_arr = np.arange(min_len) * (2 * ts)
    
    for i in range(6, 36):
        name = STATE_LABELS_36[i]
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
        
        ax1 = axes[0]
        ax1.plot(time_arr, np.sqrt(P_C[:min_len, i]), color='blue', label='Clean', alpha=0.8)
        ax1.plot(time_arr, np.sqrt(P_N[:min_len, i]), color='red',  label='Noisy KF', linestyle='--', linewidth=2)
        ax1.plot(time_arr, np.sqrt(P_S[:min_len, i]), color='green', label='LLM Staged', linewidth=2, alpha=0.7)
        for b_idx in res_staged[4][:-1]:
            if b_idx < min_len:
                ax1.axvline(x=b_idx * (2 * ts), color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
        ax1.set_title(f'[{i:02d}] {name}  —  Uncertainty (std dev)')
        ax1.legend(); ax1.grid(True)
        
        ax2 = axes[1]
        ax2.plot(time_arr, X_C[:min_len, i], color='blue', label='Clean', alpha=0.8)
        ax2.plot(time_arr, X_N[:min_len, i], color='red', label='Noisy KF', linestyle='--', linewidth=2)
        ax2.plot(time_arr, X_S[:min_len, i], color='green', label='LLM Staged', linewidth=2, alpha=0.7)
        for b_idx in res_staged[4][:-1]:
            if b_idx < min_len:
                ax2.axvline(x=b_idx * (2 * ts), color='gray', linestyle=':', linewidth=1.5, alpha=0.8)
        ax2.set_title(f'[{i:02d}] {name}  —  State Estimate')
        ax2.legend(); ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{i:02d}_{name}.svg"), format='svg')
        plt.close()
        
    print(f"\nDone! Plots saved to {out_dir}")

if __name__ == "__main__":
    main()
