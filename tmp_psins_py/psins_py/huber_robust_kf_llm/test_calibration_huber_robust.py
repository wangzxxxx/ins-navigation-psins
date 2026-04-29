"""
test_calibration_huber_robust.py
---------------------------------
Huber-Based Robust Kalman Filter with LLM-Adaptive Gamma Threshold.

Core idea: Before each ZUPT measurement update, compute the innovation's
Mahalanobis distance. If it exceeds gamma, inflate R proportionally to
bring the effective distance down to gamma. This selectively suppresses
anomalous measurements while fully absorbing good ones.

The LLM sets gamma based on the current motion phase:
  - Post-rotation transition: gamma = small (strict, suppress vibration remnants)
  - Long stationary: gamma = normal (allow standard updates)
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
    ['phi_x', 'phi_y', 'phi_z'] + ['dv_x', 'dv_y', 'dv_z'] +
    ['eb_x', 'eb_y', 'eb_z'] + ['db_x', 'db_y', 'db_z'] +
    ['Kg_xx', 'Kg_yx', 'Kg_zx', 'Kg_xy', 'Kg_yy', 'Kg_zy', 'Kg_xz', 'Kg_yz', 'Kg_zz'] +
    ['Ka_xx', 'Ka_xy', 'Ka_xz', 'Ka_yy', 'Ka_yz', 'Ka_zz'] +
    ['Ka2_x', 'Ka2_y', 'Ka2_z'] +
    ['rx_x', 'rx_y', 'rx_z'] + ['ry_x', 'ry_y', 'ry_z']
)


# =====================================================================
# LLM Gamma Advisor
# =====================================================================
def get_llm_gamma_schedule(iteration, prev_innov_stats=None):
    """
    Ask LLM for a gamma schedule: different gamma values for different
    motion phases within the 19-position calibration.
    
    Returns: dict with keys:
      'gamma_transition': gamma for the first N seconds after rotation stops
      'gamma_steady':     gamma for long stationary periods
      'transition_duration': how many seconds after rotation to use strict gamma
    """
    if client is None:
        return {'gamma_transition': 2.0, 'gamma_steady': 3.5, 'transition_duration': 3.0}
    
    stats_str = ""
    if prev_innov_stats:
        stats_str = f"""
Previous iteration innovation statistics:
  Mean Mahalanobis dist: {prev_innov_stats['mean_d']:.3f}
  Std Mahalanobis dist: {prev_innov_stats['std_d']:.3f}
  Max Mahalanobis dist: {prev_innov_stats['max_d']:.3f}
  Fraction exceeding d=3.0: {prev_innov_stats['frac_gt3']:.1%}
  Fraction exceeding d=5.0: {prev_innov_stats['frac_gt5']:.1%}
"""

    system_prompt = """You are a robust estimation expert tuning a Huber-based Robust Kalman Filter for IMU calibration.

The filter uses Mahalanobis distance of the innovation to decide whether to trust each ZUPT measurement:
  - d <= gamma: FULL update (standard KF)
  - d > gamma: R inflated by (d/gamma)^2, reducing the update proportionally

You need to set THREE parameters for the upcoming iteration:
1. GAMMA_TRANSITION: Threshold for the first few seconds after rotation stops (transition period). 
   Smaller = stricter = more noise suppression but slower convergence.
   Recommended range: 1.0 to 3.0
2. GAMMA_STEADY: Threshold for long stationary periods.
   Recommended range: 2.0 to 5.0
3. TRANSITION_DURATION: How many seconds after rotation stops to use GAMMA_TRANSITION before switching to GAMMA_STEADY.
   Recommended range: 1.0 to 5.0

For a 3-dimensional chi-squared distribution:
  - d ~ chi(3): mean ≈ 1.73, 95th percentile ≈ 2.8, 99th percentile ≈ 3.4
  - Setting gamma = 2.0 rejects ~15% of normal measurements
  - Setting gamma = 3.0 rejects ~2% of normal measurements
  - Setting gamma = 4.0 rejects ~0.3% of normal measurements

STRATEGY:
- Early iterations (1-2): Be moderately strict to suppress initial noise
- Middle iterations (3-4): Be stricter, especially in transition periods  
- Final iteration (5): Be normal to allow fine convergence

OUTPUT FORMAT:
REASONING: [Your analysis]
GAMMA_TRANSITION: [value]
GAMMA_STEADY: [value]
TRANSITION_DURATION: [value in seconds]
"""

    user_prompt = f"""
This is iteration {iteration}/5 of the calibration.
{stats_str}
What gamma schedule should I use for this iteration?
"""

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]

    try:
        response = client.chat.completions.create(model=model_name, messages=messages)
        content = response.choices[0].message.content.strip()
        
        print(f"\n{'='*60}\n[LLM Gamma Advisor - Iter {iteration}]\n{content}\n{'='*60}\n")
        
        gt = 2.0; gs = 3.5; td = 3.0
        m = re.search(r'GAMMA_TRANSITION:\s*([0-9.]+)', content)
        if m: gt = float(m.group(1))
        m = re.search(r'GAMMA_STEADY:\s*([0-9.]+)', content)
        if m: gs = float(m.group(1))
        m = re.search(r'TRANSITION_DURATION:\s*([0-9.]+)', content)
        if m: td = float(m.group(1))
        
        return {
            'gamma_transition': np.clip(gt, 0.5, 6.0),
            'gamma_steady': np.clip(gs, 1.0, 8.0),
            'transition_duration': np.clip(td, 0.5, 10.0)
        }
    except Exception as e:
        print(f"  [LLM] Error: {e}")
        return {'gamma_transition': 2.0, 'gamma_steady': 3.5, 'transition_duration': 3.0}


# =====================================================================
# Huber Robust Measurement Update
# =====================================================================
def huber_kfupdate(kf, yk, gamma):
    """
    Huber-based robust measurement update.
    If Mahalanobis distance of innovation > gamma, inflate R proportionally.
    Returns: updated kf, mahalanobis_distance, was_robustified
    """
    Hk = kf['Hk']
    Pxk = kf['Pxk']
    Rk = kf['Rk']
    xk = kf['xk']
    n = kf['n']
    
    # Innovation
    innov = yk - Hk @ xk
    
    # Innovation covariance
    S = Hk @ Pxk @ Hk.T + Rk
    
    # Mahalanobis distance
    try:
        S_inv = np.linalg.inv(S)
        d_sq = float(innov.T @ S_inv @ innov)
        d_mah = math.sqrt(max(d_sq, 0.0))
    except np.linalg.LinAlgError:
        # If S is singular, skip this update
        return kf, 0.0, False
    
    robustified = False
    if d_mah > gamma and gamma > 0:
        # Inflate R by (d/gamma)^2 to bring effective distance down to gamma
        inflation = (d_mah / gamma) ** 2
        Rk_eff = Rk * inflation
        robustified = True
    else:
        Rk_eff = Rk
    
    # Compute Kalman gain with (possibly inflated) R
    S_eff = Hk @ Pxk @ Hk.T + Rk_eff
    try:
        Kk = Pxk @ Hk.T @ np.linalg.inv(S_eff)
    except np.linalg.LinAlgError:
        return kf, d_mah, False
    
    # State update
    kf['xk'] = xk + Kk @ innov
    
    # Covariance update (Joseph form for numerical stability)
    I_KH = np.eye(n) - Kk @ Hk
    kf['Pxk'] = I_KH @ Pxk @ I_KH.T + Kk @ Rk_eff @ Kk.T
    
    return kf, d_mah, robustified


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
    pvec[6:9] = 0.1 * glv.dph; pvec[9:12] = 1.0 * glv.mg
    pvec[12:15] = [100*glv.ppm, 100*glv.sec, 100*glv.sec]
    pvec[15:18] = [100*glv.sec, 100*glv.ppm, 100*glv.sec]
    pvec[18:21] = [100*glv.sec, 100*glv.sec, 100*glv.ppm]
    pvec[21] = 100*glv.ppm; pvec[22] = 100*glv.sec; pvec[23] = 100*glv.sec
    pvec[24] = 100*glv.ppm; pvec[25] = 100*glv.sec; pvec[26] = 100*glv.ppm
    pvec[27:30] = 100 * glv.ugpg2; pvec[30:33] = 0.1; pvec[33:36] = 0.1
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
    Ca_upper[:, 0] = Cnb[:, 0]*fx; Ca_upper[:, 1] = Cnb[:, 0]*fy
    Ca_upper[:, 2] = Cnb[:, 0]*fz; Ca_upper[:, 3] = Cnb[:, 1]*fy
    Ca_upper[:, 4] = Cnb[:, 1]*fz; Ca_upper[:, 5] = Cnb[:, 2]*fz
    Ft = np.zeros((n, n))
    Ft[0:3, 0:3] = -wX; Ft[0:3, 6:9] = -Cnb
    Ft[0:3, 12:15] = -wx*Cnb; Ft[0:3, 15:18] = -wy*Cnb; Ft[0:3, 18:21] = -wz*Cnb
    Ft[3:6, 0:3] = fX; Ft[3:6, 9:12] = Cnb; Ft[3:6, 21:27] = Ca_upper
    Ft[3:6, 27:30] = CDf2; Ft[3:6, 30:36] = Cnb @ SS[:, 0:6]
    return Ft

def clbtkffeedback_pruned(kf, clbt):
    xk = kf['xk']
    dKg = xk[12:21].reshape(3, 3).T
    clbt['Kg'] = (np.eye(3) - dKg) @ clbt['Kg']
    dKa = Ka_from_upper(xk[21:27])
    clbt['Ka'] = (np.eye(3) - dKa) @ clbt['Ka']
    clbt['Ka2'] += xk[27:30]; clbt['eb'] += xk[6:9]; clbt['db'] += xk[9:12]
    clbt['rx'] += xk[30:33]; clbt['ry'] += xk[33:36]
    return clbt


# =====================================================================
# Main Calibration Engine
# =====================================================================
def run_calibration(imu1, pos0, ts, huber_mode=False, label=""):
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
    iterations = 5
    P_trace, X_trace, iter_bounds = [], [], []

    def apply_clbt(imu_s, c):
        res = np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    prev_innov_stats = None

    for it in range(iterations):
        # Get gamma schedule from LLM
        gamma_sched = None
        if huber_mode:
            gamma_sched = get_llm_gamma_schedule(it + 1, prev_innov_stats)
            print(f"  [{label}] Iter {it+1}/{iterations} | gamma_trans={gamma_sched['gamma_transition']:.2f}, "
                  f"gamma_steady={gamma_sched['gamma_steady']:.2f}, trans_dur={gamma_sched['transition_duration']:.1f}s")
        else:
            print(f"  [{label}] Iter {it+1}/{iterations}")
        
        kf = clbtkfinit_36(nts)

        if it == iterations - 1:
            kf['Pxk'] = kf['Pxk'] * 100
            kf['Pxk'][:, 2] = 0; kf['Pxk'][2, :] = 0
            kf['xk'] = np.zeros(36)

        imu_align = apply_clbt(imu1[frq2:kstatic, :], clbt)
        _, _, _, qnb = alignsb(imu_align, pos0)
        vn = np.zeros(3); t1s = 0.0; t_global = 0.0
        
        # Track rotation/stationary transitions for gamma selection
        was_rotating = False
        time_since_rotation_stopped = 999.0

        # Innovation stats for LLM feedback
        mah_distances = []
        n_robustified = 0; n_total_zupt = 0

        for k in range(2 * frq2, length - frq2, nn):
            k1 = k + nn - 1
            wm = imu1[k:k1+1, 0:3]; vm = imu1[k:k1+1, 3:6]
            dwb = np.mean(dotwf[k:k1+1, 0:3], axis=0)
            phim, dvbm = cnscl(np.hstack((wm, vm)))
            phim = clbt['Kg'] @ phim - clbt['eb'] * nts
            dvbm = clbt['Ka'] @ dvbm - clbt['db'] * nts
            wb = phim / nts; fb = dvbm / nts

            SS = imulvS(wb, dwb, Cba)
            fL = SS[:, 0:6] @ np.concatenate((clbt['rx'], clbt['ry']))
            fn = qmulv(qnb, fb - clbt['Ka2']*(fb**2) - fL)
            vn = vn + (rotv(-wnie*nts/2, fn) + gn) * nts
            qnb = qupdt2(qnb, phim, wnie * nts)

            t1s += nts; t_global += nts
            Ft = getFt_36(fb, wb, q2mat(qnb), wnie, SS)
            kf['Phikk_1'] = np.eye(36) + Ft * nts
            kf = kfupdate(kf, TimeMeasBoth='T')

            if t1s > (0.2 - ts / 2):
                t1s = 0.0
                ww = np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
                is_static = np.linalg.norm(ww) / ts < 20 * glv.dph
                
                if not is_static:
                    was_rotating = True
                    time_since_rotation_stopped = 0.0
                else:
                    if was_rotating:
                        was_rotating = False
                        time_since_rotation_stopped = 0.0
                    else:
                        time_since_rotation_stopped += 0.2
                
                if is_static:
                    n_total_zupt += 1
                    
                    if huber_mode and gamma_sched:
                        # Select gamma based on motion phase
                        if time_since_rotation_stopped < gamma_sched['transition_duration']:
                            gamma = gamma_sched['gamma_transition']
                        else:
                            gamma = gamma_sched['gamma_steady']
                        
                        kf, d_mah, was_robust = huber_kfupdate(kf, vn, gamma)
                        mah_distances.append(d_mah)
                        if was_robust:
                            n_robustified += 1
                    else:
                        kf = kfupdate(kf, yk=vn, TimeMeasBoth='M')
                
                P_trace.append(np.diag(kf['Pxk']))
                X_trace.append(np.copy(kf['xk']))

        if huber_mode and n_total_zupt > 0:
            mah_arr = np.array(mah_distances) if mah_distances else np.array([0.0])
            prev_innov_stats = {
                'mean_d': float(np.mean(mah_arr)),
                'std_d': float(np.std(mah_arr)),
                'max_d': float(np.max(mah_arr)),
                'frac_gt3': float(np.mean(mah_arr > 3.0)),
                'frac_gt5': float(np.mean(mah_arr > 5.0)),
            }
            print(f"    Robustified {n_robustified}/{n_total_zupt} ZUPTs ({100*n_robustified/n_total_zupt:.1f}%), "
                  f"mean_d={prev_innov_stats['mean_d']:.2f}, max_d={prev_innov_stats['max_d']:.2f}")

        if it != iterations - 1:
            clbt = clbtkffeedback_pruned(kf, clbt)
        
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
        [1,    0, 1, 0,  90, 9, 70, 70], [2,    0, 1, 0,  90, 9, 20, 20],
        [3,    0, 1, 0,  90, 9, 20, 20], [4,    0, 1, 0, -90, 9, 20, 20],
        [5,    0, 1, 0, -90, 9, 20, 20], [6,    0, 1, 0, -90, 9, 20, 20],
        [7,    0, 0, 1,  90, 9, 20, 20], [8,    1, 0, 0,  90, 9, 20, 20],
        [9,    1, 0, 0,  90, 9, 20, 20], [10,   1, 0, 0,  90, 9, 20, 20],
        [11,  -1, 0, 0,  90, 9, 20, 20], [12,  -1, 0, 0,  90, 9, 20, 20],
        [13,  -1, 0, 0,  90, 9, 20, 20], [14,   0, 0, 1,  90, 9, 20, 20],
        [15,   0, 0, 1,  90, 9, 20, 20], [16,   0, 0,-1,  90, 9, 20, 20],
        [17,   0, 0,-1,  90, 9, 20, 20], [18,   0, 0,-1,  90, 9, 20, 20],
    ], dtype=float)
    paras[:, 4] = paras[:, 4] * glv.deg

    ARW   = 0.005 * glv.dpsh
    VRW   = 5.0   * glv.ugpsHz
    BI_G  = 0.002 * glv.dph
    BI_A  = 5.0   * glv.ug
    TAU_G = 300.0; TAU_A = 300.0

    print("Generating IMU trajectory...")
    att  = attrottt(att0, paras, ts)
    imu, _ = avp2imu(att, pos0)
    clbt_truth = get_default_clbt()
    imu_clean  = imuclbt(imu, clbt_truth)
    imu_noisy = imuadderr_full(imu_clean, ts, arw=ARW, vrw=VRW,
                                bi_g=BI_G, tau_g=TAU_G, bi_a=BI_A, tau_a=TAU_A)

    print("\n[A] Clean (5 iter)...")
    res_clean = run_calibration(imu_clean, pos0, ts, huber_mode=False, label="Clean")

    print("\n[B] Noisy (Standard KF, 5 iter)...")
    res_noisy = run_calibration(imu_noisy, pos0, ts, huber_mode=False, label="Noisy KF")
    
    print("\n[C] Noisy + Huber Robust KF (5 iter, LLM gamma)...")
    res_huber = run_calibration(imu_noisy, pos0, ts, huber_mode=True, label="Huber")

    # ── Accuracy Comparison ──
    clbt_A, clbt_B, clbt_H = res_clean[0], res_noisy[0], res_huber[0]
    dKg_truth = clbt_truth['Kg'] - np.eye(3); dKa_truth = clbt_truth['Ka'] - np.eye(3)
    
    params = [
        ("eb_x", clbt_truth['eb'][0], lambda c: -c['eb'][0]),
        ("eb_y", clbt_truth['eb'][1], lambda c: -c['eb'][1]),
        ("eb_z", clbt_truth['eb'][2], lambda c: -c['eb'][2]),
        ("db_x", clbt_truth['db'][0], lambda c: -c['db'][0]),
        ("db_y", clbt_truth['db'][1], lambda c: -c['db'][1]),
        ("db_z", clbt_truth['db'][2], lambda c: -c['db'][2]),
        ("Kg_xx", dKg_truth[0,0], lambda c: -(c['Kg']-np.eye(3))[0,0]),
        ("Kg_yx", dKg_truth[1,0], lambda c: -(c['Kg']-np.eye(3))[1,0]),
        ("Kg_zx", dKg_truth[2,0], lambda c: -(c['Kg']-np.eye(3))[2,0]),
        ("Kg_xy", dKg_truth[0,1], lambda c: -(c['Kg']-np.eye(3))[0,1]),
        ("Kg_yy", dKg_truth[1,1], lambda c: -(c['Kg']-np.eye(3))[1,1]),
        ("Kg_zy", dKg_truth[2,1], lambda c: -(c['Kg']-np.eye(3))[2,1]),
        ("Kg_xz", dKg_truth[0,2], lambda c: -(c['Kg']-np.eye(3))[0,2]),
        ("Kg_yz", dKg_truth[1,2], lambda c: -(c['Kg']-np.eye(3))[1,2]),
        ("Kg_zz", dKg_truth[2,2], lambda c: -(c['Kg']-np.eye(3))[2,2]),
        ("Ka_xx", dKa_truth[0,0], lambda c: -(c['Ka']-np.eye(3))[0,0]),
        ("Ka_xy", dKa_truth[0,1], lambda c: -(c['Ka']-np.eye(3))[0,1]),
        ("Ka_xz", dKa_truth[0,2], lambda c: -(c['Ka']-np.eye(3))[0,2]),
        ("Ka_yy", dKa_truth[1,1], lambda c: -(c['Ka']-np.eye(3))[1,1]),
        ("Ka_yz", dKa_truth[1,2], lambda c: -(c['Ka']-np.eye(3))[1,2]),
        ("Ka_zz", dKa_truth[2,2], lambda c: -(c['Ka']-np.eye(3))[2,2]),
        ("Ka2_x", clbt_truth['Ka2'][0], lambda c: -c['Ka2'][0]),
        ("Ka2_y", clbt_truth['Ka2'][1], lambda c: -c['Ka2'][1]),
        ("Ka2_z", clbt_truth['Ka2'][2], lambda c: -c['Ka2'][2]),
        ("rx_x", clbt_truth['rx'][0], lambda c: -c['rx'][0]),
        ("rx_y", clbt_truth['rx'][1], lambda c: -c['rx'][1]),
        ("rx_z", clbt_truth['rx'][2], lambda c: -c['rx'][2]),
        ("ry_x", clbt_truth['ry'][0], lambda c: -c['ry'][0]),
        ("ry_y", clbt_truth['ry'][1], lambda c: -c['ry'][1]),
        ("ry_z", clbt_truth['ry'][2], lambda c: -c['ry'][2]),
    ]
    
    print("\n" + "=" * 150)
    print("CALIBRATION ACCURACY COMPARISON — Huber Robust KF (5 iterations each)")
    print("=" * 150)
    header = f"{'Param':<10}{'Truth':>14}{'Clean Est':>14}{'Err%':>8}{'Noisy Est':>14}{'Err%':>8}{'Huber Est':>14}{'Err%':>8}  {'Winner':>10}"
    print(header); print("-" * 150)
    
    wins = {'Noisy': 0, 'Huber': 0}
    for name, truth, get_est in params:
        eA, eB, eH = get_est(clbt_A), get_est(clbt_B), get_est(clbt_H)
        if abs(truth) > 1e-15:
            ea = abs(truth-eA)/abs(truth)*100; eb_ = abs(truth-eB)/abs(truth)*100; eh = abs(truth-eH)/abs(truth)*100
        else:
            ea = abs(eA)*1e6; eb_ = abs(eB)*1e6; eh = abs(eH)*1e6
        w = "Huber" if eh < eb_ else ("Noisy" if eb_ < eh else "Tie")
        if w == "Huber": wins['Huber'] += 1
        elif w == "Noisy": wins['Noisy'] += 1
        print(f"{name:<10}{truth:>+14.6e}{eA:>+14.6e}{ea:>7.2f}%{eB:>+14.6e}{eb_:>7.2f}%{eH:>+14.6e}{eh:>7.2f}%  {w:>10}")
    
    print("=" * 150)
    print(f"\nSCOREBOARD:")
    print(f"  Huber    wins: {wins['Huber']}/30")
    print(f"  Noisy KF wins: {wins['Noisy']}/30")
    print(f"  Ties:          {30 - wins['Huber'] - wins['Noisy']}/30")
    print("=" * 150)

    # ── Plots ──
    P_C, X_C = res_clean[2], res_clean[3]; P_N, X_N = res_noisy[2], res_noisy[3]
    P_H, X_H = res_huber[2], res_huber[3]
    out_dir = os.path.dirname(os.path.abspath(__file__))
    min_len = min(len(P_C), len(P_N), len(P_H))
    time_arr = np.arange(min_len) * (2 * ts)
    for i in range(6, 36):
        name = STATE_LABELS_36[i]
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        ax1 = axes[0]
        ax1.plot(time_arr, np.sqrt(P_C[:min_len, i]), 'b', label='Clean', alpha=0.8)
        ax1.plot(time_arr, np.sqrt(P_N[:min_len, i]), 'r--', label='Noisy KF', lw=2)
        ax1.plot(time_arr, np.sqrt(P_H[:min_len, i]), 'g', label='Huber', lw=2, alpha=0.7)
        for b in res_huber[4][:-1]:
            if b < min_len: ax1.axvline(x=b*(2*ts), color='gray', ls=':', lw=1.5, alpha=0.8)
        ax1.set_title(f'[{i:02d}] {name} — Uncertainty'); ax1.legend(); ax1.grid(True)
        ax2 = axes[1]
        ax2.plot(time_arr, X_C[:min_len, i], 'b', label='Clean', alpha=0.8)
        ax2.plot(time_arr, X_N[:min_len, i], 'r--', label='Noisy KF', lw=2)
        ax2.plot(time_arr, X_H[:min_len, i], 'g', label='Huber', lw=2, alpha=0.7)
        for b in res_huber[4][:-1]:
            if b < min_len: ax2.axvline(x=b*(2*ts), color='gray', ls=':', lw=1.5, alpha=0.8)
        ax2.set_title(f'[{i:02d}] {name} — State Estimate'); ax2.legend(); ax2.grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"{i:02d}_{name}.svg"), format='svg'); plt.close()
    print(f"\nDone! Plots saved to {out_dir}")

if __name__ == "__main__":
    main()
