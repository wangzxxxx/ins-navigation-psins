"""
test_calibration_innovation_gating.py
--------------------------------------
Innovation-Based Measurement Gating with LLM Analysis.

Two-pass architecture:
  Pass 1 (Scout Run): Standard KF records the innovation (residual) sequence
          at every ZUPT measurement update. After the run, innovations are
          grouped into time windows and their statistics (mean, std, max)
          are fed to the LLM.
  LLM Analysis: The LLM identifies "toxic zones" — time windows where
          innovations are abnormally large, biased, or structured.
  Pass 2 (Gated Run): The KF re-runs, but skips measurement updates
          during any toxic zone flagged by the LLM.

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


# =====================================================================
# LLM Innovation Analysis
# =====================================================================
def get_llm_toxic_zones(innovation_log, ts):
    """
    Analyze innovation statistics per time window and ask LLM to identify
    'toxic zones' where measurement updates should be skipped.
    
    innovation_log: list of (time_sec, innovation_norm, innovation_xyz)
    """
    if client is None or len(innovation_log) == 0:
        return set()
    
    # Group innovations into 10-second windows
    innov_arr = np.array(innovation_log)  # [N, 5]: time, norm, ix, iy, iz
    max_time = innov_arr[-1, 0]
    window_size = 10.0  # seconds
    n_windows = int(max_time / window_size) + 1
    
    report_lines = []
    for w in range(n_windows):
        t_start = w * window_size
        t_end = (w + 1) * window_size
        mask = (innov_arr[:, 0] >= t_start) & (innov_arr[:, 0] < t_end)
        subset = innov_arr[mask]
        if len(subset) == 0:
            continue
        
        norms = subset[:, 1]
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)
        max_norm = np.max(norms)
        mean_xyz = np.mean(subset[:, 2:5], axis=0)
        
        # Detect if this is a stationary or rotating window
        is_mostly_zupt = len(subset) > 3  # If there are ZUPT updates, it's stationary
        
        report_lines.append(
            f"  Window [{t_start:6.0f}s - {t_end:6.0f}s]: "
            f"n_updates={len(subset):3d}, "
            f"innov_norm: mean={mean_norm:.4e}, std={std_norm:.4e}, max={max_norm:.4e}, "
            f"mean_bias=[{mean_xyz[0]:+.3e},{mean_xyz[1]:+.3e},{mean_xyz[2]:+.3e}]"
        )
    
    report = "\n".join(report_lines)
    
    # Compute overall statistics for context
    all_norms = innov_arr[:, 1]
    global_mean = np.mean(all_norms)
    global_std = np.std(all_norms)
    
    system_prompt = f"""You are an expert in Kalman Filter innovation sequence analysis for IMU calibration.

The innovations (residuals) r_k = y_k - H*x_k|k-1 should ideally be zero-mean white Gaussian noise.
If a time window shows innovations that are:
  - Abnormally LARGE (norm >> global average): sensor noise spike or model mismatch
  - Strongly BIASED (mean far from zero): systematic error that will contaminate the filter
  - Highly VARIABLE (std >> global std): unstable period

Then that window is a "toxic zone" and measurement updates should be SKIPPED.

GLOBAL STATISTICS:
  Overall innovation norm: mean={global_mean:.4e}, std={global_std:.4e}

RULES:
- A window is TOXIC if its mean norm > 2x the global mean, OR its max > 5x the global mean
- Windows during rotation transitions (high norm spikes) are usually toxic
- Windows with mean_bias significantly different from [0,0,0] indicate systematic contamination  
- Be selective: typically 10-30% of windows are toxic. Don't flag everything.
- Output the time ranges of toxic windows

OUTPUT FORMAT:
REASONING: [Your analysis of the innovation pattern]
TOXIC_WINDOWS: [start1-end1, start2-end2, ...] (in seconds)
If no windows are toxic: TOXIC_WINDOWS: NONE
"""

    user_prompt = f"""
Here are the innovation statistics grouped by 10-second windows:
{report}

Which time windows should be flagged as toxic (skip measurement updates)?
"""

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]

    try:
        response = client.chat.completions.create(model=model_name, messages=messages)
        content = response.choices[0].message.content.strip()
        
        print("\n" + "="*60)
        print("[LLM Innovation Analysis]")
        safe = content.encode(sys.stdout.encoding or 'utf-8', errors='replace').decode(sys.stdout.encoding or 'utf-8')
        print(safe)
        print("="*60 + "\n")
        
        # Parse toxic windows
        toxic_times = set()
        match = re.search(r'TOXIC_WINDOWS:\s*(.+)', content)
        if match:
            windows_str = match.group(1).strip()
            if windows_str.upper() == 'NONE':
                return set()
            # Parse ranges like "10-20, 30-40, 50-60"
            ranges = re.findall(r'(\d+)\s*-\s*(\d+)', windows_str)
            for start_s, end_s in ranges:
                t_start = float(start_s)
                t_end = float(end_s)
                toxic_times.add((t_start, t_end))
        
        return toxic_times
    except Exception as e:
        print(f"  [LLM] Error: {e}")
        return set()


def is_in_toxic_zone(t_sec, toxic_zones):
    """Check if a given time falls within any toxic zone."""
    for t_start, t_end in toxic_zones:
        if t_start <= t_sec <= t_end:
            return True
    return False


# =====================================================================
# Standard Building Blocks (same as staged version)
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
    pvec[6:9] = 0.1 * glv.dph
    pvec[9:12] = 1.0 * glv.mg
    pvec[12:15] = [100*glv.ppm, 100*glv.sec, 100*glv.sec]
    pvec[15:18] = [100*glv.sec, 100*glv.ppm, 100*glv.sec]
    pvec[18:21] = [100*glv.sec, 100*glv.sec, 100*glv.ppm]
    pvec[21] = 100*glv.ppm; pvec[22] = 100*glv.sec; pvec[23] = 100*glv.sec
    pvec[24] = 100*glv.ppm; pvec[25] = 100*glv.sec; pvec[26] = 100*glv.ppm
    pvec[27:30] = 100 * glv.ugpg2
    pvec[30:33] = 0.1; pvec[33:36] = 0.1
    kf['Pxk'] = np.diag(pvec)**2
    Hk = np.zeros((3, n)); Hk[:, 3:6] = np.eye(3)
    kf['Hk'] = Hk; kf['xk'] = np.zeros(n); kf['I'] = np.eye(n)
    return kf

def getFt_36(fb, wb, Cnb, wnie, SS):
    n = 36
    wX = askew(wnie); fX = askew(Cnb @ fb)
    fx, fy, fz = fb[0], fb[1], fb[2]
    wx, wy, wz = wb[0], wb[1], wb[2]
    CDf2 = Cnb @ np.diag(fb**2)
    Ca_upper = np.zeros((3, 6))
    Ca_upper[:, 0] = Cnb[:, 0]*fx; Ca_upper[:, 1] = Cnb[:, 0]*fy
    Ca_upper[:, 2] = Cnb[:, 0]*fz; Ca_upper[:, 3] = Cnb[:, 1]*fy
    Ca_upper[:, 4] = Cnb[:, 1]*fz; Ca_upper[:, 5] = Cnb[:, 2]*fz
    Ft = np.zeros((n, n))
    Ft[0:3, 0:3] = -wX; Ft[0:3, 6:9] = -Cnb
    Ft[0:3, 12:15] = -wx*Cnb; Ft[0:3, 15:18] = -wy*Cnb; Ft[0:3, 18:21] = -wz*Cnb
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


# =====================================================================
# Main Calibration Engine (with optional innovation gating)
# =====================================================================
def run_calibration(imu1, pos0, ts, gating_mode=False, label=""):
    """
    gating_mode: If True, performs a two-pass approach per iteration:
      - Scout pass: records innovations
      - LLM analyzes innovations and returns toxic zones
      - Gated pass: skips measurements in toxic zones
    """
    eth  = Earth(pos0)
    wnie = glv.wie * np.array([0, math.cos(pos0[0]), math.sin(pos0[0])])
    gn   = np.array([0, 0, -eth.g])
    Cba  = np.eye(3)
    nn, _, nts, _ = nnts(2, ts)
    frq2 = int(1 / ts / 2) - 1
    n_states = 36

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

    def run_single_pass(clbt_in, toxic_zones=None, record_innovations=False):
        """Run one complete KF pass through the data."""
        kf = clbtkfinit_36(nts)
        
        imu_align = apply_clbt(imu1[frq2:kstatic, :], clbt_in)
        _, _, _, qnb = alignsb(imu_align, pos0)
        vn = np.zeros(3)
        t1s = 0.0
        t_global = 0.0
        
        pass_P = []; pass_X = []
        innovation_log = []
        n_gated = 0; n_total = 0

        for k in range(2 * frq2, length - frq2, nn):
            k1 = k + nn - 1
            wm = imu1[k:k1+1, 0:3]
            vm = imu1[k:k1+1, 3:6]
            dwb = np.mean(dotwf[k:k1+1, 0:3], axis=0)
            phim, dvbm = cnscl(np.hstack((wm, vm)))
            phim = clbt_in['Kg'] @ phim - clbt_in['eb'] * nts
            dvbm = clbt_in['Ka'] @ dvbm - clbt_in['db'] * nts
            wb = phim / nts; fb = dvbm / nts

            SS = imulvS(wb, dwb, Cba)
            fL = SS[:, 0:6] @ np.concatenate((clbt_in['rx'], clbt_in['ry']))
            fn = qmulv(qnb, fb - clbt_in['Ka2']*(fb**2) - fL)
            vn = vn + (rotv(-wnie*nts/2, fn) + gn) * nts
            qnb = qupdt2(qnb, phim, wnie * nts)

            t1s += nts
            t_global += nts
            Ft = getFt_36(fb, wb, q2mat(qnb), wnie, SS)
            kf['Phikk_1'] = np.eye(n_states) + Ft * nts
            kf = kfupdate(kf, TimeMeasBoth='T')

            if t1s > (0.2 - ts / 2):
                t1s = 0.0
                ww = np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
                if np.linalg.norm(ww) / ts < 20 * glv.dph:
                    n_total += 1
                    
                    # Record innovation before update
                    if record_innovations:
                        # Innovation = yk - Hk @ xk_predicted
                        innov = vn - kf['Hk'] @ kf['xk']
                        innov_norm = np.linalg.norm(innov)
                        innovation_log.append([t_global, innov_norm, innov[0], innov[1], innov[2]])
                    
                    # Check if in toxic zone
                    if toxic_zones and is_in_toxic_zone(t_global, toxic_zones):
                        n_gated += 1
                        # SKIP measurement update — only time update was done
                    else:
                        kf = kfupdate(kf, yk=vn, TimeMeasBoth='M')
                
                pass_P.append(np.diag(kf['Pxk']))
                pass_X.append(np.copy(kf['xk']))

        return kf, clbt_in, pass_P, pass_X, innovation_log, n_gated, n_total

    for it in range(iterations):
        print(f"  [{label}] Iter {it+1}/{iterations}")
        
        if gating_mode and it > 0:
            # --- Two-pass approach ---
            # Scout pass: record innovations
            print(f"    Scout pass (recording innovations)...")
            _, _, scout_P, scout_X, innov_log, _, _ = run_single_pass(
                clbt, toxic_zones=None, record_innovations=True
            )
            
            # LLM analysis
            toxic_zones = get_llm_toxic_zones(innov_log, ts)
            if toxic_zones:
                total_toxic_s = sum(e - s for s, e in toxic_zones)
                print(f"    LLM flagged {len(toxic_zones)} toxic zones ({total_toxic_s:.0f}s total)")
            else:
                print(f"    LLM: No toxic zones detected")
            
            # Gated pass: skip measurements in toxic zones
            print(f"    Gated pass (skipping toxic zones)...")
            kf, _, iter_P, iter_X, _, n_gated, n_total = run_single_pass(
                clbt, toxic_zones=toxic_zones, record_innovations=False
            )
            if n_total > 0:
                print(f"    Gated {n_gated}/{n_total} measurements ({100*n_gated/n_total:.1f}%)")
        else:
            # Standard single pass
            kf, _, iter_P, iter_X, _, _, _ = run_single_pass(
                clbt, toxic_zones=None, record_innovations=False
            )
        
        P_trace.extend(iter_P)
        X_trace.extend(iter_X)
        
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

    ARW   = 0.005 * glv.dpsh
    VRW   = 5.0   * glv.ugpsHz
    BI_G  = 0.002 * glv.dph
    BI_A  = 5.0   * glv.ug
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
    res_clean = run_calibration(imu_clean, pos0, ts, gating_mode=False, label="Clean")

    print("\n[B] Noisy (Standard KF, 5 iter)...")
    res_noisy = run_calibration(imu_noisy, pos0, ts, gating_mode=False, label="Noisy KF")
    
    print("\n[C] Noisy + Innovation Gating (5 iter, LLM toxic zone analysis)...")
    res_gated = run_calibration(imu_noisy, pos0, ts, gating_mode=True, label="Gated")

    # ── Calibration Accuracy Comparison ──
    clbt_A, clbt_B, clbt_G = res_clean[0], res_noisy[0], res_gated[0]
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
             f"{'Gated Est':>14}{'Err%':>8}" \
             f"  {'Winner':>10}"
    print(header)
    print("-" * 150)
    
    wins = {'Noisy': 0, 'Gated': 0}
    
    for name, truth, get_est in params:
        est_A = get_est(clbt_A)
        est_B = get_est(clbt_B)
        est_G = get_est(clbt_G)
        
        if abs(truth) > 1e-15:
            err_A = abs(truth - est_A) / abs(truth) * 100
            err_B = abs(truth - est_B) / abs(truth) * 100
            err_G = abs(truth - est_G) / abs(truth) * 100
        else:
            err_A = abs(est_A) * 1e6
            err_B = abs(est_B) * 1e6
            err_G = abs(est_G) * 1e6
        
        if err_G < err_B:
            winner = "Gated"; wins['Gated'] += 1
        elif err_B < err_G:
            winner = "Noisy"; wins['Noisy'] += 1
        else:
            winner = "Tie"
        
        print(f"{name:<10}{truth:>+14.6e}"
              f"{est_A:>+14.6e}{err_A:>7.2f}%"
              f"{est_B:>+14.6e}{err_B:>7.2f}%"
              f"{est_G:>+14.6e}{err_G:>7.2f}%"
              f"  {winner:>10}")
    
    print("=" * 150)
    print(f"\nSCOREBOARD (Noisy KF vs Innovation Gating):")
    print(f"  Gated    wins: {wins['Gated']}/30")
    print(f"  Noisy KF wins: {wins['Noisy']}/30")
    print(f"  Ties:          {30 - wins['Gated'] - wins['Noisy']}/30")
    print("=" * 150)

    # ── Plots ──
    P_C, X_C = res_clean[2], res_clean[3]
    P_N, X_N = res_noisy[2], res_noisy[3]
    P_G, X_G = res_gated[2], res_gated[3]
    
    out_dir = os.path.dirname(os.path.abspath(__file__))
    min_len = min(len(P_C), len(P_N), len(P_G))
    time_arr = np.arange(min_len) * (2 * ts)
    
    for i in range(6, 36):
        name = STATE_LABELS_36[i]
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
        
        ax1 = axes[0]
        ax1.plot(time_arr, np.sqrt(P_C[:min_len, i]), color='blue', label='Clean', alpha=0.8)
        ax1.plot(time_arr, np.sqrt(P_N[:min_len, i]), color='red',  label='Noisy KF', linestyle='--', linewidth=2)
        ax1.plot(time_arr, np.sqrt(P_G[:min_len, i]), color='green', label='Gated', linewidth=2, alpha=0.7)
        for b_idx in res_gated[4][:-1]:
            if b_idx < min_len:
                ax1.axvline(x=b_idx * (2*ts), color='gray', ls=':', lw=1.5, alpha=0.8)
        ax1.set_title(f'[{i:02d}] {name} — Uncertainty (std dev)')
        ax1.legend(); ax1.grid(True)
        
        ax2 = axes[1]
        ax2.plot(time_arr, X_C[:min_len, i], color='blue', label='Clean', alpha=0.8)
        ax2.plot(time_arr, X_N[:min_len, i], color='red', label='Noisy KF', linestyle='--', linewidth=2)
        ax2.plot(time_arr, X_G[:min_len, i], color='green', label='Gated', linewidth=2, alpha=0.7)
        for b_idx in res_gated[4][:-1]:
            if b_idx < min_len:
                ax2.axvline(x=b_idx * (2*ts), color='gray', ls=':', lw=1.5, alpha=0.8)
        ax2.set_title(f'[{i:02d}] {name} — State Estimate')
        ax2.legend(); ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{i:02d}_{name}.svg"), format='svg')
        plt.close()
        
    print(f"\nDone! Plots saved to {out_dir}")

if __name__ == "__main__":
    main()
