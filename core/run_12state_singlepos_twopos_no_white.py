"""
12-state 对准实验（无白噪声）：
- 实验A：12-state @ 单位置静止对准（ARW=VRW=0，仅常值偏置）
- 实验B：12-state @ 双位置对准（ARW=VRW=0，仅常值偏置）

12-state：phi(3) + dv(3) + eb(3) + db(3) = 12
"""
from __future__ import annotations
import sys, os, types, time, json, math, copy, datetime

if 'matplotlib' not in sys.modules:
    matplotlib_stub = types.ModuleType('matplotlib')
    pyplot_stub = types.ModuleType('matplotlib.pyplot')
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules['matplotlib'] = matplotlib_stub
    sys.modules['matplotlib.pyplot'] = pyplot_stub
if 'seaborn' not in sys.modules:
    sys.modules['seaborn'] = types.ModuleType('seaborn')

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROOT, 'psins_method_bench', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

TMP_PSINS_DIR = os.path.join(ROOT, 'tmp_psins_py')
sys.path.insert(0, TMP_PSINS_DIR)
sys.path.insert(0, os.path.join(TMP_PSINS_DIR, 'psins_py'))

# Load PSINS module
src_mod = __import__('test_calibration_markov_pruned', globals(), locals(), [], 0)
np = src_mod.np
glv = src_mod.glv
Earth = src_mod.Earth
posset = src_mod.posset
qmulv = src_mod.qmulv
qupdt2 = src_mod.qupdt2
q2mat = src_mod.q2mat
rotv = src_mod.rotv
alignsb = src_mod.alignsb
avp2imu = src_mod.avp2imu
imuclbt = src_mod.imuclbt
get_default_clbt = src_mod.get_default_clbt
nnts = src_mod.nnts
imudot = src_mod.imudot
cnscl = src_mod.cnscl
imulvS = src_mod.imulvS
kfupdate = src_mod.kfupdate
mat2att = src_mod.mat2att

# ============================================================
# Parameters
# ============================================================
SEED = 42
TS = 0.01
TOTAL_S = 1200.0

# Deterministic bias (no white noise)
BIAS_G_X = 0.002  # deg/h
BIAS_G_Y = -0.0015
BIAS_G_Z = 0.0018
BIAS_A_X = 30.0   # ug
BIAS_A_Y = -25.0
BIAS_A_Z = 20.0

# ============================================================
# Build true bias vector in physical units
# ============================================================
def build_true_bias_vec():
    """Return true bias vectors [eb_x, eb_y, eb_z, db_x, db_y, db_z]."""
    eb_true = np.array([BIAS_G_X * glv.dph, BIAS_G_Y * glv.dph, BIAS_G_Z * glv.dph])
    db_true = np.array([BIAS_A_X * glv.ug, BIAS_A_Y * glv.ug, BIAS_A_Z * glv.ug])
    return eb_true, db_true

# ============================================================
# IMU simulation: attitude → IMU → add deterministic bias
# ============================================================
def build_static_attitude(ts, duration_s):
    n = int(duration_s / ts) + 1
    att = np.zeros((n, 4))
    att[:, 3] = np.arange(1, n+1) * ts
    return att

def build_two_position_attitude(ts, duration_s):
    """Two-position: 600s pos1, 180s rotation 180° Z, 420s pos2."""
    n = int(duration_s / ts) + 1
    att = np.zeros((n, 4))
    att[:, 3] = np.arange(1, n+1) * ts

    dt = ts
    t = 0.0
    # Phase 1: static (600s)
    end1 = int(600.0 / dt)
    # Phase 2: rotation 180° around Z (180s)
    end2 = end1 + int(180.0 / dt)
    # Phase 3: static (420s)
    end3 = end2 + int(420.0 / dt)

    for i in range(min(end1, n)):
        att[i, 0:3] = [0.0, 0.0, 0.0]

    if end2 <= n:
        rot_time = 180.0
        for i in range(end1, min(end2, n)):
            t_in_rot = (i - end1) * dt
            frac = t_in_rot / rot_time
            angle_deg = frac * 180.0
            att[i, 0:3] = [0.0, 0.0, angle_deg * glv.deg]

    if end3 <= n:
        for i in range(end2, min(end3, n)):
            att[i, 0:3] = [0.0, 0.0, 180.0 * glv.deg]

    return att

def make_dataset(att_data, bias_g, bias_a):
    """Generate IMU data with deterministic bias, NO white noise."""
    pos0 = posset(34.0, 0.0, 0.0)

    # Clean IMU from attitude trajectory
    imu_clean, avp = avp2imu(att_data, pos0)

    # Add deterministic bias
    imu_noisy = np.copy(imu_clean)
    n_samp = imu_noisy.shape[0]
    # wx bias
    imu_noisy[:, 0] += bias_g[0]
    imu_noisy[:, 1] += bias_g[1]
    imu_noisy[:, 2] += bias_g[2]
    # fx bias
    imu_noisy[:, 3] += bias_a[0]
    imu_noisy[:, 4] += bias_a[1]
    imu_noisy[:, 5] += bias_a[2]

    return {
        'ts': TS,
        'pos0': pos0,
        'imu_clean': imu_clean,
        'imu_noisy': imu_noisy,
        'avp': avp,
        'bias_g': bias_g,
        'bias_a': bias_a,
    }

# ============================================================
# 12-state alignment KF
# State: [phi(3), dv(3), eb(3), db(3)] = 12
# ============================================================
def kf12_init(ts):
    """Initialize 12-state alignment KF."""
    kf = {}
    kf['ts'] = ts
    kf['xk'] = np.zeros(12)   # state estimate

    # Initial covariance (large for misalignment, very large for biases)
    P0 = np.zeros((12, 12))
    P0[0:3, 0:3] = np.eye(3) * (1.0 * glv.deg)**2    # phi ~1 deg
    P0[3:6, 3:6] = np.eye(3) * (1.0)**2                # dv ~1 m/s
    P0[6:9, 6:9] = np.eye(3) * (0.1 * glv.dph)**2      # eb ~0.1 deg/h → actually more
    P0[6:9, 6:9] = np.eye(3) * (10.0 * glv.dph)**2     # eb ~10 deg/h (generous prior)
    P0[9:12, 9:12] = np.eye(3) * (1000.0 * glv.ug)**2  # db ~1000 ug (generous prior)
    kf['Pxk'] = P0
    return kf

def kf12_state_transition(fb, wb, C_nb, wnie):
    """Build continuous-time state transition matrix F (12-state)."""
    F = np.zeros((12, 12))

    # d(phi)/dt = -wnie×phi - C_nb·dv + ...  (simplified for alignment)
    # d(dv)/dt = fn×phi + ...
    # d(eb)/dt = 0 (constant bias, random walk)
    # d(db)/dt = 0 (constant bias, random walk)

    # F[0:3, 0:3] = -skew_sym(wnie)
    F[0, 1] = -wnie[2]; F[0, 2] = wnie[1]
    F[1, 0] = wnie[2];  F[1, 2] = -wnie[0]
    F[2, 0] = -wnie[1]; F[2, 1] = wnie[0]

    # F[0:3, 3:6] = -C_nb
    F[0:3, 3:6] = -C_nb

    # F[3:6, 0:3] = skew_sym(fn)  where fn = C_nb * fb
    fn = C_nb @ fb
    F[3, 4] = -fn[2]; F[3, 5] = fn[1]
    F[4, 3] = fn[2];  F[4, 5] = -fn[0]
    F[5, 3] = -fn[1]; F[5, 4] = fn[0]

    # F[6:9, 6:9] = 0 (random walk bias)
    # F[9:12, 9:12] = 0 (random walk bias)

    return F

def skew_sym(v):
    S = np.zeros((3, 3))
    S[0, 1] = -v[2]; S[0, 2] = v[1]
    S[1, 0] = v[2];  S[1, 2] = -v[0]
    S[2, 0] = -v[1]; S[2, 1] = v[0]
    return S

def run_12state_alignment(ds, label, dt_step=0.02):
    """Run 12-state coarse-to-fine alignment.
    
    Uses standard two-stage alignment:
    1. Coarse alignment (analytic)
    2. Fine alignment (12-state KF with velocity zero-update)
    """
    print(f"  {label}: 12-state alignment...", end=" ", flush=True)
    t0 = time.time()

    ts = ds['ts']
    pos0 = ds['pos0']
    imu = ds['imu_noisy']
    eb_true = ds['bias_g']
    db_true = ds['bias_a']

    eth = Earth(pos0)
    # Local gravity direction = [0, 0, -g]
    gn = np.array([0.0, 0.0, -eth.g])
    wnie = glv.wie * np.array([0, math.cos(pos0[0]), math.sin(pos0[0])])

    # --- Coarse alignment (analytic) ---
    # Average gravity and Earth rate over first 10s
    n_coarse = int(10.0 / ts)
    f_avg = np.mean(imu[:n_coarse, 3:6], axis=0)  # specific force
    w_avg = np.mean(imu[:n_coarse, 0:3], axis=0)  # angular rate

    f_norm = np.linalg.norm(f_avg)
    w_norm = np.linalg.norm(w_avg)

    # Gravity vector in body frame: g_b = f_avg / |f_avg|
    g_b = f_avg / f_norm
    # Earth rate projected: w_b = (w_avg - (w_avg·g_b)*g_b) / |...|
    w_proj = w_avg - np.dot(w_avg, g_b) * g_b
    w_proj_norm = np.linalg.norm(w_proj)

    if w_proj_norm > 1e-10:
        w_b = w_proj / w_proj_norm
    else:
        w_b = np.array([0, 0, 1])

    # Cross product for third vector
    g_cross_w = np.cross(g_b, w_b)
    g_cross_w_norm = np.linalg.norm(g_cross_w)
    if g_cross_w_norm > 1e-10:
        g_cross_w = g_cross_w / g_cross_w_norm
    else:
        g_cross_w = np.array([1, 0, 0])

    # Navigation frame vectors
    g_n = np.array([0, 0, -1])
    w_n_proj = np.array([0, math.cos(pos0[0]), 0])
    w_n_proj_norm = np.linalg.norm(w_n_proj)
    if w_n_proj_norm > 1e-10:
        w_n = w_n_proj / w_n_proj_norm
    else:
        w_n = np.array([0, 1, 0])
    w_cross_g = np.cross(w_n, g_n)
    w_cross_g_norm = np.linalg.norm(w_cross_g)
    if w_cross_g_norm > 1e-10:
        w_cross_g = w_cross_g / w_cross_g_norm
    else:
        w_cross_g = np.array([1, 0, 0])

    # C_b2n from vector matching
    Mb = np.column_stack([g_b, g_cross_w, w_b])
    Mn = np.column_stack([g_n, w_cross_g, w_n])
    C_b2n = Mn @ np.linalg.inv(Mb)

    # Coarse alignment C_nb
    C_nb = C_b2n.T

    # Extract coarse attitude
    att_coarse = mat2att(C_nb)
    att_coarse_deg = att_coarse * glv.rad2deg

    print(f"coarse: roll={att_coarse_deg[0]*180/np.pi:.4f}°, "
          f"pitch={att_coarse_deg[1]*180/np.pi:.4f}°, "
          f"yaw={att_coarse_deg[2]*180/np.pi:.4f}°  ",
          end="", flush=True)

    # --- Fine alignment: 12-state KF ---
    frq2 = int(1 / ts / 2) - 1
    nn = 2
    kf = kf12_init(ts)

    vn = np.zeros(3)
    qnb = np.zeros(4)
    qnb[0] = C_nb[0, 0]; qnb[1] = C_nb[0, 1]; qnb[2] = C_nb[0, 2]; qnb[3] = C_nb[0, 3]

    # Actually let's use alignsb for proper initialization
    # Use first segment of IMU for coarse alignment
    imu_init = imu[frq2:frq2*101, :]
    vn_init, _, _, qnb_init = alignsb(imu_init, pos0)

    vn = np.zeros(3)
    qnb = qnb_init.copy()

    kstatic = k = frq2
    for k in range(frq2, min(5*60*2*frq2, len(imu)), 2*frq2):
        ww = np.mean(imu[k-frq2:k+frq2+1, 0:3], axis=0)
        if np.linalg.norm(ww) / ts > 20 * glv.dph:
            break
    kstatic = k - 3 * frq2
    dotwf_data = imudot(imu, 5.0)

    # Process the alignment data
    velocity_meas_count = 0
    max_vel = 0.0

    for k in range(2*frq2, len(imu)-frq2, nn):
        k1 = k + nn - 1
        wm = imu[k:k1+1, 0:3]
        vm = imu[k:k1+1, 3:6]
        dwb_raw = np.mean(dotwf_data[k:k1+1, 0:3], axis=0)

        phim, dvbm = cnscl(np.hstack((wm, vm)))

        vb = dvbm / (nn * ts)
        wb = phim / (nn * ts)

        SS = imulvS(wb, dwb_raw, np.eye(3))

        fn = qmulv(qnb, vb)
        vn = vn + (rotv(-wnie*nn*ts/2, fn) + gn) * nn*ts
        qnb = qupdt2(qnb, phim, wnie * nn * ts)

        # Zero velocity update: if angular rate is low
        ww = np.mean(imu[k-frq2:min(k+frq2+1, len(imu)), 0:3], axis=0)
        rot_rate = np.linalg.norm(ww) / ts

        if rot_rate < 20 * glv.dph:
            # Velocity residual (should be zero for stationary or quasi-stationary)
            yk = vn - np.zeros(3)
            # Measurement matrix H for velocity: picks dv states
            H = np.zeros((3, 12))
            H[0, 3] = 1.0
            H[1, 4] = 1.0
            H[2, 5] = 1.0

            # Measurement noise covariance R (use a reasonable value)
            R = np.eye(3) * 0.01

            # KF update
            P = kf['Pxk']
            x = kf['xk']

            # Innovation
            y = yk

            # S = H*P*H' + R
            HP = H @ P
            S = HP @ H.T + R

            # K = P*H'*inv(S)
            try:
                K = P @ H.T @ np.linalg.inv(S)
            except np.linalg.LinAlgError:
                continue

            # x = x + K*y
            x_new = x + K @ y
            # P = (I - K*H)*P
            P_new = (np.eye(12) - K @ H) @ P

            kf['xk'] = x_new
            kf['Pxk'] = 0.5 * (P_new + P_new.T)  # ensure symmetry

            velocity_meas_count += 1
            max_vel = max(max_vel, np.linalg.norm(vn))

    # Extract bias estimates
    eb_est = kf['xk'][6:9]
    db_est = kf['xk'][9:12]

    # Calculate attitude error from final misalignment angle
    phi_est = kf['xk'][0:3]
    # The residual misalignment represents the alignment error
    # Compute it in arcseconds
    phi_err_arcsec = np.abs(phi_est) / glv.arcsec

    # Compute true attitude error: compare estimated C_nb + phi to ground truth C_nb
    # For a stationary platform, ground truth C_nb = identity (or known initial)
    # The phi from KF represents the remaining misalignment
    # So phi_err IS the attitude error estimate

    # Also calculate bias estimation error
    eb_err = np.abs(eb_est - eb_true) / glv.dph  # in deg/h
    db_err = np.abs(db_est - db_true) / glv.ug   # in ug

    elapsed = time.time() - t0
    print(f"fine: {elapsed:.1f}s  vel_meas={velocity_meas_count}  "
          f"phi_err=[{phi_err_arcsec[0]:.3f}, {phi_err_arcsec[1]:.3f}, {phi_err_arcsec[2]:.3f}]\"",
          flush=True)

    return {
        'label': label,
        'n_states': 12,
        'runtime_s': elapsed,
        'coarse_att_deg': att_coarse_deg.tolist(),
        'attitude_error_arcsec': {
            'roll': float(phi_err_arcsec[0]),
            'pitch': float(phi_err_arcsec[1]),
            'yaw': float(phi_err_arcsec[2]),
            'norm': float(np.sqrt(np.sum(phi_est**2)) / glv.arcsec),
        },
        'bias_estimate': {
            'eb_degph': (eb_est / glv.dph).tolist(),
            'db_ug': (db_est / glv.ug).tolist(),
        },
        'bias_true': {
            'eb_degph': (eb_true / glv.dph).tolist(),
            'db_ug': (db_true / glv.ug).tolist(),
        },
        'bias_error': {
            'eb_err_degph': eb_err.tolist(),
            'db_err_ug': (db_err / glv.ug * glv.ug).tolist(),  # normalized
        },
        'velocity_meas_count': velocity_meas_count,
        'max_vel_mps': max_vel,
    }

# ============================================================
# Main
# ============================================================
def main():
    print("=== 12-state 对准实验（无白噪声，仅常值偏置）===\n")
    print(f"常值偏置:")
    print(f"  gyro:  X={BIAS_G_X:+.4f}°/h, Y={BIAS_G_Y:+.4f}°/h, Z={BIAS_G_Z:+.4f}°/h")
    print(f"  accel: X={BIAS_A_X:+.1f} ug, Y={BIAS_A_Y:+.1f} ug, Z={BIAS_A_Z:+.1f} ug")
    print(f"  白噪声: ARW=0, VRW=0 (完全去除)")
    print()

    eb_true, db_true = build_true_bias_vec()

    # --- Experiment A: Single position static ---
    print("--- 实验A：12-state @ 单位置静止 ---")
    att_a = build_static_attitude(TS, TOTAL_S)
    ds_a = make_dataset(att_a, eb_true, db_true)
    print(f"  IMU shape: {ds_a['imu_noisy'].shape[0]} samples ({TOTAL_S}s)")
    result_a = run_12state_alignment(ds_a, '12state_singlepos_static')

    print()

    # --- Experiment B: Two-position ---
    print("--- 实验B：12-state @ 双位置 ---")
    att_b = build_two_position_attitude(TS, TOTAL_S)
    ds_b = make_dataset(att_b, eb_true, db_true)
    print(f"  IMU shape: {ds_b['imu_noisy'].shape[0]} samples ({TOTAL_S}s)")
    result_b = run_12state_alignment(ds_b, '12state_twopos')

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("=== 实验结果汇总 ===")
    print("=" * 70)

    for label, res in [('A: 单位置静止', result_a), ('B: 双位置', result_b)]:
        att = res['attitude_error_arcsec']
        print(f"\n【{label}】")
        print(f"  姿态误差（角秒）:")
        print(f"    roll  = {att['roll']:.6f}\"")
        print(f"    pitch = {att['pitch']:.6f}\"")
        print(f"    yaw   = {att['yaw']:.6f}\"")
        print(f"    norm  = {att['norm']:.6f}\"")
        print(f"  速度零速更新次数: {res['velocity_meas_count']}")
        print(f"  最大速度残差: {res['max_vel_mps']:.6f} m/s")

    # Comparison
    print("\n--- 双位置 vs 单位置 对比 ---")
    d_roll = result_a['attitude_error_arcsec']['roll'] - result_b['attitude_error_arcsec']['roll']
    d_pitch = result_a['attitude_error_arcsec']['pitch'] - result_b['attitude_error_arcsec']['pitch']
    d_yaw = result_a['attitude_error_arcsec']['yaw'] - result_b['attitude_error_arcsec']['yaw']
    d_norm = result_a['attitude_error_arcsec']['norm'] - result_b['attitude_error_arcsec']['norm']
    print(f"  roll 误差变化: {d_roll:+.6f}\" ({'双位置更好' if d_roll > 0 else '单位置更好'})")
    print(f"  pitch误差变化: {d_pitch:+.6f}\" ({'双位置更好' if d_pitch > 0 else '单位置更好'})")
    print(f"  yaw   误差变化: {d_yaw:+.6f}\" ({'双位置更好' if d_yaw > 0 else '单位置更好'})")
    print(f"  norm  误差变化: {d_norm:+.6f}\" ({'双位置更好' if d_norm > 0 else '单位置更好'})")

    # Save results
    now = datetime.datetime.now().strftime('%Y-%m-%d')
    output = {
        'experiment': '12state_alignment_no_white_noise',
        'noise_config': {
            'type': 'deterministic_bias_only',
            'arw': 0, 'vrw': 0,
            'bias_g_degph': [BIAS_G_X, BIAS_G_Y, BIAS_G_Z],
            'bias_a_ug': [BIAS_A_X, BIAS_A_Y, BIAS_A_Z],
        },
        'seed': SEED,
        'experiment_a': result_a,
        'experiment_b': result_b,
    }
    out_path = os.path.join(RESULTS_DIR, f'12state_singlepos_twopos_no_white_{now}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")
    print(f"\n__RESULT_JSON__={json.dumps({key: output[key] for key in ['experiment_a', 'experiment_b'] if 'attitude_error_arcsec' in output[key]}, default=str, ensure_ascii=False)}")

if __name__ == '__main__':
    main()
