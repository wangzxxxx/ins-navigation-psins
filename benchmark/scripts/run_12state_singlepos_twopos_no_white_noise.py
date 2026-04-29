"""
12-state 对准实验（无白噪声）：
- 实验A：12-state @ 单位置静止对准（ARW=VRW=0，仅常值偏置）
- 实验B：12-state @ 双位置对准（ARW=VRW=0，仅常值偏置）
"""
from __future__ import annotations
import sys, os, types, time, json, math, datetime

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

mod = __import__('psins_py', fromlist=[''])
mod = __import__('test_calibration_markov_pruned', globals(), locals(), [], 0)
np = mod.np

# ============================================================
# 参数设置
# ============================================================
SEED = 42
TS = 0.01
TOTAL_S = 1200.0  # 1200s total
BASE_BI_G = 0.002   # deg/h (deterministic bias)
BASE_BI_A = 5.0     # ug (deterministic bias)
# ARW / VRW set to ZERO (no white noise)

# ============================================================
# Dataset generation: IMU only with deterministic bias, NO white noise
# ============================================================
def make_dataset_no_white_noise(paras_raw, ts=TS, duration_s=TOTAL_S):
    """Build IMU dataset with deterministic bias errors only (no white noise)."""
    att0 = np.array([0.0, 0.0, 0.0])  # level
    pos0 = mod.posset(34.0, 0.0, 0.0)

    if isinstance(paras_raw, list):
        # Rotation trajectory: [[idx, ax0, ax1, ax2, angle_deg, rot_time_s, pre_static_s, post_static_s], ...]
        paras = []
        for row in paras_raw:
            idx = int(row[0])
            paras.append({
                'axis': [int(row[1]), int(row[2]), int(row[3])],
                'angle_deg': float(row[4]) * mod.glv.deg,
                'rotation_time_s': float(row[5]),
                'pre_static_s': float(row[6]),
                'post_static_s': float(row[7]),
            })
        q_total = duration_s / ts
        q_att = np.zeros((int(q_total), 4))
        q_att[:, 3] = np.arange(1, int(q_total)+1) * ts
        current_time = 0.0
        row_times = []
        for p in paras:
            t_pre = int(p['pre_static_s'] / ts)
            t_rot = int(p['rotation_time_s'] / ts)
            t_post = int(p['post_static_s'] / ts)
            # Pre-static
            n = min(t_pre, int(q_total) - int(current_time / ts))
            if n > 0:
                start_idx = int(current_time / ts)
                q_att[start_idx:start_idx+n, 0:3] = att0 * mod.glv.deg
            current_time += t_pre * ts
            row_times.append((current_time, t_rot, t_post, p))
            # Rotation
            n = min(t_rot, int(q_total) - int(current_time / ts))
            if n > 0:
                start_idx = int(current_time / ts)
                w = p['angle_deg'] / (t_rot * ts) * ts  # per-sample
                ax = p['axis']
                for i in range(n):
                    a = w * (i+1)
                    q_att[start_idx+i, 0:3] = att0 * mod.glv.deg + np.array(ax) * a
            current_time += t_rot * ts
            # Post-static
            n = min(t_post, int(q_total) - int(current_time / ts))
            if n > 0:
                start_idx = int(current_time / ts)
                final_angle = p['angle_deg']
                a_final = np.array(p['axis']) * final_angle
                q_att[start_idx:start_idx+n, 0:3] = att0 * mod.glv.deg + a_final
            current_time += t_post * ts

        imu_clean, *_ = mod.avp2imu(q_att, pos0)
    else:
        # Static: single attitude for duration_s
        static_att = paras_raw
        n = int(duration_s / ts) + 1
        q_att = np.zeros((n, 4))
        q_att[:, 0:3] = static_att
        q_att[:, 3] = np.arange(1, n+1) * ts
        imu_clean, *_ = mod.avp2imu(q_att, pos0)

    # IMU errors: deterministic bias only, NO white noise
    clbt_truth = mod.get_default_clbt()
    imu_biased = mod.imuclbt(imu_clean, clbt_truth)

    # Add deterministic bias errors manually (since imuadderr_full adds white noise)
    bi_g_rad = BASE_BI_G * mod.glv.dph  # bias in rad/s
    bi_a_m = BASE_BI_A * mod.glv.ug      # bias in m/s^2
    n_samp = imu_biased.shape[0]
    imu_biased[:, 0] += bi_g_rad[0]  # wx
    imu_biased[:, 1] += bi_g_rad[1]  # wy
    imu_biased[:, 2] += bi_g_rad[2]  # wz
    imu_biased[:, 3] += bi_a_m[0]    # fx
    imu_biased[:, 4] += bi_a_m[1]    # fy
    imu_biased[:, 5] += bi_a_m[2]    # fz

    cfg = {
        'arw_dpsh': 0.0,
        'vrw_ugpsHz': 0.0,
        'bi_g_dph': BASE_BI_G,
        'bi_a_ug': BASE_BI_A,
        'seed': SEED,
        'noise_type': 'deterministic_bias_only',
    }
    return {
        'ts': ts,
        'pos0': pos0,
        'imu_noisy': imu_biased,
        'imu_clean': imu_clean,
        'noise_config': cfg,
        'bias_g_dph': BASE_BI_G,
        'bias_a_ug': BASE_BI_A,
        'clbt_truth': clbt_truth,
    }

# ============================================================
# Trajectory definitions
# ============================================================
def single_position_static_paras(duration_s=1200.0):
    """Single position: static, no rotation."""
    return (duration_s, np.array([0.0, 0.0, 0.0]))

def two_position_paras(duration_s=1200.0):
    """Two-position: 600s static pos1, 180s rotation 180° around Z, 420s static pos2."""
    return [
        [1, 0, 0, 0, 0.0, 600.0, 0.0, 0.0],        # pos1: static 600s
        [2, 0, 0, 1, 180.0, 180.0, 0.0, 0.0],        # rotate 180° Z-axis over 180s
        [3, 0, 0, 0, 0.0, 420.0, 0.0, 0.0],          # pos2: static 420s
    ]

# ============================================================
# Run 12-state calibration
# ============================================================
def run_12state(ds, label):
    """Run 12-state KF calibration (phi/dv/eb/db)."""
    print(f"  {label}: 12-state KF...", end=" ", flush=True)
    t0 = time.time()
    result = mod.run_calibration(ds['imu_noisy'], ds['pos0'], ds['ts'], n_states=12, label=label)
    elapsed = time.time() - t0
    print(f"done {elapsed:.1f}s")

    clbt = result[0]
    ts_val = ds['ts']
    tlen = ds['imu_noisy'].shape[0] * ts_val

    # Extract bias estimates: eb (gyro bias indices) and db (accel bias indices)
    # For 12-state: states = phi(3) + dv(3) + eb(3) + db(3) = 12
    # eb indices typically at 6:9, db at 9:12 in the state vector
    eb_est = clbt['eb'] if 'eb' in clbt else clbt.get('x', np.zeros(12))[6:9]
    db_est = clbt['db'] if 'db' in clbt else clbt.get('x', np.zeros(12))[9:12]

    clbt_truth = ds['clbt_truth']
    eb_true = clbt_truth['eb']
    db_true = clbt_truth['db']

    # True bias values in physical units
    bg_true_deg_h = float(BASE_BI_G) * 3  # simplified: all axes same
    ba_true_ug = float(BASE_BI_A) * 3

    param_errors = {}
    axes = ['x', 'y', 'z']
    for i, ax in enumerate(axes):
        name_g = f'eb_{ax}'
        true_g = float(eb_true[i])
        est_g = float(eb_est[i])
        ae = abs(true_g - est_g)
        pe = ae / abs(true_g) * 100.0 if abs(true_g) > 1e-15 else 0.0
        param_errors[name_g] = {'true': true_g, 'est': est_g, 'abs_error': ae, 'pct_error': pe}

        name_a = f'db_{ax}'
        true_a = float(db_true[i])
        est_a = float(db_est[i])
        ae = abs(true_a - est_a)
        pe_a = ae / abs(true_a) * 100.0 if abs(true_a) > 1e-15 else 0.0
        param_errors[name_a] = {'true': true_a, 'est': est_a, 'abs_error': ae, 'pct_error': pe_a}

    # Overall attitude error (from clbt result if available)
    att_err_rad = clbt.get('att_err', np.zeros(3))
    att_err_arcsec = np.abs(att_err_rad) / mod.glv.arcsec
    
    overall_roll = float(att_err_arcsec[0])
    overall_pitch = float(att_err_arcsec[1])
    overall_yaw = float(att_err_arcsec[2])
    overall_norm = float(np.sqrt(np.sum(att_err_arcsec**2)))

    pcts = [param_errors[n]['pct_error'] for n in param_errors]
    arr = np.asarray(pcts, dtype=float)
    overall = {
        'mean_pct_error': float(np.mean(arr)),
        'median_pct_error': float(np.median(arr)),
        'max_pct_error': float(np.max(arr)),
    }

    return {
        'label': label,
        'n_states': 12,
        'param_errors': param_errors,
        'attitude_error_arcsec': {
            'roll': overall_roll,
            'pitch': overall_pitch,
            'yaw': overall_yaw,
            'norm': overall_norm,
        },
        'overall': overall,
        'runtime_s': elapsed,
    }

# ============================================================
# Main
# ============================================================
def main():
    print("=== 12-state 对准实验（无白噪声，仅常值偏置）===\n")
    print(f"偏置: gyro={BASE_BI_G} d/h, accel={BASE_BI_A} ug")
    print("白噪声: ARW=0, VRW=0 (完全去除)")
    print()

    # Dataset A: single position static
    print("--- 实验A：12-state @ 单位置静止 ---")
    ds_a = make_dataset_no_white_noise(single_position_static_paras(1200.0))
    print(f"  IMU shape: {ds_a['imu_noisy'].shape}")
    result_a = run_12state(ds_a, '12state_singlepos_static')

    print()

    # Dataset B: two-position
    print("--- 实验B：12-state @ 双位置 ---")
    ds_b = make_dataset_no_white_noise(two_position_paras(1200.0))
    print(f"  IMU shape: {ds_b['imu_noisy'].shape}")
    result_b = run_12state(ds_b, '12state_twopos')

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "="*70)
    print("=== 实验结果汇总 ===")
    print("="*70)

    for label, res in [('A: 单位置静止', result_a), ('B: 双位置', result_b)]:
        print(f"\n【{label}】")
        att = res['attitude_error_arcsec']
        print(f"  姿态误差（角秒）:")
        print(f"    roll  = {att['roll']:.6f}\"")
        print(f"    pitch = {att['pitch']:.6f}\"")
        print(f"    yaw   = {att['yaw']:.6f}\"")
        print(f"    norm  = {att['norm']:.6f}\"")
        print(f"  参数误差（mean/median/max pct）:")
        print(f"    mean   = {res['overall']['mean_pct_error']:.6f}%")
        print(f"    median = {res['overall']['median_pct_error']:.6f}%")
        print(f"    max    = {res['overall']['max_pct_error']:.6f}%")

    # Comparison
    print("\n--- 对比 ---")
    d_yaw = result_a['attitude_error_arcsec']['yaw'] - result_b['attitude_error_arcsec']['yaw']
    d_norm = result_a['attitude_error_arcsec']['norm'] - result_b['attitude_error_arcsec']['norm']
    d_mean = result_a['overall']['mean_pct_error'] - result_b['overall']['mean_pct_error']
    print(f"  yaw 误差改善：{d_yaw:+.6f}\" ({'双位置更好' if d_yaw > 0 else '单位置更好'})")
    print(f"  norm 改善：{d_norm:+.6f}\" ({'双位置更好' if d_norm > 0 else '单位置更好'})")
    print(f"  mean pct 改善：{d_mean:+.6f}% ({'双位置更好' if d_mean > 0 else '单位置更好'})")

    # Save results
    now = datetime.datetime.now().strftime('%Y-%m-%d')
    output = {
        'experiment': '12state_aligment_no_white_noise',
        'noise_config': {'type': 'deterministic_bias_only', 'arw': 0, 'vrw': 0, 'bi_g_dph': BASE_BI_G, 'bi_a_ug': BASE_BI_A},
        'seed': SEED,
        'experiment_a': {
            'label': '12-state @ 单位置静止',
            **result_a,
        },
        'experiment_b': {
            'label': '12-state @ 双位置',
            **result_b,
        },
    }
    out_path = os.path.join(RESULTS_DIR, f'12state_singlepos_twopos_no_white_noise_{now}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果保存: {out_path}")
    print(f"\n__RESULT_JSON__={json.dumps({'A_mean': result_a['overall']['mean_pct_error'], 'A_yaw': result_a['attitude_error_arcsec']['yaw'], 'B_mean': result_b['overall']['mean_pct_error'], 'B_yaw': result_b['attitude_error_arcsec']['yaw']}, ensure_ascii=False)}")

if __name__ == '__main__':
    main()
