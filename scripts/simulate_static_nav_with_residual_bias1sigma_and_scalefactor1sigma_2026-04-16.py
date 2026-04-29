#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
ALIGN_JSON = WORKSPACE / 'psins_method_bench/results/dualpath_pure_scd_custom_noise_0p3_mc50_2026-04-13.json'
CAL_PRECISION_SOURCE = WORKSPACE / '第五章实验_补全图表版_2026-03-31.md'
BIAS_ONLY_SUMMARY = WORKSPACE / 'tmp/static_nav_pure_scd_mean_att0_residual_bias1sigma_2026-04-16/summary.json'
OUT_DIR = WORKSPACE / 'tmp/static_nav_pure_scd_mean_att0_residual_bias1sigma_scalefactor1sigma_2026-04-16'
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAT0_DEG = 39.9819
LON0_DEG = 116.3480
H0_M = 52.0
FS_HZ = 400.0
DT = 1.0 / FS_HZ
T_NAV_S = 12 * 3600
NM_PER_M = 1.0 / 1852.0
WE = 7.292115e-5


def earth_radii_and_g(lat_rad: float, h_m: float):
    a = 6378137.0
    e2 = 6.69437999014e-3
    sin_lat = math.sin(lat_rad)
    den = math.sqrt(1.0 - e2 * sin_lat * sin_lat)
    RN = a / den
    RM = a * (1.0 - e2) / (den ** 3)
    g = 9.7803253359 * (1.0 + 0.00193185265241 * sin_lat * sin_lat) / den - 3.086e-6 * h_m
    return RM, RN, g


def arcsec_to_rad(x_arcsec: float) -> float:
    return math.radians(x_arcsec / 3600.0)


def dph_to_radps(x_dph: float) -> float:
    return math.radians(x_dph) / 3600.0


def ug_to_mps2(x_ug: float) -> float:
    return x_ug * 1e-6 * 9.80665


def load_alignment_mean():
    obj = json.loads(ALIGN_JSON.read_text())
    stats = obj['statistics']
    pitch_mean_arcsec = float(stats['mean_signed_arcsec'][0])
    roll_mean_arcsec = float(stats['mean_signed_arcsec'][1])
    yaw_mean_arcsec = float(stats['mean_signed_arcsec'][2])
    return {
        'roll_arcsec': roll_mean_arcsec,
        'pitch_arcsec': pitch_mean_arcsec,
        'yaw_arcsec': yaw_mean_arcsec,
        'source_json': str(ALIGN_JSON),
    }


def residual_case_from_table55_1sigma():
    """
    Deterministic representative case based on Table 5-5 repeated-calibration 1σ values.
    Sign of residual bias follows the corresponding mean sign in Table 5-5.
    Scale-factor residuals keep positive sign because the table mean values are positive on all three axes.
    """
    return {
        'source': str(CAL_PRECISION_SOURCE),
        'rule': 'bias magnitude = 1σ and sign follows mean sign; scale-factor magnitude = 1σ and sign follows mean sign',
        'gyro_bias_dph': {'x': +0.0021, 'y': -0.0024, 'z': +0.0019},
        'acc_bias_ug': {'x': +6.2, 'y': -5.8, 'z': +7.1},
        'gyro_scale_ppm': {'x': +2.6, 'y': +2.1, 'z': +2.4},
        'acc_scale_ppm': {'x': +3.4, 'y': +2.9, 'z': +3.1},
    }


def simulate_static_nav(roll0_arcsec: float, pitch0_arcsec: float, yaw0_arcsec: float, residual_case):
    """
    Static-navigation surrogate with:
    1) initial attitude from pure-SCD mean alignment result;
    2) residual constant zero bias from calibration 1σ;
    3) residual diagonal scale-factor from calibration 1σ.

    Important physical note for this strict static case:
    - gyro x / z scale-factor acts on the earth-rate components;
    - accel z scale-factor acts on gravity magnitude;
    - gyro y and accel x / y scale-factor are almost unexcited because the corresponding true static inputs are ~0.
    """
    lat0 = math.radians(LAT0_DEG)
    RM, RN, g = earth_radii_and_g(lat0, H0_M)

    roll0_rad = arcsec_to_rad(roll0_arcsec)
    pitch0_rad = arcsec_to_rad(pitch0_arcsec)
    yaw0_rad = arcsec_to_rad(yaw0_arcsec)

    bgx = dph_to_radps(residual_case['gyro_bias_dph']['x'])
    bgy = dph_to_radps(residual_case['gyro_bias_dph']['y'])
    bgz = dph_to_radps(residual_case['gyro_bias_dph']['z'])
    bax = ug_to_mps2(residual_case['acc_bias_ug']['x'])
    bay = ug_to_mps2(residual_case['acc_bias_ug']['y'])
    baz = ug_to_mps2(residual_case['acc_bias_ug']['z'])

    kgx = residual_case['gyro_scale_ppm']['x'] * 1e-6
    kgy = residual_case['gyro_scale_ppm']['y'] * 1e-6
    kgz = residual_case['gyro_scale_ppm']['z'] * 1e-6
    kax = residual_case['acc_scale_ppm']['x'] * 1e-6
    kay = residual_case['acc_scale_ppm']['y'] * 1e-6
    kaz = residual_case['acc_scale_ppm']['z'] * 1e-6

    omega_true_n = WE * math.cos(lat0)
    omega_true_d = -WE * math.sin(lat0)

    Re_n = RM + H0_M
    Re_e = RN + H0_M
    g_eff = g * (1.0 + kaz)
    wn = math.sqrt(g_eff / Re_n)
    we = math.sqrt(g_eff / Re_e)

    # Strictly static aligned case: y-axis earth-rate is zero, body x/y specific force are zero.
    # Therefore only selected scale-factor channels are physically excited in this surrogate.
    gy_force = bgy + kgy * 0.0
    gx_force = bgx + kgx * omega_true_n
    yaw_force = bgz + kgz * omega_true_d
    ax_force = bax + kax * 0.0
    ay_force = bay + kay * 0.0

    t_1hz = np.arange(0.0, T_NAV_S + 1.0, 1.0, dtype=np.float64)
    t_preview = np.arange(0.0, 60.0 + DT, DT, dtype=np.float64)

    # North channel (pitch / north velocity / north position)
    c1_n = pitch0_rad - ax_force / g_eff
    north_m = Re_n * c1_n * (np.cos(wn * t_1hz) - 1.0) + Re_n * gy_force * (np.sin(wn * t_1hz) / wn - t_1hz)
    vn_mps = -Re_n * wn * c1_n * np.sin(wn * t_1hz) + Re_n * gy_force * (np.cos(wn * t_1hz) - 1.0)
    pitch_rad = c1_n * np.cos(wn * t_1hz) + (gy_force / wn) * np.sin(wn * t_1hz) + ax_force / g_eff

    # East channel (roll / east velocity / east position)
    c1_e = roll0_rad + ay_force / g_eff
    east_m = Re_e * c1_e * (1.0 - np.cos(we * t_1hz)) + Re_e * gx_force * (t_1hz - np.sin(we * t_1hz) / we)
    ve_mps = Re_e * we * c1_e * np.sin(we * t_1hz) + Re_e * gx_force * (1.0 - np.cos(we * t_1hz))
    roll_rad = c1_e * np.cos(we * t_1hz) + (gx_force / we) * np.sin(we * t_1hz) - ay_force / g_eff

    yaw_rad = yaw0_rad + yaw_force * t_1hz
    horiz_m = np.sqrt(north_m * north_m + east_m * east_m)
    lat_est_deg = LAT0_DEG + np.degrees(north_m / Re_n)
    lon_est_deg = LON0_DEG + np.degrees(east_m / (Re_e * math.cos(lat0)))

    north_preview = Re_n * c1_n * (np.cos(wn * t_preview) - 1.0) + Re_n * gy_force * (np.sin(wn * t_preview) / wn - t_preview)
    east_preview = Re_e * c1_e * (1.0 - np.cos(we * t_preview)) + Re_e * gx_force * (t_preview - np.sin(we * t_preview) / we)
    horiz_preview = np.sqrt(north_preview * north_preview + east_preview * east_preview)
    pitch_preview_arcsec = np.degrees(c1_n * np.cos(wn * t_preview) + (gy_force / wn) * np.sin(wn * t_preview) + ax_force / g_eff) * 3600.0
    roll_preview_arcsec = np.degrees(c1_e * np.cos(we * t_preview) + (gx_force / we) * np.sin(we * t_preview) - ay_force / g_eff) * 3600.0
    yaw_preview_arcsec = np.degrees(yaw0_rad + yaw_force * t_preview) * 3600.0

    idx_max = int(np.argmax(horiz_m))
    metrics = {
        'max_north_err_m': float(np.max(np.abs(north_m))),
        'max_east_err_m': float(np.max(np.abs(east_m))),
        'max_horizontal_err_m': float(np.max(horiz_m)),
        'max_horizontal_err_nm': float(np.max(horiz_m) * NM_PER_M),
        'end_north_err_m': float(north_m[-1]),
        'end_east_err_m': float(east_m[-1]),
        'end_horizontal_err_m': float(horiz_m[-1]),
        'end_horizontal_err_nm': float(horiz_m[-1] * NM_PER_M),
        'equivalent_end_divergence_rate_nm_per_h': float(horiz_m[-1] * NM_PER_M / (T_NAV_S / 3600.0)),
        'time_of_max_horizontal_err_s': float(t_1hz[idx_max]),
        'time_of_max_horizontal_err_h': float(t_1hz[idx_max] / 3600.0),
        'north_err_at_max_m': float(north_m[idx_max]),
        'east_err_at_max_m': float(east_m[idx_max]),
        'yaw_end_arcsec': float(np.degrees(yaw_rad[-1]) * 3600.0),
        'yaw_change_arcsec': float(np.degrees(yaw_rad[-1] - yaw0_rad) * 3600.0),
    }

    compare_to_bias_only = None
    if BIAS_ONLY_SUMMARY.exists():
        ref = json.loads(BIAS_ONLY_SUMMARY.read_text())
        rm = ref['metrics']
        compare_to_bias_only = {
            'bias_only_max_horizontal_nm': float(rm['max_horizontal_err_nm']),
            'bias_only_end_divergence_nm_per_h': float(rm['equivalent_end_divergence_rate_nm_per_h']),
            'delta_max_horizontal_nm': float(metrics['max_horizontal_err_nm'] - rm['max_horizontal_err_nm']),
            'delta_end_divergence_nm_per_h': float(metrics['equivalent_end_divergence_rate_nm_per_h'] - rm['equivalent_end_divergence_rate_nm_per_h']),
            'delta_yaw_change_arcsec': float(metrics['yaw_change_arcsec'] - rm['yaw_change_arcsec']),
        }

    static_imu_profile = {
        'gyro_true_ned_rad_s': [omega_true_n, 0.0, omega_true_d],
        'gyro_residual_bias_rad_s': [bgx, bgy, bgz],
        'gyro_scale_factor_residual': {'x': kgx, 'y': kgy, 'z': kgz},
        'gyro_measured_body_rad_s_for_static_aligned_case': [
            (1.0 + kgx) * omega_true_n + bgx,
            (1.0 + kgy) * 0.0 + bgy,
            (1.0 + kgz) * omega_true_d + bgz,
        ],
        'acc_true_body_mps2': [0.0, 0.0, -g],
        'acc_residual_bias_mps2': [bax, bay, baz],
        'acc_scale_factor_residual': {'x': kax, 'y': kay, 'z': kaz},
        'acc_measured_body_mps2_for_static_aligned_case': [
            (1.0 + kax) * 0.0 + bax,
            (1.0 + kay) * 0.0 + bay,
            (1.0 + kaz) * (-g) + baz,
        ],
        'excitation_note': 'Under strict static aligned input, gyro y and accel x/y scale-factor channels are almost not excited; gyro x/z and accel z are the dominant active scale-factor channels.',
    }

    summary = {
        'model': 'alignment_plus_residual_bias_plus_scalefactor_forced_schuler_static_nav',
        'assumptions': [
            'static base',
            'initial velocity zero',
            'initial attitude from pure-SCD MC50 mean',
            'residual constant zero bias injected using Table 5-5 1σ calibration precision',
            'residual diagonal scale factor injected using Table 5-5 1σ calibration precision',
            'north/east channels use forced Schuler response',
            'vertical channel held out of model',
        ],
        'site': {
            'name': 'Beihang University (approx. Xueyuan Road campus), Haidian, Beijing',
            'lat_deg': LAT0_DEG,
            'lon_deg': LON0_DEG,
            'h_m': H0_M,
        },
        'simulation': {
            'sample_rate_hz': FS_HZ,
            'dt_s': DT,
            'nav_duration_s': T_NAV_S,
            'nav_duration_h': T_NAV_S / 3600.0,
            'output_rate_hz': 1.0,
            'total_samples_400hz': int(T_NAV_S * FS_HZ),
        },
        'initial_conditions': {
            'att0_mean_arcsec': {'roll': roll0_arcsec, 'pitch': pitch0_arcsec, 'yaw': yaw0_arcsec},
            'vn0_mps': [0.0, 0.0, 0.0],
            'pos0': {'lat_deg': LAT0_DEG, 'lon_deg': LON0_DEG, 'h_m': H0_M},
        },
        'residual_case': residual_case,
        'effective_forcing_terms': {
            'gyro_y_force_rad_s': gy_force,
            'gyro_x_force_rad_s': gx_force,
            'yaw_force_rad_s': yaw_force,
            'acc_x_force_mps2': ax_force,
            'acc_y_force_mps2': ay_force,
            'effective_g_mps2': g_eff,
        },
        'earth_constants': {
            'RM_m': Re_n,
            'RN_m': Re_e,
            'g_mps2': g,
            'effective_g_mps2': g_eff,
            'schuler_wn_rad_s': wn,
            'schuler_we_rad_s': we,
            'schuler_period_n_s': 2.0 * math.pi / wn,
            'schuler_period_e_s': 2.0 * math.pi / we,
        },
        'static_imu_profile': static_imu_profile,
        'metrics': metrics,
        'compare_to_bias_only_version': compare_to_bias_only,
    }

    trajectory = {
        't_s': t_1hz,
        'lat_est_deg': lat_est_deg,
        'lon_est_deg': lon_est_deg,
        'north_err_m': north_m,
        'east_err_m': east_m,
        'horizontal_err_m': horiz_m,
        'vn_mps': vn_mps,
        've_mps': ve_mps,
        'roll_err_arcsec': np.degrees(roll_rad) * 3600.0,
        'pitch_err_arcsec': np.degrees(pitch_rad) * 3600.0,
        'yaw_err_arcsec': np.degrees(yaw_rad) * 3600.0,
    }
    preview = {
        't_s': t_preview,
        'north_err_m': north_preview,
        'east_err_m': east_preview,
        'horizontal_err_m': horiz_preview,
        'roll_err_arcsec': roll_preview_arcsec,
        'pitch_err_arcsec': pitch_preview_arcsec,
        'yaw_err_arcsec': yaw_preview_arcsec,
    }
    return {'summary': summary, 'trajectory': trajectory, 'preview': preview}


def write_csv(path: Path, rows, header):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    mean_att = load_alignment_mean()
    residual_case = residual_case_from_table55_1sigma()
    result = simulate_static_nav(mean_att['roll_arcsec'], mean_att['pitch_arcsec'], mean_att['yaw_arcsec'], residual_case)
    summary = result['summary']
    traj = result['trajectory']
    preview = result['preview']

    summary_path = OUT_DIR / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    imu_profile_path = OUT_DIR / 'static_imu_profile.json'
    imu_profile_path.write_text(json.dumps(summary['static_imu_profile'], ensure_ascii=False, indent=2))

    trajectory_csv = OUT_DIR / 'static_nav_residual_bias1sigma_scalefactor1sigma_1hz.csv'
    write_csv(
        trajectory_csv,
        zip(
            traj['t_s'], traj['lat_est_deg'], traj['lon_est_deg'], traj['north_err_m'], traj['east_err_m'],
            traj['horizontal_err_m'], traj['vn_mps'], traj['ve_mps'],
            traj['roll_err_arcsec'], traj['pitch_err_arcsec'], traj['yaw_err_arcsec'],
        ),
        ['t_s', 'lat_est_deg', 'lon_est_deg', 'north_err_m', 'east_err_m', 'horizontal_err_m', 'vn_mps', 've_mps', 'roll_err_arcsec', 'pitch_err_arcsec', 'yaw_err_arcsec'],
    )

    preview_csv = OUT_DIR / 'static_nav_residual_bias1sigma_scalefactor1sigma_preview_400hz_first60s.csv'
    write_csv(
        preview_csv,
        zip(
            preview['t_s'], preview['north_err_m'], preview['east_err_m'], preview['horizontal_err_m'],
            preview['roll_err_arcsec'], preview['pitch_err_arcsec'], preview['yaw_err_arcsec'],
        ),
        ['t_s', 'north_err_m', 'east_err_m', 'horizontal_err_m', 'roll_err_arcsec', 'pitch_err_arcsec', 'yaw_err_arcsec'],
    )

    report_path = OUT_DIR / 'report.md'
    m = summary['metrics']
    rc = summary['residual_case']
    cmp = summary['compare_to_bias_only_version']
    cmp_txt = ''
    if cmp:
        cmp_txt = f"""
## 与上一版“只加残余零偏”结果的差异

- bias-only 最大水平误差：`{cmp['bias_only_max_horizontal_nm']:.6f} nm`
- 当前版本最大水平误差：`{m['max_horizontal_err_nm']:.6f} nm`
- 增量：`{cmp['delta_max_horizontal_nm']:.6f} nm`
- bias-only 发散速度：`{cmp['bias_only_end_divergence_nm_per_h']:.6f} nm/h`
- 当前版本发散速度：`{m['equivalent_end_divergence_rate_nm_per_h']:.6f} nm/h`
- 增量：`{cmp['delta_end_divergence_nm_per_h']:.6f} nm/h`
- yaw 变化量增量：`{cmp['delta_yaw_change_arcsec']:.6f}"`

从这个差异可以直接看出：在**静态**条件下，加入 residual scale factor 后结果会继续变差，但幅度明显小于加入 residual bias 时带来的跃迁。这说明在严格静态导航工况中，**残余零偏仍是主导项，残余 scale factor 更像次级修正项**。
"""
    report = f"""# 静态导航仿真（pure-SCD 均值 att0 + 表5-5 的 1σ 残余零偏 + 1σ residual scale factor）

## 设置

- 位置：北京海淀区北航学院路校区近似坐标 `39.9819°N, 116.3480°E, h=52.0 m`
- 初始速度：`[0, 0, 0] m/s`
- 采样频率：`400.0 Hz`
- 导航总时长：`12.0 h`
- att0 取自 `pure-SCD @ 0.3×` 的 MC50 均值：
  - roll = `{mean_att['roll_arcsec']:.6f}"`
  - pitch = `{mean_att['pitch_arcsec']:.6f}"`
  - yaw = `{mean_att['yaw_arcsec']:.6f}"`

## 残余误差设定（按表 5-5 的 1σ 标定精度）

### 残余零偏
- 光纤陀螺 / dph：x=`{rc['gyro_bias_dph']['x']}`, y=`{rc['gyro_bias_dph']['y']}`, z=`{rc['gyro_bias_dph']['z']}`
- 加速度计 / ug：x=`{rc['acc_bias_ug']['x']}`, y=`{rc['acc_bias_ug']['y']}`, z=`{rc['acc_bias_ug']['z']}`

### 残余 scale factor
- 光纤陀螺 / ppm：x=`{rc['gyro_scale_ppm']['x']}`, y=`{rc['gyro_scale_ppm']['y']}`, z=`{rc['gyro_scale_ppm']['z']}`
- 加速度计 / ppm：x=`{rc['acc_scale_ppm']['x']}`, y=`{rc['acc_scale_ppm']['y']}`, z=`{rc['acc_scale_ppm']['z']}`

## 模型说明

- 这版在“残余零偏 1σ”基础上进一步加入了“residual scale factor 1σ”；
- 严格静态条件下，并非所有 scale factor 通道都被充分激励：
  - **陀螺 x/z scale factor** 会作用到地球自转分量；
  - **加速度计 z scale factor** 会作用到重力项；
  - **陀螺 y 与加速度计 x/y scale factor** 在严格静止对准条件下几乎不被激励；
- 因而这版结果很适合用来说明：在静态导航场景下，residual scale factor 会有影响，但通常不是主导项。

## 结果摘要

- 最大北向误差：`{m['max_north_err_m']:.6f} m`
- 最大东向误差：`{m['max_east_err_m']:.6f} m`
- 最大水平误差：`{m['max_horizontal_err_m']:.6f} m` = `{m['max_horizontal_err_nm']:.6f} nm`
- 末时刻水平误差：`{m['end_horizontal_err_m']:.6f} m` = `{m['end_horizontal_err_nm']:.6f} nm`
- 等效末值发散速度：`{m['equivalent_end_divergence_rate_nm_per_h']:.6f} nm/h`
- 最大水平误差出现时刻：`{m['time_of_max_horizontal_err_h']:.6f} h`
- 12 h 后 yaw：`{m['yaw_end_arcsec']:.6f}"`（相对初值变化 `{m['yaw_change_arcsec']:.6f}"`）
{cmp_txt}
## 文件

- 1 Hz 全程轨迹：`{trajectory_csv}`
- 400 Hz 前 60 s 预览：`{preview_csv}`
- 摘要：`{summary_path}`
- 静止 IMU 常值输入：`{imu_profile_path}`
"""
    report_path.write_text(report)

    print(json.dumps({
        'ok': True,
        'out_dir': str(OUT_DIR),
        'summary_json': str(summary_path),
        'trajectory_csv': str(trajectory_csv),
        'preview_csv': str(preview_csv),
        'report_md': str(report_path),
        'key_metrics': summary['metrics'],
        'compare_to_bias_only_version': summary['compare_to_bias_only_version'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
