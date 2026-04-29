#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
ALIGN_JSON = WORKSPACE / 'psins_method_bench/results/dualpath_pure_scd_custom_noise_0p3_mc50_2026-04-13.json'
G4_JSON = WORKSPACE / 'psins_method_bench/results/R61_42state_gm1_round61_h_scd_state20_microtight_commit_ch3corrected_symmetric20_att0zero_1200s_shared_custom_arw0p0005_vrw0p5_big0p0007_bia5_tg300_ta300_seed42_param_errors.json'
OUT_DIR = WORKSPACE / 'tmp/static_nav_pure_scd_mean_att0_g4_single_residual_bias_seed42_2026-04-16'
OUT_DIR.mkdir(parents=True, exist_ok=True)

LAT0_DEG = 39.9819
LON0_DEG = 116.3480
H0_M = 52.0
FS_HZ = 400.0
DT = 1.0 / FS_HZ
T_NAV_S = 12 * 3600
NM_PER_M = 1.0 / 1852.0


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


def signed_g4_single_bias_residuals():
    obj = json.loads(G4_JSON.read_text())
    pe = obj['param_errors']
    # signed residual = est - true
    eb = {k[-1]: float(pe[k]['est'] - pe[k]['true']) for k in ['eb_x', 'eb_y', 'eb_z']}
    db = {k[-1]: float(pe[k]['est'] - pe[k]['true']) for k in ['db_x', 'db_y', 'db_z']}

    # Convert to engineering units for report readability
    eb_dph = {axis: math.degrees(val) * 3600.0 for axis, val in eb.items()}
    db_ug = {axis: val / 9.80665 * 1e6 for axis, val in db.items()}

    return {
        'source': str(G4_JSON),
        'description': 'signed single-run residual bias from G4 / Round61 on corrected symmetric20 under custom noise seed=42',
        'custom_noise': {
            'arw_dpsh': 0.0005,
            'vrw_ugpsHz': 0.5,
            'bi_g_dph': 0.0007,
            'bi_a_ug': 5.0,
            'tau_g_s': 300.0,
            'tau_a_s': 300.0,
            'seed': 42,
        },
        'gyro_bias_rad_s': eb,
        'acc_bias_mps2': db,
        'gyro_bias_dph': eb_dph,
        'acc_bias_ug': db_ug,
    }


def simulate_static_nav_with_residual_bias(roll0_arcsec: float, pitch0_arcsec: float, yaw0_arcsec: float, residual_bias):
    lat0 = math.radians(LAT0_DEG)
    RM, RN, g = earth_radii_and_g(lat0, H0_M)

    roll0_rad = arcsec_to_rad(roll0_arcsec)
    pitch0_rad = arcsec_to_rad(pitch0_arcsec)
    yaw0_rad = arcsec_to_rad(yaw0_arcsec)

    bgx = float(residual_bias['gyro_bias_rad_s']['x'])
    bgy = float(residual_bias['gyro_bias_rad_s']['y'])
    bgz = float(residual_bias['gyro_bias_rad_s']['z'])
    bax = float(residual_bias['acc_bias_mps2']['x'])
    bay = float(residual_bias['acc_bias_mps2']['y'])

    t_1hz = np.arange(0.0, T_NAV_S + 1.0, 1.0, dtype=np.float64)
    t_preview = np.arange(0.0, 60.0 + DT, DT, dtype=np.float64)

    Re_n = RM + H0_M
    Re_e = RN + H0_M
    wn = math.sqrt(g / Re_n)
    we = math.sqrt(g / Re_e)

    c1_n = pitch0_rad - bax / g
    north_m = Re_n * c1_n * (np.cos(wn * t_1hz) - 1.0) + Re_n * bgy * (np.sin(wn * t_1hz) / wn - t_1hz)
    vn_mps = -Re_n * wn * c1_n * np.sin(wn * t_1hz) + Re_n * bgy * (np.cos(wn * t_1hz) - 1.0)
    pitch_rad = c1_n * np.cos(wn * t_1hz) + (bgy / wn) * np.sin(wn * t_1hz) + bax / g

    c1_e = roll0_rad + bay / g
    east_m = Re_e * c1_e * (1.0 - np.cos(we * t_1hz)) + Re_e * bgx * (t_1hz - np.sin(we * t_1hz) / we)
    ve_mps = Re_e * we * c1_e * np.sin(we * t_1hz) + Re_e * bgx * (1.0 - np.cos(we * t_1hz))
    roll_rad = c1_e * np.cos(we * t_1hz) + (bgx / we) * np.sin(we * t_1hz) - bay / g

    yaw_rad = yaw0_rad + bgz * t_1hz
    horiz_m = np.sqrt(north_m * north_m + east_m * east_m)
    lat_est_deg = LAT0_DEG + np.degrees(north_m / Re_n)
    lon_est_deg = LON0_DEG + np.degrees(east_m / (Re_e * math.cos(lat0)))

    north_preview = Re_n * c1_n * (np.cos(wn * t_preview) - 1.0) + Re_n * bgy * (np.sin(wn * t_preview) / wn - t_preview)
    east_preview = Re_e * c1_e * (1.0 - np.cos(we * t_preview)) + Re_e * bgx * (t_preview - np.sin(we * t_preview) / we)
    horiz_preview = np.sqrt(north_preview * north_preview + east_preview * east_preview)
    pitch_preview_arcsec = np.degrees(c1_n * np.cos(wn * t_preview) + (bgy / wn) * np.sin(wn * t_preview) + bax / g) * 3600.0
    roll_preview_arcsec = np.degrees(c1_e * np.cos(we * t_preview) + (bgx / we) * np.sin(we * t_preview) - bay / g) * 3600.0
    yaw_preview_arcsec = np.degrees(yaw0_rad + bgz * t_preview) * 3600.0

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

    static_imu_profile = {
        'note': 'Static aligned IMU profile with signed single-run G4 residual bias injected; no extra white noise / GM / scale-factor residual added.',
        'gyro_residual_bias_rad_s': [bgx, bgy, bgz],
        'gyro_residual_bias_dph': [residual_bias['gyro_bias_dph']['x'], residual_bias['gyro_bias_dph']['y'], residual_bias['gyro_bias_dph']['z']],
        'acc_residual_bias_mps2': [bax, bay, float(residual_bias['acc_bias_mps2']['z'])],
        'acc_residual_bias_ug': [residual_bias['acc_bias_ug']['x'], residual_bias['acc_bias_ug']['y'], residual_bias['acc_bias_ug']['z']],
    }

    summary = {
        'model': 'alignment_plus_single_run_g4_signed_residual_bias_forced_schuler_static_nav',
        'assumptions': [
            'static base',
            'initial velocity zero',
            'initial attitude from pure-SCD MC50 mean',
            'residual constant bias uses signed single-run G4 residual from custom-noise seed=42',
            'no extra white noise / GM / scale-factor residual injected',
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
        'residual_case': residual_bias,
        'earth_constants': {
            'RM_m': Re_n,
            'RN_m': Re_e,
            'g_mps2': g,
            'schuler_wn_rad_s': wn,
            'schuler_we_rad_s': we,
            'schuler_period_n_s': 2.0 * math.pi / wn,
            'schuler_period_e_s': 2.0 * math.pi / we,
        },
        'static_imu_profile': static_imu_profile,
        'metrics': metrics,
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
    residual_bias = signed_g4_single_bias_residuals()
    result = simulate_static_nav_with_residual_bias(mean_att['roll_arcsec'], mean_att['pitch_arcsec'], mean_att['yaw_arcsec'], residual_bias)
    summary = result['summary']
    traj = result['trajectory']
    preview = result['preview']

    summary_path = OUT_DIR / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    imu_profile_path = OUT_DIR / 'static_imu_profile.json'
    imu_profile_path.write_text(json.dumps(summary['static_imu_profile'], ensure_ascii=False, indent=2))

    trajectory_csv = OUT_DIR / 'static_nav_g4_single_residual_bias_1hz.csv'
    write_csv(
        trajectory_csv,
        zip(
            traj['t_s'], traj['lat_est_deg'], traj['lon_est_deg'], traj['north_err_m'], traj['east_err_m'],
            traj['horizontal_err_m'], traj['vn_mps'], traj['ve_mps'],
            traj['roll_err_arcsec'], traj['pitch_err_arcsec'], traj['yaw_err_arcsec'],
        ),
        ['t_s', 'lat_est_deg', 'lon_est_deg', 'north_err_m', 'east_err_m', 'horizontal_err_m', 'vn_mps', 've_mps', 'roll_err_arcsec', 'pitch_err_arcsec', 'yaw_err_arcsec'],
    )

    preview_csv = OUT_DIR / 'static_nav_g4_single_residual_bias_preview_400hz_first60s.csv'
    write_csv(
        preview_csv,
        zip(
            preview['t_s'], preview['north_err_m'], preview['east_err_m'], preview['horizontal_err_m'],
            preview['roll_err_arcsec'], preview['pitch_err_arcsec'], preview['yaw_err_arcsec'],
        ),
        ['t_s', 'north_err_m', 'east_err_m', 'horizontal_err_m', 'roll_err_arcsec', 'pitch_err_arcsec', 'yaw_err_arcsec'],
    )

    m = summary['metrics']
    rb = summary['residual_case']
    report = f"""# 静态导航仿真（pure-SCD 均值 att0 + G4 单次 signed residual bias）

## 设置

- 位置：北京海淀区北航学院路校区近似坐标 `39.9819°N, 116.3480°E, h=52.0 m`
- 初始速度：`[0, 0, 0] m/s`
- 采样频率：`400.0 Hz`
- 导航总时长：`12.0 h`
- att0 取自 `pure-SCD @ 0.3×` 的 MC50 均值：
  - roll = `{mean_att['roll_arcsec']:.6f}"`
  - pitch = `{mean_att['pitch_arcsec']:.6f}"`
  - yaw = `{mean_att['yaw_arcsec']:.6f}"`

## 代入的 G4 单次真实残余零偏（custom-noise / seed=42）

### 光纤陀螺 / dph（signed residual = est - true）
- x = `{rb['gyro_bias_dph']['x']:.6f}`
- y = `{rb['gyro_bias_dph']['y']:.6f}`
- z = `{rb['gyro_bias_dph']['z']:.6f}`

### 加速度计 / ug（signed residual = est - true）
- x = `{rb['acc_bias_ug']['x']:.6f}`
- y = `{rb['acc_bias_ug']['y']:.6f}`
- z = `{rb['acc_bias_ug']['z']:.6f}`

## 模型说明

- 当前只把 **G4 单次真实残余零偏** 替换进静态导航；
- 不额外加入白噪声 / GM / scale-factor residual；
- 因此这版结果更适合回答：**如果直接采用这一轮 G4 实际回收下来的 signed eb/db 残差，静态导航会漂成什么样。**

## 结果摘要

- 最大北向误差：`{m['max_north_err_m']:.6f} m`
- 最大东向误差：`{m['max_east_err_m']:.6f} m`
- 最大水平误差：`{m['max_horizontal_err_m']:.6f} m` = `{m['max_horizontal_err_nm']:.6f} nm`
- 末时刻水平误差：`{m['end_horizontal_err_m']:.6f} m` = `{m['end_horizontal_err_nm']:.6f} nm`
- 等效末值发散速度：`{m['equivalent_end_divergence_rate_nm_per_h']:.6f} nm/h`
- 12 h 后 yaw：`{m['yaw_end_arcsec']:.6f}"`（相对初值变化 `{m['yaw_change_arcsec']:.6f}"`）

## 文件

- 1 Hz 全程轨迹：`{trajectory_csv}`
- 400 Hz 前 60 s 预览：`{preview_csv}`
- 摘要：`{summary_path}`
- 静止 IMU 常值输入：`{imu_profile_path}`
"""
    report_path = OUT_DIR / 'report.md'
    report_path.write_text(report)

    print(json.dumps({
        'ok': True,
        'out_dir': str(OUT_DIR),
        'summary_json': str(summary_path),
        'trajectory_csv': str(trajectory_csv),
        'preview_csv': str(preview_csv),
        'report_md': str(report_path),
        'key_metrics': summary['metrics'],
        'residual_case_dph_ug': {
            'gyro_bias_dph': residual_bias['gyro_bias_dph'],
            'acc_bias_ug': residual_bias['acc_bias_ug'],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
