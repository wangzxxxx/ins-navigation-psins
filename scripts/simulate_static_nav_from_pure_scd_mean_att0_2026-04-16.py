#!/usr/bin/env python3
import csv
import json
import math
from pathlib import Path

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
ALIGN_JSON = WORKSPACE / 'psins_method_bench/results/dualpath_pure_scd_custom_noise_0p3_mc50_2026-04-13.json'
OUT_DIR = WORKSPACE / 'tmp/static_nav_pure_scd_mean_att0_2026-04-16'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Approximate coordinates for Beihang University (Xueyuan Road campus), Haidian, Beijing.
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


def load_alignment_mean():
    obj = json.loads(ALIGN_JSON.read_text())
    stats = obj['statistics']
    # Stored order is [pitch, roll, yaw]; convert to thesis-friendly [roll, pitch, yaw]
    pitch_mean_arcsec = float(stats['mean_signed_arcsec'][0])
    roll_mean_arcsec = float(stats['mean_signed_arcsec'][1])
    yaw_mean_arcsec = float(stats['mean_signed_arcsec'][2])
    return {
        'roll_arcsec': roll_mean_arcsec,
        'pitch_arcsec': pitch_mean_arcsec,
        'yaw_arcsec': yaw_mean_arcsec,
        'source_json': str(ALIGN_JSON),
    }


def simulate_alignment_only_static_nav(roll0_arcsec: float, pitch0_arcsec: float, yaw0_arcsec: float):
    """
    Minimal static-navigation simulation driven only by alignment mean attitude errors.

    Model assumptions:
    1. Static base, no additional calibration residuals / IMU biases loaded.
    2. Horizontal position drift comes from initial level-misalignment-induced Schuler dynamics.
    3. Yaw mean is tracked as a constant initial heading offset, but does not drive horizontal drift in this minimal model.
    """
    lat0 = math.radians(LAT0_DEG)
    lon0 = math.radians(LON0_DEG)
    RM, RN, g = earth_radii_and_g(lat0, H0_M)

    roll0_rad = arcsec_to_rad(roll0_arcsec)
    pitch0_rad = arcsec_to_rad(pitch0_arcsec)
    yaw0_rad = arcsec_to_rad(yaw0_arcsec)

    Rn_eff = RM + H0_M
    Re_eff = RN + H0_M
    w_n = math.sqrt(g / Rn_eff)
    w_e = math.sqrt(g / Re_eff)

    t_1hz = np.arange(0.0, T_NAV_S + 1.0, 1.0, dtype=np.float64)
    # North channel driven by pitch mean error (rotation about east axis)
    north_m = Rn_eff * pitch0_rad * (np.cos(w_n * t_1hz) - 1.0)
    vn_mps = -(g / w_n) * pitch0_rad * np.sin(w_n * t_1hz)
    pitch_arcsec = pitch0_arcsec * np.cos(w_n * t_1hz)

    # East channel driven by roll mean error (rotation about north axis)
    east_m = Re_eff * roll0_rad * (1.0 - np.cos(w_e * t_1hz))
    ve_mps = +(g / w_e) * roll0_rad * np.sin(w_e * t_1hz)
    roll_arcsec = roll0_arcsec * np.cos(w_e * t_1hz)

    yaw_arcsec = np.full_like(t_1hz, yaw0_arcsec, dtype=np.float64)
    horiz_m = np.sqrt(north_m * north_m + east_m * east_m)
    lat_est_deg = LAT0_DEG + np.degrees(north_m / Rn_eff)
    lon_est_deg = LON0_DEG + np.degrees(east_m / (Re_eff * math.cos(lat0)))

    idx_max = int(np.argmax(horiz_m))
    summary = {
        'model': 'alignment_only_schuler_static_nav',
        'assumptions': [
            'static base',
            'initial velocity zero',
            'only alignment mean attitude error injected',
            'no extra calibration residual / bias drift loaded',
            'horizontal drift follows minimal Schuler response',
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
            'att0_mean_arcsec': {
                'roll': roll0_arcsec,
                'pitch': pitch0_arcsec,
                'yaw': yaw0_arcsec,
            },
            'att0_mean_rad': {
                'roll': roll0_rad,
                'pitch': pitch0_rad,
                'yaw': yaw0_rad,
            },
            'vn0_mps': [0.0, 0.0, 0.0],
            'pos0': {
                'lat_deg': LAT0_DEG,
                'lon_deg': LON0_DEG,
                'h_m': H0_M,
            },
        },
        'earth_constants': {
            'RM_m': Rn_eff,
            'RN_m': Re_eff,
            'g_mps2': g,
            'schuler_wn_rad_s': w_n,
            'schuler_we_rad_s': w_e,
            'schuler_period_n_s': 2.0 * math.pi / w_n,
            'schuler_period_e_s': 2.0 * math.pi / w_e,
        },
        'static_imu_profile': {
            'gyro_true_ned_rad_s': [7.292115e-5 * math.cos(lat0), 0.0, -7.292115e-5 * math.sin(lat0)],
            'acc_true_body_mps2': [0.0, 0.0, -g],
            'note': 'For a truly static base aligned with NED, the ideal IMU sample is constant; the 400 Hz stream is therefore represented parametrically.',
        },
        'metrics': {
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
            'yaw_constant_arcsec': float(yaw0_arcsec),
        },
    }

    return {
        'summary': summary,
        'trajectory': {
            't_s': t_1hz,
            'north_err_m': north_m,
            'east_err_m': east_m,
            'horizontal_err_m': horiz_m,
            'vn_mps': vn_mps,
            've_mps': ve_mps,
            'roll_err_arcsec': roll_arcsec,
            'pitch_err_arcsec': pitch_arcsec,
            'yaw_err_arcsec': yaw_arcsec,
            'lat_est_deg': lat_est_deg,
            'lon_est_deg': lon_est_deg,
        }
    }


def write_csv(path: Path, rows, header):
    with path.open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def main():
    mean_att = load_alignment_mean()
    result = simulate_alignment_only_static_nav(
        mean_att['roll_arcsec'],
        mean_att['pitch_arcsec'],
        mean_att['yaw_arcsec'],
    )
    summary = result['summary']
    traj = result['trajectory']

    summary_path = OUT_DIR / 'summary.json'
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    imu_profile_path = OUT_DIR / 'static_imu_profile.json'
    imu_profile_path.write_text(json.dumps(summary['static_imu_profile'], ensure_ascii=False, indent=2))

    csv_path = OUT_DIR / 'static_nav_alignment_only_1hz.csv'
    rows = zip(
        traj['t_s'],
        traj['lat_est_deg'],
        traj['lon_est_deg'],
        traj['north_err_m'],
        traj['east_err_m'],
        traj['horizontal_err_m'],
        traj['vn_mps'],
        traj['ve_mps'],
        traj['roll_err_arcsec'],
        traj['pitch_err_arcsec'],
        traj['yaw_err_arcsec'],
    )
    write_csv(
        csv_path,
        rows,
        [
            't_s', 'lat_est_deg', 'lon_est_deg', 'north_err_m', 'east_err_m', 'horizontal_err_m',
            'vn_mps', 've_mps', 'roll_err_arcsec', 'pitch_err_arcsec', 'yaw_err_arcsec'
        ],
    )

    preview_path = OUT_DIR / 'static_nav_alignment_only_preview_400hz_first60s.csv'
    t_preview = np.arange(0.0, 60.0 + DT, DT, dtype=np.float64)
    RM = summary['earth_constants']['RM_m']
    RN = summary['earth_constants']['RN_m']
    g = summary['earth_constants']['g_mps2']
    w_n = summary['earth_constants']['schuler_wn_rad_s']
    w_e = summary['earth_constants']['schuler_we_rad_s']
    roll0 = mean_att['roll_arcsec']
    pitch0 = mean_att['pitch_arcsec']
    yaw0 = mean_att['yaw_arcsec']
    roll0_rad = arcsec_to_rad(roll0)
    pitch0_rad = arcsec_to_rad(pitch0)
    north_preview = RM * pitch0_rad * (np.cos(w_n * t_preview) - 1.0)
    east_preview = RN * roll0_rad * (1.0 - np.cos(w_e * t_preview))
    horiz_preview = np.sqrt(north_preview * north_preview + east_preview * east_preview)
    roll_preview = roll0 * np.cos(w_e * t_preview)
    pitch_preview = pitch0 * np.cos(w_n * t_preview)
    yaw_preview = np.full_like(t_preview, yaw0)
    write_csv(
        preview_path,
        zip(t_preview, north_preview, east_preview, horiz_preview, roll_preview, pitch_preview, yaw_preview),
        ['t_s', 'north_err_m', 'east_err_m', 'horizontal_err_m', 'roll_err_arcsec', 'pitch_err_arcsec', 'yaw_err_arcsec'],
    )

    report_path = OUT_DIR / 'report.md'
    metrics = summary['metrics']
    report = f"""# 静态导航最小仿真（pure-SCD 0.3× MC50 均值姿态作为 att0）\n\n## 设置\n\n- 位置：北京海淀区北航学院路校区近似坐标 `{LAT0_DEG:.4f}°N, {LON0_DEG:.4f}°E, h={H0_M:.1f} m`\n- 初始速度：`[0, 0, 0] m/s`\n- 采样频率：`{FS_HZ:.1f} Hz`\n- 导航总时长：`{T_NAV_S/3600:.1f} h`\n- att0 取自 `pure-SCD @ 0.3×` 的 MC50 均值：\n  - roll = `{mean_att['roll_arcsec']:.6f}"`\n  - pitch = `{mean_att['pitch_arcsec']:.6f}"`\n  - yaw = `{mean_att['yaw_arcsec']:.6f}"`\n\n## 模型假设\n\n- 这是 **alignment-only** 的静态导航最小仿真：只注入对准后的平均姿态误差；\n- 不额外加载标定残差、陀螺/加表零偏和比例因数残差；\n- 水平位置误差由初始水平失准角引起的 Schuler 振荡主导；\n- yaw 作为常值初始航向偏差记录，但在该最小静态模型中不主导水平位置漂移。\n\n## 结果摘要\n\n- 最大北向误差：`{metrics['max_north_err_m']:.6f} m`\n- 最大东向误差：`{metrics['max_east_err_m']:.6f} m`\n- 最大水平误差：`{metrics['max_horizontal_err_m']:.6f} m` = `{metrics['max_horizontal_err_nm']:.6f} nm`\n- 末时刻水平误差：`{metrics['end_horizontal_err_m']:.6f} m` = `{metrics['end_horizontal_err_nm']:.6f} nm`\n- 等效末值发散速度：`{metrics['equivalent_end_divergence_rate_nm_per_h']:.6f} nm/h`\n- 最大水平误差出现时刻：`{metrics['time_of_max_horizontal_err_h']:.6f} h`\n\n## 文件\n\n- 1 Hz 全程轨迹：`{csv_path}`\n- 400 Hz 前 60 s 预览：`{preview_path}`\n- 摘要：`{summary_path}`\n- 静止 IMU 常值输入：`{imu_profile_path}`\n\n## 备注\n\n如果你后面要把这个结果改造成更像第五章“长时间静态导航发散”的版本，就要进一步加入：\n1. 标定残差；\n2. 陀螺/加表 bias 或 GM 噪声；\n3. 更完整的导航 mechanization。\n\n当前这版更适合回答：**仅由 pure-SCD 平均对准误差出发，静态导航会产生多大的最小水平漂移。**\n"""
    report_path.write_text(report)

    print(json.dumps({
        'ok': True,
        'out_dir': str(OUT_DIR),
        'summary_json': str(summary_path),
        'trajectory_csv': str(csv_path),
        'preview_csv': str(preview_path),
        'report_md': str(report_path),
        'key_metrics': summary['metrics'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
