from common_setup import build_dataset, run_calibration

if __name__ == '__main__':
    imu_clean, imu_noisy, pos0, ts = build_dataset()
    res = run_calibration(imu_clean, pos0, ts, scd_mode=False, label='Clean')
    print(res.keys() if isinstance(res, dict) else type(res))
