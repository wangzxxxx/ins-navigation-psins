from pathlib import Path
from dotenv import load_dotenv
PROJECT_ENV = Path('/root/.openclaw/workspace/tmp_psins_py/psins_py/.env')
if PROJECT_ENV.exists():
    load_dotenv(PROJECT_ENV, override=True)

from common_setup import build_dataset
import sys

sys.path.insert(0, '/root/.openclaw/workspace/tmp_psins_py')
from psins_py.huber_robust_kf_llm.test_calibration_huber_robust import run_calibration


def run_method():
    imu_clean, imu_noisy, pos0, ts = build_dataset()
    return run_calibration(imu_noisy, pos0, ts, huber_mode=True, label='Huber')


if __name__ == '__main__':
    res = run_method()
    print(res[0].keys() if isinstance(res, tuple) else type(res))
