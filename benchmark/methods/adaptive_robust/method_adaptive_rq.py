from pathlib import Path
from dotenv import load_dotenv
PROJECT_ENV = Path('/root/.openclaw/workspace/tmp_psins_py/psins_py/.env')
if PROJECT_ENV.exists():
    load_dotenv(PROJECT_ENV, override=True)

from common_setup import build_dataset
import sys

sys.path.insert(0, '/root/.openclaw/workspace/tmp_psins_py')
import psins_py.test_calibration_adaptive_rq_llm as src


def run_method():
    imu_clean, imu_noisy, pos0, ts = build_dataset()
    src.is_first_llm_call = True
    src.global_llm_mask_cache = {}
    return src.run_calibration(imu_noisy, pos0, ts, llm_mode=True, label='Adaptive RQ')


if __name__ == '__main__':
    res = run_method()
    print(res[0].keys() if isinstance(res, tuple) else type(res))
