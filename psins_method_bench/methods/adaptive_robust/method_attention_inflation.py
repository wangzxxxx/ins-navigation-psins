from pathlib import Path
from dotenv import load_dotenv
PROJECT_ENV = Path('/root/.openclaw/workspace/tmp_psins_py/psins_py/.env')
if PROJECT_ENV.exists():
    load_dotenv(PROJECT_ENV, override=True)

import sys

sys.path.insert(0, '/root/.openclaw/workspace/tmp_psins_py')
import psins_py.test_system_calibration_19pos_inflation as src


def run_method():
    return src.main()


if __name__ == '__main__':
    run_method()
