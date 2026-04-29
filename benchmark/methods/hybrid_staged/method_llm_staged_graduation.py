from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv
PROJECT_ENV = Path('/root/.openclaw/workspace/tmp_psins_py/psins_py/.env')
if PROJECT_ENV.exists():
    load_dotenv(PROJECT_ENV, override=True)


import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/root/.openclaw/workspace')
BENCH = ROOT / 'psins_method_bench'
TMP = ROOT / 'tmp_psins_py'
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
if str(TMP) not in sys.path:
    sys.path.insert(0, str(TMP))

from summary.extract_metrics import extract_result_metrics
from psins_py.nav_utils import glv, posset
from psins_py.imu_utils import attrottt, avp2imu, imuclbt
from psins_py.staged_calibration_llm import test_calibration_staged_llm as src

SOURCE = 'staged_calibration_llm/test_calibration_staged_llm.py'
METHOD = 'Noisy + LLM Staged Graduation'


def build_dataset():
    ts = 0.01
    att0 = np.array([1.0, -91.0, -91.0]) * glv.deg
    pos0 = posset(34.0, 0.0, 0.0)
    paras = np.array([
        [1, 0, 1, 0, 90, 9, 70, 70],
        [2, 0, 1, 0, 90, 9, 20, 20],
        [3, 0, 1, 0, 90, 9, 20, 20],
        [4, 0, 1, 0, -90, 9, 20, 20],
        [5, 0, 1, 0, -90, 9, 20, 20],
        [6, 0, 1, 0, -90, 9, 20, 20],
        [7, 0, 0, 1, 90, 9, 20, 20],
        [8, 1, 0, 0, 90, 9, 20, 20],
        [9, 1, 0, 0, 90, 9, 20, 20],
        [10, 1, 0, 0, 90, 9, 20, 20],
        [11, -1, 0, 0, 90, 9, 20, 20],
        [12, -1, 0, 0, 90, 9, 20, 20],
        [13, -1, 0, 0, 90, 9, 20, 20],
        [14, 0, 0, 1, 90, 9, 20, 20],
        [15, 0, 0, 1, 90, 9, 20, 20],
        [16, 0, 0, -1, 90, 9, 20, 20],
        [17, 0, 0, -1, 90, 9, 20, 20],
        [18, 0, 0, -1, 90, 9, 20, 20],
    ], dtype=float)
    paras[:, 4] *= glv.deg

    att = attrottt(att0, paras, ts)
    imu, _ = avp2imu(att, pos0)
    clbt_truth = src.get_default_clbt()
    imu_clean = imuclbt(imu, clbt_truth)
    imu_noisy = src.imuadderr_full(
        imu_clean,
        ts,
        arw=0.005 * glv.dpsh,
        vrw=5.0 * glv.ugpsHz,
        bi_g=0.002 * glv.dph,
        tau_g=300.0,
        bi_a=5.0 * glv.ug,
        tau_a=300.0,
    )
    return imu_clean, imu_noisy, pos0, ts


def run_method():
    _, imu_noisy, pos0, ts = build_dataset()
    res = src.run_calibration(imu_noisy, pos0, ts, staged_mode=True, label='LLM Staged')
    metrics = extract_result_metrics('llm_staged_graduation', res)
    metrics.update({
        'source': SOURCE,
        'method_label': METHOD,
        'llm_client_enabled': bool(getattr(src, 'client', None)),
    })
    return metrics


if __name__ == '__main__':
    print(json.dumps(run_method(), ensure_ascii=False))
