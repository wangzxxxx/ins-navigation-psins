"""Shared dataset builder for adaptive/robust extracted methods."""
from __future__ import annotations

import sys
import numpy as np

sys.path.insert(0, '/root/.openclaw/workspace/tmp_psins_py')

from psins_py.nav_utils import glv, posset
from psins_py.imu_utils import attrottt, avp2imu, imuclbt
from psins_py.test_calibration_adaptive_rq_llm import get_default_clbt, imuadderr_full


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
    clbt_truth = get_default_clbt()
    imu_clean = imuclbt(imu, clbt_truth)

    ARW = 0.005 * glv.dpsh
    VRW = 5.0 * glv.ugpsHz
    BI_G = 0.002 * glv.dph
    BI_A = 5.0 * glv.ug
    TAU_G = 300.0
    TAU_A = 300.0
    imu_noisy = imuadderr_full(
        imu_clean,
        ts,
        arw=ARW,
        vrw=VRW,
        bi_g=BI_G,
        tau_g=TAU_G,
        bi_a=BI_A,
        tau_a=TAU_A,
    )
    return imu_clean, imu_noisy, pos0, ts
