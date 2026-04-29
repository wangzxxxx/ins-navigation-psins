"""Common setup extracted from test_calibration_correlation_decay.py

目标：提供 clean baseline / noisy baseline / SCD 三种方法共享的数据构造，
尽量严格对齐原始 `test_calibration_correlation_decay.py` 的 main() 部分。
"""
import numpy as np
import sys

sys.path.append('/root/.openclaw/workspace/tmp_psins_py')

from psins_py.nav_utils import glv, posset
from psins_py.imu_utils import attrottt, avp2imu, imuclbt
from psins_py.correlation_decay_llm.test_calibration_correlation_decay import (
    get_default_clbt,
    imuadderr_full,
    run_calibration,
)


def build_dataset():
    ts = 0.01
    att0 = np.array([1.0, -91.0, -91.0]) * glv.deg
    pos0 = posset(34.0, 0.0, 0.0)

    # 完全按原脚本 main() 中的 paras 构造
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

    # 原脚本：att=attrottt(att0,paras,ts); imu,_=avp2imu(att,pos0)
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
