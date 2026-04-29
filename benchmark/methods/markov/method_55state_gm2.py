from __future__ import annotations

from common_markov import TMP_PSINS, emit_result, load_module, summarize_result

SOURCE = 'test_calibration_markov_noise.py'
METHOD = '55-state GM2 correct tau'
FAMILY = 'full_markov'
VARIANT = '55state_gm2_correct_tau'


def run_method():
    mod = load_module('markov_55_correct', str(TMP_PSINS / SOURCE))
    ts = 0.01
    att0 = mod.np.array([1.0, -91.0, -91.0]) * mod.glv.deg
    pos0 = mod.posset(34.0, 0.0, 0.0)
    paras = mod.np.array([
        [1, 0, 1, 0, 90, 9, 70, 70], [2, 0, 1, 0, 90, 9, 20, 20], [3, 0, 1, 0, 90, 9, 20, 20],
        [4, 0, 1, 0, -90, 9, 20, 20], [5, 0, 1, 0, -90, 9, 20, 20], [6, 0, 1, 0, -90, 9, 20, 20],
        [7, 0, 0, 1, 90, 9, 20, 20], [8, 1, 0, 0, 90, 9, 20, 20], [9, 1, 0, 0, 90, 9, 20, 20],
        [10, 1, 0, 0, 90, 9, 20, 20], [11, -1, 0, 0, 90, 9, 20, 20], [12, -1, 0, 0, 90, 9, 20, 20],
        [13, -1, 0, 0, 90, 9, 20, 20], [14, 0, 0, 1, 90, 9, 20, 20], [15, 0, 0, 1, 90, 9, 20, 20],
        [16, 0, 0, -1, 90, 9, 20, 20], [17, 0, 0, -1, 90, 9, 20, 20], [18, 0, 0, -1, 90, 9, 20, 20],
    ], dtype=float)
    paras[:, 4] = paras[:, 4] * mod.glv.deg
    att = mod.attrottt(att0, paras, ts)
    imu, _ = mod.avp2imu(att, pos0)
    clbt_truth = mod.get_default_clbt()
    imu_clean = mod.imuclbt(imu, clbt_truth)
    imu_noisy = mod.imuadderr_full(
        imu_clean, ts,
        arw=0.001 * mod.glv.dpsh, vrw=1.0 * mod.glv.ugpsHz,
        bi_g=0.005 * mod.glv.dph, tau_g=3000.0,
        bi_a=10.0 * mod.glv.ug, tau_a=3000.0, seed=42,
    )
    return mod.run_calibration(
        imu_noisy, pos0, ts, n_states=55,
        bi_g=0.005 * mod.glv.dph, tau_g=3000.0,
        bi_a=10.0 * mod.glv.ug, tau_a=3000.0,
        label='55-SOGM'
    )


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
