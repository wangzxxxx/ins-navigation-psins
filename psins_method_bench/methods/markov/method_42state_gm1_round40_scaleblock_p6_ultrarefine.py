from __future__ import annotations

import sys
import types

if 'matplotlib' not in sys.modules:
    matplotlib_stub = types.ModuleType('matplotlib')
    pyplot_stub = types.ModuleType('matplotlib.pyplot')
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules['matplotlib'] = matplotlib_stub
    sys.modules['matplotlib.pyplot'] = pyplot_stub
if 'seaborn' not in sys.modules:
    sys.modules['seaborn'] = types.ModuleType('seaborn')

from common_markov import TMP_PSINS, emit_result, load_module, summarize_result

SOURCE = 'test_calibration_markov_pruned.py'
METHOD = '42-state GM1 round40 selective scale-block P6 + ultra-small guard refinement'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round40_scaleblock_p6_ultrarefine'

P6_PROFILE = {
    12: -2.0,   # dKg_xx
    15: +1.0,   # dKg_xy
    16: +1.2,   # dKg_yy
    20: -2.0,   # dKg_zz
    21: -2.5,   # dKa_xx
}

ROUND39_BASELINE = {
    'rx_y_guard_mult': 1.0370,
    'ry_z_guard_mult': 1.0010,
}

ULTRA_SMALL_CANDIDATES = [
    {'name': 'C1', 'rx_y_guard_mult': 1.0375, 'ry_z_guard_mult': 1.0010},
    {'name': 'C2', 'rx_y_guard_mult': 1.0380, 'ry_z_guard_mult': 1.0015},
    {'name': 'C3', 'rx_y_guard_mult': 1.0380, 'ry_z_guard_mult': 1.0020},
    {'name': 'C4', 'rx_y_guard_mult': 1.0390, 'ry_z_guard_mult': 1.0020},
]
SELECTED_CANDIDATE = ULTRA_SMALL_CANDIDATES[-1]
RX_Y_GUARD_MULT = SELECTED_CANDIDATE['rx_y_guard_mult']
RY_Z_GUARD_MULT = SELECTED_CANDIDATE['ry_z_guard_mult']


def _build_dataset(mod):
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
    bi_g = 0.002 * mod.glv.dph
    bi_a = 5.0 * mod.glv.ug
    tau_g = 300.0
    tau_a = 300.0
    imu_noisy = mod.imuadderr_full(
        imu_clean, ts,
        arw=0.005 * mod.glv.dpsh, vrw=5.0 * mod.glv.ugpsHz,
        bi_g=bi_g, tau_g=tau_g,
        bi_a=bi_a, tau_a=tau_a, seed=42,
    )
    return ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a


def _copy_clbt(mod, clbt):
    return {
        'Kg': mod.np.array(clbt['Kg'], dtype=float).copy(),
        'Ka': mod.np.array(clbt['Ka'], dtype=float).copy(),
        'Ka2': mod.np.array(clbt['Ka2'], dtype=float).copy(),
        'eb': mod.np.array(clbt['eb'], dtype=float).copy(),
        'db': mod.np.array(clbt['db'], dtype=float).copy(),
        'rx': mod.np.array(clbt['rx'], dtype=float).copy(),
        'ry': mod.np.array(clbt['ry'], dtype=float).copy(),
    }


def _apply_selective_feedback(mod, base_clbt, xk):
    refined = _copy_clbt(mod, base_clbt)
    xfb = mod.np.zeros_like(xk)
    for idx, gain in P6_PROFILE.items():
        xfb[idx] = gain * xk[idx]

    dKg = xfb[12:21].reshape(3, 3).T
    refined['Kg'] = (mod.np.eye(3) - dKg) @ refined['Kg']

    dKa = mod.Ka_from_upper(xfb[21:27])
    refined['Ka'] = (mod.np.eye(3) - dKa) @ refined['Ka']

    rx_before = float(refined['rx'][1])
    refined['rx'][1] = refined['rx'][1] * RX_Y_GUARD_MULT
    rx_after = float(refined['rx'][1])

    ry_before = float(refined['ry'][2])
    refined['ry'][2] = refined['ry'][2] * RY_Z_GUARD_MULT
    ry_after = float(refined['ry'][2])

    return refined, xfb, {
        'rx_y_guard': {
            'target_axis': 'rx_y',
            'mode': 'post_scale_multiplicative_guard',
            'multiplier': float(RX_Y_GUARD_MULT),
            'rx_y_before': rx_before,
            'rx_y_after': rx_after,
            'rx_y_delta': rx_after - rx_before,
            'delta_vs_round39_mult': float(RX_Y_GUARD_MULT - ROUND39_BASELINE['rx_y_guard_mult']),
        },
        'ry_z_guard': {
            'target_axis': 'ry_z',
            'mode': 'post_scale_multiplicative_guard',
            'multiplier': float(RY_Z_GUARD_MULT),
            'ry_z_before': ry_before,
            'ry_z_after': ry_after,
            'ry_z_delta': ry_after - ry_before,
            'delta_vs_round39_mult': float(RY_Z_GUARD_MULT - ROUND39_BASELINE['ry_z_guard_mult']),
        },
    }


def run_method():
    mod = load_module('markov_pruned_42_round40_scaleblock_p6_ultrarefine', str(TMP_PSINS / SOURCE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = _build_dataset(mod)
    base_res = mod.run_calibration(
        imu_noisy, pos0, ts, n_states=42,
        bi_g=bi_g, tau_g=tau_g, bi_a=bi_a, tau_a=tau_a,
        label='42-GM1-R40-BASE'
    )
    refined_clbt, xfb, guard_meta = _apply_selective_feedback(mod, base_res[0], base_res[1]['xk'])
    return refined_clbt, base_res[1], base_res[2], base_res[3], {
        'iter_bounds': base_res[4],
        'profile': {str(k): float(v) for k, v in P6_PROFILE.items()},
        'selected_state_labels': {
            '12': 'dKg_xx',
            '15': 'dKg_xy',
            '16': 'dKg_yy',
            '20': 'dKg_zz',
            '21': 'dKa_xx',
        },
        'selected_xk': {str(k): float(base_res[1]['xk'][k]) for k in sorted(P6_PROFILE)},
        'applied_feedback': {str(k): float(xfb[k]) for k in sorted(P6_PROFILE)},
        'round39_baseline': ROUND39_BASELINE,
        'ultra_small_candidates_considered': ULTRA_SMALL_CANDIDATES,
        'selected_candidate': SELECTED_CANDIDATE,
        'rx_y_guard': guard_meta['rx_y_guard'],
        'ry_z_guard': guard_meta['ry_z_guard'],
        'micro_refinement': 'Round39 scale-block P6 preserved exactly; only ultra-small post-scale guard multipliers were adjusted.',
        'policy': 'Selective Kg/Ka main-block feedback only, plus ultra-small post-scale rx_y and ry_z multiplicative guards; eb/db/Ka2/other lever states unchanged.',
    }


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
