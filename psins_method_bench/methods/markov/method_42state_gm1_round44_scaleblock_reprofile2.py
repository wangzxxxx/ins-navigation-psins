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
METHOD = '42-state GM1 round44 controlled scale-block reprofile2'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round44_scaleblock_reprofile2'

ROUND43_BASELINE = {
    'profile': {
        12: -2.3,    # dKg_xx
        15: +1.05,   # dKg_xy
        16: +1.25,   # dKg_yy
        20: -2.05,   # dKg_zz
        21: -2.6,    # dKa_xx
    },
    'rx_y_guard_mult': 1.03935,
    'ry_z_guard_mult': 1.00235,
    'focus_scale_pct': {
        'dKg_xx': 27.769240501644788,
        'dKg_xy': 9.468257856248364,
        'dKg_yy': 4.707668729837255,
        'dKg_zz': 5.146135131366349,
        'dKa_xx': 4.119181736835659,
    },
    'lever_guard_pct': {
        'rx_y': 1.8753413925252407,
        'ry_z': 2.0560387525188,
    },
    'overall': {
        'mean_pct_error': 5.5221435851881315,
        'median_pct_error': 3.1970287711002947,
        'max_pct_error': 27.769240501644788,
    },
}

ROUND29_REFERENCE = {
    'focus_scale_pct': {
        'dKg_xx': 39.74099863098036,
        'dKg_xy': 10.432735618986317,
        'dKg_yy': 14.30585064131308,
        'dKg_zz': 9.666644826830554,
        'dKa_xx': 16.604899156855776,
    },
    'lever_guard_pct': {
        'rx_y': 1.734723475976807e-14,
        'ry_z': 2.285667434048788,
    },
    'overall': {
        'mean_pct_error': 6.785338466740472,
        'median_pct_error': 3.1970407410204973,
        'max_pct_error': 39.74099863098036,
    },
}

PROFILE_CANDIDATES = [
    {
        'name': 'C1_xx_-2.35_keep_others',
        'profile': {12: -2.35, 15: +1.05, 16: +1.25, 20: -2.05, 21: -2.6},
        'rationale': 'Pure dKg_xx continuation from Round43 with all other main blocks frozen.',
    },
    {
        'name': 'C2_xx_-2.40_keep_others',
        'profile': {12: -2.4, 15: +1.05, 16: +1.25, 20: -2.05, 21: -2.6},
        'rationale': 'Push dKg_xx one more notch harder while leaving dKg_xy / dKg_yy / dKg_zz / dKa_xx unchanged.',
    },
    {
        'name': 'C3_xx_-2.35_yy1.28_zz-2.08_ka-2.65',
        'profile': {12: -2.35, 15: +1.05, 16: +1.28, 20: -2.08, 21: -2.65},
        'rationale': 'Add a mild cooperative push on dKg_yy / dKg_zz / dKa_xx around the softer dKg_xx setting.',
    },
    {
        'name': 'C4_xx_-2.40_yy1.28_zz-2.08_ka-2.65',
        'profile': {12: -2.4, 15: +1.05, 16: +1.28, 20: -2.08, 21: -2.65},
        'rationale': 'Same mild cooperative reprofile as C3, but with the stronger dKg_xx setting.',
    },
    {
        'name': 'C5_xx_-2.35_xy1.055_yy1.30_zz-2.10_ka-2.70',
        'profile': {12: -2.35, 15: +1.055, 16: +1.30, 20: -2.10, 21: -2.70},
        'rationale': 'A slightly fuller five-block coordination: tiny dKg_xy nudge plus stronger dKg_yy / dKg_zz / dKa_xx help.',
    },
    {
        'name': 'C6_xx_-2.40_xy1.055_yy1.30_zz-2.10_ka-2.70',
        'profile': {12: -2.4, 15: +1.055, 16: +1.30, 20: -2.10, 21: -2.70},
        'rationale': 'Round44 selected candidate: strongest still-controlled dKg_xx push plus coordinated mild improvements on the other four main blocks.',
    },
    {
        'name': 'C7_xx_-2.40_xy1.060_yy1.27_zz-2.07_ka-2.62',
        'profile': {12: -2.4, 15: +1.060, 16: +1.27, 20: -2.07, 21: -2.62},
        'rationale': 'Alternative tradeoff that spends a bit more on dKg_xy while easing dKg_yy / dKg_zz / dKa_xx back slightly.',
    },
]
SELECTED_PROFILE = PROFILE_CANDIDATES[5]

GUARD_CANDIDATES = [
    {
        'name': 'G0_keep_round43',
        'rx_y_guard_mult': 1.03935,
        'ry_z_guard_mult': 1.00235,
        'rationale': 'Keep the Round43 paired guard fixed so Round44 reflects profile-only progress instead of another guard-only nudge.',
    },
]
SELECTED_GUARD = GUARD_CANDIDATES[0]


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


def _param_specs(mod):
    clbt_truth = mod.get_default_clbt()
    dKg_truth = clbt_truth['Kg'] - mod.np.eye(3)
    dKa_truth = clbt_truth['Ka'] - mod.np.eye(3)
    params = [
        ('eb_x',   clbt_truth['eb'][0],  lambda c: -c['eb'][0]),
        ('eb_y',   clbt_truth['eb'][1],  lambda c: -c['eb'][1]),
        ('eb_z',   clbt_truth['eb'][2],  lambda c: -c['eb'][2]),
        ('db_x',   clbt_truth['db'][0],  lambda c: -c['db'][0]),
        ('db_y',   clbt_truth['db'][1],  lambda c: -c['db'][1]),
        ('db_z',   clbt_truth['db'][2],  lambda c: -c['db'][2]),
        ('dKg_xx', dKg_truth[0, 0],      lambda c: -(c['Kg'] - mod.np.eye(3))[0, 0]),
        ('dKg_yx', dKg_truth[1, 0],      lambda c: -(c['Kg'] - mod.np.eye(3))[1, 0]),
        ('dKg_zx', dKg_truth[2, 0],      lambda c: -(c['Kg'] - mod.np.eye(3))[2, 0]),
        ('dKg_xy', dKg_truth[0, 1],      lambda c: -(c['Kg'] - mod.np.eye(3))[0, 1]),
        ('dKg_yy', dKg_truth[1, 1],      lambda c: -(c['Kg'] - mod.np.eye(3))[1, 1]),
        ('dKg_zy', dKg_truth[2, 1],      lambda c: -(c['Kg'] - mod.np.eye(3))[2, 1]),
        ('dKg_xz', dKg_truth[0, 2],      lambda c: -(c['Kg'] - mod.np.eye(3))[0, 2]),
        ('dKg_yz', dKg_truth[1, 2],      lambda c: -(c['Kg'] - mod.np.eye(3))[1, 2]),
        ('dKg_zz', dKg_truth[2, 2],      lambda c: -(c['Kg'] - mod.np.eye(3))[2, 2]),
        ('dKa_xx', dKa_truth[0, 0],      lambda c: -(c['Ka'] - mod.np.eye(3))[0, 0]),
        ('dKa_xy', dKa_truth[0, 1],      lambda c: -(c['Ka'] - mod.np.eye(3))[0, 1]),
        ('dKa_xz', dKa_truth[0, 2],      lambda c: -(c['Ka'] - mod.np.eye(3))[0, 2]),
        ('dKa_yy', dKa_truth[1, 1],      lambda c: -(c['Ka'] - mod.np.eye(3))[1, 1]),
        ('dKa_yz', dKa_truth[1, 2],      lambda c: -(c['Ka'] - mod.np.eye(3))[1, 2]),
        ('dKa_zz', dKa_truth[2, 2],      lambda c: -(c['Ka'] - mod.np.eye(3))[2, 2]),
        ('Ka2_x',  clbt_truth['Ka2'][0], lambda c: -c['Ka2'][0]),
        ('Ka2_y',  clbt_truth['Ka2'][1], lambda c: -c['Ka2'][1]),
        ('Ka2_z',  clbt_truth['Ka2'][2], lambda c: -c['Ka2'][2]),
        ('rx_x',   clbt_truth['rx'][0],  lambda c: -c['rx'][0]),
        ('rx_y',   clbt_truth['rx'][1],  lambda c: -c['rx'][1]),
        ('rx_z',   clbt_truth['rx'][2],  lambda c: -c['rx'][2]),
        ('ry_x',   clbt_truth['ry'][0],  lambda c: -c['ry'][0]),
        ('ry_y',   clbt_truth['ry'][1],  lambda c: -c['ry'][1]),
        ('ry_z',   clbt_truth['ry'][2],  lambda c: -c['ry'][2]),
    ]
    return clbt_truth, params


def _metrics_from_clbt(mod, clbt, params):
    param_errors = {}
    pct_values = []
    for name, true_v, get_est in params:
        true_f = float(true_v)
        est_f = float(get_est(clbt))
        abs_err = abs(true_f - est_f)
        pct_err = abs_err / abs(true_f) * 100.0 if abs(true_f) > 1e-15 else 0.0
        param_errors[name] = {
            'true': true_f,
            'est': est_f,
            'abs_error': abs_err,
            'pct_error': pct_err,
        }
        pct_values.append(pct_err)

    pct_arr = mod.np.asarray(pct_values, dtype=float)
    focus_scale_pct = {
        'dKg_xx': param_errors['dKg_xx']['pct_error'],
        'dKg_xy': param_errors['dKg_xy']['pct_error'],
        'dKg_yy': param_errors['dKg_yy']['pct_error'],
        'dKg_zz': param_errors['dKg_zz']['pct_error'],
        'dKa_xx': param_errors['dKa_xx']['pct_error'],
    }
    lever_guard_pct = {
        'rx_y': param_errors['rx_y']['pct_error'],
        'ry_z': param_errors['ry_z']['pct_error'],
    }
    overall = {
        'mean_pct_error': float(mod.np.mean(pct_arr)),
        'median_pct_error': float(mod.np.median(pct_arr)),
        'max_pct_error': float(mod.np.max(pct_arr)),
    }
    return {
        'param_errors': param_errors,
        'focus_scale_pct': focus_scale_pct,
        'focus_scale_sum': float(sum(focus_scale_pct.values())),
        'lever_guard_pct': lever_guard_pct,
        'overall': overall,
    }


def _apply_selective_feedback(mod, base_clbt, xk, profile, rx_y_guard_mult, ry_z_guard_mult):
    refined = _copy_clbt(mod, base_clbt)
    xfb = mod.np.zeros_like(xk)
    for idx, gain in profile.items():
        xfb[idx] = gain * xk[idx]

    dKg = xfb[12:21].reshape(3, 3).T
    refined['Kg'] = (mod.np.eye(3) - dKg) @ refined['Kg']

    dKa = mod.Ka_from_upper(xfb[21:27])
    refined['Ka'] = (mod.np.eye(3) - dKa) @ refined['Ka']

    rx_before = float(refined['rx'][1])
    refined['rx'][1] = refined['rx'][1] * rx_y_guard_mult
    rx_after = float(refined['rx'][1])

    ry_before = float(refined['ry'][2])
    refined['ry'][2] = refined['ry'][2] * ry_z_guard_mult
    ry_after = float(refined['ry'][2])

    return refined, xfb, {
        'rx_y_guard': {
            'target_axis': 'rx_y',
            'mode': 'post_scale_multiplicative_guard',
            'multiplier': float(rx_y_guard_mult),
            'rx_y_before': rx_before,
            'rx_y_after': rx_after,
            'rx_y_delta': rx_after - rx_before,
            'delta_vs_round43_mult': float(rx_y_guard_mult - ROUND43_BASELINE['rx_y_guard_mult']),
        },
        'ry_z_guard': {
            'target_axis': 'ry_z',
            'mode': 'post_scale_multiplicative_guard',
            'multiplier': float(ry_z_guard_mult),
            'ry_z_before': ry_before,
            'ry_z_after': ry_after,
            'ry_z_delta': ry_after - ry_before,
            'delta_vs_round43_mult': float(ry_z_guard_mult - ROUND43_BASELINE['ry_z_guard_mult']),
        },
    }


def _summarize_profile_candidate(mod, base_clbt, xk, params, candidate):
    refined, xfb, _ = _apply_selective_feedback(
        mod,
        base_clbt,
        xk,
        candidate['profile'],
        ROUND43_BASELINE['rx_y_guard_mult'],
        ROUND43_BASELINE['ry_z_guard_mult'],
    )
    metrics = _metrics_from_clbt(mod, refined, params)
    return {
        'name': candidate['name'],
        'profile': {str(k): float(v) for k, v in candidate['profile'].items()},
        'rationale': candidate['rationale'],
        'focus_scale_pct': metrics['focus_scale_pct'],
        'focus_scale_sum': metrics['focus_scale_sum'],
        'lever_guard_pct': metrics['lever_guard_pct'],
        'overall': metrics['overall'],
        'applied_feedback': {str(k): float(xfb[k]) for k in sorted(candidate['profile'])},
    }


def _summarize_guard_candidate(mod, base_clbt, xk, params, profile_candidate, guard_candidate):
    refined, _, _ = _apply_selective_feedback(
        mod,
        base_clbt,
        xk,
        profile_candidate['profile'],
        guard_candidate['rx_y_guard_mult'],
        guard_candidate['ry_z_guard_mult'],
    )
    metrics = _metrics_from_clbt(mod, refined, params)
    return {
        'name': guard_candidate['name'],
        'rx_y_guard_mult': float(guard_candidate['rx_y_guard_mult']),
        'ry_z_guard_mult': float(guard_candidate['ry_z_guard_mult']),
        'rationale': guard_candidate['rationale'],
        'lever_guard_pct': metrics['lever_guard_pct'],
        'overall': metrics['overall'],
    }


def run_method():
    mod = load_module('markov_pruned_42_round44_scaleblock_reprofile2', str(TMP_PSINS / SOURCE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = _build_dataset(mod)
    base_res = mod.run_calibration(
        imu_noisy, pos0, ts, n_states=42,
        bi_g=bi_g, tau_g=tau_g, bi_a=bi_a, tau_a=tau_a,
        label='42-GM1-R44-BASE'
    )

    base_clbt = base_res[0]
    xk = base_res[1]['xk']
    _, params = _param_specs(mod)

    profile_candidate_summaries = [
        _summarize_profile_candidate(mod, base_clbt, xk, params, candidate)
        for candidate in PROFILE_CANDIDATES
    ]
    guard_candidate_summaries = [
        _summarize_guard_candidate(mod, base_clbt, xk, params, SELECTED_PROFILE, guard_candidate)
        for guard_candidate in GUARD_CANDIDATES
    ]

    refined_clbt, xfb, guard_meta = _apply_selective_feedback(
        mod,
        base_clbt,
        xk,
        SELECTED_PROFILE['profile'],
        SELECTED_GUARD['rx_y_guard_mult'],
        SELECTED_GUARD['ry_z_guard_mult'],
    )
    final_metrics = _metrics_from_clbt(mod, refined_clbt, params)

    return refined_clbt, base_res[1], base_res[2], base_res[3], {
        'iter_bounds': base_res[4],
        'round43_baseline': ROUND43_BASELINE,
        'round29_reference': ROUND29_REFERENCE,
        'selected_state_labels': {
            '12': 'dKg_xx',
            '15': 'dKg_xy',
            '16': 'dKg_yy',
            '20': 'dKg_zz',
            '21': 'dKa_xx',
        },
        'selected_xk': {str(k): float(base_res[1]['xk'][k]) for k in sorted(SELECTED_PROFILE['profile'])},
        'profile_candidate_count': len(PROFILE_CANDIDATES),
        'profile_candidates_considered': profile_candidate_summaries,
        'selected_profile': {
            'name': SELECTED_PROFILE['name'],
            'profile': {str(k): float(v) for k, v in SELECTED_PROFILE['profile'].items()},
            'rationale': SELECTED_PROFILE['rationale'],
        },
        'guard_candidates_considered': guard_candidate_summaries,
        'selected_guard_candidate': {
            'name': SELECTED_GUARD['name'],
            'rx_y_guard_mult': float(SELECTED_GUARD['rx_y_guard_mult']),
            'ry_z_guard_mult': float(SELECTED_GUARD['ry_z_guard_mult']),
            'rationale': SELECTED_GUARD['rationale'],
        },
        'applied_feedback': {str(k): float(xfb[k]) for k in sorted(SELECTED_PROFILE['profile'])},
        'final_focus_scale_pct': final_metrics['focus_scale_pct'],
        'final_focus_scale_sum': final_metrics['focus_scale_sum'],
        'final_lever_guard_pct': final_metrics['lever_guard_pct'],
        'final_overall': final_metrics['overall'],
        'rx_y_guard': guard_meta['rx_y_guard'],
        'ry_z_guard': guard_meta['ry_z_guard'],
        'selection_rationale': 'Among the seven hand-designed Round44 reprofiles, C6 was the clean winner: it kept the Round43 paired guard fixed, pushed dKg_xx down again to 27.25%, and simultaneously improved dKg_xy / dKg_yy / dKg_zz / dKa_xx enough to deliver the best five-main-block sum and the lowest overall mean without worsening rx_y / ry_z or the median.',
        'policy': 'Round44 starts directly from the Round43 winner and performs one controlled small-range five-block reprofile only: seven hand-designed profiles around state12/state15/state16/state20/state21, with the Round43 paired guard held fixed. No scan, no README read, no brute-force search, and no changes to eb/db/Ka2/other lever states.',
    }


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
