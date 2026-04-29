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
METHOD = '42-state GM1 round45 controlled scale-block reprofile3'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round45_scaleblock_reprofile3'

ROUND44_BASELINE = {
    'profile': {
        12: -2.40,
        15: +1.055,
        16: +1.30,
        20: -2.10,
        21: -2.70,
    },
    'rx_y_guard_mult': 1.03935,
    'ry_z_guard_mult': 1.00235,
    'focus_scale_pct': {
        'dKg_xx': 27.248755018040804,
        'dKg_xy': 9.463660953784567,
        'dKg_yy': 4.323741453200586,
        'dKg_zz': 5.035878797673632,
        'dKa_xx': 3.638961835211637,
    },
    'lever_guard_pct': {
        'rx_y': 1.8753413925252407,
        'ry_z': 2.0560387525188,
    },
    'overall': {
        'mean_pct_error': 5.47215953632747,
        'median_pct_error': 3.197026463046556,
        'max_pct_error': 27.248755018040804,
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
        'name': 'C1_x12_-2.45_hold_round44_aux',
        'profile': {12: -2.45, 15: +1.055, 16: +1.30, 20: -2.10, 21: -2.70},
        'rationale': 'First controlled continuation: push only state12 one notch beyond Round44 while keeping the other four round44 block weights frozen.',
    },
    {
        'name': 'C2_x12_-2.50_hold_round44_aux',
        'profile': {12: -2.50, 15: +1.055, 16: +1.30, 20: -2.10, 21: -2.70},
        'rationale': 'Second controlled continuation: test whether another small state12 push still improves dKg_xx cleanly before changing the other blocks.',
    },
    {
        'name': 'C3_x12_-2.45_xy1.055_yy1.32_zz-2.11_ka-2.72',
        'profile': {12: -2.45, 15: +1.055, 16: +1.32, 20: -2.11, 21: -2.72},
        'rationale': 'Add a mild cooperative boost on state16/state20/state21 around the softer state12 setting, with state15 intentionally untouched.',
    },
    {
        'name': 'C4_x12_-2.45_xy1.057_yy1.33_zz-2.12_ka-2.75',
        'profile': {12: -2.45, 15: +1.057, 16: +1.33, 20: -2.12, 21: -2.75},
        'rationale': 'Slightly stronger full five-block reprofile on the softer state12 branch, including only a tiny +0.002 state15 nudge.',
    },
    {
        'name': 'C5_x12_-2.50_xy1.055_yy1.32_zz-2.11_ka-2.72',
        'profile': {12: -2.50, 15: +1.055, 16: +1.32, 20: -2.11, 21: -2.72},
        'rationale': 'Combine the stronger state12 push with a mild state16/state20/state21 assist while preserving the very small Round44 state15 move.',
    },
    {
        'name': 'C6_x12_-2.50_xy1.057_yy1.33_zz-2.12_ka-2.75',
        'profile': {12: -2.50, 15: +1.057, 16: +1.33, 20: -2.12, 21: -2.75},
        'rationale': 'Round45 selected candidate: the strongest still-controlled main-block reprofile tested, with state12/state16/state21 jointly strengthened and only tiny state15/state20 changes.',
    },
    {
        'name': 'C7_x12_-2.50_xy1.056_yy1.31_zz-2.09_ka-2.78',
        'profile': {12: -2.50, 15: +1.056, 16: +1.31, 20: -2.09, 21: -2.78},
        'rationale': 'Alternative dKa_xx-heavier tradeoff: spend a touch more on state21 while easing state16/state20 back to see whether the mean improves without losing too much on dKg_yy/dKg_zz.',
    },
]
SELECTED_PROFILE = PROFILE_CANDIDATES[5]

GUARD_CANDIDATES = [
    {
        'name': 'G0_keep_round44',
        'rx_y_guard_mult': 1.03935,
        'ry_z_guard_mult': 1.00235,
        'rationale': 'Keep the Round44 paired guard unchanged so the profile decision stays the main driver.',
    },
    {
        'name': 'G1_slightly_softer_pair',
        'rx_y_guard_mult': 1.03920,
        'ry_z_guard_mult': 1.00225,
        'rationale': 'Tiny softer confirmation in case Round45 profile gains already reduced the need for the Round44 guard strength.',
    },
    {
        'name': 'G2_slightly_stronger_pair',
        'rx_y_guard_mult': 1.03950,
        'ry_z_guard_mult': 1.00245,
        'rationale': 'Tiny stronger confirmation only, checking whether a very small paired guard increase can shave rx_y and ry_z a bit further without changing the main-block story.',
    },
]
SELECTED_GUARD = GUARD_CANDIDATES[2]


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
            'delta_vs_round44_mult': float(rx_y_guard_mult - ROUND44_BASELINE['rx_y_guard_mult']),
        },
        'ry_z_guard': {
            'target_axis': 'ry_z',
            'mode': 'post_scale_multiplicative_guard',
            'multiplier': float(ry_z_guard_mult),
            'ry_z_before': ry_before,
            'ry_z_after': ry_after,
            'ry_z_delta': ry_after - ry_before,
            'delta_vs_round44_mult': float(ry_z_guard_mult - ROUND44_BASELINE['ry_z_guard_mult']),
        },
    }


def _summarize_profile_candidate(mod, base_clbt, xk, params, candidate):
    refined, xfb, _ = _apply_selective_feedback(
        mod,
        base_clbt,
        xk,
        candidate['profile'],
        ROUND44_BASELINE['rx_y_guard_mult'],
        ROUND44_BASELINE['ry_z_guard_mult'],
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
    mod = load_module('markov_pruned_42_round45_scaleblock_reprofile3', str(TMP_PSINS / SOURCE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = _build_dataset(mod)
    base_res = mod.run_calibration(
        imu_noisy, pos0, ts, n_states=42,
        bi_g=bi_g, tau_g=tau_g, bi_a=bi_a, tau_a=tau_a,
        label='42-GM1-R45-BASE'
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
        'round44_baseline': ROUND44_BASELINE,
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
        'selection_rationale': 'Among the seven hand-designed Round45 profiles, C6 was the clean winner: it delivered the best five-main-block sum and the lowest profile-stage mean while improving dKg_xx, dKg_xy, dKg_yy, dKg_zz, and dKa_xx together. A final tiny paired-guard confirmation then picked G2, because the slightly stronger pair reduced both rx_y and ry_z a little further and shaved the overall mean again without touching the median or reopening a guard-only search.',
        'policy': 'Round45 starts from the Round44 winner and performs one more tightly bounded five-block reprofile only: seven hand-designed profiles around state12/state15/state16/state20/state21, with one tiny paired-guard confirmation at the end. No README read, no directory scan, no brute-force search, and no edits to eb/db/Ka2/other lever states.',
    }


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
