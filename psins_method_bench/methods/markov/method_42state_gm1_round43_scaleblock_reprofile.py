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
METHOD = '42-state GM1 round43 controlled scale-block reprofile'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round43_scaleblock_reprofile'

ROUND42_BASELINE = {
    'profile': {
        12: -2.0,   # dKg_xx
        15: +1.0,   # dKg_xy
        16: +1.2,   # dKg_yy
        20: -2.0,   # dKg_zz
        21: -2.5,   # dKa_xx
    },
    'rx_y_guard_mult': 1.0393,
    'ry_z_guard_mult': 1.0023,
    'focus_scale_pct': {
        'dKg_xx': 29.330719110287873,
        'dKg_xy': 9.514194248358994,
        'dKg_yy': 5.091596005363701,
        'dKg_zz': 5.256391465799213,
        'dKa_xx': 4.599401636239235,
    },
    'lever_guard_pct': {
        'rx_y': 1.880061874490295,
        'ry_z': 2.0609244691471043,
    },
    'overall': {
        'mean_pct_error': 5.608525799540521,
        'median_pct_error': 3.1970341561855493,
        'max_pct_error': 29.330719110287873,
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
        'name': 'P1_xx_-2.1',
        'profile': {12: -2.1, 15: +1.0, 16: +1.2, 20: -2.0, 21: -2.5},
        'rationale': 'Only strengthen dKg_xx correction slightly.',
    },
    {
        'name': 'P2_xx_-2.2',
        'profile': {12: -2.2, 15: +1.0, 16: +1.2, 20: -2.0, 21: -2.5},
        'rationale': 'Medium dKg_xx strengthening without touching the other four blocks.',
    },
    {
        'name': 'P3_xx_-2.3',
        'profile': {12: -2.3, 15: +1.0, 16: +1.2, 20: -2.0, 21: -2.5},
        'rationale': 'Largest allowed dKg_xx-only push in this round.',
    },
    {
        'name': 'P4_xx_-2.2_zz_-2.05',
        'profile': {12: -2.2, 15: +1.0, 16: +1.2, 20: -2.05, 21: -2.5},
        'rationale': 'Test whether a tiny extra dKg_zz push helps after the dKg_xx gain.',
    },
    {
        'name': 'P5_xx_-2.2_xy_1.05_yy_1.25_ka_-2.6',
        'profile': {12: -2.2, 15: +1.05, 16: +1.25, 20: -2.0, 21: -2.6},
        'rationale': 'Keep dKg_zz fixed, but conservatively tighten dKg_xy / dKg_yy / dKa_xx together.',
    },
    {
        'name': 'P6_xx_-2.2_balanced_all',
        'profile': {12: -2.2, 15: +1.05, 16: +1.25, 20: -2.05, 21: -2.6},
        'rationale': 'Round42 profile with a balanced mild reprofile on all five main blocks.',
    },
    {
        'name': 'P7_xx_-2.3_xy_1.05_yy_1.25_ka_-2.6',
        'profile': {12: -2.3, 15: +1.05, 16: +1.25, 20: -2.0, 21: -2.6},
        'rationale': 'Carry the stronger dKg_xx push while keeping dKg_zz untouched.',
    },
    {
        'name': 'P8_xx_-2.3_balanced_all',
        'profile': {12: -2.3, 15: +1.05, 16: +1.25, 20: -2.05, 21: -2.6},
        'rationale': 'Final balanced reprofile: strongest dKg_xx push plus mild dKg_xy / dKg_yy / dKg_zz / dKa_xx cooperation.',
    },
]
SELECTED_PROFILE = PROFILE_CANDIDATES[-1]

GUARD_CANDIDATES = [
    {
        'name': 'G0_keep_round42',
        'rx_y_guard_mult': 1.0393,
        'ry_z_guard_mult': 1.0023,
        'rationale': 'Keep the Round42 paired ultra-small guards unchanged.',
    },
    {
        'name': 'G1_paired_plus_5e-5',
        'rx_y_guard_mult': 1.03935,
        'ry_z_guard_mult': 1.00235,
        'rationale': 'One last paired +5e-5 guard confirmation on the selected Round43 profile.',
    },
]
SELECTED_GUARD = GUARD_CANDIDATES[-1]


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
            'delta_vs_round42_mult': float(rx_y_guard_mult - ROUND42_BASELINE['rx_y_guard_mult']),
        },
        'ry_z_guard': {
            'target_axis': 'ry_z',
            'mode': 'post_scale_multiplicative_guard',
            'multiplier': float(ry_z_guard_mult),
            'ry_z_before': ry_before,
            'ry_z_after': ry_after,
            'ry_z_delta': ry_after - ry_before,
            'delta_vs_round42_mult': float(ry_z_guard_mult - ROUND42_BASELINE['ry_z_guard_mult']),
        },
    }


def _summarize_profile_candidate(mod, base_clbt, xk, params, candidate):
    refined, xfb, _ = _apply_selective_feedback(
        mod,
        base_clbt,
        xk,
        candidate['profile'],
        ROUND42_BASELINE['rx_y_guard_mult'],
        ROUND42_BASELINE['ry_z_guard_mult'],
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
    mod = load_module('markov_pruned_42_round43_scaleblock_reprofile', str(TMP_PSINS / SOURCE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = _build_dataset(mod)
    base_res = mod.run_calibration(
        imu_noisy, pos0, ts, n_states=42,
        bi_g=bi_g, tau_g=tau_g, bi_a=bi_a, tau_a=tau_a,
        label='42-GM1-R43-BASE'
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
        'round42_baseline': ROUND42_BASELINE,
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
        'selection_rationale': 'Among the eight hand-designed P6 reprofiles, P8 delivered the strongest combined five-main-block improvement while also pushing dKg_xx down materially versus Round42; a final paired +5e-5 guard confirmation then shaved rx_y / ry_z and the overall mean a little further without disturbing the new scale-block gains.',
        'policy': 'Round43 keeps the Round42 mainline and only performs a controlled small-range P6 reprofile around the five dominant scale-factor blocks, plus one paired guard confirmation on the selected profile. No scan, no brute-force sweep, no changes to eb/db/Ka2/other lever states.',
    }


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
