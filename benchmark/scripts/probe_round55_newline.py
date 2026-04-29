from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

if 'matplotlib' not in sys.modules:
    matplotlib_stub = types.ModuleType('matplotlib')
    pyplot_stub = types.ModuleType('matplotlib.pyplot')
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules['matplotlib'] = matplotlib_stub
    sys.modules['matplotlib.pyplot'] = pyplot_stub
if 'seaborn' not in sys.modules:
    sys.modules['seaborn'] = types.ModuleType('seaborn')

ROOT = Path('/root/.openclaw/workspace')
METHOD_DIR = ROOT / 'psins_method_bench' / 'methods' / 'markov'
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'
R53_JSON = RESULTS_DIR / 'R53_42state_gm1_round53_internalized_trustcov_release_param_errors.json'
OUTPUT_JSON = RESULTS_DIR / 'round55_newline_probe_summary.json'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from common_markov import load_module


FOCUS_KEYS = ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx']
LEVER_KEYS = ['rx_y', 'ry_z']
TARGET_STATE_LABELS = {'15': 'dKg_xy', '20': 'dKg_zz'}


CANDIDATES = [
    {
        'name': 'alpha_split_xy_up_zz_damp',
        'description': 'Pure new-line asymmetric alpha route: push state15 a bit harder in iter1 and damp its iter2 rollback, while directly damping state20 in both passes.',
        'iter_patches': {
            0: {
                'state_alpha_mult': {15: 1.020, 20: 0.970},
            },
            1: {
                'state_alpha_mult': {15: 0.760, 20: 0.900},
            },
        },
    },
    {
        'name': 'priorq_split_xy_up_zz_damp',
        'description': 'No alpha surgery; only selected-state prior/q asymmetry so 15 gets a bit more room while 20 gets slightly tighter covariance.',
        'iter_patches': {
            0: {
                'state_prior_diag_mult': {15: 1.080, 20: 0.920},
                'state_q_static_mult': {15: 1.100, 20: 0.900},
                'state_q_dynamic_mult': {15: 1.050, 20: 0.950},
                'state_q_late_mult': {15: 1.040, 20: 0.960},
            },
            1: {
                'state_prior_diag_mult': {15: 1.040, 20: 0.950},
                'state_q_dynamic_mult': {15: 1.040, 20: 0.960},
                'state_q_late_mult': {15: 1.030, 20: 0.970},
            },
        },
    },
    {
        'name': 'hybrid_xy_up_zz_damp_ryz_guard',
        'description': 'Hybrid 15-up / 20-damp branch with both asym prior/q and alpha, then a very small ry_z post-guard to see if zz and ry_z can fall together.',
        'iter_patches': {
            0: {
                'state_prior_diag_mult': {15: 1.060, 20: 0.940},
                'state_q_static_mult': {15: 1.080, 20: 0.920},
                'state_q_dynamic_mult': {15: 1.030, 20: 0.960},
                'state_alpha_mult': {15: 1.015, 20: 0.975},
            },
            1: {
                'state_prior_diag_mult': {15: 1.030, 20: 0.960},
                'state_q_dynamic_mult': {15: 1.030, 20: 0.970},
                'state_alpha_mult': {15: 0.820, 20: 0.920},
            },
        },
        'post_ry_z_mult': 1.0009,
    },
    {
        'name': 'hybrid_micro_lever_pair',
        'description': 'Same new line but with a very small paired rx_y / ry_z lever guard, checking whether the selected-state repair can coexist with a micro lever confirmation.',
        'iter_patches': {
            0: {
                'state_prior_diag_mult': {15: 1.050, 20: 0.950},
                'state_q_static_mult': {15: 1.060, 20: 0.930},
                'state_q_dynamic_mult': {15: 1.025, 20: 0.970},
                'state_alpha_mult': {15: 1.012, 20: 0.978},
            },
            1: {
                'state_prior_diag_mult': {15: 1.020, 20: 0.970},
                'state_q_dynamic_mult': {15: 1.020, 20: 0.980},
                'state_alpha_mult': {15: 0.840, 20: 0.940},
            },
        },
        'post_rx_y_mult': 1.0003,
        'post_ry_z_mult': 1.0010,
    },
    {
        'name': 'commit_protect_121621_with_1520_repair',
        'description': 'Let 15/20 take the new asymmetric repair, but add a very small iter2 protection on 12/16/21 so xx/yy/Ka_xx stay closer to the R53 plateau.',
        'iter_patches': {
            0: {
                'state_prior_diag_mult': {15: 1.040, 20: 0.960},
                'state_alpha_mult': {15: 1.018, 20: 0.972},
            },
            1: {
                'state_prior_diag_mult': {15: 1.020, 20: 0.970},
                'state_alpha_mult': {12: 0.995, 15: 0.800, 16: 0.997, 20: 0.900, 21: 0.997},
            },
        },
        'post_ry_z_mult': 1.0007,
    },
]


def _param_specs(mod):
    clbt_truth = mod.get_default_clbt()
    dKg_truth = clbt_truth['Kg'] - mod.np.eye(3)
    dKa_truth = clbt_truth['Ka'] - mod.np.eye(3)
    params = [
        ('eb_x', clbt_truth['eb'][0], lambda c: -c['eb'][0]),
        ('eb_y', clbt_truth['eb'][1], lambda c: -c['eb'][1]),
        ('eb_z', clbt_truth['eb'][2], lambda c: -c['eb'][2]),
        ('db_x', clbt_truth['db'][0], lambda c: -c['db'][0]),
        ('db_y', clbt_truth['db'][1], lambda c: -c['db'][1]),
        ('db_z', clbt_truth['db'][2], lambda c: -c['db'][2]),
        ('dKg_xx', dKg_truth[0, 0], lambda c: -(c['Kg'] - mod.np.eye(3))[0, 0]),
        ('dKg_yx', dKg_truth[1, 0], lambda c: -(c['Kg'] - mod.np.eye(3))[1, 0]),
        ('dKg_zx', dKg_truth[2, 0], lambda c: -(c['Kg'] - mod.np.eye(3))[2, 0]),
        ('dKg_xy', dKg_truth[0, 1], lambda c: -(c['Kg'] - mod.np.eye(3))[0, 1]),
        ('dKg_yy', dKg_truth[1, 1], lambda c: -(c['Kg'] - mod.np.eye(3))[1, 1]),
        ('dKg_zy', dKg_truth[2, 1], lambda c: -(c['Kg'] - mod.np.eye(3))[2, 1]),
        ('dKg_xz', dKg_truth[0, 2], lambda c: -(c['Kg'] - mod.np.eye(3))[0, 2]),
        ('dKg_yz', dKg_truth[1, 2], lambda c: -(c['Kg'] - mod.np.eye(3))[1, 2]),
        ('dKg_zz', dKg_truth[2, 2], lambda c: -(c['Kg'] - mod.np.eye(3))[2, 2]),
        ('dKa_xx', dKa_truth[0, 0], lambda c: -(c['Ka'] - mod.np.eye(3))[0, 0]),
        ('dKa_xy', dKa_truth[0, 1], lambda c: -(c['Ka'] - mod.np.eye(3))[0, 1]),
        ('dKa_xz', dKa_truth[0, 2], lambda c: -(c['Ka'] - mod.np.eye(3))[0, 2]),
        ('dKa_yy', dKa_truth[1, 1], lambda c: -(c['Ka'] - mod.np.eye(3))[1, 1]),
        ('dKa_yz', dKa_truth[1, 2], lambda c: -(c['Ka'] - mod.np.eye(3))[1, 2]),
        ('dKa_zz', dKa_truth[2, 2], lambda c: -(c['Ka'] - mod.np.eye(3))[2, 2]),
        ('Ka2_x', clbt_truth['Ka2'][0], lambda c: -c['Ka2'][0]),
        ('Ka2_y', clbt_truth['Ka2'][1], lambda c: -c['Ka2'][1]),
        ('Ka2_z', clbt_truth['Ka2'][2], lambda c: -c['Ka2'][2]),
        ('rx_x', clbt_truth['rx'][0], lambda c: -c['rx'][0]),
        ('rx_y', clbt_truth['rx'][1], lambda c: -c['rx'][1]),
        ('rx_z', clbt_truth['rx'][2], lambda c: -c['rx'][2]),
        ('ry_x', clbt_truth['ry'][0], lambda c: -c['ry'][0]),
        ('ry_y', clbt_truth['ry'][1], lambda c: -c['ry'][1]),
        ('ry_z', clbt_truth['ry'][2], lambda c: -c['ry'][2]),
    ]
    return params


def _compute_metrics(source_mod, clbt):
    params = _param_specs(source_mod)
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

    pct_arr = source_mod.np.asarray(pct_values, dtype=float)
    focus = {k: param_errors[k]['pct_error'] for k in FOCUS_KEYS}
    lever = {k: param_errors[k]['pct_error'] for k in LEVER_KEYS}
    overall = {
        'mean_pct_error': float(source_mod.np.mean(pct_arr)),
        'median_pct_error': float(source_mod.np.median(pct_arr)),
        'max_pct_error': float(source_mod.np.max(pct_arr)),
    }
    return param_errors, focus, lever, overall


def _delta_block(curr: dict, ref: dict):
    return {k: float(curr[k] - ref[k]) for k in curr}


def _sorted_policy_patch(iter_patches: dict):
    out = {}
    for iter_idx, patch in sorted(iter_patches.items()):
        out[str(iter_idx + 1)] = {
            key: {str(k): float(v) for k, v in value.items()} if isinstance(value, dict) else value
            for key, value in patch.items()
        }
    return out


# --- patched helpers over R53 route ---

def _build_patched_method(method_mod, candidate: dict):
    original_configure = method_mod._configure_iteration_prior
    original_set_cov = method_mod._set_cov_schedule
    np = method_mod.load_module('tmp_probe_np_src', str(SOURCE_FILE)).np

    def patched_configure(mod, kf, policy):
        prior_diag = original_configure(mod, kf, policy)
        if policy.get('readout_only'):
            return prior_diag
        overrides = policy.get('state_prior_diag_mult', {})
        if overrides:
            for idx, mult in overrides.items():
                i = int(idx)
                kf['Pxk'][i, i] *= float(mult)
            prior_diag = mod.np.diag(kf['Pxk']).copy()
        return prior_diag

    def patched_set_cov_schedule(kf, base_q, base_r, policy, progress, is_static):
        original_set_cov(kf, base_q, base_r, policy, progress, is_static)
        if policy.get('readout_only'):
            return
        late = progress >= policy.get('late_release_frac', 2.0)
        q_map_name = 'state_q_static_mult' if is_static else 'state_q_dynamic_mult'
        for idx, mult in policy.get(q_map_name, {}).items():
            i = int(idx)
            kf['Qk'][i, i] *= float(mult)
        if late:
            for idx, mult in policy.get('state_q_late_mult', {}).items():
                i = int(idx)
                kf['Qk'][i, i] *= float(mult)

    def patched_apply_feedback(mod, base_clbt, kf, prior_diag, policy):
        out = method_mod._copy_clbt(mod, base_clbt)
        xk = mod.np.array(kf['xk'], dtype=float).copy()
        xfb = xk.copy()
        trust_log = {}

        alpha_mult = {int(k): float(v) for k, v in policy.get('state_alpha_mult', {}).items()}
        alpha_add = {int(k): float(v) for k, v in policy.get('state_alpha_add', {}).items()}
        alpha_clip = policy.get('state_alpha_clip', (0.70, 1.08))
        low_clip = float(alpha_clip[0])
        high_clip = float(alpha_clip[1])

        for idx in method_mod.SELECTED_SCALE_STATES:
            tc = method_mod._trust_components(mod, kf, prior_diag, idx, policy)
            base_alpha = float(tc['alpha'])
            adj_alpha = base_alpha * alpha_mult.get(idx, 1.0) + alpha_add.get(idx, 0.0)
            if adj_alpha < low_clip:
                adj_alpha = low_clip
            if adj_alpha > high_clip:
                adj_alpha = high_clip
            xfb[idx] = adj_alpha * xk[idx]
            tc['label'] = method_mod.SELECTED_STATE_LABELS[str(idx)]
            tc['base_alpha'] = base_alpha
            tc['alpha_mult'] = float(alpha_mult.get(idx, 1.0))
            tc['alpha_add'] = float(alpha_add.get(idx, 0.0))
            tc['alpha'] = float(adj_alpha)
            tc['x_feedback'] = float(xfb[idx])
            trust_log[str(idx)] = tc

        for idx in method_mod.OTHER_SCALE_STATES:
            xfb[idx] = policy['other_scale_alpha'] * xk[idx]

        dKg = xfb[12:21].reshape(3, 3).T
        out['Kg'] = (mod.np.eye(3) - dKg) @ out['Kg']

        dKa = mod.Ka_from_upper(xfb[21:27])
        out['Ka'] = (mod.np.eye(3) - dKa) @ out['Ka']

        out['Ka2'] = out['Ka2'] + policy['ka2_alpha'] * xfb[27:30]
        out['eb'] = out['eb'] + xfb[6:9]
        out['db'] = out['db'] + xfb[9:12]
        out['rx'] = out['rx'] + policy['lever_alpha'] * xfb[30:33]
        out['ry'] = out['ry'] + policy['lever_alpha'] * xfb[33:36]
        out['eb'] = out['eb'] + policy['markov_alpha'] * xfb[36:39]
        out['db'] = out['db'] + policy['markov_alpha'] * xfb[39:42]

        post_rx_y_mult = float(policy.get('post_rx_y_mult', 1.0))
        post_ry_z_mult = float(policy.get('post_ry_z_mult', 1.0))
        rx_before = float(out['rx'][1])
        ry_before = float(out['ry'][2])
        out['rx'][1] *= post_rx_y_mult
        out['ry'][2] *= post_ry_z_mult

        return out, {
            'policy_name': policy['name'],
            'selected_state_labels': method_mod.SELECTED_STATE_LABELS,
            'selected_trust': trust_log,
            'other_scale_alpha': float(policy['other_scale_alpha']),
            'ka2_alpha': float(policy['ka2_alpha']),
            'lever_alpha': float(policy['lever_alpha']),
            'markov_alpha': float(policy['markov_alpha']),
            'state_prior_diag_mult': {str(k): float(v) for k, v in policy.get('state_prior_diag_mult', {}).items()},
            'state_q_static_mult': {str(k): float(v) for k, v in policy.get('state_q_static_mult', {}).items()},
            'state_q_dynamic_mult': {str(k): float(v) for k, v in policy.get('state_q_dynamic_mult', {}).items()},
            'state_q_late_mult': {str(k): float(v) for k, v in policy.get('state_q_late_mult', {}).items()},
            'state_alpha_mult': {str(k): float(v) for k, v in alpha_mult.items()},
            'state_alpha_add': {str(k): float(v) for k, v in alpha_add.items()},
            'post_rx_y_mult': post_rx_y_mult,
            'post_ry_z_mult': post_ry_z_mult,
            'post_rx_y_before': rx_before,
            'post_rx_y_after': float(out['rx'][1]),
            'post_ry_z_before': ry_before,
            'post_ry_z_after': float(out['ry'][2]),
        }

    method_mod._configure_iteration_prior = patched_configure
    method_mod._set_cov_schedule = patched_set_cov_schedule
    method_mod._apply_trust_internalized_feedback = patched_apply_feedback

    patched_policies = copy.deepcopy(method_mod.ITERATION_POLICIES)
    for iter_idx, patch in candidate.get('iter_patches', {}).items():
        patched_policies[iter_idx].update(copy.deepcopy(patch))
    for policy in patched_policies:
        if policy.get('readout_only'):
            continue
        if candidate.get('post_rx_y_mult') is not None:
            policy['post_rx_y_mult'] = float(candidate['post_rx_y_mult'])
        if candidate.get('post_ry_z_mult') is not None:
            policy['post_ry_z_mult'] = float(candidate['post_ry_z_mult'])

    method_mod.ITERATION_POLICIES = patched_policies
    method_mod.METHOD = f"42-state GM1 round55 probe {candidate['name']}"
    method_mod.VARIANT = f"42state_gm1_round55_probe_{candidate['name']}"
    return method_mod


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    source_mod = load_module('markov_pruned_source_for_round55_probe', str(SOURCE_FILE))
    r53_payload = json.loads(R53_JSON.read_text(encoding='utf-8'))

    out = {
        'baseline_r53': {
            'focus': r53_payload['focus_scale_pct'],
            'lever': r53_payload['lever_guard_pct'],
            'overall': r53_payload['overall'],
        },
        'target_state_labels': TARGET_STATE_LABELS,
        'candidates': {},
    }

    for idx, candidate in enumerate(CANDIDATES, start=1):
        method_mod = load_module(f'markov_method_round55_probe_{idx}', str(R53_METHOD_FILE))
        method_mod = _build_patched_method(method_mod, candidate)
        result = method_mod.run_method()
        clbt = result[0]
        extra = result[4] if len(result) >= 5 else {}
        _, focus, lever, overall = _compute_metrics(source_mod, clbt)

        probe_info = {
            'description': candidate['description'],
            'policy_patch': _sorted_policy_patch(candidate.get('iter_patches', {})),
            'post_rx_y_mult': float(candidate.get('post_rx_y_mult', 1.0)),
            'post_ry_z_mult': float(candidate.get('post_ry_z_mult', 1.0)),
            'focus': focus,
            'lever': lever,
            'overall': overall,
            'delta_vs_r53': {
                **_delta_block(focus, r53_payload['focus_scale_pct']),
                **_delta_block(lever, r53_payload['lever_guard_pct']),
                **_delta_block(overall, r53_payload['overall']),
            },
            'key_newline_delta': {
                'dKg_xy': float(focus['dKg_xy'] - r53_payload['focus_scale_pct']['dKg_xy']),
                'dKg_zz': float(focus['dKg_zz'] - r53_payload['focus_scale_pct']['dKg_zz']),
                'ry_z': float(lever['ry_z'] - r53_payload['lever_guard_pct']['ry_z']),
                'protect_dKg_xx': float(focus['dKg_xx'] - r53_payload['focus_scale_pct']['dKg_xx']),
                'protect_dKg_yy': float(focus['dKg_yy'] - r53_payload['focus_scale_pct']['dKg_yy']),
                'protect_dKa_xx': float(focus['dKa_xx'] - r53_payload['focus_scale_pct']['dKa_xx']),
                'protect_mean': float(overall['mean_pct_error'] - r53_payload['overall']['mean_pct_error']),
            },
            'extra': {
                'schedule_log': extra.get('schedule_log'),
                'feedback_log': extra.get('feedback_log'),
            },
        }
        out['candidates'][candidate['name']] = probe_info
        print(candidate['name'], json.dumps({
            'focus': focus,
            'lever': lever,
            'overall': overall,
            'delta_vs_r53': probe_info['delta_vs_r53'],
        }, ensure_ascii=False))

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {OUTPUT_JSON}')
    print('__RESULT_JSON__=' + json.dumps({'output_json': str(OUTPUT_JSON)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
