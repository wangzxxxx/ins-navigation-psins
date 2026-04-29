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
ROUND54_FIRST_PROBE_JSON = RESULTS_DIR / 'round54_probe_summary.json'
OUTPUT_JSON = RESULTS_DIR / 'round54_second_probe_summary.json'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from common_markov import load_module


FOCUS_KEYS = ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx']
LEVER_KEYS = ['rx_y', 'ry_z']


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
    out = {}
    for key, value in curr.items():
        out[key] = float(value - ref[key])
    return out


def _load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _apply_candidate(policy_set, candidate_name: str):
    p = copy.deepcopy(policy_set)
    iter2 = p[1]
    if candidate_name == 'soft_late_commit_yyzz_qpush':
        iter2['selected_q_dynamic_scale'] = 1.005
        iter2['selected_q_late_mult'] = 1.292
        iter2['late_release_frac'] = 0.575
    elif candidate_name == 'soft_late_commit_yyzz_uniform':
        iter2['selected_q_dynamic_scale'] = 1.005
        iter2['selected_q_late_mult'] = 1.292
        iter2['late_release_frac'] = 0.575
        iter2['selected_alpha_floor'] = 0.962
        iter2['selected_alpha_span'] = 0.118
    elif candidate_name == 'soft_late_commit_yyzz_staticlate':
        iter2['selected_q_static_scale'] = 0.784
        iter2['selected_q_late_mult'] = 1.288
        iter2['late_release_frac'] = 0.577
    else:
        raise KeyError(candidate_name)
    return p


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    source_mod = load_module('markov_pruned_source_for_round54_second_probe', str(SOURCE_FILE))
    r53_payload = _load_json(R53_JSON)
    round54_first = _load_json(ROUND54_FIRST_PROBE_JSON)
    soft_ref = round54_first['soft_late_commit']

    candidates = [
        'soft_late_commit_yyzz_qpush',
        'soft_late_commit_yyzz_uniform',
        'soft_late_commit_yyzz_staticlate',
    ]

    out = {
        'baseline_r53': {
            'focus': r53_payload['focus_scale_pct'],
            'lever': r53_payload['lever_guard_pct'],
            'overall': r53_payload['overall'],
        },
        'soft_late_commit_reference': soft_ref,
        'candidates': {},
    }

    for idx, name in enumerate(candidates, start=1):
        method_mod = load_module(f'markov_method_round54_second_probe_{idx}', str(R53_METHOD_FILE))
        patched = _apply_candidate(method_mod.ITERATION_POLICIES, name)
        method_mod.ITERATION_POLICIES = patched
        method_mod.METHOD = f'42-state GM1 round54 second probe {name}'
        method_mod.VARIANT = f'42state_gm1_round54_second_probe_{name}'
        result = method_mod.run_method()
        clbt = result[0]
        extra = result[4] if len(result) >= 5 else {}
        _, focus, lever, overall = _compute_metrics(source_mod, clbt)

        probe_info = {
            'policy_patch': {
                'iter2': {
                    k: patched[1][k]
                    for k in [
                        'selected_q_static_scale', 'selected_q_dynamic_scale', 'selected_q_late_mult',
                        'late_release_frac', 'selected_alpha_floor', 'selected_alpha_span',
                    ]
                }
            },
            'focus': focus,
            'lever': lever,
            'overall': overall,
            'delta_vs_r53': {
                **_delta_block(focus, r53_payload['focus_scale_pct']),
                **_delta_block(lever, r53_payload['lever_guard_pct']),
                **_delta_block(overall, r53_payload['overall']),
            },
            'delta_vs_soft_late_commit': {
                **_delta_block(focus, soft_ref['focus']),
                **_delta_block(lever, soft_ref['lever']),
                **_delta_block(overall, soft_ref['overall']),
            },
            'extra': {
                'schedule_log': extra.get('schedule_log'),
                'feedback_log': extra.get('feedback_log'),
            },
        }
        out['candidates'][name] = probe_info
        print(name, json.dumps({
            'focus': focus,
            'lever': lever,
            'overall': overall,
            'delta_vs_r53': probe_info['delta_vs_r53'],
            'delta_vs_soft_late_commit': probe_info['delta_vs_soft_late_commit'],
        }, ensure_ascii=False))

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {OUTPUT_JSON}')
    print('__RESULT_JSON__=' + json.dumps({'output_json': str(OUTPUT_JSON)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
