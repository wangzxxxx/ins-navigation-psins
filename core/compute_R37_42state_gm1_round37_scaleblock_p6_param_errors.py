from __future__ import annotations

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
METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round37_scaleblock_p6.py'
OUTPUT_JSON = RESULTS_DIR / 'R37_42state_gm1_round37_scaleblock_p6_param_errors.json'
VARIANT = '42state_gm1_round37_scaleblock_p6'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))

from common_markov import load_module


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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    source_mod = load_module('markov_pruned_source_for_r37', str(SOURCE_FILE))
    method_mod = load_module('markov_method_r37_scaleblock_p6', str(METHOD_FILE))

    method_res = method_mod.run_method()
    clbt = method_res[0]
    extra = method_res[4] if len(method_res) >= 5 else {}

    _, params = _param_specs(source_mod)
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
        'mean_pct_error': float(source_mod.np.mean(pct_arr)),
        'median_pct_error': float(source_mod.np.median(pct_arr)),
        'max_pct_error': float(source_mod.np.max(pct_arr)),
    }

    payload = {
        'variant': VARIANT,
        'method_file': str(METHOD_FILE),
        'source_file': str(SOURCE_FILE),
        'param_order': [name for name, _, _ in params],
        'param_errors': param_errors,
        'focus_scale_pct': focus_scale_pct,
        'lever_guard_pct': lever_guard_pct,
        'overall': overall,
        'extra': extra,
    }

    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote param errors to {OUTPUT_JSON}')
    print('__RESULT_JSON__=' + json.dumps({
        'output_json': str(OUTPUT_JSON),
        'focus_scale_pct': focus_scale_pct,
        'lever_guard_pct': lever_guard_pct,
        'overall': overall,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
