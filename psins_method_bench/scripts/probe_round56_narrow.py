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
R53_JSON = RESULTS_DIR / 'R53_42state_gm1_round53_internalized_trustcov_release_param_errors.json'
R55_JSON = RESULTS_DIR / 'R55_42state_gm1_round55_internalized_xyzz_targeted_repair_param_errors.json'
OUTPUT_JSON = RESULTS_DIR / 'round56_narrow_probe_summary.json'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(ROOT / 'psins_method_bench' / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'psins_method_bench' / 'scripts'))

from common_markov import load_module
from probe_round55_newline import CANDIDATES as ROUND55_CANDIDATES, _build_patched_method


FOCUS_KEYS = ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx']
LEVER_KEYS = ['rx_y', 'ry_z']
BASELINE_NAME = 'alpha_split_xy_up_zz_damp'


ROUND56_CANDIDATES = [
    {
        'name': 'lever_micro_pair_guard',
        'description': 'Pure R55 preservation check plus a very small paired rx_y/ry_z post guard.',
        'post_rx_y_mult': 1.00035,
        'post_ry_z_mult': 1.00095,
    },
    {
        'name': 'commit_yyka_micro_up_pair',
        'description': 'Keep R55 15/20 line untouched, but add a tiny iter2 16/21 upward repair plus the same micro lever pair.',
        'iter_patches': {
            1: {
                'state_alpha_mult': {15: 0.760, 16: 1.004, 20: 0.900, 21: 1.003},
            },
        },
        'post_rx_y_mult': 1.00035,
        'post_ry_z_mult': 1.00095,
    },
    {
        'name': 'dualpass_yyka_micro_up_pair',
        'description': 'Same idea, but also give 16/21 a near-invisible iter1 nudge so the repair is distributed across both passes.',
        'iter_patches': {
            0: {
                'state_alpha_mult': {15: 1.020, 16: 1.002, 20: 0.970, 21: 1.0015},
            },
            1: {
                'state_alpha_mult': {15: 0.760, 16: 1.004, 20: 0.900, 21: 1.003},
            },
        },
        'post_rx_y_mult': 1.00035,
        'post_ry_z_mult': 1.00095,
    },
    {
        'name': 'commit_yyka_micro_up_pair_stronger',
        'description': 'Slightly stronger version to test the edge of useful 16/21 repair while still staying narrow.',
        'iter_patches': {
            1: {
                'state_alpha_mult': {15: 0.760, 16: 1.008, 20: 0.900, 21: 1.006},
            },
        },
        'post_rx_y_mult': 1.00045,
        'post_ry_z_mult': 1.00110,
    },
]


def _load_baseline_candidate():
    for candidate in ROUND55_CANDIDATES:
        if candidate['name'] == BASELINE_NAME:
            return copy.deepcopy(candidate)
    raise KeyError(BASELINE_NAME)


def _merge_candidate(extra_candidate: dict):
    merged = _load_baseline_candidate()
    merged['name'] = extra_candidate['name']
    merged['description'] = extra_candidate['description']
    merged['post_rx_y_mult'] = float(extra_candidate.get('post_rx_y_mult', merged.get('post_rx_y_mult', 1.0)))
    merged['post_ry_z_mult'] = float(extra_candidate.get('post_ry_z_mult', merged.get('post_ry_z_mult', 1.0)))

    merged_patches = copy.deepcopy(merged.get('iter_patches', {}))
    for iter_idx, patch in extra_candidate.get('iter_patches', {}).items():
        dst = merged_patches.setdefault(iter_idx, {})
        for key, value in patch.items():
            if isinstance(value, dict):
                current = copy.deepcopy(dst.get(key, {}))
                current.update(copy.deepcopy(value))
                dst[key] = current
            else:
                dst[key] = copy.deepcopy(value)
    merged['iter_patches'] = merged_patches
    merged['round56_extra_patch'] = extra_candidate.get('iter_patches', {})
    return merged


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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    source_mod = load_module('markov_pruned_source_for_round56_probe', str(SOURCE_FILE))
    r53_payload = json.loads(R53_JSON.read_text(encoding='utf-8'))
    r55_payload = json.loads(R55_JSON.read_text(encoding='utf-8'))

    out = {
        'baseline_r55': {
            'focus': r55_payload['focus_scale_pct'],
            'lever': r55_payload['lever_guard_pct'],
            'overall': r55_payload['overall'],
        },
        'baseline_r53': {
            'focus': r53_payload['focus_scale_pct'],
            'lever': r53_payload['lever_guard_pct'],
            'overall': r53_payload['overall'],
        },
        'candidates': {},
    }

    for idx, extra_candidate in enumerate(ROUND56_CANDIDATES, start=1):
        merged_candidate = _merge_candidate(extra_candidate)
        method_mod = load_module(f'markov_method_round56_probe_{idx}', str(METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'))
        method_mod = _build_patched_method(method_mod, merged_candidate)
        method_mod.METHOD = f"42-state GM1 round56 probe {merged_candidate['name']}"
        method_mod.VARIANT = f"42state_gm1_round56_probe_{merged_candidate['name']}"

        result = method_mod.run_method()
        clbt = result[0]
        extra = result[4] if len(result) >= 5 else {}
        _, focus, lever, overall = _compute_metrics(source_mod, clbt)

        probe_info = {
            'description': merged_candidate['description'],
            'base_round55_candidate': BASELINE_NAME,
            'policy_patch': _sorted_policy_patch(merged_candidate.get('iter_patches', {})),
            'round56_extra_patch': _sorted_policy_patch(merged_candidate.get('round56_extra_patch', {})),
            'post_rx_y_mult': float(merged_candidate.get('post_rx_y_mult', 1.0)),
            'post_ry_z_mult': float(merged_candidate.get('post_ry_z_mult', 1.0)),
            'focus': focus,
            'lever': lever,
            'overall': overall,
            'delta_vs_r55': {
                **_delta_block(focus, r55_payload['focus_scale_pct']),
                **_delta_block(lever, r55_payload['lever_guard_pct']),
                **_delta_block(overall, r55_payload['overall']),
            },
            'delta_vs_r53': {
                **_delta_block(focus, r53_payload['focus_scale_pct']),
                **_delta_block(lever, r53_payload['lever_guard_pct']),
                **_delta_block(overall, r53_payload['overall']),
            },
            'key_round56_delta': {
                'protect_dKg_xx': float(focus['dKg_xx'] - r55_payload['focus_scale_pct']['dKg_xx']),
                'protect_dKg_xy': float(focus['dKg_xy'] - r55_payload['focus_scale_pct']['dKg_xy']),
                'repair_dKg_yy': float(focus['dKg_yy'] - r55_payload['focus_scale_pct']['dKg_yy']),
                'protect_dKg_zz': float(focus['dKg_zz'] - r55_payload['focus_scale_pct']['dKg_zz']),
                'repair_dKa_xx': float(focus['dKa_xx'] - r55_payload['focus_scale_pct']['dKa_xx']),
                'repair_rx_y': float(lever['rx_y'] - r55_payload['lever_guard_pct']['rx_y']),
                'repair_ry_z': float(lever['ry_z'] - r55_payload['lever_guard_pct']['ry_z']),
                'protect_mean': float(overall['mean_pct_error'] - r55_payload['overall']['mean_pct_error']),
                'protect_max': float(overall['max_pct_error'] - r55_payload['overall']['max_pct_error']),
            },
            'extra': {
                'schedule_log': extra.get('schedule_log'),
                'feedback_log': extra.get('feedback_log'),
            },
        }
        out['candidates'][merged_candidate['name']] = probe_info
        print(merged_candidate['name'], json.dumps({
            'focus': focus,
            'lever': lever,
            'overall': overall,
            'delta_vs_r55': probe_info['delta_vs_r55'],
        }, ensure_ascii=False))

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {OUTPUT_JSON}')
    print('__RESULT_JSON__=' + json.dumps({'output_json': str(OUTPUT_JSON)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
