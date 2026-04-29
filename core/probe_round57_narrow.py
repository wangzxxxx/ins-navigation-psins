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
R55_JSON = RESULTS_DIR / 'R55_42state_gm1_round55_internalized_xyzz_targeted_repair_param_errors.json'
R56_JSON = RESULTS_DIR / 'R56_42state_gm1_round56_internalized_xyzz_yyka_micro_guard_param_errors.json'
OUTPUT_JSON = RESULTS_DIR / 'round57_narrow_probe_summary.json'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(ROOT / 'psins_method_bench' / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'psins_method_bench' / 'scripts'))

from common_markov import load_module
from probe_round55_newline import _build_patched_method
from probe_round56_narrow import ROUND56_CANDIDATES, _merge_candidate, _compute_metrics


FOCUS_KEYS = ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx']
LEVER_KEYS = ['rx_y', 'ry_z']
BASELINE_NAME = 'commit_yyka_micro_up_pair_stronger'


ROUND57_CANDIDATES = [
    {
        'name': 'yyka_backoff_tiny',
        'description': 'Slightly reclaim the iter2 16/21 micro-up while leaving the lever pair untouched.',
        'iter_patches': {
            1: {
                'state_alpha_mult': {15: 0.760, 16: 1.007, 20: 0.900, 21: 1.005},
            },
        },
        'post_rx_y_mult': 1.00045,
        'post_ry_z_mult': 1.00110,
    },
    {
        'name': 'yyka_backoff_soft',
        'description': 'One more notch of 16/21 backoff to test whether xx/max rebounds without giving back yy/Ka_xx too much.',
        'iter_patches': {
            1: {
                'state_alpha_mult': {15: 0.760, 16: 1.006, 20: 0.900, 21: 1.0045},
            },
        },
        'post_rx_y_mult': 1.00045,
        'post_ry_z_mult': 1.00110,
    },
    {
        'name': 'state12_guard_keep_yyka',
        'description': 'Keep the R56 16/21 repair, but add a tiny iter2 state12 lift to directly suppress the max holder.',
        'iter_patches': {
            1: {
                'state_alpha_mult': {12: 1.0015, 15: 0.760, 16: 1.008, 20: 0.900, 21: 1.006},
            },
        },
        'post_rx_y_mult': 1.00045,
        'post_ry_z_mult': 1.00110,
    },
    {
        'name': 'state12_guard_tiny_backoff',
        'description': 'Combine a very small state12 lift with a half-step 16/21 reclaim so xx/max gets help while yy/Ka_xx stays near R56.',
        'iter_patches': {
            1: {
                'state_alpha_mult': {12: 1.0015, 15: 0.760, 16: 1.007, 20: 0.900, 21: 1.005},
            },
        },
        'post_rx_y_mult': 1.00045,
        'post_ry_z_mult': 1.00110,
    },
    {
        'name': 'state12_guard_keep_yyka_lever_retrim',
        'description': 'Same state12 micro guard, with a barely smaller post lever pair to see if mean can come back without hurting rx_y/ry_z too much.',
        'iter_patches': {
            1: {
                'state_alpha_mult': {12: 1.0015, 15: 0.760, 16: 1.008, 20: 0.900, 21: 1.006},
            },
        },
        'post_rx_y_mult': 1.00042,
        'post_ry_z_mult': 1.00105,
    },
]


def _load_round56_candidate():
    for candidate in ROUND56_CANDIDATES:
        if candidate['name'] == BASELINE_NAME:
            return _merge_candidate(candidate)
    raise KeyError(BASELINE_NAME)


def _merge_round57_candidate(extra_candidate: dict):
    merged = copy.deepcopy(_load_round56_candidate())
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
    merged['round57_extra_patch'] = extra_candidate.get('iter_patches', {})
    return merged


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
    source_mod = load_module('markov_pruned_source_for_round57_probe', str(SOURCE_FILE))
    r55_payload = json.loads(R55_JSON.read_text(encoding='utf-8'))
    r56_payload = json.loads(R56_JSON.read_text(encoding='utf-8'))

    out = {
        'baseline_r56': {
            'focus': r56_payload['focus_scale_pct'],
            'lever': r56_payload['lever_guard_pct'],
            'overall': r56_payload['overall'],
        },
        'baseline_r55': {
            'focus': r55_payload['focus_scale_pct'],
            'lever': r55_payload['lever_guard_pct'],
            'overall': r55_payload['overall'],
        },
        'candidates': {},
    }

    for idx, extra_candidate in enumerate(ROUND57_CANDIDATES, start=1):
        merged_candidate = _merge_round57_candidate(extra_candidate)
        method_mod = load_module(f'markov_method_round57_probe_{idx}', str(R53_METHOD_FILE))
        method_mod = _build_patched_method(method_mod, merged_candidate)
        method_mod.METHOD = f"42-state GM1 round57 probe {merged_candidate['name']}"
        method_mod.VARIANT = f"42state_gm1_round57_probe_{merged_candidate['name']}"

        result = method_mod.run_method()
        clbt = result[0]
        extra = result[4] if len(result) >= 5 else {}
        _, focus, lever, overall = _compute_metrics(source_mod, clbt)

        probe_info = {
            'description': merged_candidate['description'],
            'base_round56_candidate': BASELINE_NAME,
            'policy_patch': _sorted_policy_patch(merged_candidate.get('iter_patches', {})),
            'round57_extra_patch': _sorted_policy_patch(merged_candidate.get('round57_extra_patch', {})),
            'post_rx_y_mult': float(merged_candidate.get('post_rx_y_mult', 1.0)),
            'post_ry_z_mult': float(merged_candidate.get('post_ry_z_mult', 1.0)),
            'focus': focus,
            'lever': lever,
            'overall': overall,
            'delta_vs_r56': {
                **_delta_block(focus, r56_payload['focus_scale_pct']),
                **_delta_block(lever, r56_payload['lever_guard_pct']),
                **_delta_block(overall, r56_payload['overall']),
            },
            'delta_vs_r55': {
                **_delta_block(focus, r55_payload['focus_scale_pct']),
                **_delta_block(lever, r55_payload['lever_guard_pct']),
                **_delta_block(overall, r55_payload['overall']),
            },
            'key_round57_delta': {
                'protect_dKg_xy': float(focus['dKg_xy'] - r56_payload['focus_scale_pct']['dKg_xy']),
                'protect_dKg_zz': float(focus['dKg_zz'] - r56_payload['focus_scale_pct']['dKg_zz']),
                'protect_dKg_yy': float(focus['dKg_yy'] - r56_payload['focus_scale_pct']['dKg_yy']),
                'protect_dKa_xx': float(focus['dKa_xx'] - r56_payload['focus_scale_pct']['dKa_xx']),
                'repair_dKg_xx': float(focus['dKg_xx'] - r56_payload['focus_scale_pct']['dKg_xx']),
                'protect_rx_y': float(lever['rx_y'] - r56_payload['lever_guard_pct']['rx_y']),
                'protect_ry_z': float(lever['ry_z'] - r56_payload['lever_guard_pct']['ry_z']),
                'repair_mean': float(overall['mean_pct_error'] - r56_payload['overall']['mean_pct_error']),
                'repair_max': float(overall['max_pct_error'] - r56_payload['overall']['max_pct_error']),
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
            'delta_vs_r56': probe_info['delta_vs_r56'],
        }, ensure_ascii=False))

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {OUTPUT_JSON}')
    print('__RESULT_JSON__=' + json.dumps({'output_json': str(OUTPUT_JSON)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
