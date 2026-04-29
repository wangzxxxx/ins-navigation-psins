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
R57_JSON = RESULTS_DIR / 'R57_42state_gm1_round57_internalized_xyzz_yyka_state12_guard_param_errors.json'
OUTPUT_JSON = RESULTS_DIR / 'round58_llm_guided_probe_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'round58_llm_guided_candidates.json'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(ROOT / 'psins_method_bench' / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'psins_method_bench' / 'scripts'))

from common_markov import load_module
from probe_round55_newline import _build_patched_method
from probe_round57_narrow import _compute_metrics, _merge_round57_candidate, ROUND57_CANDIDATES


FOCUS_KEYS = ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx']
LEVER_KEYS = ['rx_y', 'ry_z']
BASELINE_NAME = 'state12_guard_keep_yyka'


ROUND58_LLM_CANDIDATES = [
    {
        'name': 'llm_alpha12_plus',
        'description': 'Extend the only proven R57 win direction: a barely-stronger iter2 state12 alpha lift, touching nothing else.',
        'rationale': 'R57 improved dKg_xx/max with a 12-only alpha micro-guard; first LLM step is to test an even smaller same-direction nudge before trying orthogonal knobs.',
        'iter_patches': {
            1: {
                'state_alpha_mult': {12: 1.0018},
            },
        },
        'post_rx_y_mult': 1.00045,
        'post_ry_z_mult': 1.00110,
    },
    {
        'name': 'llm_prior12_micro',
        'description': 'Keep the R57 alpha/lever stack and add only a tiny iter2 state12 prior-diagonal opening.',
        'rationale': 'If state12 is still slightly under-moving, a microscopic prior widening may help dKg_xx without perturbing the preserved 15/16/20/21 route.',
        'iter_patches': {
            1: {
                'state_prior_diag_mult': {12: 1.006},
            },
        },
        'post_rx_y_mult': 1.00045,
        'post_ry_z_mult': 1.00110,
    },
    {
        'name': 'llm_q12_dynlate_micro',
        'description': 'Keep R57 intact, but add a nearly invisible iter2 state12 dynamic/late Q lift.',
        'rationale': 'If the remaining xx drift is mostly in the commit/late phase, a tiny Q increase might unlock state12 correction more selectively than more alpha.',
        'iter_patches': {
            1: {
                'state_q_dynamic_mult': {12: 1.004},
                'state_q_late_mult': {12: 1.008},
            },
        },
        'post_rx_y_mult': 1.00045,
        'post_ry_z_mult': 1.00110,
    },
    {
        'name': 'llm_alpha12_plus_lever_plus',
        'description': 'Pair the slightly stronger state12 alpha lift with an equally tiny lever confirmation.',
        'rationale': 'Tests whether xx/max and mean can move together by combining the proven state12 path with a near-no-op rx_y/ry_z confirmation, still within narrow bounds.',
        'iter_patches': {
            1: {
                'state_alpha_mult': {12: 1.0018},
            },
        },
        'post_rx_y_mult': 1.00047,
        'post_ry_z_mult': 1.00113,
    },
]


def _load_round57_candidate():
    for candidate in ROUND57_CANDIDATES:
        if candidate['name'] == BASELINE_NAME:
            return _merge_round57_candidate(candidate)
    raise KeyError(BASELINE_NAME)


def _merge_round58_candidate(extra_candidate: dict):
    merged = copy.deepcopy(_load_round57_candidate())
    merged['name'] = extra_candidate['name']
    merged['description'] = extra_candidate['description']
    merged['rationale'] = extra_candidate['rationale']
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
    merged['round58_llm_extra_patch'] = extra_candidate.get('iter_patches', {})
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
    source_mod = load_module('markov_pruned_source_for_round58_probe', str(SOURCE_FILE))
    r57_payload = json.loads(R57_JSON.read_text(encoding='utf-8'))

    candidate_dump = {
        'baseline': BASELINE_NAME,
        'allowed_knobs': [
            'state_alpha_mult',
            'state_prior_diag_mult',
            'state_q_static_mult',
            'state_q_dynamic_mult',
            'state_q_late_mult',
            'post_rx_y_mult',
            'post_ry_z_mult',
        ],
        'candidates': ROUND58_LLM_CANDIDATES,
    }
    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    out = {
        'baseline_r57': {
            'focus': r57_payload['focus_scale_pct'],
            'lever': r57_payload['lever_guard_pct'],
            'overall': r57_payload['overall'],
        },
        'baseline_name': BASELINE_NAME,
        'allowed_knobs': candidate_dump['allowed_knobs'],
        'candidates': {},
    }

    for idx, extra_candidate in enumerate(ROUND58_LLM_CANDIDATES, start=1):
        merged_candidate = _merge_round58_candidate(extra_candidate)
        method_mod = load_module(f'markov_method_round58_llm_guided_probe_{idx}', str(R53_METHOD_FILE))
        method_mod = _build_patched_method(method_mod, merged_candidate)
        method_mod.METHOD = f"42-state GM1 round58 llm-guided probe {merged_candidate['name']}"
        method_mod.VARIANT = f"42state_gm1_round58_llm_guided_probe_{merged_candidate['name']}"

        result = method_mod.run_method()
        clbt = result[0]
        extra = result[4] if len(result) >= 5 else {}
        _, focus, lever, overall = _compute_metrics(source_mod, clbt)

        probe_info = {
            'description': merged_candidate['description'],
            'rationale': merged_candidate['rationale'],
            'base_round57_candidate': BASELINE_NAME,
            'policy_patch': _sorted_policy_patch(merged_candidate.get('iter_patches', {})),
            'round58_llm_extra_patch': _sorted_policy_patch(merged_candidate.get('round58_llm_extra_patch', {})),
            'post_rx_y_mult': float(merged_candidate.get('post_rx_y_mult', 1.0)),
            'post_ry_z_mult': float(merged_candidate.get('post_ry_z_mult', 1.0)),
            'focus': focus,
            'lever': lever,
            'overall': overall,
            'delta_vs_r57': {
                **_delta_block(focus, r57_payload['focus_scale_pct']),
                **_delta_block(lever, r57_payload['lever_guard_pct']),
                **_delta_block(overall, r57_payload['overall']),
            },
            'key_round58_delta': {
                'repair_dKg_xx': float(focus['dKg_xx'] - r57_payload['focus_scale_pct']['dKg_xx']),
                'protect_dKg_xy': float(focus['dKg_xy'] - r57_payload['focus_scale_pct']['dKg_xy']),
                'protect_dKg_yy': float(focus['dKg_yy'] - r57_payload['focus_scale_pct']['dKg_yy']),
                'protect_dKg_zz': float(focus['dKg_zz'] - r57_payload['focus_scale_pct']['dKg_zz']),
                'protect_dKa_xx': float(focus['dKa_xx'] - r57_payload['focus_scale_pct']['dKa_xx']),
                'protect_rx_y': float(lever['rx_y'] - r57_payload['lever_guard_pct']['rx_y']),
                'protect_ry_z': float(lever['ry_z'] - r57_payload['lever_guard_pct']['ry_z']),
                'repair_mean': float(overall['mean_pct_error'] - r57_payload['overall']['mean_pct_error']),
                'repair_max': float(overall['max_pct_error'] - r57_payload['overall']['max_pct_error']),
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
            'delta_vs_r57': probe_info['delta_vs_r57'],
        }, ensure_ascii=False))

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {OUTPUT_JSON}')
    print('__RESULT_JSON__=' + json.dumps({
        'output_json': str(OUTPUT_JSON),
        'candidate_json': str(CANDIDATE_JSON),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
