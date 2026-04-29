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
R58_JSON = RESULTS_DIR / 'R58_42state_gm1_round58_llm_guided_alpha12_lever_plus_param_errors.json'
R59_JSON = RESULTS_DIR / 'R59_42state_gm1_round59_h_scd_scale_once_commit_param_errors.json'
R60_JSON = RESULTS_DIR / 'R60_42state_gm1_round60_h_scd_state20_ryz_micro_commit_param_errors.json'
OUTPUT_JSON = RESULTS_DIR / 'round61_hybrid_micro_probe_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'round61_hybrid_micro_candidates.json'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(ROOT / 'psins_method_bench' / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'psins_method_bench' / 'scripts'))

from common_markov import load_module
from probe_round55_newline import _build_patched_method
from probe_round56_narrow import _compute_metrics
from probe_round59_h_scd_hybrid import _run_internalized_hybrid_scd
from probe_round60_conservative import ROUND60_CONSERVATIVE_CANDIDATES, _merge_round60_candidate


ROUND60_BASE_NAME = 'scale_once_a0999_s20tight_0899_ryz00116'

ROUND61_CANDIDATES = [
    {
        'name': 'r61_s20_08988_ryz00116',
        'description': 'Keep exact R60 route, only tighten iter2 state20 from 0.899 to 0.8988.',
        'rationale': 'Most conservative zz-only follow-up: preserve the R60 lever repair and nudge only the dKg_zz line.',
        'iter_patches': {
            1: {
                'state_alpha_mult': {20: 0.8988},
            },
        },
    },
    {
        'name': 'r61_s20_08988_ryz00118',
        'description': 'Same 0.8988 state20 tighten, plus a tiny ry_z confirmation bump.',
        'rationale': 'Pair the slight zz tighten with the smallest additional ry_z repair that still stays on the R60 route.',
        'iter_patches': {
            1: {
                'state_alpha_mult': {20: 0.8988},
            },
        },
        'post_ry_z_mult': 1.00118,
    },
    {
        'name': 'r61_s20_08986_ryz00117',
        'description': 'Slightly stronger state20 tighten with a mid-step ry_z confirmation.',
        'rationale': 'Checks whether zz still improves monotonically before the R60-protected terms begin to move.',
        'iter_patches': {
            1: {
                'state_alpha_mult': {20: 0.8986},
            },
        },
        'post_ry_z_mult': 1.00117,
    },
    {
        'name': 'r61_td18_s20_0899_ryz00116',
        'description': 'Keep R60 exactly, but start the one-shot scale-block SCD a touch earlier after transition.',
        'rationale': 'Tests whether a micro cadence shift helps zz / median without softening the established SCD body.',
        'scd_patch': {
            'transition_duration': 1.8,
        },
    },
    {
        'name': 'r61_td22_s20_0899_ryz00116',
        'description': 'Keep R60 exactly, but start the one-shot scale-block SCD a touch later after transition.',
        'rationale': 'Checks the opposite cadence direction to see if a slightly later once-per-phase cut is safer for the central bulk stats.',
        'scd_patch': {
            'transition_duration': 2.2,
        },
    },
]


def _load_round60_base_candidate():
    for candidate in ROUND60_CONSERVATIVE_CANDIDATES:
        if candidate['name'] == ROUND60_BASE_NAME:
            return _merge_round60_candidate(candidate)
    raise KeyError(ROUND60_BASE_NAME)


def _merge_round61_candidate(extra_candidate: dict):
    merged = copy.deepcopy(_load_round60_base_candidate())
    merged['name'] = extra_candidate['name']
    merged['description'] = extra_candidate['description']
    merged['rationale'] = extra_candidate['rationale']

    merged['scd'] = copy.deepcopy(merged.get('scd', {}))
    merged['scd'].update(copy.deepcopy(extra_candidate.get('scd_patch', {})))

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

    if extra_candidate.get('post_rx_y_mult') is not None:
        merged['post_rx_y_mult'] = float(extra_candidate['post_rx_y_mult'])
    if extra_candidate.get('post_ry_z_mult') is not None:
        merged['post_ry_z_mult'] = float(extra_candidate['post_ry_z_mult'])
    merged['round61_extra_patch'] = copy.deepcopy(extra_candidate)
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


def _run_candidate(candidate: dict, idx: int):
    merged_candidate = _merge_round61_candidate(candidate)
    method_mod = load_module(f'markov_method_round61_micro_probe_{idx}', str(R53_METHOD_FILE))
    method_mod = _build_patched_method(method_mod, merged_candidate)
    method_mod.METHOD = f"42-state GM1 round61 micro probe {merged_candidate['name']}"
    method_mod.VARIANT = f"42state_gm1_round61_micro_probe_{merged_candidate['name']}"

    source_mod = load_module(f'markov_pruned_source_for_round61_micro_probe_{idx}', str(SOURCE_FILE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = method_mod._build_dataset(source_mod)
    result = _run_internalized_hybrid_scd(
        method_mod,
        source_mod,
        imu_noisy,
        pos0,
        ts,
        bi_g=bi_g,
        bi_a=bi_a,
        tau_g=tau_g,
        tau_a=tau_a,
        label=f'42-GM1-R61M-{idx}',
        scd_cfg=merged_candidate['scd'],
    )
    clbt = result[0]
    extra = result[4] if len(result) >= 5 else {}
    _, focus, lever, overall = _compute_metrics(source_mod, clbt)
    return merged_candidate, result, focus, lever, overall, extra


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    r58_payload = json.loads(R58_JSON.read_text(encoding='utf-8'))
    r59_payload = json.loads(R59_JSON.read_text(encoding='utf-8'))
    r60_payload = json.loads(R60_JSON.read_text(encoding='utf-8'))

    candidate_dump = {
        'baseline': ROUND60_BASE_NAME,
        'round61_micro_candidates': ROUND61_CANDIDATES,
    }
    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    out = {
        'baseline_r60': {
            'focus': r60_payload['focus_scale_pct'],
            'lever': r60_payload['lever_guard_pct'],
            'overall': r60_payload['overall'],
        },
        'baseline_r59': {
            'focus': r59_payload['focus_scale_pct'],
            'lever': r59_payload['lever_guard_pct'],
            'overall': r59_payload['overall'],
        },
        'baseline_r58': {
            'focus': r58_payload['focus_scale_pct'],
            'lever': r58_payload['lever_guard_pct'],
            'overall': r58_payload['overall'],
        },
        'baseline_name': ROUND60_BASE_NAME,
        'candidates': {},
    }

    for idx, candidate in enumerate(ROUND61_CANDIDATES, start=1):
        merged_candidate, result, focus, lever, overall, extra = _run_candidate(candidate, idx)
        probe_info = {
            'description': merged_candidate['description'],
            'rationale': merged_candidate['rationale'],
            'base_round60_candidate': ROUND60_BASE_NAME,
            'policy_patch': _sorted_policy_patch(merged_candidate.get('iter_patches', {})),
            'round61_extra_patch': copy.deepcopy(candidate),
            'scd': copy.deepcopy(merged_candidate['scd']),
            'post_rx_y_mult': float(merged_candidate.get('post_rx_y_mult', 1.0)),
            'post_ry_z_mult': float(merged_candidate.get('post_ry_z_mult', 1.0)),
            'focus': focus,
            'lever': lever,
            'overall': overall,
            'delta_vs_r60': {
                **_delta_block(focus, r60_payload['focus_scale_pct']),
                **_delta_block(lever, r60_payload['lever_guard_pct']),
                **_delta_block(overall, r60_payload['overall']),
            },
            'delta_vs_r59': {
                **_delta_block(focus, r59_payload['focus_scale_pct']),
                **_delta_block(lever, r59_payload['lever_guard_pct']),
                **_delta_block(overall, r59_payload['overall']),
            },
            'delta_vs_r58': {
                **_delta_block(focus, r58_payload['focus_scale_pct']),
                **_delta_block(lever, r58_payload['lever_guard_pct']),
                **_delta_block(overall, r58_payload['overall']),
            },
            'key_round61_delta': {
                'hold_vs_r60_dKg_xx': float(focus['dKg_xx'] - r60_payload['focus_scale_pct']['dKg_xx']),
                'hold_vs_r60_dKg_xy': float(focus['dKg_xy'] - r60_payload['focus_scale_pct']['dKg_xy']),
                'hold_vs_r60_dKg_yy': float(focus['dKg_yy'] - r60_payload['focus_scale_pct']['dKg_yy']),
                'repair_vs_r60_dKg_zz': float(focus['dKg_zz'] - r60_payload['focus_scale_pct']['dKg_zz']),
                'hold_vs_r60_dKa_xx': float(focus['dKa_xx'] - r60_payload['focus_scale_pct']['dKa_xx']),
                'hold_vs_r60_rx_y': float(lever['rx_y'] - r60_payload['lever_guard_pct']['rx_y']),
                'repair_vs_r60_ry_z': float(lever['ry_z'] - r60_payload['lever_guard_pct']['ry_z']),
                'repair_vs_r60_mean': float(overall['mean_pct_error'] - r60_payload['overall']['mean_pct_error']),
                'repair_vs_r60_median': float(overall['median_pct_error'] - r60_payload['overall']['median_pct_error']),
                'hold_vs_r60_max': float(overall['max_pct_error'] - r60_payload['overall']['max_pct_error']),
            },
            'extra': {
                'schedule_log': extra.get('schedule_log'),
                'feedback_log': extra.get('feedback_log'),
                'scd_log': extra.get('scd_log'),
            },
        }
        out['candidates'][merged_candidate['name']] = probe_info
        print(merged_candidate['name'], json.dumps({
            'focus': focus,
            'lever': lever,
            'overall': overall,
            'delta_vs_r60': probe_info['delta_vs_r60'],
            'delta_vs_r59': probe_info['delta_vs_r59'],
        }, ensure_ascii=False))

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {OUTPUT_JSON}')
    print('__RESULT_JSON__=' + json.dumps({
        'output_json': str(OUTPUT_JSON),
        'candidate_json': str(CANDIDATE_JSON),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
