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
R59_JSON = RESULTS_DIR / 'R59_42state_gm1_round59_scd_hybrid_scaleblock_once_param_errors.json'
OUTPUT_JSON = RESULTS_DIR / 'round60_narrow_probe_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'round60_narrow_candidates.json'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(ROOT / 'psins_method_bench' / 'scripts') not in sys.path:
    sys.path.insert(0, str(ROOT / 'psins_method_bench' / 'scripts'))

from common_markov import load_module
from probe_round55_newline import _build_patched_method
from probe_round56_narrow import _compute_metrics
from probe_round59_h_scd_hybrid import _merge_hybrid_candidate, _run_internalized_hybrid_scd


ROUND59_BASE_NAME = 'scd_scale_once_a0999_biaslink_commit'
FOCUS_KEYS = ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx']
LEVER_KEYS = ['rx_y', 'ry_z']


ROUND60_CANDIDATES = [
    {
        'name': 'scale_once_a09992_soft',
        'description': 'Keep the R59-H scale-block once-per-phase route, but weaken alpha a notch to reduce zz / median back-pressure.',
        'rationale': 'First-order probe: preserve the whole hybrid route and only slightly soften the SCD cut.',
        'scd_patch': {
            'alpha': 0.9992,
        },
    },
    {
        'name': 'scale_once_a09992_soft_zzryz',
        'description': 'Same slightly softer scale-block SCD, plus a tiny iter2 state20 relief and a tiny ry_z confirmation.',
        'rationale': 'Direct repair attempt for the exact two items R59 gave back: dKg_zz and ry_z.',
        'scd_patch': {
            'alpha': 0.9992,
        },
        'iter_patches': {
            1: {
                'state_alpha_mult': {20: 0.905},
            },
        },
        'post_ry_z_mult': 1.00118,
    },
    {
        'name': 'scale_once_a09994_soft_zzryz',
        'description': 'Even softer scale-block SCD with the same tiny state20 / ry_z repair pair.',
        'rationale': 'Checks whether the round59 gain is robust enough to survive a second notch softer while better repairing zz / median.',
        'scd_patch': {
            'alpha': 0.9994,
        },
        'iter_patches': {
            1: {
                'state_alpha_mult': {20: 0.905},
            },
        },
        'post_ry_z_mult': 1.00118,
    },
    {
        'name': 'scale_once_a09992_soft_zzryz_xxhold',
        'description': 'Same soft alpha + zz/ry_z repair, but add an ultra-small state12 hold so xx / max do not drift back.',
        'rationale': 'If the softer SCD relaxes xx too much, this keeps the proven state12 line intact with only a 12-only micro-hold.',
        'scd_patch': {
            'alpha': 0.9992,
        },
        'iter_patches': {
            1: {
                'state_alpha_mult': {12: 1.0019, 20: 0.905},
            },
        },
        'post_ry_z_mult': 1.00118,
    },
]


def _load_round59_base_candidate():
    return _merge_hybrid_candidate({
        'name': ROUND59_BASE_NAME,
        'description': 'Round59 formal best base candidate.',
        'rationale': 'Base for narrow Round60 refinements.',
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.999,
            'transition_duration': 2.0,
            'target': 'scale_block',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    })


def _merge_round60_candidate(extra_candidate: dict):
    merged = copy.deepcopy(_load_round59_base_candidate())
    merged['name'] = extra_candidate['name']
    merged['description'] = extra_candidate['description']
    merged['rationale'] = extra_candidate['rationale']

    merged['scd'] = copy.deepcopy(merged['scd'])
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
    merged['round60_extra_patch'] = copy.deepcopy(extra_candidate)
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
    merged_candidate = _merge_round60_candidate(candidate)
    method_mod = load_module(f'markov_method_round60_probe_{idx}', str(R53_METHOD_FILE))
    method_mod = _build_patched_method(method_mod, merged_candidate)
    method_mod.METHOD = f"42-state GM1 round60 probe {merged_candidate['name']}"
    method_mod.VARIANT = f"42state_gm1_round60_probe_{merged_candidate['name']}"

    source_mod = load_module(f'markov_pruned_source_for_round60_probe_{idx}', str(SOURCE_FILE))
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
        label=f'42-GM1-R60-{idx}',
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

    candidate_dump = {
        'baseline': ROUND59_BASE_NAME,
        'round60_candidates': ROUND60_CANDIDATES,
    }
    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    out = {
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
        'baseline_name': ROUND59_BASE_NAME,
        'candidates': {},
    }

    for idx, candidate in enumerate(ROUND60_CANDIDATES, start=1):
        merged_candidate, result, focus, lever, overall, extra = _run_candidate(candidate, idx)
        probe_info = {
            'description': merged_candidate['description'],
            'rationale': merged_candidate['rationale'],
            'base_round59_candidate': ROUND59_BASE_NAME,
            'policy_patch': _sorted_policy_patch(merged_candidate.get('iter_patches', {})),
            'round60_extra_patch': copy.deepcopy(candidate),
            'scd': copy.deepcopy(merged_candidate['scd']),
            'post_rx_y_mult': float(merged_candidate.get('post_rx_y_mult', 1.0)),
            'post_ry_z_mult': float(merged_candidate.get('post_ry_z_mult', 1.0)),
            'focus': focus,
            'lever': lever,
            'overall': overall,
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
            'key_round60_delta': {
                'protect_vs_r58_dKg_xx': float(focus['dKg_xx'] - r58_payload['focus_scale_pct']['dKg_xx']),
                'protect_vs_r58_dKg_xy': float(focus['dKg_xy'] - r58_payload['focus_scale_pct']['dKg_xy']),
                'protect_vs_r58_dKg_yy': float(focus['dKg_yy'] - r58_payload['focus_scale_pct']['dKg_yy']),
                'repair_vs_r59_dKg_zz': float(focus['dKg_zz'] - r59_payload['focus_scale_pct']['dKg_zz']),
                'protect_vs_r58_dKa_xx': float(focus['dKa_xx'] - r58_payload['focus_scale_pct']['dKa_xx']),
                'protect_vs_r58_rx_y': float(lever['rx_y'] - r58_payload['lever_guard_pct']['rx_y']),
                'repair_vs_r59_ry_z': float(lever['ry_z'] - r59_payload['lever_guard_pct']['ry_z']),
                'protect_vs_r58_mean': float(overall['mean_pct_error'] - r58_payload['overall']['mean_pct_error']),
                'repair_vs_r59_median': float(overall['median_pct_error'] - r59_payload['overall']['median_pct_error']),
                'protect_vs_r58_max': float(overall['max_pct_error'] - r58_payload['overall']['max_pct_error']),
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
            'delta_vs_r59': probe_info['delta_vs_r59'],
            'delta_vs_r58': probe_info['delta_vs_r58'],
        }, ensure_ascii=False))

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Wrote {OUTPUT_JSON}')
    print('__RESULT_JSON__=' + json.dumps({
        'output_json': str(OUTPUT_JSON),
        'candidate_json': str(CANDIDATE_JSON),
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
