from __future__ import annotations

import json
import sys
import types
from datetime import datetime
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
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
REPORTS_DIR = ROOT / 'reports'
METHOD_DIR = ROOT / 'psins_method_bench' / 'methods' / 'markov'
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'

for p in [METHOD_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module
from compare_ch3_12pos_path_baselines import build_ch3_path_paras, build_dataset_with_path
from compare_four_methods_shared_noise import _load_json, compute_payload
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs
from method_42state_gm1_round17_static_consensus_frozen_weakrefine import (
    _aux_static_consensus_weakstates,
    _run_strong_refine_with_frozen_weak,
)

NOISE_SCALE = 0.08
NOISE_TAG = 'noise0p08'
BASELINE_JSON = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
FREEZE_JSON = RESULTS_DIR / 'FZ42_markov_weakfreeze_ch3faithful12_shared_noise0p08_param_errors.json'
SUMMARY_JSON = RESULTS_DIR / 'compare_markov_vs_freeze_ch3faithful12_shared_noise0p08.json'
REPORT_MD = REPORTS_DIR / f'psins_markov_vs_freeze_ch3faithful12_{datetime.now().strftime("%Y-%m-%d")}.md'


def reconstruct_clbt_from_payload(mod, payload: dict):
    pe = payload['param_errors']
    eye = mod.np.eye(3)
    kg = eye.copy()
    ka = eye.copy()

    kg[0, 0] = 1.0 - pe['dKg_xx']['est']
    kg[1, 0] = 0.0 - pe['dKg_yx']['est']
    kg[2, 0] = 0.0 - pe['dKg_zx']['est']
    kg[0, 1] = 0.0 - pe['dKg_xy']['est']
    kg[1, 1] = 1.0 - pe['dKg_yy']['est']
    kg[2, 1] = 0.0 - pe['dKg_zy']['est']
    kg[0, 2] = 0.0 - pe['dKg_xz']['est']
    kg[1, 2] = 0.0 - pe['dKg_yz']['est']
    kg[2, 2] = 1.0 - pe['dKg_zz']['est']

    ka[0, 0] = 1.0 - pe['dKa_xx']['est']
    ka[0, 1] = 0.0 - pe['dKa_xy']['est']
    ka[0, 2] = 0.0 - pe['dKa_xz']['est']
    ka[1, 0] = 0.0
    ka[1, 1] = 1.0 - pe['dKa_yy']['est']
    ka[1, 2] = 0.0 - pe['dKa_yz']['est']
    ka[2, 0] = 0.0
    ka[2, 1] = 0.0
    ka[2, 2] = 1.0 - pe['dKa_zz']['est']

    return {
        'Kg': kg,
        'Ka': ka,
        'Ka2': mod.np.array([
            -pe['Ka2_x']['est'],
            -pe['Ka2_y']['est'],
            -pe['Ka2_z']['est'],
        ], dtype=float),
        'eb': mod.np.array([
            -pe['eb_x']['est'],
            -pe['eb_y']['est'],
            -pe['eb_z']['est'],
        ], dtype=float),
        'db': mod.np.array([
            -pe['db_x']['est'],
            -pe['db_y']['est'],
            -pe['db_z']['est'],
        ], dtype=float),
        'rx': mod.np.array([
            -pe['rx_x']['est'],
            -pe['rx_y']['est'],
            -pe['rx_z']['est'],
        ], dtype=float),
        'ry': mod.np.array([
            -pe['ry_x']['est'],
            -pe['ry_y']['est'],
            -pe['ry_z']['est'],
        ], dtype=float),
    }


def overall_triplet(payload: dict):
    o = payload['overall']
    return {
        'mean_pct_error': float(o['mean_pct_error']),
        'median_pct_error': float(o['median_pct_error']),
        'max_pct_error': float(o['max_pct_error']),
    }


def delta(base: dict, challenger: dict):
    out = {}
    for k in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        out[k] = {
            'baseline': float(base[k]),
            'freeze': float(challenger[k]),
            'improvement_pct_points': float(base[k] - challenger[k]),
        }
    return out


def verdict(deltas: dict) -> str:
    vals = [deltas[k]['improvement_pct_points'] for k in ['mean_pct_error', 'median_pct_error', 'max_pct_error']]
    if all(v > 0 for v in vals):
        return 'freeze_better'
    if all(v < 0 for v in vals):
        return 'freeze_worse'
    return 'mixed'


def render_report(summary: dict) -> str:
    b = summary['baseline']['overall']
    f = summary['freeze']['overall']
    d = summary['delta']
    lines = []
    lines.append('# Faithful12 Markov42 vs freeze-weak refine')
    lines.append('')
    lines.append('## Fixed setup')
    lines.append('')
    lines.append('- path = chapter-3 faithful 12-position path')
    lines.append('- initial attitude = (0, 0, 0) deg')
    lines.append('- source model = 42-state GM1 / ordinary Kalman baseline from `test_calibration_markov_pruned.py`')
    lines.append('- freeze variant = reuse the same faithful12 noisy dataset, first obtain a weak-state static consensus seed, then run a strong-only refine with weak states frozen')
    lines.append(f"- noise_scale = {summary['noise_scale']}")
    lines.append('')
    lines.append('## Overall metrics (lower is better)')
    lines.append('')
    lines.append('| method | mean | median | max |')
    lines.append('|---|---:|---:|---:|')
    lines.append(f"| Markov42 baseline | {b['mean_pct_error']:.6f} | {b['median_pct_error']:.6f} | {b['max_pct_error']:.6f} |")
    lines.append(f"| Freeze weak-state refine | {f['mean_pct_error']:.6f} | {f['median_pct_error']:.6f} | {f['max_pct_error']:.6f} |")
    lines.append('')
    lines.append('## Delta (baseline - freeze)')
    lines.append('')
    lines.append(f"- mean: {d['mean_pct_error']['improvement_pct_points']:+.6f} pct-points")
    lines.append(f"- median: {d['median_pct_error']['improvement_pct_points']:+.6f} pct-points")
    lines.append(f"- max: {d['max_pct_error']['improvement_pct_points']:+.6f} pct-points")
    lines.append(f"- verdict: **{summary['verdict']}**")
    lines.append('')
    lines.append('## Files')
    lines.append('')
    lines.append(f"- baseline_json: `{summary['files']['baseline_json']}`")
    lines.append(f"- freeze_json: `{summary['files']['freeze_json']}`")
    lines.append(f"- summary_json: `{summary['files']['summary_json']}`")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    mod = load_module('markov_pruned_42_faithful12_freeze_compare', str(SOURCE_FILE))
    paras = build_ch3_path_paras(mod)
    dataset = build_dataset_with_path(mod, NOISE_SCALE, paras)
    params = _param_specs(mod)

    baseline_payload = _load_json(BASELINE_JSON)
    baseline_clbt = reconstruct_clbt_from_payload(mod, baseline_payload)

    consensus_clbt = _aux_static_consensus_weakstates(mod, dataset['imu_noisy'], dataset['pos0'], dataset['ts'], baseline_clbt)
    freeze_res = _run_strong_refine_with_frozen_weak(
        mod,
        dataset['imu_noisy'],
        dataset['pos0'],
        dataset['ts'],
        dataset['bi_g'],
        dataset['bi_a'],
        dataset['tau_g'],
        dataset['tau_a'],
        init_clbt=consensus_clbt,
        label='CH3-FAITHFUL12-FREEZE',
    )

    freeze_payload = compute_payload(
        mod,
        freeze_res[0],
        params,
        variant='42state_gm1_ch3faithful12_freezeweak_shared_noise0p08',
        method_file='compare_ch3_faithful12_markov_vs_freeze.py -> round17 frozen-weak refine on faithful12 path',
        extra={
            'noise_scale': NOISE_SCALE,
            'att0_deg': [0.0, 0.0, 0.0],
            'path_key': 'chapter3_12pos_reconstructed',
            'path_tag': 'ch3faithful12',
            'candidate_name': 'faithful12_freezeweak',
            'comparison_mode': 'markov42_vs_freeze_on_faithful12',
            'baseline_json': str(BASELINE_JSON),
            'freeze_policy': 'reuse faithful12 baseline seed -> static consensus on weak states -> rerun strong refine with weak states frozen (indices 27:36)',
        },
    )
    FREEZE_JSON.write_text(json.dumps(freeze_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    baseline_overall = overall_triplet(baseline_payload)
    freeze_overall = overall_triplet(freeze_payload)
    deltas = delta(baseline_overall, freeze_overall)
    summary = {
        'noise_scale': NOISE_SCALE,
        'baseline': {
            'name': 'markov42_baseline',
            'overall': baseline_overall,
        },
        'freeze': {
            'name': 'freeze_weak_refine',
            'overall': freeze_overall,
        },
        'delta': deltas,
        'verdict': verdict(deltas),
        'files': {
            'baseline_json': str(BASELINE_JSON),
            'freeze_json': str(FREEZE_JSON),
            'summary_json': str(SUMMARY_JSON),
            'report_md': str(REPORT_MD),
        },
    }
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    REPORT_MD.write_text(render_report(summary), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
