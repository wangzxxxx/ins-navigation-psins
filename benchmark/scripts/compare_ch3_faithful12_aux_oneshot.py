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
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'
METHOD_DIR = ROOT / 'psins_method_bench' / 'methods' / 'markov'
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'

for p in [SCRIPTS_DIR, METHOD_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module
from compare_ch3_12pos_path_baselines import build_ch3_path_paras, build_dataset_with_path
from compare_four_methods_shared_noise import _load_json, compute_payload
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs
from method_42state_gm1_round18_global_aux_progressive_weakrelease import _global_aux_weaksolve

NOISE_SCALE = 0.08
BASELINE_JSON = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
OUT_SUMMARY_JSON = RESULTS_DIR / 'compare_markov_vs_aux_oneshot_ch3faithful12_shared_noise0p08.json'
OUT_REPORT_MD = REPORTS_DIR / f'psins_aux_oneshot_ch3faithful12_{datetime.now().strftime("%Y-%m-%d")}.md'


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


def copy_clbt(mod, clbt):
    return {
        'Kg': mod.np.array(clbt['Kg'], dtype=float).copy(),
        'Ka': mod.np.array(clbt['Ka'], dtype=float).copy(),
        'Ka2': mod.np.array(clbt['Ka2'], dtype=float).copy(),
        'eb': mod.np.array(clbt['eb'], dtype=float).copy(),
        'db': mod.np.array(clbt['db'], dtype=float).copy(),
        'rx': mod.np.array(clbt['rx'], dtype=float).copy(),
        'ry': mod.np.array(clbt['ry'], dtype=float).copy(),
    }


def overall_triplet(payload: dict):
    o = payload['overall']
    return {
        'mean_pct_error': float(o['mean_pct_error']),
        'median_pct_error': float(o['median_pct_error']),
        'max_pct_error': float(o['max_pct_error']),
    }


def delta(base: dict, cand: dict):
    return {k: float(base[k] - cand[k]) for k in ['mean_pct_error', 'median_pct_error', 'max_pct_error']}


def classify(d):
    vals = [d['mean_pct_error'], d['median_pct_error'], d['max_pct_error']]
    if all(v > 0 for v in vals):
        return 'better'
    if all(v < 0 for v in vals):
        return 'worse'
    return 'mixed'


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def build_candidates(mod, baseline_clbt, aux_clbt):
    rows = []

    def add(name, ka2_alpha, lever_alpha):
        c = copy_clbt(mod, baseline_clbt)
        c['Ka2'] = (1 - ka2_alpha) * baseline_clbt['Ka2'] + ka2_alpha * aux_clbt['Ka2']
        c['rx'] = (1 - lever_alpha) * baseline_clbt['rx'] + lever_alpha * aux_clbt['rx']
        c['ry'] = (1 - lever_alpha) * baseline_clbt['ry'] + lever_alpha * aux_clbt['ry']
        rows.append((name, c, ka2_alpha, lever_alpha))

    add('aux_blend_ka2_0p25_lever_0p00', 0.25, 0.00)
    add('aux_blend_ka2_0p50_lever_0p00', 0.50, 0.00)
    add('aux_blend_ka2_0p25_lever_0p25', 0.25, 0.25)
    add('aux_blend_ka2_0p50_lever_0p25', 0.50, 0.25)
    add('aux_full_ka2_lever', 1.00, 1.00)
    return rows


def render_report(summary: dict) -> str:
    b = summary['baseline']['overall']
    lines = []
    lines.append('# Faithful12 auxiliary one-shot weak-state writeback')
    lines.append('')
    lines.append('## Fixed setup')
    lines.append('')
    lines.append('- path = chapter-3 faithful 12-position path')
    lines.append('- initial attitude = (0, 0, 0) deg')
    lines.append('- strong-state baseline = existing faithful12 Markov42 result')
    lines.append('- weak-state method = static-window auxiliary solve, no main-filter rerun')
    lines.append('')
    lines.append('## Overall metrics (lower is better)')
    lines.append('')
    lines.append('| method | mean | median | max | verdict vs baseline |')
    lines.append('|---|---:|---:|---:|---|')
    lines.append(f"| baseline | {b['mean_pct_error']:.6f} | {b['median_pct_error']:.6f} | {b['max_pct_error']:.6f} | ref |")
    for row in summary['candidates']:
        o = row['overall']
        lines.append(f"| {row['name']} | {o['mean_pct_error']:.6f} | {o['median_pct_error']:.6f} | {o['max_pct_error']:.6f} | **{row['verdict_vs_baseline']}** |")
    lines.append('')
    lines.append(f"## Best candidate\n\n- **{summary['best_candidate']}**")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    mod = load_module('markov_pruned_42_aux_oneshot_ch3faithful12', str(SOURCE_FILE))
    paras = build_ch3_path_paras(mod)
    dataset = build_dataset_with_path(mod, NOISE_SCALE, paras)
    params = _param_specs(mod)

    baseline_payload = _load_json(BASELINE_JSON)
    baseline_clbt = reconstruct_clbt_from_payload(mod, baseline_payload)
    aux_clbt = _global_aux_weaksolve(mod, dataset['imu_noisy'], dataset['pos0'], dataset['ts'], baseline_clbt)

    baseline_overall = overall_triplet(baseline_payload)
    candidate_rows = []
    for name, clbt, ka2_alpha, lever_alpha in build_candidates(mod, baseline_clbt, aux_clbt):
        payload = compute_payload(
            mod,
            clbt,
            params,
            variant=f'42state_gm1_{name}_ch3faithful12_shared_noise0p08',
            method_file='compare_ch3_faithful12_aux_oneshot.py',
            extra={
                'noise_scale': NOISE_SCALE,
                'att0_deg': [0.0, 0.0, 0.0],
                'path_key': 'chapter3_12pos_reconstructed',
                'path_tag': 'ch3faithful12',
                'comparison_mode': 'aux_oneshot_writeback',
                'ka2_alpha': ka2_alpha,
                'lever_alpha': lever_alpha,
            },
        )
        out_json = RESULTS_DIR / f'AUX42_markov_{name}_ch3faithful12_shared_noise0p08_param_errors.json'
        write_json(out_json, payload)
        o = overall_triplet(payload)
        d = delta(baseline_overall, o)
        candidate_rows.append({
            'name': name,
            'overall': o,
            'delta_vs_baseline': d,
            'verdict_vs_baseline': classify(d),
            'json_path': str(out_json),
        })

    best = min(candidate_rows, key=lambda r: (r['overall']['mean_pct_error'], r['overall']['max_pct_error'], r['overall']['median_pct_error']))
    summary = {
        'baseline': {
            'name': 'markov42_baseline',
            'overall': baseline_overall,
            'json_path': str(BASELINE_JSON),
        },
        'candidates': candidate_rows,
        'best_candidate': best['name'],
        'noise_scale': NOISE_SCALE,
        'att0_deg': [0.0, 0.0, 0.0],
    }
    write_json(OUT_SUMMARY_JSON, summary)
    OUT_REPORT_MD.write_text(render_report(summary), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
