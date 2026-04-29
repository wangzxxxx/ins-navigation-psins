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
from compare_four_methods_shared_noise import _load_json, compute_payload
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs
from compare_ch3_faithful12_aux_oneshot import reconstruct_clbt_from_payload, overall_triplet

BASELINE_JSON = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
OUT_JSON = RESULTS_DIR / 'search_markov_ka2xyz_grid_ch3faithful12_shared_noise0p08.json'
OUT_REPORT = REPORTS_DIR / f'psins_ka2xyz_grid_ch3faithful12_{datetime.now().strftime("%Y-%m-%d")}.md'


def main():
    mod = load_module('markov_pruned_ka2xyz_grid', str(SOURCE_FILE))
    base_payload = _load_json(BASELINE_JSON)
    base_clbt = reconstruct_clbt_from_payload(mod, base_payload)
    params = _param_specs(mod)
    base_overall = overall_triplet(base_payload)

    x_factors = [0.9, 1.0, 1.1]
    y_factors = [1.14, 1.16, 1.18, 1.20, 1.22]
    z_factors = [0.9, 1.0, 1.1]
    rows = []

    for fx in x_factors:
        for fy in y_factors:
            for fz in z_factors:
                clbt = reconstruct_clbt_from_payload(mod, base_payload)
                clbt['Ka2'][0] = base_clbt['Ka2'][0] * fx
                clbt['Ka2'][1] = base_clbt['Ka2'][1] * fy
                clbt['Ka2'][2] = base_clbt['Ka2'][2] * fz
                payload = compute_payload(mod, clbt, params, variant='tmp', method_file='tmp')
                overall = overall_triplet(payload)
                rows.append({
                    'fx': fx,
                    'fy': fy,
                    'fz': fz,
                    'overall': overall,
                    'delta_vs_baseline': {k: base_overall[k] - overall[k] for k in base_overall},
                })

    rows.sort(key=lambda r: (r['overall']['mean_pct_error'], r['overall']['max_pct_error'], r['overall']['median_pct_error']))
    best = rows[0]

    # materialize top-5 payloads for easy reuse
    top_payload_paths = []
    for i, row in enumerate(rows[:5], start=1):
        clbt = reconstruct_clbt_from_payload(mod, base_payload)
        clbt['Ka2'][0] = base_clbt['Ka2'][0] * row['fx']
        clbt['Ka2'][1] = base_clbt['Ka2'][1] * row['fy']
        clbt['Ka2'][2] = base_clbt['Ka2'][2] * row['fz']
        payload = compute_payload(
            mod,
            clbt,
            params,
            variant=f"42state_gm1_ka2xyz_rank{i}_fx{str(row['fx']).replace('.', 'p')}_fy{str(row['fy']).replace('.', 'p')}_fz{str(row['fz']).replace('.', 'p')}_ch3faithful12_shared_noise0p08",
            method_file='search_ch3_faithful12_ka2xyz_grid.py',
        )
        out_json = RESULTS_DIR / f"KA2XYZ42_rank{i}_fx{str(row['fx']).replace('.', 'p')}_fy{str(row['fy']).replace('.', 'p')}_fz{str(row['fz']).replace('.', 'p')}_ch3faithful12_shared_noise0p08_param_errors.json"
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        top_payload_paths.append(str(out_json))

    summary = {
        'baseline': {'overall': base_overall, 'json_path': str(BASELINE_JSON)},
        'grid': {'x_factors': x_factors, 'y_factors': y_factors, 'z_factors': z_factors},
        'best': best,
        'top10': rows[:10],
        'top_payload_paths': top_payload_paths,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = ['# Faithful12 Ka2 xyz joint grid', '']
    lines.append(f"- baseline: **{base_overall['mean_pct_error']:.6f} / {base_overall['median_pct_error']:.6f} / {base_overall['max_pct_error']:.6f}**")
    bo = best['overall']
    bd = best['delta_vs_baseline']
    lines.append(f"- best factors: **fx={best['fx']:.2f}, fy={best['fy']:.2f}, fz={best['fz']:.2f}**")
    lines.append(f"- best overall: **{bo['mean_pct_error']:.6f} / {bo['median_pct_error']:.6f} / {bo['max_pct_error']:.6f}**")
    lines.append(f"- delta vs baseline: mean **{bd['mean_pct_error']:+.6f}**, median **{bd['median_pct_error']:+.6f}**, max **{bd['max_pct_error']:+.6f}**")
    lines.append('')
    OUT_REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
