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
OUT_JSON = RESULTS_DIR / 'search_markov_ka2y_finegrid_ch3faithful12_shared_noise0p08.json'
OUT_REPORT = REPORTS_DIR / f'psins_ka2y_finegrid_ch3faithful12_{datetime.now().strftime("%Y-%m-%d")}.md'


def main():
    mod = load_module('markov_pruned_ka2y_finegrid', str(SOURCE_FILE))
    base_payload = _load_json(BASELINE_JSON)
    base_clbt = reconstruct_clbt_from_payload(mod, base_payload)
    params = _param_specs(mod)
    base_overall = overall_triplet(base_payload)

    factors = [1.14, 1.16, 1.18, 1.19, 1.20, 1.21, 1.22, 1.24, 1.26]
    rows = []
    for fy in factors:
        clbt = reconstruct_clbt_from_payload(mod, base_payload)
        clbt['Ka2'][1] = base_clbt['Ka2'][1] * fy
        payload = compute_payload(mod, clbt, params, variant='tmp', method_file='tmp')
        overall = overall_triplet(payload)
        out_json = RESULTS_DIR / f'KA2Y42_markov_ka2y_fine_{str(fy).replace('.', 'p')}_ch3faithful12_shared_noise0p08_param_errors.json'
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        rows.append({
            'ka2y_factor': fy,
            'overall': overall,
            'delta_vs_baseline': {k: base_overall[k] - overall[k] for k in base_overall},
            'json_path': str(out_json),
        })

    best = min(rows, key=lambda r: (r['overall']['mean_pct_error'], r['overall']['max_pct_error'], r['overall']['median_pct_error']))
    summary = {
        'baseline': {'overall': base_overall, 'json_path': str(BASELINE_JSON)},
        'rows': rows,
        'best': best,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = ['# Faithful12 Ka2_y fine-grid search', '']
    lines.append(f"- baseline: **{base_overall['mean_pct_error']:.6f} / {base_overall['median_pct_error']:.6f} / {base_overall['max_pct_error']:.6f}**")
    lines.append(f"- best factor: **{best['ka2y_factor']:.2f}**")
    bo = best['overall']
    bd = best['delta_vs_baseline']
    lines.append(f"- best overall: **{bo['mean_pct_error']:.6f} / {bo['median_pct_error']:.6f} / {bo['max_pct_error']:.6f}**")
    lines.append(f"- delta vs baseline: mean **{bd['mean_pct_error']:+.6f}**, median **{bd['median_pct_error']:+.6f}**, max **{bd['max_pct_error']:+.6f}**")
    lines.append('')
    OUT_REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
