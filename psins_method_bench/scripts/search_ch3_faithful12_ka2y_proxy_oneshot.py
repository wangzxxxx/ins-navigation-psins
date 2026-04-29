from __future__ import annotations

import json
import math
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
from compare_ch3_faithful12_aux_oneshot import reconstruct_clbt_from_payload, overall_triplet

NOISE_SCALE = 0.08
BASELINE_JSON = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
OUT_JSON = RESULTS_DIR / 'search_markov_ka2y_proxy_oneshot_ch3faithful12_shared_noise0p08.json'
OUT_REPORT = REPORTS_DIR / f'psins_ka2y_proxy_oneshot_ch3faithful12_{datetime.now().strftime("%Y-%m-%d")}.md'


def static_window_bounds(mod, imu1, ts):
    frq2 = int(1 / ts / 2) - 1
    windows = []
    in_static = False
    start = None
    for k in range(frq2, len(imu1) - frq2, 2 * frq2):
        ww = mod.np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
        is_static = mod.np.linalg.norm(ww) / ts < 20 * mod.glv.dph
        if is_static and not in_static:
            in_static = True
            start = max(0, k - frq2)
        elif (not is_static) and in_static:
            in_static = False
            windows.append((start, min(len(imu1), k + frq2)))
            start = None
    if in_static and start is not None:
        windows.append((start, len(imu1)))
    return windows


def objective_for_ka2y(mod, imu1, pos0, ts, clbt, ka2y_candidate):
    eth = mod.Earth(pos0)
    wnie = mod.glv.wie * mod.np.array([0, mod.math.cos(pos0[0]), mod.math.sin(pos0[0])])
    gn = mod.np.array([0, 0, -eth.g])
    windows = static_window_bounds(mod, imu1, ts)
    total = 0.0
    count = 0

    def apply_strong(seg):
        out = mod.np.copy(seg)
        for i in range(len(out)):
            out[i, 0:3] = clbt['Kg'] @ out[i, 0:3] - clbt['eb'] * ts
            out[i, 3:6] = clbt['Ka'] @ out[i, 3:6] - clbt['db'] * ts
        return out

    for s, e in windows:
        seg_raw = imu1[s:e, :]
        if len(seg_raw) < 80:
            continue
        seg = apply_strong(seg_raw)
        dotwf = mod.imudot(seg, 5.0)
        qnb = mod.alignsb(seg, pos0)[3]
        vn = mod.np.zeros(3)
        ka2 = clbt['Ka2'].copy()
        ka2[1] = ka2y_candidate
        for k in range(len(seg)):
            wm = seg[k, 0:3]
            vm = seg[k, 3:6]
            dwb = dotwf[k, 0:3]
            fb = vm / ts
            SS = mod.imulvS(wm / ts, dwb, mod.np.eye(3))
            fL = SS[:, 0:6] @ mod.np.concatenate((clbt['rx'], clbt['ry']))
            fn = mod.qmulv(qnb, fb - ka2 * (fb**2) - fL)
            vn = vn + (mod.rotv(-wnie * ts / 2, fn) + gn) * ts
            qnb = mod.qupdt2(qnb, wm, wnie * ts)
        total += float(mod.np.dot(vn, vn))
        count += 1
    return total / max(count, 1)


def render_report(summary: dict) -> str:
    lines = []
    lines.append('# Faithful12 Ka2_y posterior one-shot search')
    lines.append('')
    lines.append('## Fixed setup')
    lines.append('')
    lines.append('- path = chapter-3 faithful12')
    lines.append('- att0 = (0,0,0)')
    lines.append('- strong block + lever fixed to Markov42 baseline')
    lines.append('- only Ka2_y varied')
    lines.append('')
    lines.append('## Best candidate')
    lines.append('')
    b = summary['baseline']['overall']
    c = summary['best_candidate']['overall']
    lines.append(f"- baseline: **{b['mean_pct_error']:.6f} / {b['median_pct_error']:.6f} / {b['max_pct_error']:.6f}**")
    lines.append(f"- best candidate `{summary['best_candidate']['name']}`: **{c['mean_pct_error']:.6f} / {c['median_pct_error']:.6f} / {c['max_pct_error']:.6f}**")
    d = summary['best_candidate']['delta_vs_baseline']
    lines.append(f"- delta vs baseline: mean {d['mean_pct_error']:+.6f}, median {d['median_pct_error']:+.6f}, max {d['max_pct_error']:+.6f}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    mod = load_module('markov_pruned_ka2y_proxy_oneshot', str(SOURCE_FILE))
    paras = build_ch3_path_paras(mod)
    dataset = build_dataset_with_path(mod, NOISE_SCALE, paras)
    params = _param_specs(mod)

    baseline_payload = _load_json(BASELINE_JSON)
    baseline_clbt = reconstruct_clbt_from_payload(mod, baseline_payload)
    baseline_overall = overall_triplet(baseline_payload)

    base_y = float(baseline_clbt['Ka2'][1])
    factors = [0.70, 0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.30]
    rows = []
    best_obj = math.inf
    best_y = base_y
    for fac in factors:
        ky = base_y * fac
        obj = objective_for_ka2y(mod, dataset['imu_noisy'], dataset['pos0'], dataset['ts'], baseline_clbt, ky)
        rows.append({'factor': fac, 'ka2y_candidate': ky, 'proxy_obj': obj})
        if obj < best_obj:
            best_obj = obj
            best_y = ky

    candidates = []
    for fac, name in [(best_y / base_y, 'ka2y_proxy_best'), (1.10, 'ka2y_plus10pct'), (1.15, 'ka2y_plus15pct'), (1.20, 'ka2y_plus20pct')]:
        clbt = reconstruct_clbt_from_payload(mod, baseline_payload)
        clbt['Ka2'][1] = base_y * fac
        payload = compute_payload(
            mod,
            clbt,
            params,
            variant=f'42state_gm1_{name}_ch3faithful12_shared_noise0p08',
            method_file='search_ch3_faithful12_ka2y_proxy_oneshot.py',
            extra={
                'noise_scale': NOISE_SCALE,
                'att0_deg': [0.0, 0.0, 0.0],
                'path_key': 'chapter3_12pos_reconstructed',
                'path_tag': 'ch3faithful12',
                'comparison_mode': 'ka2y_proxy_oneshot',
                'ka2y_factor': fac,
                'base_ka2y': base_y,
            },
        )
        out_json = RESULTS_DIR / f'KA2Y42_markov_{name}_ch3faithful12_shared_noise0p08_param_errors.json'
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        overall = overall_triplet(payload)
        d = {k: baseline_overall[k] - overall[k] for k in baseline_overall}
        candidates.append({
            'name': name,
            'ka2y_factor': fac,
            'overall': overall,
            'delta_vs_baseline': d,
            'json_path': str(out_json),
        })

    best = min(candidates, key=lambda r: (r['overall']['mean_pct_error'], r['overall']['max_pct_error'], r['overall']['median_pct_error']))
    summary = {
        'baseline': {
            'overall': baseline_overall,
            'json_path': str(BASELINE_JSON),
            'base_ka2y': base_y,
        },
        'proxy_scan': rows,
        'best_candidate': best,
        'all_candidates': candidates,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    OUT_REPORT.write_text(render_report(summary), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
