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

BASELINE_JSON = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
OUT_JSON = RESULTS_DIR / 'compare_truefreeze_family_ch3faithful12_shared_noise0p08.json'
OUT_REPORT = REPORTS_DIR / f'psins_truefreeze_family_ch3faithful12_{datetime.now().strftime("%Y-%m-%d")}.md'

STRONG_IDX = list(range(6, 27)) + list(range(36, 42))
WEAK_IDX = list(range(27, 36))
TARGETED_KG_DIAG_IDX = [12, 16, 20]


def overall_triplet(payload: dict):
    o = payload['overall']
    return {
        'mean_pct_error': float(o['mean_pct_error']),
        'median_pct_error': float(o['median_pct_error']),
        'max_pct_error': float(o['max_pct_error']),
    }


def apply_strong_feedback(mod, clbt, xk):
    subkf = {'xk': mod.np.zeros_like(xk)}
    subkf['xk'][6:27] = xk[6:27]
    subkf['xk'][36:42] = xk[36:42]
    return mod.clbtkffeedback_pruned(subkf, clbt, 42)


def apply_selected_weak_feedback(mod, clbt, xk, *, use_ka2=(True, True, True), use_rx=(False, False, False), use_ry=(False, False, False), kg_diag_gain=0.0):
    subkf = {'xk': mod.np.zeros_like(xk)}
    if use_ka2[0]:
        subkf['xk'][27] = xk[27]
    if use_ka2[1]:
        subkf['xk'][28] = xk[28]
    if use_ka2[2]:
        subkf['xk'][29] = xk[29]
    for i, flag in enumerate(use_rx):
        if flag:
            subkf['xk'][30 + i] = xk[30 + i]
    for i, flag in enumerate(use_ry):
        if flag:
            subkf['xk'][33 + i] = xk[33 + i]
    if kg_diag_gain > 0:
        for idx in TARGETED_KG_DIAG_IDX:
            subkf['xk'][idx] = kg_diag_gain * xk[idx]
    return mod.clbtkffeedback_pruned(subkf, clbt, 42)


def run_pass(mod, dataset, clbt, mode: str, *, weak_p_inflate=1.0, freeze_strong=False):
    ts = dataset['ts']
    pos0 = dataset['pos0']
    imu_noisy = dataset['imu_noisy']
    bi_g = dataset['bi_g']
    bi_a = dataset['bi_a']
    tau_g = dataset['tau_g']
    tau_a = dataset['tau_a']

    eth = mod.Earth(pos0)
    wnie = mod.glv.wie * mod.np.array([0, mod.math.cos(pos0[0]), mod.math.sin(pos0[0])])
    gn = mod.np.array([0, 0, -eth.g])
    Cba = mod.np.eye(3)
    nn, _, nts, _ = mod.nnts(2, ts)
    frq2 = int(1 / ts / 2) - 1

    k = frq2
    for k in range(frq2, min(5 * 60 * 2 * frq2, len(imu_noisy)), 2 * frq2):
        ww = mod.np.mean(imu_noisy[k-frq2:k+frq2+1, 0:3], axis=0)
        if mod.np.linalg.norm(ww) / ts > 20 * mod.glv.dph:
            break
    kstatic = k - 3 * frq2

    kf = mod.clbtkfinit_42(nts, bi_g, tau_g, bi_a, tau_a)
    if weak_p_inflate != 1.0:
        kf['Pxk'][27:36, 27:36] *= weak_p_inflate

    def apply_clbt(imu_s, c):
        res = mod.np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    imu_align = apply_clbt(imu_noisy[frq2:kstatic, :], clbt)
    _, _, _, qnb = mod.alignsb(imu_align, pos0)
    vn = mod.np.zeros(3)
    t1s = 0.0
    dotwf = mod.imudot(imu_noisy, 5.0)
    p_trace, x_trace = [], []

    for k in range(2 * frq2, len(imu_noisy) - frq2, nn):
        k1 = k + nn - 1
        wm = imu_noisy[k:k1+1, 0:3]
        vm = imu_noisy[k:k1+1, 3:6]
        dwb = mod.np.mean(dotwf[k:k1+1, 0:3], axis=0)
        phim, dvbm = mod.cnscl(mod.np.hstack((wm, vm)))
        phim = clbt['Kg'] @ phim - clbt['eb'] * nts
        dvbm = clbt['Ka'] @ dvbm - clbt['db'] * nts
        wb = phim / nts
        fb = dvbm / nts
        SS = mod.imulvS(wb, dwb, Cba)
        fL = SS[:, 0:6] @ mod.np.concatenate((clbt['rx'], clbt['ry']))
        fn = mod.qmulv(qnb, fb - clbt['Ka2'] * (fb**2) - fL)
        vn = vn + (mod.rotv(-wnie * nts / 2, fn) + gn) * nts
        qnb = mod.qupdt2(qnb, phim, wnie * nts)
        t1s += nts

        Ft = mod.getFt_42(fb, wb, mod.q2mat(qnb), wnie, SS, tau_g, tau_a)
        kf['Phikk_1'] = mod.np.eye(42) + Ft * nts
        kf = mod.kfupdate(kf, TimeMeasBoth='T')
        if freeze_strong:
            kf['xk'][6:27] = 0.0
            kf['xk'][36:42] = 0.0

        if t1s > (0.2 - ts / 2):
            t1s = 0.0
            ww = mod.np.mean(imu_noisy[k-frq2:k+frq2+1, 0:3], axis=0)
            if mod.np.linalg.norm(ww) / ts < 20 * mod.glv.dph:
                kf = mod.kfupdate(kf, yk=vn, TimeMeasBoth='M')
                if freeze_strong:
                    kf['xk'][6:27] = 0.0
                    kf['xk'][36:42] = 0.0
            p_trace.append(mod.np.diag(kf['Pxk']))
            x_trace.append(mod.np.copy(kf['xk']))

    return kf, mod.np.array(p_trace), mod.np.array(x_trace)


def run_candidate(mod, dataset, cfg):
    clbt = {
        'Kg': mod.np.eye(3), 'Ka': mod.np.eye(3), 'Ka2': mod.np.zeros(3),
        'eb': mod.np.zeros(3), 'db': mod.np.zeros(3), 'rx': mod.np.zeros(3), 'ry': mod.np.zeros(3),
    }
    p_all, x_all, iter_bounds = [], [], []

    for _ in range(cfg['strong_passes']):
        kf, p, x = run_pass(mod, dataset, clbt, 'strong', weak_p_inflate=1.0, freeze_strong=False)
        clbt = apply_strong_feedback(mod, clbt, kf['xk'])
        p_all.extend(list(p)); x_all.extend(list(x)); iter_bounds.append(len(p_all))

    kf, p, x = run_pass(mod, dataset, clbt, 'weak', weak_p_inflate=cfg.get('weak_p_inflate', 8.0), freeze_strong=True)
    clbt = apply_selected_weak_feedback(
        mod, clbt, kf['xk'],
        use_ka2=cfg['use_ka2'], use_rx=cfg['use_rx'], use_ry=cfg['use_ry'], kg_diag_gain=cfg.get('kg_diag_gain', 0.0)
    )
    p_all.extend(list(p)); x_all.extend(list(x)); iter_bounds.append(len(p_all))

    if cfg.get('final_strong_refine', False):
        kf, p, x = run_pass(mod, dataset, clbt, 'final_strong', weak_p_inflate=1.0, freeze_strong=False)
        clbt = apply_strong_feedback(mod, clbt, kf['xk'])
        p_all.extend(list(p)); x_all.extend(list(x)); iter_bounds.append(len(p_all))

    return clbt, kf, mod.np.array(p_all), mod.np.array(x_all), iter_bounds


def main():
    mod = load_module('markov_pruned_truefreeze_family', str(SOURCE_FILE))
    paras = build_ch3_path_paras(mod)
    dataset = build_dataset_with_path(mod, 0.08, paras)
    params = _param_specs(mod)
    baseline_payload = _load_json(BASELINE_JSON)
    baseline = overall_triplet(baseline_payload)

    candidates = [
        {
            'name': 'freeze_strong2_then_ka2block',
            'strong_passes': 2,
            'use_ka2': (True, True, True),
            'use_rx': (False, False, False),
            'use_ry': (False, False, False),
            'weak_p_inflate': 10.0,
            'final_strong_refine': False,
        },
        {
            'name': 'freeze_strong2_then_ka2y_only',
            'strong_passes': 2,
            'use_ka2': (False, True, False),
            'use_rx': (False, False, False),
            'use_ry': (False, False, False),
            'weak_p_inflate': 10.0,
            'final_strong_refine': False,
        },
        {
            'name': 'freeze_strong1_then_ka2y_only_then_strong',
            'strong_passes': 1,
            'use_ka2': (False, True, False),
            'use_rx': (False, False, False),
            'use_ry': (False, False, False),
            'weak_p_inflate': 10.0,
            'final_strong_refine': True,
        },
    ]

    rows = []
    for cfg in candidates:
        print(f"=== {cfg['name']} ===")
        result = run_candidate(mod, dataset, cfg)
        payload = compute_payload(
            mod,
            result[0],
            params,
            variant=f"42state_gm1_{cfg['name']}_ch3faithful12_shared_noise0p08",
            method_file='compare_ch3_faithful12_truefreeze_family.py',
            extra={
                'path_key': 'chapter3_12pos_reconstructed',
                'att0_deg': [0.0, 0.0, 0.0],
                'comparison_mode': 'true_freeze_family',
                'freeze_cfg': cfg,
            },
        )
        out_json = RESULTS_DIR / f"TF42_markov_{cfg['name']}_ch3faithful12_shared_noise0p08_param_errors.json"
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        overall = overall_triplet(payload)
        delta = {k: baseline[k] - overall[k] for k in baseline}
        rows.append({
            'name': cfg['name'],
            'overall': overall,
            'delta_vs_baseline': delta,
            'verdict': 'better' if all(v > 0 for v in delta.values()) else ('worse' if all(v < 0 for v in delta.values()) else 'mixed'),
            'json_path': str(out_json),
        })

    best = min(rows, key=lambda r: (r['overall']['mean_pct_error'], r['overall']['max_pct_error'], r['overall']['median_pct_error']))
    summary = {
        'baseline': {'overall': baseline, 'json_path': str(BASELINE_JSON)},
        'candidates': rows,
        'best': best,
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = ['# Faithful12 true freeze-family benchmark', '']
    lines.append(f"- baseline: **{baseline['mean_pct_error']:.6f} / {baseline['median_pct_error']:.6f} / {baseline['max_pct_error']:.6f}**")
    bo = best['overall']; bd = best['delta_vs_baseline']
    lines.append(f"- best candidate: **{best['name']}**")
    lines.append(f"- best overall: **{bo['mean_pct_error']:.6f} / {bo['median_pct_error']:.6f} / {bo['max_pct_error']:.6f}**")
    lines.append(f"- delta vs baseline: mean **{bd['mean_pct_error']:+.6f}**, median **{bd['median_pct_error']:+.6f}**, max **{bd['max_pct_error']:+.6f}**")
    lines.append('')
    OUT_REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
