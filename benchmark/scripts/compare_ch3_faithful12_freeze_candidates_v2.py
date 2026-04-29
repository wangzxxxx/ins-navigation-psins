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

NOISE_SCALE = 0.08
BASELINE_JSON = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
OUT_SUMMARY_JSON = RESULTS_DIR / 'compare_markov_vs_reasonable_freeze_ch3faithful12_shared_noise0p08.json'
OUT_REPORT_MD = REPORTS_DIR / f'psins_reasonable_freeze_candidates_ch3faithful12_{datetime.now().strftime("%Y-%m-%d")}.md'

KA2_IDX = slice(27, 30)
LEVER_IDX = slice(30, 36)
WEAK_IDX = slice(27, 36)
STRONG_TO_WEAK = (slice(WEAK_IDX.start, WEAK_IDX.stop), slice(12, 27))
WEAK_TO_STRONG = (slice(12, 27), slice(WEAK_IDX.start, WEAK_IDX.stop))


def overall_triplet(payload: dict):
    o = payload['overall']
    return {
        'mean_pct_error': float(o['mean_pct_error']),
        'median_pct_error': float(o['median_pct_error']),
        'max_pct_error': float(o['max_pct_error']),
    }


def delta_triplet(base: dict, cand: dict):
    return {
        k: float(base[k] - cand[k])
        for k in ['mean_pct_error', 'median_pct_error', 'max_pct_error']
    }


def classify(triplet_delta: dict) -> str:
    vals = [triplet_delta['mean_pct_error'], triplet_delta['median_pct_error'], triplet_delta['max_pct_error']]
    if all(v > 0 for v in vals):
        return 'better'
    if all(v < 0 for v in vals):
        return 'worse'
    return 'mixed'


def write_payload(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def make_clbt_zero(mod):
    return {
        'Kg': mod.np.eye(3),
        'Ka': mod.np.eye(3),
        'Ka2': mod.np.zeros(3),
        'eb': mod.np.zeros(3),
        'db': mod.np.zeros(3),
        'rx': mod.np.zeros(3),
        'ry': mod.np.zeros(3),
    }


def run_variant(mod, dataset: dict, cfg: dict):
    ts = dataset['ts']
    pos0 = dataset['pos0']
    imu1 = dataset['imu_noisy']
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
    for k in range(frq2, min(5 * 60 * 2 * frq2, len(imu1)), 2 * frq2):
        ww = mod.np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
        if mod.np.linalg.norm(ww) / ts > 20 * mod.glv.dph:
            break
    kstatic = k - 3 * frq2

    clbt = make_clbt_zero(mod)
    length = len(imu1)
    dotwf = mod.imudot(imu1, 5.0)
    P_trace, X_trace, iter_bounds = [], [], []
    iterations = 3

    def apply_clbt(imu_s, c):
        res = mod.np.copy(imu_s)
        for i in range(len(res)):
            res[i, 0:3] = c['Kg'] @ res[i, 0:3] - c['eb'] * ts
            res[i, 3:6] = c['Ka'] @ res[i, 3:6] - c['db'] * ts
        return res

    for it in range(iterations):
        print(f"  [{cfg['label']}] Iter {it+1}/{iterations}")
        kf = mod.clbtkfinit_42(nts, bi_g, tau_g, bi_a, tau_a)
        if it == iterations - 1:
            kf['Pxk'] = kf['Pxk'] * 100
            kf['Pxk'][:, 2] = 0
            kf['Pxk'][2, :] = 0
            kf['xk'] = mod.np.zeros(42)

        # stage-wise prior shrinkage on weak states
        ka2_p = cfg['ka2_p_scale'][it]
        lever_p = cfg['lever_p_scale'][it]
        kf['Pxk'][KA2_IDX, KA2_IDX] *= ka2_p
        kf['Pxk'][LEVER_IDX, LEVER_IDX] *= lever_p

        imu_align = apply_clbt(imu1[frq2:kstatic, :], clbt)
        _, _, _, qnb = mod.alignsb(imu_align, pos0)
        vn = mod.np.zeros(3)
        t1s = 0.0

        for k in range(2 * frq2, length - frq2, nn):
            k1 = k + nn - 1
            wm = imu1[k:k1+1, 0:3]
            vm = imu1[k:k1+1, 3:6]
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
            Ft[STRONG_TO_WEAK] *= cfg['coupling_scale'][it]
            Ft[WEAK_TO_STRONG] *= cfg['coupling_scale'][it]
            kf['Phikk_1'] = mod.np.eye(42) + Ft * nts
            kf = mod.kfupdate(kf, TimeMeasBoth='T')

            if t1s > (0.2 - ts / 2):
                t1s = 0.0
                ww = mod.np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
                static_phase = (mod.np.linalg.norm(ww) / ts < 20 * mod.glv.dph)
                if static_phase:
                    kf['Qk'][WEAK_IDX, WEAK_IDX] *= cfg['static_q_scale'][it]
                    kf['Rk'] *= cfg['static_r_scale'][it]
                    kf = mod.kfupdate(kf, yk=vn, TimeMeasBoth='M')
                else:
                    kf['Qk'][WEAK_IDX, WEAK_IDX] *= cfg['dynamic_q_scale'][it]

                P_trace.append(mod.np.diag(kf['Pxk']))
                X_trace.append(mod.np.copy(kf['xk']))

        if it != iterations - 1:
            kf_fb = {k: (v.copy() if hasattr(v, 'copy') else v) for k, v in kf.items()}
            kf_fb['xk'] = mod.np.copy(kf['xk'])
            kf_fb['xk'][KA2_IDX] *= cfg['ka2_feedback_gain'][it]
            kf_fb['xk'][LEVER_IDX] *= cfg['lever_feedback_gain'][it]
            clbt = mod.clbtkffeedback_pruned(kf_fb, clbt, 42)
        else:
            clbt = mod.clbtkffeedback_pruned(kf, clbt, 42)

        iter_bounds.append(len(P_trace))

    return clbt, kf, mod.np.array(P_trace), mod.np.array(X_trace), iter_bounds


def render_report(summary: dict) -> str:
    lines = []
    lines.append('# Faithful12 reasonable freeze candidates')
    lines.append('')
    lines.append('## Fixed setup')
    lines.append('')
    lines.append('- path = chapter-3 faithful 12-position path')
    lines.append('- initial attitude = (0, 0, 0) deg')
    lines.append('- same Markov42 / same noise0.08 / same seed42')
    lines.append('- baseline reference = existing faithful12 Markov42 run')
    lines.append('')
    lines.append('## Overall metrics (lower is better)')
    lines.append('')
    lines.append('| method | mean | median | max | verdict vs baseline |')
    lines.append('|---|---:|---:|---:|---|')
    b = summary['baseline']['overall']
    lines.append(f"| baseline | {b['mean_pct_error']:.6f} | {b['median_pct_error']:.6f} | {b['max_pct_error']:.6f} | ref |")
    for row in summary['candidates']:
        o = row['overall']
        lines.append(f"| {row['name']} | {o['mean_pct_error']:.6f} | {o['median_pct_error']:.6f} | {o['max_pct_error']:.6f} | **{row['verdict_vs_baseline']}** |")
    lines.append('')
    lines.append('## Deltas vs baseline (baseline - candidate)')
    lines.append('')
    for row in summary['candidates']:
        d = row['delta_vs_baseline']
        lines.append(f"- **{row['name']}**: Δmean {d['mean_pct_error']:+.6f}, Δmedian {d['median_pct_error']:+.6f}, Δmax {d['max_pct_error']:+.6f}")
    lines.append('')
    lines.append(f"## Best candidate\n\n- **{summary['best_candidate']}**")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    mod = load_module('markov_pruned_42_reasonable_freeze_candidates', str(SOURCE_FILE))
    paras = build_ch3_path_paras(mod)
    dataset = build_dataset_with_path(mod, NOISE_SCALE, paras)
    params = _param_specs(mod)

    baseline_payload = _load_json(BASELINE_JSON)
    baseline_overall = overall_triplet(baseline_payload)

    candidate_cfgs = [
        {
            'name': 'soft_weak_release',
            'label': 'SOFT-WEAK-RELEASE',
            'ka2_feedback_gain': [0.25, 0.55],
            'lever_feedback_gain': [0.10, 0.35],
            'ka2_p_scale': [0.35, 0.60, 1.00],
            'lever_p_scale': [0.20, 0.40, 1.00],
            'coupling_scale': [0.75, 0.90, 1.00],
            'static_q_scale': [0.45, 0.65, 1.00],
            'dynamic_q_scale': [1.00, 1.10, 1.00],
            'static_r_scale': [1.10, 1.05, 1.00],
        },
        {
            'name': 'lever_late_release',
            'label': 'LEVER-LATE-RELEASE',
            'ka2_feedback_gain': [0.75, 1.00],
            'lever_feedback_gain': [0.00, 0.25],
            'ka2_p_scale': [0.70, 0.90, 1.00],
            'lever_p_scale': [0.12, 0.28, 1.00],
            'coupling_scale': [0.70, 0.85, 1.00],
            'static_q_scale': [0.40, 0.60, 1.00],
            'dynamic_q_scale': [1.00, 1.05, 1.00],
            'static_r_scale': [1.08, 1.03, 1.00],
        },
        {
            'name': 'staged_qr_softfreeze',
            'label': 'STAGED-QR-SOFTFREEZE',
            'ka2_feedback_gain': [0.45, 0.80],
            'lever_feedback_gain': [0.20, 0.50],
            'ka2_p_scale': [0.30, 0.55, 1.00],
            'lever_p_scale': [0.25, 0.60, 1.00],
            'coupling_scale': [0.85, 0.95, 1.00],
            'static_q_scale': [0.20, 0.50, 1.00],
            'dynamic_q_scale': [1.40, 1.20, 1.00],
            'static_r_scale': [1.15, 1.08, 1.00],
        },
    ]

    candidate_rows = []
    for cfg in candidate_cfgs:
        result = run_variant(mod, dataset, cfg)
        payload = compute_payload(
            mod,
            result[0],
            params,
            variant=f"42state_gm1_{cfg['name']}_ch3faithful12_shared_noise0p08",
            method_file='compare_ch3_faithful12_freeze_candidates_v2.py',
            extra={
                'noise_scale': NOISE_SCALE,
                'att0_deg': [0.0, 0.0, 0.0],
                'path_key': 'chapter3_12pos_reconstructed',
                'path_tag': 'ch3faithful12',
                'comparison_mode': 'reasonable_freeze_candidates',
                'freeze_cfg': cfg,
            },
        )
        out_json = RESULTS_DIR / f"RZ42_markov_{cfg['name']}_ch3faithful12_shared_noise0p08_param_errors.json"
        write_payload(out_json, payload)
        overall = overall_triplet(payload)
        d = delta_triplet(baseline_overall, overall)
        candidate_rows.append({
            'name': cfg['name'],
            'overall': overall,
            'delta_vs_baseline': d,
            'verdict_vs_baseline': classify(d),
            'json_path': str(out_json),
        })

    # pick best by lexicographic (mean, max, median)
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
    write_payload(OUT_SUMMARY_JSON, summary)
    OUT_REPORT_MD.write_text(render_report(summary), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
