#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
PSINS_ROOT = WORKSPACE / 'tmp_psins_py'
if str(PSINS_ROOT) not in sys.path:
    sys.path.insert(0, str(PSINS_ROOT))

BASE12_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
ACC18_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_accel_colored_py_2026-03-30.py'
REF_JSON = WORKSPACE / 'psins_method_bench' / 'results' / 'sweep_g2_18state_param_outer_extrap3_2026-04-07.json'
RESULTS_DIR = WORKSPACE / 'psins_method_bench' / 'results'
REPORTS_DIR = WORKSPACE / 'reports'
OUT_JSON = RESULTS_DIR / 'sweep_g2_18state_param_outer_extrap4_2026-04-07.json'
OUT_MD = REPORTS_DIR / 'psins_g2_18state_param_outer_extrap4_2026-04-07.md'

MAX_WORKERS = min(4, os.cpu_count() or 1)
SEEDS = [0, 1, 2, 3, 4]
TS = 0.01
SINGLE_AXIS_ATT0_DEG = np.array([1.0, 0.0, 10.0])
WVN = np.array([0.01, 0.01, 0.01])
PHI_DEG = np.array([0.1, 0.1, 0.5])
NG_SIGMA_DPH = np.array([0.05, 0.05, 0.05])
TAU_G_S = np.array([300.0, 300.0, 300.0])
XA_SIGMA_UG = np.array([0.01, 0.01, 0.01])
TAU_A_S = np.array([100.0, 100.0, 100.0])

CONFIGS = [
    {'label': 'param_outer_fb18_it12', 'feedback_scale': 1.8, 'outer_iterations': 12},
    {'label': 'param_outer_fb20_it12', 'feedback_scale': 2.0, 'outer_iterations': 12},
    {'label': 'param_outer_fb18_it13', 'feedback_scale': 1.8, 'outer_iterations': 13},
    {'label': 'param_outer_fb20_it13', 'feedback_scale': 2.0, 'outer_iterations': 13},
]

_BASE12 = None
_ACC18 = None


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_base12():
    global _BASE12
    if _BASE12 is None:
        _BASE12 = load_module('small_param_outer_base12_20260407', BASE12_PATH)
    return _BASE12


def load_acc18():
    global _ACC18
    if _ACC18 is None:
        _ACC18 = load_module('small_param_outer_acc18_20260407', ACC18_PATH)
    return _ACC18


def build_single_axis_att(base12):
    att0 = SINGLE_AXIS_ATT0_DEG * base12.glv.deg
    paras = np.array([[1, 0, 0, 1, 3000 * base12.glv.deg, 300.0, 0.0, 0.0]], dtype=float)
    return att0, base12.attrottt(att0, paras, TS)


def extract_last_metrics(last: Any) -> dict[str, Any]:
    if is_dataclass(last):
        return {
            'att_err_arcsec': [float(x) for x in last.att_err_arcsec],
            'att_err_norm_arcsec': float(last.att_err_norm_arcsec),
            'yaw_abs_arcsec': float(last.yaw_abs_arcsec),
            'final_att_deg': [float(x) for x in last.final_att_deg],
            'est_eb_dph': [float(x) for x in last.est_eb_dph],
            'est_db_ug': [float(x) for x in last.est_db_ug],
        }
    return {
        'att_err_arcsec': [float(x) for x in last['att_err_arcsec']],
        'att_err_norm_arcsec': float(last['att_err_norm_arcsec']),
        'yaw_abs_arcsec': float(last['yaw_abs_arcsec']),
        'final_att_deg': [float(x) for x in last['final_att_deg']],
        'est_eb_dph': [float(x) for x in last['est_eb_dph']],
        'est_db_ug': [float(x) for x in last['est_db_ug']],
    }


def run_param_outer(base12, acc18, imu_noisy, att0_guess, pos0, phi, imuerr, truth_att, cfg):
    eb_accum = np.zeros(3)
    db_accum = np.zeros(3)
    outer_logs = []
    last = None
    for outer_it in range(1, cfg['outer_iterations'] + 1):
        imu_corr = imu_noisy.copy()
        imu_corr[:, 0:3] -= eb_accum * TS
        imu_corr[:, 3:6] -= db_accum * TS
        _, _, _, iter_logs = acc18.alignvn_18state_iter(
            imu=imu_corr,
            qnb=att0_guess,
            pos=pos0,
            phi0=phi,
            imuerr=imuerr,
            wvn=WVN.copy(),
            max_iter=1,
            truth_att=truth_att,
            ng_sigma=NG_SIGMA_DPH * base12.glv.dph,
            tau_g_s=TAU_G_S.copy(),
            xa_sigma=XA_SIGMA_UG * base12.glv.ug,
            tau_a_s=TAU_A_S.copy(),
            wash_scale=0.0,
            carry_att_seed=False,
        )
        last = extract_last_metrics(iter_logs[-1])
        eb_est = np.array(last['est_eb_dph'], dtype=float) * base12.glv.dph
        db_est = np.array(last['est_db_ug'], dtype=float) * base12.glv.ug
        eb_accum = eb_accum + cfg['feedback_scale'] * eb_est
        db_accum = db_accum + cfg['feedback_scale'] * db_est
        outer_logs.append({
            'outer_iteration': outer_it,
            'att_err_arcsec': last['att_err_arcsec'],
            'yaw_abs_arcsec': last['yaw_abs_arcsec'],
            'accum_eb_dph': (eb_accum / base12.glv.dph).tolist(),
            'accum_db_ug': (db_accum / base12.glv.ug).tolist(),
        })
    assert last is not None
    return {
        'final_att_deg': last['final_att_deg'],
        'final_att_err_arcsec': last['att_err_arcsec'],
        'final_att_err_abs_arcsec': [abs(v) for v in last['att_err_arcsec']],
        'final_att_err_norm_arcsec': last['att_err_norm_arcsec'],
        'final_yaw_abs_arcsec': last['yaw_abs_arcsec'],
        'outer_logs': outer_logs,
    }


def run_single(task: tuple[dict[str, Any], int]) -> dict[str, Any]:
    cfg, seed = task
    base12 = load_base12()
    acc18 = load_acc18()

    np.random.seed(seed)
    pos0 = base12.posset(34, 116, 480, isdeg=1)
    att0_ref, att_truth = build_single_axis_att(base12)
    imu, _ = base12.avp2imu(att_truth, pos0)
    imuerr = base12.build_imuerr()
    imu_noisy = base12.imuadderr(imu, imuerr)
    phi = PHI_DEG * base12.glv.deg
    att0_guess = base12.q2att(base12.qaddphi(base12.a2qua(att0_ref), phi))
    truth_att = att_truth[-1, 0:3]

    out = run_param_outer(base12, acc18, imu_noisy, att0_guess, pos0, phi, imuerr, truth_att, cfg)
    return {
        'label': cfg['label'],
        'feedback_scale': cfg['feedback_scale'],
        'outer_iterations': cfg['outer_iterations'],
        'seed': seed,
        **out,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errs = np.array([row['final_att_err_arcsec'] for row in rows], dtype=float)
    abs_errs = np.abs(errs)
    norms = np.array([row['final_att_err_norm_arcsec'] for row in rows], dtype=float)
    yaw_abs = np.array([row['final_yaw_abs_arcsec'] for row in rows], dtype=float)
    return {
        'pitch_mean_abs_arcsec': float(abs_errs[:, 1].mean()),
        'yaw_abs_mean_arcsec': float(yaw_abs.mean()),
        'norm_mean_arcsec': float(norms.mean()),
        'yaw_abs_median_arcsec': float(np.median(yaw_abs)),
        'yaw_abs_max_arcsec': float(yaw_abs.max()),
        'mean_signed_arcsec': errs.mean(axis=0).tolist(),
        'mean_abs_arcsec': abs_errs.mean(axis=0).tolist(),
        'per_seed_final_yaw_abs_arcsec': yaw_abs.tolist(),
        'per_seed_final_norm_arcsec': norms.tolist(),
        'per_seed': rows,
    }


def load_ref_best():
    payload = json.loads(REF_JSON.read_text(encoding='utf-8'))
    return payload['best_config']


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tasks = [(cfg, seed) for cfg in CONFIGS for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as ex:
        rows = list(ex.map(run_single, tasks))

    grouped = {cfg['label']: [] for cfg in CONFIGS}
    meta = {cfg['label']: cfg for cfg in CONFIGS}
    for row in rows:
        grouped[row['label']].append(row)

    ref_best = load_ref_best()
    summaries = []
    for label, cfg_rows in grouped.items():
        cfg_rows.sort(key=lambda x: x['seed'])
        stats = summarize_rows(cfg_rows)
        summaries.append({
            'label': label,
            **meta[label],
            **stats,
            'delta_vs_ref_yaw_arcsec': float(stats['yaw_abs_mean_arcsec'] - ref_best['yaw_abs_mean_arcsec']),
            'delta_vs_ref_pitch_arcsec': float(stats['pitch_mean_abs_arcsec'] - ref_best['pitch_mean_abs_arcsec']),
            'delta_vs_ref_norm_arcsec': float(stats['norm_mean_arcsec'] - ref_best['norm_mean_arcsec']),
        })

    summaries.sort(key=lambda x: (x['yaw_abs_mean_arcsec'], x['norm_mean_arcsec'], x['pitch_mean_abs_arcsec']))
    best = summaries[0]

    payload = {
        'task': 'sweep_g2_18state_param_outer_extrap4_2026_04_07',
        'reference_best': ref_best,
        'configs': CONFIGS,
        'best_config': best,
        'all_summaries': summaries,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = []
    lines.append('# G2 18-state param-outer extrap sweep 4 (2026-04-07)')
    lines.append('')
    lines.append('## 最优配置')
    lines.append('')
    lines.append(f"- label: `{best['label']}`")
    lines.append(f"- feedback_scale: `{best['feedback_scale']}`")
    lines.append(f"- outer_iterations: `{best['outer_iterations']}`")
    lines.append(f"- pitch mean abs: `{best['pitch_mean_abs_arcsec']:.6f}\"")
    lines.append(f"- yaw abs mean: `{best['yaw_abs_mean_arcsec']:.6f}\"")
    lines.append(f"- norm mean: `{best['norm_mean_arcsec']:.6f}\"")
    lines.append(f"- Δyaw vs ref: `{best['delta_vs_ref_yaw_arcsec']:+.6f}\"")
    lines.append('')
    lines.append('## 全部排序')
    lines.append('')
    for idx, rec in enumerate(summaries, start=1):
        lines.append(
            f"{idx}. `{rec['label']}` | fb={rec['feedback_scale']} | outer={rec['outer_iterations']} | "
            f"yaw=`{rec['yaw_abs_mean_arcsec']:.3f}\"` | pitch=`{rec['pitch_mean_abs_arcsec']:.3f}\"` | norm=`{rec['norm_mean_arcsec']:.3f}\"` | Δyaw=`{rec['delta_vs_ref_yaw_arcsec']:+.3f}\"`"
        )
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({
        'best_config': best,
        'top5': summaries[:5],
        'out_json': str(OUT_JSON),
        'out_md': str(OUT_MD),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
