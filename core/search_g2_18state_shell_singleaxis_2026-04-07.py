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
BASELINE_JSON = WORKSPACE / 'psins_method_bench' / 'results' / 'compare_four_group_alignment_arcsec_g2_18state_2026-04-07.json'
RESULTS_DIR = WORKSPACE / 'psins_method_bench' / 'results'
REPORTS_DIR = WORKSPACE / 'reports'
OUT_JSON = RESULTS_DIR / 'search_g2_18state_shell_singleaxis_2026-04-07.json'
OUT_MD = REPORTS_DIR / 'psins_g2_18state_shell_singleaxis_2026-04-07.md'

MAX_WORKERS = min(4, os.cpu_count() or 1)
SEEDS = [0, 1, 2, 3, 4]
TS = 0.01
SINGLE_AXIS_ATT0_DEG = np.array([1.0, 0.0, 10.0])
ITER_CHOICES = [1, 2, 3, 5]
CARRY_CHOICES = [False, True]
WASH_CHOICES = [0.0, 0.05, 0.1, 0.2, 0.5]

NG_SIGMA_DPH = np.array([0.05, 0.05, 0.05])
TAU_G_S = np.array([300.0, 300.0, 300.0])
XA_SIGMA_UG = np.array([0.01, 0.01, 0.01])
TAU_A_S = np.array([100.0, 100.0, 100.0])
WVN = np.array([0.01, 0.01, 0.01])
PHI_DEG = np.array([0.1, 0.1, 0.5])


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
        _BASE12 = load_module('g2shell_base12_20260407', BASE12_PATH)
    return _BASE12


def load_acc18():
    global _ACC18
    if _ACC18 is None:
        _ACC18 = load_module('g2shell_acc18_20260407', ACC18_PATH)
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
        }
    return {
        'att_err_arcsec': [float(x) for x in last['att_err_arcsec']],
        'att_err_norm_arcsec': float(last['att_err_norm_arcsec']),
        'yaw_abs_arcsec': float(last['yaw_abs_arcsec']),
        'final_att_deg': [float(x) for x in last['final_att_deg']],
    }


def run_single(task: tuple[int, int, bool, float]) -> dict[str, Any]:
    seed, max_iter, carry_att_seed, wash_scale = task
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

    _, _, _, iter_logs = acc18.alignvn_18state_iter(
        imu=imu_noisy.copy(),
        qnb=att0_guess,
        pos=pos0,
        phi0=phi,
        imuerr=imuerr,
        wvn=WVN.copy(),
        max_iter=max_iter,
        truth_att=truth_att,
        ng_sigma=NG_SIGMA_DPH * base12.glv.dph,
        tau_g_s=TAU_G_S.copy(),
        xa_sigma=XA_SIGMA_UG * base12.glv.ug,
        tau_a_s=TAU_A_S.copy(),
        wash_scale=wash_scale,
        carry_att_seed=carry_att_seed,
    )
    last = extract_last_metrics(iter_logs[-1])
    err = np.array(last['att_err_arcsec'], dtype=float)
    return {
        'seed': seed,
        'max_iter': max_iter,
        'carry_att_seed': carry_att_seed,
        'wash_scale': wash_scale,
        'final_att_deg': last['final_att_deg'],
        'final_att_err_arcsec': [float(x) for x in err],
        'final_att_err_abs_arcsec': [float(x) for x in np.abs(err)],
        'final_att_err_norm_arcsec': float(last['att_err_norm_arcsec']),
        'final_yaw_abs_arcsec': float(last['yaw_abs_arcsec']),
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


def config_key(max_iter: int, carry_att_seed: bool, wash_scale: float) -> str:
    carry_tag = 'carry' if carry_att_seed else 'nocarry'
    wash_tag = str(wash_scale).replace('.', 'p')
    return f'iter{max_iter}_{carry_tag}_wash{wash_tag}'


def load_baseline() -> dict[str, Any]:
    return json.loads(BASELINE_JSON.read_text(encoding='utf-8'))


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = []
    for max_iter in ITER_CHOICES:
        for carry_att_seed in CARRY_CHOICES:
            for wash_scale in WASH_CHOICES:
                for seed in SEEDS:
                    tasks.append((seed, max_iter, carry_att_seed, wash_scale))

    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as ex:
        rows = list(ex.map(run_single, tasks))

    grouped: dict[str, list[dict[str, Any]]] = {}
    config_meta: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = config_key(row['max_iter'], row['carry_att_seed'], row['wash_scale'])
        grouped.setdefault(key, []).append(row)
        config_meta[key] = {
            'max_iter': row['max_iter'],
            'carry_att_seed': row['carry_att_seed'],
            'wash_scale': row['wash_scale'],
        }

    summaries = []
    for key, cfg_rows in grouped.items():
        cfg_rows = sorted(cfg_rows, key=lambda x: x['seed'])
        stats = summarize_rows(cfg_rows)
        rec = {
            'config_key': key,
            **config_meta[key],
            **stats,
        }
        summaries.append(rec)

    summaries.sort(key=lambda x: (x['yaw_abs_mean_arcsec'], x['norm_mean_arcsec'], x['pitch_mean_abs_arcsec']))

    baseline = load_baseline()
    g1 = next(row for row in baseline['summary_rows'] if row['group_key'] == 'g1_plain12_singleaxis')
    g2_bad = next(row for row in baseline['summary_rows'] if row['group_key'] == 'g2_markov_singleaxis_18state')

    best = summaries[0]
    for rec in summaries:
        rec['vs_g1_yaw_gain_arcsec'] = float(g1['yaw_abs_mean_arcsec'] - rec['yaw_abs_mean_arcsec'])
        rec['vs_g1_norm_gain_arcsec'] = float(g1['norm_mean_arcsec'] - rec['norm_mean_arcsec'])
        rec['vs_bad_iter5_yaw_gain_arcsec'] = float(g2_bad['yaw_abs_mean_arcsec'] - rec['yaw_abs_mean_arcsec'])
        rec['vs_bad_iter5_norm_gain_arcsec'] = float(g2_bad['norm_mean_arcsec'] - rec['norm_mean_arcsec'])

    stable_frontier = [
        rec for rec in summaries
        if rec['yaw_abs_mean_arcsec'] < g1['yaw_abs_mean_arcsec'] and rec['pitch_mean_abs_arcsec'] < g1['pitch_mean_abs_arcsec']
    ]

    payload = {
        'task': 'search_g2_18state_shell_singleaxis_2026_04_07',
        'baseline_reference': str(BASELINE_JSON),
        'search_space': {
            'iter_choices': ITER_CHOICES,
            'carry_choices': CARRY_CHOICES,
            'wash_choices': WASH_CHOICES,
            'seeds': SEEDS,
            'single_axis_att0_deg': SINGLE_AXIS_ATT0_DEG.tolist(),
            'phi_deg': PHI_DEG.tolist(),
            'ng_sigma_dph': NG_SIGMA_DPH.tolist(),
            'tau_g_s': TAU_G_S.tolist(),
            'xa_sigma_ug': XA_SIGMA_UG.tolist(),
            'tau_a_s': TAU_A_S.tolist(),
        },
        'baseline_rows': {
            'g1_plain12_singleaxis': g1,
            'g2_markov_singleaxis_18state_bad_iter5': g2_bad,
        },
        'best_config': best,
        'stable_frontier': stable_frontier,
        'top10_by_yaw': summaries[:10],
        'all_summaries': summaries,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = []
    lines.append('# G2 18-state 单轴外壳重调结果（2026-04-07）')
    lines.append('')
    lines.append('## 最优配置')
    lines.append('')
    lines.append(f"- config: `{best['config_key']}`")
    lines.append(f"- max_iter: `{best['max_iter']}`")
    lines.append(f"- carry_att_seed: `{best['carry_att_seed']}`")
    lines.append(f"- wash_scale: `{best['wash_scale']}`")
    lines.append(f"- pitch mean abs: `{best['pitch_mean_abs_arcsec']:.6f}\"")
    lines.append(f"- yaw abs mean: `{best['yaw_abs_mean_arcsec']:.6f}\"")
    lines.append(f"- norm mean: `{best['norm_mean_arcsec']:.6f}\"")
    lines.append(f"- vs G1 yaw gain: `{best['vs_g1_yaw_gain_arcsec']:.6f}\"")
    lines.append(f"- vs bad iter5 yaw gain: `{best['vs_bad_iter5_yaw_gain_arcsec']:.6f}\"")
    lines.append('')
    lines.append('## Top 10')
    lines.append('')
    for idx, rec in enumerate(summaries[:10], start=1):
        lines.append(
            f"{idx}. `{rec['config_key']}` | yaw=`{rec['yaw_abs_mean_arcsec']:.3f}\"` | "
            f"pitch=`{rec['pitch_mean_abs_arcsec']:.3f}\"` | norm=`{rec['norm_mean_arcsec']:.3f}\"`"
        )
    lines.append('')
    lines.append(f"JSON: `{OUT_JSON}`")
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps({
        'best_config': best,
        'top5_by_yaw': summaries[:5],
        'stable_frontier_count': len(stable_frontier),
        'out_json': str(OUT_JSON),
        'out_md': str(OUT_MD),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
