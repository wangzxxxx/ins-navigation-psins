#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
SCRIPTS_DIR = WORKSPACE / 'scripts'
OUT_DIR = WORKSPACE / 'tmp' / 'alignment_strategy_sweep'
OUT_JSON = OUT_DIR / 'ch4_yaw_hold_baseline_probe_2026-03-31.json'
OUT_MD = OUT_DIR / 'ch4_yaw_hold_baseline_probe_2026-03-31.md'
OUT_CSV = OUT_DIR / 'ch4_yaw_hold_baseline_probe_table_2026-03-31.csv'
YAW_HOLD_PATH = SCRIPTS_DIR / 'alignvn_dar_hybrid24_truth_gm_yaw_hold_probe_2026-03-31.py'
MATCHED_REF_JSON = OUT_DIR / 'ch4_plain24_staged24_truth_gm_matched_2026-03-31.json'
SEEDS = [0, 1, 2, 3, 4]

_mod = None


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_mod():
    global _mod
    if _mod is None:
        _mod = load_module('alignvn_yaw_hold_baseline_20260331', YAW_HOLD_PATH)
    return _mod


def run_seed(seed: int) -> dict[str, Any]:
    mod = load_mod()
    base12 = mod.load_base12()
    h24 = mod.load_hybrid24()
    gm_helper = mod.load_gm_helper()
    acc18 = h24.load_acc18()

    np.random.seed(seed)
    ts = 0.01
    att0 = np.array([0.0, 0.0, 0.0])
    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    rot_paras = base12.build_rot_paras()
    att_truth = acc18.attrottt(att0, rot_paras, ts)
    imu_clean, _ = acc18.avp2imu(att_truth, pos0)

    imuerr = gm_helper.build_truth_imuerr_variant(profile='baseline')
    imu_noisy = gm_helper.apply_truth_imu_errors(imu_clean, imuerr)

    phi = np.array([0.1, 0.1, 0.5]) * acc18.glv.deg
    att0_guess = acc18.q2att(base12.qaddphi(acc18.a2qua(att0), phi))
    wvn = np.array([0.01, 0.01, 0.01])
    truth_att = att_truth[-1, 0:3]

    cfg = mod.MethodConfig(
        name='staged24_yaw_hold_kgz_iter3_baseline',
        label='yaw_hold24 baseline',
        seeds=SEEDS,
        max_iter=5,
        staged_release=True,
        release_iter=2,
        yaw_gyro_release_iter=3,
        note='Baseline truth (no extra GM). Same as yaw_hold24 but evaluated on baseline/no-GM data.',
    )

    _, iter_logs = mod.alignvn_24state_iter_yaw_hold(
        imu=imu_noisy.copy(),
        qnb=att0_guess,
        pos=pos0,
        phi0=phi,
        imuerr=imuerr,
        wvn=wvn,
        cfg=cfg,
        truth_att=truth_att,
    )
    last = iter_logs[-1]
    return {
        'seed': seed,
        'final_att_err_arcsec': [float(x) for x in last['att_err_arcsec']],
        'final_att_err_abs_arcsec': [float(abs(x)) for x in last['att_err_arcsec']],
        'final_att_err_norm_arcsec': float(last['att_err_norm_arcsec']),
        'final_yaw_abs_arcsec': float(last['yaw_abs_arcsec']),
        'final_iter_gate_stats': {
            'high_rot_ratio': float(last['high_rot_ratio']),
            'yaw_kgz_coupled_ratio': float(last['yaw_kgz_coupled_ratio']),
            'yaw_gyro_scale_active': bool(last['yaw_gyro_scale_active']),
        },
        'iter_logs': iter_logs,
    }


def load_reference_baseline_rows() -> dict[str, Any]:
    obj = json.loads(MATCHED_REF_JSON.read_text(encoding='utf-8'))
    rows = [r for r in obj['summary_rows'] if r['condition'] == 'baseline']
    return {r['method']: r for r in rows}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errs = np.array([row['final_att_err_arcsec'] for row in rows], dtype=float)
    abs_errs = np.abs(errs)
    norms = np.array([row['final_att_err_norm_arcsec'] for row in rows], dtype=float)
    yaw_abs = np.array([row['final_yaw_abs_arcsec'] for row in rows], dtype=float)
    gate_rows = [row['final_iter_gate_stats'] for row in rows]
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
        'final_iter_gate_mean': {
            'high_rot_ratio': float(np.mean([g['high_rot_ratio'] for g in gate_rows])),
            'yaw_kgz_coupled_ratio': float(np.mean([g['yaw_kgz_coupled_ratio'] for g in gate_rows])),
        },
    }


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ref = load_reference_baseline_rows()
    runs = [run_seed(seed) for seed in SEEDS]
    yaw_hold = summarize(runs)

    delta_vs_staged = {
        'pitch_mean_abs_arcsec': yaw_hold['pitch_mean_abs_arcsec'] - ref['staged24']['pitch_mean_abs_arcsec'],
        'yaw_abs_mean_arcsec': yaw_hold['yaw_abs_mean_arcsec'] - ref['staged24']['yaw_abs_mean_arcsec'],
        'norm_mean_arcsec': yaw_hold['norm_mean_arcsec'] - ref['staged24']['norm_mean_arcsec'],
        'yaw_abs_median_arcsec': yaw_hold['yaw_abs_median_arcsec'] - ref['staged24']['yaw_abs_median_arcsec'],
        'yaw_abs_max_arcsec': yaw_hold['yaw_abs_max_arcsec'] - ref['staged24']['yaw_abs_max_arcsec'],
    }

    summary_rows = [
        {'condition': 'baseline', 'method': 'plain24', **ref['plain24']},
        {'condition': 'baseline', 'method': 'staged24', **ref['staged24']},
        {'condition': 'baseline', 'method': 'yaw_hold24', **yaw_hold},
    ]
    for row in summary_rows:
        row.pop('condition', None)
        row.pop('method', None)
    summary_rows = [
        {'condition': 'baseline', 'method': 'plain24', **ref['plain24']},
        {'condition': 'baseline', 'method': 'staged24', **ref['staged24']},
        {'condition': 'baseline', 'method': 'yaw_hold24', **yaw_hold},
    ]

    csv_lines = ['condition,method,pitch_mean_abs_arcsec,yaw_abs_mean_arcsec,norm_mean_arcsec,yaw_abs_median_arcsec,yaw_abs_max_arcsec']
    for row in summary_rows:
        csv_lines.append(
            f"{row['condition']},{row['method']},{row['pitch_mean_abs_arcsec']:.6f},{row['yaw_abs_mean_arcsec']:.6f},{row['norm_mean_arcsec']:.6f},{row['yaw_abs_median_arcsec']:.6f},{row['yaw_abs_max_arcsec']:.6f}"
        )

    md = f'''# Chapter 4 DAR baseline yaw-hold probe (2026-03-31)

## Goal
- Evaluate `yaw_hold24` on the **baseline / no extra GM drift** condition.
- Use the same DAR path, same seeds `[0,1,2,3,4]`, same `iter=5` setup.
- Compare directly against the existing baseline `plain24` and `staged24` rows.

## What changed
- No extra truth-side GM drift was added.
- The filter-side setting stays at the current Chapter-4 baseline reference.
- The only method change remains: keep `kg_z` frozen for one extra outer iteration, i.e. release `kg_z` from `iter>=3` instead of `iter>=2`.

## Summary table

| condition | method | pitch mean abs (") | yaw abs mean (") | norm mean (") | yaw abs median (") | yaw abs max (") |
|---|---|---:|---:|---:|---:|---:|
| baseline | plain24 | {ref['plain24']['pitch_mean_abs_arcsec']:.3f} | {ref['plain24']['yaw_abs_mean_arcsec']:.3f} | {ref['plain24']['norm_mean_arcsec']:.3f} | {ref['plain24']['yaw_abs_median_arcsec']:.3f} | {ref['plain24']['yaw_abs_max_arcsec']:.3f} |
| baseline | staged24 | {ref['staged24']['pitch_mean_abs_arcsec']:.3f} | {ref['staged24']['yaw_abs_mean_arcsec']:.3f} | {ref['staged24']['norm_mean_arcsec']:.3f} | {ref['staged24']['yaw_abs_median_arcsec']:.3f} | {ref['staged24']['yaw_abs_max_arcsec']:.3f} |
| baseline | yaw_hold24 | {yaw_hold['pitch_mean_abs_arcsec']:.3f} | {yaw_hold['yaw_abs_mean_arcsec']:.3f} | {yaw_hold['norm_mean_arcsec']:.3f} | {yaw_hold['yaw_abs_median_arcsec']:.3f} | {yaw_hold['yaw_abs_max_arcsec']:.3f} |

## Verdict vs current staged24 under baseline

- Δpitch_mean_abs = {delta_vs_staged['pitch_mean_abs_arcsec']:+.3f}"
- Δyaw_mean = {delta_vs_staged['yaw_abs_mean_arcsec']:+.3f}"
- Δnorm_mean = {delta_vs_staged['norm_mean_arcsec']:+.3f}"
- Δyaw_median = {delta_vs_staged['yaw_abs_median_arcsec']:+.3f}"
- Δyaw_max = {delta_vs_staged['yaw_abs_max_arcsec']:+.3f}"

## Final-iteration activation diagnostic

- high-rotation ratio = {yaw_hold['final_iter_gate_mean']['high_rot_ratio']:.3f}
- yaw `kg_z` coupled ratio within high-rotation steps = {yaw_hold['final_iter_gate_mean']['yaw_kgz_coupled_ratio']:.3f}

## Files
- json: `{OUT_JSON}`
- md: `{OUT_MD}`
- csv: `{OUT_CSV}`
'''

    payload = {
        'meta': {
            'date': '2026-03-31',
            'condition': 'baseline',
            'seeds': SEEDS,
            'method_change': 'hold kg_z one extra outer iteration; release from iter>=3 instead of iter>=2',
            'runtime_s': round(time.time() - t0, 3),
        },
        'summary_rows': summary_rows,
        'yaw_hold_summary': yaw_hold,
        'delta_vs_staged24': delta_vs_staged,
        'seed_runs': runs,
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    OUT_MD.write_text(md, encoding='utf-8')
    OUT_CSV.write_text('\n'.join(csv_lines) + '\n', encoding='utf-8')
    print(json.dumps({'summary_rows': summary_rows, 'delta_vs_staged24': delta_vs_staged, 'out_json': str(OUT_JSON)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
