#!/usr/bin/env python3
"""Controlled robustness sweep for Chapter 4 DAR plain24 vs staged24.

Design choices:
- keep the same DAR path, same initial error, same outer-iteration setup as the
  current main comparison (`alignvn_dar_hybrid24_staged_py_2026-03-30.py`)
- scale `eb/db/web/wdb` together by a noise factor, because `build_imuerr()` is
  the shared source for both truth-side injection and the filter's assumed IMU
  error level / initial prior scale in the current code path
- keep diagonal `dKg/dKa` fixed at 30 ppm so the stronger-noise probe stays
  focused on random-error / bias-intensity robustness rather than changing the
  deterministic scale-factor mismatch itself
- for each `(noise_factor, seed)` pair, generate ONE noisy IMU realization and
  reuse it for both plain24 and staged24, ensuring same-path / same-seed /
  same-noisy-input comparison
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
SCRIPTS_DIR = WORKSPACE / 'scripts'
OUT_DIR = WORKSPACE / 'tmp' / 'alignment_strategy_sweep'
OUT_JSON = OUT_DIR / 'ch4_plain24_staged24_noise_robustness_2026-03-31.json'
OUT_MD = OUT_DIR / 'ch4_plain24_staged24_noise_robustness_2026-03-31.md'
OUT_CSV = OUT_DIR / 'ch4_plain24_staged24_noise_robustness_table_2026-03-31.csv'
BASE12_PATH = SCRIPTS_DIR / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
HYBRID24_PATH = SCRIPTS_DIR / 'alignvn_dar_hybrid24_staged_py_2026-03-30.py'
MAX_WORKERS = min(4, os.cpu_count() or 1)
SEEDS = [0, 1, 2, 3, 4]
NOISE_FACTORS = [1.0, 2.0, 5.0]

_BASE12 = None
_HYBRID24 = None


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module



def load_base12():
    global _BASE12
    if _BASE12 is None:
        _BASE12 = load_module('alignvn_base12_noise_robust_20260331', BASE12_PATH)
    return _BASE12



def load_hybrid24():
    global _HYBRID24
    if _HYBRID24 is None:
        _HYBRID24 = load_module('alignvn_hybrid24_noise_robust_20260331', HYBRID24_PATH)
    return _HYBRID24



def build_scaled_imuerr(noise_factor: float) -> dict[str, np.ndarray]:
    base12 = load_base12()
    imuerr = base12.build_imuerr()
    scaled: dict[str, np.ndarray] = {}
    for key, value in imuerr.items():
        arr = np.array(value, copy=True)
        if key in ('eb', 'db', 'web', 'wdb'):
            arr = arr * noise_factor
        scaled[key] = arr
    return scaled



def build_method_configs(h24) -> dict[str, Any]:
    return {
        'plain24': h24.Hybrid24Config(
            name='plain24_iter5',
            label='plain24 iter=5',
            seeds=SEEDS,
            max_iter=5,
            staged_release=False,
            note='kg/ka active from iteration 1; otherwise identical to the current main result setup.',
        ),
        'staged24': h24.Hybrid24Config(
            name='staged24_iter5',
            label='staged24 iter=5',
            seeds=SEEDS,
            max_iter=5,
            staged_release=True,
            release_iter=2,
            rot_gate_dps=5.0,
            scale_wash_scale=0.5,
            note='iter1 freezes kg/ka; iter>=2 releases them with the same gate as the current main result setup.',
        ),
    }



def run_seed_noise(task: tuple[float, int]) -> dict[str, Any]:
    noise_factor, seed = task
    base12 = load_base12()
    h24 = load_hybrid24()
    acc18 = h24.load_acc18()

    np.random.seed(seed)

    ts = 0.01
    att0 = np.array([0.0, 0.0, 0.0])
    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    rot_paras = base12.build_rot_paras()
    att_truth = acc18.attrottt(att0, rot_paras, ts)
    imu_clean, _ = acc18.avp2imu(att_truth, pos0)

    imuerr = build_scaled_imuerr(noise_factor)
    imu_noisy = acc18.imuadderr(imu_clean, imuerr)

    phi = np.array([0.1, 0.1, 0.5]) * acc18.glv.deg
    att0_guess = acc18.q2att(base12.qaddphi(acc18.a2qua(att0), phi))
    wvn = np.array([0.01, 0.01, 0.01])
    truth_att = att_truth[-1, 0:3]

    method_cfgs = build_method_configs(h24)
    out = {
        'noise_factor': noise_factor,
        'seed': seed,
        'methods': {},
    }
    for method_name, cfg in method_cfgs.items():
        _, iter_logs = h24.alignvn_24state_iter(
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
        out['methods'][method_name] = {
            'final_att_err_arcsec': [float(x) for x in last['att_err_arcsec']],
            'final_att_err_abs_arcsec': [float(abs(x)) for x in last['att_err_arcsec']],
            'final_att_err_norm_arcsec': float(last['att_err_norm_arcsec']),
            'final_yaw_abs_arcsec': float(last['yaw_abs_arcsec']),
            'iter_logs': iter_logs,
        }
    return out



def summarize_method(rows: list[dict[str, Any]], method_name: str) -> dict[str, Any]:
    errs = np.array([row['methods'][method_name]['final_att_err_arcsec'] for row in rows], dtype=float)
    abs_errs = np.abs(errs)
    norms = np.array([row['methods'][method_name]['final_att_err_norm_arcsec'] for row in rows], dtype=float)
    yaw_abs = np.array([row['methods'][method_name]['final_yaw_abs_arcsec'] for row in rows], dtype=float)
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
    }



def build_judgement(plain_stats: dict[str, Any], staged_stats: dict[str, Any]) -> dict[str, str]:
    yaw_better = staged_stats['yaw_abs_mean_arcsec'] < plain_stats['yaw_abs_mean_arcsec']
    norm_better = staged_stats['norm_mean_arcsec'] < plain_stats['norm_mean_arcsec']
    if yaw_better and norm_better:
        verdict = 'staged still beats plain on both yaw and norm'
    elif (not yaw_better) and (not norm_better):
        verdict = 'staged advantage disappears or reverses on both yaw and norm'
    elif yaw_better and (not norm_better):
        verdict = 'staged still wins on yaw, but its norm advantage disappears/reverses'
    else:
        verdict = 'staged still wins on norm, but its yaw advantage disappears/reverses'
    return {
        'verdict': verdict,
        'delta_yaw_abs_mean_arcsec': f"{staged_stats['yaw_abs_mean_arcsec'] - plain_stats['yaw_abs_mean_arcsec']:+.3f}",
        'delta_norm_mean_arcsec': f"{staged_stats['norm_mean_arcsec'] - plain_stats['norm_mean_arcsec']:+.3f}",
    }



def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        '# Chapter 4 DAR plain24 vs staged24 noise robustness (2026-03-31)',
        '',
        '## Controlled setup',
        '- same DAR path / same initial misalignment / same outer iteration count (`iter=5`) as the current main result',
        '- same seed set: `[0, 1, 2, 3, 4]`',
        '- for each `(noise_factor, seed)`, the same noisy IMU realization is reused for both plain24 and staged24',
        '- scaled together: `eb/db/web/wdb`',
        '- kept fixed: diagonal `dKg/dKa = 30 ppm`',
        '',
        '## Summary table',
        '',
        '| noise | method | pitch mean abs (") | yaw abs mean (") | norm mean (") | yaw abs median (") | yaw abs max (") |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in payload['summary_rows']:
        lines.append(
            f"| {row['noise_label']} | {row['method']} | {row['pitch_mean_abs_arcsec']:.3f} | {row['yaw_abs_mean_arcsec']:.3f} | {row['norm_mean_arcsec']:.3f} | {row['yaw_abs_median_arcsec']:.3f} | {row['yaw_abs_max_arcsec']:.3f} |"
        )
    lines.extend([
        '',
        '## Noise-level verdicts',
        '',
    ])
    for item in payload['judgements']:
        lines.append(
            f"- {item['noise_label']}: {item['verdict']} (Δyaw_mean={item['delta_yaw_abs_mean_arcsec']}\", Δnorm_mean={item['delta_norm_mean_arcsec']}\")"
        )
    lines.extend([
        '',
        '## Files',
        f'- script: `{SCRIPTS_DIR / "run_ch4_plain24_staged24_noise_robustness_2026-03-31.py"}`',
        f'- json: `{OUT_JSON}`',
        f'- md: `{OUT_MD}`',
        f'- csv: `{OUT_CSV}`',
        '',
    ])
    return '\n'.join(lines) + '\n'



def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(noise_factor, seed) for noise_factor in NOISE_FACTORS for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as ex:
        seed_runs = list(ex.map(run_seed_noise, tasks))

    seed_runs.sort(key=lambda x: (x['noise_factor'], x['seed']))

    grouped: dict[float, list[dict[str, Any]]] = {factor: [] for factor in NOISE_FACTORS}
    for item in seed_runs:
        grouped[item['noise_factor']].append(item)

    summary_by_noise: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    judgements: list[dict[str, str]] = []

    for factor in NOISE_FACTORS:
        rows = grouped[factor]
        plain_stats = summarize_method(rows, 'plain24')
        staged_stats = summarize_method(rows, 'staged24')
        noise_label = f'noise_x{int(factor) if float(factor).is_integer() else factor}'
        summary_by_noise[noise_label] = {
            'noise_factor': factor,
            'plain24': plain_stats,
            'staged24': staged_stats,
        }
        summary_rows.append({'noise_label': noise_label, 'method': 'plain24', **plain_stats})
        summary_rows.append({'noise_label': noise_label, 'method': 'staged24', **staged_stats})
        judgement = build_judgement(plain_stats, staged_stats)
        judgement['noise_label'] = noise_label
        judgements.append(judgement)

    csv_lines = ['noise,method,pitch_mean_abs_arcsec,yaw_abs_mean_arcsec,norm_mean_arcsec,yaw_abs_median_arcsec,yaw_abs_max_arcsec']
    for row in summary_rows:
        csv_lines.append(
            f"{row['noise_label']},{row['method']},{row['pitch_mean_abs_arcsec']:.6f},{row['yaw_abs_mean_arcsec']:.6f},{row['norm_mean_arcsec']:.6f},{row['yaw_abs_median_arcsec']:.6f},{row['yaw_abs_max_arcsec']:.6f}"
        )

    payload = {
        'meta': {
            'date': '2026-03-31',
            'purpose': 'controlled robustness check for Chapter 4 DAR plain24 vs staged24 under stronger IMU errors',
            'seeds': SEEDS,
            'noise_factors': NOISE_FACTORS,
            'scaled_terms': ['eb', 'db', 'web', 'wdb'],
            'fixed_terms': ['dKg', 'dKa'],
            'same_noisy_imu_reused_within_seed': True,
            'outer_iteration_setup_matches_main_result': True,
            'runtime_sec': time.time() - t0,
        },
        'seed_runs': seed_runs,
        'summary_by_noise': summary_by_noise,
        'summary_rows': summary_rows,
        'judgements': judgements,
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    OUT_MD.write_text(build_markdown(payload))
    OUT_CSV.write_text('\n'.join(csv_lines) + '\n')

    print(json.dumps({
        'summary_rows': summary_rows,
        'judgements': judgements,
        'out_json': str(OUT_JSON),
        'out_md': str(OUT_MD),
        'out_csv': str(OUT_CSV),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
