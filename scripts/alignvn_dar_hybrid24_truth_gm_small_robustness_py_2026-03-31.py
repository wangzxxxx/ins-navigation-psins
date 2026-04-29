#!/usr/bin/env python3
"""Controlled SMALL-GM robustness check for Chapter 4 DAR plain24 vs staged24."""

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
OUT_JSON = OUT_DIR / 'ch4_plain24_staged24_truth_gm_small_robustness_2026-03-31.json'
OUT_MD = OUT_DIR / 'ch4_plain24_staged24_truth_gm_small_robustness_2026-03-31.md'
OUT_CSV = OUT_DIR / 'ch4_plain24_staged24_truth_gm_small_robustness_table_2026-03-31.csv'
BASE12_PATH = SCRIPTS_DIR / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
HYBRID24_PATH = SCRIPTS_DIR / 'alignvn_dar_hybrid24_staged_py_2026-03-30.py'
GM_HELPER_PATH = SCRIPTS_DIR / 'alignvn_dar_truth_gm_helper_2026-03-31.py'
MAX_WORKERS = min(4, os.cpu_count() or 1)
SEEDS = [0, 1, 2, 3, 4]
CONDITIONS = ['baseline', 'tiny_gm', 'small_gm']

_BASE12 = None
_HYBRID24 = None
_GM_HELPER = None


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
        _BASE12 = load_module('alignvn_base12_truth_gm_small_robust_20260331', BASE12_PATH)
    return _BASE12



def load_hybrid24():
    global _HYBRID24
    if _HYBRID24 is None:
        _HYBRID24 = load_module('alignvn_hybrid24_truth_gm_small_robust_20260331', HYBRID24_PATH)
    return _HYBRID24



def load_gm_helper():
    global _GM_HELPER
    if _GM_HELPER is None:
        _GM_HELPER = load_module('alignvn_truth_gm_small_helper_20260331', GM_HELPER_PATH)
    return _GM_HELPER



def build_method_configs(h24) -> dict[str, Any]:
    return {
        'plain24': h24.Hybrid24Config(
            name='plain24_iter5',
            label='plain24 iter=5',
            seeds=SEEDS,
            max_iter=5,
            staged_release=False,
            note='kg/ka active from iteration 1; otherwise identical to the current main comparison.',
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
            note='iter1 freezes kg/ka; iter>=2 releases them with the current main comparison gate.',
        ),
    }



def run_seed_condition(task: tuple[str, int]) -> dict[str, Any]:
    condition_name, seed = task
    base12 = load_base12()
    h24 = load_hybrid24()
    gm_helper = load_gm_helper()
    acc18 = h24.load_acc18()

    np.random.seed(seed)

    ts = 0.01
    att0 = np.array([0.0, 0.0, 0.0])
    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    rot_paras = base12.build_rot_paras()
    att_truth = acc18.attrottt(att0, rot_paras, ts)
    imu_clean, _ = acc18.avp2imu(att_truth, pos0)

    imuerr = gm_helper.build_truth_imuerr_variant(profile=condition_name)
    imu_noisy = gm_helper.apply_truth_imu_errors(imu_clean, imuerr)

    phi = np.array([0.1, 0.1, 0.5]) * acc18.glv.deg
    att0_guess = acc18.q2att(base12.qaddphi(acc18.a2qua(att0), phi))
    wvn = np.array([0.01, 0.01, 0.01])
    truth_att = att_truth[-1, 0:3]

    method_cfgs = build_method_configs(h24)
    out = {
        'condition': condition_name,
        'seed': seed,
        'profile': gm_helper.describe_truth_profile(condition_name),
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
        '# Chapter 4 DAR plain24 vs staged24 truth-side SMALL-GM robustness (2026-03-31)',
        '',
        '## What was added',
        '- kept the original truth-side injection path (`eb/db/web/wdb/dKg/dKa`) intact',
        '- then added an extra first-order Gauss-Markov bias drift on the truth IMU only',
        '- per sample: `b_k = exp(-dt/tau) * b_{k-1} + sigma * sqrt(1-exp(-2dt/tau)) * N(0,1)` and `imu += b_k * dt`',
        '- the filter model itself was not given explicit GM states; this is a controlled robustness test against unmodeled slow drift',
        '- here the GM levels are intentionally kept mild/interpretable for thesis-supplement sensitivity analysis',
        '',
        '## GM settings',
    ]
    for profile_name, profile in payload['profiles'].items():
        lines.append(
            f"- {profile_name}: gyro sigma={profile['gyro_sigma_dph']} deg/h, accel sigma={profile['accel_sigma_ug']} ug, tau_g={profile['tau_g_s']} s, tau_a={profile['tau_a_s']} s. {profile['note']}"
        )
    lines.extend([
        '',
        '## Controlled setup',
        '- same DAR path / same initial misalignment / same outer iteration count (`iter=5`) as the current main comparison',
        '- same seed set: `[0, 1, 2, 3, 4]`',
        '- for each `(condition, seed)`, one noisy IMU realization is generated and then reused for both plain24 and staged24',
        '',
        '## Summary table',
        '',
        '| condition | method | pitch mean abs (") | yaw abs mean (") | norm mean (") | yaw abs median (") | yaw abs max (") |',
        '|---|---|---:|---:|---:|---:|---:|',
    ])
    for row in payload['summary_rows']:
        lines.append(
            f"| {row['condition']} | {row['method']} | {row['pitch_mean_abs_arcsec']:.3f} | {row['yaw_abs_mean_arcsec']:.3f} | {row['norm_mean_arcsec']:.3f} | {row['yaw_abs_median_arcsec']:.3f} | {row['yaw_abs_max_arcsec']:.3f} |"
        )
    lines.extend([
        '',
        '## Verdict by condition',
        '',
    ])
    for item in payload['judgements']:
        lines.append(
            f"- {item['condition']}: {item['verdict']} (Δyaw_mean={item['delta_yaw_abs_mean_arcsec']}\", Δnorm_mean={item['delta_norm_mean_arcsec']}\")"
        )
    lines.extend([
        '',
        '## Files',
        f'- helper: `{GM_HELPER_PATH}`',
        f'- script: `{SCRIPTS_DIR / "alignvn_dar_hybrid24_truth_gm_small_robustness_py_2026-03-31.py"}`',
        f'- json: `{OUT_JSON}`',
        f'- md: `{OUT_MD}`',
        f'- csv: `{OUT_CSV}`',
        '',
    ])
    return '\n'.join(lines) + '\n'



def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    gm_helper = load_gm_helper()
    tasks = [(condition, seed) for condition in CONDITIONS for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as ex:
        seed_runs = list(ex.map(run_seed_condition, tasks))

    seed_runs.sort(key=lambda x: (x['condition'], x['seed']))

    grouped: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for item in seed_runs:
        grouped[item['condition']].append(item)

    profiles = {condition: gm_helper.describe_truth_profile(condition) for condition in CONDITIONS}
    summary_by_condition: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    judgements: list[dict[str, str]] = []

    for condition in CONDITIONS:
        rows = grouped[condition]
        plain_stats = summarize_method(rows, 'plain24')
        staged_stats = summarize_method(rows, 'staged24')
        summary_by_condition[condition] = {
            'profile': profiles[condition],
            'plain24': plain_stats,
            'staged24': staged_stats,
        }
        summary_rows.append({'condition': condition, 'method': 'plain24', **plain_stats})
        summary_rows.append({'condition': condition, 'method': 'staged24', **staged_stats})
        judgement = build_judgement(plain_stats, staged_stats)
        judgement['condition'] = condition
        judgements.append(judgement)

    csv_lines = ['condition,method,pitch_mean_abs_arcsec,yaw_abs_mean_arcsec,norm_mean_arcsec,yaw_abs_median_arcsec,yaw_abs_max_arcsec']
    for row in summary_rows:
        csv_lines.append(
            f"{row['condition']},{row['method']},{row['pitch_mean_abs_arcsec']:.6f},{row['yaw_abs_mean_arcsec']:.6f},{row['norm_mean_arcsec']:.6f},{row['yaw_abs_median_arcsec']:.6f},{row['yaw_abs_max_arcsec']:.6f}"
        )

    payload = {
        'meta': {
            'date': '2026-03-31',
            'purpose': 'controlled SMALL-GM robustness check for Chapter 4 DAR plain24 vs staged24 under added truth-side GM drift',
            'seeds': SEEDS,
            'conditions': CONDITIONS,
            'same_noisy_imu_reused_within_condition_seed': True,
            'outer_iteration_setup_matches_main_result': True,
            'runtime_sec': time.time() - t0,
        },
        'profiles': profiles,
        'seed_runs': seed_runs,
        'summary_by_condition': summary_by_condition,
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
