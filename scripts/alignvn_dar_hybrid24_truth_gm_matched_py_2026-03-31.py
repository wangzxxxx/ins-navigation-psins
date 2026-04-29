#!/usr/bin/env python3
"""Matched-GM Chapter 4 DAR comparison for plain24 vs staged24.

Goal:
- keep the same truth-side GM drift injection used in the earlier 2026-03-31
  robustness probes
- but, for the GM cases, also set the filter-side 24-state ng/xa GM parameters
  to the same stationary sigma/tau as the truth-side GM process
- therefore distinguish pure model mismatch from intrinsic plain24 vs staged24
  behavior under the current 24-state formulation

Conservative mapping choice used here:
- truth helper GM uses stationary sigma + correlation time tau for first-order GM
  bias drift, injected as bias * dt on the IMU increments
- 24-state filter ng/xa states also use stationary sigma + tau in avnkfinit_24
- so the cleanest matched mapping is direct 1:1:
    truth gyro_sigma_dph -> cfg.ng_sigma_dph
    truth tau_g_s       -> cfg.tau_g_s
    truth accel_sigma_ug -> cfg.xa_sigma_ug
    truth tau_a_s        -> cfg.tau_a_s
- baseline remains the current Chapter-4 reference setup (truth baseline + default
  filter ng/xa settings), because there is no extra truth GM process to match
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
SCRIPTS_DIR = WORKSPACE / 'scripts'
OUT_DIR = WORKSPACE / 'tmp' / 'alignment_strategy_sweep'
OUT_JSON = OUT_DIR / 'ch4_plain24_staged24_truth_gm_matched_2026-03-31.json'
OUT_MD = OUT_DIR / 'ch4_plain24_staged24_truth_gm_matched_2026-03-31.md'
OUT_CSV = OUT_DIR / 'ch4_plain24_staged24_truth_gm_matched_table_2026-03-31.csv'
BASE12_PATH = SCRIPTS_DIR / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
HYBRID24_PATH = SCRIPTS_DIR / 'alignvn_dar_hybrid24_staged_py_2026-03-30.py'
GM_HELPER_PATH = SCRIPTS_DIR / 'alignvn_dar_truth_gm_helper_2026-03-31.py'
UNMATCHED_JSON = OUT_DIR / 'ch4_plain24_staged24_truth_gm_small_robustness_2026-03-31.json'
MAX_WORKERS = min(4, os.cpu_count() or 1)
SEEDS = [0, 1, 2, 3, 4]
CONDITIONS = [
    'baseline',
    'tiny_gm_matched',
    'small_gm_matched',
]

_BASE12 = None
_HYBRID24 = None
_GM_HELPER = None


DEFAULT_FILTER_GM = {
    'ng_sigma_dph': [0.05, 0.05, 0.05],
    'tau_g_s': [300.0, 300.0, 300.0],
    'xa_sigma_ug': [0.01, 0.01, 0.01],
    'tau_a_s': [100.0, 100.0, 100.0],
    'source': 'current Hybrid24Config defaults in alignvn_dar_hybrid24_staged_py_2026-03-30.py',
}


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
        _BASE12 = load_module('alignvn_base12_truth_gm_matched_20260331', BASE12_PATH)
    return _BASE12



def load_hybrid24():
    global _HYBRID24
    if _HYBRID24 is None:
        _HYBRID24 = load_module('alignvn_hybrid24_truth_gm_matched_20260331', HYBRID24_PATH)
    return _HYBRID24



def load_gm_helper():
    global _GM_HELPER
    if _GM_HELPER is None:
        _GM_HELPER = load_module('alignvn_truth_gm_matched_helper_20260331', GM_HELPER_PATH)
    return _GM_HELPER



def get_condition_spec(condition_name: str) -> dict[str, Any]:
    if condition_name == 'baseline':
        return {
            'condition': 'baseline',
            'truth_profile': 'baseline',
            'filter_mode': 'chapter4_default_reference',
            'label': 'baseline',
            'note': 'No added truth-side GM. Keep the current Chapter-4 24-state filter default ng/xa sigma/tau as the reference baseline.',
        }
    if condition_name == 'tiny_gm_matched':
        return {
            'condition': 'tiny_gm_matched',
            'truth_profile': 'tiny_gm',
            'filter_mode': 'matched_truth_gm',
            'label': 'tiny_gm_matched',
            'note': 'Truth uses tiny GM drift; filter ng/xa sigma/tau are matched 1:1 to the same tiny GM stationary sigma/tau.',
        }
    if condition_name == 'small_gm_matched':
        return {
            'condition': 'small_gm_matched',
            'truth_profile': 'small_gm',
            'filter_mode': 'matched_truth_gm',
            'label': 'small_gm_matched',
            'note': 'Truth uses small GM drift; filter ng/xa sigma/tau are matched 1:1 to the same small GM stationary sigma/tau.',
        }
    raise KeyError(f'unknown condition: {condition_name}')



def build_filter_gm_setting(condition_name: str) -> dict[str, Any]:
    gm_helper = load_gm_helper()
    spec = get_condition_spec(condition_name)
    truth_profile = gm_helper.describe_truth_profile(spec['truth_profile'])

    if spec['filter_mode'] == 'chapter4_default_reference':
        return {
            'mode': 'chapter4_default_reference',
            'matched': False,
            'ng_sigma_dph': list(DEFAULT_FILTER_GM['ng_sigma_dph']),
            'tau_g_s': list(DEFAULT_FILTER_GM['tau_g_s']),
            'xa_sigma_ug': list(DEFAULT_FILTER_GM['xa_sigma_ug']),
            'tau_a_s': list(DEFAULT_FILTER_GM['tau_a_s']),
            'source': DEFAULT_FILTER_GM['source'],
            'mapping_note': (
                'Baseline has no extra truth GM process, so it is kept as the current Chapter-4 '
                'reference rather than forcing ng/xa to zero.'
            ),
        }

    return {
        'mode': 'matched_truth_gm',
        'matched': True,
        'ng_sigma_dph': list(truth_profile['gyro_sigma_dph']),
        'tau_g_s': list(truth_profile['tau_g_s']),
        'xa_sigma_ug': list(truth_profile['accel_sigma_ug']),
        'tau_a_s': list(truth_profile['tau_a_s']),
        'source': 'direct 1:1 mapping from truth GM helper profile',
        'mapping_note': (
            'Direct sigma/tau match. This is the most conservative reasonable choice because both '
            'the truth-side GM injector and the filter-side ng/xa states use stationary sigma + tau '
            'for first-order GM evolution.'
        ),
    }



def build_method_configs(h24, filter_gm: dict[str, Any]) -> dict[str, Any]:
    common = dict(
        seeds=SEEDS,
        max_iter=5,
        ng_sigma_dph=list(filter_gm['ng_sigma_dph']),
        tau_g_s=list(filter_gm['tau_g_s']),
        xa_sigma_ug=list(filter_gm['xa_sigma_ug']),
        tau_a_s=list(filter_gm['tau_a_s']),
    )
    return {
        'plain24': h24.Hybrid24Config(
            name='plain24_iter5',
            label='plain24 iter=5',
            staged_release=False,
            note=(
                'kg/ka active from iteration 1; filter ng/xa sigma/tau follow the condition-level '
                f"setting ({filter_gm['mode']})."
            ),
            **common,
        ),
        'staged24': h24.Hybrid24Config(
            name='staged24_iter5',
            label='staged24 iter=5',
            staged_release=True,
            release_iter=2,
            rot_gate_dps=5.0,
            scale_wash_scale=0.5,
            note=(
                'iter1 freezes kg/ka; iter>=2 releases them with the current main comparison gate; '
                f"filter ng/xa sigma/tau follow the condition-level setting ({filter_gm['mode']})."
            ),
            **common,
        ),
    }



def run_seed_condition(task: tuple[str, int]) -> dict[str, Any]:
    condition_name, seed = task
    base12 = load_base12()
    h24 = load_hybrid24()
    gm_helper = load_gm_helper()
    acc18 = h24.load_acc18()

    spec = get_condition_spec(condition_name)
    filter_gm = build_filter_gm_setting(condition_name)

    np.random.seed(seed)

    ts = 0.01
    att0 = np.array([0.0, 0.0, 0.0])
    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    rot_paras = base12.build_rot_paras()
    att_truth = acc18.attrottt(att0, rot_paras, ts)
    imu_clean, _ = acc18.avp2imu(att_truth, pos0)

    imuerr = gm_helper.build_truth_imuerr_variant(profile=spec['truth_profile'])
    imu_noisy = gm_helper.apply_truth_imu_errors(imu_clean, imuerr)

    phi = np.array([0.1, 0.1, 0.5]) * acc18.glv.deg
    att0_guess = acc18.q2att(base12.qaddphi(acc18.a2qua(att0), phi))
    wvn = np.array([0.01, 0.01, 0.01])
    truth_att = att_truth[-1, 0:3]

    method_cfgs = build_method_configs(h24, filter_gm)
    out = {
        'condition': condition_name,
        'seed': seed,
        'condition_spec': spec,
        'truth_profile': gm_helper.describe_truth_profile(spec['truth_profile']),
        'filter_gm': filter_gm,
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
            'config': {
                'name': cfg.name,
                'label': cfg.label,
                'staged_release': bool(cfg.staged_release),
                'release_iter': int(cfg.release_iter),
                'rot_gate_dps': float(cfg.rot_gate_dps),
                'max_iter': int(cfg.max_iter),
                'ng_sigma_dph': list(cfg.ng_sigma_dph),
                'tau_g_s': list(cfg.tau_g_s),
                'xa_sigma_ug': list(cfg.xa_sigma_ug),
                'tau_a_s': list(cfg.tau_a_s),
            },
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



def load_unmatched_reference() -> dict[str, Any] | None:
    if not UNMATCHED_JSON.exists():
        return None
    try:
        return json.loads(UNMATCHED_JSON.read_text())
    except Exception:
        return None



def index_unmatched_summary_rows(unmatched: dict[str, Any] | None) -> dict[tuple[str, str], dict[str, Any]]:
    if unmatched is None:
        return {}
    rows = unmatched.get('summary_rows', [])
    return {(row['condition'], row['method']): row for row in rows}



def compare_with_unmatched(matched_summary: dict[str, Any], unmatched_rows: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for matched_condition in ('tiny_gm_matched', 'small_gm_matched'):
        unmatched_condition = matched_condition.replace('_matched', '')
        for method in ('plain24', 'staged24'):
            matched_stats = matched_summary[matched_condition][method]
            unmatched_stats = unmatched_rows.get((unmatched_condition, method))
            if unmatched_stats is None:
                continue
            comparisons.append({
                'condition': matched_condition,
                'method': method,
                'reference_unmatched_condition': unmatched_condition,
                'matched_yaw_abs_mean_arcsec': matched_stats['yaw_abs_mean_arcsec'],
                'unmatched_yaw_abs_mean_arcsec': unmatched_stats['yaw_abs_mean_arcsec'],
                'delta_yaw_abs_mean_arcsec': matched_stats['yaw_abs_mean_arcsec'] - unmatched_stats['yaw_abs_mean_arcsec'],
                'matched_norm_mean_arcsec': matched_stats['norm_mean_arcsec'],
                'unmatched_norm_mean_arcsec': unmatched_stats['norm_mean_arcsec'],
                'delta_norm_mean_arcsec': matched_stats['norm_mean_arcsec'] - unmatched_stats['norm_mean_arcsec'],
                'matched_pitch_mean_abs_arcsec': matched_stats['pitch_mean_abs_arcsec'],
                'unmatched_pitch_mean_abs_arcsec': unmatched_stats['pitch_mean_abs_arcsec'],
                'delta_pitch_mean_abs_arcsec': matched_stats['pitch_mean_abs_arcsec'] - unmatched_stats['pitch_mean_abs_arcsec'],
                'matched_yaw_abs_median_arcsec': matched_stats['yaw_abs_median_arcsec'],
                'unmatched_yaw_abs_median_arcsec': unmatched_stats['yaw_abs_median_arcsec'],
                'delta_yaw_abs_median_arcsec': matched_stats['yaw_abs_median_arcsec'] - unmatched_stats['yaw_abs_median_arcsec'],
                'matched_yaw_abs_max_arcsec': matched_stats['yaw_abs_max_arcsec'],
                'unmatched_yaw_abs_max_arcsec': unmatched_stats['yaw_abs_max_arcsec'],
                'delta_yaw_abs_max_arcsec': matched_stats['yaw_abs_max_arcsec'] - unmatched_stats['yaw_abs_max_arcsec'],
            })
    return comparisons



def build_recovery_verdict(comparisons: list[dict[str, Any]]) -> str:
    if not comparisons:
        return 'unmatched light-GM reference file not found, so matched-vs-unmatched recovery cannot be judged automatically.'

    improved = [
        item for item in comparisons
        if item['delta_yaw_abs_mean_arcsec'] < 0.0 and item['delta_norm_mean_arcsec'] < 0.0
    ]
    worsened = [
        item for item in comparisons
        if item['delta_yaw_abs_mean_arcsec'] > 0.0 and item['delta_norm_mean_arcsec'] > 0.0
    ]

    if len(improved) == len(comparisons):
        return 'Matching the GM model improves both yaw-mean and norm-mean for all compared light-GM cases/methods.'
    if len(worsened) == len(comparisons):
        return 'Matching the GM model consistently worsens both yaw-mean and norm-mean for the compared light-GM cases/methods.'
    return 'Matched-vs-unmatched recovery is mixed across cases/methods; inspect the per-condition deltas rather than claiming a universal gain.'



def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        '# Chapter 4 DAR plain24 vs staged24 matched-GM check (2026-03-31)',
        '',
        '## Why this run exists',
        '- earlier 2026-03-31 light-GM tests added truth-side GM drift but kept the filter-side ng/xa parameters at their old defaults',
        '- that mixed together two effects: intrinsic method behavior and model mismatch',
        '- this run keeps the same truth-side GM profiles, but for GM cases also matches the filter ng/xa stationary sigma/tau to the same GM setting',
        '',
        '## Current 24-state filter-side GM defaults',
        f"- ng sigma default: {payload['filter_defaults']['ng_sigma_dph']} deg/h",
        f"- tau_g default: {payload['filter_defaults']['tau_g_s']} s",
        f"- xa sigma default: {payload['filter_defaults']['xa_sigma_ug']} ug",
        f"- tau_a default: {payload['filter_defaults']['tau_a_s']} s",
        '',
        '## Mapping rule used here',
        '- conservative direct 1:1 mapping because both sides use first-order GM stationary sigma + tau',
        '- truth `gyro_sigma_dph` -> filter `ng_sigma_dph`',
        '- truth `tau_g_s` -> filter `tau_g_s`',
        '- truth `accel_sigma_ug` -> filter `xa_sigma_ug`',
        '- truth `tau_a_s` -> filter `tau_a_s`',
        '- baseline is kept as the current Chapter-4 reference rather than forcing ng/xa to zero',
        '',
        '## Condition setup',
    ]
    for condition in CONDITIONS:
        item = payload['condition_details'][condition]
        lines.append(
            f"- {condition}: truth={item['truth_profile']['profile']}, filter_mode={item['filter_gm']['mode']}, "
            f"ng_sigma={item['filter_gm']['ng_sigma_dph']} deg/h, xa_sigma={item['filter_gm']['xa_sigma_ug']} ug, "
            f"tau_g={item['filter_gm']['tau_g_s']} s, tau_a={item['filter_gm']['tau_a_s']} s. {item['filter_gm']['mapping_note']}"
        )

    lines.extend([
        '',
        '## Controlled setup',
        '- same DAR path / same initial misalignment / same outer iteration count (`iter=5`) as the current Chapter-4 comparison',
        '- same seed set: `[0, 1, 2, 3, 4]`',
        '- for each `(condition, seed)`, one noisy IMU realization is generated and reused for both plain24 and staged24',
        '',
        '## Summary table',
        '',
        '| condition | method | pitch mean abs (\") | yaw abs mean (\") | norm mean (\") | yaw abs median (\") | yaw abs max (\") |',
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
        '## Matched vs previous unmatched light-GM result',
        '',
    ])
    if payload['comparison_vs_unmatched']:
        lines.extend([
            '| matched condition | method | yaw mean Δ (matched-unmatched, \") | norm mean Δ (\") | pitch mean abs Δ (\") | yaw median Δ (\") | yaw max Δ (\") |',
            '|---|---|---:|---:|---:|---:|---:|',
        ])
        for item in payload['comparison_vs_unmatched']:
            lines.append(
                f"| {item['condition']} | {item['method']} | {item['delta_yaw_abs_mean_arcsec']:.3f} | {item['delta_norm_mean_arcsec']:.3f} | {item['delta_pitch_mean_abs_arcsec']:.3f} | {item['delta_yaw_abs_median_arcsec']:.3f} | {item['delta_yaw_abs_max_arcsec']:.3f} |"
            )
        lines.append('')
        lines.append(f"- Recovery verdict: {payload['recovery_verdict']}")
    else:
        lines.append('- Unmatched light-GM reference file was not available, so no automatic recovery comparison was produced.')

    lines.extend([
        '',
        '## Files',
        f'- helper: `{GM_HELPER_PATH}`',
        f'- script: `{SCRIPTS_DIR / "alignvn_dar_hybrid24_truth_gm_matched_py_2026-03-31.py"}`',
        f'- unmatched reference: `{UNMATCHED_JSON}`',
        f'- json: `{OUT_JSON}`',
        f'- md: `{OUT_MD}`',
        f'- csv: `{OUT_CSV}`',
        '',
    ])
    return '\n'.join(lines) + '\n'



def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(condition, seed) for condition in CONDITIONS for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as ex:
        seed_runs = list(ex.map(run_seed_condition, tasks))

    seed_runs.sort(key=lambda x: (x['condition'], x['seed']))

    grouped: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for item in seed_runs:
        grouped[item['condition']].append(item)

    condition_details: dict[str, Any] = {}
    summary_by_condition: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    judgements: list[dict[str, str]] = []

    for condition in CONDITIONS:
        rows = grouped[condition]
        condition_details[condition] = {
            'condition_spec': deepcopy(rows[0]['condition_spec']),
            'truth_profile': deepcopy(rows[0]['truth_profile']),
            'filter_gm': deepcopy(rows[0]['filter_gm']),
        }
        plain_stats = summarize_method(rows, 'plain24')
        staged_stats = summarize_method(rows, 'staged24')
        summary_by_condition[condition] = {
            'condition_spec': deepcopy(rows[0]['condition_spec']),
            'truth_profile': deepcopy(rows[0]['truth_profile']),
            'filter_gm': deepcopy(rows[0]['filter_gm']),
            'plain24': plain_stats,
            'staged24': staged_stats,
        }
        summary_rows.append({'condition': condition, 'method': 'plain24', **plain_stats})
        summary_rows.append({'condition': condition, 'method': 'staged24', **staged_stats})
        judgement = build_judgement(plain_stats, staged_stats)
        judgement['condition'] = condition
        judgements.append(judgement)

    unmatched = load_unmatched_reference()
    unmatched_rows = index_unmatched_summary_rows(unmatched)
    comparison_vs_unmatched = compare_with_unmatched(summary_by_condition, unmatched_rows)
    recovery_verdict = build_recovery_verdict(comparison_vs_unmatched)

    csv_lines = ['condition,method,pitch_mean_abs_arcsec,yaw_abs_mean_arcsec,norm_mean_arcsec,yaw_abs_median_arcsec,yaw_abs_max_arcsec']
    for row in summary_rows:
        csv_lines.append(
            f"{row['condition']},{row['method']},{row['pitch_mean_abs_arcsec']:.6f},{row['yaw_abs_mean_arcsec']:.6f},{row['norm_mean_arcsec']:.6f},{row['yaw_abs_median_arcsec']:.6f},{row['yaw_abs_max_arcsec']:.6f}"
        )

    payload = {
        'meta': {
            'date': '2026-03-31',
            'purpose': 'matched-GM check for Chapter 4 DAR plain24 vs staged24',
            'seeds': SEEDS,
            'conditions': CONDITIONS,
            'same_noisy_imu_reused_within_condition_seed': True,
            'outer_iteration_setup_matches_main_result': True,
            'runtime_sec': time.time() - t0,
        },
        'filter_defaults': deepcopy(DEFAULT_FILTER_GM),
        'condition_details': condition_details,
        'seed_runs': seed_runs,
        'summary_by_condition': summary_by_condition,
        'summary_rows': summary_rows,
        'judgements': judgements,
        'unmatched_reference_json': str(UNMATCHED_JSON),
        'comparison_vs_unmatched': comparison_vs_unmatched,
        'recovery_verdict': recovery_verdict,
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    OUT_MD.write_text(build_markdown(payload))
    OUT_CSV.write_text('\n'.join(csv_lines) + '\n')

    print(json.dumps({
        'summary_rows': summary_rows,
        'judgements': judgements,
        'comparison_vs_unmatched': comparison_vs_unmatched,
        'recovery_verdict': recovery_verdict,
        'out_json': str(OUT_JSON),
        'out_md': str(OUT_MD),
        'out_csv': str(OUT_CSV),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
