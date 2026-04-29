#!/usr/bin/env python3
"""Targeted yaw-directed matched-GM probe for Chapter 4 DAR hybrid24.

Probe idea:
- keep the current staged24 DAR path and matched-GM setting
- only change the yaw-sensitive gyro-z scale state release timing
- compared with current staged24, keep kg_x/kg_y and ka_x/ka_y/ka_z release at iter>=2
  as before, but hold kg_z frozen for one extra outer iteration and only release it
  from iter>=3

Plain-language intent:
- heading is the weakest channel in this DAR path under matched small-GM drift
- the z-axis gyro scale state (kg_z) is the most direct scale term for yaw drift
- so this probe asks a narrow question: if we stop kg_z from learning during the
  earliest scale-release round, does staged24 keep its pitch benefit while avoiding
  some extra yaw contamination?
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
SCRIPTS_DIR = WORKSPACE / 'scripts'
OUT_DIR = WORKSPACE / 'tmp' / 'alignment_strategy_sweep'
OUT_JSON = OUT_DIR / 'ch4_plain24_staged24_yaw_hold_truth_gm_matched_2026-03-31.json'
OUT_MD = OUT_DIR / 'ch4_plain24_staged24_yaw_hold_truth_gm_matched_2026-03-31.md'
OUT_CSV = OUT_DIR / 'ch4_plain24_staged24_yaw_hold_truth_gm_matched_table_2026-03-31.csv'
BASE12_PATH = SCRIPTS_DIR / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
HYBRID24_PATH = SCRIPTS_DIR / 'alignvn_dar_hybrid24_staged_py_2026-03-30.py'
GM_HELPER_PATH = SCRIPTS_DIR / 'alignvn_dar_truth_gm_helper_2026-03-31.py'
REFERENCE_MATCHED_JSON = OUT_DIR / 'ch4_plain24_staged24_truth_gm_matched_2026-03-31.json'
MAX_WORKERS = min(4, os.cpu_count() or 1)
SEEDS = [0, 1, 2, 3, 4]
CONDITIONS = ['small_gm_matched', 'tiny_gm_matched']

_BASE12 = None
_HYBRID24 = None
_GM_HELPER = None


@dataclass
class MethodConfig:
    name: str
    label: str
    seeds: list[int]
    max_iter: int = 5
    wash_scale: float = 0.5
    scale_wash_scale: float = 0.5
    carry_att_seed: bool = True
    staged_release: bool = True
    release_iter: int = 2
    yaw_gyro_release_iter: int = 3
    rot_gate_dps: float = 5.0
    ng_sigma_dph: list[float] | None = None
    tau_g_s: list[float] | None = None
    xa_sigma_ug: list[float] | None = None
    tau_a_s: list[float] | None = None
    note: str = ''

    def __post_init__(self):
        if self.ng_sigma_dph is None:
            self.ng_sigma_dph = [0.05, 0.05, 0.05]
        if self.tau_g_s is None:
            self.tau_g_s = [300.0, 300.0, 300.0]
        if self.xa_sigma_ug is None:
            self.xa_sigma_ug = [0.01, 0.01, 0.01]
        if self.tau_a_s is None:
            self.tau_a_s = [100.0, 100.0, 100.0]


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
        _BASE12 = load_module('alignvn_base12_truth_gm_yaw_hold_20260331', BASE12_PATH)
    return _BASE12


def load_hybrid24():
    global _HYBRID24
    if _HYBRID24 is None:
        _HYBRID24 = load_module('alignvn_hybrid24_truth_gm_yaw_hold_20260331', HYBRID24_PATH)
    return _HYBRID24


def load_gm_helper():
    global _GM_HELPER
    if _GM_HELPER is None:
        _GM_HELPER = load_module('alignvn_truth_gm_yaw_hold_helper_20260331', GM_HELPER_PATH)
    return _GM_HELPER


def get_condition_spec(condition_name: str) -> dict[str, Any]:
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
    return {
        'mode': 'matched_truth_gm',
        'matched': True,
        'ng_sigma_dph': list(truth_profile['gyro_sigma_dph']),
        'tau_g_s': list(truth_profile['tau_g_s']),
        'xa_sigma_ug': list(truth_profile['accel_sigma_ug']),
        'tau_a_s': list(truth_profile['tau_a_s']),
        'source': 'direct 1:1 mapping from truth GM helper profile',
        'mapping_note': 'Direct sigma/tau match for the 24-state ng/xa GM states.',
    }


def load_reference_payload() -> dict[str, Any]:
    if not REFERENCE_MATCHED_JSON.exists():
        raise FileNotFoundError(f'matched reference not found: {REFERENCE_MATCHED_JSON}')
    return json.loads(REFERENCE_MATCHED_JSON.read_text())


def summarize_method_rows(rows: list[dict[str, Any]], method_name: str) -> dict[str, Any]:
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


def build_method_config(filter_gm: dict[str, Any]) -> MethodConfig:
    return MethodConfig(
        name='staged24_yaw_hold_kgz_iter3',
        label='staged24 yaw-hold kg_z iter>=3',
        seeds=SEEDS,
        max_iter=5,
        staged_release=True,
        release_iter=2,
        yaw_gyro_release_iter=3,
        rot_gate_dps=5.0,
        scale_wash_scale=0.5,
        ng_sigma_dph=list(filter_gm['ng_sigma_dph']),
        tau_g_s=list(filter_gm['tau_g_s']),
        xa_sigma_ug=list(filter_gm['xa_sigma_ug']),
        tau_a_s=list(filter_gm['tau_a_s']),
        note=(
            'Same as current staged24 except the yaw-sensitive gyro-z scale state kg_z is held '
            'frozen for one extra outer iteration: kg_x/kg_y and all ka states still release from '
            'iter>=2, but kg_z only releases from iter>=3.'
        ),
    )


def alignvn_24state_iter_yaw_hold(imu: np.ndarray, qnb: np.ndarray, pos: np.ndarray, phi0: np.ndarray,
                                  imuerr: dict[str, np.ndarray], wvn: np.ndarray, cfg: MethodConfig,
                                  truth_att: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    h24 = load_hybrid24()
    acc18 = h24.load_acc18()
    glv = acc18.glv

    imu_corr = imu.copy()
    nn = 2
    ts = float(imu_corr[1, -1] - imu_corr[0, -1])
    nts = nn * ts
    qnb_seed = acc18.a2qua(qnb) if len(qnb) == 3 else np.asarray(qnb).reshape(4)
    length = (len(imu_corr) // nn) * nn
    imu_corr = imu_corr[:length]

    eth = acc18.Earth(pos)
    cnn = acc18.rv2m(-eth.wnie * nts / 2)
    rot_gate_rad = cfg.rot_gate_dps * glv.deg

    iter_logs: list[dict[str, Any]] = []
    final_att = None

    for iteration in range(1, cfg.max_iter + 1):
        scale_active = (not cfg.staged_release) or (iteration >= cfg.release_iter)
        yaw_gyro_scale_active = scale_active and (iteration >= cfg.yaw_gyro_release_iter)
        kf = h24.avnkfinit_24(
            nts, pos, phi0, imuerr, wvn,
            np.array(cfg.ng_sigma_dph) * glv.dph,
            np.array(cfg.tau_g_s),
            np.array(cfg.xa_sigma_ug) * glv.ug,
            np.array(cfg.tau_a_s),
            enable_scale_states=scale_active,
        )
        vn = np.zeros(3)
        qnbi = qnb_seed.copy()
        high_rot_steps = 0
        scale_coupled_steps = 0
        yaw_kgz_coupled_steps = 0

        for k in range(0, length, nn):
            wvm = imu_corr[k:k + nn, 0:6]
            phim, dvbm = acc18.cnscl(wvm)
            cnb = acc18.q2mat(qnbi)
            dvn = cnn @ cnb @ dvbm
            vn = vn + dvn + eth.gn * nts
            qnbi = acc18.qupdt2(qnbi, phim, eth.wnin * nts)

            phi_k = kf['Phikk_1'].copy()
            cnbts = cnb * nts
            phi_k[3:6, 0:3] = acc18.askew(dvn)
            phi_k[3:6, 9:12] = cnbts
            phi_k[3:6, 15:18] = cnbts
            phi_k[0:3, 6:9] = -cnbts
            phi_k[0:3, 12:15] = -cnbts
            phi_k[12:15, 12:15] = np.diag(kf['fg'])
            phi_k[15:18, 15:18] = np.diag(kf['fa'])

            if scale_active:
                high_rot = bool(np.max(np.abs(phim / nts)) > rot_gate_rad)
                if high_rot:
                    high_rot_steps += 1
                    phi_k[0:3, 18:21] = -cnb @ np.diag(phim[0:3])
                    phi_k[3:6, 21:24] = cnb @ np.diag(dvbm[0:3])
                    scale_coupled_steps += 1
                    if not yaw_gyro_scale_active:
                        phi_k[0:3, 20] = 0.0
                    else:
                        yaw_kgz_coupled_steps += 1
                else:
                    phi_k[0:3, 18:21] = 0.0
                    phi_k[3:6, 21:24] = 0.0
            else:
                phi_k[0:3, 18:21] = 0.0
                phi_k[3:6, 21:24] = 0.0

            kf['Phikk_1'] = phi_k
            kf = acc18.kfupdate(kf, vn)

            qnbi = acc18.qdelphi(qnbi, 0.91 * kf['xk'][0:3])
            kf['xk'][0:3] *= 0.09
            vn = vn - 0.91 * kf['xk'][3:6]
            kf['xk'][3:6] *= 0.09

            if not scale_active:
                kf['xk'][18:24] = 0.0
            elif not yaw_gyro_scale_active:
                kf['xk'][20] = 0.0

        final_att = acc18.q2att(qnbi)
        att_err_arcsec = acc18.qq2phi(acc18.a2qua(final_att), acc18.a2qua(truth_att)) / glv.sec
        total_steps = max(length // nn, 1)
        iter_logs.append({
            'iteration': iteration,
            'scale_active': bool(scale_active),
            'yaw_gyro_scale_active': bool(yaw_gyro_scale_active),
            'att_err_arcsec': [float(x) for x in att_err_arcsec],
            'att_err_norm_arcsec': float(np.linalg.norm(att_err_arcsec)),
            'yaw_abs_arcsec': float(abs(att_err_arcsec[2])),
            'est_kg_ppm': (kf['xk'][18:21] / glv.ppm).tolist(),
            'est_ka_ppm': (kf['xk'][21:24] / glv.ppm).tolist(),
            'high_rot_steps': int(high_rot_steps),
            'scale_coupled_steps': int(scale_coupled_steps),
            'yaw_kgz_coupled_steps': int(yaw_kgz_coupled_steps),
            'high_rot_ratio': float(high_rot_steps / total_steps),
            'yaw_kgz_coupled_ratio': float(yaw_kgz_coupled_steps / max(high_rot_steps, 1)) if high_rot_steps > 0 else 0.0,
        })

        if iteration < cfg.max_iter:
            if cfg.carry_att_seed:
                qnb_seed = qnbi.copy()
            imu_corr[:, 0:3] -= cfg.wash_scale * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= cfg.wash_scale * kf['xk'][9:12] * ts
            if scale_active and cfg.scale_wash_scale > 0.0:
                kg_for_wash = kf['xk'][18:21].copy()
                if not yaw_gyro_scale_active:
                    kg_for_wash[2] = 0.0
                imu_corr = h24.apply_scale_wash(imu_corr, kg_for_wash, kf['xk'][21:24], cfg.scale_wash_scale)

    assert final_att is not None
    return final_att, iter_logs


def run_seed_condition(task: tuple[str, int]) -> dict[str, Any]:
    condition_name, seed = task
    base12 = load_base12()
    h24 = load_hybrid24()
    gm_helper = load_gm_helper()
    acc18 = h24.load_acc18()

    spec = get_condition_spec(condition_name)
    filter_gm = build_filter_gm_setting(condition_name)
    cfg = build_method_config(filter_gm)

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

    _, iter_logs = alignvn_24state_iter_yaw_hold(
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
        'condition': condition_name,
        'seed': seed,
        'condition_spec': spec,
        'truth_profile': gm_helper.describe_truth_profile(spec['truth_profile']),
        'filter_gm': filter_gm,
        'methods': {
            'yaw_hold24': {
                'config': asdict(cfg),
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
        },
    }


def build_reference_index(reference_payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = reference_payload.get('summary_rows', [])
    return {(row['condition'], row['method']): row for row in rows}


def build_condition_judgement(reference_index: dict[tuple[str, str], dict[str, Any]],
                              condition: str,
                              yaw_hold_stats: dict[str, Any]) -> dict[str, Any]:
    plain = reference_index[(condition, 'plain24')]
    staged = reference_index[(condition, 'staged24')]
    delta_vs_staged = {
        'pitch_mean_abs_arcsec': yaw_hold_stats['pitch_mean_abs_arcsec'] - staged['pitch_mean_abs_arcsec'],
        'yaw_abs_mean_arcsec': yaw_hold_stats['yaw_abs_mean_arcsec'] - staged['yaw_abs_mean_arcsec'],
        'norm_mean_arcsec': yaw_hold_stats['norm_mean_arcsec'] - staged['norm_mean_arcsec'],
        'yaw_abs_median_arcsec': yaw_hold_stats['yaw_abs_median_arcsec'] - staged['yaw_abs_median_arcsec'],
        'yaw_abs_max_arcsec': yaw_hold_stats['yaw_abs_max_arcsec'] - staged['yaw_abs_max_arcsec'],
    }
    delta_vs_plain = {
        'yaw_abs_mean_arcsec': yaw_hold_stats['yaw_abs_mean_arcsec'] - plain['yaw_abs_mean_arcsec'],
        'norm_mean_arcsec': yaw_hold_stats['norm_mean_arcsec'] - plain['norm_mean_arcsec'],
    }

    if delta_vs_staged['yaw_abs_mean_arcsec'] < 0.0 and delta_vs_staged['norm_mean_arcsec'] < 0.0:
        verdict = 'yaw-hold variant improves over current staged24 on both yaw-mean and norm-mean'
    elif delta_vs_staged['yaw_abs_mean_arcsec'] > 0.0 and delta_vs_staged['norm_mean_arcsec'] > 0.0:
        verdict = 'yaw-hold variant is worse than current staged24 on both yaw-mean and norm-mean'
    elif delta_vs_staged['yaw_abs_mean_arcsec'] < 0.0 and delta_vs_staged['norm_mean_arcsec'] > 0.0:
        verdict = 'yaw-hold variant helps yaw-mean but hurts norm-mean versus current staged24'
    elif delta_vs_staged['yaw_abs_mean_arcsec'] > 0.0 and delta_vs_staged['norm_mean_arcsec'] < 0.0:
        verdict = 'yaw-hold variant helps norm-mean but hurts yaw-mean versus current staged24'
    else:
        verdict = 'yaw-hold variant is effectively tied with current staged24 on the main metrics'

    return {
        'condition': condition,
        'verdict': verdict,
        'delta_vs_staged24': delta_vs_staged,
        'delta_vs_plain24': delta_vs_plain,
    }


def build_global_verdict(judgements: list[dict[str, Any]]) -> str:
    rows = {item['condition']: item for item in judgements}
    small = rows['small_gm_matched']
    tiny = rows.get('tiny_gm_matched')
    dy = small['delta_vs_staged24']['yaw_abs_mean_arcsec']
    dn = small['delta_vs_staged24']['norm_mean_arcsec']
    if dy < 0.0 and dn < 0.0:
        core = 'Under the primary small_gm_matched condition, the yaw-hold variant beat current staged24.'
    elif dy > 0.0 and dn > 0.0:
        core = 'Under the primary small_gm_matched condition, the yaw-hold variant did not beat current staged24; it was worse on both yaw-mean and norm-mean.'
    else:
        core = 'Under the primary small_gm_matched condition, the yaw-hold variant did not give a clean win over current staged24; it only produced a mixed trade-off.'

    if tiny is None:
        return core
    ty = tiny['delta_vs_staged24']['yaw_abs_mean_arcsec']
    tn = tiny['delta_vs_staged24']['norm_mean_arcsec']
    if ty < 0.0 and tn < 0.0:
        tail = ' On tiny_gm_matched it also improved.'
    elif ty > 0.0 and tn > 0.0:
        tail = ' On tiny_gm_matched it also got worse.'
    else:
        tail = ' On tiny_gm_matched it was mixed.'
    return core + tail


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        '# Chapter 4 DAR matched-GM yaw-hold probe (2026-03-31)',
        '',
        '## Goal',
        '- Run one narrow yaw-directed probe on top of the current matched-GM Chapter 4 DAR setup.',
        '- Keep the same DAR path, same seeds `[0,1,2,3,4]`, same `iter=5`, and same matched-GM truth/filter condition.',
        '',
        '## Exactly what changed',
        '- Start from the current `staged24` method.',
        '- Keep the usual staged release at `iter>=2` for `kg_x`, `kg_y`, and all three accelerometer scale states `ka_x/ka_y/ka_z`.',
        '- **Only** keep the yaw-sensitive gyro-z scale state `kg_z` frozen for one extra outer iteration.',
        '- So compared with current staged24:',
        '  - `iter1`: all scale states still frozen, same as staged24',
        '  - `iter2`: `kg_x/kg_y` and all `ka` states are released, but `kg_z` is still forced off',
        '  - `iter>=3`: `kg_z` is released and the method becomes the normal staged24 path again',
        '- No new states, no broad adaptive-R/GM logic, no global gate tightening.',
        '',
        '## Why this is yaw-directed',
        '- `kg_z` is the most direct gyro scale state for heading/yaw contamination in this 24-state layout.',
        '- This probe tests whether early `kg_z` learning is too eager under matched light-GM drift, while leaving the rest of the staged24 mechanism intact.',
        '',
        '## Summary table',
        '',
        '| condition | method | pitch mean abs (") | yaw abs mean (") | norm mean (") | yaw abs median (") | yaw abs max (") |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for row in payload['summary_rows']:
        lines.append(
            f"| {row['condition']} | {row['method']} | {row['pitch_mean_abs_arcsec']:.3f} | {row['yaw_abs_mean_arcsec']:.3f} | {row['norm_mean_arcsec']:.3f} | {row['yaw_abs_median_arcsec']:.3f} | {row['yaw_abs_max_arcsec']:.3f} |"
        )

    lines.extend([
        '',
        '## Final-iteration activation diagnostic for the new variant',
        '',
        '| condition | high-rotation ratio | yaw kg_z coupled ratio within high-rotation steps |',
        '|---|---:|---:|',
    ])
    for condition in CONDITIONS:
        stats = payload['yaw_hold_summary_by_condition'][condition].get('final_iter_gate_mean', {})
        if not stats:
            continue
        lines.append(
            f"| {condition} | {stats['high_rot_ratio']:.3f} | {stats['yaw_kgz_coupled_ratio']:.3f} |"
        )

    lines.extend([
        '',
        '## Verdict vs current staged24',
        '',
    ])
    for item in payload['judgements']:
        d = item['delta_vs_staged24']
        lines.append(
            f"- {item['condition']}: {item['verdict']} (Δpitch_mean_abs={d['pitch_mean_abs_arcsec']:+.3f}\", Δyaw_mean={d['yaw_abs_mean_arcsec']:+.3f}\", Δnorm_mean={d['norm_mean_arcsec']:+.3f}\", Δyaw_median={d['yaw_abs_median_arcsec']:+.3f}\", Δyaw_max={d['yaw_abs_max_arcsec']:+.3f}\")"
        )

    lines.extend([
        '',
        '## Crisp verdict',
        f"- {payload['global_verdict']}",
        '',
        '## Files',
        f'- script: `{SCRIPTS_DIR / "alignvn_dar_hybrid24_truth_gm_yaw_hold_probe_2026-03-31.py"}`',
        f'- matched reference used for plain24/staged24: `{REFERENCE_MATCHED_JSON}`',
        f'- json: `{OUT_JSON}`',
        f'- md: `{OUT_MD}`',
        f'- csv: `{OUT_CSV}`',
        '',
    ])
    return '\n'.join(lines) + '\n'


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    reference_payload = load_reference_payload()
    reference_index = build_reference_index(reference_payload)

    tasks = [(condition, seed) for condition in CONDITIONS for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as ex:
        seed_runs = list(ex.map(run_seed_condition, tasks))

    seed_runs.sort(key=lambda x: (x['condition'], x['seed']))

    grouped: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for item in seed_runs:
        grouped[item['condition']].append(item)

    yaw_hold_summary_by_condition: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    judgements: list[dict[str, Any]] = []

    for condition in CONDITIONS:
        rows = grouped[condition]
        yaw_hold_stats = summarize_method_rows(rows, 'yaw_hold24')
        gate_rows = [row['methods']['yaw_hold24']['final_iter_gate_stats'] for row in rows]
        yaw_hold_stats['final_iter_gate_mean'] = {
            'high_rot_ratio': float(np.mean([g['high_rot_ratio'] for g in gate_rows])),
            'yaw_kgz_coupled_ratio': float(np.mean([g['yaw_kgz_coupled_ratio'] for g in gate_rows])),
        }
        yaw_hold_summary_by_condition[condition] = yaw_hold_stats

        summary_rows.append({'condition': condition, 'method': 'plain24', **deepcopy(reference_index[(condition, 'plain24')])})
        summary_rows[-1].pop('condition', None)
        summary_rows[-1].pop('method', None)
        summary_rows[-1]['condition'] = condition
        summary_rows[-1]['method'] = 'plain24'

        summary_rows.append({'condition': condition, 'method': 'staged24', **deepcopy(reference_index[(condition, 'staged24')])})
        summary_rows[-1].pop('condition', None)
        summary_rows[-1].pop('method', None)
        summary_rows[-1]['condition'] = condition
        summary_rows[-1]['method'] = 'staged24'

        summary_rows.append({'condition': condition, 'method': 'yaw_hold24', **yaw_hold_stats})

        judgements.append(build_condition_judgement(reference_index, condition, yaw_hold_stats))

    global_verdict = build_global_verdict(judgements)

    csv_lines = ['condition,method,pitch_mean_abs_arcsec,yaw_abs_mean_arcsec,norm_mean_arcsec,yaw_abs_median_arcsec,yaw_abs_max_arcsec']
    for row in summary_rows:
        csv_lines.append(
            f"{row['condition']},{row['method']},{row['pitch_mean_abs_arcsec']:.6f},{row['yaw_abs_mean_arcsec']:.6f},{row['norm_mean_arcsec']:.6f},{row['yaw_abs_median_arcsec']:.6f},{row['yaw_abs_max_arcsec']:.6f}"
        )

    payload = {
        'meta': {
            'date': '2026-03-31',
            'purpose': 'targeted yaw-directed matched-GM probe for Chapter 4 DAR hybrid24',
            'conditions': CONDITIONS,
            'seeds': SEEDS,
            'methods': ['plain24', 'staged24', 'yaw_hold24'],
            'runtime_sec': time.time() - t0,
            'reference_plain_staged_loaded_from_artifact': True,
        },
        'reference_matched_json': str(REFERENCE_MATCHED_JSON),
        'reference_summary_by_condition': {
            condition: {
                'plain24': deepcopy(reference_index[(condition, 'plain24')]),
                'staged24': deepcopy(reference_index[(condition, 'staged24')]),
            }
            for condition in CONDITIONS
        },
        'seed_runs': seed_runs,
        'yaw_hold_summary_by_condition': yaw_hold_summary_by_condition,
        'summary_rows': summary_rows,
        'judgements': judgements,
        'global_verdict': global_verdict,
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    OUT_MD.write_text(build_markdown(payload))
    OUT_CSV.write_text('\n'.join(csv_lines) + '\n')

    print(json.dumps({
        'summary_rows': [
            {
                'condition': row['condition'],
                'method': row['method'],
                'pitch_mean_abs_arcsec': row['pitch_mean_abs_arcsec'],
                'yaw_abs_mean_arcsec': row['yaw_abs_mean_arcsec'],
                'norm_mean_arcsec': row['norm_mean_arcsec'],
                'yaw_abs_median_arcsec': row['yaw_abs_median_arcsec'],
                'yaw_abs_max_arcsec': row['yaw_abs_max_arcsec'],
            }
            for row in summary_rows
        ],
        'judgements': judgements,
        'global_verdict': global_verdict,
        'out_json': str(OUT_JSON),
        'out_md': str(OUT_MD),
        'out_csv': str(OUT_CSV),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
