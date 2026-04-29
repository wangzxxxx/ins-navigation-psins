#!/usr/bin/env python3
"""Targeted matched-GM adaptive probe for Chapter 4 DAR hybrid24.

This is a single focused method-improvement probe on top of the current
matched-GM Chapter-4 comparison.

Compared methods:
1. plain24 matched-GM reference
2. staged24 matched-GM reference
3. staged24_adaptive_gm_innov_gate

New method idea (lightweight, no new states):
- keep the same 24-state DAR path and the same iter=5 outer loop
- make ng/xa GM absorption stronger in early outer iterations, then taper back
  to the matched-GM setting later
- keep the staged release idea for kg/ka, but also require innovation behavior
  to stay reasonably calm before enabling scale-state coupling on a high-rotation
  segment

Innovation gate used here:
- compute a whitened innovation score at each inner update
- convert it to an equivalent per-axis RMS sigma ratio:
    innov_sigma = sqrt(NIS / 3)
- maintain an EMA across inner updates
- allow scale-state coupling only when:
    high rotation AND innovation_sigma_ema <= threshold
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
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
OUT_JSON = OUT_DIR / 'ch4_plain24_staged24_adaptive_truth_gm_matched_2026-03-31.json'
OUT_MD = OUT_DIR / 'ch4_plain24_staged24_adaptive_truth_gm_matched_2026-03-31.md'
OUT_CSV = OUT_DIR / 'ch4_plain24_staged24_adaptive_truth_gm_matched_table_2026-03-31.csv'
BASE12_PATH = SCRIPTS_DIR / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
HYBRID24_PATH = SCRIPTS_DIR / 'alignvn_dar_hybrid24_staged_py_2026-03-30.py'
GM_HELPER_PATH = SCRIPTS_DIR / 'alignvn_dar_truth_gm_helper_2026-03-31.py'
REFERENCE_MATCHED_JSON = OUT_DIR / 'ch4_plain24_staged24_truth_gm_matched_2026-03-31.json'
MAX_WORKERS = min(4, os.cpu_count() or 1)
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
DEFAULT_CONDITIONS = ['baseline', 'tiny_gm_matched', 'small_gm_matched']

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


@dataclass
class MethodConfig:
    name: str
    label: str
    seeds: list[int]
    max_iter: int = 5
    wash_scale: float = 0.5
    scale_wash_scale: float = 0.5
    carry_att_seed: bool = True
    staged_release: bool = False
    release_iter: int = 2
    rot_gate_dps: float = 5.0
    ng_sigma_dph: list[float] | None = None
    tau_g_s: list[float] | None = None
    xa_sigma_ug: list[float] | None = None
    tau_a_s: list[float] | None = None
    adaptive_gm_sigma_schedule: list[float] | None = None
    innovation_gate_enabled: bool = False
    innovation_gate_sigma_thresh: float = 0.2
    innovation_gate_ema_alpha: float = 0.999
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
        if self.adaptive_gm_sigma_schedule is None:
            self.adaptive_gm_sigma_schedule = [1.0] * self.max_iter
        if len(self.adaptive_gm_sigma_schedule) < self.max_iter:
            last = self.adaptive_gm_sigma_schedule[-1]
            self.adaptive_gm_sigma_schedule = self.adaptive_gm_sigma_schedule + [last] * (self.max_iter - len(self.adaptive_gm_sigma_schedule))
        elif len(self.adaptive_gm_sigma_schedule) > self.max_iter:
            self.adaptive_gm_sigma_schedule = self.adaptive_gm_sigma_schedule[:self.max_iter]


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
        _BASE12 = load_module('alignvn_base12_truth_gm_adaptive_20260331', BASE12_PATH)
    return _BASE12



def load_hybrid24():
    global _HYBRID24
    if _HYBRID24 is None:
        _HYBRID24 = load_module('alignvn_hybrid24_truth_gm_adaptive_20260331', HYBRID24_PATH)
    return _HYBRID24



def load_gm_helper():
    global _GM_HELPER
    if _GM_HELPER is None:
        _GM_HELPER = load_module('alignvn_truth_gm_adaptive_helper_20260331', GM_HELPER_PATH)
    return _GM_HELPER



def parse_csv_list(text: str | None, cast):
    if text is None or text.strip() == '':
        return None
    return [cast(item.strip()) for item in text.split(',') if item.strip()]



def get_condition_spec(condition_name: str) -> dict[str, Any]:
    if condition_name == 'baseline':
        return {
            'condition': 'baseline',
            'truth_profile': 'baseline',
            'filter_mode': 'chapter4_default_reference',
            'label': 'baseline',
            'note': 'No added truth-side GM. Keep the current Chapter-4 24-state filter default ng/xa sigma/tau as the baseline reference.',
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
            'mapping_note': 'Baseline has no extra truth GM process, so it keeps the current Chapter-4 filter reference.',
        }

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



def compute_predicted_innovation_metrics(kf: dict[str, Any], yk: np.ndarray) -> dict[str, Any]:
    Hk = kf['Hk']
    Pxy = kf['Pxk'] @ Hk.T
    Py0 = Hk @ Pxy
    ykk_1 = Hk @ kf['xk']
    rk = yk - ykk_1
    Py = Py0 + kf['Rk']
    try:
        Py_inv = np.linalg.inv(Py)
    except np.linalg.LinAlgError:
        Py_inv = np.linalg.pinv(Py)
    nis = float(rk.T @ Py_inv @ rk)
    sigma_ratio = float(math.sqrt(max(nis, 0.0) / max(len(rk), 1)))
    return {
        'rk': rk,
        'Py': Py,
        'Pxy': Pxy,
        'nis': nis,
        'sigma_ratio': sigma_ratio,
    }



def kfupdate_with_metrics(kf: dict[str, Any], yk: np.ndarray) -> tuple[dict[str, Any], dict[str, Any]]:
    kf['xk'] = kf['Phikk_1'] @ kf['xk']
    kf['Pxk'] = kf['Phikk_1'] @ kf['Pxk'] @ kf['Phikk_1'].T + kf['Qk']

    metrics = compute_predicted_innovation_metrics(kf, yk)
    try:
        Kk = metrics['Pxy'] @ np.linalg.inv(metrics['Py'])
    except np.linalg.LinAlgError:
        Kk = metrics['Pxy'] @ np.linalg.pinv(metrics['Py'])
    kf['Kk'] = Kk
    kf['rk'] = metrics['rk']
    kf['xk'] = kf['xk'] + Kk @ metrics['rk']
    kf['Pxk'] = kf['Pxk'] - Kk @ metrics['Py'] @ Kk.T
    kf['Pxk'] = (kf['Pxk'] + kf['Pxk'].T) * 0.5
    return kf, metrics



def alignvn_24state_iter_custom(imu: np.ndarray, qnb: np.ndarray, pos: np.ndarray, phi0: np.ndarray,
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

    base_ng_sigma = np.array(cfg.ng_sigma_dph, dtype=float) * glv.dph
    base_xa_sigma = np.array(cfg.xa_sigma_ug, dtype=float) * glv.ug
    tau_g_s = np.array(cfg.tau_g_s, dtype=float)
    tau_a_s = np.array(cfg.tau_a_s, dtype=float)

    for iteration in range(1, cfg.max_iter + 1):
        scale_active = (not cfg.staged_release) or (iteration >= cfg.release_iter)
        gm_sigma_scale = float(cfg.adaptive_gm_sigma_schedule[iteration - 1])
        ng_sigma = base_ng_sigma * gm_sigma_scale
        xa_sigma = base_xa_sigma * gm_sigma_scale
        kf = h24.avnkfinit_24(
            nts, pos, phi0, imuerr, wvn,
            ng_sigma,
            tau_g_s,
            xa_sigma,
            tau_a_s,
            enable_scale_states=scale_active,
        )
        vn = np.zeros(3)
        qnbi = qnb_seed.copy()
        innovation_sigma_ema = 1.0
        innovation_sigma_sum = 0.0
        innovation_sigma_max = 0.0
        innovation_nis_sum = 0.0
        high_rot_steps = 0
        innovation_ok_steps = 0
        coupled_steps = 0
        innovation_blocked_steps = 0

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

            high_rot = bool(np.max(np.abs(phim / nts)) > rot_gate_rad)
            innovation_ok = (not cfg.innovation_gate_enabled) or (innovation_sigma_ema <= cfg.innovation_gate_sigma_thresh)
            allow_scale_coupling = bool(scale_active and high_rot and innovation_ok)

            if high_rot:
                high_rot_steps += 1
            if innovation_ok:
                innovation_ok_steps += 1
            if scale_active and high_rot and (not innovation_ok):
                innovation_blocked_steps += 1
            if allow_scale_coupling:
                coupled_steps += 1
                phi_k[0:3, 18:21] = -cnb @ np.diag(phim[0:3])
                phi_k[3:6, 21:24] = cnb @ np.diag(dvbm[0:3])
            else:
                phi_k[0:3, 18:21] = 0.0
                phi_k[3:6, 21:24] = 0.0

            kf['Phikk_1'] = phi_k
            kf, metrics = kfupdate_with_metrics(kf, vn)
            innovation_sigma = metrics['sigma_ratio']
            innovation_sigma_sum += innovation_sigma
            innovation_sigma_max = max(innovation_sigma_max, innovation_sigma)
            innovation_nis_sum += metrics['nis']
            innovation_sigma_ema = cfg.innovation_gate_ema_alpha * innovation_sigma_ema + (1.0 - cfg.innovation_gate_ema_alpha) * innovation_sigma

            qnbi = acc18.qdelphi(qnbi, 0.91 * kf['xk'][0:3])
            kf['xk'][0:3] *= 0.09
            vn = vn - 0.91 * kf['xk'][3:6]
            kf['xk'][3:6] *= 0.09

            if not scale_active:
                kf['xk'][18:24] = 0.0

        final_att = acc18.q2att(qnbi)
        att_err_arcsec = acc18.qq2phi(acc18.a2qua(final_att), acc18.a2qua(truth_att)) / glv.sec
        total_steps = max(length // nn, 1)
        iter_logs.append({
            'iteration': iteration,
            'scale_active': scale_active,
            'gm_sigma_scale': gm_sigma_scale,
            'innovation_gate_enabled': bool(cfg.innovation_gate_enabled),
            'innovation_gate_sigma_thresh': float(cfg.innovation_gate_sigma_thresh),
            'innovation_gate_ema_alpha': float(cfg.innovation_gate_ema_alpha),
            'att_err_arcsec': [float(x) for x in att_err_arcsec],
            'att_err_norm_arcsec': float(np.linalg.norm(att_err_arcsec)),
            'yaw_abs_arcsec': float(abs(att_err_arcsec[2])),
            'est_kg_ppm': (kf['xk'][18:21] / glv.ppm).tolist(),
            'est_ka_ppm': (kf['xk'][21:24] / glv.ppm).tolist(),
            'high_rot_steps': int(high_rot_steps),
            'coupled_steps': int(coupled_steps),
            'innovation_blocked_steps': int(innovation_blocked_steps),
            'coupled_ratio': float(coupled_steps / max(high_rot_steps, 1)) if high_rot_steps > 0 else 0.0,
            'high_rot_ratio': float(high_rot_steps / total_steps),
            'innovation_ok_ratio': float(innovation_ok_steps / total_steps),
            'innovation_sigma_mean': float(innovation_sigma_sum / total_steps),
            'innovation_sigma_max': float(innovation_sigma_max),
            'innovation_sigma_ema_final': float(innovation_sigma_ema),
            'innovation_nis_mean': float(innovation_nis_sum / total_steps),
        })

        if iteration < cfg.max_iter:
            if cfg.carry_att_seed:
                qnb_seed = qnbi.copy()
            imu_corr[:, 0:3] -= cfg.wash_scale * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= cfg.wash_scale * kf['xk'][9:12] * ts
            if scale_active and cfg.scale_wash_scale > 0.0:
                imu_corr = h24.apply_scale_wash(imu_corr, kf['xk'][18:21], kf['xk'][21:24], cfg.scale_wash_scale)

    assert final_att is not None
    return final_att, iter_logs



def build_method_configs(seeds: list[int], filter_gm: dict[str, Any]) -> dict[str, MethodConfig]:
    common = dict(
        seeds=seeds,
        max_iter=5,
        ng_sigma_dph=list(filter_gm['ng_sigma_dph']),
        tau_g_s=list(filter_gm['tau_g_s']),
        xa_sigma_ug=list(filter_gm['xa_sigma_ug']),
        tau_a_s=list(filter_gm['tau_a_s']),
    )
    return {
        'plain24': MethodConfig(
            name='plain24_iter5',
            label='plain24 iter=5',
            staged_release=False,
            note='kg/ka active from iteration 1; matched-GM reference.',
            **common,
        ),
        'staged24': MethodConfig(
            name='staged24_iter5',
            label='staged24 iter=5',
            staged_release=True,
            release_iter=2,
            rot_gate_dps=5.0,
            scale_wash_scale=0.5,
            note='iter1 freezes kg/ka; iter>=2 releases them with the current high-rotation gate.',
            **common,
        ),
        'adaptive_staged24': MethodConfig(
            name='staged24_adaptive_gm_innov_gate_iter5',
            label='staged24 adaptive-GM + innov-gate iter=5',
            staged_release=True,
            release_iter=2,
            rot_gate_dps=5.0,
            scale_wash_scale=0.5,
            adaptive_gm_sigma_schedule=[2.0, 1.5, 1.0, 1.0, 1.0],
            innovation_gate_enabled=True,
            innovation_gate_sigma_thresh=0.2,
            innovation_gate_ema_alpha=0.999,
            note=(
                'Same staged release as staged24, but early outer iterations inflate ng/xa GM sigma by '
                '[2.0, 1.5, 1.0, 1.0, 1.0]; scale-state coupling is allowed only when both high rotation is '
                'present and the long-memory EMA innovation RMS sigma ratio stays <= 0.2.'
            ),
            **common,
        ),
    }



def run_seed_condition(task: tuple[str, int, list[int]]) -> dict[str, Any]:
    condition_name, seed, all_seeds = task
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

    method_cfgs = build_method_configs(all_seeds, filter_gm)
    out = {
        'condition': condition_name,
        'seed': seed,
        'condition_spec': spec,
        'truth_profile': gm_helper.describe_truth_profile(spec['truth_profile']),
        'filter_gm': filter_gm,
        'methods': {},
    }
    for method_name, cfg in method_cfgs.items():
        _, iter_logs = alignvn_24state_iter_custom(
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
            'config': asdict(cfg),
            'final_att_err_arcsec': [float(x) for x in last['att_err_arcsec']],
            'final_att_err_abs_arcsec': [float(abs(x)) for x in last['att_err_arcsec']],
            'final_att_err_norm_arcsec': float(last['att_err_norm_arcsec']),
            'final_yaw_abs_arcsec': float(last['yaw_abs_arcsec']),
            'final_iter_gate_stats': {
                'gm_sigma_scale': float(last['gm_sigma_scale']),
                'coupled_ratio': float(last['coupled_ratio']),
                'high_rot_ratio': float(last['high_rot_ratio']),
                'innovation_ok_ratio': float(last['innovation_ok_ratio']),
                'innovation_sigma_mean': float(last['innovation_sigma_mean']),
                'innovation_sigma_max': float(last['innovation_sigma_max']),
                'innovation_sigma_ema_final': float(last['innovation_sigma_ema_final']),
                'innovation_nis_mean': float(last['innovation_nis_mean']),
            },
            'iter_logs': iter_logs,
        }
    return out



def summarize_method(rows: list[dict[str, Any]], method_name: str) -> dict[str, Any]:
    errs = np.array([row['methods'][method_name]['final_att_err_arcsec'] for row in rows], dtype=float)
    abs_errs = np.abs(errs)
    norms = np.array([row['methods'][method_name]['final_att_err_norm_arcsec'] for row in rows], dtype=float)
    yaw_abs = np.array([row['methods'][method_name]['final_yaw_abs_arcsec'] for row in rows], dtype=float)
    gate_rows = [row['methods'][method_name].get('final_iter_gate_stats') for row in rows]
    gate_rows = [g for g in gate_rows if g]
    out = {
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
    if gate_rows:
        out['final_iter_gate_mean'] = {
            'gm_sigma_scale': float(np.mean([g['gm_sigma_scale'] for g in gate_rows])),
            'coupled_ratio': float(np.mean([g['coupled_ratio'] for g in gate_rows])),
            'high_rot_ratio': float(np.mean([g['high_rot_ratio'] for g in gate_rows])),
            'innovation_ok_ratio': float(np.mean([g['innovation_ok_ratio'] for g in gate_rows])),
            'innovation_sigma_mean': float(np.mean([g['innovation_sigma_mean'] for g in gate_rows])),
            'innovation_sigma_max': float(np.mean([g['innovation_sigma_max'] for g in gate_rows])),
            'innovation_sigma_ema_final': float(np.mean([g['innovation_sigma_ema_final'] for g in gate_rows])),
            'innovation_nis_mean': float(np.mean([g['innovation_nis_mean'] for g in gate_rows])),
        }
    return out



def build_condition_judgement(summary: dict[str, Any]) -> dict[str, Any]:
    plain = summary['plain24']
    staged = summary['staged24']
    adaptive = summary['adaptive_staged24']

    delta_vs_staged = {
        'pitch_mean_abs_arcsec': adaptive['pitch_mean_abs_arcsec'] - staged['pitch_mean_abs_arcsec'],
        'yaw_abs_mean_arcsec': adaptive['yaw_abs_mean_arcsec'] - staged['yaw_abs_mean_arcsec'],
        'norm_mean_arcsec': adaptive['norm_mean_arcsec'] - staged['norm_mean_arcsec'],
        'yaw_abs_median_arcsec': adaptive['yaw_abs_median_arcsec'] - staged['yaw_abs_median_arcsec'],
        'yaw_abs_max_arcsec': adaptive['yaw_abs_max_arcsec'] - staged['yaw_abs_max_arcsec'],
    }
    delta_vs_plain = {
        'yaw_abs_mean_arcsec': adaptive['yaw_abs_mean_arcsec'] - plain['yaw_abs_mean_arcsec'],
        'norm_mean_arcsec': adaptive['norm_mean_arcsec'] - plain['norm_mean_arcsec'],
    }

    improved_main = (delta_vs_staged['yaw_abs_mean_arcsec'] < 0.0) and (delta_vs_staged['norm_mean_arcsec'] < 0.0)
    hurt_main = (delta_vs_staged['yaw_abs_mean_arcsec'] > 0.0) and (delta_vs_staged['norm_mean_arcsec'] > 0.0)

    if improved_main:
        verdict = 'adaptive variant improves over current staged24 on both yaw-mean and norm-mean'
    elif hurt_main:
        verdict = 'adaptive variant is worse than current staged24 on both yaw-mean and norm-mean'
    elif delta_vs_staged['yaw_abs_mean_arcsec'] < 0.0 and delta_vs_staged['norm_mean_arcsec'] > 0.0:
        verdict = 'adaptive variant helps yaw-mean but hurts norm-mean versus current staged24'
    elif delta_vs_staged['yaw_abs_mean_arcsec'] > 0.0 and delta_vs_staged['norm_mean_arcsec'] < 0.0:
        verdict = 'adaptive variant helps norm-mean but hurts yaw-mean versus current staged24'
    else:
        verdict = 'adaptive variant is essentially tied with current staged24 on the main metrics'

    return {
        'verdict': verdict,
        'delta_vs_staged24': delta_vs_staged,
        'delta_vs_plain24': delta_vs_plain,
    }



def build_global_verdict(judgements: list[dict[str, Any]]) -> str:
    rows = {item['condition']: item for item in judgements}
    small = rows.get('small_gm_matched')
    tiny = rows.get('tiny_gm_matched')
    if small is None:
        return 'small_gm_matched was not run, so the requested primary verdict is unavailable.'

    dy = small['delta_vs_staged24']['yaw_abs_mean_arcsec']
    dn = small['delta_vs_staged24']['norm_mean_arcsec']
    if dy < 0.0 and dn < 0.0:
        core = 'Under the primary small_gm_matched condition, the adaptive variant improved over current staged24.'
    elif dy > 0.0 and dn > 0.0:
        core = 'Under the primary small_gm_matched condition, the adaptive variant did not improve over current staged24; it was clearly worse on both yaw-mean and norm-mean.'
    else:
        core = 'Under the primary small_gm_matched condition, the adaptive variant gave only a mixed trade-off rather than a clean improvement over current staged24.'

    if tiny is None:
        return core

    ty = tiny['delta_vs_staged24']['yaw_abs_mean_arcsec']
    tn = tiny['delta_vs_staged24']['norm_mean_arcsec']
    if abs(ty) < 0.2 and abs(tn) < 0.2:
        tail = ' On tiny_gm_matched it was basically tied.'
    elif ty < 0.0 and tn < 0.0:
        tail = ' On tiny_gm_matched it also improved.'
    elif ty > 0.0 and tn > 0.0:
        tail = ' On tiny_gm_matched it also got worse.'
    else:
        tail = ' On tiny_gm_matched it was mixed.'
    return core + tail



def maybe_load_reference() -> dict[str, Any] | None:
    if not REFERENCE_MATCHED_JSON.exists():
        return None
    try:
        return json.loads(REFERENCE_MATCHED_JSON.read_text())
    except Exception:
        return None



def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        '# Chapter 4 DAR matched-GM adaptive probe (2026-03-31)',
        '',
        '## Goal',
        '- Run one focused method-improvement probe on top of the current matched-GM Chapter-4 DAR setup.',
        '- Keep the same DAR path, seeds `[0,1,2,3,4]`, and `iter=5` outer loop.',
        '- Test whether a lightweight adaptive-GM + innovation-gate staged variant can further suppress slow-drift/noise effects without adding states.',
        '',
        '## New method change',
        '- **Adaptive GM absorption:** ng/xa GM sigma schedule across outer iterations = `[2.0, 1.5, 1.0, 1.0, 1.0] × matched-GM sigma`.',
        '- **Innovation gate for scale coupling:** kg/ka coupling is enabled only when both:',
        '  - local segment rotation exceeds the existing high-rotation gate (`rot_gate_dps = 5.0`), and',
        '  - the long-memory EMA whitened innovation RMS sigma ratio stays `<= 0.2` (`ema_alpha = 0.999`).',
        '- No new states were introduced; this is only a change to outer-iteration scheduling and inner coupling gate logic.',
        '',
        '## Controlled setup',
        f"- conditions: {payload['meta']['conditions']}",
        f"- seeds: {payload['meta']['seeds']}",
        '- same noisy IMU realization is reused across all compared methods within each `(condition, seed)`.',
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
        '## Gate diagnostics for the new adaptive variant (final outer iteration, seed-average)',
        '',
        '| condition | coupled ratio in high-rot steps | high-rot ratio | innovation-ok ratio | innovation sigma mean | innovation sigma max | innovation sigma EMA final |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])
    for condition in payload['meta']['conditions']:
        stats = payload['summary_by_condition'][condition]['adaptive_staged24'].get('final_iter_gate_mean', {})
        if not stats:
            continue
        lines.append(
            f"| {condition} | {stats['coupled_ratio']:.3f} | {stats['high_rot_ratio']:.3f} | {stats['innovation_ok_ratio']:.3f} | {stats['innovation_sigma_mean']:.3f} | {stats['innovation_sigma_max']:.3f} | {stats['innovation_sigma_ema_final']:.3f} |"
        )

    lines.extend([
        '',
        '## Condition-by-condition verdict vs current staged24',
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
        f'- script: `{SCRIPTS_DIR / "alignvn_dar_hybrid24_truth_gm_adaptive_probe_2026-03-31.py"}`',
        f'- matched reference used for context: `{REFERENCE_MATCHED_JSON}`',
        f'- json: `{OUT_JSON}`',
        f'- md: `{OUT_MD}`',
        f'- csv: `{OUT_CSV}`',
        '',
    ])
    return '\n'.join(lines) + '\n'



def main() -> None:
    parser = argparse.ArgumentParser(description='Run the targeted matched-GM adaptive probe for Chapter 4 DAR hybrid24.')
    parser.add_argument('--conditions', type=str, default=None, help='Comma-separated condition list. Default: baseline,tiny_gm_matched,small_gm_matched')
    parser.add_argument('--seeds', type=str, default=None, help='Comma-separated integer seeds. Default: 0,1,2,3,4')
    parser.add_argument('--max-workers', type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    conditions = parse_csv_list(args.conditions, str) or list(DEFAULT_CONDITIONS)
    seeds = parse_csv_list(args.seeds, int) or list(DEFAULT_SEEDS)
    max_workers = max(1, min(int(args.max_workers), os.cpu_count() or 1))

    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(condition, seed, seeds) for condition in conditions for seed in seeds]
    with ProcessPoolExecutor(max_workers=min(max_workers, len(tasks))) as ex:
        seed_runs = list(ex.map(run_seed_condition, tasks))

    seed_runs.sort(key=lambda x: (x['condition'], x['seed']))

    grouped: dict[str, list[dict[str, Any]]] = {condition: [] for condition in conditions}
    for item in seed_runs:
        grouped[item['condition']].append(item)

    condition_details: dict[str, Any] = {}
    summary_by_condition: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    judgements: list[dict[str, Any]] = []
    methods = ['plain24', 'staged24', 'adaptive_staged24']

    for condition in conditions:
        rows = grouped[condition]
        condition_details[condition] = {
            'condition_spec': deepcopy(rows[0]['condition_spec']),
            'truth_profile': deepcopy(rows[0]['truth_profile']),
            'filter_gm': deepcopy(rows[0]['filter_gm']),
        }
        summary_by_condition[condition] = {
            'condition_spec': deepcopy(rows[0]['condition_spec']),
            'truth_profile': deepcopy(rows[0]['truth_profile']),
            'filter_gm': deepcopy(rows[0]['filter_gm']),
        }
        for method in methods:
            stats = summarize_method(rows, method)
            summary_by_condition[condition][method] = stats
            summary_rows.append({'condition': condition, 'method': method, **stats})
        judgement = build_condition_judgement(summary_by_condition[condition])
        judgement['condition'] = condition
        judgements.append(judgement)

    global_verdict = build_global_verdict(judgements)

    csv_lines = ['condition,method,pitch_mean_abs_arcsec,yaw_abs_mean_arcsec,norm_mean_arcsec,yaw_abs_median_arcsec,yaw_abs_max_arcsec']
    for row in summary_rows:
        csv_lines.append(
            f"{row['condition']},{row['method']},{row['pitch_mean_abs_arcsec']:.6f},{row['yaw_abs_mean_arcsec']:.6f},{row['norm_mean_arcsec']:.6f},{row['yaw_abs_median_arcsec']:.6f},{row['yaw_abs_max_arcsec']:.6f}"
        )

    payload = {
        'meta': {
            'date': '2026-03-31',
            'purpose': 'targeted matched-GM adaptive probe for Chapter 4 DAR hybrid24',
            'seeds': seeds,
            'conditions': conditions,
            'methods': methods,
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
        'global_verdict': global_verdict,
        'reference_matched_json': str(REFERENCE_MATCHED_JSON),
        'reference_matched_loaded': maybe_load_reference() is not None,
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    OUT_MD.write_text(build_markdown(payload))
    OUT_CSV.write_text('\n'.join(csv_lines) + '\n')

    print(json.dumps({
        'summary_rows': summary_rows,
        'judgements': judgements,
        'global_verdict': global_verdict,
        'out_json': str(OUT_JSON),
        'out_md': str(OUT_MD),
        'out_csv': str(OUT_CSV),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
