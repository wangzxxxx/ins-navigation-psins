#!/usr/bin/env python3
"""Offline terminal-yaw smoothing probe on top of the Chapter 4 DAR yaw-hold matched-GM run.

This is intentionally *not* a full 24-state RTS rewrite.
Instead, it adds one conservative fixed-interval smoothing layer on top of the
current best yaw-directed variant (`yaw_hold24`):

1. run the same forward yaw_hold24 filter/outer-iteration path;
2. on the final outer iteration, collect the posterior quaternion after each step;
3. for each posterior step, propagate it open-loop to the interval end to get a
   sequence of *candidate terminal attitudes*;
4. on the last 50% of that sequence, extract the candidate terminal yaw correction
   relative to the forward terminal estimate;
5. run a scalar random-walk RTS smoother on that terminal-yaw correction sequence;
6. apply the smoothed constant yaw correction once to the forward terminal attitude.

So the smoothed quantity is:
- **terminal yaw correction** (a scalar, in navigation-frame small-angle form)
- derived from the full final-iteration posterior trajectory
- smoothed offline with an RTS-style forward-backward scalar smoother.

This is a lightweight "forward-backward smoothing layer" probe aimed directly at
whether the current yaw tail is mostly endpoint noise or a deeper structural issue.
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
OUT_JSON = OUT_DIR / 'ch4_plain24_staged24_yaw_hold_terminal_smoothing_truth_gm_matched_2026-03-31.json'
OUT_MD = OUT_DIR / 'ch4_plain24_staged24_yaw_hold_terminal_smoothing_truth_gm_matched_2026-03-31.md'
OUT_CSV = OUT_DIR / 'ch4_plain24_staged24_yaw_hold_terminal_smoothing_truth_gm_matched_table_2026-03-31.csv'
BASE12_PATH = SCRIPTS_DIR / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
HYBRID24_PATH = SCRIPTS_DIR / 'alignvn_dar_hybrid24_staged_py_2026-03-30.py'
GM_HELPER_PATH = SCRIPTS_DIR / 'alignvn_dar_truth_gm_helper_2026-03-31.py'
MATCHED_JSON = OUT_DIR / 'ch4_plain24_staged24_truth_gm_matched_2026-03-31.json'
YAW_HOLD_JSON = OUT_DIR / 'ch4_plain24_staged24_yaw_hold_truth_gm_matched_2026-03-31.json'
MAX_WORKERS = min(4, os.cpu_count() or 1)
SEEDS = [0, 1, 2, 3, 4]
CONDITIONS = ['tiny_gm_matched', 'small_gm_matched']
TAIL_FRACTION = 0.5
RTS_PROCESS_RATIO = 1.0e-4
RTS_MEAS_FLOOR_ARCSEC = 0.5
RTS_INIT_SCALE = 100.0

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
        _BASE12 = load_module('alignvn_base12_truth_gm_yaw_terminal_smooth_20260331', BASE12_PATH)
    return _BASE12


def load_hybrid24():
    global _HYBRID24
    if _HYBRID24 is None:
        _HYBRID24 = load_module('alignvn_hybrid24_truth_gm_yaw_terminal_smooth_20260331', HYBRID24_PATH)
    return _HYBRID24


def load_gm_helper():
    global _GM_HELPER
    if _GM_HELPER is None:
        _GM_HELPER = load_module('alignvn_truth_gm_yaw_terminal_smooth_helper_20260331', GM_HELPER_PATH)
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


def quat_left_matrix(q: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = np.asarray(q).reshape(4)
    return np.array([
        [q0, -q1, -q2, -q3],
        [q1,  q0, -q3,  q2],
        [q2,  q3,  q0, -q1],
        [q3, -q2,  q1,  q0],
    ], dtype=float)


def quat_right_matrix(q: np.ndarray) -> np.ndarray:
    q0, q1, q2, q3 = np.asarray(q).reshape(4)
    return np.array([
        [q0, -q1, -q2, -q3],
        [q1,  q0,  q3, -q2],
        [q2, -q3,  q0,  q1],
        [q3,  q2, -q1,  q0],
    ], dtype=float)


def build_qupdt2_matrix(acc18, rv_ib: np.ndarray, rv_in: np.ndarray) -> np.ndarray:
    a = acc18.rv2q(-rv_in)
    b = acc18.rv2q(rv_ib)
    return quat_left_matrix(a) @ quat_right_matrix(b)


def scalar_rts_constant_smoother(z: np.ndarray, glv) -> dict[str, Any]:
    z = np.asarray(z, dtype=float).reshape(-1)
    if len(z) == 0:
        return {
            'smoothed_value_rad': 0.0,
            'smoothed_series_rad': [],
            'measurement_var_rad2': 0.0,
            'process_var_rad2': 0.0,
        }
    if len(z) == 1:
        return {
            'smoothed_value_rad': float(z[0]),
            'smoothed_series_rad': [float(z[0])],
            'measurement_var_rad2': float(max(z[0] * z[0], (RTS_MEAS_FLOOR_ARCSEC * glv.sec) ** 2)),
            'process_var_rad2': 0.0,
        }

    med = float(np.median(z))
    mad = float(np.median(np.abs(z - med)))
    sigma = 1.4826 * mad
    meas_floor = RTS_MEAS_FLOOR_ARCSEC * glv.sec
    r = max(sigma * sigma, meas_floor * meas_floor)
    q = max(r * RTS_PROCESS_RATIO, (0.01 * glv.sec) ** 2)
    p0 = max(RTS_INIT_SCALE * r, (5.0 * glv.sec) ** 2)

    n = len(z)
    x_pred = np.zeros(n, dtype=float)
    p_pred = np.zeros(n, dtype=float)
    x_filt = np.zeros(n, dtype=float)
    p_filt = np.zeros(n, dtype=float)

    x_prev = med
    p_prev = p0
    for k in range(n):
        x_pred[k] = x_prev
        p_pred[k] = p_prev + q
        K = p_pred[k] / (p_pred[k] + r)
        x_filt[k] = x_pred[k] + K * (z[k] - x_pred[k])
        p_filt[k] = (1.0 - K) * p_pred[k]
        x_prev = x_filt[k]
        p_prev = p_filt[k]

    x_s = np.zeros(n, dtype=float)
    p_s = np.zeros(n, dtype=float)
    x_s[-1] = x_filt[-1]
    p_s[-1] = p_filt[-1]
    for k in range(n - 2, -1, -1):
        C = p_filt[k] / max(p_pred[k + 1], 1e-30)
        x_s[k] = x_filt[k] + C * (x_s[k + 1] - x_pred[k + 1])
        p_s[k] = p_filt[k] + C * C * (p_s[k + 1] - p_pred[k + 1])

    return {
        'smoothed_value_rad': float(x_s[0]),
        'smoothed_series_rad': [float(v) for v in x_s],
        'measurement_var_rad2': float(r),
        'process_var_rad2': float(q),
        'raw_mean_rad': float(np.mean(z)),
        'raw_median_rad': float(np.median(z)),
        'raw_std_rad': float(np.std(z, ddof=1)) if len(z) > 1 else 0.0,
    }


def mechanize_from_post_sequence(acc18, q_post_seq: list[np.ndarray], step_mats: list[np.ndarray]) -> list[np.ndarray]:
    n = len(q_post_seq)
    suffix = [np.eye(4) for _ in range(n + 1)]
    for k in range(n - 1, -1, -1):
        suffix[k] = suffix[k + 1] @ step_mats[k]

    candidates: list[np.ndarray] = []
    for k in range(n):
        q_end = suffix[k + 1] @ q_post_seq[k]
        q_end = q_end / np.linalg.norm(q_end)
        candidates.append(q_end)
    return candidates


def smooth_terminal_yaw_from_trace(acc18, final_iter_trace: dict[str, Any]) -> dict[str, Any]:
    glv = acc18.glv
    q_post_seq = [np.asarray(q, dtype=float).reshape(4) for q in final_iter_trace['q_post_seq']]
    step_mats = [np.asarray(m, dtype=float).reshape(4, 4) for m in final_iter_trace['step_qupdt2_mats']]
    q_end_forward = np.asarray(final_iter_trace['q_end_forward'], dtype=float).reshape(4)

    candidates = mechanize_from_post_sequence(acc18, q_post_seq, step_mats)
    n = len(candidates)
    tail_start = max(int((1.0 - TAIL_FRACTION) * n), 0)
    tail_candidates = candidates[tail_start:]

    yaw_corr = np.array([
        acc18.qq2phi(q_cand, q_end_forward)[2] for q_cand in tail_candidates
    ], dtype=float)
    smoother = scalar_rts_constant_smoother(yaw_corr, glv)
    delta_yaw = smoother['smoothed_value_rad']

    q_end_smooth = acc18.qdelphi(q_end_forward.copy(), np.array([0.0, 0.0, delta_yaw]))
    q_end_smooth = q_end_smooth / np.linalg.norm(q_end_smooth)

    return {
        'tail_fraction': TAIL_FRACTION,
        'num_total_candidates': int(n),
        'num_tail_candidates': int(len(tail_candidates)),
        'tail_start_index': int(tail_start),
        'yaw_tail_raw_arcsec': [float(v / glv.sec) for v in yaw_corr],
        'yaw_tail_rts': {
            'smoothed_value_arcsec': float(delta_yaw / glv.sec),
            'raw_mean_arcsec': float(smoother.get('raw_mean_rad', 0.0) / glv.sec),
            'raw_median_arcsec': float(smoother.get('raw_median_rad', 0.0) / glv.sec),
            'raw_std_arcsec': float(smoother.get('raw_std_rad', 0.0) / glv.sec),
            'measurement_sigma_arcsec': float(np.sqrt(smoother['measurement_var_rad2']) / glv.sec),
            'process_sigma_arcsec': float(np.sqrt(smoother['process_var_rad2']) / glv.sec),
        },
        'q_end_smoothed': q_end_smooth.tolist(),
    }


def alignvn_24state_iter_yaw_hold_with_trace(
    imu: np.ndarray,
    qnb: np.ndarray,
    pos: np.ndarray,
    phi0: np.ndarray,
    imuerr: dict[str, np.ndarray],
    wvn: np.ndarray,
    cfg: MethodConfig,
    truth_att: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, Any]]:
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
    final_iter_trace: dict[str, Any] | None = None

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

        collect_trace = (iteration == cfg.max_iter)
        q_post_seq: list[list[float]] = []
        step_qupdt2_mats: list[list[list[float]]] = []

        for k in range(0, length, nn):
            wvm = imu_corr[k:k + nn, 0:6]
            phim, dvbm = acc18.cnscl(wvm)
            cnb = acc18.q2mat(qnbi)
            dvn = cnn @ cnb @ dvbm
            vn = vn + dvn + eth.gn * nts

            if collect_trace:
                step_qupdt2_mats.append(build_qupdt2_matrix(acc18, phim, eth.wnin * nts).tolist())
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

            if collect_trace:
                q_post_seq.append(qnbi.tolist())

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

        if collect_trace:
            final_iter_trace = {
                'q_post_seq': q_post_seq,
                'step_qupdt2_mats': step_qupdt2_mats,
                'q_end_forward': qnbi.tolist(),
            }

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
    assert final_iter_trace is not None
    return final_att, iter_logs, final_iter_trace


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

    final_att, iter_logs, final_iter_trace = alignvn_24state_iter_yaw_hold_with_trace(
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
    smoothing = smooth_terminal_yaw_from_trace(acc18, final_iter_trace)
    smoothed_att = acc18.q2att(np.asarray(smoothing['q_end_smoothed']))
    smoothed_err_arcsec = acc18.qq2phi(acc18.a2qua(smoothed_att), acc18.a2qua(truth_att)) / acc18.glv.sec

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
            },
            'yaw_hold24_terminal_smooth': {
                'config': {
                    **asdict(cfg),
                    'terminal_smoothing': {
                        'type': 'scalar_rts_on_terminal_candidate_yaw',
                        'tail_fraction': TAIL_FRACTION,
                        'process_ratio': RTS_PROCESS_RATIO,
                        'measurement_floor_arcsec': RTS_MEAS_FLOOR_ARCSEC,
                    },
                },
                'final_att_err_arcsec': [float(x) for x in smoothed_err_arcsec],
                'final_att_err_abs_arcsec': [float(abs(x)) for x in smoothed_err_arcsec],
                'final_att_err_norm_arcsec': float(np.linalg.norm(smoothed_err_arcsec)),
                'final_yaw_abs_arcsec': float(abs(smoothed_err_arcsec[2])),
                'terminal_smoothing_debug': smoothing,
            },
        },
    }


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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def build_reference_index(reference_payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    rows = reference_payload.get('summary_rows', [])
    return {(row['condition'], row['method']): row for row in rows}


def build_condition_judgement(reference_index: dict[tuple[str, str], dict[str, Any]],
                              condition: str,
                              smooth_stats: dict[str, Any]) -> dict[str, Any]:
    staged = reference_index[(condition, 'staged24')]
    yaw_hold = reference_index[(condition, 'yaw_hold24')]
    delta_vs_yaw_hold = {
        'pitch_mean_abs_arcsec': smooth_stats['pitch_mean_abs_arcsec'] - yaw_hold['pitch_mean_abs_arcsec'],
        'yaw_abs_mean_arcsec': smooth_stats['yaw_abs_mean_arcsec'] - yaw_hold['yaw_abs_mean_arcsec'],
        'norm_mean_arcsec': smooth_stats['norm_mean_arcsec'] - yaw_hold['norm_mean_arcsec'],
        'yaw_abs_median_arcsec': smooth_stats['yaw_abs_median_arcsec'] - yaw_hold['yaw_abs_median_arcsec'],
        'yaw_abs_max_arcsec': smooth_stats['yaw_abs_max_arcsec'] - yaw_hold['yaw_abs_max_arcsec'],
    }
    delta_vs_staged = {
        'yaw_abs_mean_arcsec': smooth_stats['yaw_abs_mean_arcsec'] - staged['yaw_abs_mean_arcsec'],
        'norm_mean_arcsec': smooth_stats['norm_mean_arcsec'] - staged['norm_mean_arcsec'],
    }

    if delta_vs_yaw_hold['yaw_abs_mean_arcsec'] < -5.0:
        verdict = 'terminal smoothing gives a material yaw-mean gain over yaw_hold24'
    elif delta_vs_yaw_hold['yaw_abs_mean_arcsec'] < -1.0:
        verdict = 'terminal smoothing gives only a modest yaw-mean gain over yaw_hold24'
    elif delta_vs_yaw_hold['yaw_abs_mean_arcsec'] < 1.0:
        verdict = 'terminal smoothing is effectively a wash versus yaw_hold24'
    else:
        verdict = 'terminal smoothing makes yaw_hold24 worse'

    return {
        'condition': condition,
        'verdict': verdict,
        'delta_vs_yaw_hold24': delta_vs_yaw_hold,
        'delta_vs_staged24': delta_vs_staged,
    }


def build_global_verdict(judgements: list[dict[str, Any]]) -> str:
    rows = {item['condition']: item for item in judgements}
    tiny = rows['tiny_gm_matched']
    dy = tiny['delta_vs_yaw_hold24']['yaw_abs_mean_arcsec']
    if dy < -5.0:
        core = 'On tiny_gm_matched, the offline terminal smoothing layer gives a real gain.'
    elif dy < -1.0:
        core = 'On tiny_gm_matched, the offline terminal smoothing layer only gives a small gain.'
    elif dy < 1.0:
        core = 'On tiny_gm_matched, the offline terminal smoothing layer is basically a wash.'
    else:
        core = 'On tiny_gm_matched, the offline terminal smoothing layer makes the result worse.'

    if rows['tiny_gm_matched']['delta_vs_yaw_hold24']['yaw_abs_mean_arcsec'] <= -40.0:
        plausibility = 'That would make ~20 arcsec look newly plausible under the current setup.'
    else:
        plausibility = 'This still does not make ~20 arcsec look realistic under the current setup.'
    return core + ' ' + plausibility


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        '# Chapter 4 DAR yaw-hold + terminal smoothing probe (2026-03-31)',
        '',
        '## Goal',
        '- Test one lightweight fixed-interval smoothing layer on top of the current best yaw-directed matched-GM variant.',
        '- Keep the same DAR path, same seeds `[0,1,2,3,4]`, same `iter=5`, same matched-GM setup.',
        '',
        '## What was actually smoothed',
        '- Base method: current `yaw_hold24` (same `kg_z` delayed release probe already saved separately).',
        '- Only the **terminal yaw correction** was smoothed offline.',
        '- Construction:',
        '  1. run the final `yaw_hold24` outer iteration forward as usual;',
        '  2. save the posterior quaternion after every step;',
        '  3. from each saved posterior, propagate open-loop to the interval end to build a sequence of **candidate terminal attitudes**;',
        '  4. on the **last 50%** of that candidate-terminal sequence, extract the candidate terminal yaw correction relative to the forward terminal estimate;',
        '  5. run a **scalar random-walk RTS smoother** on that yaw-correction sequence;',
        '  6. apply the smoothed constant yaw correction once to the forward terminal quaternion.',
        '- So this is an RTS-style forward-backward smoother on a **derived terminal-yaw correction sequence**, not a full 24-state RTS rewrite.',
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
        '## Smoother diagnostics (new method only)',
        '',
        '| condition | mean smoothed yaw correction (") | tail raw yaw std (") | tail candidate count |',
        '|---|---:|---:|---:|',
    ])
    for item in payload['smoothing_diagnostics_rows']:
        lines.append(
            f"| {item['condition']} | {item['smoothed_yaw_correction_arcsec']:.3f} | {item['tail_raw_yaw_std_arcsec']:.3f} | {item['num_tail_candidates']} |"
        )

    lines.extend([
        '',
        '## Verdict vs yaw_hold24',
        '',
    ])
    for item in payload['judgements']:
        d = item['delta_vs_yaw_hold24']
        lines.append(
            f"- {item['condition']}: {item['verdict']} (Δpitch_mean_abs={d['pitch_mean_abs_arcsec']:+.3f}\", Δyaw_mean={d['yaw_abs_mean_arcsec']:+.3f}\", Δnorm_mean={d['norm_mean_arcsec']:+.3f}\", Δyaw_median={d['yaw_abs_median_arcsec']:+.3f}\", Δyaw_max={d['yaw_abs_max_arcsec']:+.3f}\")"
        )

    lines.extend([
        '',
        '## Crisp verdict',
        f"- {payload['global_verdict']}",
        '',
        '## Files',
        f'- script: `{SCRIPTS_DIR / "alignvn_dar_hybrid24_truth_gm_yaw_hold_terminal_smoothing_probe_2026-03-31.py"}`',
        f'- plain/staged matched reference: `{MATCHED_JSON}`',
        f'- yaw-hold reference: `{YAW_HOLD_JSON}`',
        f'- json: `{OUT_JSON}`',
        f'- md: `{OUT_MD}`',
        f'- csv: `{OUT_CSV}`',
        '',
    ])
    return '\n'.join(lines) + '\n'


def main() -> None:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    matched_reference = load_json(MATCHED_JSON)
    yaw_hold_reference = load_json(YAW_HOLD_JSON)
    matched_index = build_reference_index(matched_reference)
    yaw_hold_index = build_reference_index(yaw_hold_reference)
    merged_reference_index = {**matched_index, **yaw_hold_index}

    tasks = [(condition, seed) for condition in CONDITIONS for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as ex:
        seed_runs = list(ex.map(run_seed_condition, tasks))

    seed_runs.sort(key=lambda x: (x['condition'], x['seed']))

    grouped: dict[str, list[dict[str, Any]]] = {condition: [] for condition in CONDITIONS}
    for item in seed_runs:
        grouped[item['condition']].append(item)

    smooth_summary_by_condition: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    judgements: list[dict[str, Any]] = []
    smoothing_diagnostics_rows: list[dict[str, Any]] = []

    for condition in CONDITIONS:
        rows = grouped[condition]
        smooth_stats = summarize_method_rows(rows, 'yaw_hold24_terminal_smooth')
        smooth_summary_by_condition[condition] = smooth_stats

        summary_rows.append({'condition': condition, 'method': 'plain24', **deepcopy(merged_reference_index[(condition, 'plain24')])})
        summary_rows[-1].pop('condition', None)
        summary_rows[-1].pop('method', None)
        summary_rows[-1]['condition'] = condition
        summary_rows[-1]['method'] = 'plain24'

        summary_rows.append({'condition': condition, 'method': 'staged24', **deepcopy(merged_reference_index[(condition, 'staged24')])})
        summary_rows[-1].pop('condition', None)
        summary_rows[-1].pop('method', None)
        summary_rows[-1]['condition'] = condition
        summary_rows[-1]['method'] = 'staged24'

        summary_rows.append({'condition': condition, 'method': 'yaw_hold24', **deepcopy(merged_reference_index[(condition, 'yaw_hold24')])})
        summary_rows[-1].pop('condition', None)
        summary_rows[-1].pop('method', None)
        summary_rows[-1]['condition'] = condition
        summary_rows[-1]['method'] = 'yaw_hold24'

        summary_rows.append({'condition': condition, 'method': 'yaw_hold24_terminal_smooth', **smooth_stats})

        judgements.append(build_condition_judgement(merged_reference_index, condition, smooth_stats))

        tail_debug = rows[0]['methods']['yaw_hold24_terminal_smooth']['terminal_smoothing_debug']['yaw_tail_rts']
        # Use per-seed average diagnostics across all seeds, not seed0 only.
        smooth_debug_rows = [row['methods']['yaw_hold24_terminal_smooth']['terminal_smoothing_debug'] for row in rows]
        smoothing_diagnostics_rows.append({
            'condition': condition,
            'smoothed_yaw_correction_arcsec': float(np.mean([d['yaw_tail_rts']['smoothed_value_arcsec'] for d in smooth_debug_rows])),
            'tail_raw_yaw_std_arcsec': float(np.mean([d['yaw_tail_rts']['raw_std_arcsec'] for d in smooth_debug_rows])),
            'num_tail_candidates': int(np.mean([d['num_tail_candidates'] for d in smooth_debug_rows])),
        })

    global_verdict = build_global_verdict(judgements)

    csv_lines = ['condition,method,pitch_mean_abs_arcsec,yaw_abs_mean_arcsec,norm_mean_arcsec,yaw_abs_median_arcsec,yaw_abs_max_arcsec']
    for row in summary_rows:
        csv_lines.append(
            f"{row['condition']},{row['method']},{row['pitch_mean_abs_arcsec']:.6f},{row['yaw_abs_mean_arcsec']:.6f},{row['norm_mean_arcsec']:.6f},{row['yaw_abs_median_arcsec']:.6f},{row['yaw_abs_max_arcsec']:.6f}"
        )

    payload = {
        'meta': {
            'date': '2026-03-31',
            'purpose': 'offline terminal-yaw smoothing probe on top of yaw_hold24 under matched-GM Chapter 4 DAR conditions',
            'conditions': CONDITIONS,
            'seeds': SEEDS,
            'methods': ['plain24', 'staged24', 'yaw_hold24', 'yaw_hold24_terminal_smooth'],
            'runtime_sec': time.time() - t0,
            'tail_fraction': TAIL_FRACTION,
            'rts_process_ratio': RTS_PROCESS_RATIO,
            'rts_measurement_floor_arcsec': RTS_MEAS_FLOOR_ARCSEC,
            'reference_plain_staged_loaded_from_artifact': True,
            'reference_yaw_hold_loaded_from_artifact': True,
        },
        'reference_matched_json': str(MATCHED_JSON),
        'reference_yaw_hold_json': str(YAW_HOLD_JSON),
        'seed_runs': seed_runs,
        'smooth_summary_by_condition': smooth_summary_by_condition,
        'summary_rows': summary_rows,
        'judgements': judgements,
        'smoothing_diagnostics_rows': smoothing_diagnostics_rows,
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
        'smoothing_diagnostics_rows': smoothing_diagnostics_rows,
        'global_verdict': global_verdict,
        'out_json': str(OUT_JSON),
        'out_md': str(OUT_MD),
        'out_csv': str(OUT_CSV),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
