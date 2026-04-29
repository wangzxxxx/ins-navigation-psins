#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
PSINS_ROOT = WORKSPACE / 'tmp_psins_py'
if str(PSINS_ROOT) not in sys.path:
    sys.path.insert(0, str(PSINS_ROOT))

BASE12_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
H24_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_hybrid24_staged_py_2026-03-30.py'
SCALE18_PATH = WORKSPACE / 'psins_method_bench' / 'scripts' / 'compare_dualpath_scaleonly_g2_vs_g3g4_2026-04-09.py'
PURE_SCD_PATH = WORKSPACE / 'scripts' / 'compare_ch4_pure_scd_vs_freeze_2026-04-03.py'

RESULTS_DIR = WORKSPACE / 'psins_method_bench' / 'results'
REPORTS_DIR = WORKSPACE / 'reports'
OUT_JSON = RESULTS_DIR / 'compare_dualpath_three_methods_custom_noise_half_2026-04-12.json'
OUT_MD = REPORTS_DIR / 'psins_dualpath_three_methods_custom_noise_half_2026-04-12.md'

SEEDS = [0, 1, 2, 3, 4]
TS = 0.01
WVN = np.array([0.01, 0.01, 0.01], dtype=float)
PHI_DEG = np.array([0.1, 0.1, 0.5], dtype=float)
ROT_GATE_DPS = 5.0
MAX_ITER = 5
WASH_SCALE = 0.5
SCALE_WASH_SCALE = 0.5
PURE_SCD_ALPHA = 0.995
PURE_SCD_TRANSITION_DURATION_S = 2.0
PURE_SCD_APPLY_AFTER_RELEASE_ITER = 1

ARW_DPS_SQRT_H = 0.00025
VRW_UG_SQRT_HZ = 0.25
BI_G_DPH = 0.00035
BI_A_UG = 2.5
TAU_G_S = 300.0
TAU_A_S = 300.0
TRUTH_SCALE_PPM = 30.0

_BASE12 = None
_H24 = None
_SCALE18 = None
_PURE_SCD = None


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_base12():
    global _BASE12
    if _BASE12 is None:
        _BASE12 = load_module('dualpath_custom_noise_half_base12_20260412', BASE12_PATH)
    return _BASE12


def load_h24():
    global _H24
    if _H24 is None:
        _H24 = load_module('dualpath_custom_noise_half_h24_20260412', H24_PATH)
    return _H24


def load_scale18():
    global _SCALE18
    if _SCALE18 is None:
        _SCALE18 = load_module('dualpath_custom_noise_half_scale18_20260412', SCALE18_PATH)
    return _SCALE18


def load_pure_scd():
    global _PURE_SCD
    if _PURE_SCD is None:
        _PURE_SCD = load_module('dualpath_custom_noise_half_pure_scd_20260412', PURE_SCD_PATH)
    return _PURE_SCD


def build_noise_config(glv) -> dict[str, Any]:
    return {
        'semantics': 'truth IMU = white increment noise + first-order Gauss-Markov bias, not legacy imuerrset constant-bias semantics',
        'truth_noise_units': {
            'arw': 'dps/sqrt(h)',
            'vrw': 'ug/sqrt(Hz)',
            'bi_g': 'deg/hour (GM steady-state sigma)',
            'bi_a': 'ug (GM steady-state sigma)',
            'tau_g': 's',
            'tau_a': 's',
        },
        'truth_noise_values': {
            'arw_dps_sqrt_h': ARW_DPS_SQRT_H,
            'vrw_ug_sqrt_hz': VRW_UG_SQRT_HZ,
            'bi_g_dph': BI_G_DPH,
            'bi_a_ug': BI_A_UG,
            'tau_g_s': TAU_G_S,
            'tau_a_s': TAU_A_S,
        },
        'truth_noise_internal_units': {
            'arw_radps_sqrt_h': float(ARW_DPS_SQRT_H * glv.dpsh),
            'vrw_mps2_sqrt_hz': float(VRW_UG_SQRT_HZ * glv.ugpsHz),
            'bi_g_radps_sigma': float(BI_G_DPH * glv.dph),
            'bi_a_mps2_sigma': float(BI_A_UG * glv.ug),
        },
        'truth_scale_injection': {
            'dKg_diag_ppm': [TRUTH_SCALE_PPM, TRUTH_SCALE_PPM, TRUTH_SCALE_PPM],
            'dKa_diag_ppm': [TRUTH_SCALE_PPM, TRUTH_SCALE_PPM, TRUTH_SCALE_PPM],
            'source': 'kept from base12.build_imuerr() convention',
        },
        'filter_mapping': {
            '24state_ng_xa_mapping': 'matched_GM',
            'ng_sigma_dph': [BI_G_DPH, BI_G_DPH, BI_G_DPH],
            'tau_g_s': [TAU_G_S, TAU_G_S, TAU_G_S],
            'xa_sigma_ug': [BI_A_UG, BI_A_UG, BI_A_UG],
            'tau_a_s': [TAU_A_S, TAU_A_S, TAU_A_S],
            '18state_scale_only': 'no ng/xa states',
        },
        'filter_prior_proxy_for_eb_db': {
            'eb_dph': [BI_G_DPH, BI_G_DPH, BI_G_DPH],
            'db_ug': [BI_A_UG, BI_A_UG, BI_A_UG],
            'web_dps_sqrt_h': [ARW_DPS_SQRT_H, ARW_DPS_SQRT_H, ARW_DPS_SQRT_H],
            'wdb_ug_sqrt_hz': [VRW_UG_SQRT_HZ, VRW_UG_SQRT_HZ, VRW_UG_SQRT_HZ],
        },
    }


def build_filter_imuerr(glv) -> dict[str, np.ndarray]:
    return {
        'eb': np.full(3, BI_G_DPH * glv.dph, dtype=float),
        'db': np.full(3, BI_A_UG * glv.ug, dtype=float),
        'web': np.full(3, ARW_DPS_SQRT_H * glv.dpsh, dtype=float),
        'wdb': np.full(3, VRW_UG_SQRT_HZ * glv.ugpsHz, dtype=float),
        'dKg': np.diag(np.full(3, TRUTH_SCALE_PPM * glv.ppm, dtype=float)),
        'dKa': np.diag(np.full(3, TRUTH_SCALE_PPM * glv.ppm, dtype=float)),
    }


def _gm_sequence(m: int, ts: float, sigma_ss: float, tau_s: float, rng: np.random.Generator) -> np.ndarray:
    if sigma_ss <= 0.0 or tau_s <= 0.0:
        return np.zeros((m, 3), dtype=float)
    coeff = math.exp(-ts / tau_s)
    sigma_drive = sigma_ss * math.sqrt(max(1.0 - coeff ** 2, 0.0))
    out = np.zeros((m, 3), dtype=float)
    b = np.zeros(3, dtype=float)
    for k in range(m):
        b = coeff * b + sigma_drive * rng.standard_normal(3)
        out[k] = b
    return out


def imuadderr_full_with_scale(
    imu_in: np.ndarray,
    ts: float,
    *,
    dKg: np.ndarray | None,
    dKa: np.ndarray | None,
    arw: float,
    vrw: float,
    bi_g: float,
    tau_g: float,
    bi_a: float,
    tau_a: float,
    seed: int,
) -> np.ndarray:
    """Apply diagonal scale truth injection + white increments + GM bias increments.

    Semantics:
    - arw / vrw are increment-domain white noise coefficients
    - bi_g / bi_a are steady-state sigma of first-order GM rate/accel bias
    - generated GM biases are converted to increments via bias * ts
    """
    imu = np.array(imu_in, copy=True, dtype=float)
    rng = np.random.default_rng(seed)
    m = imu.shape[0]
    sts = math.sqrt(ts)

    if dKg is not None:
        Kg = np.eye(3) + np.asarray(dKg, dtype=float)
        imu[:, 0:3] = imu[:, 0:3] @ Kg.T
    if dKa is not None:
        Ka = np.eye(3) + np.asarray(dKa, dtype=float)
        imu[:, 3:6] = imu[:, 3:6] @ Ka.T

    if arw > 0.0:
        imu[:, 0:3] += arw * sts * rng.standard_normal((m, 3))
    if vrw > 0.0:
        imu[:, 3:6] += vrw * sts * rng.standard_normal((m, 3))

    if bi_g > 0.0 and tau_g > 0.0:
        gyro_bias = _gm_sequence(m, ts, bi_g, tau_g, rng)
        imu[:, 0:3] += gyro_bias * ts
    if bi_a > 0.0 and tau_a > 0.0:
        acc_bias = _gm_sequence(m, ts, bi_a, tau_a, rng)
        imu[:, 3:6] += acc_bias * ts

    return imu


def extract_method_result(group_key: str, seed: int, last: dict[str, Any]) -> dict[str, Any]:
    err = np.array(last['att_err_arcsec'], dtype=float)
    return {
        'group_key': group_key,
        'seed': seed,
        'final_att_err_arcsec': [float(x) for x in err],
        'final_att_err_abs_arcsec': [float(x) for x in np.abs(err)],
        'final_pitch_arcsec': float(err[0]),
        'final_roll_arcsec': float(err[1]),
        'final_yaw_arcsec': float(err[2]),
        'final_yaw_abs_arcsec': float(abs(err[2])),
        'final_att_err_norm_arcsec': float(last['att_err_norm_arcsec']),
        'iter_logs': last.get('iter_logs_full') or [],
    }


def run_seed(seed: int) -> dict[str, Any]:
    base12 = load_base12()
    h24 = load_h24()
    scale18 = load_scale18()
    pure = load_pure_scd()
    acc18 = h24.load_acc18()
    glv = acc18.glv

    att0_ref = np.array([0.0, 0.0, 0.0], dtype=float)
    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    rot_paras = base12.build_rot_paras()
    att_truth = acc18.attrottt(att0_ref, rot_paras, TS)
    imu_clean, _ = acc18.avp2imu(att_truth, pos0)

    filter_imuerr = build_filter_imuerr(glv)
    imu_noisy = imuadderr_full_with_scale(
        imu_clean,
        TS,
        dKg=filter_imuerr['dKg'],
        dKa=filter_imuerr['dKa'],
        arw=float(ARW_DPS_SQRT_H * glv.dpsh),
        vrw=float(VRW_UG_SQRT_HZ * glv.ugpsHz),
        bi_g=float(BI_G_DPH * glv.dph),
        tau_g=TAU_G_S,
        bi_a=float(BI_A_UG * glv.ug),
        tau_a=TAU_A_S,
        seed=seed,
    )

    phi = PHI_DEG * glv.deg
    att0_guess = acc18.q2att(base12.qaddphi(acc18.a2qua(att0_ref), phi))
    truth_att = att_truth[-1, 0:3]

    _, scale_logs = scale18.alignvn_scale18_iter(
        imu=imu_noisy.copy(),
        qnb=att0_guess,
        pos=pos0,
        phi0=phi,
        imuerr=filter_imuerr,
        wvn=WVN.copy(),
        max_iter=MAX_ITER,
        truth_att=truth_att,
        wash_scale=WASH_SCALE,
        scale_wash_scale=SCALE_WASH_SCALE,
        carry_att_seed=True,
    )
    scale_last = dict(scale_logs[-1])
    scale_last['iter_logs_full'] = scale_logs

    cfg_plain24 = h24.Hybrid24Config(
        name='g3_markov_rotation_custom_noise',
        label='G3 Markov plain24 @ dual-axis custom-noise rerun',
        seeds=[seed],
        max_iter=MAX_ITER,
        wash_scale=WASH_SCALE,
        scale_wash_scale=SCALE_WASH_SCALE,
        carry_att_seed=True,
        staged_release=False,
        rot_gate_dps=ROT_GATE_DPS,
        ng_sigma_dph=[BI_G_DPH, BI_G_DPH, BI_G_DPH],
        tau_g_s=[TAU_G_S, TAU_G_S, TAU_G_S],
        xa_sigma_ug=[BI_A_UG, BI_A_UG, BI_A_UG],
        tau_a_s=[TAU_A_S, TAU_A_S, TAU_A_S],
        note='matched GM mapping from truth noise to 24-state ng/xa states',
    )
    _, plain24_logs = h24.alignvn_24state_iter(
        imu=imu_noisy.copy(),
        qnb=att0_guess,
        pos=pos0,
        phi0=phi,
        imuerr=filter_imuerr,
        wvn=WVN.copy(),
        cfg=cfg_plain24,
        truth_att=truth_att,
    )
    plain24_last = dict(plain24_logs[-1])
    plain24_last['iter_logs_full'] = plain24_logs

    cfg_scd24 = pure.h24.Hybrid24Config(
        name='g4_scd_rotation_custom_noise',
        label='G4 Markov + pure-SCD @ dual-axis custom-noise rerun',
        seeds=[seed],
        max_iter=MAX_ITER,
        wash_scale=WASH_SCALE,
        scale_wash_scale=SCALE_WASH_SCALE,
        carry_att_seed=True,
        staged_release=False,
        rot_gate_dps=ROT_GATE_DPS,
        ng_sigma_dph=[BI_G_DPH, BI_G_DPH, BI_G_DPH],
        tau_g_s=[TAU_G_S, TAU_G_S, TAU_G_S],
        xa_sigma_ug=[BI_A_UG, BI_A_UG, BI_A_UG],
        tau_a_s=[TAU_A_S, TAU_A_S, TAU_A_S],
        note='pure-SCD line with matched GM mapping and scale states active from iter1',
    )
    scd_cfg = pure.SCDConfig(
        enabled=True,
        alpha=PURE_SCD_ALPHA,
        transition_duration_s=PURE_SCD_TRANSITION_DURATION_S,
        apply_after_release_iter=PURE_SCD_APPLY_AFTER_RELEASE_ITER,
        note='pure-SCD once-per-static-phase covariance suppression after rotation release',
    )
    _, scd24_logs = pure.alignvn_24state_iter_pure_scd(
        imu=imu_noisy.copy(),
        qnb=att0_guess,
        pos=pos0,
        phi0=phi,
        imuerr=filter_imuerr,
        wvn=WVN.copy(),
        cfg=cfg_scd24,
        truth_att=truth_att,
        scd=scd_cfg,
    )
    scd24_last = dict(scd24_logs[-1])
    scd24_last['iter_logs_full'] = scd24_logs

    return {
        'seed': seed,
        'g2_scaleonly_rotation': extract_method_result('g2_scaleonly_rotation', seed, scale_last),
        'g3_markov_rotation': extract_method_result('g3_markov_rotation', seed, plain24_last),
        'g4_scd_rotation': extract_method_result('g4_scd_rotation', seed, scd24_last),
    }


def summarize_method(group_key: str, seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row[group_key] for row in seed_rows]
    errs = np.array([row['final_att_err_arcsec'] for row in rows], dtype=float)
    abs_errs = np.abs(errs)
    norms = np.array([row['final_att_err_norm_arcsec'] for row in rows], dtype=float)
    pitch_signed = errs[:, 0]
    roll_signed = errs[:, 1]
    yaw_signed = errs[:, 2]

    return {
        'component_order_signed_att_err_arcsec': ['pitch', 'roll', 'yaw'],
        'pitch_mean_abs_arcsec': float(abs_errs[:, 0].mean()),
        'pitch_1sigma_arcsec': float(pitch_signed.std(ddof=1)) if len(rows) > 1 else 0.0,
        'roll_mean_abs_arcsec': float(abs_errs[:, 1].mean()),
        'roll_1sigma_arcsec': float(roll_signed.std(ddof=1)) if len(rows) > 1 else 0.0,
        'yaw_mean_abs_arcsec': float(abs_errs[:, 2].mean()),
        'yaw_1sigma_arcsec': float(yaw_signed.std(ddof=1)) if len(rows) > 1 else 0.0,
        'norm_mean_arcsec': float(norms.mean()),
        'norm_1sigma_arcsec': float(norms.std(ddof=1)) if len(rows) > 1 else 0.0,
        'mean_signed_arcsec': errs.mean(axis=0).tolist(),
        'mean_abs_arcsec': abs_errs.mean(axis=0).tolist(),
        'per_seed_final_pitch_arcsec': pitch_signed.tolist(),
        'per_seed_final_roll_arcsec': roll_signed.tolist(),
        'per_seed_final_yaw_arcsec': yaw_signed.tolist(),
        'per_seed_final_yaw_abs_arcsec': np.abs(yaw_signed).tolist(),
        'per_seed_final_norm_arcsec': norms.tolist(),
        'per_seed': rows,
    }


def build_method_definitions() -> dict[str, Any]:
    return {
        'g2_scaleonly_rotation': {
            'display': '双轴 18-state (scale-only)',
            'source_script': str(SCALE18_PATH),
            'state_layout': ['phi(3)', 'dv(3)', 'eb(3)', 'db(3)', 'kg(3)', 'ka(3)'],
            'path': 'old Chapter-4 dual-axis build_rot_paras()',
            'notes': [
                'no ng/xa Markov states',
                'same iterative semantics as accepted dualpath three-method line',
                'high-rotation gate only controls scale-state coupling',
            ],
            'config': {
                'max_iter': MAX_ITER,
                'carry_att_seed': True,
                'wash_scale': WASH_SCALE,
                'scale_wash_scale': SCALE_WASH_SCALE,
                'rot_gate_dps': ROT_GATE_DPS,
                'phi_guess_deg': PHI_DEG.tolist(),
                'wvn': WVN.tolist(),
            },
        },
        'g3_markov_rotation': {
            'display': '双轴 24-state (Markov/plain24)',
            'source_script': str(H24_PATH),
            'state_layout': ['phi(3)', 'dv(3)', 'eb(3)', 'db(3)', 'ng(3)', 'xa(3)', 'kg(3)', 'ka(3)'],
            'path': 'old Chapter-4 dual-axis build_rot_paras()',
            'notes': [
                'plain24 / Markov baseline line',
                'matched GM mapping used for ng/xa states',
                'scale states active from iteration 1',
            ],
            'config': {
                'max_iter': MAX_ITER,
                'carry_att_seed': True,
                'wash_scale': WASH_SCALE,
                'scale_wash_scale': SCALE_WASH_SCALE,
                'rot_gate_dps': ROT_GATE_DPS,
                'phi_guess_deg': PHI_DEG.tolist(),
                'wvn': WVN.tolist(),
                'ng_sigma_dph': [BI_G_DPH, BI_G_DPH, BI_G_DPH],
                'tau_g_s': [TAU_G_S, TAU_G_S, TAU_G_S],
                'xa_sigma_ug': [BI_A_UG, BI_A_UG, BI_A_UG],
                'tau_a_s': [TAU_A_S, TAU_A_S, TAU_A_S],
            },
        },
        'g4_scd_rotation': {
            'display': '双轴 24-state + SCD (pure-SCD)',
            'source_script': str(PURE_SCD_PATH),
            'state_layout': ['phi(3)', 'dv(3)', 'eb(3)', 'db(3)', 'ng(3)', 'xa(3)', 'kg(3)', 'ka(3)'],
            'path': 'old Chapter-4 dual-axis build_rot_paras()',
            'notes': [
                'pure-SCD line; scale states active from iteration 1',
                'matched GM mapping used for ng/xa states',
                'SCD is applied once per static phase after transition duration is met',
            ],
            'config': {
                'max_iter': MAX_ITER,
                'carry_att_seed': True,
                'wash_scale': WASH_SCALE,
                'scale_wash_scale': SCALE_WASH_SCALE,
                'rot_gate_dps': ROT_GATE_DPS,
                'phi_guess_deg': PHI_DEG.tolist(),
                'wvn': WVN.tolist(),
                'ng_sigma_dph': [BI_G_DPH, BI_G_DPH, BI_G_DPH],
                'tau_g_s': [TAU_G_S, TAU_G_S, TAU_G_S],
                'xa_sigma_ug': [BI_A_UG, BI_A_UG, BI_A_UG],
                'tau_a_s': [TAU_A_S, TAU_A_S, TAU_A_S],
                'scd_alpha': PURE_SCD_ALPHA,
                'scd_transition_duration_s': PURE_SCD_TRANSITION_DURATION_S,
                'scd_apply_after_release_iter': PURE_SCD_APPLY_AFTER_RELEASE_ITER,
            },
        },
    }


def build_summary_rows(groups: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for group_key in ['g2_scaleonly_rotation', 'g3_markov_rotation', 'g4_scd_rotation']:
        st = groups[group_key]
        rows.append({
            'group_key': group_key,
            'display': build_method_definitions()[group_key]['display'],
            'roll_mean_abs_arcsec': st['roll_mean_abs_arcsec'],
            'roll_1sigma_arcsec': st['roll_1sigma_arcsec'],
            'pitch_mean_abs_arcsec': st['pitch_mean_abs_arcsec'],
            'pitch_1sigma_arcsec': st['pitch_1sigma_arcsec'],
            'yaw_mean_abs_arcsec': st['yaw_mean_abs_arcsec'],
            'yaw_1sigma_arcsec': st['yaw_1sigma_arcsec'],
            'norm_mean_arcsec': st['norm_mean_arcsec'],
            'norm_1sigma_arcsec': st['norm_1sigma_arcsec'],
        })
    return rows


def build_markdown(payload: dict[str, Any]) -> str:
    rows = payload['summary_rows']
    lines = [
        '# PSINS 双轴三方法自对准自定义噪声×0.5重跑（2026-04-12）',
        '',
        '- 路径统一：旧 Chapter-4 双轴路径 `build_rot_paras()`',
        f'- seeds: `{SEEDS}`',
        f'- phi guess (deg): `{PHI_DEG.tolist()}`；wvn: `{WVN.tolist()}`',
        '- truth 侧保留 `dKg/dKa = 30 ppm` 对角注入',
        '- 新噪声口径：white increment + first-order GM bias（不是旧 `imuerrset` 常值 bias 口径）',
        '- 误差分量顺序严格按 PSINS `q2att/qq2phi`：`[pitch, roll, yaw]`；因此下面 roll 指第 2 分量、pitch 指第 1 分量',
        '',
        '## 核心结果',
        '',
        '| 方法 | roll mean abs (") | roll 1σ (") | pitch mean abs (") | pitch 1σ (") | yaw mean abs (") | yaw 1σ (") | norm mean (") | norm 1σ (") |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in rows:
        lines.append(
            f"| {row['display']} | {row['roll_mean_abs_arcsec']:.6f} | {row['roll_1sigma_arcsec']:.6f} | {row['pitch_mean_abs_arcsec']:.6f} | {row['pitch_1sigma_arcsec']:.6f} | {row['yaw_mean_abs_arcsec']:.6f} | {row['yaw_1sigma_arcsec']:.6f} | {row['norm_mean_arcsec']:.6f} | {row['norm_1sigma_arcsec']:.6f} |"
        )

    lines.extend([
        '',
        '## Per-seed final yaw / norm',
        '',
    ])
    for group_key in ['g2_scaleonly_rotation', 'g3_markov_rotation', 'g4_scd_rotation']:
        st = payload['groups'][group_key]
        lines.extend([
            f"### {payload['method_definition'][group_key]['display']}",
            f"- final yaw signed (arcsec): `{[round(x, 6) for x in st['per_seed_final_yaw_arcsec']]}`",
            f"- final yaw abs (arcsec): `{[round(x, 6) for x in st['per_seed_final_yaw_abs_arcsec']]}`",
            f"- final norm (arcsec): `{[round(x, 6) for x in st['per_seed_final_norm_arcsec']]}`",
            '',
        ])

    lines.extend([
        '## 关键假设 / 口径说明',
        '',
        '- 24-state / 24-state+SCD 的 `ng/xa` 采用 **matched GM mapping**：`sigma = bi`，`tau = 300 s`。',
        '- 18-state(scale-only) 不含 `ng/xa`，仅保留 `eb/db/kg/ka`。',
        '- `imuerr` 中的 `eb/db/web/wdb` 在本次实现里作为 filter 先验/Qk 代理口径；truth 侧实际注入采用 GM + white-noise。',
        '- pure-SCD 使用 `alpha=0.995`、`transition_duration_s=2.0`、`apply_after_release_iter=1`。',
        '',
        '## 文件',
        '',
        f'- script: `{payload["files"]["script"]}`',
        f'- json: `{payload["files"]["json"]}`',
        f'- markdown: `{payload["files"]["md"]}`',
    ])
    return '\n'.join(lines) + '\n'


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    h24 = load_h24()
    acc18 = h24.load_acc18()
    noise_config = build_noise_config(acc18.glv)
    method_definition = build_method_definitions()

    seed_rows = [run_seed(seed) for seed in SEEDS]
    seed_rows.sort(key=lambda x: x['seed'])

    groups = {
        'g2_scaleonly_rotation': summarize_method('g2_scaleonly_rotation', seed_rows),
        'g3_markov_rotation': summarize_method('g3_markov_rotation', seed_rows),
        'g4_scd_rotation': summarize_method('g4_scd_rotation', seed_rows),
    }
    summary_rows = build_summary_rows(groups)

    payload = {
        'task': 'compare_dualpath_three_methods_custom_noise_half_2026_04_12',
        'path_note': 'All three methods use the same old Chapter-4 dual-axis path build_rot_paras().',
        'component_order_note': 'Signed terminal attitude error components follow PSINS q2att order [pitch, roll, yaw].',
        'noise_config': noise_config,
        'experiment_config': {
            'seeds': SEEDS,
            'ts': TS,
            'phi_guess_deg': PHI_DEG.tolist(),
            'wvn': WVN.tolist(),
            'max_iter': MAX_ITER,
            'carry_att_seed': True,
            'wash_scale': WASH_SCALE,
            'scale_wash_scale': SCALE_WASH_SCALE,
            'rot_gate_dps': ROT_GATE_DPS,
        },
        'method_definition': method_definition,
        'summary_rows': summary_rows,
        'groups': groups,
        'per_seed_rows': seed_rows,
        'files': {
            'script': str(Path(__file__)),
            'json': str(OUT_JSON),
            'md': str(OUT_MD),
        },
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    OUT_MD.write_text(build_markdown(payload), encoding='utf-8')
    print(json.dumps({
        'script': str(Path(__file__)),
        'json': str(OUT_JSON),
        'md': str(OUT_MD),
        'summary_rows': summary_rows,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
