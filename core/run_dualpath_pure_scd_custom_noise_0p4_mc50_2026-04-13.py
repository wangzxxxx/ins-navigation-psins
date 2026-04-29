#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
PSINS_ROOT = WORKSPACE / 'tmp_psins_py'
if str(PSINS_ROOT) not in sys.path:
    sys.path.insert(0, str(PSINS_ROOT))

BASE12_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
H24_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_hybrid24_staged_py_2026-03-30.py'
PURE_SCD_PATH = WORKSPACE / 'scripts' / 'compare_ch4_pure_scd_vs_freeze_2026-04-03.py'

RESULTS_DIR = WORKSPACE / 'psins_method_bench' / 'results'
REPORTS_DIR = WORKSPACE / 'reports'
OUT_JSON = RESULTS_DIR / 'dualpath_pure_scd_custom_noise_0p4_mc50_2026-04-13.json'
OUT_MD = REPORTS_DIR / 'psins_dualpath_pure_scd_custom_noise_0p4_mc50_2026-04-13.md'
DEFAULT_WORKERS = min(4, os.cpu_count() or 1)

SEEDS = list(range(50))
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

ARW_DPS_SQRT_H = 0.00020
VRW_UG_SQRT_HZ = 0.20
BI_G_DPH = 0.00028
BI_A_UG = 2.0
TAU_G_S = 300.0
TAU_A_S = 300.0
TRUTH_SCALE_PPM = 30.0

_BASE12 = None
_H24 = None
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
        _BASE12 = load_module('pure_scd_mc50_base12_20260413', BASE12_PATH)
    return _BASE12


def load_h24():
    global _H24
    if _H24 is None:
        _H24 = load_module('pure_scd_mc50_h24_20260413', H24_PATH)
    return _H24


def load_pure_scd():
    global _PURE_SCD
    if _PURE_SCD is None:
        _PURE_SCD = load_module('pure_scd_mc50_g4_20260413', PURE_SCD_PATH)
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


def run_seed(seed: int) -> dict[str, Any]:
    base12 = load_base12()
    h24 = load_h24()
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

    cfg_scd24 = pure.h24.Hybrid24Config(
        name='g4_scd_rotation_custom_noise_0p4_mc50',
        label='G4 Markov + pure-SCD @ dual-axis custom-noise 0.4x MC50',
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
    _, logs = pure.alignvn_24state_iter_pure_scd(
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
    last = dict(logs[-1])
    err = np.array(last['att_err_arcsec'], dtype=float)
    return {
        'seed': int(seed),
        'final_att_err_arcsec': [float(x) for x in err],
        'final_att_err_abs_arcsec': [float(x) for x in np.abs(err)],
        'final_pitch_arcsec': float(err[0]),
        'final_roll_arcsec': float(err[1]),
        'final_yaw_arcsec': float(err[2]),
        'final_yaw_abs_arcsec': float(abs(err[2])),
        'final_att_err_norm_arcsec': float(last['att_err_norm_arcsec']),
        'iter_logs': last.get('iter_logs_full') or logs,
    }


def run_seed_worker(seed: int) -> dict[str, Any]:
    return run_seed(seed)


def compute_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errs = np.array([row['final_att_err_arcsec'] for row in rows], dtype=float)
    abs_errs = np.abs(errs)
    norms = np.array([row['final_att_err_norm_arcsec'] for row in rows], dtype=float)
    sigmas = errs.std(axis=0, ddof=1) if len(rows) > 1 else np.zeros(3)
    rms = np.sqrt(np.mean(errs ** 2, axis=0))
    within20 = (abs_errs < 20.0).mean(axis=0)
    pitch = errs[:, 0]
    roll = errs[:, 1]
    yaw = errs[:, 2]
    yaw_abs = abs_errs[:, 2]
    return {
        'component_order_signed_att_err_arcsec': ['pitch', 'roll', 'yaw'],
        'mean_signed_arcsec': errs.mean(axis=0).tolist(),
        'std_signed_arcsec_1sigma': sigmas.tolist(),
        'rms_arcsec': rms.tolist(),
        'mean_abs_arcsec': abs_errs.mean(axis=0).tolist(),
        'median_abs_arcsec': np.median(abs_errs, axis=0).tolist(),
        'within20_abs_rate': within20.tolist(),
        'max_abs_arcsec': abs_errs.max(axis=0).tolist(),
        'norm_mean_arcsec': float(norms.mean()),
        'norm_median_arcsec': float(np.median(norms)),
        'norm_1sigma_arcsec': float(norms.std(ddof=1)) if len(rows) > 1 else 0.0,
        'yaw_abs_mean_arcsec': float(yaw_abs.mean()),
        'yaw_abs_median_arcsec': float(np.median(yaw_abs)),
        'pitch_signed_range_arcsec': [float(pitch.min()), float(pitch.max())],
        'per_seed_final_pitch_arcsec': pitch.tolist(),
        'per_seed_final_roll_arcsec': roll.tolist(),
        'per_seed_final_yaw_arcsec': yaw.tolist(),
        'per_seed_final_yaw_abs_arcsec': yaw_abs.tolist(),
        'per_seed_final_norm_arcsec': norms.tolist(),
    }


def build_markdown(payload: dict[str, Any]) -> str:
    st = payload['statistics']
    roll_idx = 1
    pitch_idx = 0
    yaw_idx = 2

    def pct(x: float) -> float:
        return 100.0 * x

    lines = [
        '# PSINS 双轴 24-state + pure-SCD 在 0.4× 噪声口径下的 MC50 重复性实验（2026-04-13）',
        '',
        '- 对象：旧 Chapter-4 双轴路径 `build_rot_paras()` 下的 **24-state + pure-SCD**',
        '- Monte Carlo seeds: `0..49`',
        f'- phi guess (deg): `{PHI_DEG.tolist()}`；wvn: `{WVN.tolist()}`',
        '- truth 侧保留 `dKg/dKa = 30 ppm` 对角注入',
        '- truth 噪声口径：white increment + first-order GM bias',
        '- 分量顺序严格按 PSINS `q2att/qq2phi = [pitch, roll, yaw]`；下表按论文阅读习惯改写为 **横滚 / 俯仰 / 航向**。',
        '',
        '## 表：MC50 统计结果（arcsec）',
        '',
        '| 指标 | 横滚 | 俯仰 | 航向 |',
        '|---|---:|---:|---:|',
        f"| 平均误差 | {st['mean_signed_arcsec'][roll_idx]:.3f} | {st['mean_signed_arcsec'][pitch_idx]:.3f} | {st['mean_signed_arcsec'][yaw_idx]:.3f} |",
        f"| 标准差（1σ） | {st['std_signed_arcsec_1sigma'][roll_idx]:.3f} | {st['std_signed_arcsec_1sigma'][pitch_idx]:.3f} | {st['std_signed_arcsec_1sigma'][yaw_idx]:.3f} |",
        f"| RMS | {st['rms_arcsec'][roll_idx]:.3f} | {st['rms_arcsec'][pitch_idx]:.3f} | {st['rms_arcsec'][yaw_idx]:.3f} |",
        f"| 平均绝对误差 | {st['mean_abs_arcsec'][roll_idx]:.3f} | {st['mean_abs_arcsec'][pitch_idx]:.3f} | {st['mean_abs_arcsec'][yaw_idx]:.3f} |",
        f"| 中位绝对误差 | {st['median_abs_arcsec'][roll_idx]:.3f} | {st['median_abs_arcsec'][pitch_idx]:.3f} | {st['median_abs_arcsec'][yaw_idx]:.3f} |",
        f"| |误差|<20\" 比例 / % | {pct(st['within20_abs_rate'][roll_idx]):.1f} | {pct(st['within20_abs_rate'][pitch_idx]):.1f} | {pct(st['within20_abs_rate'][yaw_idx]):.1f} |",
        '',
        '## Norm / yaw 汇总',
        '',
        f"- norm mean = `{st['norm_mean_arcsec']:.3f}\"`",
        f"- norm median = `{st['norm_median_arcsec']:.3f}\"`",
        f"- norm 1σ = `{st['norm_1sigma_arcsec']:.3f}\"`",
        f"- yaw abs mean = `{st['yaw_abs_mean_arcsec']:.3f}\"`",
        f"- yaw abs median = `{st['yaw_abs_median_arcsec']:.3f}\"`",
        f"- pitch signed range = `[{st['pitch_signed_range_arcsec'][0]:.3f}, {st['pitch_signed_range_arcsec'][1]:.3f}]\"`",
        '',
        '## 文件',
        '',
        f"- script: `{payload['files']['script']}`",
        f"- json: `{payload['files']['json']}`",
        f"- markdown: `{payload['files']['md']}`",
    ]
    return '\n'.join(lines) + '\n'


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description='Run MC50 repeatability for dual-axis pure-SCD @ noise x0.4.')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    h24 = load_h24()
    acc18 = h24.load_acc18()
    workers = max(1, min(args.workers, len(SEEDS)))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(run_seed_worker, SEEDS))
    elapsed_s = time.time() - t0
    rows.sort(key=lambda row: row['seed'])

    payload = {
        'task': 'dualpath_pure_scd_custom_noise_0p4_mc50_2026_04_13',
        'path_note': 'old Chapter-4 dual-axis path build_rot_paras()',
        'method': 'dual-axis 24-state + pure-SCD',
        'component_order_note': 'Signed terminal attitude error components follow PSINS q2att order [pitch, roll, yaw].',
        'noise_config': build_noise_config(acc18.glv),
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
            'workers': workers,
            'elapsed_s': elapsed_s,
            'scd_alpha': PURE_SCD_ALPHA,
            'scd_transition_duration_s': PURE_SCD_TRANSITION_DURATION_S,
            'scd_apply_after_release_iter': PURE_SCD_APPLY_AFTER_RELEASE_ITER,
        },
        'statistics': compute_statistics(rows),
        'per_seed': rows,
        'files': {
            'script': str(Path(__file__)),
            'json': str(OUT_JSON),
            'md': str(OUT_MD),
        },
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    OUT_MD.write_text(build_markdown(payload), encoding='utf-8')

    print(json.dumps({
        'json': str(OUT_JSON),
        'md': str(OUT_MD),
        'elapsed_s': elapsed_s,
        'yaw_abs_mean_arcsec': payload['statistics']['yaw_abs_mean_arcsec'],
        'pitch_mean_abs_arcsec': payload['statistics']['mean_abs_arcsec'][0],
        'roll_mean_abs_arcsec': payload['statistics']['mean_abs_arcsec'][1],
        'norm_mean_arcsec': payload['statistics']['norm_mean_arcsec'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
