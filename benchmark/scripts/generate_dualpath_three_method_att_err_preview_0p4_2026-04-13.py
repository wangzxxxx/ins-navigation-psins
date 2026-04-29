#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

WORKSPACE = Path('/root/.openclaw/workspace')
BASE_SCRIPT = WORKSPACE / 'psins_method_bench' / 'scripts' / 'generate_dualpath_three_method_att_err_preview_2026-04-09.py'
OUT_DIR = WORKSPACE / 'tmp' / 'psins_dualpath_three_method_att_err_preview_0p4_2026-04-13'
OUT_JSON = OUT_DIR / 'summary.json'
OUT_COMBINED = OUT_DIR / 'dualpath_three_method_att_err_preview_0p4_2026-04-13.png'
OUT_SVG_DIR = OUT_DIR / 'svg_panels'
OUT_PNG_DIR = OUT_DIR / 'png_panels'

ARW_DPS_SQRT_H = 0.00020
VRW_UG_SQRT_HZ = 0.20
BI_G_DPH = 0.00028
BI_A_UG = 2.0
TAU_G_S = 300.0
TAU_A_S = 300.0
TRUTH_SCALE_PPM = 30.0
SEED = 0


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


base = load_module('dualpath_atterr_preview_base_0p4_20260413', BASE_SCRIPT)


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


def build_shared_dual_dataset_custom_noise_0p4():
    base12 = base.bp.load_base12()
    h24 = base.bp.load_h24()
    acc18 = h24.load_acc18()
    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    att0 = np.array([0.0, 0.0, 0.0])
    rot_paras = base12.build_rot_paras()
    att_truth = acc18.attrottt(att0, rot_paras, base.bp.TS)
    imu, _ = acc18.avp2imu(att_truth, pos0)
    imuerr = build_filter_imuerr(acc18.glv)
    imu_noisy = imuadderr_full_with_scale(
        imu,
        base.bp.TS,
        dKg=imuerr['dKg'],
        dKa=imuerr['dKa'],
        arw=float(ARW_DPS_SQRT_H * acc18.glv.dpsh),
        vrw=float(VRW_UG_SQRT_HZ * acc18.glv.ugpsHz),
        bi_g=float(BI_G_DPH * acc18.glv.dph),
        tau_g=TAU_G_S,
        bi_a=float(BI_A_UG * acc18.glv.ug),
        tau_a=TAU_A_S,
        seed=SEED,
    )
    phi = np.array([0.1, 0.1, 0.5]) * acc18.glv.deg
    att0_guess = acc18.q2att(base12.qaddphi(acc18.a2qua(att0), phi))
    return {
        'pos0': pos0,
        'att0_deg': [0.0, 0.0, 0.0],
        'rot_paras': rot_paras,
        'duration_s': float(att_truth[-1, 3]),
        'truth_att': att_truth[-1, 0:3].copy(),
        'imu_noisy': imu_noisy,
        'imuerr': imuerr,
        'phi': phi,
        'att0_guess': att0_guess,
    }


def combine_pngs(paths: list[Path], out_path: Path):
    images = [Image.open(p).convert('RGB') for p in paths]
    pad = 24
    title_h = 56
    width = max(img.width for img in images) + 2 * pad
    height = title_h + sum(img.height for img in images) + pad * (len(images) + 1)
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    title = '0.4× 噪声口径下三方法姿态对准误差收敛对比图（seed=0）'
    draw.text((pad, 18), title, fill=(24, 36, 54))
    y = title_h
    for img in images:
        x = (width - img.width) // 2
        canvas.paste(img, (x, y))
        y += img.height + pad
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, format='PNG', optimize=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_SVG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PNG_DIR.mkdir(parents=True, exist_ok=True)

    base.OUT_DIR = OUT_DIR
    base.OUT_JSON = OUT_JSON
    base.OUTER_ITERS = 5
    base.bp.build_shared_dual_dataset = build_shared_dual_dataset_custom_noise_0p4

    shared = base.bp.build_shared_dual_dataset()
    groups = [base.run_scale18_att_err(shared), base.run_plain24_att_err(shared), base.run_purescd_att_err(shared)]

    axis_specs = [
        ('pitch_error_arcsec', 'att_err_x'),
        ('roll_error_arcsec', 'att_err_y'),
        ('yaw_error_arcsec', 'att_err_z'),
    ]
    summary = {
        'task': 'dualpath_three_method_att_err_preview_0p4_2026_04_13',
        'noise': '0.4x custom noise',
        'seed': SEED,
        'path_note': 'old Chapter-4 dual-axis rotation strategy build_rot_paras()',
        'plots': [],
        'combined_png': str(OUT_COMBINED),
    }

    png_paths = []
    for axis_title, axis_key in axis_specs:
        series_list = [{'group_key': g['group_key'], 'label': g['label'], 'x': g['x'], 'y': g[axis_key], 'iter_bounds_s': g['iter_bounds_s']} for g in groups]
        svg_path = OUT_SVG_DIR / f'{axis_key}_preview.svg'
        png_path = OUT_PNG_DIR / f'{axis_key}_preview.png'
        base.render_axis_preview(svg_path, png_path, axis_title, series_list)
        png_paths.append(png_path)
        summary['plots'].append({'axis': axis_key, 'title': axis_title, 'svg': str(svg_path), 'png': str(png_path)})

    combine_pngs(png_paths, OUT_COMBINED)
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
