#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import math
import sys
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
PSINS_ROOT = WORKSPACE / 'tmp_psins_py'
if str(PSINS_ROOT) not in sys.path:
    sys.path.insert(0, str(PSINS_ROOT))

BASE12_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
H24_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_hybrid24_staged_py_2026-03-30.py'
PURE_SCD_PATH = WORKSPACE / 'scripts' / 'compare_ch4_pure_scd_vs_freeze_2026-04-03.py'

OUT_DIR = WORKSPACE / 'tmp' / 'psins_repeatability'
OUT_PNG = OUT_DIR / 'fig_dual_noise_phi_t_convergence_2026-04-13.png'
OUT_SVG = OUT_DIR / 'fig_dual_noise_phi_t_convergence_2026-04-13.svg'
OUT_JSON = OUT_DIR / 'fig_dual_noise_phi_t_convergence_2026-04-13.json'

TS = 0.01
TRACE_DT_S = 0.2
WVN = np.array([0.01, 0.01, 0.01], dtype=float)
PHI_DEG = np.array([0.1, 0.1, 0.5], dtype=float)
ROT_GATE_DPS = 5.0
MAX_ITER = 5
WASH_SCALE = 0.5
SCALE_WASH_SCALE = 0.5
PURE_SCD_ALPHA = 0.995
PURE_SCD_TRANSITION_DURATION_S = 2.0
PURE_SCD_APPLY_AFTER_RELEASE_ITER = 1
SEED = 0

CASES = {
    '0.4×': {
        'tag': '0p4',
        'color': '#C44E52',
        'arw_dps_sqrt_h': 0.00020,
        'vrw_ug_sqrt_hz': 0.20,
        'bi_g_dph': 0.00028,
        'bi_a_ug': 2.0,
        'tau_g_s': 300.0,
        'tau_a_s': 300.0,
    },
    '0.3×': {
        'tag': '0p3',
        'color': '#4C78A8',
        'arw_dps_sqrt_h': 0.00015,
        'vrw_ug_sqrt_hz': 0.15,
        'bi_g_dph': 0.00021,
        'bi_a_ug': 1.5,
        'tau_g_s': 300.0,
        'tau_a_s': 300.0,
    },
}
TRUTH_SCALE_PPM = 30.0

_BASE12 = None
_H24 = None
_PURE = None


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
        _BASE12 = load_module('phi_t_base12_20260413', BASE12_PATH)
    return _BASE12


def load_h24():
    global _H24
    if _H24 is None:
        _H24 = load_module('phi_t_h24_20260413', H24_PATH)
    return _H24


def load_pure():
    global _PURE
    if _PURE is None:
        _PURE = load_module('phi_t_pure_20260413', PURE_SCD_PATH)
    return _PURE


def build_filter_imuerr(glv, cfg) -> dict[str, np.ndarray]:
    return {
        'eb': np.full(3, cfg['bi_g_dph'] * glv.dph, dtype=float),
        'db': np.full(3, cfg['bi_a_ug'] * glv.ug, dtype=float),
        'web': np.full(3, cfg['arw_dps_sqrt_h'] * glv.dpsh, dtype=float),
        'wdb': np.full(3, cfg['vrw_ug_sqrt_hz'] * glv.ugpsHz, dtype=float),
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


def trace_case(case_name: str, cfg: dict[str, Any]) -> dict[str, Any]:
    base12 = load_base12()
    h24 = load_h24()
    pure = load_pure()
    acc18 = h24.load_acc18()
    glv = acc18.glv

    att0 = np.array([0.0, 0.0, 0.0], dtype=float)
    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    rot_paras = base12.build_rot_paras()
    att_truth = acc18.attrottt(att0, rot_paras, TS)
    imu_clean, _ = acc18.avp2imu(att_truth, pos0)
    duration_s = float(att_truth[-1, 3])

    filter_imuerr = build_filter_imuerr(glv, cfg)
    imu_corr = imuadderr_full_with_scale(
        imu_clean,
        TS,
        dKg=filter_imuerr['dKg'],
        dKa=filter_imuerr['dKa'],
        arw=float(cfg['arw_dps_sqrt_h'] * glv.dpsh),
        vrw=float(cfg['vrw_ug_sqrt_hz'] * glv.ugpsHz),
        bi_g=float(cfg['bi_g_dph'] * glv.dph),
        tau_g=cfg['tau_g_s'],
        bi_a=float(cfg['bi_a_ug'] * glv.ug),
        tau_a=cfg['tau_a_s'],
        seed=SEED,
    )

    phi0 = PHI_DEG * glv.deg
    qnb_seed = acc18.a2qua(base12.q2att(base12.qaddphi(acc18.a2qua(att0), phi0)))
    nn = 2
    ts = float(imu_corr[1, -1] - imu_corr[0, -1])
    nts = nn * ts
    length = (len(imu_corr) // nn) * nn
    imu_corr = imu_corr[:length]

    eth = acc18.Earth(pos0)
    cnn = acc18.rv2m(-eth.wnie * nts / 2)
    rot_gate_rad = ROT_GATE_DPS * glv.deg
    scd_cfg = pure.SCDConfig(enabled=True, alpha=PURE_SCD_ALPHA, transition_duration_s=PURE_SCD_TRANSITION_DURATION_S, apply_after_release_iter=PURE_SCD_APPLY_AFTER_RELEASE_ITER, note='hard_a995_td2_i1')

    times = []
    x_trace = []
    sigma_trace = []
    iter_bounds = []
    last_saved_global_t = -1e9

    for iteration in range(1, MAX_ITER + 1):
        kf = h24.avnkfinit_24(
            nts, pos0, phi0, filter_imuerr, WVN.copy(),
            np.array([cfg['bi_g_dph'], cfg['bi_g_dph'], cfg['bi_g_dph']]) * glv.dph,
            np.array([cfg['tau_g_s'], cfg['tau_g_s'], cfg['tau_g_s']], dtype=float),
            np.array([cfg['bi_a_ug'], cfg['bi_a_ug'], cfg['bi_a_ug']]) * glv.ug,
            np.array([cfg['tau_a_s'], cfg['tau_a_s'], cfg['tau_a_s']], dtype=float),
            enable_scale_states=True,
        )
        vn = np.zeros(3)
        qnbi = qnb_seed.copy()
        time_since_rot_stop = 0.0
        scd_applied_this_phase = False
        elapsed_s = 0.0

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
            high_rot = np.max(np.abs(phim / nts)) > rot_gate_rad
            if high_rot:
                phi_k[0:3, 18:21] = -cnb @ np.diag(phim[0:3])
                phi_k[3:6, 21:24] = cnb @ np.diag(dvbm[0:3])
                time_since_rot_stop = 0.0
                scd_applied_this_phase = False
            else:
                phi_k[0:3, 18:21] = 0.0
                phi_k[3:6, 21:24] = 0.0
                time_since_rot_stop += nts
            kf['Phikk_1'] = phi_k
            kf = acc18.kfupdate(kf, vn)

            qnbi = acc18.qdelphi(qnbi, 0.91 * kf['xk'][0:3])
            kf['xk'][0:3] *= 0.09
            vn = vn - 0.91 * kf['xk'][3:6]
            kf['xk'][3:6] *= 0.09

            if scd_cfg.enabled and iteration >= scd_cfg.apply_after_release_iter and (not high_rot):
                if (time_since_rot_stop >= scd_cfg.transition_duration_s) and (not scd_applied_this_phase):
                    kf = pure.apply_scd_once(kf, scd_cfg)
                    scd_applied_this_phase = True

            elapsed_s += nts
            current_global_t = elapsed_s + (iteration - 1) * duration_s
            if (not times) or (current_global_t - last_saved_global_t >= TRACE_DT_S - 1e-12):
                times.append(float(current_global_t))
                x_trace.append((np.array(kf['xk'][0:3]) / glv.deg).tolist())
                sigma_trace.append((np.sqrt(np.diag(kf['Pxk'])[0:3]) / glv.deg).tolist())
                last_saved_global_t = current_global_t

        iter_bounds.append(len(times))
        if iteration < MAX_ITER:
            qnb_seed = qnbi.copy()
            imu_corr[:, 0:3] -= WASH_SCALE * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= WASH_SCALE * kf['xk'][9:12] * ts
            imu_corr = h24.apply_scale_wash(imu_corr, kf['xk'][18:21], kf['xk'][21:24], SCALE_WASH_SCALE)

    return {
        'case': case_name,
        'seed': SEED,
        'duration_s_per_iter': duration_s,
        'times_s': times,
        'phi_deg': x_trace,
        'sigma_deg': sigma_trace,
        'iter_bounds': iter_bounds,
        'initial_phi_deg': PHI_DEG.tolist(),
    }


def main() -> None:
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'Noto Sans CJK JP', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    traces = {name: trace_case(name, cfg) for name, cfg in CASES.items()}

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.3), dpi=180, sharex=False)
    state_labels = ['phi_x', 'phi_y', 'phi_z']
    state_title = ['φx(t)', 'φy(t)', 'φz(t)']

    for r, (case_name, cfg) in enumerate(CASES.items()):
        trace = traces[case_name]
        t = np.array(trace['times_s'])
        x = np.array(trace['phi_deg'])
        s = np.array(trace['sigma_deg'])
        bounds = trace['iter_bounds'][:-1]
        duration_s = trace['duration_s_per_iter']
        color = cfg['color']

        for c in range(3):
            ax = axes[r, c]
            ax.plot(t, x[:, c], color=color, linewidth=1.4)
            ax.fill_between(t, x[:, c] - s[:, c], x[:, c] + s[:, c], color=color, alpha=0.16)
            for b in bounds:
                xb = t[b - 1]
                ax.axvline(xb, color='#7A7A7A', linestyle='--', linewidth=0.8, alpha=0.7)
            ax.axhline(0.0, color='#444', linestyle=':', linewidth=0.9)
            ax.set_title(f'{case_name} · {state_title[c]}', fontsize=12)
            ax.set_ylabel('deg')
            ax.set_xlabel('cumulative filter update time (s)')
            ax.grid(True, linestyle='--', alpha=0.22)
            ax.text(0.02, 0.96,
                    f'φ0 = {trace["initial_phi_deg"][c]:.3f} deg\nseed = {SEED}\nshaded = ±1σ(P)',
                    transform=ax.transAxes, va='top', ha='left', fontsize=8.8,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.78))
            total_s = duration_s * MAX_ITER
            ax.set_xlim(0, total_s)

    fig.suptitle('pure-SCD 双轴对准的内部残余失准角 φ(t) 收敛图（0.3× vs 0.4× 噪声，seed=0）', fontsize=15)
    fig.text(0.5, 0.02,
             '说明：这里展示的是滤波器内部的残余失准角状态 xk[0:3]=phi(t) 及其协方差开方 ±1σ(P)，\n'
             '不是最终用真值重新计算的 roll / pitch / yaw 终值误差；虚线表示 outer iteration 边界。',
             ha='center', va='bottom', fontsize=10.5, color='#334E68')
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.95])
    fig.savefig(OUT_PNG, format='png', bbox_inches='tight')
    fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
    OUT_JSON.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'png': str(OUT_PNG), 'svg': str(OUT_SVG), 'json': str(OUT_JSON)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
