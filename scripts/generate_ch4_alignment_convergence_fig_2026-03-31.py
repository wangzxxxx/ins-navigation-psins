#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
PSINS_ROOT = WORKSPACE / 'tmp_psins_py'
if str(PSINS_ROOT) not in sys.path:
    sys.path.insert(0, str(PSINS_ROOT))

SCRIPT_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_hybrid24_staged_py_2026-03-30.py'
OUT_PNG = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'fig_ch4_alignment_convergence_pub_2026-03-31.png'
OUT_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'ch4_alignment_convergence_2026-03-31.json'


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = load_module('align_h24_convergence_20260331', SCRIPT_PATH)
acc18 = mod.load_acc18()
base12 = mod.load_base12()
glv = acc18.glv


def trace_final_iteration(cfg: mod.Hybrid24Config, seed: int = 0, sample_stride_raw: int = 100):
    np.random.seed(seed)
    ts = 0.01
    att0 = np.array([0.0, 0.0, 0.0])
    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    rot_paras = base12.build_rot_paras()
    att_truth = acc18.attrottt(att0, rot_paras, ts)
    imu, _ = acc18.avp2imu(att_truth, pos0)
    imuerr = base12.build_imuerr()
    imu_noisy = acc18.imuadderr(imu, imuerr)

    phi = np.array([0.1, 0.1, 0.5]) * glv.deg
    att0_guess = acc18.q2att(base12.qaddphi(acc18.a2qua(att0), phi))
    wvn = np.array([0.01, 0.01, 0.01])

    imu_corr = imu_noisy.copy()
    nn = 2
    ts = float(imu_corr[1, -1] - imu_corr[0, -1])
    nts = nn * ts
    qnb_seed = acc18.a2qua(att0_guess)
    length = (len(imu_corr) // nn) * nn
    imu_corr = imu_corr[:length]

    eth = acc18.Earth(pos0)
    cnn = acc18.rv2m(-eth.wnie * nts / 2)
    rot_gate_rad = cfg.rot_gate_dps * glv.deg

    final_trace = None
    final_att = None
    for iteration in range(1, cfg.max_iter + 1):
        scale_active = (not cfg.staged_release) or (iteration >= cfg.release_iter)
        kf = mod.avnkfinit_24(
            nts, pos0, phi, imuerr, wvn,
            np.array(cfg.ng_sigma_dph) * glv.dph,
            np.array(cfg.tau_g_s),
            np.array(cfg.xa_sigma_ug) * glv.ug,
            np.array(cfg.tau_a_s),
            enable_scale_states=scale_active,
        )
        vn = np.zeros(3)
        qnbi = qnb_seed.copy()
        iter_trace = []

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
                high_rot = np.max(np.abs(phim / nts)) > rot_gate_rad
                if high_rot:
                    phi_k[0:3, 18:21] = -cnb @ np.diag(phim[0:3])
                    phi_k[3:6, 21:24] = cnb @ np.diag(dvbm[0:3])
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

            if k % sample_stride_raw == 0 or k + nn >= length:
                att_est = acc18.q2att(qnbi)
                truth = att_truth[min(k + nn - 1, len(att_truth) - 1), 0:3]
                err = acc18.qq2phi(acc18.a2qua(att_est), acc18.a2qua(truth)) / glv.sec
                iter_trace.append({
                    'time_s': float((k + nn) * ts),
                    'roll_abs_arcsec': float(abs(err[0])),
                    'pitch_signed_arcsec': float(err[1]),
                    'pitch_abs_arcsec': float(abs(err[1])),
                    'yaw_abs_arcsec': float(abs(err[2])),
                    'norm_arcsec': float(np.linalg.norm(err)),
                })

        final_att = acc18.q2att(qnbi)
        if iteration == cfg.max_iter:
            final_trace = iter_trace

        if iteration < cfg.max_iter:
            if cfg.carry_att_seed:
                qnb_seed = qnbi.copy()
            imu_corr[:, 0:3] -= cfg.wash_scale * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= cfg.wash_scale * kf['xk'][9:12] * ts
            if scale_active and cfg.scale_wash_scale > 0.0:
                imu_corr = mod.apply_scale_wash(imu_corr, kf['xk'][18:21], kf['xk'][21:24], cfg.scale_wash_scale)

    return {
        'seed': seed,
        'trace': final_trace,
        'final_att_arcsec': [
            float(x) for x in (
                acc18.qq2phi(acc18.a2qua(final_att), acc18.a2qua(att_truth[-1, 0:3])) / glv.sec
            )
        ],
    }


def main():
    plain_cfg = mod.Hybrid24Config(name='plain24_iter5', label='plain24 iter=5', seeds=[0], staged_release=False)
    staged_cfg = mod.Hybrid24Config(name='staged24_iter5', label='staged24 iter=5', seeds=[0], staged_release=True)

    plain = trace_final_iteration(plain_cfg, seed=0)
    staged = trace_final_iteration(staged_cfg, seed=0)

    tp = np.array([row['time_s'] for row in plain['trace']])
    ts = np.array([row['time_s'] for row in staged['trace']])

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.0))

    axes[0, 0].plot(tp, [row['roll_abs_arcsec'] for row in plain['trace']], label='plain24', linewidth=2.0)
    axes[0, 0].plot(ts, [row['roll_abs_arcsec'] for row in staged['trace']], label='staged24', linewidth=2.0)
    axes[0, 0].set_title('Roll absolute error')
    axes[0, 0].set_xlabel('alignment time / s')
    axes[0, 0].set_ylabel('arcsec')
    axes[0, 0].grid(True, linestyle='--', alpha=0.35)
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(tp, [row['pitch_signed_arcsec'] for row in plain['trace']], label='plain24', linewidth=2.0)
    axes[0, 1].plot(ts, [row['pitch_signed_arcsec'] for row in staged['trace']], label='staged24', linewidth=2.0)
    axes[0, 1].axhline(0.0, color='k', linestyle='--', linewidth=1.0)
    axes[0, 1].set_title('Pitch signed error')
    axes[0, 1].set_xlabel('alignment time / s')
    axes[0, 1].set_ylabel('arcsec')
    axes[0, 1].grid(True, linestyle='--', alpha=0.35)
    axes[0, 1].legend(frameon=False)

    axes[1, 0].plot(tp, [row['yaw_abs_arcsec'] for row in plain['trace']], label='plain24', linewidth=2.0)
    axes[1, 0].plot(ts, [row['yaw_abs_arcsec'] for row in staged['trace']], label='staged24', linewidth=2.0)
    axes[1, 0].axhline(20.0, color='r', linestyle='--', linewidth=1.0, label='20 arcsec')
    axes[1, 0].set_title('Yaw absolute error')
    axes[1, 0].set_xlabel('alignment time / s')
    axes[1, 0].set_ylabel('arcsec')
    axes[1, 0].grid(True, linestyle='--', alpha=0.35)
    axes[1, 0].legend(frameon=False)

    axes[1, 1].plot(tp, [row['norm_arcsec'] for row in plain['trace']], label='plain24', linewidth=2.0)
    axes[1, 1].plot(ts, [row['norm_arcsec'] for row in staged['trace']], label='staged24', linewidth=2.0)
    axes[1, 1].axhline(20.0, color='r', linestyle='--', linewidth=1.0, label='20 arcsec')
    axes[1, 1].set_title('Attitude error norm')
    axes[1, 1].set_xlabel('alignment time / s')
    axes[1, 1].set_ylabel('arcsec')
    axes[1, 1].grid(True, linestyle='--', alpha=0.35)
    axes[1, 1].legend(frameon=False)

    fig.suptitle('Representative filter convergence of alignment accuracy (seed0, final iter=5)')
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=240, bbox_inches='tight')
    plt.close(fig)

    payload = {
        'plain24_seed0_final': plain,
        'staged24_seed0_final': staged,
        'figure': str(OUT_PNG),
        'note': 'Representative seed0 convergence traces from the final outer iteration (iter=5).',
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'plain_final': plain['final_att_arcsec'],
        'staged_final': staged['final_att_arcsec'],
        'figure': str(OUT_PNG),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
