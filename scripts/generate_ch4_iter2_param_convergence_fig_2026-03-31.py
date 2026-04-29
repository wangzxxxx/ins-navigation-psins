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
OUT_PNG = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'fig_ch4_iter2_param_convergence_pub_2026-03-31.png'
OUT_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'ch4_iter2_param_convergence_2026-03-31.json'


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = load_module('align_h24_iter2_param_20260331', SCRIPT_PATH)
acc18 = mod.load_acc18()
base12 = mod.load_base12()
glv = acc18.glv


def trace_to_iter2(cfg: mod.Hybrid24Config, seed: int = 0, sample_stride_raw: int = 200):
    np.random.seed(seed)
    ts = 0.01
    att0 = np.array([0.0, 0.0, 0.0])
    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    rot_paras = base12.build_rot_paras()
    att_truth = acc18.attrottt(att0, rot_paras, ts)
    imu, _ = acc18.avp2imu(att_truth, pos0)
    imuerr = base12.build_imuerr()
    imu_corr = acc18.imuadderr(imu, imuerr)

    phi = np.array([0.1, 0.1, 0.5]) * glv.deg
    att0_guess = acc18.q2att(base12.qaddphi(acc18.a2qua(att0), phi))
    wvn = np.array([0.01, 0.01, 0.01])

    nn = 2
    nts = nn * ts
    qnb_seed = acc18.a2qua(att0_guess)
    length = (len(imu_corr) // nn) * nn
    imu_corr = imu_corr[:length]

    eth = acc18.Earth(pos0)
    cnn = acc18.rv2m(-eth.wnie * nts / 2)
    rot_gate_rad = cfg.rot_gate_dps * glv.deg
    trace = None

    for iteration in range(1, 3):
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
        cur = []

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
            if scale_active:
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

            if iteration == 2 and (k % sample_stride_raw == 0 or k + nn >= length):
                cur.append({
                    'time_s': float((k + nn) * ts),
                    'kg_y_ppm': float(kf['xk'][19] / glv.ppm),
                    'ka_z_ppm': float(kf['xk'][23] / glv.ppm),
                    'yaw_abs_arcsec': float(abs(acc18.qq2phi(qnbi, acc18.a2qua(att_truth[min(k + nn - 1, len(att_truth) - 1), 0:3]))[2] / glv.sec)),
                    'high_rot': bool(high_rot),
                })
        if iteration == 2:
            trace = cur

        if iteration < 2:
            qnb_seed = qnbi.copy() if cfg.carry_att_seed else qnb_seed
            imu_corr[:, 0:3] -= cfg.wash_scale * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= cfg.wash_scale * kf['xk'][9:12] * ts
            if scale_active and cfg.scale_wash_scale > 0.0:
                imu_corr = mod.apply_scale_wash(imu_corr, kf['xk'][18:21], kf['xk'][21:24], cfg.scale_wash_scale)

    return trace


def spans_from_gate(times: list[float], high_rot: list[bool]):
    spans = []
    start = None
    prev_t = None
    for t, flag in zip(times, high_rot):
        if flag and start is None:
            start = t
        if not flag and start is not None:
            spans.append((start, prev_t if prev_t is not None else t))
            start = None
        prev_t = t
    if start is not None and prev_t is not None:
        spans.append((start, prev_t))
    return spans


def main():
    plain_cfg = mod.Hybrid24Config(name='plain24', label='plain24', seeds=[0], staged_release=False)
    staged_cfg = mod.Hybrid24Config(name='staged24', label='staged24', seeds=[0], staged_release=True)

    plain = trace_to_iter2(plain_cfg, seed=0)
    staged = trace_to_iter2(staged_cfg, seed=0)

    tp = [row['time_s'] for row in plain]
    ts = [row['time_s'] for row in staged]
    gate_spans = spans_from_gate(ts, [row['high_rot'] for row in staged])

    fig, axes = plt.subplots(3, 1, figsize=(12.4, 8.2), sharex=True)
    for ax in axes:
        for a, b in gate_spans:
            ax.axvspan(a, b, color='#d9d9d9', alpha=0.25)
        ax.grid(True, linestyle='--', alpha=0.35)

    axes[0].plot(tp, [row['kg_y_ppm'] for row in plain], label='plain24 iter2', linewidth=2.0)
    axes[0].plot(ts, [row['kg_y_ppm'] for row in staged], label='staged24 iter2', linewidth=2.0)
    axes[0].axhline(30.0, color='r', linestyle='--', linewidth=1.0, label='true value = 30 ppm')
    axes[0].set_ylabel('estimated kg_y / ppm')
    axes[0].set_title('Representative scale-state convergence in iter=2')
    axes[0].legend(frameon=False, loc='upper left')

    axes[1].plot(tp, [row['ka_z_ppm'] for row in plain], label='plain24 iter2', linewidth=2.0)
    axes[1].plot(ts, [row['ka_z_ppm'] for row in staged], label='staged24 iter2', linewidth=2.0)
    axes[1].axhline(30.0, color='r', linestyle='--', linewidth=1.0, label='true value = 30 ppm')
    axes[1].set_ylabel('estimated ka_z / ppm')
    axes[1].legend(frameon=False, loc='upper left')

    axes[2].step(ts, [1.0 if row['high_rot'] else 0.0 for row in staged], where='mid', linewidth=1.8)
    axes[2].set_ylim(-0.1, 1.1)
    axes[2].set_yticks([0, 1])
    axes[2].set_yticklabels(['gate off', 'gate on'])
    axes[2].set_xlabel('alignment time in iter=2 / s')
    axes[2].set_ylabel('staged gate')

    fig.suptitle('Time-domain convergence of representative released parameters (iter=2, release + gate)')
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=240, bbox_inches='tight')
    plt.close(fig)

    payload = {
        'plain24_iter2_seed0': plain,
        'staged24_iter2_seed0': staged,
        'gate_spans': gate_spans,
        'figure': str(OUT_PNG),
        'note': 'Representative parameter-time convergence in iter=2. kg_y and ka_z are chosen because they are among the strongest activated scale states under the current trajectory.',
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'figure': str(OUT_PNG), 'gate_spans_count': len(gate_spans)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
