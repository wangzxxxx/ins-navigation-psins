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
OUT_PNG = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'fig_ch4_iter1_iter2_transition_pub_2026-03-31.png'
OUT_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'ch4_iter1_iter2_transition_2026-03-31.json'


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = load_module('align_h24_iter12_20260331', SCRIPT_PATH)
acc18 = mod.load_acc18()
base12 = mod.load_base12()
glv = acc18.glv


def trace_iter1_iter2(cfg: mod.Hybrid24Config, seed: int, sample_stride_raw: int = 200):
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
    traces = []

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

            if k % sample_stride_raw == 0 or k + nn >= length:
                truth = att_truth[min(k + nn - 1, len(att_truth) - 1), 0:3]
                err = acc18.qq2phi(qnbi, acc18.a2qua(truth)) / glv.sec
                t = float((iteration - 1) * (length * ts) + (k + nn) * ts)
                traces.append({
                    'time_s': t,
                    'iteration': iteration,
                    'roll_abs_arcsec': float(abs(err[0])),
                    'pitch_signed_arcsec': float(err[1]),
                    'yaw_abs_arcsec': float(abs(err[2])),
                    'norm_arcsec': float(np.linalg.norm(err)),
                    'high_rot': bool(high_rot) if iteration >= cfg.release_iter else False,
                })

        if iteration < 2:
            qnb_seed = qnbi.copy() if cfg.carry_att_seed else qnb_seed
            imu_corr[:, 0:3] -= cfg.wash_scale * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= cfg.wash_scale * kf['xk'][9:12] * ts
            if scale_active and cfg.scale_wash_scale > 0.0:
                imu_corr = mod.apply_scale_wash(imu_corr, kf['xk'][18:21], kf['xk'][21:24], cfg.scale_wash_scale)

    return traces


def average_traces(cfg: mod.Hybrid24Config, seeds: list[int]):
    traces = [trace_iter1_iter2(cfg, s) for s in seeds]
    n = len(traces[0])
    out = []
    for i in range(n):
        rows = [tr[i] for tr in traces]
        out.append({
            'time_s': rows[0]['time_s'],
            'iteration': rows[0]['iteration'],
            'roll_abs_arcsec': float(np.mean([r['roll_abs_arcsec'] for r in rows])),
            'pitch_signed_arcsec': float(np.mean([r['pitch_signed_arcsec'] for r in rows])),
            'yaw_abs_arcsec': float(np.mean([r['yaw_abs_arcsec'] for r in rows])),
            'norm_arcsec': float(np.mean([r['norm_arcsec'] for r in rows])),
            'high_rot': bool(rows[0]['high_rot']),
        })
    return out


def spans_from_mask(times: list[float], mask: list[bool]):
    spans = []
    start = None
    prev_t = None
    for t, flag in zip(times, mask):
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
    seeds = [0, 1, 2, 3, 4]
    plain_cfg = mod.Hybrid24Config(name='plain24', label='plain24', seeds=seeds, staged_release=False)
    staged_cfg = mod.Hybrid24Config(name='staged24', label='staged24', seeds=seeds, staged_release=True)

    plain = average_traces(plain_cfg, seeds)
    staged = average_traces(staged_cfg, seeds)

    tp = [row['time_s'] for row in plain]
    ts = [row['time_s'] for row in staged]
    gate_spans = spans_from_mask(ts, [row['high_rot'] for row in staged])
    iter_boundary = max(row['time_s'] for row in plain if row['iteration'] == 1)

    fig, axes = plt.subplots(4, 1, figsize=(12.6, 9.0), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1, 1, 0.5]})
    for ax in axes:
        ax.axvspan(0.0, iter_boundary, color='#d9d9d9', alpha=0.18)
        for a, b in gate_spans:
            ax.axvspan(a, b, color='#bdbdbd', alpha=0.28)
        ax.axvline(iter_boundary, color='gray', linestyle='--', linewidth=1.0)
        ax.grid(True, linestyle='--', alpha=0.35)

    axes[0].plot(tp, [row['roll_abs_arcsec'] for row in plain], linewidth=2.0, label='plain24')
    axes[0].plot(ts, [row['roll_abs_arcsec'] for row in staged], linewidth=2.0, label='staged24')
    axes[0].set_ylabel('roll |error| / arcsec')
    axes[0].legend(frameon=False, loc='upper right')
    axes[0].set_title('Mean attitude-state evolution across iter1→iter2 (5-seed average)')

    axes[1].plot(tp, [row['pitch_signed_arcsec'] for row in plain], linewidth=2.0, label='plain24')
    axes[1].plot(ts, [row['pitch_signed_arcsec'] for row in staged], linewidth=2.0, label='staged24')
    axes[1].axhline(0.0, color='k', linestyle='--', linewidth=1.0)
    axes[1].set_ylabel('pitch error / arcsec')

    axes[2].plot(tp, [row['yaw_abs_arcsec'] for row in plain], linewidth=2.0, label='plain24')
    axes[2].plot(ts, [row['yaw_abs_arcsec'] for row in staged], linewidth=2.0, label='staged24')
    axes[2].set_ylabel('yaw |error| / arcsec')

    gate_mask = [1.0 if row['high_rot'] else 0.0 for row in staged]
    axes[3].step(ts, gate_mask, where='mid', linewidth=1.7)
    axes[3].set_ylim(-0.1, 1.1)
    axes[3].set_yticks([0, 1])
    axes[3].set_yticklabels(['off', 'on'])
    axes[3].set_ylabel('gate')
    axes[3].set_xlabel('concatenated alignment time across iter1→iter2 / s')

    ymax = max(max([row['yaw_abs_arcsec'] for row in plain]), max([row['yaw_abs_arcsec'] for row in staged]))
    axes[2].text(iter_boundary / 2, ymax * 1.03, 'iter1: freeze (staged)', ha='center', va='bottom', fontsize=9)
    axes[2].text(iter_boundary + (tp[-1] - iter_boundary) / 2, ymax * 1.03, 'iter2: release + gate', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=240, bbox_inches='tight')
    plt.close(fig)

    payload = {
        'plain24_mean_iter1_iter2': plain,
        'staged24_mean_iter1_iter2': staged,
        'iter_boundary_s': iter_boundary,
        'gate_spans': gate_spans,
        'figure': str(OUT_PNG),
        'note': 'This figure concatenates iter1 and iter2 on a single time axis to show freeze→release transition on the key attitude states. Gray background before the boundary denotes staged freeze in iter1; darker gray spans in iter2 denote gate-on high-rotation intervals.',
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'figure': str(OUT_PNG), 'iter_boundary_s': iter_boundary, 'gate_span_count': len(gate_spans)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
