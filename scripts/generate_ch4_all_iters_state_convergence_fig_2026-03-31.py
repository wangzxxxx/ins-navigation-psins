#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
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
OUT_PNG = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'fig_ch4_all_iters_state_convergence_pub_2026-03-31.png'
OUT_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'ch4_all_iters_state_convergence_2026-03-31.json'
MAX_WORKERS = min(6, os.cpu_count() or 1)


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def trace_seed_method(seed: int, staged_release: bool, sample_stride_raw: int = 500):
    mod = load_module(f'align_h24_alliters_{seed}_{int(staged_release)}', SCRIPT_PATH)
    acc18 = mod.load_acc18()
    base12 = mod.load_base12()
    glv = acc18.glv

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
    cfg = mod.Hybrid24Config(name='cfg', label='cfg', seeds=[seed], staged_release=staged_release)

    nn = 2
    nts = nn * ts
    qnb_seed = acc18.a2qua(att0_guess)
    length = (len(imu_corr) // nn) * nn
    imu_corr = imu_corr[:length]

    eth = acc18.Earth(pos0)
    cnn = acc18.rv2m(-eth.wnie * nts / 2)
    rot_gate_rad = cfg.rot_gate_dps * glv.deg
    traces = {it: [] for it in range(1, 6)}

    for iteration in range(1, 6):
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

            if k % sample_stride_raw == 0 or k + nn >= length:
                truth = att_truth[min(k + nn - 1, len(att_truth) - 1), 0:3]
                err = acc18.qq2phi(qnbi, acc18.a2qua(truth)) / glv.sec
                cur.append({
                    'time_s': float((k + nn) * ts),
                    'roll_abs_arcsec': float(abs(err[0])),
                    'pitch_signed_arcsec': float(err[1]),
                    'yaw_abs_arcsec': float(abs(err[2])),
                    'high_rot': bool(high_rot) if iteration >= cfg.release_iter and cfg.staged_release else False,
                })
        traces[iteration] = cur

        if iteration < 5:
            qnb_seed = qnbi.copy() if cfg.carry_att_seed else qnb_seed
            imu_corr[:, 0:3] -= cfg.wash_scale * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= cfg.wash_scale * kf['xk'][9:12] * ts
            if scale_active and cfg.scale_wash_scale > 0.0:
                imu_corr = mod.apply_scale_wash(imu_corr, kf['xk'][18:21], kf['xk'][21:24], cfg.scale_wash_scale)

    return traces


def average_method(staged_release: bool, seeds: list[int]):
    args = [(s, staged_release) for s in seeds]
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(seeds))) as ex:
        traces_list = list(ex.map(_trace_seed_method_star, args))
    avg = {}
    for it in range(1, 6):
        n = len(traces_list[0][it])
        rows = []
        for idx in range(n):
            block = [tr[it][idx] for tr in traces_list]
            rows.append({
                'time_s': block[0]['time_s'],
                'roll_abs_arcsec': float(np.mean([r['roll_abs_arcsec'] for r in block])),
                'pitch_signed_arcsec': float(np.mean([r['pitch_signed_arcsec'] for r in block])),
                'yaw_abs_arcsec': float(np.mean([r['yaw_abs_arcsec'] for r in block])),
                'high_rot': bool(block[0]['high_rot']),
            })
        avg[it] = rows
    return avg


def _trace_seed_method_star(args):
    seed, staged_release = args
    return trace_seed_method(seed, staged_release)


def gate_spans(rows):
    spans = []
    start = None
    prev = None
    for row in rows:
        t = row['time_s']
        flag = row['high_rot']
        if flag and start is None:
            start = t
        if not flag and start is not None:
            spans.append((start, prev if prev is not None else t))
            start = None
        prev = t
    if start is not None and prev is not None:
        spans.append((start, prev))
    return spans


def main():
    seeds = [0, 1, 2, 3, 4]
    plain = average_method(False, seeds)
    staged = average_method(True, seeds)

    fig, axes = plt.subplots(3, 5, figsize=(18, 9), sharex=False)
    row_names = ['roll_abs_arcsec', 'pitch_signed_arcsec', 'yaw_abs_arcsec']
    ylabels = ['roll |error| / arcsec', 'pitch error / arcsec', 'yaw |error| / arcsec']

    for col, it in enumerate(range(1, 6)):
        gate = gate_spans(staged[it])
        for row_idx, (key, ylabel) in enumerate(zip(row_names, ylabels)):
            ax = axes[row_idx, col]
            for a, b in gate:
                ax.axvspan(a, b, color='#d9d9d9', alpha=0.22)
            ax.plot([r['time_s'] for r in plain[it]], [r[key] for r in plain[it]], linewidth=1.8, label='plain24')
            ax.plot([r['time_s'] for r in staged[it]], [r[key] for r in staged[it]], linewidth=1.8, label='staged24')
            ax.grid(True, linestyle='--', alpha=0.3)
            if row_idx == 0:
                title = f'iter={it}'
                if it == 1:
                    title += ' (freeze)'
                elif it >= 2:
                    title += ' (release+gate)'
                ax.set_title(title)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row_idx == 2:
                ax.set_xlabel('time / s')
            if row_idx == 1:
                ax.axhline(0.0, color='k', linestyle='--', linewidth=0.8)
            if row_idx == 0 and col == 4:
                ax.legend(frameon=False, loc='upper right')

    fig.suptitle('State-convergence comparison of plain24 vs staged24 for every outer iteration (5-seed mean)')
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220, bbox_inches='tight')
    plt.close(fig)

    payload = {
        'plain24_mean': plain,
        'staged24_mean': staged,
        'figure': str(OUT_PNG),
        'note': 'Each column is one outer iteration; gray spans denote gate-on high-rotation intervals in the staged method. Curves are 5-seed means.',
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'figure': str(OUT_PNG)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
