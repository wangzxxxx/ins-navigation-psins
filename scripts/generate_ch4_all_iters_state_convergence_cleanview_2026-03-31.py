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
RESULT_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'alignvn_dar_hybrid24_staged_result_2026-03-30.json'
OUT_PNG = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'fig_ch4_all_iters_state_convergence_cleanview_pub_2026-03-31.png'
OUT_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'ch4_all_iters_state_convergence_cleanview_2026-03-31.json'


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = load_module('align_h24_cleanview_20260331', SCRIPT_PATH)
acc18 = mod.load_acc18()
base12 = mod.load_base12()
glv = acc18.glv


def build_setup(seed: int = 0):
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
    return {
        'ts': ts,
        'att0': att0,
        'pos0': pos0,
        'att_truth': att_truth,
        'imuerr': imuerr,
        'imu_noisy': imu_noisy,
        'phi': phi,
        'att0_guess': att0_guess,
        'wvn': wvn,
    }


def run_one_iteration(imu_input: np.ndarray, qnb_init, pos0: np.ndarray, phi0: np.ndarray,
                      imuerr: dict, wvn: np.ndarray, scale_active: bool, sample_stride_raw: int = 500):
    nn = 2
    ts = float(imu_input[1, -1] - imu_input[0, -1])
    nts = nn * ts
    rot_gate_rad = 5.0 * glv.deg
    length = (len(imu_input) // nn) * nn
    imu_corr = imu_input[:length]
    qnbi = acc18.a2qua(qnb_init) if len(np.asarray(qnb_init).reshape(-1)) == 3 else np.asarray(qnb_init).reshape(4)

    eth = acc18.Earth(pos0)
    cnn = acc18.rv2m(-eth.wnie * nts / 2)
    kf = mod.avnkfinit_24(
        nts, pos0, phi0, imuerr, wvn,
        np.array([0.05, 0.05, 0.05]) * glv.dph,
        np.array([300.0, 300.0, 300.0]),
        np.array([0.01, 0.01, 0.01]) * glv.ug,
        np.array([100.0, 100.0, 100.0]),
        enable_scale_states=scale_active,
    )

    vn = np.zeros(3)
    trace = []
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

        high_rot = False
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
            kf['xk'][18:24] = 0.0

        kf['Phikk_1'] = phi_k
        kf = acc18.kfupdate(kf, vn)
        qnbi = acc18.qdelphi(qnbi, 0.91 * kf['xk'][0:3])
        kf['xk'][0:3] *= 0.09
        vn = vn - 0.91 * kf['xk'][3:6]
        kf['xk'][3:6] *= 0.09
        if not scale_active:
            kf['xk'][18:24] = 0.0

        if k % sample_stride_raw == 0 or k + nn >= length:
            truth = SETUP['att_truth'][min(k + nn - 1, len(SETUP['att_truth']) - 1), 0:3]
            err = acc18.qq2phi(qnbi, acc18.a2qua(truth)) / glv.sec
            trace.append({
                'time_s': float((k + nn) * ts),
                'roll_abs_arcsec': float(abs(err[0])),
                'pitch_signed_arcsec': float(err[1]),
                'yaw_abs_arcsec': float(abs(err[2])),
                'high_rot': bool(high_rot),
            })

    return {
        'trace': trace,
        'final_qnbi': qnbi.copy(),
        'final_xk': np.array(kf['xk'], dtype=float),
    }


def run_method_cleanview(staged_release: bool):
    cfg = mod.Hybrid24Config(name='tmp', label='tmp', seeds=[0], staged_release=staged_release)
    imu_corr = SETUP['imu_noisy'].copy()
    qnb_seed_actual = acc18.a2qua(SETUP['att0_guess'])
    traces = {}

    for iteration in range(1, 6):
        scale_active = (not cfg.staged_release) or (iteration >= cfg.release_iter)
        # clean-view convergence: same initial coarse misalignment every iter, on the current corrected dataset snapshot
        clean = run_one_iteration(
            imu_input=imu_corr.copy(),
            qnb_init=SETUP['att0_guess'],
            pos0=SETUP['pos0'],
            phi0=SETUP['phi'],
            imuerr=SETUP['imuerr'],
            wvn=SETUP['wvn'],
            scale_active=scale_active,
        )
        traces[str(iteration)] = clean['trace']

        # actual outer-iteration update: preserve carry_att_seed + wash logic for next iteration snapshot
        actual = run_one_iteration(
            imu_input=imu_corr,
            qnb_init=qnb_seed_actual,
            pos0=SETUP['pos0'],
            phi0=SETUP['phi'],
            imuerr=SETUP['imuerr'],
            wvn=SETUP['wvn'],
            scale_active=scale_active,
        )

        if iteration < 5:
            if cfg.carry_att_seed:
                qnb_seed_actual = actual['final_qnbi'].copy()
            ts = SETUP['ts']
            imu_corr[:, 0:3] -= cfg.wash_scale * actual['final_xk'][6:9] * ts
            imu_corr[:, 3:6] -= cfg.wash_scale * actual['final_xk'][9:12] * ts
            if scale_active and cfg.scale_wash_scale > 0.0:
                imu_corr = mod.apply_scale_wash(imu_corr, actual['final_xk'][18:21], actual['final_xk'][21:24], cfg.scale_wash_scale)

    return traces


def gate_spans(rows):
    spans = []
    start = None
    prev = None
    for row in rows:
        t = row['time_s']
        flag = row['high_rot']
        if flag and start is None:
            start = t
        if (not flag) and start is not None:
            spans.append((start, prev if prev is not None else t))
            start = None
        prev = t
    if start is not None and prev is not None:
        spans.append((start, prev))
    return spans


def build_table_rows():
    obj = json.loads(RESULT_JSON.read_text(encoding='utf-8'))
    rows = []
    for method_key, method_label in [('plain24_iter5', 'plain24'), ('staged24_iter5', 'staged24')]:
        per_seed = obj[method_key]['per_seed']
        for it in range(1, 6):
            errs = []
            for seed_row in per_seed:
                log = next(x for x in seed_row['iter_logs'] if x['iteration'] == it)
                errs.append(np.array(log['att_err_arcsec'], dtype=float))
            errs = np.array(errs)
            abs_errs = np.abs(errs)
            norms = np.linalg.norm(errs, axis=1)
            rows.append({
                'method': method_label,
                'iter': it,
                'roll_mean_abs_arcsec': float(abs_errs[:, 0].mean()),
                'pitch_mean_abs_arcsec': float(abs_errs[:, 1].mean()),
                'yaw_mean_abs_arcsec': float(abs_errs[:, 2].mean()),
                'norm_mean_arcsec': float(norms.mean()),
            })
    return rows


SETUP = build_setup(seed=0)


def main():
    plain = run_method_cleanview(False)
    staged = run_method_cleanview(True)
    table_rows = build_table_rows()

    fig, axes = plt.subplots(3, 5, figsize=(18, 9))
    row_keys = ['roll_abs_arcsec', 'pitch_signed_arcsec', 'yaw_abs_arcsec']
    ylabels = ['roll |error| / arcsec', 'pitch error / arcsec', 'yaw |error| / arcsec']

    for col, it in enumerate(range(1, 6)):
        gate = gate_spans(staged[str(it)])
        for ridx, (key, ylabel) in enumerate(zip(row_keys, ylabels)):
            ax = axes[ridx, col]
            for a, b in gate:
                ax.axvspan(a, b, color='#d9d9d9', alpha=0.20)
            ax.plot([r['time_s'] for r in plain[str(it)]], [r[key] for r in plain[str(it)]], linewidth=1.8, label='plain24')
            ax.plot([r['time_s'] for r in staged[str(it)]], [r[key] for r in staged[str(it)]], linewidth=1.8, label='staged24')
            if ridx == 1:
                ax.axhline(0.0, color='k', linestyle='--', linewidth=0.8)
            ax.grid(True, linestyle='--', alpha=0.3)
            if col == 0:
                ax.set_ylabel(ylabel)
            if ridx == 2:
                ax.set_xlabel('alignment time / s')
            if ridx == 0:
                title = f'iter={it}'
                if it == 1:
                    title += ' (freeze for staged)'
                else:
                    title += ' (release+gate for staged)'
                ax.set_title(title)
            if ridx == 0 and col == 4:
                ax.legend(frameon=False, loc='upper right')

    fig.suptitle('State convergence under each outer iteration (seed0 replay; same initial misalignment per iter)')
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220, bbox_inches='tight')
    plt.close(fig)

    payload = {
        'figure': str(OUT_PNG),
        'plain24_seed0_cleanview': plain,
        'staged24_seed0_cleanview': staged,
        'table_rows_5seed': table_rows,
        'note': 'For readability, each iter panel replays the current corrected dataset from the same coarse initial misalignment (seed0), while the table still reports 5-seed iteration statistics from the actual iterative pipeline.',
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'figure': str(OUT_PNG), 'rows': len(table_rows)}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
