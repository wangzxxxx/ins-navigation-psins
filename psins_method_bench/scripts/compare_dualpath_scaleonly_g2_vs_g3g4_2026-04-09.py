#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import sys
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
BASELINE_JSON = WORKSPACE / 'psins_method_bench' / 'results' / 'compare_four_group_alignment_arcsec_2026-04-05.json'

RESULTS_DIR = WORKSPACE / 'psins_method_bench' / 'results'
REPORTS_DIR = WORKSPACE / 'reports'
OUT_JSON = RESULTS_DIR / 'compare_dualpath_scaleonly_g2_vs_g3g4_2026-04-09.json'
OUT_MD = REPORTS_DIR / 'psins_dualpath_scaleonly_g2_vs_g3g4_2026-04-09.md'

MAX_WORKERS = min(4, os.cpu_count() or 1)
SEEDS = [0, 1, 2, 3, 4]
TS = 0.01
WVN = np.array([0.01, 0.01, 0.01])
PHI_DEG = np.array([0.1, 0.1, 0.5])
ROT_GATE_DPS = 5.0
MAX_ITER = 5
WASH_SCALE = 0.5
SCALE_WASH_SCALE = 0.5

_BASE12 = None
_H24 = None


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
        _BASE12 = load_module('dualpath_scaleonly_base12_20260409', BASE12_PATH)
    return _BASE12


def load_h24():
    global _H24
    if _H24 is None:
        _H24 = load_module('dualpath_scaleonly_h24_20260409', H24_PATH)
    return _H24


def avnkfinit_scale18(nts: float, pos: np.ndarray, phi0: np.ndarray, imuerr: dict[str, np.ndarray], wvn: np.ndarray) -> dict[str, Any]:
    h24 = load_h24()
    acc18 = h24.load_acc18()
    glv = acc18.glv
    eth = acc18.Earth(pos)

    web = np.asarray(imuerr['web']).reshape(3)
    wdb = np.asarray(imuerr['wdb']).reshape(3)
    eb = np.asarray(imuerr['eb']).reshape(3)
    db = np.asarray(imuerr['db']).reshape(3)

    init_eb_p = np.maximum(eb, 0.1 * glv.dph)
    init_db_p = np.maximum(db, 1000 * glv.ug)
    init_scale_p = np.full(3, 100.0 * glv.ppm)

    qk = np.zeros((18, 18))
    qk[0:3, 0:3] = np.diag(web**2 * nts)
    qk[3:6, 3:6] = np.diag(wdb**2 * nts)

    ft = np.zeros((18, 18))
    ft[0:3, 0:3] = acc18.askew(-eth.wnie)
    phikk_1 = np.eye(18) + ft * nts

    return {
        'n': 18,
        'm': 3,
        'nts': nts,
        'Qk': qk,
        'Rk': np.diag(wvn.reshape(3)) ** 2 / nts,
        'Pxk': np.diag(np.r_[phi0, np.ones(3), init_eb_p, init_db_p, init_scale_p, init_scale_p]) ** 2,
        'Phikk_1': phikk_1,
        'Hk': np.hstack([np.zeros((3, 3)), np.eye(3), np.zeros((3, 12))]),
        'xk': np.zeros(18),
    }


def alignvn_scale18_iter(
    imu: np.ndarray,
    qnb: np.ndarray,
    pos: np.ndarray,
    phi0: np.ndarray,
    imuerr: dict[str, np.ndarray],
    wvn: np.ndarray,
    max_iter: int,
    truth_att: np.ndarray,
    wash_scale: float = WASH_SCALE,
    scale_wash_scale: float = SCALE_WASH_SCALE,
    carry_att_seed: bool = True,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    h24 = load_h24()
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
    rot_gate_rad = ROT_GATE_DPS * glv.deg

    iter_logs: list[dict[str, Any]] = []
    final_att = None

    for iteration in range(1, max_iter + 1):
        kf = avnkfinit_scale18(nts, pos, phi0, imuerr, wvn)
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
            phi_k[0:3, 6:9] = -cnbts

            high_rot = np.max(np.abs(phim / nts)) > rot_gate_rad
            if high_rot:
                phi_k[0:3, 12:15] = -cnb @ np.diag(phim[0:3])
                phi_k[3:6, 15:18] = cnb @ np.diag(dvbm[0:3])
            else:
                phi_k[0:3, 12:15] = 0.0
                phi_k[3:6, 15:18] = 0.0

            kf['Phikk_1'] = phi_k
            kf = acc18.kfupdate(kf, vn)

            qnbi = acc18.qdelphi(qnbi, 0.91 * kf['xk'][0:3])
            kf['xk'][0:3] *= 0.09
            vn = vn - 0.91 * kf['xk'][3:6]
            kf['xk'][3:6] *= 0.09

        final_att = acc18.q2att(qnbi)
        att_err_arcsec = acc18.qq2phi(acc18.a2qua(final_att), acc18.a2qua(truth_att)) / glv.sec
        iter_logs.append({
            'iteration': iteration,
            'final_att_deg': (final_att / glv.deg).tolist(),
            'att_err_arcsec': [float(x) for x in att_err_arcsec],
            'att_err_norm_arcsec': float(np.linalg.norm(att_err_arcsec)),
            'yaw_abs_arcsec': float(abs(att_err_arcsec[2])),
            'est_eb_dph': (kf['xk'][6:9] / glv.dph).tolist(),
            'est_db_ug': (kf['xk'][9:12] / glv.ug).tolist(),
            'est_kg_ppm': (kf['xk'][12:15] / glv.ppm).tolist(),
            'est_ka_ppm': (kf['xk'][15:18] / glv.ppm).tolist(),
            'high_rot_gate_used': bool(high_rot),
        })

        if iteration < max_iter:
            if carry_att_seed:
                qnb_seed = qnbi.copy()
            imu_corr[:, 0:3] -= wash_scale * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= wash_scale * kf['xk'][9:12] * ts
            if scale_wash_scale > 0.0:
                imu_corr = h24.apply_scale_wash(imu_corr, kf['xk'][12:15], kf['xk'][15:18], scale_wash_scale)

    assert final_att is not None
    return final_att, iter_logs


def run_single(seed: int) -> dict[str, Any]:
    base12 = load_base12()
    h24 = load_h24()
    acc18 = h24.load_acc18()

    np.random.seed(seed)

    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    att0_ref = np.array([0.0, 0.0, 0.0])
    rot_paras = base12.build_rot_paras()
    att_truth = acc18.attrottt(att0_ref, rot_paras, TS)
    imu, _ = acc18.avp2imu(att_truth, pos0)
    imuerr = base12.build_imuerr()  # keep truth dKg/dKa = 30 ppm
    imu_noisy = acc18.imuadderr(imu, imuerr)

    phi = PHI_DEG * acc18.glv.deg
    att0_guess = acc18.q2att(base12.qaddphi(acc18.a2qua(att0_ref), phi))
    truth_att = att_truth[-1, 0:3]

    _, iter_logs = alignvn_scale18_iter(
        imu=imu_noisy.copy(),
        qnb=att0_guess,
        pos=pos0,
        phi0=phi,
        imuerr=imuerr,
        wvn=WVN.copy(),
        max_iter=MAX_ITER,
        truth_att=truth_att,
        wash_scale=WASH_SCALE,
        scale_wash_scale=SCALE_WASH_SCALE,
        carry_att_seed=True,
    )
    last = iter_logs[-1]
    err = np.array(last['att_err_arcsec'], dtype=float)
    return {
        'group_key': 'g2_scaleonly_rotation',
        'seed': seed,
        'final_att_err_arcsec': [float(x) for x in err],
        'final_att_err_abs_arcsec': [float(x) for x in np.abs(err)],
        'final_att_err_norm_arcsec': float(last['att_err_norm_arcsec']),
        'final_yaw_abs_arcsec': float(last['yaw_abs_arcsec']),
        'iter_logs': iter_logs,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errs = np.array([row['final_att_err_arcsec'] for row in rows], dtype=float)
    abs_errs = np.abs(errs)
    norms = np.array([row['final_att_err_norm_arcsec'] for row in rows], dtype=float)
    yaw_abs = np.array([row['final_yaw_abs_arcsec'] for row in rows], dtype=float)
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
        'per_seed': rows,
    }


def build_summary_row(group_key: str, display: str, st: dict[str, Any]) -> dict[str, Any]:
    return {
        'group_key': group_key,
        'display': display,
        'pitch_mean_abs_arcsec': st['pitch_mean_abs_arcsec'],
        'yaw_abs_mean_arcsec': st['yaw_abs_mean_arcsec'],
        'norm_mean_arcsec': st['norm_mean_arcsec'],
        'yaw_abs_median_arcsec': st['yaw_abs_median_arcsec'],
        'yaw_abs_max_arcsec': st['yaw_abs_max_arcsec'],
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline = json.loads(BASELINE_JSON.read_text(encoding='utf-8'))
    baseline_groups = baseline['groups']

    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(SEEDS))) as ex:
        rows = list(ex.map(run_single, SEEDS))
    rows.sort(key=lambda x: x['seed'])
    g2_scale = summarize_rows(rows)

    g3 = baseline_groups['g3_markov_rotation']
    g4 = baseline_groups['g4_scd_rotation']

    summary_rows = [
        build_summary_row('g2_scaleonly_rotation', 'G2 scale-only (eb/db/kg/ka) @ 双轴旋转对准策略', g2_scale),
        build_summary_row('g3_markov_rotation', 'G3 Markov/GM-family plain24 @ 双轴旋转对准策略', g3),
        build_summary_row('g4_scd_rotation', 'G4 Markov + SCD @ 双轴旋转对准策略', g4),
    ]

    pairwise = {}
    for left, right in [('g2_scaleonly_rotation', 'g3_markov_rotation'), ('g3_markov_rotation', 'g4_scd_rotation'), ('g2_scaleonly_rotation', 'g4_scd_rotation')]:
        l = next(x for x in summary_rows if x['group_key'] == left)
        r = next(x for x in summary_rows if x['group_key'] == right)
        pairwise[f'{left}_vs_{right}'] = {
            'pitch_delta_arcsec': float(l['pitch_mean_abs_arcsec'] - r['pitch_mean_abs_arcsec']),
            'yaw_delta_arcsec': float(l['yaw_abs_mean_arcsec'] - r['yaw_abs_mean_arcsec']),
            'norm_delta_arcsec': float(l['norm_mean_arcsec'] - r['norm_mean_arcsec']),
            'right_better_pitch': bool(r['pitch_mean_abs_arcsec'] < l['pitch_mean_abs_arcsec']),
            'right_better_yaw': bool(r['yaw_abs_mean_arcsec'] < l['yaw_abs_mean_arcsec']),
            'right_better_norm': bool(r['norm_mean_arcsec'] < l['norm_mean_arcsec']),
        }

    result = {
        'task': 'compare_dualpath_scaleonly_g2_vs_g3g4_2026_04_09',
        'reference_json': str(BASELINE_JSON),
        'metric_definition': 'pitch mean abs / yaw abs mean / norm mean in arcsec',
        'path_note': 'All three groups use the same old Chapter-4 dual-axis rotation strategy build_rot_paras().',
        'variation_note': (
            'New G2 keeps the same dual-axis path and truth/noise semantics as the accepted 2026-04-05 alignment baseline, '
            'but uses scale-only state modeling x=[phi(3), dv(3), eb(3), db(3), kg(3), ka(3)] without ng/xa Markov states. '
            'G3/G4 are reused from the accepted baseline JSON.'
        ),
        'g2_scaleonly_config': {
            'state_layout': ['phi(3)', 'dv(3)', 'eb(3)', 'db(3)', 'kg(3)', 'ka(3)'],
            'max_iter': MAX_ITER,
            'wash_scale': WASH_SCALE,
            'scale_wash_scale': SCALE_WASH_SCALE,
            'carry_att_seed': True,
            'rot_gate_dps': ROT_GATE_DPS,
            'truth_scale_injection_kept': True,
            'truth_path': 'build_rot_paras()',
            'truth_att0_deg': [0.0, 0.0, 0.0],
            'phi_guess_deg': PHI_DEG.tolist(),
            'wvn': WVN.tolist(),
            'seeds': SEEDS,
        },
        'summary_rows': summary_rows,
        'pairwise': pairwise,
        'groups': {
            'g2_scaleonly_rotation': g2_scale,
            'g3_markov_rotation': g3,
            'g4_scd_rotation': g4,
        },
        'files': {
            'json': str(OUT_JSON),
            'md': str(OUT_MD),
        },
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# 双轴路径上新增 G2(scale-only) 与 G3/G4 对比',
        '',
        '- 路径统一：旧第四章双轴旋转对准策略 `build_rot_paras()`',
        '- truth / noise 口径统一复用 accepted baseline (`compare_four_group_alignment_arcsec_2026-04-05.json`)',
        '- 新 G2：`x = [phi(3), dv(3), eb(3), db(3), kg(3), ka(3)]`，不含 `ng/xa` Markov 误差建模',
        '- G3/G4：直接复用 accepted baseline 结果',
        '',
        '| 组别 | pitch mean abs (") | yaw abs mean (") | norm mean (") | yaw median (") | yaw max (") |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['display']} | {row['pitch_mean_abs_arcsec']:.6f} | {row['yaw_abs_mean_arcsec']:.6f} | {row['norm_mean_arcsec']:.6f} | {row['yaw_abs_median_arcsec']:.6f} | {row['yaw_abs_max_arcsec']:.6f} |"
        )
    lines.extend([
        '',
        '## Pairwise deltas (left - right, positive means right is better)',
        '',
    ])
    for key, item in pairwise.items():
        lines.append(f'- **{key}**: pitch Δ={item["pitch_delta_arcsec"]:.6f}", yaw Δ={item["yaw_delta_arcsec"]:.6f}", norm Δ={item["norm_delta_arcsec"]:.6f}"')
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
