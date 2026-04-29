#!/usr/bin/env python3
"""DAR minimal hybrid 24-state staged prototype.

State layout:
    x = [phi(3), dv(3), eb(3), db(3), ng(3), xa(3), kg(3), ka(3)]

Goal:
- start from the working 18-state accel-colored prototype
- add diagonal dKg/dKa states only
- compare two 24-state variants under the same truth:
  1) plain-24: kg/ka active from iteration 1
  2) staged-24: iteration 1 keeps kg/ka frozen; from iteration 2 onward,
     kg/ka are released and only coupled during high-rotation segments
- test whether staged release avoids over-parameterization while preserving the
  pitch-bias repair discovered in the minimal hybrid24 probe
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
PSINS_ROOT = WORKSPACE / 'tmp_psins_py'
if str(PSINS_ROOT) not in sys.path:
    sys.path.insert(0, str(PSINS_ROOT))

SCRIPT_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_hybrid24_staged_py_2026-03-30.py'
ACC18_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_accel_colored_py_2026-03-30.py'
BASE12_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
BASELINE_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'alignvn_dar_hybrid24_pitch_repair_probe_2026-03-30.json'
OUT_DIR = WORKSPACE / 'tmp' / 'alignment_strategy_sweep'
OUT_JSON = OUT_DIR / 'alignvn_dar_hybrid24_staged_result_2026-03-30.json'
OUT_MD = OUT_DIR / 'alignvn_dar_hybrid24_staged_summary_2026-03-30.md'
MAX_WORKERS = min(4, os.cpu_count() or 1)

ACC18 = None
BASE12 = None


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_acc18():
    global ACC18
    if ACC18 is None:
        ACC18 = load_module('alignvn_acc18_h24staged_20260330', ACC18_PATH)
    return ACC18


def load_base12():
    global BASE12
    if BASE12 is None:
        BASE12 = load_module('alignvn_base12_h24staged_20260330', BASE12_PATH)
    return BASE12


@dataclass
class Hybrid24Config:
    name: str
    label: str
    seeds: list[int]
    max_iter: int = 5
    wash_scale: float = 0.5
    scale_wash_scale: float = 0.5
    carry_att_seed: bool = True
    staged_release: bool = False
    release_iter: int = 2
    rot_gate_dps: float = 5.0
    ng_sigma_dph: list[float] | None = None
    tau_g_s: list[float] | None = None
    xa_sigma_ug: list[float] | None = None
    tau_a_s: list[float] | None = None
    note: str = ''

    def __post_init__(self):
        if self.ng_sigma_dph is None:
            self.ng_sigma_dph = [0.05, 0.05, 0.05]
        if self.tau_g_s is None:
            self.tau_g_s = [300.0, 300.0, 300.0]
        if self.xa_sigma_ug is None:
            self.xa_sigma_ug = [0.01, 0.01, 0.01]
        if self.tau_a_s is None:
            self.tau_a_s = [100.0, 100.0, 100.0]


def avnkfinit_24(nts: float, pos: np.ndarray, phi0: np.ndarray, imuerr: dict[str, np.ndarray], wvn: np.ndarray,
                 ng_sigma: np.ndarray, tau_g_s: np.ndarray, xa_sigma: np.ndarray, tau_a_s: np.ndarray,
                 enable_scale_states: bool) -> dict[str, Any]:
    acc18 = load_acc18()
    glv = acc18.glv
    eth = acc18.Earth(pos)
    web = np.asarray(imuerr['web']).reshape(3)
    wdb = np.asarray(imuerr['wdb']).reshape(3)
    eb = np.asarray(imuerr['eb']).reshape(3)
    db = np.asarray(imuerr['db']).reshape(3)
    ng_sigma = np.asarray(ng_sigma).reshape(3)
    tau_g_s = np.asarray(tau_g_s).reshape(3)
    xa_sigma = np.asarray(xa_sigma).reshape(3)
    tau_a_s = np.asarray(tau_a_s).reshape(3)

    init_eb_p = np.maximum(eb, 0.1 * glv.dph)
    init_db_p = np.maximum(db, 1000 * glv.ug)
    init_xa_p = np.maximum(xa_sigma, 5.0 * glv.ug)
    init_scale_p = np.full(3, 100.0 * glv.ppm)

    fg = np.exp(-nts / tau_g_s)
    fa = np.exp(-nts / tau_a_s)
    q_ng = ng_sigma * np.sqrt(np.maximum(1.0 - fg**2, 0.0))
    q_xa = xa_sigma * np.sqrt(np.maximum(1.0 - fa**2, 0.0))

    qk = np.zeros((24, 24))
    qk[0:3, 0:3] = np.diag(web**2 * nts)
    qk[3:6, 3:6] = np.diag(wdb**2 * nts)
    qk[12:15, 12:15] = np.diag(q_ng**2)
    qk[15:18, 15:18] = np.diag(q_xa**2)

    ft = np.zeros((24, 24))
    ft[0:3, 0:3] = acc18.askew(-eth.wnie)
    phikk_1 = np.eye(24) + ft * nts
    phikk_1[12:15, 12:15] = np.diag(fg)
    phikk_1[15:18, 15:18] = np.diag(fa)

    p_diag = np.r_[phi0, np.ones(3), init_eb_p, init_db_p, ng_sigma, init_xa_p,
                   init_scale_p if enable_scale_states else np.zeros(3),
                   init_scale_p if enable_scale_states else np.zeros(3)]

    return {
        'n': 24,
        'm': 3,
        'nts': nts,
        'Qk': qk,
        'Rk': np.diag(wvn.reshape(3)) ** 2 / nts,
        'Pxk': np.diag(p_diag) ** 2,
        'Phikk_1': phikk_1,
        'Hk': np.hstack([np.zeros((3, 3)), np.eye(3), np.zeros((3, 18))]),
        'xk': np.zeros(24),
        'fg': fg,
        'fa': fa,
    }


def apply_scale_wash(imu_corr: np.ndarray, kg: np.ndarray, ka: np.ndarray, scale_wash_scale: float) -> np.ndarray:
    imu_new = imu_corr.copy()
    kgs = scale_wash_scale * np.asarray(kg).reshape(3)
    kas = scale_wash_scale * np.asarray(ka).reshape(3)
    for i in range(3):
        imu_new[:, i] = imu_new[:, i] / (1.0 + kgs[i])
        imu_new[:, 3 + i] = imu_new[:, 3 + i] / (1.0 + kas[i])
    return imu_new


def alignvn_24state_iter(imu: np.ndarray, qnb: np.ndarray, pos: np.ndarray, phi0: np.ndarray,
                         imuerr: dict[str, np.ndarray], wvn: np.ndarray, cfg: Hybrid24Config,
                         truth_att: np.ndarray) -> tuple[np.ndarray, list[dict[str, Any]]]:
    acc18 = load_acc18()
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
    rot_gate_rad = cfg.rot_gate_dps * glv.deg

    iter_logs: list[dict[str, Any]] = []
    final_att = None

    for iteration in range(1, cfg.max_iter + 1):
        scale_active = (not cfg.staged_release) or (iteration >= cfg.release_iter)
        kf = avnkfinit_24(
            nts, pos, phi0, imuerr, wvn,
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

        final_att = acc18.q2att(qnbi)
        att_err_arcsec = acc18.qq2phi(acc18.a2qua(final_att), acc18.a2qua(truth_att)) / glv.sec
        iter_logs.append({
            'iteration': iteration,
            'scale_active': scale_active,
            'att_err_arcsec': [float(x) for x in att_err_arcsec],
            'att_err_norm_arcsec': float(np.linalg.norm(att_err_arcsec)),
            'yaw_abs_arcsec': float(abs(att_err_arcsec[2])),
            'est_kg_ppm': (kf['xk'][18:21] / glv.ppm).tolist(),
            'est_ka_ppm': (kf['xk'][21:24] / glv.ppm).tolist(),
        })

        if iteration < cfg.max_iter:
            if cfg.carry_att_seed:
                qnb_seed = qnbi.copy()
            imu_corr[:, 0:3] -= cfg.wash_scale * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= cfg.wash_scale * kf['xk'][9:12] * ts
            if scale_active and cfg.scale_wash_scale > 0.0:
                imu_corr = apply_scale_wash(imu_corr, kf['xk'][18:21], kf['xk'][21:24], cfg.scale_wash_scale)

    assert final_att is not None
    return final_att, iter_logs


def run_single_seed(task: tuple[Hybrid24Config, int]) -> dict[str, Any]:
    cfg, seed = task
    acc18 = load_acc18()
    base12 = load_base12()

    np.random.seed(seed)

    ts = 0.01
    att0 = np.array([0.0, 0.0, 0.0])
    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    rot_paras = base12.build_rot_paras()
    att_truth = acc18.attrottt(att0, rot_paras, ts)
    imu, _ = acc18.avp2imu(att_truth, pos0)
    imuerr = base12.build_imuerr()
    imu_noisy = acc18.imuadderr(imu, imuerr)

    phi = np.array([0.1, 0.1, 0.5]) * acc18.glv.deg
    att0_guess = acc18.q2att(base12.qaddphi(acc18.a2qua(att0), phi))
    wvn = np.array([0.01, 0.01, 0.01])

    _, iter_logs = alignvn_24state_iter(
        imu=imu_noisy.copy(),
        qnb=att0_guess,
        pos=pos0,
        phi0=phi,
        imuerr=imuerr,
        wvn=wvn,
        cfg=cfg,
        truth_att=att_truth[-1, 0:3],
    )
    last = iter_logs[-1]
    return {
        'seed': seed,
        'final_att_err_arcsec': last['att_err_arcsec'],
        'final_att_err_abs_arcsec': np.abs(np.array(last['att_err_arcsec'])).tolist(),
        'final_att_err_norm_arcsec': last['att_err_norm_arcsec'],
        'final_yaw_abs_arcsec': last['yaw_abs_arcsec'],
        'iter_logs': iter_logs,
    }


def summarize_config(cfg: Hybrid24Config) -> dict[str, Any]:
    tasks = [(cfg, seed) for seed in cfg.seeds]
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as ex:
        per_seed = list(ex.map(run_single_seed, tasks))
    per_seed.sort(key=lambda x: x['seed'])

    errs = np.array([row['final_att_err_arcsec'] for row in per_seed], dtype=float)
    abs_errs = np.abs(errs)
    norms = np.array([row['final_att_err_norm_arcsec'] for row in per_seed], dtype=float)
    yaw_abs = np.array([row['final_yaw_abs_arcsec'] for row in per_seed], dtype=float)
    pitch = errs[:, 1]

    return {
        'name': cfg.name,
        'label': cfg.label,
        'config': asdict(cfg),
        'statistics': {
            'mean_signed_arcsec': errs.mean(axis=0).tolist(),
            'std_signed_arcsec_1sigma': errs.std(axis=0, ddof=1).tolist() if len(per_seed) > 1 else [0.0, 0.0, 0.0],
            'mean_abs_arcsec': abs_errs.mean(axis=0).tolist(),
            'median_abs_arcsec': np.median(abs_errs, axis=0).tolist(),
            'norm_mean_arcsec': float(norms.mean()),
            'yaw_abs_mean_arcsec': float(yaw_abs.mean()),
            'yaw_abs_median_arcsec': float(np.median(yaw_abs)),
            'pitch_signed_range_arcsec': [float(pitch.min()), float(pitch.max())],
        },
        'per_seed': per_seed,
    }


def load_baseline_reference() -> dict[str, Any] | None:
    if not BASELINE_JSON.exists():
        return None
    try:
        return json.loads(BASELINE_JSON.read_text())
    except Exception:
        return None


def build_markdown(payload: dict[str, Any]) -> str:
    lines = [
        '# Hybrid24 staged probe (2026-03-30)',
        '',
        '## Setup',
        '- same truth as current 18-state / hybrid24 tests',
        '- injected diagonal `dKg/dKa = 30 ppm` remains unchanged',
        '- compare plain 24-state vs staged 24-state under the same 5-seed set',
        '',
        '## Main conclusion',
        f"- {payload['judgement']}",
        '',
        '| config | pitch mean signed (\") | pitch mean abs (\") | yaw abs mean (\") | norm mean (\") |',
        '|---|---:|---:|---:|---:|',
    ]
    for item in payload['rows_for_table']:
        st = item['statistics']
        lines.append(
            f"| {item['name']} | {st['mean_signed_arcsec'][1]:.2f} | {st['mean_abs_arcsec'][1]:.2f} | {st['yaw_abs_mean_arcsec']:.2f} | {st['norm_mean_arcsec']:.2f} |"
        )
    lines.extend([
        '',
        '## Interpretation',
        f"- {payload['interpretation_1']}",
        f"- {payload['interpretation_2']}",
        f"- {payload['interpretation_3']}",
        '',
        '## Files',
        f'- script: `{SCRIPT_PATH}`',
        f'- json: `{OUT_JSON}`',
        f'- md: `{OUT_MD}`',
    ])
    return '\n'.join(lines) + '\n'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    baseline_ref = load_baseline_reference()

    cfg_plain = Hybrid24Config(
        name='hybrid24_plain_iter5',
        label='plain 24-state iter=5',
        seeds=[0, 1, 2, 3, 4],
        max_iter=5,
        staged_release=False,
        note='kg/ka active from iteration 1; high-rotation gate still enabled inside run.',
    )
    cfg_staged = Hybrid24Config(
        name='hybrid24_staged_iter5',
        label='staged 24-state iter=5',
        seeds=[0, 1, 2, 3, 4],
        max_iter=5,
        staged_release=True,
        release_iter=2,
        rot_gate_dps=5.0,
        scale_wash_scale=0.5,
        note='iter1 freezes kg/ka; iter>=2 releases them and only couples on high-rotation segments.',
    )

    plain = summarize_config(cfg_plain)
    staged = summarize_config(cfg_staged)

    payload = {
        'baseline_reference': baseline_ref,
        'plain24_iter5': plain,
        'staged24_iter5': staged,
    }

    rows_for_table = []
    if baseline_ref is not None and 'baseline18_iter5' in baseline_ref:
        rows_for_table.append({
            'name': 'baseline18_iter5',
            'statistics': {
                'mean_signed_arcsec': [0.0, baseline_ref['baseline18_iter5']['pitch_mean_signed'], 0.0],
                'mean_abs_arcsec': [0.0, baseline_ref['baseline18_iter5']['pitch_mean_abs'], 0.0],
                'yaw_abs_mean_arcsec': baseline_ref['baseline18_iter5']['yaw_abs_mean'],
                'norm_mean_arcsec': baseline_ref['baseline18_iter5']['norm_mean'],
            },
        })
    rows_for_table.extend([plain, staged])
    payload['rows_for_table'] = rows_for_table

    p_pitch = plain['statistics']['mean_abs_arcsec'][1]
    s_pitch = staged['statistics']['mean_abs_arcsec'][1]
    p_yaw = plain['statistics']['yaw_abs_mean_arcsec']
    s_yaw = staged['statistics']['yaw_abs_mean_arcsec']
    p_norm = plain['statistics']['norm_mean_arcsec']
    s_norm = staged['statistics']['norm_mean_arcsec']

    if s_norm <= p_norm and s_pitch <= p_pitch:
        judgement = 'staged 24-state is at least as stable as plain 24-state on this probe, and does not lose the pitch repair.'
    elif s_pitch <= p_pitch and s_yaw > p_yaw:
        judgement = 'staged 24-state keeps the pitch repair but trades some yaw performance; this means staged release is safer, not strictly better.'
    else:
        judgement = 'staged 24-state did not beat plain 24-state on this probe; staged release may be too conservative under the current path.'

    payload['judgement'] = judgement
    payload['interpretation_1'] = (
        f"plain24 iter=5: pitch mean abs={p_pitch:.2f}\", yaw abs mean={p_yaw:.2f}\", norm mean={p_norm:.2f}\"."
    )
    payload['interpretation_2'] = (
        f"staged24 iter=5: pitch mean abs={s_pitch:.2f}\", yaw abs mean={s_yaw:.2f}\", norm mean={s_norm:.2f}\"."
    )
    payload['interpretation_3'] = (
        'If staged and plain are close, that means diagonal dKg/dKa are not too many for this DAR path; '
        'if staged is clearly better, it means delayed release helps avoid early false learning.'
    )

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    OUT_MD.write_text(build_markdown(payload))
    print(json.dumps({
        'baseline18_iter5': baseline_ref.get('baseline18_iter5') if baseline_ref else None,
        'plain24_iter5': plain['statistics'],
        'staged24_iter5': staged['statistics'],
        'judgement': judgement,
        'out_json': str(OUT_JSON),
        'out_md': str(OUT_MD),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
