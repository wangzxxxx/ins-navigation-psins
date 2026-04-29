#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
PSINS_ROOT = WORKSPACE / 'tmp_psins_py'
if str(PSINS_ROOT) not in sys.path:
    sys.path.insert(0, str(PSINS_ROOT))

H24_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_hybrid24_staged_py_2026-03-30.py'
MATCHED_REF_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'ch4_plain24_staged24_truth_gm_matched_2026-03-31.json'
OUT_DIR = WORKSPACE / 'tmp' / 'alignment_strategy_sweep'
OUT_JSON = OUT_DIR / 'ch4_pure_scd_vs_freeze_2026-04-03.json'
OUT_MD = OUT_DIR / 'ch4_pure_scd_vs_freeze_2026-04-03.md'


def load_module(module_name: str, path: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


h24 = load_module('alignvn_h24_pure_scd_20260403', H24_PATH)
acc18 = h24.load_acc18()
base12 = h24.load_base12()


@dataclass
class SCDConfig:
    enabled: bool = False
    alpha: float = 0.999
    transition_duration_s: float = 2.0
    core_slice: tuple[int, int] = (0, 18)
    scale_slice: tuple[int, int] = (18, 24)
    apply_after_release_iter: int = 1
    note: str = ''


def apply_scd_once(kf: dict[str, Any], scd: SCDConfig):
    a0, a1 = scd.core_slice
    b0, b1 = scd.scale_slice
    P = kf['Pxk']
    P[a0:a1, b0:b1] *= scd.alpha
    P[b0:b1, a0:a1] *= scd.alpha
    kf['Pxk'] = (P + P.T) * 0.5
    return kf


def alignvn_24state_iter_pure_scd(imu: np.ndarray, qnb: np.ndarray, pos: np.ndarray, phi0: np.ndarray,
                                  imuerr: dict[str, np.ndarray], wvn: np.ndarray, cfg: h24.Hybrid24Config,
                                  truth_att: np.ndarray, scd: SCDConfig) -> tuple[np.ndarray, list[dict[str, Any]]]:
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
        scale_active = True  # pure-SCD keeps scale states active from iter1
        kf = h24.avnkfinit_24(
            nts, pos, phi0, imuerr, wvn,
            np.array(cfg.ng_sigma_dph) * glv.dph,
            np.array(cfg.tau_g_s),
            np.array(cfg.xa_sigma_ug) * glv.ug,
            np.array(cfg.tau_a_s),
            enable_scale_states=True,
        )
        vn = np.zeros(3)
        qnbi = qnb_seed.copy()
        time_since_rot_stop = 0.0
        scd_applied_this_phase = False

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

            if scd.enabled and iteration >= scd.apply_after_release_iter and (not high_rot):
                if (time_since_rot_stop >= scd.transition_duration_s) and (not scd_applied_this_phase):
                    kf = apply_scd_once(kf, scd)
                    scd_applied_this_phase = True

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
            if cfg.scale_wash_scale > 0.0:
                imu_corr = h24.apply_scale_wash(imu_corr, kf['xk'][18:21], kf['xk'][21:24], cfg.scale_wash_scale)

    assert final_att is not None
    return final_att, iter_logs


def run_seed(seed: int) -> dict[str, Any]:
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
    truth_att = att_truth[-1, 0:3]

    cfg_plain = h24.Hybrid24Config(name='plain24', label='plain24', seeds=[seed], max_iter=5, staged_release=False)
    cfg_staged = h24.Hybrid24Config(name='staged24', label='staged24', seeds=[seed], max_iter=5, staged_release=True, release_iter=2, rot_gate_dps=5.0, scale_wash_scale=0.5)
    cfg_scd = h24.Hybrid24Config(name='pure_scd24', label='pure_scd24', seeds=[seed], max_iter=5, staged_release=False, rot_gate_dps=5.0, scale_wash_scale=0.5)
    scd_cfg = SCDConfig(enabled=True, alpha=0.999, transition_duration_s=2.0, note='once-per-static-phase gentle cross-cov suppression on scale block')

    _, logs_plain = h24.alignvn_24state_iter(imu_noisy.copy(), att0_guess, pos0, phi, imuerr, wvn, cfg_plain, truth_att)
    _, logs_staged = h24.alignvn_24state_iter(imu_noisy.copy(), att0_guess, pos0, phi, imuerr, wvn, cfg_staged, truth_att)
    _, logs_scd = alignvn_24state_iter_pure_scd(imu_noisy.copy(), att0_guess, pos0, phi, imuerr, wvn, cfg_scd, truth_att, scd_cfg)

    out = {}
    for name, logs in [('plain24', logs_plain), ('staged24', logs_staged), ('pure_scd24', logs_scd)]:
        last = logs[-1]
        out[name] = {
            'final_att_err_arcsec': [float(x) for x in last['att_err_arcsec']],
            'final_att_err_abs_arcsec': [float(abs(x)) for x in last['att_err_arcsec']],
            'final_att_err_norm_arcsec': float(last['att_err_norm_arcsec']),
            'final_yaw_abs_arcsec': float(last['yaw_abs_arcsec']),
            'iter_logs': logs,
        }
    return {'seed': seed, 'methods': out}


def summarize_method(seed_rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    errs = np.array([row['methods'][method]['final_att_err_arcsec'] for row in seed_rows], dtype=float)
    abs_errs = np.abs(errs)
    norms = np.array([row['methods'][method]['final_att_err_norm_arcsec'] for row in seed_rows], dtype=float)
    yaw_abs = np.array([row['methods'][method]['final_yaw_abs_arcsec'] for row in seed_rows], dtype=float)
    return {
        'pitch_mean_abs_arcsec': float(abs_errs[:, 1].mean()),
        'yaw_abs_mean_arcsec': float(yaw_abs.mean()),
        'norm_mean_arcsec': float(norms.mean()),
        'yaw_abs_median_arcsec': float(np.median(yaw_abs)),
        'yaw_abs_max_arcsec': float(yaw_abs.max()),
        'mean_signed_arcsec': errs.mean(axis=0).tolist(),
        'mean_abs_arcsec': abs_errs.mean(axis=0).tolist(),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = [0, 1, 2, 3, 4]
    rows = [run_seed(seed) for seed in seeds]

    summary = {
        'plain24': summarize_method(rows, 'plain24'),
        'staged24': summarize_method(rows, 'staged24'),
        'pure_scd24': summarize_method(rows, 'pure_scd24'),
        'per_seed': rows,
    }
    if MATCHED_REF_JSON.exists():
        try:
            summary['matched_reference'] = json.loads(MATCHED_REF_JSON.read_text())
        except Exception:
            summary['matched_reference'] = None

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    p = summary['plain24']
    s = summary['staged24']
    c = summary['pure_scd24']
    lines = [
        '# Chapter 4 pure-SCD vs freeze minimal compare (2026-04-03)',
        '',
        '- same path / same seed / same noisy IMU per seed',
        '- methods: plain24, staged24(freeze→release), pure_scd24(no freeze, scale active from iter1, gentle once-per-static-phase SCD)',
        '',
        '| method | pitch mean abs (") | yaw abs mean (") | norm mean (") | yaw median (") | yaw max (") |',
        '|---|---:|---:|---:|---:|---:|',
        f"| plain24 | {p['pitch_mean_abs_arcsec']:.3f} | {p['yaw_abs_mean_arcsec']:.3f} | {p['norm_mean_arcsec']:.3f} | {p['yaw_abs_median_arcsec']:.3f} | {p['yaw_abs_max_arcsec']:.3f} |",
        f"| staged24 | {s['pitch_mean_abs_arcsec']:.3f} | {s['yaw_abs_mean_arcsec']:.3f} | {s['norm_mean_arcsec']:.3f} | {s['yaw_abs_median_arcsec']:.3f} | {s['yaw_abs_max_arcsec']:.3f} |",
        f"| pure_scd24 | {c['pitch_mean_abs_arcsec']:.3f} | {c['yaw_abs_mean_arcsec']:.3f} | {c['norm_mean_arcsec']:.3f} | {c['yaw_abs_median_arcsec']:.3f} | {c['yaw_abs_max_arcsec']:.3f} |",
        '',
    ]
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'out_json': str(OUT_JSON), 'plain24': p, 'staged24': s, 'pure_scd24': c}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
