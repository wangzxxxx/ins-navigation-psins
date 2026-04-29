#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
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

PURE_SCD_COMPARE = WORKSPACE / 'scripts' / 'compare_ch4_pure_scd_vs_freeze_2026-04-03.py'
PURE_SCD_SWEEP_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'ch4_pure_scd_param_sweep_2026-04-03.json'
OUT_DIR = WORKSPACE / 'tmp' / 'alignment_strategy_sweep'
OUT_JSON = OUT_DIR / 'ch4_llm_scd_only_alignment_2026-04-03.json'
OUT_MD = OUT_DIR / 'ch4_llm_scd_only_alignment_2026-04-03.md'
MAX_WORKERS = 6
SEEDS = [0, 1, 2, 3, 4]


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


cmp = load_module('compare_ch4_pure_scd_mod_llm_20260403', PURE_SCD_COMPARE)
acc18 = cmp.acc18
base12 = cmp.base12
h24 = cmp.h24


@dataclass(frozen=True)
class Candidate:
    name: str
    label: str
    description: str
    alpha: float
    transition_duration_s: float
    apply_after_release_iter: int
    row_indices: tuple[int, ...]
    col_indices: tuple[int, ...]


def _idx(*parts: tuple[int, int] | list[int]):
    out: list[int] = []
    for part in parts:
        if isinstance(part, tuple):
            out.extend(list(range(part[0], part[1])))
        else:
            out.extend(list(part))
    return tuple(out)


CORE_ALL = _idx((0, 18))
NAV_BIAS = _idx((0, 12))
ATT_BIAS = _idx((0, 3), (6, 12))
ATT_ONLY = _idx((0, 3))
BIAS_ONLY = _idx((6, 12))
SCALE_ALL = _idx((18, 24))
KG_ONLY = _idx((18, 21))
KA_ONLY = _idx((21, 24))


# LLM-designed constrained candidate batch:
# - same pure alignment 24-state skeleton
# - no staged/freeze
# - only SCD scope/alpha/trigger timing may move
# - hypotheses emphasize whether yaw pollution is mostly carried by attitude/bias -> kg couplings
CANDIDATES = [
    Candidate(
        name='llm_anchor_fullscale',
        label='A',
        description='Anchor candidate from the best pure-SCD sweep: full core-to-scale cut, alpha=0.995, td=2.0, iter1.',
        alpha=0.995,
        transition_duration_s=2.0,
        apply_after_release_iter=1,
        row_indices=CORE_ALL,
        col_indices=SCALE_ALL,
    ),
    Candidate(
        name='llm_navbias_fullscale',
        label='B',
        description='Hypothesis: most harmful yaw coupling sits in nav+bias -> scale rather than GM colored states; narrower row scope, stronger alpha.',
        alpha=0.992,
        transition_duration_s=2.0,
        apply_after_release_iter=1,
        row_indices=NAV_BIAS,
        col_indices=SCALE_ALL,
    ),
    Candidate(
        name='llm_attbias_fullscale_early',
        label='C',
        description='Focus directly on attitude+bias to whole scale, with slightly earlier trigger and stronger cut.',
        alpha=0.990,
        transition_duration_s=1.5,
        apply_after_release_iter=1,
        row_indices=ATT_BIAS,
        col_indices=SCALE_ALL,
    ),
    Candidate(
        name='llm_core_kgonly',
        label='D',
        description='Hypothesis: yaw damage is mostly gyro-scale driven; cut only core <-> kg couplings.',
        alpha=0.992,
        transition_duration_s=2.0,
        apply_after_release_iter=1,
        row_indices=CORE_ALL,
        col_indices=KG_ONLY,
    ),
    Candidate(
        name='llm_attbias_kgonly_early',
        label='E',
        description='Strong focused candidate: only attitude+bias <-> kg, earlier trigger, stronger cut.',
        alpha=0.989,
        transition_duration_s=1.5,
        apply_after_release_iter=1,
        row_indices=ATT_BIAS,
        col_indices=KG_ONLY,
    ),
    Candidate(
        name='llm_bias_kgonly',
        label='F',
        description='Conservative focused candidate: only eb/db <-> kg, keeping attitude-to-scale correlation untouched.',
        alpha=0.992,
        transition_duration_s=2.0,
        apply_after_release_iter=1,
        row_indices=BIAS_ONLY,
        col_indices=KG_ONLY,
    ),
    Candidate(
        name='llm_att_only_kaonly_fast',
        label='G',
        description='Counter-hypothesis probe: if pitch is mostly accel-scale contaminated, target attitude <-> ka early.',
        alpha=0.990,
        transition_duration_s=1.0,
        apply_after_release_iter=1,
        row_indices=ATT_ONLY,
        col_indices=KA_ONLY,
    ),
]


def apply_masked_scd_once(kf: dict[str, Any], row_indices: tuple[int, ...], col_indices: tuple[int, ...], alpha: float):
    P = kf['Pxk']
    rows = np.array(row_indices, dtype=int)
    cols = np.array(col_indices, dtype=int)
    P[np.ix_(rows, cols)] *= alpha
    P[np.ix_(cols, rows)] *= alpha
    kf['Pxk'] = (P + P.T) * 0.5
    return kf


def alignvn_24state_iter_llm_scd(imu: np.ndarray, qnb: np.ndarray, pos: np.ndarray, phi0: np.ndarray,
                                 imuerr: dict[str, np.ndarray], wvn: np.ndarray, cfg: h24.Hybrid24Config,
                                 truth_att: np.ndarray, candidate: Candidate) -> tuple[np.ndarray, list[dict[str, Any]]]:
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

            if iteration >= candidate.apply_after_release_iter and (not high_rot):
                if (time_since_rot_stop >= candidate.transition_duration_s) and (not scd_applied_this_phase):
                    kf = apply_masked_scd_once(kf, candidate.row_indices, candidate.col_indices, candidate.alpha)
                    scd_applied_this_phase = True

        final_att = acc18.q2att(qnbi)
        att_err_arcsec = acc18.qq2phi(acc18.a2qua(final_att), acc18.a2qua(truth_att)) / glv.sec
        iter_logs.append({
            'iteration': iteration,
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


def run_candidate_seed(task: tuple[Candidate, int]) -> dict[str, Any]:
    candidate, seed = task
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

    cfg = h24.Hybrid24Config(
        name=candidate.name,
        label=candidate.name,
        seeds=[seed],
        max_iter=5,
        staged_release=False,
        rot_gate_dps=5.0,
        scale_wash_scale=0.5,
    )
    _, logs = alignvn_24state_iter_llm_scd(
        imu_noisy.copy(),
        att0_guess,
        pos0,
        phi,
        imuerr,
        wvn,
        cfg,
        truth_att,
        candidate,
    )
    last = logs[-1]
    return {
        'candidate': asdict(candidate),
        'seed': seed,
        'final_att_err_arcsec': [float(x) for x in last['att_err_arcsec']],
        'final_att_err_abs_arcsec': [float(abs(x)) for x in last['att_err_arcsec']],
        'final_att_err_norm_arcsec': float(last['att_err_norm_arcsec']),
        'final_yaw_abs_arcsec': float(last['yaw_abs_arcsec']),
        'iter_logs': logs,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errs = np.array([row['final_att_err_arcsec'] for row in rows], dtype=float)
    abs_errs = np.abs(errs)
    norms = np.array([row['final_att_err_norm_arcsec'] for row in rows], dtype=float)
    yaw_abs = np.array([row['final_yaw_abs_arcsec'] for row in rows], dtype=float)
    return {
        'mean_signed_arcsec': errs.mean(axis=0).tolist(),
        'mean_abs_arcsec': abs_errs.mean(axis=0).tolist(),
        'pitch_mean_abs_arcsec': float(abs_errs[:, 1].mean()),
        'yaw_abs_mean_arcsec': float(yaw_abs.mean()),
        'yaw_abs_median_arcsec': float(np.median(yaw_abs)),
        'yaw_abs_max_arcsec': float(yaw_abs.max()),
        'norm_mean_arcsec': float(norms.mean()),
        'per_seed_yaw_abs_arcsec': yaw_abs.tolist(),
        'per_seed_norm_arcsec': norms.tolist(),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ref = json.loads(PURE_SCD_SWEEP_JSON.read_text()) if PURE_SCD_SWEEP_JSON.exists() else {}
    ref_plain = ref.get('reference', {}).get('plain24', {})
    ref_staged = ref.get('reference', {}).get('staged24', {})
    ref_best_pure = None
    for item in ref.get('candidates', []):
        if item['candidate']['name'] == 'hard_a995_td2_i1':
            ref_best_pure = item['statistics']
            break

    tasks = [(cand, seed) for cand in CANDIDATES for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as ex:
        rows = list(ex.map(run_candidate_seed, tasks))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row['candidate']['name'], []).append(row)
    for k in grouped:
        grouped[k].sort(key=lambda x: x['seed'])

    summary_rows = []
    for cand in CANDIDATES:
        stats = summarize_rows(grouped[cand.name])
        if ref_best_pure is not None:
            stats['delta_vs_best_pure_pitch'] = float(stats['pitch_mean_abs_arcsec'] - ref_best_pure['pitch_mean_abs_arcsec'])
            stats['delta_vs_best_pure_yaw'] = float(stats['yaw_abs_mean_arcsec'] - ref_best_pure['yaw_abs_mean_arcsec'])
            stats['delta_vs_best_pure_norm'] = float(stats['norm_mean_arcsec'] - ref_best_pure['norm_mean_arcsec'])
        else:
            stats['delta_vs_best_pure_pitch'] = None
            stats['delta_vs_best_pure_yaw'] = None
            stats['delta_vs_best_pure_norm'] = None
        summary_rows.append({
            'candidate': asdict(cand),
            'statistics': stats,
            'per_seed': grouped[cand.name],
        })

    by_yaw = sorted(summary_rows, key=lambda x: (x['statistics']['yaw_abs_mean_arcsec'], x['statistics']['norm_mean_arcsec']))
    pitch_safe = [r for r in summary_rows if r['statistics']['pitch_mean_abs_arcsec'] <= (ref_best_pure['pitch_mean_abs_arcsec'] + 0.02 if ref_best_pure else 1e9)]
    by_balanced = sorted(pitch_safe or summary_rows, key=lambda x: (x['statistics']['norm_mean_arcsec'], x['statistics']['yaw_abs_mean_arcsec']))

    payload = {
        'reference': {
            'plain24': ref_plain,
            'staged24': ref_staged,
            'best_pure_scd_sweep': ref_best_pure,
        },
        'candidates': summary_rows,
        'rankings': {
            'best_by_yaw_mean': by_yaw[0]['candidate']['name'] if by_yaw else None,
            'best_balanced_pitch_safe': by_balanced[0]['candidate']['name'] if by_balanced else None,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# Chapter 4 LLM+SCD-only alignment probe (2026-04-03)',
        '',
        '- fixed DAR 24-state alignment skeleton',
        '- no staged / no freeze',
        '- LLM only changes constrained SCD knobs: scope, alpha, trigger timing',
        '',
        '## Reference',
        f"- best pure-SCD sweep: pitch {ref_best_pure.get('pitch_mean_abs_arcsec', float('nan')):.4f} / yaw {ref_best_pure.get('yaw_abs_mean_arcsec', float('nan')):.4f} / norm {ref_best_pure.get('norm_mean_arcsec', float('nan')):.4f}" if ref_best_pure else '- best pure-SCD sweep: N/A',
        f"- staged24: pitch {ref_staged.get('pitch_mean_abs_arcsec', float('nan')):.4f} / yaw {ref_staged.get('yaw_abs_mean_arcsec', float('nan')):.4f} / norm {ref_staged.get('norm_mean_arcsec', float('nan')):.4f}" if ref_staged else '- staged24: N/A',
        '',
        '| candidate | alpha | td | iter | row scope | col scope | pitch | yaw | norm | Δyaw vs best pure |',
        '|---|---:|---:|---:|---|---|---:|---:|---:|---:|',
    ]
    for row in by_yaw:
        c = row['candidate']
        s = row['statistics']
        lines.append(
            f"| {c['name']} | {c['alpha']:.3f} | {c['transition_duration_s']:.1f} | {c['apply_after_release_iter']} | {list(c['row_indices'])} | {list(c['col_indices'])} | {s['pitch_mean_abs_arcsec']:.4f} | {s['yaw_abs_mean_arcsec']:.4f} | {s['norm_mean_arcsec']:.4f} | {s['delta_vs_best_pure_yaw']:+.4f} |"
        )
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({
        'out_json': str(OUT_JSON),
        'best_by_yaw_mean': payload['rankings']['best_by_yaw_mean'],
        'best_balanced_pitch_safe': payload['rankings']['best_balanced_pitch_safe'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
