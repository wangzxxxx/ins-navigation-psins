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
PREV_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'ch4_pure_scd_vs_freeze_2026-04-03.json'
OUT_DIR = WORKSPACE / 'tmp' / 'alignment_strategy_sweep'
OUT_JSON = OUT_DIR / 'ch4_pure_scd_param_sweep_2026-04-03.json'
OUT_MD = OUT_DIR / 'ch4_pure_scd_param_sweep_2026-04-03.md'
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


cmp = load_module('compare_ch4_pure_scd_mod_20260403', PURE_SCD_COMPARE)
acc18 = cmp.acc18
base12 = cmp.base12
h24 = cmp.h24


@dataclass(frozen=True)
class Candidate:
    name: str
    alpha: float
    transition_duration_s: float
    apply_after_release_iter: int
    core_slice: tuple[int, int] = (0, 18)
    scale_slice: tuple[int, int] = (18, 24)
    note: str = ''


def build_candidates() -> list[Candidate]:
    return [
        Candidate('base_a999_td2_i1', 0.999, 2.0, 1, note='current pure-SCD baseline'),
        Candidate('weak_a9995_td2_i1', 0.9995, 2.0, 1, note='weaker covariance cut'),
        Candidate('strong_a998_td2_i1', 0.998, 2.0, 1, note='slightly stronger covariance cut'),
        Candidate('hard_a995_td2_i1', 0.995, 2.0, 1, note='much stronger covariance cut'),
        Candidate('early_a999_td1_i1', 0.999, 1.0, 1, note='same alpha, earlier after stop'),
        Candidate('late_a999_td4_i1', 0.999, 4.0, 1, note='same alpha, later after stop'),
        Candidate('delay_i2_a999_td2', 0.999, 2.0, 2, note='do not apply during iter1'),
        Candidate('delay_i2_a998_td2', 0.998, 2.0, 2, note='iter2-only with slightly stronger covariance cut'),
    ]


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

    cfg_scd = h24.Hybrid24Config(
        name=candidate.name,
        label=candidate.name,
        seeds=[seed],
        max_iter=5,
        staged_release=False,
        rot_gate_dps=5.0,
        scale_wash_scale=0.5,
    )
    scd_cfg = cmp.SCDConfig(
        enabled=True,
        alpha=candidate.alpha,
        transition_duration_s=candidate.transition_duration_s,
        core_slice=candidate.core_slice,
        scale_slice=candidate.scale_slice,
        apply_after_release_iter=candidate.apply_after_release_iter,
        note=candidate.note,
    )

    _, logs = cmp.alignvn_24state_iter_pure_scd(
        imu_noisy.copy(),
        att0_guess,
        pos0,
        phi,
        imuerr,
        wvn,
        cfg_scd,
        truth_att,
        scd_cfg,
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
    candidates = build_candidates()
    tasks = [(cand, seed) for cand in candidates for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as ex:
        rows = list(ex.map(run_candidate_seed, tasks))

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row['candidate']['name'], []).append(row)
    for k in grouped:
        grouped[k].sort(key=lambda x: x['seed'])

    prev = json.loads(PREV_JSON.read_text()) if PREV_JSON.exists() else {}
    plain = prev.get('plain24', {})
    staged = prev.get('staged24', {})
    baseline_pure = prev.get('pure_scd24', {})

    summary_rows = []
    for cand in candidates:
        stats = summarize_rows(grouped[cand.name])
        stats['delta_vs_prev_pure_yaw_mean'] = float(stats['yaw_abs_mean_arcsec'] - baseline_pure.get('yaw_abs_mean_arcsec', 0.0)) if baseline_pure else None
        stats['delta_vs_prev_pure_pitch_mean'] = float(stats['pitch_mean_abs_arcsec'] - baseline_pure.get('pitch_mean_abs_arcsec', 0.0)) if baseline_pure else None
        stats['delta_vs_prev_pure_norm_mean'] = float(stats['norm_mean_arcsec'] - baseline_pure.get('norm_mean_arcsec', 0.0)) if baseline_pure else None
        summary_rows.append({
            'candidate': asdict(cand),
            'statistics': stats,
            'per_seed': grouped[cand.name],
        })

    by_yaw = sorted(summary_rows, key=lambda x: (x['statistics']['yaw_abs_mean_arcsec'], x['statistics']['norm_mean_arcsec']))
    by_norm = sorted(summary_rows, key=lambda x: (x['statistics']['norm_mean_arcsec'], x['statistics']['yaw_abs_mean_arcsec']))
    pitch_safe = [r for r in summary_rows if r['statistics']['pitch_mean_abs_arcsec'] <= baseline_pure.get('pitch_mean_abs_arcsec', 1e9) + 0.02]
    by_balanced = sorted(pitch_safe or summary_rows, key=lambda x: (x['statistics']['norm_mean_arcsec'], x['statistics']['yaw_abs_mean_arcsec']))

    payload = {
        'reference': {
            'plain24': plain,
            'staged24': staged,
            'prev_pure_scd24': baseline_pure,
        },
        'candidates': summary_rows,
        'rankings': {
            'best_by_yaw_mean': by_yaw[0]['candidate']['name'] if by_yaw else None,
            'best_by_norm_mean': by_norm[0]['candidate']['name'] if by_norm else None,
            'best_balanced_pitch_safe': by_balanced[0]['candidate']['name'] if by_balanced else None,
        },
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# Chapter 4 pure-SCD parameter sweep (2026-04-03)',
        '',
        '- same path / same seeds / same noisy IMU per seed',
        '- only pure-SCD parameters are changed',
        '',
        '## Reference',
        f"- plain24: pitch {plain.get('pitch_mean_abs_arcsec', float('nan')):.3f} / yaw {plain.get('yaw_abs_mean_arcsec', float('nan')):.3f} / norm {plain.get('norm_mean_arcsec', float('nan')):.3f}",
        f"- staged24: pitch {staged.get('pitch_mean_abs_arcsec', float('nan')):.3f} / yaw {staged.get('yaw_abs_mean_arcsec', float('nan')):.3f} / norm {staged.get('norm_mean_arcsec', float('nan')):.3f}",
        f"- prev pure_scd24: pitch {baseline_pure.get('pitch_mean_abs_arcsec', float('nan')):.3f} / yaw {baseline_pure.get('yaw_abs_mean_arcsec', float('nan')):.3f} / norm {baseline_pure.get('norm_mean_arcsec', float('nan')):.3f}",
        '',
        '| candidate | alpha | td(s) | first_iter | pitch | yaw | norm | Δyaw vs prev | Δpitch vs prev |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in sorted(summary_rows, key=lambda x: (x['statistics']['yaw_abs_mean_arcsec'], x['statistics']['norm_mean_arcsec'])):
        c = row['candidate']
        s = row['statistics']
        lines.append(
            f"| {c['name']} | {c['alpha']:.4f} | {c['transition_duration_s']:.1f} | {c['apply_after_release_iter']} | {s['pitch_mean_abs_arcsec']:.3f} | {s['yaw_abs_mean_arcsec']:.3f} | {s['norm_mean_arcsec']:.3f} | {s['delta_vs_prev_pure_yaw_mean']:+.3f} | {s['delta_vs_prev_pure_pitch_mean']:+.3f} |"
        )
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({
        'out_json': str(OUT_JSON),
        'best_by_yaw_mean': payload['rankings']['best_by_yaw_mean'],
        'best_by_norm_mean': payload['rankings']['best_by_norm_mean'],
        'best_balanced_pitch_safe': payload['rankings']['best_balanced_pitch_safe'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
