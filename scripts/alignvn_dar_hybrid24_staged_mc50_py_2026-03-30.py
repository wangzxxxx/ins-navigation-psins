#!/usr/bin/env python3
"""MC50 runner for the staged DAR hybrid 24-state alignment variant.

This wrapper preserves the staged-24 semantics from:
  scripts/alignvn_dar_hybrid24_staged_py_2026-03-30.py
and only extends it to a 50-seed Monte Carlo with richer summary stats.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
BASE_SCRIPT = WORKSPACE / 'scripts' / 'alignvn_dar_hybrid24_staged_py_2026-03-30.py'
OUT_DIR = WORKSPACE / 'tmp' / 'alignment_strategy_sweep'
OUT_JSON = OUT_DIR / 'alignvn_dar_hybrid24_staged_mc50_result_2026-03-30.json'
OUT_MD = OUT_DIR / 'alignvn_dar_hybrid24_staged_mc50_summary_2026-03-30.md'
REF_5SEED = OUT_DIR / 'alignvn_dar_hybrid24_staged_result_2026-03-30.json'
REF_ACC18_MC50 = OUT_DIR / 'alignvn_dar_accel18_mc50_iter1_2026-03-30.json'
DEFAULT_WORKERS = min(4, os.cpu_count() or 1)

_BASE_MODULE = None


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_base_module():
    global _BASE_MODULE
    if _BASE_MODULE is None:
        _BASE_MODULE = load_module('alignvn_dar_hybrid24_staged_base_mc50_20260330', BASE_SCRIPT)
    return _BASE_MODULE


def build_cfg_dict(seed_start: int, seed_end: int) -> dict[str, Any]:
    return {
        'name': 'hybrid24_staged_iter5_mc50',
        'label': 'staged 24-state iter=5 mc50',
        'seeds': list(range(seed_start, seed_end + 1)),
        'max_iter': 5,
        'wash_scale': 0.5,
        'scale_wash_scale': 0.5,
        'carry_att_seed': True,
        'staged_release': True,
        'release_iter': 2,
        'rot_gate_dps': 5.0,
        'ng_sigma_dph': [0.05, 0.05, 0.05],
        'tau_g_s': [300.0, 300.0, 300.0],
        'xa_sigma_ug': [0.01, 0.01, 0.01],
        'tau_a_s': [100.0, 100.0, 100.0],
        'note': 'MC50 wrapper: keep iter1 frozen for kg/ka; release from iter2; high-rotation gate at 5 dps; same wash/carry-att defaults as the staged prototype.',
    }


def run_seed_worker(task: tuple[dict[str, Any], int]) -> dict[str, Any]:
    cfg_dict, seed = task
    base = load_base_module()
    cfg = base.Hybrid24Config(**cfg_dict)
    return base.run_single_seed((cfg, seed))


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def compute_statistics(errs: np.ndarray) -> dict[str, Any]:
    abs_errs = np.abs(errs)
    norms = np.linalg.norm(errs, axis=1)
    sigmas = errs.std(axis=0, ddof=1) if len(errs) > 1 else np.zeros(3)
    rms = np.sqrt(np.mean(errs ** 2, axis=0))
    within20 = (abs_errs < 20.0).mean(axis=0)
    pitch = errs[:, 1]
    yaw_abs = abs_errs[:, 2]
    return {
        'mean_signed_arcsec': errs.mean(axis=0).tolist(),
        'std_signed_arcsec_1sigma': sigmas.tolist(),
        'rms_arcsec': rms.tolist(),
        'mean_abs_arcsec': abs_errs.mean(axis=0).tolist(),
        'median_abs_arcsec': np.median(abs_errs, axis=0).tolist(),
        'within20_abs_rate': within20.tolist(),
        'norm_mean_arcsec': float(norms.mean()),
        'norm_median_arcsec': float(np.median(norms)),
        'yaw_abs_mean_arcsec': float(yaw_abs.mean()),
        'yaw_abs_median_arcsec': float(np.median(yaw_abs)),
        'pitch_signed_range_arcsec': [float(pitch.min()), float(pitch.max())],
        'max_abs_arcsec': abs_errs.max(axis=0).tolist(),
    }


def build_payload(per_seed: list[dict[str, Any]], cfg_dict: dict[str, Any], workers: int, elapsed_s: float) -> dict[str, Any]:
    errs = np.array([row['final_att_err_arcsec'] for row in per_seed], dtype=float)
    stats = compute_statistics(errs)
    sigmas = np.array(stats['std_signed_arcsec_1sigma'], dtype=float)
    rms = np.array(stats['rms_arcsec'], dtype=float)
    pitch_mean = float(stats['mean_signed_arcsec'][1])
    pitch_sigma = float(stats['std_signed_arcsec_1sigma'][1])
    pitch_rms = float(stats['rms_arcsec'][1])
    pitch_mean_abs = float(stats['mean_abs_arcsec'][1])
    pitch_median_abs = float(stats['median_abs_arcsec'][1])
    pitch_within20 = float(stats['within20_abs_rate'][1])
    pitch_range = stats['pitch_signed_range_arcsec']

    yaw_sigma_lt_20 = bool(sigmas[2] < 20.0)
    all_axis_sigma_lt_20 = bool(np.all(sigmas < 20.0))
    all_axis_rms_lt_20 = bool(np.all(rms < 20.0))
    pitch_near_zero_mc50 = bool(abs(pitch_mean) < 1.0 and pitch_rms < 1.0 and pitch_within20 == 1.0)

    if all_axis_sigma_lt_20:
        stability_note = 'MC50 shows low dispersion on all three axes; the staged variant is statistically stable under this truth/noise setting.'
    else:
        stability_note = 'MC50 still shows an axis with sigma above 20 arcsec, so the staged variant is not yet statistically tight enough.'

    if all_axis_rms_lt_20:
        target_note = 'It meets the 20-arcsec style target on RMS for all three axes.'
    else:
        target_note = 'It does not fully meet the 20-arcsec style target because at least one axis RMS is above 20 arcsec.'

    if pitch_near_zero_mc50:
        pitch_note = (
            f'Pitch stays near zero across MC50: mean={pitch_mean:.3f}\", sigma={pitch_sigma:.3f}\", '
            f'RMS={pitch_rms:.3f}\", mean|.|={pitch_mean_abs:.3f}\", median|.|={pitch_median_abs:.3f}\", '
            f'within20={pitch_within20:.1%}, range=[{pitch_range[0]:.3f}, {pitch_range[1]:.3f}]\".'
        )
    else:
        pitch_note = (
            f'Pitch is not convincingly near zero across MC50: mean={pitch_mean:.3f}\", sigma={pitch_sigma:.3f}\", '
            f'RMS={pitch_rms:.3f}\", mean|.|={pitch_mean_abs:.3f}\", median|.|={pitch_median_abs:.3f}\", '
            f'within20={pitch_within20:.1%}, range=[{pitch_range[0]:.3f}, {pitch_range[1]:.3f}]\".'
        )

    return {
        'method': 'hybrid24_staged_iter5_mc50',
        'source_script': str(BASE_SCRIPT),
        'mc': {
            'n_seeds': len(per_seed),
            'seed_range': [int(cfg_dict['seeds'][0]), int(cfg_dict['seeds'][-1])],
            'max_iter': int(cfg_dict['max_iter']),
            'staged_release': bool(cfg_dict['staged_release']),
            'release_iter': int(cfg_dict['release_iter']),
            'rot_gate_dps': float(cfg_dict['rot_gate_dps']),
            'workers': int(workers),
            'elapsed_s': float(elapsed_s),
        },
        'config': cfg_dict,
        'statistics': stats,
        'judgement': {
            'yaw_sigma_lt_20': yaw_sigma_lt_20,
            'all_axis_sigma_lt_20': all_axis_sigma_lt_20,
            'all_axis_rms_lt_20': all_axis_rms_lt_20,
            'statistically_stable': all_axis_sigma_lt_20,
            'meets_20arcsec_style_targets': all_axis_rms_lt_20,
        },
        'pitch_assessment': {
            'mean_signed_arcsec': pitch_mean,
            'std_signed_arcsec_1sigma': pitch_sigma,
            'rms_arcsec': pitch_rms,
            'mean_abs_arcsec': pitch_mean_abs,
            'median_abs_arcsec': pitch_median_abs,
            'within20_abs_rate': pitch_within20,
            'signed_range_arcsec': pitch_range,
            'near_zero_mc50': pitch_near_zero_mc50,
            'note': pitch_note,
        },
        'summary_notes': {
            'stability': stability_note,
            'target': target_note,
            'pitch': pitch_note,
        },
        'references': {
            'prior_5seed_staged': str(REF_5SEED) if REF_5SEED.exists() else None,
            'accel18_mc50_iter1': str(REF_ACC18_MC50) if REF_ACC18_MC50.exists() else None,
        },
        'per_seed': per_seed,
    }


def build_markdown(payload: dict[str, Any], ref_5seed: dict[str, Any] | None, ref_acc18: dict[str, Any] | None) -> str:
    st = payload['statistics']
    jd = payload['judgement']
    pitch = payload['pitch_assessment']

    def row(name: str, vals: list[float]) -> str:
        return f'| {name} | {vals[0]:.3f} | {vals[1]:.3f} | {vals[2]:.3f} |'

    lines = [
        '# Staged hybrid24 MC50 summary (2026-03-30)',
        '',
        '## Run setup',
        f"- variant: `{payload['method']}`",
        f"- source: `{payload['source_script']}`",
        '- semantics preserved: 24-state = phi/dv/eb/db/ng/xa + diagonal dKg(3)+dKa(3)',
        '- staged release preserved: iter1 freezes kg/ka; iter>=2 releases them',
        '- exact run knobs: max_iter=5, staged_release=true, release_iter=2, rot_gate_dps=5.0, wash/carry-att unchanged',
        f"- seeds: {payload['mc']['seed_range'][0]}..{payload['mc']['seed_range'][1]} (n={payload['mc']['n_seeds']})",
        f"- workers: {payload['mc']['workers']}, elapsed: {payload['mc']['elapsed_s']:.1f}s",
        '',
        '## Bottom line',
        f"- Statistical stability: **{'YES' if jd['statistically_stable'] else 'NO'}**",
        f"- Meets 20-arcsec style RMS target on all axes: **{'YES' if jd['meets_20arcsec_style_targets'] else 'NO'}**",
        f"- Pitch remains near zero over MC50: **{'YES' if pitch['near_zero_mc50'] else 'NO'}**",
        '',
        '## Axis statistics (arcsec)',
        '| metric | roll | pitch | yaw |',
        '|---|---:|---:|---:|',
        row('mean signed', st['mean_signed_arcsec']),
        row('std signed (1σ)', st['std_signed_arcsec_1sigma']),
        row('RMS', st['rms_arcsec']),
        row('mean abs', st['mean_abs_arcsec']),
        row('median abs', st['median_abs_arcsec']),
        row('within20 abs rate', [100.0 * x for x in st['within20_abs_rate']]),
        row('max abs', st['max_abs_arcsec']),
        '',
        '## Norm / target checks',
        f"- norm mean: {st['norm_mean_arcsec']:.3f}\"",
        f"- norm median: {st['norm_median_arcsec']:.3f}\"",
        f"- yaw abs mean / median: {st['yaw_abs_mean_arcsec']:.3f}\" / {st['yaw_abs_median_arcsec']:.3f}\"",
        f"- yaw_sigma_lt_20: `{jd['yaw_sigma_lt_20']}`",
        f"- all_axis_sigma_lt_20: `{jd['all_axis_sigma_lt_20']}`",
        f"- all_axis_rms_lt_20: `{jd['all_axis_rms_lt_20']}`",
        '',
        '## Pitch assessment',
        f"- {pitch['note']}",
    ]

    if ref_5seed is not None:
        staged5 = ref_5seed.get('staged24_iter5', {}).get('statistics', {})
        if staged5:
            lines.extend([
                '',
                '## Relation to prior 5-seed staged probe',
                f"- prior 5-seed pitch mean abs: {staged5.get('mean_abs_arcsec', [None, None, None])[1]:.3f}\"",
                f"- prior 5-seed yaw abs mean: {staged5.get('yaw_abs_mean_arcsec', float('nan')):.3f}\"",
                f"- prior 5-seed norm mean: {staged5.get('norm_mean_arcsec', float('nan')):.3f}\"",
                '- MC50 should be trusted over the 5-seed probe for stability judgement.',
            ])

    if ref_acc18 is not None:
        acc18_stats = ref_acc18.get('statistics', {})
        if acc18_stats:
            lines.extend([
                '',
                '## Reference vs accel18 MC50 iter1',
                f"- accel18 pitch mean abs: {acc18_stats.get('mean_abs_arcsec', [None, None, None])[1]:.3f}\"",
                f"- accel18 pitch RMS: {acc18_stats.get('rms_arcsec', [None, None, None])[1]:.3f}\"",
                f"- accel18 yaw abs mean: {acc18_stats.get('yaw_abs_mean_arcsec', float('nan')):.3f}\"",
                '- This staged24 run is mainly about whether pitch repair survives MC50 while keeping yaw statistically controlled.',
            ])

    lines.extend([
        '',
        '## Files',
        f'- json: `{OUT_JSON}`',
        f'- md: `{OUT_MD}`',
    ])
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser(description='Run MC50 for staged DAR hybrid24 iter=5.')
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--seed-end', type=int, default=49)
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    args = parser.parse_args()

    if args.seed_end < args.seed_start:
        raise SystemExit('--seed-end must be >= --seed-start')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg_dict = build_cfg_dict(args.seed_start, args.seed_end)
    tasks = [(cfg_dict, seed) for seed in cfg_dict['seeds']]
    workers = max(1, min(args.workers, len(tasks)))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        per_seed = list(ex.map(run_seed_worker, tasks))
    elapsed_s = time.time() - t0
    per_seed.sort(key=lambda row: row['seed'])

    payload = build_payload(per_seed, cfg_dict, workers, elapsed_s)
    ref_5seed = load_json_if_exists(REF_5SEED)
    ref_acc18 = load_json_if_exists(REF_ACC18_MC50)

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    OUT_MD.write_text(build_markdown(payload, ref_5seed, ref_acc18))

    print(json.dumps({
        'out_json': str(OUT_JSON),
        'out_md': str(OUT_MD),
        'judgement': payload['judgement'],
        'pitch_assessment': payload['pitch_assessment'],
        'statistics': payload['statistics'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
