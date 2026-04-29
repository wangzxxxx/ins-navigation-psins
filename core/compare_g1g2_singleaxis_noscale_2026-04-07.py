#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import is_dataclass
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
OUT_JSON = RESULTS_DIR / 'compare_g1g2_singleaxis_noscale_alignment_arcsec_2026-04-07.json'
OUT_MD = REPORTS_DIR / 'psins_g1g2_singleaxis_noscale_alignment_arcsec_2026-04-07.md'

MAX_WORKERS = min(4, os.cpu_count() or 1)
SEEDS = [0, 1, 2, 3, 4]
TS = 0.01
SINGLE_AXIS_ATT0_DEG = np.array([1.0, 0.0, 10.0])
GROUP_ORDER = ['g1_plain12_singleaxis_noscale', 'g2_markov_singleaxis_noscale']
GROUP_DISPLAY = {
    'g1_plain12_singleaxis_noscale': 'G1 普通模型(12-state) @ 单轴旋转对准（去掉 truth 比例误差）',
    'g2_markov_singleaxis_noscale': 'G2 GM-family plain24 @ 单轴旋转对准（去掉 truth 比例误差）',
}
BASELINE_GROUP_MAP = {
    'g1_plain12_singleaxis_noscale': 'g1_plain12_singleaxis',
    'g2_markov_singleaxis_noscale': 'g2_markov_singleaxis',
}

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
        _BASE12 = load_module('g1g2_noscale_base12_20260407', BASE12_PATH)
    return _BASE12



def load_h24():
    global _H24
    if _H24 is None:
        _H24 = load_module('g1g2_noscale_h24_20260407', H24_PATH)
    return _H24



def build_single_axis_att(acc18):
    att0 = SINGLE_AXIS_ATT0_DEG * acc18.glv.deg
    paras = np.array([[1, 0, 0, 1, 3000 * acc18.glv.deg, 300.0, 0.0, 0.0]], dtype=float)
    return acc18.attrottt(att0, paras, TS)



def build_imuerr_no_scale(base12):
    imuerr = base12.build_imuerr()
    imuerr = dict(imuerr)
    imuerr['dKg'] = np.zeros_like(imuerr['dKg'])
    imuerr['dKa'] = np.zeros_like(imuerr['dKa'])
    return imuerr



def extract_last_metrics(last: Any) -> dict[str, Any]:
    if is_dataclass(last):
        return {
            'att_err_arcsec': [float(x) for x in last.att_err_arcsec],
            'att_err_norm_arcsec': float(last.att_err_norm_arcsec),
            'yaw_abs_arcsec': float(last.yaw_abs_arcsec),
        }
    return {
        'att_err_arcsec': [float(x) for x in last['att_err_arcsec']],
        'att_err_norm_arcsec': float(last['att_err_norm_arcsec']),
        'yaw_abs_arcsec': float(last['yaw_abs_arcsec']),
    }



def run_single(task: tuple[str, int]) -> dict[str, Any]:
    group_key, seed = task
    base12 = load_base12()
    h24 = load_h24()
    acc18 = h24.load_acc18()

    np.random.seed(seed)

    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    att_truth = build_single_axis_att(acc18)
    att0_ref = SINGLE_AXIS_ATT0_DEG * acc18.glv.deg

    imu, _ = acc18.avp2imu(att_truth, pos0)
    imuerr = build_imuerr_no_scale(base12)
    imu_noisy = acc18.imuadderr(imu, imuerr)

    phi = np.array([0.1, 0.1, 0.5]) * acc18.glv.deg
    att0_guess = acc18.q2att(base12.qaddphi(acc18.a2qua(att0_ref), phi))
    wvn = np.array([0.01, 0.01, 0.01])
    truth_att = att_truth[-1, 0:3]

    if group_key == 'g1_plain12_singleaxis_noscale':
        _, _, _, iter_logs = base12.alignvn_12state_iter(
            imu=imu_noisy.copy(),
            qnb=att0_guess,
            pos=pos0,
            phi0=phi,
            imuerr=imuerr,
            wvn=wvn,
            max_iter=5,
            truth_att=truth_att,
        )
        last = extract_last_metrics(iter_logs[-1])
    elif group_key == 'g2_markov_singleaxis_noscale':
        cfg = h24.Hybrid24Config(
            name=group_key,
            label=group_key,
            seeds=[seed],
            max_iter=5,
            staged_release=False,
            rot_gate_dps=5.0,
            scale_wash_scale=0.5,
            note='single-axis G2 with zero truth dKg/dKa; scale states kept active to isolate mismatch effect',
        )
        _, iter_logs = h24.alignvn_24state_iter(
            imu=imu_noisy.copy(),
            qnb=att0_guess,
            pos=pos0,
            phi0=phi,
            imuerr=imuerr,
            wvn=wvn,
            cfg=cfg,
            truth_att=truth_att,
        )
        last = extract_last_metrics(iter_logs[-1])
    else:
        raise KeyError(group_key)

    err = np.array(last['att_err_arcsec'], dtype=float)
    return {
        'group_key': group_key,
        'seed': seed,
        'final_att_err_arcsec': [float(x) for x in err],
        'final_att_err_abs_arcsec': [float(x) for x in np.abs(err)],
        'final_att_err_norm_arcsec': float(last['att_err_norm_arcsec']),
        'final_yaw_abs_arcsec': float(last['yaw_abs_arcsec']),
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



def load_baseline_reference() -> dict[str, Any]:
    return json.loads(BASELINE_JSON.read_text(encoding='utf-8'))



def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline = load_baseline_reference()
    baseline_groups = baseline['groups']

    tasks = [(gk, seed) for gk in GROUP_ORDER for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as ex:
        out_rows = list(ex.map(run_single, tasks))

    grouped: dict[str, list[dict[str, Any]]] = {gk: [] for gk in GROUP_ORDER}
    for row in out_rows:
        grouped[row['group_key']].append(row)
    for gk in GROUP_ORDER:
        grouped[gk].sort(key=lambda x: x['seed'])

    groups = {}
    summary_rows = []
    deltas_vs_baseline = {}
    for gk in GROUP_ORDER:
        st = summarize_rows(grouped[gk])
        groups[gk] = st
        base_gk = BASELINE_GROUP_MAP[gk]
        ref = baseline_groups[base_gk]
        deltas_vs_baseline[gk] = {
            'baseline_group_key': base_gk,
            'pitch_mean_abs_arcsec_delta': float(ref['pitch_mean_abs_arcsec'] - st['pitch_mean_abs_arcsec']),
            'yaw_abs_mean_arcsec_delta': float(ref['yaw_abs_mean_arcsec'] - st['yaw_abs_mean_arcsec']),
            'norm_mean_arcsec_delta': float(ref['norm_mean_arcsec'] - st['norm_mean_arcsec']),
            'pitch_improved': bool(st['pitch_mean_abs_arcsec'] < ref['pitch_mean_abs_arcsec']),
            'yaw_improved': bool(st['yaw_abs_mean_arcsec'] < ref['yaw_abs_mean_arcsec']),
            'norm_improved': bool(st['norm_mean_arcsec'] < ref['norm_mean_arcsec']),
        }
        summary_rows.append({
            'group_key': gk,
            'display': GROUP_DISPLAY[gk],
            'pitch_mean_abs_arcsec': st['pitch_mean_abs_arcsec'],
            'yaw_abs_mean_arcsec': st['yaw_abs_mean_arcsec'],
            'norm_mean_arcsec': st['norm_mean_arcsec'],
            'yaw_abs_median_arcsec': st['yaw_abs_median_arcsec'],
            'yaw_abs_max_arcsec': st['yaw_abs_max_arcsec'],
            'baseline_pitch_mean_abs_arcsec': ref['pitch_mean_abs_arcsec'],
            'baseline_yaw_abs_mean_arcsec': ref['yaw_abs_mean_arcsec'],
            'baseline_norm_mean_arcsec': ref['norm_mean_arcsec'],
            'delta_pitch_mean_abs_arcsec': deltas_vs_baseline[gk]['pitch_mean_abs_arcsec_delta'],
            'delta_yaw_abs_mean_arcsec': deltas_vs_baseline[gk]['yaw_abs_mean_arcsec_delta'],
            'delta_norm_mean_arcsec': deltas_vs_baseline[gk]['norm_mean_arcsec_delta'],
        })

    result = {
        'task': 'compare_g1g2_singleaxis_noscale_alignment_arcsec_2026_04_07',
        'reference_json': str(BASELINE_JSON),
        'variation_note': 'Only G1/G2 were rerun. Truth injection dKg/dKa were set to zero. eb/db/web/wdb, seeds, single-axis trajectory, phi0, and filter configs were kept unchanged. G2 still keeps scale states active.',
        'metric_definition': 'pitch mean abs / yaw abs mean / norm mean in arcsec',
        'seeds': SEEDS,
        'summary_rows': summary_rows,
        'groups': groups,
        'deltas_vs_baseline': deltas_vs_baseline,
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# G1/G2 单轴对准：去掉 truth 比例误差后的复跑',
        '',
        '- 只重跑 G1/G2 单轴 baseline',
        '- 只改 truth 注入：`dKg = 0`, `dKa = 0`',
        '- `eb/db/web/wdb`、单轴轨迹、`phi0`、seeds、滤波配置保持不变',
        '- G2 仍保留 scale states，用于隔离“truth scale mismatch”影响',
        '',
        '| 组别 | pitch new (") | yaw new (") | norm new (") | pitch baseline (") | yaw baseline (") | norm baseline (") | Δpitch (") | Δyaw (") | Δnorm (") |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['display']} | {row['pitch_mean_abs_arcsec']:.6f} | {row['yaw_abs_mean_arcsec']:.6f} | {row['norm_mean_arcsec']:.6f} | {row['baseline_pitch_mean_abs_arcsec']:.6f} | {row['baseline_yaw_abs_mean_arcsec']:.6f} | {row['baseline_norm_mean_arcsec']:.6f} | {row['delta_pitch_mean_abs_arcsec']:.6f} | {row['delta_yaw_abs_mean_arcsec']:.6f} | {row['delta_norm_mean_arcsec']:.6f} |"
        )
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
