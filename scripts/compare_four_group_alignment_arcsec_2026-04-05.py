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
PURE_SCD_PATH = WORKSPACE / 'scripts' / 'compare_ch4_pure_scd_vs_freeze_2026-04-03.py'

RESULTS_DIR = WORKSPACE / 'psins_method_bench' / 'results'
REPORTS_DIR = WORKSPACE / 'reports'
OUT_JSON = RESULTS_DIR / 'compare_four_group_alignment_arcsec_iter1_2026-04-05.json'
OUT_MD = REPORTS_DIR / 'psins_four_group_alignment_arcsec_iter1_2026-04-05.md'

MAX_WORKERS = min(4, os.cpu_count() or 1)
SEEDS = [0, 1, 2, 3, 4]
TS = 0.01

GROUP_ORDER = ['g1_plain12_singleaxis', 'g2_markov_singleaxis', 'g3_markov_rotation', 'g4_scd_rotation']
GROUP_DISPLAY = {
    'g1_plain12_singleaxis': 'G1 普通模型(12-state) @ 单轴旋转对准',
    'g2_markov_singleaxis': 'G2 Markov/GM-family plain24 @ 单轴旋转对准',
    'g3_markov_rotation': 'G3 Markov/GM-family plain24 @ 双轴旋转对准策略',
    'g4_scd_rotation': 'G4 Markov + SCD @ 双轴旋转对准策略',
}

_BASE12 = None
_H24 = None
_PURE = None


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
        _BASE12 = load_module('four_group_alignment_base12_20260405', BASE12_PATH)
    return _BASE12


def load_h24():
    global _H24
    if _H24 is None:
        _H24 = load_module('four_group_alignment_h24_20260405', H24_PATH)
    return _H24


def load_pure():
    global _PURE
    if _PURE is None:
        _PURE = load_module('four_group_alignment_pure_20260405', PURE_SCD_PATH)
    return _PURE


def build_single_axis_att(acc18):
    att0 = np.array([1.0, 0.0, 10.0]) * acc18.glv.deg
    paras = np.array([[1, 0, 0, 1, 3000 * acc18.glv.deg, 300.0, 0.0, 0.0]], dtype=float)
    return att0, acc18.attrottt(att0, paras, TS)


def build_dual_axis_att(base12, acc18):
    att0 = np.array([0.0, 0.0, 0.0])
    paras = base12.build_rot_paras()
    return att0, acc18.attrottt(att0, paras, TS)


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
    pure = load_pure()
    acc18 = h24.load_acc18()

    np.random.seed(seed)

    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    if group_key in ('g1_plain12_singleaxis', 'g2_markov_singleaxis'):
        att0_ref, att_truth = build_single_axis_att(acc18)
    else:
        att0_ref, att_truth = build_dual_axis_att(base12, acc18)

    imu, _ = acc18.avp2imu(att_truth, pos0)
    imuerr = base12.build_imuerr()
    imu_noisy = acc18.imuadderr(imu, imuerr)

    phi = np.array([0.1, 0.1, 0.5]) * acc18.glv.deg
    att0_guess = acc18.q2att(base12.qaddphi(acc18.a2qua(att0_ref), phi))
    wvn = np.array([0.01, 0.01, 0.01])
    truth_att = att_truth[-1, 0:3]

    if group_key == 'g1_plain12_singleaxis':
        _, _, _, iter_logs = base12.alignvn_12state_iter(
            imu=imu_noisy.copy(),
            qnb=att0_guess,
            pos=pos0,
            phi0=phi,
            imuerr=imuerr,
            wvn=wvn,
            max_iter=1,
            truth_att=truth_att,
        )
        last = extract_last_metrics(iter_logs[-1])
    elif group_key in ('g2_markov_singleaxis', 'g3_markov_rotation'):
        cfg = h24.Hybrid24Config(
            name=group_key,
            label=group_key,
            seeds=[seed],
            max_iter=1,
            staged_release=False,
            rot_gate_dps=5.0,
            scale_wash_scale=0.5,
            note='plain24 baseline on the old working alignment core (iter=1 unified semantic)',
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
    elif group_key == 'g4_scd_rotation':
        cfg = h24.Hybrid24Config(
            name=group_key,
            label=group_key,
            seeds=[seed],
            max_iter=1,
            staged_release=False,
            rot_gate_dps=5.0,
            scale_wash_scale=0.5,
            note='best manual pure-SCD config on the old working alignment core (iter=1 unified semantic)',
        )
        scd_cfg = pure.SCDConfig(
            enabled=True,
            alpha=0.995,
            transition_duration_s=2.0,
            apply_after_release_iter=1,
            note='hard_a995_td2_i1',
        )
        _, iter_logs = pure.alignvn_24state_iter_pure_scd(
            imu=imu_noisy.copy(),
            qnb=att0_guess,
            pos=pos0,
            phi0=phi,
            imuerr=imuerr,
            wvn=wvn,
            cfg=cfg,
            truth_att=truth_att,
            scd=scd_cfg,
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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    tasks = [(gk, seed) for gk in GROUP_ORDER for seed in SEEDS]
    with ProcessPoolExecutor(max_workers=min(MAX_WORKERS, len(tasks))) as ex:
        out_rows = list(ex.map(run_single, tasks))

    grouped: dict[str, list[dict[str, Any]]] = {gk: [] for gk in GROUP_ORDER}
    for row in out_rows:
        grouped[row['group_key']].append(row)
    for gk in GROUP_ORDER:
        grouped[gk].sort(key=lambda x: x['seed'])

    summary_rows = []
    groups = {}
    for gk in GROUP_ORDER:
        st = summarize_rows(grouped[gk])
        groups[gk] = st
        summary_rows.append({
            'group_key': gk,
            'display': GROUP_DISPLAY[gk],
            'pitch_mean_abs_arcsec': st['pitch_mean_abs_arcsec'],
            'yaw_abs_mean_arcsec': st['yaw_abs_mean_arcsec'],
            'norm_mean_arcsec': st['norm_mean_arcsec'],
            'yaw_abs_median_arcsec': st['yaw_abs_median_arcsec'],
            'yaw_abs_max_arcsec': st['yaw_abs_max_arcsec'],
        })

    progression = {}
    for metric in ['pitch_mean_abs_arcsec', 'yaw_abs_mean_arcsec', 'norm_mean_arcsec']:
        steps = []
        for i in range(1, len(summary_rows)):
            prev = summary_rows[i - 1]
            cur = summary_rows[i]
            delta = float(prev[metric] - cur[metric])
            steps.append({
                'from_group': prev['group_key'],
                'to_group': cur['group_key'],
                'delta': delta,
                'improved': bool(delta > 0),
            })
        progression[metric] = {
            'strict_progression': all(s['improved'] for s in steps),
            'best_group': min(summary_rows, key=lambda x: x[metric])['group_key'],
            'steps': steps,
        }

    report = []
    report.append('# 四组实验：严格按对准角误差口径 (arcsec)')
    report.append('')
    report.append('- 评估：最终姿态误差 `pitch mean abs / yaw abs mean / norm mean`，单位 arcsec')
    report.append('- G1/G2 baseline：用户给定单轴旋转 alignvn baseline（att0=[1,0,10]deg，z轴 3000deg / 300s）')
    report.append('- G3/G4 rotation strategy：旧第四章双轴旋转对准策略（`build_rot_paras()`）')
    report.append('- G4 使用 best manual pure-SCD setting：`alpha=0.995, transition_duration=2.0s, iter1`')
    report.append('- 统一评估语义：四组全部按 **单次对准 (`iter=1`)** 统计，不再混入多轮 wash')
    report.append('- seeds = [0,1,2,3,4]')
    report.append('')
    report.append('| 组别 | pitch mean abs (\") | yaw abs mean (\") | norm mean (\") | yaw median (\") | yaw max (\") |')
    report.append('|---|---:|---:|---:|---:|---:|')
    for row in summary_rows:
        report.append(
            f"| {row['display']} | {row['pitch_mean_abs_arcsec']:.3f} | {row['yaw_abs_mean_arcsec']:.3f} | {row['norm_mean_arcsec']:.3f} | {row['yaw_abs_median_arcsec']:.3f} | {row['yaw_abs_max_arcsec']:.3f} |"
        )
    report.append('')
    report.append('## 递进判断')
    report.append('')
    for metric, label in [
        ('pitch_mean_abs_arcsec', 'pitch mean abs'),
        ('yaw_abs_mean_arcsec', 'yaw abs mean'),
        ('norm_mean_arcsec', 'norm mean'),
    ]:
        item = progression[metric]
        verdict = '严格递进' if item['strict_progression'] else '非严格单调'
        report.append(f"- **{label}**：{verdict}；best = {GROUP_DISPLAY[item['best_group']]}")
        for step in item['steps']:
            tag = '改善' if step['improved'] else '退化'
            report.append(f"  - {GROUP_DISPLAY[step['from_group']]} -> {GROUP_DISPLAY[step['to_group']]}：{tag} {step['delta']:+.3f}\"")
    report.append('')

    payload = {
        'task': 'four_group_alignment_arcsec_iter1_2026_04_05',
        'metric_definition': 'pitch mean abs / yaw abs mean / norm mean in arcsec',
        'group_mapping_note': 'G1=12-state plain alignvn on single-axis baseline; G2=24-state plain24 on same single-axis baseline; G3=24-state plain24 on old Chapter-4 dual-axis rotation strategy; G4=24-state pure-SCD on the same dual-axis rotation strategy',
        'seeds': SEEDS,
        'summary_rows': summary_rows,
        'progression': progression,
        'groups': groups,
        'files': {
            'json': str(OUT_JSON),
            'md': str(OUT_MD),
        },
    }

    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    OUT_MD.write_text('\n'.join(report) + '\n', encoding='utf-8')

    print(json.dumps({'json': str(OUT_JSON), 'md': str(OUT_MD), 'summary_rows': summary_rows}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
