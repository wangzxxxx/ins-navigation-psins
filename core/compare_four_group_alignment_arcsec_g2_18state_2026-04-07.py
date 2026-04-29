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
ACC18_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_accel_colored_py_2026-03-30.py'
BASELINE_JSON = WORKSPACE / 'psins_method_bench' / 'results' / 'compare_four_group_alignment_arcsec_2026-04-05.json'

RESULTS_DIR = WORKSPACE / 'psins_method_bench' / 'results'
REPORTS_DIR = WORKSPACE / 'reports'
OUT_JSON = RESULTS_DIR / 'compare_four_group_alignment_arcsec_g2_18state_2026-04-07.json'
OUT_MD = REPORTS_DIR / 'psins_four_group_alignment_arcsec_g2_18state_2026-04-07.md'

MAX_WORKERS = min(4, os.cpu_count() or 1)
SEEDS = [0, 1, 2, 3, 4]
TS = 0.01
SINGLE_AXIS_ATT0_DEG = np.array([1.0, 0.0, 10.0])

_BASE12 = None
_ACC18 = None


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
        _BASE12 = load_module('g2_18state_base12_20260407', BASE12_PATH)
    return _BASE12



def load_acc18():
    global _ACC18
    if _ACC18 is None:
        _ACC18 = load_module('g2_18state_acc18_20260407', ACC18_PATH)
    return _ACC18



def build_single_axis_att(base12):
    att0 = SINGLE_AXIS_ATT0_DEG * base12.glv.deg
    paras = np.array([[1, 0, 0, 1, 3000 * base12.glv.deg, 300.0, 0.0, 0.0]], dtype=float)
    return att0, base12.attrottt(att0, paras, TS)



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



def run_single(seed: int) -> dict[str, Any]:
    base12 = load_base12()
    acc18 = load_acc18()

    np.random.seed(seed)

    pos0 = base12.posset(34, 116, 480, isdeg=1)
    att0_ref, att_truth = build_single_axis_att(base12)
    imu, _ = base12.avp2imu(att_truth, pos0)
    imuerr = base12.build_imuerr()  # keep truth dKg/dKa unchanged on purpose
    imu_noisy = base12.imuadderr(imu, imuerr)

    phi = np.array([0.1, 0.1, 0.5]) * base12.glv.deg
    att0_guess = base12.q2att(base12.qaddphi(base12.a2qua(att0_ref), phi))
    wvn = np.array([0.01, 0.01, 0.01])
    truth_att = att_truth[-1, 0:3]

    _, _, _, iter_logs = acc18.alignvn_18state_iter(
        imu=imu_noisy.copy(),
        qnb=att0_guess,
        pos=pos0,
        phi0=phi,
        imuerr=imuerr,
        wvn=wvn,
        max_iter=5,
        truth_att=truth_att,
        ng_sigma=np.array([0.05, 0.05, 0.05]) * base12.glv.dph,
        tau_g_s=np.array([300.0, 300.0, 300.0]),
        xa_sigma=np.array([0.01, 0.01, 0.01]) * base12.glv.ug,
        tau_a_s=np.array([100.0, 100.0, 100.0]),
        wash_scale=0.5,
        carry_att_seed=True,
    )
    last = extract_last_metrics(iter_logs[-1])
    err = np.array(last['att_err_arcsec'], dtype=float)
    return {
        'group_key': 'g2_markov_singleaxis_18state',
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
    g2_18 = summarize_rows(rows)

    g1 = baseline_groups['g1_plain12_singleaxis']
    g3 = baseline_groups['g3_markov_rotation']
    g4 = baseline_groups['g4_scd_rotation']
    old_g2 = baseline_groups['g2_markov_singleaxis']

    summary_rows = [
        build_summary_row('g1_plain12_singleaxis', 'G1 普通模型(12-state) @ 单轴旋转对准', g1),
        build_summary_row('g2_markov_singleaxis_18state', 'G2 18-state 随机误差模型 @ 单轴旋转对准', g2_18),
        build_summary_row('g3_markov_rotation', 'G3 Markov/GM-family plain24 @ 双轴旋转对准策略', g3),
        build_summary_row('g4_scd_rotation', 'G4 Markov + SCD @ 双轴旋转对准策略', g4),
    ]

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
                'improved': bool(delta > 0.0),
            })
        vals = [row[metric] for row in summary_rows]
        progression[metric] = {
            'strict_progression': all(step['improved'] for step in steps),
            'best_group': summary_rows[int(np.argmin(vals))]['group_key'],
            'steps': steps,
        }

    g2_delta_vs_old = {
        'pitch_mean_abs_arcsec_delta': float(old_g2['pitch_mean_abs_arcsec'] - g2_18['pitch_mean_abs_arcsec']),
        'yaw_abs_mean_arcsec_delta': float(old_g2['yaw_abs_mean_arcsec'] - g2_18['yaw_abs_mean_arcsec']),
        'norm_mean_arcsec_delta': float(old_g2['norm_mean_arcsec'] - g2_18['norm_mean_arcsec']),
        'pitch_improved': bool(g2_18['pitch_mean_abs_arcsec'] < old_g2['pitch_mean_abs_arcsec']),
        'yaw_improved': bool(g2_18['yaw_abs_mean_arcsec'] < old_g2['yaw_abs_mean_arcsec']),
        'norm_improved': bool(g2_18['norm_mean_arcsec'] < old_g2['norm_mean_arcsec']),
    }

    result = {
        'task': 'compare_four_group_alignment_arcsec_g2_18state_2026_04_07',
        'reference_json': str(BASELINE_JSON),
        'variation_note': (
            'G1 was unchanged and reused from the accepted four-group arcsec baseline. '
            'G2 was rerun as an 18-state model x=[phi(3), dv(3), eb(3), db(3), ng(3), xa(3)] without kg/ka states. '
            'Truth dKg/dKa injection was intentionally kept unchanged at 30 ppm to isolate state-structure / over-parameterization effects under the single-axis baseline. '
            'G3/G4 were reused from the accepted baseline.'
        ),
        'metric_definition': 'pitch mean abs / yaw abs mean / norm mean in arcsec',
        'seeds': SEEDS,
        'summary_rows': summary_rows,
        'progression': progression,
        'groups': {
            'g1_plain12_singleaxis': g1,
            'g2_markov_singleaxis_18state': g2_18,
            'g3_markov_rotation': g3,
            'g4_scd_rotation': g4,
        },
        'g2_old_plain24_reference': old_g2,
        'g2_delta_vs_old_plain24': g2_delta_vs_old,
        'g2_18state_config': {
            'state_layout': ['phi(3)', 'dv(3)', 'eb(3)', 'db(3)', 'ng(3)', 'xa(3)'],
            'ng_sigma_dph': [0.05, 0.05, 0.05],
            'tau_g_s': [300.0, 300.0, 300.0],
            'xa_sigma_ug': [0.01, 0.01, 0.01],
            'tau_a_s': [100.0, 100.0, 100.0],
            'max_iter': 5,
            'wash_scale': 0.5,
            'carry_att_seed': True,
            'single_axis_att0_deg': SINGLE_AXIS_ATT0_DEG.tolist(),
            'truth_scale_injection_kept': True,
        },
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = [
        '# 四组对准精度对比：把 G2 改成 18-state（去掉 scale states）',
        '',
        '- G1 保持原 12-state，不重定义；直接复用已接受的 baseline',
        '- G2 改成 18-state：`phi(3), dv(3), eb(3), db(3), ng(3), xa(3)`',
        '- 这轮**不改 truth dKg/dKa 注入**，就是要测单轴下 plain24 是否过参数化',
        '- G3/G4 继续复用已接受 baseline',
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
        '## G2（18-state）相对旧 G2（plain24）的变化',
        '',
        f"- pitch: {old_g2['pitch_mean_abs_arcsec']:.6f}\" -> {g2_18['pitch_mean_abs_arcsec']:.6f}\" (Δ={g2_delta_vs_old['pitch_mean_abs_arcsec_delta']:.6f}\")",
        f"- yaw: {old_g2['yaw_abs_mean_arcsec']:.6f}\" -> {g2_18['yaw_abs_mean_arcsec']:.6f}\" (Δ={g2_delta_vs_old['yaw_abs_mean_arcsec_delta']:.6f}\")",
        f"- norm: {old_g2['norm_mean_arcsec']:.6f}\" -> {g2_18['norm_mean_arcsec']:.6f}\" (Δ={g2_delta_vs_old['norm_mean_arcsec_delta']:.6f}\")",
        '',
        '## 读法',
        '',
        '- 如果 G2(18-state) 明显优于旧 G2(plain24)，说明单轴下确实存在 **scale states 过参数化/假学习** 问题。',
        '- 如果只改善很少，说明主因仍更偏向 **单轴轨迹整体激励不足**。',
    ])
    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
