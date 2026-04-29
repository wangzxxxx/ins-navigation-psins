#!/usr/bin/env python3
from __future__ import annotations

"""Rerun the original four alignment groups with the pure param-outer method.

Default method choice follows the latest stable single-axis boundary-confirmation sweep:
- feedback_scale = 2.2
- outer_iterations = 15

Per outer round:
- start from the SAME initial attitude seed (no carry)
- run exactly one inner pass
- do NO inner wash
- accumulate eb/db estimates and apply them back to the raw IMU before next round

Important state-slice note used here:
- G1 12-state: x = [phi(3), dv(3), eb(3), db(3)]
- G2/G3/G4 24-state: x = [phi(3), dv(3), eb(3), db(3), ng(3), xa(3), kg(3), ka(3)]
  so eb = x[6:9], db = x[9:12]
"""

import argparse
import importlib.util
import json
import math
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
BASELINE_JSON = WORKSPACE / 'psins_method_bench' / 'results' / 'compare_four_group_alignment_arcsec_2026-04-05.json'
RESULTS_DIR = WORKSPACE / 'psins_method_bench' / 'results'
REPORTS_DIR = WORKSPACE / 'reports'

DEFAULT_OUTER_ITERATIONS = 15
DEFAULT_FEEDBACK_SCALE = 2.2
DEFAULT_SEEDS = [0, 1, 2, 3, 4]
TS = 0.01
WVN = np.array([0.01, 0.01, 0.01])
PHI_DEG = np.array([0.1, 0.1, 0.5])
DATE_TAG = '2026-04-07'

GROUP_ORDER = [
    'g1_plain12_singleaxis',
    'g2_markov_singleaxis',
    'g3_markov_rotation',
    'g4_scd_rotation',
]
GROUP_DISPLAY = {
    'g1_plain12_singleaxis': 'G1 普通模型(12-state) @ 单轴旋转对准',
    'g2_markov_singleaxis': 'G2 Markov/GM-family plain24 @ 单轴旋转对准',
    'g3_markov_rotation': 'G3 Markov/GM-family plain24 @ 双轴旋转对准策略',
    'g4_scd_rotation': 'G4 Markov + SCD @ 双轴旋转对准策略',
}
GROUP_FAMILY = {
    'g1_plain12_singleaxis': 'singleaxis',
    'g2_markov_singleaxis': 'singleaxis',
    'g3_markov_rotation': 'rotation',
    'g4_scd_rotation': 'rotation',
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
        _BASE12 = load_module('four_group_param_outer_base12_20260407', BASE12_PATH)
    return _BASE12



def load_h24():
    global _H24
    if _H24 is None:
        _H24 = load_module('four_group_param_outer_h24_20260407', H24_PATH)
    return _H24



def load_pure():
    global _PURE
    if _PURE is None:
        _PURE = load_module('four_group_param_outer_pure_20260407', PURE_SCD_PATH)
    return _PURE



def sanitize_float_tag(x: float) -> str:
    s = f'{x:.6g}'
    s = s.replace('-', 'm').replace('.', 'p')
    return s



def parse_int_list(text: str) -> list[int]:
    if not text.strip():
        return []
    return [int(x.strip()) for x in text.split(',') if x.strip()]



def parse_group_list(text: str) -> list[str]:
    if not text.strip():
        return []
    groups = [x.strip() for x in text.split(',') if x.strip()]
    unknown = [g for g in groups if g not in GROUP_ORDER]
    if unknown:
        raise ValueError(f'unknown groups: {unknown}')
    return groups



def build_single_axis_att(acc18):
    att0 = np.array([1.0, 0.0, 10.0]) * acc18.glv.deg
    paras = np.array([[1, 0, 0, 1, 3000 * acc18.glv.deg, 300.0, 0.0, 0.0]], dtype=float)
    return att0, acc18.attrottt(att0, paras, TS)



def build_dual_axis_att(base12, acc18):
    att0 = np.array([0.0, 0.0, 0.0])
    paras = base12.build_rot_paras()
    return att0, acc18.attrottt(att0, paras, TS)



def extract_last_metrics_generic(last: Any) -> dict[str, Any]:
    if is_dataclass(last):
        return {
            'att_err_arcsec': [float(x) for x in last.att_err_arcsec],
            'att_err_norm_arcsec': float(last.att_err_norm_arcsec),
            'yaw_abs_arcsec': float(last.yaw_abs_arcsec),
            'final_att_deg': [float(x) for x in last.final_att_deg],
            'est_eb_dph': [float(x) for x in last.est_eb_dph],
            'est_db_ug': [float(x) for x in last.est_db_ug],
        }
    return {
        'att_err_arcsec': [float(x) for x in last['att_err_arcsec']],
        'att_err_norm_arcsec': float(last['att_err_norm_arcsec']),
        'yaw_abs_arcsec': float(last['yaw_abs_arcsec']),
        'final_att_deg': [float(x) for x in last['final_att_deg']],
        'est_eb_dph': [float(x) for x in last['est_eb_dph']],
        'est_db_ug': [float(x) for x in last['est_db_ug']],
    }



def run_g1_one_pass(base12, imu_corr: np.ndarray, att0_guess: np.ndarray, pos0: np.ndarray,
                    phi: np.ndarray, imuerr: dict[str, np.ndarray], truth_att: np.ndarray) -> dict[str, Any]:
    _, _, _, iter_logs = base12.alignvn_12state_iter(
        imu=imu_corr,
        qnb=att0_guess,
        pos=pos0,
        phi0=phi,
        imuerr=imuerr,
        wvn=WVN.copy(),
        max_iter=1,
        truth_att=truth_att,
        wash_scale=0.0,
        carry_att_seed=False,
    )
    return extract_last_metrics_generic(iter_logs[-1])



def run_24state_one_pass(acc18, h24, pure, imu_corr: np.ndarray, att0_guess: np.ndarray, pos0: np.ndarray,
                         phi: np.ndarray, imuerr: dict[str, np.ndarray], truth_att: np.ndarray,
                         *, pure_scd: bool, scd_cfg: Any | None) -> dict[str, Any]:
    """One-pass 24-state runner that exposes eb/db slices explicitly.

    This mirrors the accepted plain24 / pure-SCD alignment cores, but returns the
    hidden eb/db estimates needed by the outer bias-feedback loop.
    """
    glv = acc18.glv
    nn = 2
    ts = float(imu_corr[1, -1] - imu_corr[0, -1])
    nts = nn * ts
    qnb_seed = acc18.a2qua(att0_guess) if len(att0_guess) == 3 else np.asarray(att0_guess).reshape(4)
    length = (len(imu_corr) // nn) * nn
    imu_corr = imu_corr[:length]

    eth = acc18.Earth(pos0)
    cnn = acc18.rv2m(-eth.wnie * nts / 2)
    rot_gate_rad = 5.0 * glv.deg

    kf = h24.avnkfinit_24(
        nts,
        pos0,
        phi,
        imuerr,
        WVN.copy(),
        np.array([0.05, 0.05, 0.05]) * glv.dph,
        np.array([300.0, 300.0, 300.0]),
        np.array([0.01, 0.01, 0.01]) * glv.ug,
        np.array([100.0, 100.0, 100.0]),
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
            if pure_scd:
                time_since_rot_stop = 0.0
                scd_applied_this_phase = False
        else:
            phi_k[0:3, 18:21] = 0.0
            phi_k[3:6, 21:24] = 0.0
            if pure_scd:
                time_since_rot_stop += nts

        kf['Phikk_1'] = phi_k
        kf = acc18.kfupdate(kf, vn)

        qnbi = acc18.qdelphi(qnbi, 0.91 * kf['xk'][0:3])
        kf['xk'][0:3] *= 0.09
        vn = vn - 0.91 * kf['xk'][3:6]
        kf['xk'][3:6] *= 0.09

        if pure_scd and scd_cfg is not None and scd_cfg.enabled and (not high_rot):
            if (time_since_rot_stop >= scd_cfg.transition_duration_s) and (not scd_applied_this_phase):
                kf = pure.apply_scd_once(kf, scd_cfg)
                scd_applied_this_phase = True

    final_att = acc18.q2att(qnbi)
    att_err_arcsec = acc18.qq2phi(acc18.a2qua(final_att), acc18.a2qua(truth_att)) / glv.sec
    return {
        'final_att_deg': (final_att / glv.deg).tolist(),
        'att_err_arcsec': [float(x) for x in att_err_arcsec],
        'att_err_norm_arcsec': float(np.linalg.norm(att_err_arcsec)),
        'yaw_abs_arcsec': float(abs(att_err_arcsec[2])),
        'est_eb_dph': (kf['xk'][6:9] / glv.dph).tolist(),
        'est_db_ug': (kf['xk'][9:12] / glv.ug).tolist(),
        'est_kg_ppm': (kf['xk'][18:21] / glv.ppm).tolist(),
        'est_ka_ppm': (kf['xk'][21:24] / glv.ppm).tolist(),
    }



def run_single(task: tuple[str, int, int, float]) -> dict[str, Any]:
    group_key, seed, outer_iterations, feedback_scale = task
    base12 = load_base12()
    h24 = load_h24()
    pure = load_pure()
    acc18 = h24.load_acc18()

    np.random.seed(seed)

    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    if GROUP_FAMILY[group_key] == 'singleaxis':
        att0_ref, att_truth = build_single_axis_att(acc18)
    else:
        att0_ref, att_truth = build_dual_axis_att(base12, acc18)

    imu, _ = acc18.avp2imu(att_truth, pos0)
    imuerr = base12.build_imuerr()
    imu_noisy = acc18.imuadderr(imu, imuerr)

    phi = PHI_DEG * acc18.glv.deg
    att0_guess = acc18.q2att(base12.qaddphi(acc18.a2qua(att0_ref), phi))
    truth_att = att_truth[-1, 0:3]

    eb_accum = np.zeros(3)
    db_accum = np.zeros(3)
    outer_logs: list[dict[str, Any]] = []

    for outer_it in range(1, outer_iterations + 1):
        imu_corr = imu_noisy.copy()
        imu_corr[:, 0:3] -= eb_accum * TS
        imu_corr[:, 3:6] -= db_accum * TS

        if group_key == 'g1_plain12_singleaxis':
            result = run_g1_one_pass(base12, imu_corr, att0_guess, pos0, phi, imuerr, truth_att)
        elif group_key in ('g2_markov_singleaxis', 'g3_markov_rotation'):
            result = run_24state_one_pass(
                acc18,
                h24,
                pure,
                imu_corr,
                att0_guess,
                pos0,
                phi,
                imuerr,
                truth_att,
                pure_scd=False,
                scd_cfg=None,
            )
        elif group_key == 'g4_scd_rotation':
            scd_cfg = pure.SCDConfig(
                enabled=True,
                alpha=0.995,
                transition_duration_s=2.0,
                apply_after_release_iter=1,
                note='hard_a995_td2_i1',
            )
            result = run_24state_one_pass(
                acc18,
                h24,
                pure,
                imu_corr,
                att0_guess,
                pos0,
                phi,
                imuerr,
                truth_att,
                pure_scd=True,
                scd_cfg=scd_cfg,
            )
        else:
            raise KeyError(group_key)

        eb_est = np.array(result['est_eb_dph'], dtype=float) * acc18.glv.dph
        db_est = np.array(result['est_db_ug'], dtype=float) * acc18.glv.ug
        eb_accum = eb_accum + feedback_scale * eb_est
        db_accum = db_accum + feedback_scale * db_est

        err = np.array(result['att_err_arcsec'], dtype=float)
        outer_logs.append({
            'outer_iteration': outer_it,
            'final_att_deg': [float(x) for x in result['final_att_deg']],
            'final_att_err_arcsec': [float(x) for x in err],
            'final_att_err_abs_arcsec': [float(x) for x in np.abs(err)],
            'final_att_err_norm_arcsec': float(result['att_err_norm_arcsec']),
            'final_yaw_abs_arcsec': float(result['yaw_abs_arcsec']),
            'est_eb_dph': [float(x) for x in result['est_eb_dph']],
            'est_db_ug': [float(x) for x in result['est_db_ug']],
            'accum_eb_dph': (eb_accum / acc18.glv.dph).tolist(),
            'accum_db_ug': (db_accum / acc18.glv.ug).tolist(),
            'est_kg_ppm': [float(x) for x in result.get('est_kg_ppm', [])],
            'est_ka_ppm': [float(x) for x in result.get('est_ka_ppm', [])],
        })

    return {
        'group_key': group_key,
        'seed': seed,
        'outer_logs': outer_logs,
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



def build_summary_row(group_key: str, display: str, st: dict[str, Any], best_yaw_it: int) -> dict[str, Any]:
    return {
        'group_key': group_key,
        'display': display,
        'best_outer_iteration': int(st['outer_iteration']),
        'best_outer_iteration_by_yaw': int(best_yaw_it),
        'pitch_mean_abs_arcsec': st['pitch_mean_abs_arcsec'],
        'yaw_abs_mean_arcsec': st['yaw_abs_mean_arcsec'],
        'norm_mean_arcsec': st['norm_mean_arcsec'],
        'yaw_abs_median_arcsec': st['yaw_abs_median_arcsec'],
        'yaw_abs_max_arcsec': st['yaw_abs_max_arcsec'],
    }



def build_progression(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    return progression



def make_default_paths(outer_iterations: int, feedback_scale: float) -> tuple[Path, Path]:
    fb_tag = sanitize_float_tag(feedback_scale)
    stem = f'compare_four_group_alignment_arcsec_param_outer{outer_iterations}_fb{fb_tag}_{DATE_TAG}'
    md_stem = f'psins_four_group_alignment_arcsec_param_outer{outer_iterations}_fb{fb_tag}_{DATE_TAG}'
    return RESULTS_DIR / f'{stem}.json', REPORTS_DIR / f'{md_stem}.md'



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outer-iterations', type=int, default=DEFAULT_OUTER_ITERATIONS)
    parser.add_argument('--feedback-scale', type=float, default=DEFAULT_FEEDBACK_SCALE)
    parser.add_argument('--seeds', type=str, default=','.join(str(x) for x in DEFAULT_SEEDS))
    parser.add_argument('--groups', type=str, default=','.join(GROUP_ORDER))
    parser.add_argument('--max-workers', type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument('--out-json', type=str, default='')
    parser.add_argument('--out-md', type=str, default='')
    args = parser.parse_args()

    seeds = parse_int_list(args.seeds)
    groups = parse_group_list(args.groups)
    if not seeds:
        raise ValueError('seeds cannot be empty')
    if not groups:
        raise ValueError('groups cannot be empty')

    out_json, out_md = make_default_paths(args.outer_iterations, args.feedback_scale)
    if args.out_json:
        out_json = Path(args.out_json)
    if args.out_md:
        out_md = Path(args.out_md)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    baseline = json.loads(BASELINE_JSON.read_text(encoding='utf-8'))
    baseline_groups = baseline['groups']

    tasks = [(gk, seed, args.outer_iterations, args.feedback_scale) for gk in groups for seed in seeds]
    with ProcessPoolExecutor(max_workers=min(args.max_workers, len(tasks))) as ex:
        rows = list(ex.map(run_single, tasks))

    grouped: dict[str, list[dict[str, Any]]] = {gk: [] for gk in groups}
    for row in rows:
        grouped[row['group_key']].append(row)
    for gk in groups:
        grouped[gk].sort(key=lambda x: x['seed'])

    group_payloads: dict[str, Any] = {}
    best_round_rows: list[dict[str, Any]] = []

    for gk in groups:
        per_seed = grouped[gk]
        per_outer_stats = []
        for outer_it in range(1, args.outer_iterations + 1):
            outer_rows = []
            for row in per_seed:
                item = row['outer_logs'][outer_it - 1]
                outer_rows.append({
                    'seed': row['seed'],
                    'final_att_err_arcsec': item['final_att_err_arcsec'],
                    'final_att_err_abs_arcsec': item['final_att_err_abs_arcsec'],
                    'final_att_err_norm_arcsec': item['final_att_err_norm_arcsec'],
                    'final_yaw_abs_arcsec': item['final_yaw_abs_arcsec'],
                    'est_eb_dph': item['est_eb_dph'],
                    'est_db_ug': item['est_db_ug'],
                    'accum_eb_dph': item['accum_eb_dph'],
                    'accum_db_ug': item['accum_db_ug'],
                    'final_att_deg': item['final_att_deg'],
                    'est_kg_ppm': item['est_kg_ppm'],
                    'est_ka_ppm': item['est_ka_ppm'],
                })
            st = summarize_rows(outer_rows)
            st['outer_iteration'] = outer_it
            per_outer_stats.append(st)

        best_by_norm = min(per_outer_stats, key=lambda x: (x['norm_mean_arcsec'], x['yaw_abs_mean_arcsec'], x['pitch_mean_abs_arcsec']))
        best_by_yaw = min(per_outer_stats, key=lambda x: (x['yaw_abs_mean_arcsec'], x['norm_mean_arcsec'], x['pitch_mean_abs_arcsec']))
        baseline_ref = baseline_groups[gk]

        first_round = per_outer_stats[0]
        best_round_rows.append(build_summary_row(gk, GROUP_DISPLAY[gk], best_by_norm, best_by_yaw['outer_iteration']))
        group_payloads[gk] = {
            'display': GROUP_DISPLAY[gk],
            'trajectory_family': GROUP_FAMILY[gk],
            'baseline_iter1_reference': baseline_ref,
            'outer_iteration_stats': per_outer_stats,
            'best_round_by_norm': best_by_norm,
            'best_round_by_yaw': best_by_yaw,
            'best_round_by_yaw_differs': bool(best_by_norm['outer_iteration'] != best_by_yaw['outer_iteration']),
            'delta_best_vs_baseline_iter1': {
                'pitch_mean_abs_arcsec_delta': float(baseline_ref['pitch_mean_abs_arcsec'] - best_by_norm['pitch_mean_abs_arcsec']),
                'yaw_abs_mean_arcsec_delta': float(baseline_ref['yaw_abs_mean_arcsec'] - best_by_norm['yaw_abs_mean_arcsec']),
                'norm_mean_arcsec_delta': float(baseline_ref['norm_mean_arcsec'] - best_by_norm['norm_mean_arcsec']),
            },
            'delta_best_vs_outer1': {
                'pitch_mean_abs_arcsec_delta': float(first_round['pitch_mean_abs_arcsec'] - best_by_norm['pitch_mean_abs_arcsec']),
                'yaw_abs_mean_arcsec_delta': float(first_round['yaw_abs_mean_arcsec'] - best_by_norm['yaw_abs_mean_arcsec']),
                'norm_mean_arcsec_delta': float(first_round['norm_mean_arcsec'] - best_by_norm['norm_mean_arcsec']),
            },
            'per_seed_outer_logs': per_seed,
        }

    best_round_rows.sort(key=lambda x: GROUP_ORDER.index(x['group_key']))
    progression_best_round = build_progression(best_round_rows)

    payload = {
        'task': 'compare_four_group_alignment_arcsec_param_outer_2026_04_07',
        'reference_baseline_json': str(BASELINE_JSON),
        'metric_definition': 'pitch mean abs / yaw abs mean / norm mean in arcsec',
        'method': {
            'name': 'pure_param_outer_multi_round',
            'outer_iterations': args.outer_iterations,
            'feedback_scale': args.feedback_scale,
            'inner_pass_per_outer_round': 1,
            'carry_att_seed': False,
            'inner_wash_scale': 0.0,
            'inner_scale_wash_scale': 0.0,
            'feedback_slices_note': {
                'g1_plain12_singleaxis': '12-state x=[phi(3), dv(3), eb(3), db(3)] so eb=x[6:9], db=x[9:12]',
                'g2_g3_g4_24state': '24-state x=[phi(3), dv(3), eb(3), db(3), ng(3), xa(3), kg(3), ka(3)] so eb=x[6:9], db=x[9:12]',
            },
            'scd_note': 'G4 keeps the accepted pure-SCD setting alpha=0.995, transition_duration_s=2.0, apply_after_release_iter=1',
            'feedback_scale_origin_note': 'Default feedback_scale=2.2 follows the latest single-axis boundary-confirmation sweep (extrap5) as the best stable pure param-outer setting.',
        },
        'groups_order': groups,
        'seeds': seeds,
        'best_round_comparison_rows': best_round_rows,
        'progression_on_best_rounds': progression_best_round,
        'groups': group_payloads,
        'files': {
            'json': str(out_json),
            'md': str(out_md),
        },
    }
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    lines = []
    lines.append('# 原始四组对准实验：pure param-outer 多轮重跑（按每组 best outer round 比较）')
    lines.append('')
    lines.append(f'- baseline 参考：`{BASELINE_JSON}`')
    lines.append(f'- outer iterations = `{args.outer_iterations}`；feedback_scale = `{args.feedback_scale}`')
    lines.append('- 每个 outer round 都从**同一个初始姿态种子**重新起跑（no carry）')
    lines.append('- 每个 outer round 只跑 **1 次 inner pass**，并且**不做 inner wash**')
    lines.append('- 只把 `eb/db` 参数反馈回 IMU；G2/G3/G4 的切片来自 24-state 布局 `x=[phi,dv,eb,db,ng,xa,kg,ka]`，即 `eb=x[6:9], db=x[9:12]`')
    lines.append(f'- seeds = `{seeds}`')
    lines.append('')
    lines.append('## 每组 best round（按 norm mean 最小选）')
    lines.append('')
    lines.append('| 组别 | best round by norm | best round by yaw | same? | pitch mean abs (") | yaw abs mean (") | norm mean (") | Δnorm vs baseline iter1 (") |')
    lines.append('|---|---:|---:|:---:|---:|---:|---:|---:|')
    for row in best_round_rows:
        gk = row['group_key']
        grp = group_payloads[gk]
        same = 'Y' if not grp['best_round_by_yaw_differs'] else 'N'
        delta_norm = grp['delta_best_vs_baseline_iter1']['norm_mean_arcsec_delta']
        lines.append(
            f"| {row['display']} | {row['best_outer_iteration']} | {row['best_outer_iteration_by_yaw']} | {same} | {row['pitch_mean_abs_arcsec']:.6f} | {row['yaw_abs_mean_arcsec']:.6f} | {row['norm_mean_arcsec']:.6f} | {delta_norm:+.6f} |"
        )
    lines.append('')
    lines.append('## 用每组 best round 做四组比较')
    lines.append('')
    lines.append('| 组别 | best round | pitch mean abs (") | yaw abs mean (") | norm mean (") | yaw median (") | yaw max (") |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for row in best_round_rows:
        lines.append(
            f"| {row['display']} | {row['best_outer_iteration']} | {row['pitch_mean_abs_arcsec']:.6f} | {row['yaw_abs_mean_arcsec']:.6f} | {row['norm_mean_arcsec']:.6f} | {row['yaw_abs_median_arcsec']:.6f} | {row['yaw_abs_max_arcsec']:.6f} |"
        )
    lines.append('')
    lines.append('## 递进判断（使用每组各自 best round）')
    lines.append('')
    for metric, label in [
        ('pitch_mean_abs_arcsec', 'pitch mean abs'),
        ('yaw_abs_mean_arcsec', 'yaw abs mean'),
        ('norm_mean_arcsec', 'norm mean'),
    ]:
        item = progression_best_round[metric]
        verdict = '严格递进' if item['strict_progression'] else '非严格单调'
        lines.append(f"- **{label}**：{verdict}；best group = {GROUP_DISPLAY[item['best_group']]}")
        for step in item['steps']:
            tag = '改善' if step['improved'] else '退化'
            lines.append(f"  - {GROUP_DISPLAY[step['from_group']]} -> {GROUP_DISPLAY[step['to_group']]}：{tag} {step['delta']:+.6f}\"")
    lines.append('')

    for gk in groups:
        grp = group_payloads[gk]
        best_norm = grp['best_round_by_norm']
        best_yaw = grp['best_round_by_yaw']
        delta_base = grp['delta_best_vs_baseline_iter1']
        delta_outer1 = grp['delta_best_vs_outer1']
        lines.append(f'## {GROUP_DISPLAY[gk]}')
        lines.append('')
        lines.append(f"- best round by norm = `{best_norm['outer_iteration']}` | norm = `{best_norm['norm_mean_arcsec']:.6f}\"`")
        lines.append(f"- best round by yaw = `{best_yaw['outer_iteration']}` | yaw = `{best_yaw['yaw_abs_mean_arcsec']:.6f}\"`")
        lines.append(f"- best-by-yaw differs? `{'yes' if grp['best_round_by_yaw_differs'] else 'no'}`")
        lines.append(
            f"- Δ vs baseline iter1: pitch `{delta_base['pitch_mean_abs_arcsec_delta']:+.6f}\"`, yaw `{delta_base['yaw_abs_mean_arcsec_delta']:+.6f}\"`, norm `{delta_base['norm_mean_arcsec_delta']:+.6f}\"`"
        )
        lines.append(
            f"- Δ vs this rerun outer1: pitch `{delta_outer1['pitch_mean_abs_arcsec_delta']:+.6f}\"`, yaw `{delta_outer1['yaw_abs_mean_arcsec_delta']:+.6f}\"`, norm `{delta_outer1['norm_mean_arcsec_delta']:+.6f}\"`"
        )
        lines.append('')
        lines.append('| outer round | pitch mean abs (") | yaw abs mean (") | norm mean (") | yaw median (") | yaw max (") | tag |')
        lines.append('|---:|---:|---:|---:|---:|---:|---|')
        for rec in grp['outer_iteration_stats']:
            tags = []
            if rec['outer_iteration'] == best_norm['outer_iteration']:
                tags.append('best-norm')
            if rec['outer_iteration'] == best_yaw['outer_iteration']:
                tags.append('best-yaw')
            tag_text = ', '.join(tags) if tags else ''
            lines.append(
                f"| {rec['outer_iteration']} | {rec['pitch_mean_abs_arcsec']:.6f} | {rec['yaw_abs_mean_arcsec']:.6f} | {rec['norm_mean_arcsec']:.6f} | {rec['yaw_abs_median_arcsec']:.6f} | {rec['yaw_abs_max_arcsec']:.6f} | {tag_text} |"
            )
        lines.append('')

    out_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print(json.dumps({
        'out_json': str(out_json),
        'out_md': str(out_md),
        'best_round_comparison_rows': best_round_rows,
        'progression_on_best_rounds': progression_best_round,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
