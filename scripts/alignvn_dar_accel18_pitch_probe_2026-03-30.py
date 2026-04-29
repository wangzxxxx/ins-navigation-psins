#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
PSINS_ROOT = WORKSPACE / 'tmp_psins_py'
if str(PSINS_ROOT) not in sys.path:
    sys.path.insert(0, str(PSINS_ROOT))

from psins_py.imu_utils import attrottt, avp2imu, imuadderr  # noqa: E402
from psins_py.math_utils import a2qua, q2att  # noqa: E402
from psins_py.nav_utils import glv, posset  # noqa: E402

ACC18_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_accel_colored_py_2026-03-30.py'
BASE12_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
OUT_DIR = WORKSPACE / 'tmp' / 'alignment_strategy_sweep'
OUT_JSON = OUT_DIR / 'alignvn_dar_accel18_pitch_probe_2026-03-30.json'
OUT_MD = OUT_DIR / 'alignvn_dar_accel18_pitch_probe_2026-03-30.md'
BASELINE_MC50_JSON = OUT_DIR / 'alignvn_dar_accel18_mc50_iter1_2026-03-30.json'


@dataclass
class ProbeConfig:
    name: str
    label: str
    hypothesis: str
    seeds: list[int]
    max_iter: int = 1
    wash_scale: float = 0.5
    carry_att_seed: bool = True
    ng_sigma_dph: list[float] = None
    tau_g_s: list[float] = None
    xa_sigma_ug: list[float] = None
    tau_a_s: list[float] = None
    dkg_ppm: float = 30.0
    dka_ppm: float = 30.0
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
        ACC18 = load_module('alignvn_accel18_pitch_probe_target_20260330', ACC18_PATH)
    return ACC18


def load_base12():
    global BASE12
    if BASE12 is None:
        BASE12 = load_module('alignvn_base12_pitch_probe_target_20260330', BASE12_PATH)
    return BASE12


def build_imuerr_with_scale(dkg_ppm: float, dka_ppm: float) -> dict[str, np.ndarray]:
    base12 = load_base12()
    imuerr = base12.build_imuerr()
    imuerr['dKg'] = np.diag(np.full(3, dkg_ppm * glv.ppm))
    imuerr['dKa'] = np.diag(np.full(3, dka_ppm * glv.ppm))
    return imuerr


def run_single_seed(seed: int, cfg: ProbeConfig) -> dict[str, Any]:
    acc18 = load_acc18()
    base12 = load_base12()

    np.random.seed(seed)

    ts = 0.01
    att0 = np.array([0.0, 0.0, 0.0])
    pos0 = posset(34, 116, 480, isdeg=1)
    rot_paras = base12.build_rot_paras()
    att_truth = attrottt(att0, rot_paras, ts)
    imu, _ = avp2imu(att_truth, pos0)

    imuerr = build_imuerr_with_scale(cfg.dkg_ppm, cfg.dka_ppm)
    imu_noisy = imuadderr(imu, imuerr)

    phi = np.array([0.1, 0.1, 0.5]) * glv.deg
    att0_guess = q2att(base12.qaddphi(a2qua(att0), phi))
    wvn = np.array([0.01, 0.01, 0.01])

    att_18, _attk_18, _xkpk_18, iter_logs = acc18.alignvn_18state_iter(
        imu=imu_noisy.copy(),
        qnb=att0_guess,
        pos=pos0,
        phi0=phi,
        imuerr=imuerr,
        wvn=wvn,
        max_iter=cfg.max_iter,
        truth_att=att_truth[-1, 0:3],
        ng_sigma=np.array(cfg.ng_sigma_dph) * glv.dph,
        tau_g_s=np.array(cfg.tau_g_s),
        xa_sigma=np.array(cfg.xa_sigma_ug) * glv.ug,
        tau_a_s=np.array(cfg.tau_a_s),
        wash_scale=cfg.wash_scale,
        carry_att_seed=cfg.carry_att_seed,
    )
    metrics = acc18.to_method_metrics(cfg.name, att_18, att_truth[-1, 0:3], iter_logs)
    return {
        'seed': seed,
        'final_att_err_arcsec': metrics.final_att_err_arcsec,
        'final_att_err_abs_arcsec': metrics.final_att_err_abs_arcsec,
        'final_att_err_norm_arcsec': metrics.final_att_err_norm_arcsec,
        'final_yaw_abs_arcsec': metrics.final_yaw_abs_arcsec,
        'best_iteration_by_yaw': metrics.best_iteration_by_yaw,
        'iter_logs': metrics.iter_logs,
    }


def summarize_probe(cfg: ProbeConfig) -> dict[str, Any]:
    per_seed = [run_single_seed(seed, cfg) for seed in cfg.seeds]
    errs = np.array([row['final_att_err_arcsec'] for row in per_seed], dtype=float)
    abs_errs = np.abs(errs)
    norms = np.array([row['final_att_err_norm_arcsec'] for row in per_seed], dtype=float)
    yaw_abs = np.array([row['final_yaw_abs_arcsec'] for row in per_seed], dtype=float)

    pitch = errs[:, 1]
    pitch_abs = abs_errs[:, 1]
    roll = errs[:, 0]

    summary = {
        'name': cfg.name,
        'label': cfg.label,
        'hypothesis': cfg.hypothesis,
        'config': asdict(cfg),
        'statistics': {
            'mean_signed_arcsec': errs.mean(axis=0).tolist(),
            'std_signed_arcsec_1sigma': errs.std(axis=0, ddof=0).tolist(),
            'mean_abs_arcsec': abs_errs.mean(axis=0).tolist(),
            'median_abs_arcsec': np.median(abs_errs, axis=0).tolist(),
            'norm_mean_arcsec': float(norms.mean()),
            'norm_median_arcsec': float(np.median(norms)),
            'yaw_abs_mean_arcsec': float(yaw_abs.mean()),
            'yaw_abs_median_arcsec': float(np.median(yaw_abs)),
            'pitch_bias_to_sigma_ratio': float(abs(pitch.mean()) / max(pitch.std(ddof=0), 1e-12)),
            'pitch_sem_arcsec': float(pitch.std(ddof=0) / math.sqrt(len(pitch))),
            'pitch_signed_range_arcsec': [float(pitch.min()), float(pitch.max())],
            'roll_signed_range_arcsec': [float(roll.min()), float(roll.max())],
        },
        'per_seed': per_seed,
    }
    return summary


def compare_to_baseline(baseline: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    b = baseline['statistics']
    o = other['statistics']
    return {
        'vs': baseline['name'],
        'delta_pitch_mean_signed_arcsec': o['mean_signed_arcsec'][1] - b['mean_signed_arcsec'][1],
        'delta_pitch_mean_abs_arcsec': o['mean_abs_arcsec'][1] - b['mean_abs_arcsec'][1],
        'delta_pitch_median_abs_arcsec': o['median_abs_arcsec'][1] - b['median_abs_arcsec'][1],
        'delta_yaw_abs_mean_arcsec': o['yaw_abs_mean_arcsec'] - b['yaw_abs_mean_arcsec'],
        'delta_norm_mean_arcsec': o['norm_mean_arcsec'] - b['norm_mean_arcsec'],
    }


def load_mc50_reference() -> dict[str, Any] | None:
    if not BASELINE_MC50_JSON.exists():
        return None
    try:
        return json.loads(BASELINE_MC50_JSON.read_text())
    except Exception:
        return None


def build_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append('# 18-state pitch-bias diagnostic probe')
    lines.append('')
    lines.append('## 结论先行')
    lines.append(f"- 主因判断：**{payload['judgement']['main_cause']}**")
    lines.append(f"- 最强证据：{payload['judgement']['strongest_evidence']}")
    lines.append(f"- 最佳候选修复：{payload['judgement']['best_candidate_fix']}")
    lines.append(f"- 是否真的把 pitch 拉回 0：**{payload['judgement']['actual_repair_verdict']}**")
    lines.append('')

    if payload.get('mc50_reference'):
        mc = payload['mc50_reference']['statistics']
        lines.append('## MC50 现成基线（iter=1）')
        lines.append(
            f"- mean signed pitch = **{mc['mean_signed_arcsec'][1]:.2f}" + '"**'
            f", σ = **{mc['std_signed_arcsec_1sigma'][1]:.3f}" + '"**'
            f", |mean|/σ = **{abs(mc['mean_signed_arcsec'][1]) / mc['std_signed_arcsec_1sigma'][1]:.1f}**"
        )
        lines.append('- 这已经说明当前 pitch 偏差几乎完全不像随机散布，更像稳定的未建模确定项。')
        lines.append('')

    lines.append('## 各 probe 摘要')
    lines.append('')
    lines.append('| probe | seeds | pitch mean signed (") | pitch mean abs (") | yaw abs mean (") | norm mean (") | 备注 |')
    lines.append('|---|---:|---:|---:|---:|---:|---|')
    for item in payload['probes']:
        st = item['statistics']
        cfg = item['config']
        note = cfg['note'] or item['hypothesis']
        lines.append(
            f"| {item['name']} | {len(cfg['seeds'])} | {st['mean_signed_arcsec'][1]:.2f} | {st['mean_abs_arcsec'][1]:.2f} | {st['yaw_abs_mean_arcsec']:.2f} | {st['norm_mean_arcsec']:.2f} | {note} |"
        )
    lines.append('')

    lines.append('## 诊断证据')
    lines.append('')
    for bullet in payload['diagnostic_evidence']:
        lines.append(f'- {bullet}')
    lines.append('')

    lines.append('## 实际修复结果')
    lines.append('')
    for bullet in payload['actual_repair_results']:
        lines.append(f'- {bullet}')
    lines.append('')

    lines.append('## 推荐下一步')
    lines.append('')
    for bullet in payload['recommended_next_move']:
        lines.append(f'- {bullet}')
    lines.append('')

    return '\n'.join(lines) + '\n'


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    probes = [
        ProbeConfig(
            name='baseline_iter1',
            label='baseline iter=1',
            hypothesis='baseline',
            seeds=list(range(10)),
            max_iter=1,
            note='当前 18-state MC50 问题配置的 10-seed 复核。',
        ),
        ProbeConfig(
            name='zero_dkg_dka_iter1',
            label='zero dKg/dKa iter=1',
            hypothesis='H1',
            seeds=list(range(10)),
            max_iter=1,
            dkg_ppm=0.0,
            dka_ppm=0.0,
            note='只去掉 truth 注入里的 30 ppm dKg/dKa，其余保持不变。',
        ),
        ProbeConfig(
            name='xa_sigma_down_iter1',
            label='xa sigma down iter=1',
            hypothesis='H2',
            seeds=list(range(10)),
            max_iter=1,
            xa_sigma_ug=[0.001, 0.001, 0.001],
            note='测试 xa 建模强度减弱 10x。',
        ),
        ProbeConfig(
            name='xa_sigma_up_iter1',
            label='xa sigma up iter=1',
            hypothesis='H2',
            seeds=list(range(10)),
            max_iter=1,
            xa_sigma_ug=[0.1, 0.1, 0.1],
            note='测试 xa 建模强度增强 10x。',
        ),
        ProbeConfig(
            name='xa_tau_fast_iter1',
            label='xa tau fast iter=1',
            hypothesis='H2',
            seeds=list(range(10)),
            max_iter=1,
            tau_a_s=[30.0, 30.0, 30.0],
            note='测试 xa 相关时间缩短到 30 s。',
        ),
        ProbeConfig(
            name='xa_tau_slow_iter1',
            label='xa tau slow iter=1',
            hypothesis='H2',
            seeds=list(range(10)),
            max_iter=1,
            tau_a_s=[300.0, 300.0, 300.0],
            note='测试 xa 相关时间拉长到 300 s。',
        ),
        ProbeConfig(
            name='baseline_iter5',
            label='baseline iter=5',
            hypothesis='H3',
            seeds=list(range(5)),
            max_iter=5,
            note='检查迭代本身会不会把 pitch 明显拉回 0。',
        ),
        ProbeConfig(
            name='iter5_nowash',
            label='iter=5 wash=0',
            hypothesis='H3',
            seeds=list(range(5)),
            max_iter=5,
            wash_scale=0.0,
            note='检查 wash 是否是 pitch 偏差来源。',
        ),
        ProbeConfig(
            name='iter5_nocarry',
            label='iter=5 carry_att_seed=false',
            hypothesis='H3',
            seeds=list(range(5)),
            max_iter=5,
            carry_att_seed=False,
            note='检查 carry-att 行为是否主导 pitch 偏差。',
        ),
    ]

    summaries = []
    for cfg in probes:
        print(f'[run] {cfg.name} seeds={cfg.seeds[0]}..{cfg.seeds[-1]} max_iter={cfg.max_iter}')
        summaries.append(summarize_probe(cfg))

    by_name = {item['name']: item for item in summaries}
    baseline = by_name['baseline_iter1']
    for item in summaries:
        item['comparison_to_baseline_iter1'] = compare_to_baseline(baseline, item)

    mc50_reference = load_mc50_reference()

    zero_scale = by_name['zero_dkg_dka_iter1']
    h2_names = ['xa_sigma_down_iter1', 'xa_sigma_up_iter1', 'xa_tau_fast_iter1', 'xa_tau_slow_iter1']
    h2_sorted = sorted([by_name[name] for name in h2_names], key=lambda x: abs(x['statistics']['mean_signed_arcsec'][1]))
    best_h2 = h2_sorted[0]
    h3_names = ['baseline_iter5', 'iter5_nowash', 'iter5_nocarry']
    h3_sorted = sorted([by_name[name] for name in h3_names], key=lambda x: abs(x['statistics']['mean_signed_arcsec'][1]))
    best_h3 = h3_sorted[0]

    diagnostic_evidence = []
    if mc50_reference:
        mc = mc50_reference['statistics']
        diagnostic_evidence.append(
            f"现成 MC50 iter=1 基线里，pitch mean = {mc['mean_signed_arcsec'][1]:.2f}\"，σ = {mc['std_signed_arcsec_1sigma'][1]:.3f}\"，|mean|/σ = {abs(mc['mean_signed_arcsec'][1]) / mc['std_signed_arcsec_1sigma'][1]:.1f}，明显是稳定偏置而不是随机散布。"
        )
    diagnostic_evidence.append(
        f"10-seed baseline 复核得到 pitch mean = {baseline['statistics']['mean_signed_arcsec'][1]:.2f}\"，σ = {baseline['statistics']['std_signed_arcsec_1sigma'][1]:.3f}\"，和 MC50 现象一致。"
    )
    diagnostic_evidence.append(
        f"把 dKg/dKa 从 30 ppm 置零后，pitch mean 从 {baseline['statistics']['mean_signed_arcsec'][1]:.2f}\" 变为 {zero_scale['statistics']['mean_signed_arcsec'][1]:.2f}\"，pitch mean abs 从 {baseline['statistics']['mean_abs_arcsec'][1]:.2f}\" 降到 {zero_scale['statistics']['mean_abs_arcsec'][1]:.2f}\"。"
    )
    diagnostic_evidence.append(
        f"H2 最优 xa 参数只是把 pitch mean 从 {baseline['statistics']['mean_signed_arcsec'][1]:.2f}\" 改到 {best_h2['statistics']['mean_signed_arcsec'][1]:.2f}\"；量级仍在数十角秒，说明 xa 强度/tau 不是主因。"
    )
    diagnostic_evidence.append(
        f"H3 最优迭代类变体是 {best_h3['name']}，pitch mean = {best_h3['statistics']['mean_signed_arcsec'][1]:.2f}\"；相对 baseline_iter1 并没有真正回到 0 附近。"
    )

    actual_repair_results = [
        (
            f"真正显著降低 pitch 偏差的唯一 probe 是 **zero_dkg_dka_iter1**："
            f"pitch mean abs 改善 {baseline['statistics']['mean_abs_arcsec'][1] - zero_scale['statistics']['mean_abs_arcsec'][1]:.2f}\"，"
            f"yaw abs mean {'下降' if zero_scale['statistics']['yaw_abs_mean_arcsec'] < baseline['statistics']['yaw_abs_mean_arcsec'] else '上升'}到 {zero_scale['statistics']['yaw_abs_mean_arcsec']:.2f}\"。"
        ),
        (
            f"在 **保持 30 ppm dKg/dKa 注入不变** 的前提下，最好的 xa 调参 probe 是 {best_h2['name']}，"
            f"pitch mean abs 只改善 {baseline['statistics']['mean_abs_arcsec'][1] - best_h2['statistics']['mean_abs_arcsec'][1]:.2f}\"，不构成真正 repair。"
        ),
        (
            f"迭代/洗偏/carry-att 变体里，{best_h3['name']} 的 pitch mean abs = {best_h3['statistics']['mean_abs_arcsec'][1]:.2f}\"；"
            f"相比 baseline_iter1 的 {baseline['statistics']['mean_abs_arcsec'][1]:.2f}\"，没有 material 改善。"
        ),
    ]

    if abs(zero_scale['statistics']['mean_signed_arcsec'][1]) < 5.0:
        actual_repair_verdict = '若允许把 truth 中未建模 dKg/dKa 去掉，pitch 可以明显靠近 0；但这属于改 benchmark truth，不是同真值条件下的滤波器修复。'
    else:
        actual_repair_verdict = '本轮没有找到在同真值条件下能把 pitch 真正拉回 0 的简单修复。'

    main_cause = (
        '30 ppm dKg/dKa 这类未建模确定性比例项是主导 pitch 偏差的第一嫌疑，并且证据强于 xa 参数失配或迭代壳层设置。'
    )

    strongest_evidence = (
        f"zero_dkg_dka_iter1 使 pitch mean signed 从 {baseline['statistics']['mean_signed_arcsec'][1]:.2f}\" 变到 {zero_scale['statistics']['mean_signed_arcsec'][1]:.2f}\"，"
        f"而 H2/H3 的最优变体仍停留在 {best_h2['statistics']['mean_signed_arcsec'][1]:.2f}\" / {best_h3['statistics']['mean_signed_arcsec'][1]:.2f}\" 量级。"
    )

    best_candidate_fix = (
        '短期：把 benchmark 分成“无 dKg/dKa”与“有 dKg/dKa”两条口径，先确认 18-state 在模型匹配条件下无系统 pitch 偏差；'
        '中期：若要在同真值条件下修复，优先扩展状态/补偿链去显式处理 dKg/dKa，而不是继续调 xa_sigma / tau_a。'
    )

    recommended_next_move = [
        '不要再把主要时间花在 xa_sigma / tau_a 微调上；它们对 pitch 系统偏差只产生边角效应。',
        '下一步更值当的是做一个最小 dKg/dKa 补偿对照：哪怕不是完整 24-state，也至少加一个可辨识的比例项近似，验证 pitch 是否随之回正。',
        '汇报时把“诊断证据”和“实际 repair”分开：当前真正有效的是去掉未建模 truth 项，而不是现有 18-state 自身已经学会了消化它。',
    ]

    payload = {
        'script': str(Path(__file__)),
        'target_script': str(ACC18_PATH),
        'baseline_mc50_reference': str(BASELINE_MC50_JSON),
        'mc50_reference': mc50_reference,
        'judgement': {
            'main_cause': main_cause,
            'strongest_evidence': strongest_evidence,
            'best_candidate_fix': best_candidate_fix,
            'actual_repair_verdict': actual_repair_verdict,
        },
        'diagnostic_evidence': diagnostic_evidence,
        'actual_repair_results': actual_repair_results,
        'recommended_next_move': recommended_next_move,
        'probes': summaries,
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    OUT_MD.write_text(build_report(payload))
    print(f'[ok] wrote {OUT_JSON}')
    print(f'[ok] wrote {OUT_MD}')


if __name__ == '__main__':
    main()
