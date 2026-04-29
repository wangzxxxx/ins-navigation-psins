#!/usr/bin/env python3
"""DAR accel-colored-state alignment prototype.

Built directly on top of the existing 15-state random-model Python port.

Goal of this first workable version:
- keep the same continuous DAR trajectory, same initial misalignment, same seed(s)
- keep the 15-state random-model shell: phi / dv / eb / db / ng
- extend it with explicit accel colored-noise states xa
- use a minimal per-axis first-order AR(1) / GM state for xa so the filter becomes
  a practical 18-state prototype:
      x = [phi(3), dv(3), eb(3), db(3), ng(3), xa(3)]
- xa is added to the *state equation* through dv <- Cbn * xa * dt, not merely Q retuning

Important boundary:
- this is still not the full chapter-4 24-state ARMA(3,1) implementation
- it is the smallest truthful prototype that puts accel colored states inside the KF
"""

from __future__ import annotations

import importlib.util
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

from psins_py.imu_utils import attrottt, avp2imu, cnscl, imuadderr  # noqa: E402
from psins_py.kf_utils import kfupdate  # noqa: E402
from psins_py.math_utils import a2qua, askew, m2rv, q2att, q2mat, qmul, qupdt2, rv2m, rv2q  # noqa: E402
from psins_py.nav_utils import Earth, glv, posset  # noqa: E402

SCRIPT_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_accel_colored_py_2026-03-30.py'
OUT_DIR = WORKSPACE / 'tmp' / 'alignment_strategy_sweep'
OUT_JSON = OUT_DIR / 'alignvn_dar_accel_colored_result_2026-03-30.json'
OUT_MD = OUT_DIR / 'alignvn_dar_accel_colored_summary_2026-03-30.md'
PREV15_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_random_model_py_2026-03-30.py'
PREV15_JSON = OUT_DIR / 'alignvn_dar_random_model_result_2026-03-30.json'

PREV15 = None
BASE12 = None


@dataclass
class IterationLog18:
    iteration: int
    final_att_deg: list[float]
    att_err_arcsec: list[float]
    att_err_norm_arcsec: float
    yaw_abs_arcsec: float
    est_eb_dph: list[float]
    est_db_ug: list[float]
    est_ng_dph: list[float]
    est_xa_ug: list[float]


@dataclass
class MethodMetrics:
    method: str
    final_att_deg: list[float]
    final_att_err_arcsec: list[float]
    final_att_err_abs_arcsec: list[float]
    final_att_err_norm_arcsec: float
    final_yaw_abs_arcsec: float
    best_iteration_by_norm: dict[str, Any]
    best_iteration_by_yaw: dict[str, Any]
    iter_logs: list[dict[str, Any]]


@dataclass
class SeedComparison:
    seed: int
    random15: dict[str, Any]
    accel18: dict[str, Any]
    yaw_gain_arcsec: float
    best_yaw_gain_arcsec: float


@dataclass
class SeedAggregate:
    seeds: list[int]
    random15_mean_final_yaw_abs_arcsec: float
    accel18_mean_final_yaw_abs_arcsec: float
    random15_mean_best_yaw_abs_arcsec: float
    accel18_mean_best_yaw_abs_arcsec: float
    random15_median_final_yaw_abs_arcsec: float
    accel18_median_final_yaw_abs_arcsec: float
    per_seed: list[dict[str, Any]]


@dataclass
class AccelColoredConfig:
    ng_sigma_dph: list[float]
    tau_g_s: list[float]
    xa_sigma_ug: list[float]
    tau_a_s: list[float]
    xa_model: str
    wash_scale: float
    carry_att_seed: bool
    max_iter: int
    note: str


def load_prev15_module():
    global PREV15
    if PREV15 is not None:
        return PREV15
    spec = importlib.util.spec_from_file_location('alignvn_random15_20260330', PREV15_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load previous 15-state module from {PREV15_PATH}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    PREV15 = module
    return module


def load_iterfix12_module():
    global BASE12
    if BASE12 is not None:
        return BASE12
    prev15 = load_prev15_module()
    BASE12 = prev15.load_iterfix12_module()
    return BASE12


def left_quat_update(q: np.ndarray, phi: np.ndarray) -> np.ndarray:
    qq = qmul(rv2q(phi), q)
    return qq / np.linalg.norm(qq)


# Match the existing iterfix / random-model sign convention.
def qdelphi(q: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return left_quat_update(q, phi)


# Return q1 relative to q2 so that q1=qaddphi(q2,phi) => qq2phi(q1,q2)~=phi.
def qq2phi(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    return m2rv(q2mat(q1) @ q2mat(q2).T)



def avnkfinit_18_accel_colored(
    nts: float,
    pos: np.ndarray,
    phi0: np.ndarray,
    imuerr: dict[str, np.ndarray],
    wvn: np.ndarray,
    ng_sigma: np.ndarray,
    tau_g_s: np.ndarray,
    xa_sigma: np.ndarray,
    tau_a_s: np.ndarray,
) -> dict[str, Any]:
    """18-state KF init: phi/dv/eb/db/ng/xa.

    ng: 3-axis gyro first-order GM random-error states.
    xa: 3-axis accel colored-noise states, implemented as minimal per-axis AR(1)/GM states.
    """
    eth = Earth(pos)
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

    fg = np.exp(-nts / tau_g_s)
    fa = np.exp(-nts / tau_a_s)
    q_ng = ng_sigma * np.sqrt(np.maximum(1.0 - fg**2, 0.0))
    q_xa = xa_sigma * np.sqrt(np.maximum(1.0 - fa**2, 0.0))

    qk = np.zeros((18, 18))
    qk[0:3, 0:3] = np.diag(web**2 * nts)
    qk[3:6, 3:6] = np.diag(wdb**2 * nts)
    qk[12:15, 12:15] = np.diag(q_ng**2)
    qk[15:18, 15:18] = np.diag(q_xa**2)

    ft = np.zeros((18, 18))
    ft[0:3, 0:3] = askew(-eth.wnie)
    phikk_1 = np.eye(18) + ft * nts
    phikk_1[12:15, 12:15] = np.diag(fg)
    phikk_1[15:18, 15:18] = np.diag(fa)

    return {
        'n': 18,
        'm': 3,
        'nts': nts,
        'Qk': qk,
        'Rk': np.diag(wvn.reshape(3)) ** 2 / nts,
        'Pxk': np.diag(np.r_[phi0, np.ones(3), init_eb_p, init_db_p, ng_sigma, init_xa_p]) ** 2,
        'Phikk_1': phikk_1,
        'Hk': np.hstack([np.zeros((3, 3)), np.eye(3), np.zeros((3, 12))]),
        'xk': np.zeros(18),
        'fg': fg,
        'fa': fa,
    }



def alignvn_18state_iter(
    imu: np.ndarray,
    qnb: np.ndarray,
    pos: np.ndarray,
    phi0: np.ndarray,
    imuerr: dict[str, np.ndarray],
    wvn: np.ndarray,
    max_iter: int,
    truth_att: np.ndarray,
    ng_sigma: np.ndarray,
    tau_g_s: np.ndarray,
    xa_sigma: np.ndarray,
    tau_a_s: np.ndarray,
    wash_scale: float = 0.5,
    carry_att_seed: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[IterationLog18]]:
    """Iterative DAR alignment with explicit gyro GM + accel colored states.

    State vector:
        x = [phi(3), dv(3), eb(3), db(3), ng(3), xa(3)]

    Minimal accel-colored prototype:
        xa_k = Fa * xa_{k-1} + w_a

    Key new coupling vs 15-state:
        dv_k gets driven by both db and xa through +Cbn*dt
    """
    imu_corr = imu.copy()

    nn = 2
    ts = float(imu_corr[1, -1] - imu_corr[0, -1])
    nts = nn * ts
    qnb_seed = a2qua(qnb) if len(qnb) == 3 else np.asarray(qnb).reshape(4)

    length = (len(imu_corr) // nn) * nn
    imu_corr = imu_corr[:length]

    eth = Earth(pos)
    cnn = rv2m(-eth.wnie * nts / 2)

    final_attk = None
    final_xkpk = None
    iter_logs: list[IterationLog18] = []

    for iteration in range(1, max_iter + 1):
        kf = avnkfinit_18_accel_colored(nts, pos, phi0, imuerr, wvn, ng_sigma, tau_g_s, xa_sigma, tau_a_s)
        vn = np.zeros(3)
        qnbi = qnb_seed.copy()

        attk_rows: list[np.ndarray] = []
        xkpk_rows: list[np.ndarray] = []

        for k in range(0, length, nn):
            wvm = imu_corr[k:k + nn, 0:6]
            t = float(imu_corr[k + nn - 1, -1])
            phim, dvbm = cnscl(wvm)

            cnb = q2mat(qnbi)
            dvn = cnn @ cnb @ dvbm
            vn = vn + dvn + eth.gn * nts
            qnbi = qupdt2(qnbi, phim, eth.wnin * nts)

            phi_k = kf['Phikk_1'].copy()
            cnbts = cnb * nts
            phi_k[3:6, 0:3] = askew(dvn)
            phi_k[3:6, 9:12] = cnbts
            phi_k[3:6, 15:18] = cnbts
            phi_k[0:3, 6:9] = -cnbts
            phi_k[0:3, 12:15] = -cnbts
            phi_k[12:15, 12:15] = np.diag(kf['fg'])
            phi_k[15:18, 15:18] = np.diag(kf['fa'])
            kf['Phikk_1'] = phi_k

            kf = kfupdate(kf, vn)

            qnbi = qdelphi(qnbi, 0.91 * kf['xk'][0:3])
            kf['xk'][0:3] *= 0.09
            vn = vn - 0.91 * kf['xk'][3:6]
            kf['xk'][3:6] *= 0.09

            attk_rows.append(np.r_[q2att(qnbi), vn, t])
            xkpk_rows.append(np.r_[kf['xk'], np.diag(kf['Pxk']), t])

        attk = np.asarray(attk_rows)
        xkpk = np.asarray(xkpk_rows)
        final_attk = attk
        final_xkpk = xkpk

        est_att = attk[-1, 0:3]
        att_err_arcsec = qq2phi(a2qua(est_att), a2qua(truth_att)) / glv.sec
        iter_logs.append(IterationLog18(
            iteration=iteration,
            final_att_deg=(est_att / glv.deg).tolist(),
            att_err_arcsec=att_err_arcsec.tolist(),
            att_err_norm_arcsec=float(np.linalg.norm(att_err_arcsec)),
            yaw_abs_arcsec=float(abs(att_err_arcsec[2])),
            est_eb_dph=(kf['xk'][6:9] / glv.dph).tolist(),
            est_db_ug=(kf['xk'][9:12] / glv.ug).tolist(),
            est_ng_dph=(kf['xk'][12:15] / glv.dph).tolist(),
            est_xa_ug=(kf['xk'][15:18] / glv.ug).tolist(),
        ))

        if iteration < max_iter:
            if carry_att_seed:
                qnb_seed = qnbi.copy()
            imu_corr[:, 0:3] -= wash_scale * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= wash_scale * kf['xk'][9:12] * ts

    assert final_attk is not None and final_xkpk is not None
    return final_attk[-1, 0:3], final_attk, final_xkpk, iter_logs



def to_method_metrics(method: str, att_aligned: np.ndarray, truth_att: np.ndarray, iter_logs: list[Any]) -> MethodMetrics:
    final_att_err_arcsec = qq2phi(a2qua(att_aligned), a2qua(truth_att)) / glv.sec
    final_att_err_abs_arcsec = np.abs(final_att_err_arcsec)
    best_iter_by_norm = min(iter_logs, key=lambda x: x.att_err_norm_arcsec)
    best_iter_by_yaw = min(iter_logs, key=lambda x: x.yaw_abs_arcsec)
    return MethodMetrics(
        method=method,
        final_att_deg=(att_aligned / glv.deg).tolist(),
        final_att_err_arcsec=final_att_err_arcsec.tolist(),
        final_att_err_abs_arcsec=final_att_err_abs_arcsec.tolist(),
        final_att_err_norm_arcsec=float(np.linalg.norm(final_att_err_arcsec)),
        final_yaw_abs_arcsec=float(abs(final_att_err_arcsec[2])),
        best_iteration_by_norm=asdict(best_iter_by_norm),
        best_iteration_by_yaw=asdict(best_iter_by_yaw),
        iter_logs=[asdict(x) for x in iter_logs],
    )



def load_prev15_reference() -> dict[str, Any] | None:
    if not PREV15_JSON.exists():
        return None
    try:
        return json.loads(PREV15_JSON.read_text())
    except Exception:
        return None



def run_seed_case(seed: int, cfg: AccelColoredConfig) -> dict[str, Any]:
    prev15 = load_prev15_module()
    base12 = load_iterfix12_module()

    np.random.seed(seed)

    ts = 0.01
    att0 = np.array([0.0, 0.0, 0.0])
    pos0 = posset(34, 116, 480, isdeg=1)
    rot_paras = base12.build_rot_paras()
    att_truth = attrottt(att0, rot_paras, ts)
    imu, _ = avp2imu(att_truth, pos0)

    imuerr = base12.build_imuerr()
    imu_noisy = imuadderr(imu, imuerr)

    phi = np.array([0.1, 0.1, 0.5]) * glv.deg
    att0_guess = q2att(base12.qaddphi(a2qua(att0), phi))
    wvn = np.array([0.01, 0.01, 0.01])

    att_15, attk_15, xkpk_15, iter_logs_15 = prev15.alignvn_15state_iter(
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
        wash_scale=cfg.wash_scale,
        carry_att_seed=cfg.carry_att_seed,
    )
    metrics_15 = to_method_metrics('random_model_15state', att_15, att_truth[-1, 0:3], iter_logs_15)

    att_18, attk_18, xkpk_18, iter_logs_18 = alignvn_18state_iter(
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
    metrics_18 = to_method_metrics('accel_colored_18state', att_18, att_truth[-1, 0:3], iter_logs_18)

    return {
        'seed': seed,
        'trajectory': {
            'ts': ts,
            'total_time_s': float(att_truth[-1, -1]),
            'rot_paras': rot_paras.tolist(),
            'initial_att_deg': (att0 / glv.deg).tolist(),
            'truth_final_att_deg': (att_truth[-1, 0:3] / glv.deg).tolist(),
            'phi0_deg': (phi / glv.deg).tolist(),
            'wvn_mps': wvn.tolist(),
        },
        'imuerr': {
            'eb_dph': (imuerr['eb'] / glv.dph).tolist(),
            'db_ug': (imuerr['db'] / glv.ug).tolist(),
            'web_dpsh': (imuerr['web'] / glv.dpsh).tolist(),
            'wdb_ugpsHz': (imuerr['wdb'] / glv.ugpsHz).tolist(),
            'dKg_ppm_diag': (np.diag(imuerr['dKg']) / glv.ppm).tolist(),
            'dKa_ppm_diag': (np.diag(imuerr['dKa']) / glv.ppm).tolist(),
        },
        'random15': asdict(metrics_15),
        'accel18': asdict(metrics_18),
        'final_rows': {
            'random15_attk_last': attk_15[-1].tolist(),
            'random15_xkpk_last': xkpk_15[-1].tolist(),
            'accel18_attk_last': attk_18[-1].tolist(),
            'accel18_xkpk_last': xkpk_18[-1].tolist(),
        },
    }



def summarize_seed_review(seed_cases: list[dict[str, Any]]) -> SeedAggregate:
    per_seed: list[SeedComparison] = []
    random15_final = []
    accel18_final = []
    random15_best = []
    accel18_best = []

    for item in seed_cases:
        rnd = item['random15']
        acc = item['accel18']
        random15_final.append(rnd['final_yaw_abs_arcsec'])
        accel18_final.append(acc['final_yaw_abs_arcsec'])
        random15_best.append(rnd['best_iteration_by_yaw']['yaw_abs_arcsec'])
        accel18_best.append(acc['best_iteration_by_yaw']['yaw_abs_arcsec'])
        per_seed.append(SeedComparison(
            seed=item['seed'],
            random15={
                'final_yaw_abs_arcsec': rnd['final_yaw_abs_arcsec'],
                'best_yaw_abs_arcsec': rnd['best_iteration_by_yaw']['yaw_abs_arcsec'],
                'final_norm_arcsec': rnd['final_att_err_norm_arcsec'],
            },
            accel18={
                'final_yaw_abs_arcsec': acc['final_yaw_abs_arcsec'],
                'best_yaw_abs_arcsec': acc['best_iteration_by_yaw']['yaw_abs_arcsec'],
                'final_norm_arcsec': acc['final_att_err_norm_arcsec'],
            },
            yaw_gain_arcsec=rnd['final_yaw_abs_arcsec'] - acc['final_yaw_abs_arcsec'],
            best_yaw_gain_arcsec=rnd['best_iteration_by_yaw']['yaw_abs_arcsec'] - acc['best_iteration_by_yaw']['yaw_abs_arcsec'],
        ))

    return SeedAggregate(
        seeds=[item['seed'] for item in seed_cases],
        random15_mean_final_yaw_abs_arcsec=float(np.mean(random15_final)),
        accel18_mean_final_yaw_abs_arcsec=float(np.mean(accel18_final)),
        random15_mean_best_yaw_abs_arcsec=float(np.mean(random15_best)),
        accel18_mean_best_yaw_abs_arcsec=float(np.mean(accel18_best)),
        random15_median_final_yaw_abs_arcsec=float(np.median(random15_final)),
        accel18_median_final_yaw_abs_arcsec=float(np.median(accel18_final)),
        per_seed=[asdict(x) for x in per_seed],
    )



def build_markdown(
    cfg: AccelColoredConfig,
    seed0_case: dict[str, Any],
    seed_review: SeedAggregate,
    prev15_ref: dict[str, Any] | None,
) -> str:
    rnd0 = seed0_case['random15']
    acc0 = seed0_case['accel18']
    seed0_better = acc0['final_yaw_abs_arcsec'] < rnd0['final_yaw_abs_arcsec']
    mean_better = seed_review.accel18_mean_final_yaw_abs_arcsec < seed_review.random15_mean_final_yaw_abs_arcsec

    headline = (
        f"accel 有色噪声版跑通；seed0 最终 yaw≈{acc0['final_yaw_abs_arcsec']:.2f}\"；"
        f"比 15-state {'更好' if seed0_better else '更差'}。"
    )

    ref_line = '- 没读到上一版 15-state 的归档 JSON。'
    if prev15_ref is not None:
        try:
            ref_seed0 = prev15_ref['seed0_direct_comparison']['random_model']['final_yaw_abs_arcsec']
            ref_mean = prev15_ref['seed_review_0_1_2']['random_model_mean_final_yaw_abs_arcsec']
            ref_line = (
                f'- 上一版 15-state 归档参考：seed0 final yaw≈`{ref_seed0:.2f}"`，'
                f'seed0-2 mean final yaw≈`{ref_mean:.2f}"`。'
            )
        except Exception:
            ref_line = '- 读到了上一版 15-state JSON，但关键摘要字段解析失败。'

    lines: list[str] = [
        headline,
        '',
        '# DAR accel colored 版（18-state, phi/dv/eb/db/ng/xa）首版结果',
        '',
        '## 1. accel 有色噪声状态是否真的加进滤波器了？',
        '',
        '- **是，真的加进去了**，不是只调 `Q`。',
        '- 这版状态向量是：`[phi(3), dv(3), eb(3), db(3), ng(3), xa(3)]`，总维数 **18**。',
        '- `ng` 继续保留为三轴陀螺一阶 GM 状态；`xa` 则新增为 **三轴 accel colored / AR(1) 状态**。',
        '- `xa` 的离散状态方程明确写进 KF：',
        '  - `xa_k = Fa * xa_{k-1} + w_a`',
        '  - `Fa = diag(exp(-ΔT/τ_a))`',
        '- `xa` 还通过速度误差传播块显式耦合进滤波器：`dv <- dv + Cbn*ΔT*xa`。',
        '- 也就是说，这版不是“把 accel colored 当作额外白噪声处理”，而是把它当成**真实动态状态**在线估。',
        '- 但它仍是 **最小原型**：每轴只先放 1 个一阶 colored state，还不是第四章完整 `ARMA(3,1)` 的 24-state 复刻版。',
        '',
        '## 2. 本版参数设置',
        '',
        f"- `ng_sigma` = `{cfg.ng_sigma_dph}` deg/h",
        f"- `tau_g` = `{cfg.tau_g_s}` s",
        f"- `xa_sigma` = `{cfg.xa_sigma_ug}` ug",
        f"- `tau_a` = `{cfg.tau_a_s}` s",
        f"- `xa_model` = `{cfg.xa_model}`",
        f"- 迭代次数 = `{cfg.max_iter}`，wash_scale = `{cfg.wash_scale}`，carry_att_seed = `{cfg.carry_att_seed}`",
        f"- 备注：{cfg.note}",
        '',
        '## 3. 同口径 seed0 直接对比（同轨迹 / 同初始失准角 / 同 seed）',
        '',
        f"- 真值末态姿态（deg）：`[{seed0_case['trajectory']['truth_final_att_deg'][0]:.6f}, {seed0_case['trajectory']['truth_final_att_deg'][1]:.6f}, {seed0_case['trajectory']['truth_final_att_deg'][2]:.6f}]`",
        f"- 初始失准角（deg）：`[{seed0_case['trajectory']['phi0_deg'][0]:.3f}, {seed0_case['trajectory']['phi0_deg'][1]:.3f}, {seed0_case['trajectory']['phi0_deg'][2]:.3f}]`",
        ref_line,
        '- 为了保持同口径，这次 IMU 仿真输入仍沿用上一版同一条 DAR 轨迹和同一套 `imuadderr` 误差生成；新加的是**滤波器里的 accel colored states**，不是另起一套轨迹。',
        '',
        '| method | final err p/r/y (arcsec) | final norm (arcsec) | final |yaw| (arcsec) | best iter | best iter |yaw| (arcsec) |',
        '|---|---|---:|---:|---:|---:|',
        f"| random-model 15-state | [{rnd0['final_att_err_arcsec'][0]:.2f}, {rnd0['final_att_err_arcsec'][1]:.2f}, {rnd0['final_att_err_arcsec'][2]:.2f}] | {rnd0['final_att_err_norm_arcsec']:.2f} | {rnd0['final_yaw_abs_arcsec']:.2f} | {rnd0['best_iteration_by_yaw']['iteration']} | {rnd0['best_iteration_by_yaw']['yaw_abs_arcsec']:.2f} |",
        f"| accel-colored 18-state | [{acc0['final_att_err_arcsec'][0]:.2f}, {acc0['final_att_err_arcsec'][1]:.2f}, {acc0['final_att_err_arcsec'][2]:.2f}] | {acc0['final_att_err_norm_arcsec']:.2f} | {acc0['final_yaw_abs_arcsec']:.2f} | {acc0['best_iteration_by_yaw']['iteration']} | {acc0['best_iteration_by_yaw']['yaw_abs_arcsec']:.2f} |",
        '',
        f"- **seed0 结论**：最终 yaw 从 15-state 的 `{rnd0['final_yaw_abs_arcsec']:.2f}\"` 变成 `{acc0['final_yaw_abs_arcsec']:.2f}\"`，"
        f"因此 seed0 上是 **{'更好' if seed0_better else '更差'}**。",
        f"- seed0 的总范数从 `{rnd0['final_att_err_norm_arcsec']:.2f}\"` 变成 `{acc0['final_att_err_norm_arcsec']:.2f}\"`。",
        '',
        '## 4. 小规模 seed0-2 复核',
        '',
        '| seed | 15-state final |yaw| | accel18 final |yaw| | gain | 15-state best |yaw| | accel18 best |yaw| |',
        '|---:|---:|---:|---:|---:|---:|',
    ]

    for item in seed_review.per_seed:
        lines.append(
            f"| {item['seed']} | {item['random15']['final_yaw_abs_arcsec']:.2f} | {item['accel18']['final_yaw_abs_arcsec']:.2f} | "
            f"{item['yaw_gain_arcsec']:.2f} | {item['random15']['best_yaw_abs_arcsec']:.2f} | {item['accel18']['best_yaw_abs_arcsec']:.2f} |"
        )

    lines.extend([
        '',
        f"- 15-state mean final |yaw| = `{seed_review.random15_mean_final_yaw_abs_arcsec:.2f}\"`",
        f"- accel18 mean final |yaw| = `{seed_review.accel18_mean_final_yaw_abs_arcsec:.2f}\"`",
        f"- 15-state mean best-iter |yaw| = `{seed_review.random15_mean_best_yaw_abs_arcsec:.2f}\"`",
        f"- accel18 mean best-iter |yaw| = `{seed_review.accel18_mean_best_yaw_abs_arcsec:.2f}\"`",
        f"- 15-state median final |yaw| = `{seed_review.random15_median_final_yaw_abs_arcsec:.2f}\"`",
        f"- accel18 median final |yaw| = `{seed_review.accel18_median_final_yaw_abs_arcsec:.2f}\"`",
        f"- **seed0-2 mean 结论**：相比上一版 15-state 的 `seed0-2 mean≈2.67\"`，这版是 **{'更好' if mean_better else '更差'}**。",
        '',
        '## 5. 必答结论',
        '',
        '- **accel 有色噪声状态是否真的加进滤波器了？**',
        '  - **是**。状态维数从 15 扩到 18，`xa` 进了状态转移矩阵、过程噪声矩阵，并通过 `Phi(dv, xa)` 真正耦合进速度误差传播。',
        '- **新版最终 yaw 大概多少角秒？**',
        f"  - 直接同口径 seed0：final yaw 约 **`{acc0['final_yaw_abs_arcsec']:.2f}\"`**。",
        f"  - 小规模 seed0-2 复核：mean final yaw 约 **`{seed_review.accel18_mean_final_yaw_abs_arcsec:.2f}\"`**。",
        '- **相比 15-state (`seed0 final yaw≈1.02"`, `seed0-2 mean≈2.67"`) 是更好还是更差？**',
        f"  - seed0：**{'更好' if seed0_better else '更差'}**。",
        f"  - seed0-2 mean：**{'更好' if mean_better else '更差'}**。",
    ])

    if not seed0_better or not mean_better:
        lines.extend([
            '- **如果没更好，最可能原因是什么？**',
            '  - 更像是这三个原因之一，且很可能是混合出现：',
            '    1. **有色噪声建模方式不对**：当前 `xa` 只是每轴 1 个一阶 AR(1)/GM 状态，离第四章 `ARMA(3,1)` 还差两级状态维数；',
            '    2. **观测对 accel colored states 不敏感**：当前量测仍是零速速度观测，`xa` 主要通过 `dv` 间接进入，短时内可辨识度可能不够；',
            '    3. **耦合还不够**：现在只加了最小 `Phi(dv, xa)=Cbn*ΔT` 块，尚未做更完整的 colored-accel 建模与反馈链路。',
        ])
    else:
        lines.extend([
            '- **这次为什么可能有改善？**',
            '  - 说明最小版 `xa` 状态确实有机会吸收一部分速度通道里的低频残差；',
            '  - 但因为这还不是完整 24-state，现阶段更像是“方向验证”，不是最终论文口径结果。',
        ])

    lines.extend([
        '',
        '## 6. 我对这版的判断',
        '',
        '- 这版已经满足“**accel colored states 真进滤波器**”这个核心要求。',
        '- 从研究流程看，它是一个 **从 15-state 向第四章 24-state 靠拢的最小 Python 原型**。',
        '- 如果继续往前走，更合理的下一步是：',
        '  1. 把 `xa` 从每轴 1 个状态升级成更接近章节描述的 `r=3` 状态实现；',
        '  2. 重新检查 `xa` 的离散化参数与单位映射；',
        '  3. 若要更公平评估 `xa`，再补一组“输入里也显式带 accel colored truth”的对照实验。',
        '',
        '## 7. 结果文件',
        '',
        f'- 脚本：`{SCRIPT_PATH}`',
        f'- JSON：`{OUT_JSON}`',
        f'- 摘要：`{OUT_MD}`',
    ])

    return '\n'.join([line for line in lines if line != '']) + '\n'



def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cfg = AccelColoredConfig(
        ng_sigma_dph=[0.05, 0.05, 0.05],
        tau_g_s=[300.0, 300.0, 300.0],
        xa_sigma_ug=[0.01, 0.01, 0.01],
        tau_a_s=[100.0, 100.0, 100.0],
        xa_model='per-axis first-order AR(1)/GM colored accel state',
        wash_scale=0.5,
        carry_att_seed=True,
        max_iter=5,
        note=(
            '在 15-state random-model 基础上最小增补 xa(3)；'
            '首轮稳定性复核后，采用较小的 xa_sigma=0.01 ug、tau_a=100 s 作为首个可工作配置，'
            '避免 xa 先验过大时把零速观测通道带偏；暂不一步追到完整 ARMA(3,1) 24-state。'
        ),
    )

    seed_cases = [run_seed_case(seed, cfg) for seed in [0, 1, 2]]
    seed0_case = seed_cases[0]
    seed_review = summarize_seed_review(seed_cases)
    prev15_ref = load_prev15_reference()

    rnd0 = seed0_case['random15']
    acc0 = seed0_case['accel18']

    payload = {
        'script': str(SCRIPT_PATH),
        'previous_15state_script': str(PREV15_PATH),
        'previous_15state_reference_json': str(PREV15_JSON),
        'model': {
            'name': 'dar_accel_colored_18state',
            'state_dimension': 18,
            'state_layout': [
                'phi_E', 'phi_N', 'phi_U',
                'dV_E', 'dV_N', 'dV_U',
                'eb_x', 'eb_y', 'eb_z',
                'db_x', 'db_y', 'db_z',
                'ng_x', 'ng_y', 'ng_z',
                'xa_x', 'xa_y', 'xa_z',
            ],
            'accel_colored_state_really_added': True,
            'accel_colored_truth_injected_into_imu': False,
            'accel_colored_model': {
                'type': 'per_axis_first_order_ar1_gm',
                'sigma_ug': cfg.xa_sigma_ug,
                'tau_s': cfg.tau_a_s,
                'state_equation': 'xa_k = Fa * xa_{k-1} + w_a',
                'coupling': 'dv <- dv + Cbn*dt*xa',
            },
            'gyro_random_model': {
                'type': 'first_order_gauss_markov',
                'sigma_dph': cfg.ng_sigma_dph,
                'tau_s': cfg.tau_g_s,
            },
            'core_difference_vs_15state': [
                'added xa(3) accel colored states into filter state vector',
                'added Phi(dv, xa)=Cbn*dt coupling block',
                'added Fa transition and Q_xa process-noise block',
                'still only washes constant eb/db between iterations; xa remains dynamic state',
            ],
            'chapter4_gap': 'still minimal r=1 per axis, not full ARMA(3,1) / 24-state yet',
        },
        'shared_settings': asdict(cfg),
        'seed0_direct_comparison': seed0_case,
        'seed_review_0_1_2': asdict(seed_review),
        'previous_15state_archive_reference': prev15_ref,
        'judgement': {
            'accel_colored_really_added_to_filter_state': True,
            'seed0_final_yaw_abs_arcsec': acc0['final_yaw_abs_arcsec'],
            'seed0_vs_15state': 'better' if acc0['final_yaw_abs_arcsec'] < rnd0['final_yaw_abs_arcsec'] else 'worse',
            'seed_review_mean_vs_15state': 'better' if seed_review.accel18_mean_final_yaw_abs_arcsec < seed_review.random15_mean_final_yaw_abs_arcsec else 'worse',
            'headline': (
                f"accel 有色噪声版跑通；最终大概 {acc0['final_yaw_abs_arcsec']:.2f} arcsec；"
                f"比 15-state {'更好' if acc0['final_yaw_abs_arcsec'] < rnd0['final_yaw_abs_arcsec'] else '更差'}"
            ),
            'likely_reason_if_not_better': [
                'colored-accel model still too minimal / not the right structure',
                'zero-velocity observation may be insufficiently sensitive to xa states',
                'coupling is still too weak compared with the full chapter-4 model',
            ],
        },
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    OUT_MD.write_text(build_markdown(cfg, seed0_case, seed_review, prev15_ref))

    print(f'[ok] wrote {OUT_JSON}')
    print(f'[ok] wrote {OUT_MD}')
    print(
        '[result] '
        f"seed0 accel18 final yaw = {acc0['final_yaw_abs_arcsec']:.2f} arcsec, "
        f"random15 = {rnd0['final_yaw_abs_arcsec']:.2f} arcsec, "
        f"seed0-2 mean accel18 = {seed_review.accel18_mean_final_yaw_abs_arcsec:.2f} arcsec"
    )


if __name__ == '__main__':
    main()
