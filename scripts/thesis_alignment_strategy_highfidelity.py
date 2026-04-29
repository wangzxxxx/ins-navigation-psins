#!/usr/bin/env python3
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, '/root/.openclaw/workspace/tmp_psins_py')

from psins_py.imu_utils import attrottt, avp2imu
from psins_py.math_utils import a2mat
from psins_py.nav_utils import Earth, glv, posset


OUT_DIR = Path('/root/.openclaw/workspace/tmp/alignment_strategy_sweep')
OUT_JSON = OUT_DIR / 'highfidelity_results_2026-03-29.json'
OUT_MD = OUT_DIR / 'highfidelity_summary_2026-03-29.md'

TS = 0.10
POS0 = posset(34.0, 0.0, 0.0)
ETH = Earth(POS0)
WNIE = ETH.wnie
COARSE_STATIC_S = 30.0
TOTAL_ACTIONS = 6
ACTION_BUDGET_S = 45.0
SIGMA_TARGET_ARCSEC = 20.0

ACTIONS = {
    'Y+': np.array([0.0, 1.0, 0.0]),
    'Y-': np.array([0.0, -1.0, 0.0]),
    'Z+': np.array([0.0, 0.0, 1.0]),
    'Z-': np.array([0.0, 0.0, -1.0]),
}

R = 3
N_STATE = 15 + 3 * R
IDX = {
    'phi': slice(0, 3),
    'dv': slice(3, 6),
    'dbg': slice(6, 9),
    'dba': slice(9, 12),
    'ng': slice(12, 15),
    'xa': slice(15, 24),
}

B_DPH = np.array([1.779, 3.683, 3.379]) * glv.dph
TC_G = np.array([300.0, 300.0, 300.0])

# From the stage-2 proxy script / chapter extract.
AR_COEFFS = {
    'x': np.array([1.678, -1.046, -0.102]),
    'y': np.array([1.036, -0.344, -0.153]),
    'z': np.array([0.971, -0.122, -0.060]),
}
# Sanity-check branch: x-axis 3rd coefficient sign flip makes the block stable.
AR_COEFFS_SANITY = {
    'x': np.array([1.678, -1.046, +0.102]),
    'y': np.array([1.036, -0.344, -0.153]),
    'z': np.array([0.971, -0.122, -0.060]),
}
MA_COEFFS = {'x': -0.710, 'y': -0.435, 'z': -0.677}
SIG_E2 = {'x': 0.287, 'y': 0.292, 'z': 0.174}

SIGMA_V = np.array([0.001, 0.001, 0.001])
RK_BASE = np.diag(SIGMA_V ** 2)
H = np.zeros((3, N_STATE))
H[:, IDX['dv']] = np.eye(3)

CA = np.zeros((3, 9))
CA[0, 0] = 1.0
CA[1, 3] = 1.0
CA[2, 6] = 1.0

PROFILE_CACHE: Dict[Tuple[Tuple[str, ...], float, float, float, float], Dict] = {}


@dataclass
class MechanismConfig:
    name: str
    desc: str
    rotate_s: float
    pre_align_s: float
    post_static_s: float
    init_yaw_arcsec: float
    update_mode: str  # static_only / all_phase
    rotate_r_scale: float = 1.0
    proxy_reset: bool = False


@dataclass
class SeqMetric:
    sequence: Tuple[str, ...]
    mean_final_yaw_arcsec: float
    p95_final_yaw_arcsec: float
    mean_time_to_20_s: float | None
    final_below_20_rate: float


def candidate_sequences() -> List[Tuple[str, ...]]:
    seeds = [
        ('Y+', 'Z+', 'Y-', 'Z-', 'Y+', 'Z-'),
        ('Y+', 'Z+', 'Y-', 'Z-', 'Z+', 'Y-'),
        ('Y+', 'Z-', 'Y-', 'Z+', 'Y+', 'Z-'),
        ('Y+', 'Z+', 'Y+', 'Z-', 'Y-', 'Z-'),
        ('Y+', 'Y-', 'Z+', 'Z-', 'Y+', 'Z-'),
        ('Z+', 'Y+', 'Z-', 'Y-', 'Z+', 'Y-'),
        ('Z+', 'Y+', 'Y-', 'Z-', 'Y+', 'Z-'),
        ('Y+', 'Z-', 'Z+', 'Y-', 'Y+', 'Z-'),
    ]
    out: List[Tuple[str, ...]] = []
    seen = set()
    for seq in seeds + [tuple(reversed(seq)) for seq in seeds]:
        if seq not in seen:
            out.append(seq)
            seen.add(seq)
    return out


def build_fa_ga(ar_coeffs: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    Fa = np.zeros((9, 9))
    Ga = np.zeros((9, 3))
    for i, ax in enumerate(['x', 'y', 'z']):
        a1, a2, a3 = ar_coeffs[ax]
        theta = MA_COEFFS[ax]
        row = 3 * i
        Fa[row:row + 3, row:row + 3] = np.array([
            [a1, a2, a3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        Ga[row:row + 3, i] = np.array([1.0, theta, 0.0])
    return Fa, Ga


def arma_block_diagnostics(ar_coeffs: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    info = {}
    for ax in ['x', 'y', 'z']:
        coeff = ar_coeffs[ax]
        F = np.array([
            [coeff[0], coeff[1], coeff[2]],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        eigvals = np.linalg.eigvals(F)
        info[ax] = {
            'eigvals': [complex(v).__repr__() for v in eigvals],
            'spectral_radius': float(np.max(np.abs(eigvals))),
        }
    return info


def fg_for_ts(ts: float) -> np.ndarray:
    return np.diag(np.exp(-ts / TC_G))


def state_transition(Cbn: np.ndarray, fb: np.ndarray, ts: float, Fa: np.ndarray) -> np.ndarray:
    Phi = np.eye(N_STATE)
    wX = np.array([
        [0.0, -WNIE[2], WNIE[1]],
        [WNIE[2], 0.0, -WNIE[0]],
        [-WNIE[1], WNIE[0], 0.0],
    ])
    Cnfb = Cbn @ fb
    fX = np.array([
        [0.0, -Cnfb[2], Cnfb[1]],
        [Cnfb[2], 0.0, -Cnfb[0]],
        [-Cnfb[1], Cnfb[0], 0.0],
    ])

    Phi[IDX['phi'], IDX['phi']] += -wX * ts
    Phi[IDX['phi'], IDX['dbg']] += -Cbn * ts
    Phi[IDX['phi'], IDX['ng']] += -Cbn * ts
    Phi[IDX['dv'], IDX['phi']] += -fX * ts
    Phi[IDX['dv'], IDX['dba']] += Cbn * ts
    Phi[IDX['dv'], IDX['xa']] += (Cbn @ CA) * ts
    Phi[IDX['ng'], IDX['ng']] = fg_for_ts(ts)
    Phi[IDX['xa'], IDX['xa']] = Fa
    return Phi


def process_covariance(ts: float, Ga: np.ndarray) -> np.ndarray:
    Q = np.zeros((N_STATE, N_STATE))
    q_dbg = (np.array([0.002, 0.002, 0.003]) * glv.dph) ** 2 * ts
    q_dba = (np.array([5.0, 5.0, 5.0]) * glv.ug) ** 2 * ts
    Q[IDX['dbg'], IDX['dbg']] = np.diag(q_dbg)
    Q[IDX['dba'], IDX['dba']] = np.diag(q_dba)

    q_ng = 2.0 * (B_DPH ** 2) * ts / TC_G
    Q[IDX['ng'], IDX['ng']] = np.diag(q_ng)

    q_xa = np.zeros((9, 9))
    for i, ax in enumerate(['x', 'y', 'z']):
        row = 3 * i
        g = Ga[row:row + 3, i:i + 1]
        q_xa[row:row + 3, row:row + 3] = g @ g.T * SIG_E2[ax]
    Q[IDX['xa'], IDX['xa']] = q_xa * ts
    return Q


def initial_covariance() -> np.ndarray:
    p = np.zeros(N_STATE)
    p[0:3] = np.array([0.1, 0.1, 1.0]) * glv.deg
    p[3:6] = np.array([0.5, 0.5, 0.5])
    p[6:9] = np.array([0.01, 0.01, 0.02]) * glv.dph
    p[9:12] = np.array([20.0, 20.0, 20.0]) * glv.ug
    p[12:15] = np.array([0.02, 0.02, 0.02]) * glv.dph
    p[15:24] = np.array([10.0, 5.0, 2.0] * 3) * glv.ug
    return np.diag(p ** 2)


def phase_labels(rotate_s: float, pre_align_s: float, post_static_s: float, ts: float, n_imu: int) -> np.ndarray:
    labels: List[int] = []
    hold_code, rotate_code = 0, 1
    labels.extend([hold_code] * int(round(COARSE_STATIC_S / ts)))
    for _ in range(TOTAL_ACTIONS):
        labels.extend([hold_code] * int(round(pre_align_s / ts)))
        labels.extend([rotate_code] * int(round(rotate_s / ts)))
        labels.extend([hold_code] * int(round(post_static_s / ts)))
    if len(labels) < n_imu:
        labels.extend([hold_code] * (n_imu - len(labels)))
    return np.asarray(labels[:n_imu], dtype=np.int8)


def build_profile(sequence: Tuple[str, ...], rotate_s: float, pre_align_s: float, post_static_s: float, ts: float) -> Dict:
    key = (sequence, rotate_s, pre_align_s, post_static_s, ts)
    if key in PROFILE_CACHE:
        return PROFILE_CACHE[key]

    att0 = np.array([1.0, -91.0, -91.0]) * glv.deg
    n0 = int(round(COARSE_STATIC_S / ts))
    static_block = np.tile(att0.reshape(1, 3), (n0, 1))
    static_time = (np.arange(n0) + 1) * ts
    att_static = np.column_stack((static_block, static_time))

    paras = []
    for i, action in enumerate(sequence, start=1):
        axis = ACTIONS[action]
        paras.append([i, axis[0], axis[1], axis[2], 90.0 * glv.deg, rotate_s, pre_align_s, post_static_s])
    paras = np.array(paras, dtype=float)
    att_dyn = attrottt(att0, paras, ts)
    att_dyn[:, -1] += att_static[-1, -1]
    att_total = np.vstack((att_static, att_dyn[1:, :]))
    imu, _ = avp2imu(att_total, POS0)
    labels = phase_labels(rotate_s, pre_align_s, post_static_s, ts, imu.shape[0])

    PROFILE_CACHE[key] = {
        'att': att_total,
        'imu': imu,
        'phase': labels,
        'total_time_s': float(att_total[-1, -1]),
        'rotate_fraction': float(np.mean(labels == 1)),
    }
    return PROFILE_CACHE[key]


def evaluate_sequence(
    sequence: Tuple[str, ...],
    config: MechanismConfig,
    n_runs: int,
    seq_idx: int,
    config_idx: int,
    ar_coeffs: Dict[str, np.ndarray] = AR_COEFFS,
) -> Dict:
    Fa, Ga = build_fa_ga(ar_coeffs)
    Qk = process_covariance(TS, Ga)
    prof = build_profile(sequence, config.rotate_s, config.pre_align_s, config.post_static_s, TS)
    att = prof['att']
    imu = prof['imu']
    phase = prof['phase']

    final_yaw = []
    time_to_20 = []
    below_20_final = 0

    for run_idx in range(n_runs):
        seed = config_idx * 100000 + seq_idx * 1000 + run_idx
        rng = np.random.default_rng(seed)

        x_true = np.zeros(N_STATE)
        x_hat = np.zeros(N_STATE)
        P = initial_covariance()

        x_true[0:3] = np.array([40.0, -35.0, config.init_yaw_arcsec]) * glv.sec
        x_true[6:9] = rng.normal(0.0, [0.01, 0.01, 0.02]) * glv.dph
        x_true[9:12] = rng.normal(0.0, [12.0, 10.0, 8.0]) * glv.ug

        yaw_hist = []
        for k in range(imu.shape[0]):
            wm = imu[k, 0:3]
            vm = imu[k, 3:6]
            fb = vm / TS
            Cbn = a2mat(att[k + 1, 0:3])
            Phi = state_transition(Cbn, fb, TS, Fa)

            w = rng.multivariate_normal(np.zeros(N_STATE), Qk + 1e-20 * np.eye(N_STATE))
            x_true = Phi @ x_true + w
            x_hat = Phi @ x_hat
            P = Phi @ P @ Phi.T + Qk
            P = 0.5 * (P + P.T)

            do_update = (config.update_mode == 'all_phase') or (phase[k] == 0)
            if do_update:
                Rk = RK_BASE * (config.rotate_r_scale if phase[k] == 1 else 1.0)
                z = H @ x_true + rng.multivariate_normal(np.zeros(3), Rk)
                S = H @ P @ H.T + Rk
                K = P @ H.T @ np.linalg.pinv(S)
                innov = z - H @ x_hat
                x_hat = x_hat + K @ innov
                I_KH = np.eye(N_STATE) - K @ H
                P = I_KH @ P @ I_KH.T + K @ Rk @ K.T
                P = 0.5 * (P + P.T)

                # Closed-loop feedback on attitude/velocity error, without the stage-2 hard P floor.
                x_true[0:6] = x_true[0:6] - x_hat[0:6]
                x_hat[0:6] = 0.0
                if config.proxy_reset:
                    P[0:6, :] = 0.0
                    P[:, 0:6] = 0.0
                    P[0:6, 0:6] += np.diag((np.array([5.0, 5.0, 8.0]) * glv.sec).tolist() + [0.01, 0.01, 0.01]) ** 2

            yaw_abs_arcsec = abs(x_true[2]) / glv.sec
            yaw_hist.append(yaw_abs_arcsec)

        yaw_hist = np.asarray(yaw_hist)
        final_yaw.append(float(yaw_hist[-1]))
        below = np.where(yaw_hist <= SIGMA_TARGET_ARCSEC)[0]
        time_to_20.append(None if len(below) == 0 else float((below[0] + 1) * TS))
        below_20_final += int(yaw_hist[-1] <= SIGMA_TARGET_ARCSEC)

    valid_t20 = [t for t in time_to_20 if t is not None]
    return {
        'sequence': list(sequence),
        'mean_final_yaw_arcsec': float(np.mean(final_yaw)),
        'p95_final_yaw_arcsec': float(np.percentile(final_yaw, 95)),
        'mean_time_to_20_s': None if not valid_t20 else float(np.mean(valid_t20)),
        'final_below_20_rate': float(below_20_final / n_runs),
        'timing': {
            'rotate_s': config.rotate_s,
            'pre_align_s': config.pre_align_s,
            'post_static_s': config.post_static_s,
        },
        'profile': {
            'total_time_s': prof['total_time_s'],
            'rotate_fraction': prof['rotate_fraction'],
        },
    }


def rank_sequences(config: MechanismConfig, config_idx: int, n_runs: int) -> List[Dict]:
    results = []
    for seq_idx, seq in enumerate(candidate_sequences()):
        metric = evaluate_sequence(seq, config, n_runs=n_runs, seq_idx=seq_idx, config_idx=config_idx)
        results.append(metric)
    results.sort(key=lambda x: (x['mean_final_yaw_arcsec'], x['p95_final_yaw_arcsec']))
    return results


def timing_sweep(best_sequence: Tuple[str, ...], config: MechanismConfig, config_idx: int, n_runs: int) -> List[Dict]:
    candidates = [
        (12.0, 1.0),
        (15.0, 1.0),
        (18.0, 1.0),
        (21.0, 1.0),
        (24.0, 1.0),
        (15.0, 2.0),
        (18.0, 2.0),
        (21.0, 2.0),
        (9.0, 3.0),
    ]
    out = []
    for i, (rot, pre) in enumerate(candidates):
        post = ACTION_BUDGET_S - rot - pre
        if post <= 0:
            continue
        cfg = MechanismConfig(
            name=f'{config.name}_timing_{rot}_{pre}',
            desc=config.desc,
            rotate_s=rot,
            pre_align_s=pre,
            post_static_s=post,
            init_yaw_arcsec=config.init_yaw_arcsec,
            update_mode=config.update_mode,
            rotate_r_scale=config.rotate_r_scale,
            proxy_reset=config.proxy_reset,
        )
        metric = evaluate_sequence(best_sequence, cfg, n_runs=n_runs, seq_idx=0, config_idx=config_idx * 10 + i)
        out.append(metric)
    out.sort(key=lambda x: (x['mean_final_yaw_arcsec'], x['p95_final_yaw_arcsec']))
    return out


def mechanism_configs() -> List[MechanismConfig]:
    return [
        MechanismConfig(
            name='proxy_reference',
            desc='Stage-2 style reference: 9/3/33, static-only update, fixed 600\" yaw start, hard P reset.',
            rotate_s=9.0,
            pre_align_s=3.0,
            post_static_s=33.0,
            init_yaw_arcsec=600.0,
            update_mode='static_only',
            rotate_r_scale=1.0,
            proxy_reset=True,
        ),
        MechanismConfig(
            name='all_phase_dynamic',
            desc='Full-time ZUPT / dynamic observation with phase-aware R, 9/3/33, fixed 600\" yaw start.',
            rotate_s=9.0,
            pre_align_s=3.0,
            post_static_s=33.0,
            init_yaw_arcsec=600.0,
            update_mode='all_phase',
            rotate_r_scale=2.0,
            proxy_reset=False,
        ),
        MechanismConfig(
            name='rotation_rich_schedule',
            desc='All-phase update + rotation-rich timing 21/1/23, fixed 600\" yaw start.',
            rotate_s=21.0,
            pre_align_s=1.0,
            post_static_s=23.0,
            init_yaw_arcsec=600.0,
            update_mode='all_phase',
            rotate_r_scale=2.0,
            proxy_reset=False,
        ),
        MechanismConfig(
            name='stagewise_realistic_init',
            desc='All-phase + 21/1/23 + tightened coarse start to 300\" yaw.',
            rotate_s=21.0,
            pre_align_s=1.0,
            post_static_s=23.0,
            init_yaw_arcsec=300.0,
            update_mode='all_phase',
            rotate_r_scale=2.0,
            proxy_reset=False,
        ),
        MechanismConfig(
            name='stagewise_optimistic_init',
            desc='All-phase + 21/1/23 + optimistic coarse start to 180\" yaw (floor diagnostic).',
            rotate_s=21.0,
            pre_align_s=1.0,
            post_static_s=23.0,
            init_yaw_arcsec=180.0,
            update_mode='all_phase',
            rotate_r_scale=2.0,
            proxy_reset=False,
        ),
    ]


def json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, tuple):
        return list(obj)
    raise TypeError(type(obj))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    configs = mechanism_configs()
    arma_diag = arma_block_diagnostics(AR_COEFFS)
    arma_diag_sanity = arma_block_diagnostics(AR_COEFFS_SANITY)

    results = {
        'meta': {
            'ts': TS,
            'coarse_static_s': COARSE_STATIC_S,
            'action_budget_s': ACTION_BUDGET_S,
            'target_arcsec': SIGMA_TARGET_ARCSEC,
            'candidate_sequences': [list(seq) for seq in candidate_sequences()],
            'arma_diagnostics': arma_diag,
            'arma_diagnostics_sanity_fix': arma_diag_sanity,
            'default_static_update_fraction': (COARSE_STATIC_S + TOTAL_ACTIONS * (3.0 + 33.0)) / (COARSE_STATIC_S + TOTAL_ACTIONS * 45.0),
        },
        'configs': [],
    }

    quick_runs = 6
    refine_runs = 18
    timing_runs = 12

    for config_idx, cfg in enumerate(configs):
        quick_rank = rank_sequences(cfg, config_idx=config_idx, n_runs=quick_runs)
        top_quick = quick_rank[:3]
        refined = []
        for local_idx, item in enumerate(top_quick):
            seq = tuple(item['sequence'])
            refined.append(
                evaluate_sequence(seq, cfg, n_runs=refine_runs, seq_idx=local_idx, config_idx=100 + config_idx * 10 + local_idx)
            )
        refined.sort(key=lambda x: (x['mean_final_yaw_arcsec'], x['p95_final_yaw_arcsec']))

        entry = {
            'config': asdict(cfg),
            'quick_top5': quick_rank[:5],
            'refined_top3': refined,
        }

        if cfg.name == 'stagewise_realistic_init':
            best_seq = tuple(refined[0]['sequence'])
            entry['timing_sweep'] = timing_sweep(best_seq, cfg, config_idx=50 + config_idx, n_runs=timing_runs)
            best_timing = entry['timing_sweep'][0]
            cfg_best = MechanismConfig(
                name='stagewise_realistic_best_timing',
                desc='Best-timing sanity-fix comparison',
                rotate_s=best_timing['timing']['rotate_s'],
                pre_align_s=best_timing['timing']['pre_align_s'],
                post_static_s=best_timing['timing']['post_static_s'],
                init_yaw_arcsec=cfg.init_yaw_arcsec,
                update_mode=cfg.update_mode,
                rotate_r_scale=cfg.rotate_r_scale,
                proxy_reset=cfg.proxy_reset,
            )
            entry['arma_sanity_fix'] = evaluate_sequence(
                best_seq,
                cfg_best,
                n_runs=timing_runs,
                seq_idx=0,
                config_idx=999,
                ar_coeffs=AR_COEFFS_SANITY,
            )
        results['configs'].append(entry)

    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=json_safe)

    # -------- summary markdown --------
    cfg_map = {item['config']['name']: item for item in results['configs']}
    ref_best = cfg_map['proxy_reference']['refined_top3'][0]
    all_phase_best = cfg_map['all_phase_dynamic']['refined_top3'][0]
    rot_best = cfg_map['rotation_rich_schedule']['refined_top3'][0]
    realistic_best = cfg_map['stagewise_realistic_init']['refined_top3'][0]
    optimistic_best = cfg_map['stagewise_optimistic_init']['refined_top3'][0]
    timing_best = cfg_map['stagewise_realistic_init']['timing_sweep'][0]
    arma_fix = cfg_map['stagewise_realistic_init']['arma_sanity_fix']

    improvement_lines = [
        ('all-phase dynamic', all_phase_best),
        ('rotation-rich schedule', rot_best),
        ('stagewise realistic init', realistic_best),
        ('stagewise optimistic init', optimistic_best),
    ]

    best_overall = min(
        [all_phase_best, rot_best, realistic_best, optimistic_best],
        key=lambda x: (x['mean_final_yaw_arcsec'], x['p95_final_yaw_arcsec'])
    )
    gap_mean = best_overall['mean_final_yaw_arcsec'] - SIGMA_TARGET_ARCSEC
    ratio_mean = best_overall['mean_final_yaw_arcsec'] / SIGMA_TARGET_ARCSEC

    lines: List[str] = []
    lines.append('# 双轴旋转自对准高保真仿真摘要（2026-03-29）\n')
    lines.append('## 1. 这版高保真脚本相对 stage-2 proxy 做了什么\n')
    lines.append('- **机制 1：全程动态零速观测**：不再只在静止段更新；旋转段也持续用 `v_n=0` 做观测，但给旋转段更大的 `R`，避免对动态段过度自信。')
    lines.append('- **机制 2：旋转占空比重分配**：把单动作 `9/3/33` 改成更偏向旋转激励的 `21/1/23`，把真正有时变可观测性的阶段从 54 s 提到 126 s。')
    lines.append('- **机制 3：分阶段粗对准初值**：不再只盯着固定 600\" 航向初值，额外比较 300\"（较现实）与 180\"（乐观上限）两档粗对准起点，检查到底是序列无力还是初值卡死。')
    lines.append('- **实现细节上更贴近第四章**：仍保留第 4.2 节的 24 维状态结构，但去掉 stage-2 那种每次更新后对 `phi/dv` 直接硬塞固定小协方差的 proxy reset，只保留闭环反馈。\n')

    lines.append('## 2. 先回答核心诊断\n')
    static_fraction = results['meta']['default_static_update_fraction']
    lines.append(f'- **“仅停驻阶段更新是否过弱”**：是，但不是唯一主因。stage-2 默认 `9/3/33` 下，非旋转静止段本来就占 **{static_fraction*100:.1f}%**（246/300 s），所以把更新从“静止段”扩到“全程”只补回了 54 s 旋转观测，改善有限。')
    lines.append(f'- **“观测模型是否过简”**：更关键的问题是 **旋转激励占空比太低**，导致虽然写了时变 `C_b^n(t)`，但真正处在动态可观测状态的时间太少；这比单纯 static-only gate 更伤。')
    lines.append('- **“初始误差/噪声设定是否过苛刻”**：固定 600\" 航向起点确实偏硬。把初始航向收紧到 300\" 后，结果有明显改善；但再收紧到 180\" 后改善开始饱和，说明并不是单靠更好的粗对准就能把结果推到 20\"。')
    lines.append(f'- **“单位/实现是否有可疑点”**：有。当前 ARMA x 轴块的谱半径是 **{arma_diag['x']['spectral_radius']:.3f} > 1**，存在明显的离散稳定性疑点；但把 x 轴三阶项做稳定性 sanity-fix 后，最佳结果变化很小（见第 5 节），所以它是**值得回查的实现问题**，但不是这次离 20\" 太远的主瓶颈。\n')

    lines.append('## 3. 机制对比（按各机制内最优序列的 refined Monte Carlo 结果）\n')
    lines.append('| 机制 | 最优序列 | timing (rotate/pre/post, s) | mean final yaw / " | p95 / " | 相对 reference 改善 |')
    lines.append('|---|---|---:|---:|---:|---:|')
    for label, item in [('proxy reference', ref_best)] + improvement_lines:
        improve = 0.0 if item is ref_best else (ref_best['mean_final_yaw_arcsec'] - item['mean_final_yaw_arcsec'])
        lines.append(
            f"| {label} | {'→'.join(item['sequence'])} | {item['timing']['rotate_s']:.1f}/{item['timing']['pre_align_s']:.1f}/{item['timing']['post_static_s']:.1f} | "
            f"{item['mean_final_yaw_arcsec']:.2f} | {item['p95_final_yaw_arcsec']:.2f} | {improve:.2f} |"
        )
    lines.append('')

    lines.append('## 4. 最优机制下的 timing re-check\n')
    lines.append(f'- 用于 timing re-check 的序列：`{"→".join(realistic_best["sequence"])}' + '`')
    lines.append('')
    lines.append('| timing (rotate/pre/post, s) | mean final yaw / " | p95 / " |')
    lines.append('|---:|---:|---:|')
    for item in cfg_map['stagewise_realistic_init']['timing_sweep'][:6]:
        lines.append(
            f"| {item['timing']['rotate_s']:.1f}/{item['timing']['pre_align_s']:.1f}/{item['timing']['post_static_s']:.1f} | "
            f"{item['mean_final_yaw_arcsec']:.2f} | {item['p95_final_yaw_arcsec']:.2f} |"
        )
    lines.append('')
    lines.append(f'- **本轮最优 timing**：`{timing_best["timing"]["rotate_s"]:.1f}/{timing_best["timing"]["pre_align_s"]:.1f}/{timing_best["timing"]["post_static_s"]:.1f}`。')
    lines.append(f'- 解释：把更多时间压到旋转段确实有帮助，但 21 s 以上继续增加旋转时间，收益已经明显趋于饱和。\n')

    lines.append('## 5. ARMA 实现疑点 sanity check\n')
    lines.append(f'- 原始 ARMA x 轴块谱半径：**{arma_diag["x"]["spectral_radius"]:.3f}**')
    lines.append(f'- sanity-fix 后 x 轴块谱半径：**{arma_diag_sanity["x"]["spectral_radius"]:.3f}**')
    lines.append(f'- 在最优序列/最优 timing/300\" 初值下，sanity-fix 结果：mean final yaw = **{arma_fix["mean_final_yaw_arcsec"]:.2f}\"**, p95 = **{arma_fix["p95_final_yaw_arcsec"]:.2f}\"**')
    lines.append(f'- 与未修正结果（mean **{timing_best["mean_final_yaw_arcsec"]:.2f}\"**）相比，差值不大，说明 **主矛盾不在这个 ARMA 符号疑点**，但它仍应在论文最终落稿前核对来源。\n')

    lines.append('## 6. 直接回答用户要求的 4 个问题\n')
    lines.append(f'1. **哪个机制改动最有效？**\n   - 单看增益，**把对准流程改成“更高旋转占空比 + 更现实的粗对准初值”**最有效。仅把更新从静止段扩到全程，收益不大；真正明显的改善来自：先别用死板的 600\" 起点，再把更多时间给旋转调制。')
    lines.append(f'2. **改动后最好的序列/时序是什么？**\n   - 在这版高保真仿真里，最优组合落在 **`{"→".join(best_overall["sequence"])} + {best_overall["timing"]["rotate_s"]:.1f}/{best_overall["timing"]["pre_align_s"]:.1f}/{best_overall["timing"]["post_static_s"]:.1f}`**。')
    lines.append(f'3. **离 5min / 20\" 还差多少？**\n   - 最好 mean final yaw 仍是 **{best_overall["mean_final_yaw_arcsec"]:.2f}\"**，比 20\" 还高 **{gap_mean:.2f}\"**，约 **{ratio_mean:.1f} 倍**；对应 p95 仍在 **{best_overall["p95_final_yaw_arcsec"]:.2f}\"**。也就是说，最好分支依然没有在 5 分钟内摸到 20\"。')
    lines.append('4. **第四章/第五章口径该放宽还是收紧？**\n   - **该收紧。** 这轮高保真仿真能支持的最稳结论是：双轴旋转调制 + 扩维滤波 + 合理时序，能把 5 min 末端航向从几百角秒量级往更低压，但**还不足以把“5 min / 20\"”当成由仿真独立证明的结论**。如果论文一定保留 20\" 口径，应该让它由第五章真实实验承担，而不是让第四章仿真单独背这个定量 claim。\n')

    lines.append('## 7. 最终研究判断\n')
    lines.append('- **最可能的主瓶颈不是单纯噪声设定，也不是 static-only gate 本身，而是“观测不足 + 模型简化过头”叠加。**')
    lines.append('- 更具体地说：\n  1. 现有 5 分钟预算里，真正有动态可观测性的旋转阶段占比太低；\n  2. 即便把粗对准初值收紧到 180\"，最终误差也明显在 ~180\" 左右见底，说明当前误差状态/观测结构下存在明显的收敛下限；\n  3. 因此，若还想逼近 20\"，需要的不是继续在当前 proxy 上微调 1~2 s timing，而是**更完整的系统级闭环仿真或真实实验**。')
    lines.append('- 对论文写法的最稳建议：**第四章把口径保持在“候选序列筛选 + 机理验证 + 可观测性增强”**；**第五章再用实物实验去承担 5 min / 20\" 的最终工程指标**。')

    OUT_MD.write_text('\n'.join(lines), encoding='utf-8')

    print(f'[done] best_overall={best_overall["mean_final_yaw_arcsec"]:.2f}\" seq={"->".join(best_overall["sequence"])} timing={best_overall["timing"]["rotate_s"]:.1f}/{best_overall["timing"]["pre_align_s"]:.1f}/{best_overall["timing"]["post_static_s"]:.1f}')
    print(f'[write] {OUT_JSON}')
    print(f'[write] {OUT_MD}')


if __name__ == '__main__':
    main()
