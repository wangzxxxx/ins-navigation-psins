import itertools
import json
import math
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, '/root/.openclaw/workspace/tmp_psins_py')

from psins_py.nav_utils import glv, posset, Earth
from psins_py.imu_utils import attrottt, avp2imu
from psins_py.math_utils import a2mat

SEARCH_TS = 0.10
MC_TS = 0.05
POS0 = posset(34.0, 0.0, 0.0)
ETH = Earth(POS0)
WNIE = ETH.wnie
GN = ETH.gn

COARSE_STATIC_S = 30.0
ROT_S = 9.0
PRE_ALIGN_S = 3.0
POST_STATIC_S = 33.0  # total 45s per action => 30 + 6*45 = 300s
TOTAL_ACTIONS = 6

# Candidate dual-axis actions: Y+/Y-/Z+/Z-
ACTIONS = {
    'Y+': np.array([0.0, 1.0, 0.0]),
    'Y-': np.array([0.0, -1.0, 0.0]),
    'Z+': np.array([0.0, 0.0, 1.0]),
    'Z-': np.array([0.0, 0.0, -1.0]),
}

# Alignment filter dimensions
R = 3  # ARMA order used in chapter text
N_STATE = 15 + 3 * R
IDX = {
    'phi': slice(0, 3),
    'dv': slice(3, 6),
    'dbg': slice(6, 9),
    'dba': slice(9, 12),
    'ng': slice(12, 15),
    'xa': slice(15, 24),
}

# Noise identification values (from thesis draft)
B_DPH = np.array([1.779, 3.683, 3.379]) * glv.dph
TC_G = np.array([300.0, 300.0, 300.0])

AR_COEFFS = {
    'x': np.array([1.678, -1.046, -0.102]),
    'y': np.array([1.036, -0.344, -0.153]),
    'z': np.array([0.971, -0.122, -0.060]),
}
MA_COEFFS = {'x': -0.710, 'y': -0.435, 'z': -0.677}
SIG_E2 = {'x': 0.287, 'y': 0.292, 'z': 0.174}

SIGMA_V = np.array([0.001, 0.001, 0.001])
SIGMA_20ARCSEC = 20.0 * glv.sec


def build_fa_ga():
    Fa = np.zeros((9, 9))
    Ga = np.zeros((9, 3))
    axes = ['x', 'y', 'z']
    for i, ax in enumerate(axes):
        a1, a2, a3 = AR_COEFFS[ax]
        theta = MA_COEFFS[ax]
        row = 3 * i
        Fa[row:row + 3, row:row + 3] = np.array([
            [a1, a2, a3],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        Ga[row:row + 3, i] = np.array([1.0, theta, 0.0])
    return Fa, Ga


FA, GA = build_fa_ga()
CA = np.zeros((3, 9))
CA[0, 0] = 1.0
CA[1, 3] = 1.0
CA[2, 6] = 1.0
def fg_for_ts(ts):
    return np.diag(np.exp(-ts / TC_G))


def state_transition(Cbn, fb, ts):
    Phi = np.eye(N_STATE)
    FG = fg_for_ts(ts)
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

    Phi[IDX['ng'], IDX['ng']] = FG
    Phi[IDX['xa'], IDX['xa']] = FA
    return Phi


def process_covariance(ts):
    Q = np.zeros((N_STATE, N_STATE))
    # residual constant terms: small random walk
    q_dbg = (np.array([0.002, 0.002, 0.003]) * glv.dph) ** 2 * ts
    q_dba = (np.array([5.0, 5.0, 5.0]) * glv.ug) ** 2 * ts
    Q[IDX['dbg'], IDX['dbg']] = np.diag(q_dbg)
    Q[IDX['dba'], IDX['dba']] = np.diag(q_dba)

    # GM process driving noise
    q_ng = 2.0 * (B_DPH ** 2) * ts / TC_G
    Q[IDX['ng'], IDX['ng']] = np.diag(q_ng)

    # ARMA driving white noise
    q_xa = np.zeros((9, 9))
    for i, ax in enumerate(['x', 'y', 'z']):
        row = 3 * i
        g = GA[row:row + 3, i:i + 1]
        q_xa[row:row + 3, row:row + 3] = g @ g.T * SIG_E2[ax]
    Q[IDX['xa'], IDX['xa']] = q_xa * ts
    return Q


RK = np.diag(SIGMA_V ** 2)
H = np.zeros((3, N_STATE))
H[:, IDX['dv']] = np.eye(3)


def initial_covariance():
    p = np.zeros(N_STATE)
    p[0:3] = np.array([0.1, 0.1, 1.0]) * glv.deg
    p[3:6] = np.array([0.5, 0.5, 0.5])
    p[6:9] = np.array([0.01, 0.01, 0.02]) * glv.dph
    p[9:12] = np.array([20.0, 20.0, 20.0]) * glv.ug
    p[12:15] = np.array([0.02, 0.02, 0.02]) * glv.dph
    p[15:24] = np.array([10.0, 5.0, 2.0] * 3) * glv.ug
    return np.diag(p ** 2)


def build_attitude_profile(action_seq, att0, ts):
    n0 = int(round(COARSE_STATIC_S / ts))
    static_block = np.tile(att0.reshape(1, 3), (n0, 1))
    static_time = (np.arange(n0) + 1) * ts
    att_static = np.column_stack((static_block, static_time))

    paras = []
    for i, action in enumerate(action_seq, start=1):
        axis = ACTIONS[action]
        paras.append([i, axis[0], axis[1], axis[2], 90.0 * glv.deg, ROT_S, PRE_ALIGN_S, POST_STATIC_S])
    paras = np.array(paras, dtype=float)
    att_dyn = attrottt(att0, paras, ts)
    att_dyn[:, -1] += att_static[-1, -1]
    att_total = np.vstack((att_static, att_dyn[1:, :]))
    return att_total


def covariance_score(action_seq, att0, ts):
    att = build_attitude_profile(action_seq, att0, ts)
    imu, _ = avp2imu(att, POS0)

    P = initial_covariance()
    QK = process_covariance(ts)
    sigma_hist = []
    static_mask = []

    for k in range(imu.shape[0]):
        wm = imu[k, 0:3]
        vm = imu[k, 3:6]
        wb = wm / ts
        fb = vm / ts
        Cbn = a2mat(att[k + 1, 0:3])  # att matches imu shifted by 1 sample
        Phi = state_transition(Cbn, fb, ts)
        P = Phi @ P @ Phi.T + QK
        P = 0.5 * (P + P.T)

        is_static = np.linalg.norm(wb) < 20.0 * glv.dph
        static_mask.append(bool(is_static))
        if is_static:
            S = H @ P @ H.T + RK
            K = P @ H.T @ np.linalg.pinv(S)
            I_KH = np.eye(N_STATE) - K @ H
            P = I_KH @ P @ I_KH.T + K @ RK @ K.T
            P = 0.5 * (P + P.T)
        diagP = np.maximum(np.diag(P)[0:3], 0.0)
        sigma_hist.append(np.sqrt(diagP))

    sigma_hist = np.array(sigma_hist)
    yaw_sigma = sigma_hist[:, 2]
    final_yaw = float(yaw_sigma[-1])
    final_roll = float(sigma_hist[-1, 0])
    final_pitch = float(sigma_hist[-1, 1])

    below = np.where(yaw_sigma <= SIGMA_20ARCSEC)[0]
    t20 = None if len(below) == 0 else float((below[0] + 1) * ts)

    # favor balanced roll/pitch/yaw + early yaw lock
    score = final_yaw * (1.0 + 0.15 * (final_roll + final_pitch) / max(final_yaw, 1e-12))
    if t20 is None:
        score *= 1.5
    else:
        score *= (1.0 + t20 / 300.0)

    axis_use = {a[0] for a in action_seq}
    sign_use = set(action_seq)
    if len(axis_use) < 2:
        score *= 2.0
    if len(sign_use) < 4:
        score *= 1.2

    return {
        'sequence': action_seq,
        'final_yaw_sigma_arcsec': final_yaw / glv.sec,
        'final_roll_sigma_arcsec': final_roll / glv.sec,
        'final_pitch_sigma_arcsec': final_pitch / glv.sec,
        'time_to_20arcsec_s': t20,
        'score': score,
        'sigma_hist_arcsec': sigma_hist / glv.sec,
        'total_time_s': float(att[-1, -1]),
    }


def monte_carlo_validate(result, att0, n_runs=30, seed=42, ts=MC_TS):
    rng = np.random.default_rng(seed)
    att = build_attitude_profile(result['sequence'], att0, ts)
    imu, _ = avp2imu(att, POS0)
    n = imu.shape[0]
    QK = process_covariance(ts)

    yaw_abs_err = np.zeros((n_runs, n))
    final_stats = []

    for run in range(n_runs):
        x_true = np.zeros(N_STATE)
        x_hat = np.zeros(N_STATE)
        P = initial_covariance()

        # initial errors after coarse alignment
        x_true[0:3] = np.array([40.0, -35.0, 600.0]) * glv.sec
        x_true[6:9] = np.array([0.010, -0.012, 0.018]) * glv.dph
        x_true[9:12] = np.array([12.0, -10.0, 8.0]) * glv.ug

        for k in range(n):
            wm = imu[k, 0:3]
            vm = imu[k, 3:6]
            wb = wm / ts
            fb = vm / ts
            Cbn = a2mat(att[k + 1, 0:3])
            Phi = state_transition(Cbn, fb, ts)

            # sample process noise from diagonal Q approximation
            w = rng.multivariate_normal(np.zeros(N_STATE), QK + 1e-18 * np.eye(N_STATE))
            x_true = Phi @ x_true + w
            x_hat = Phi @ x_hat
            P = Phi @ P @ Phi.T + QK
            P = 0.5 * (P + P.T)

            is_static = np.linalg.norm(wb) < 20.0 * glv.dph
            if is_static:
                v = rng.multivariate_normal(np.zeros(3), RK)
                z = H @ x_true + v
                S = H @ P @ H.T + RK
                K = P @ H.T @ np.linalg.pinv(S)
                innov = z - H @ x_hat
                x_hat = x_hat + K @ innov
                I_KH = np.eye(N_STATE) - K @ H
                P = I_KH @ P @ I_KH.T + K @ RK @ K.T
                P = 0.5 * (P + P.T)

                # feedback-reset for phi and dv only
                x_true[0:3] = x_true[0:3] - x_hat[0:3]
                x_true[3:6] = x_true[3:6] - x_hat[3:6]
                x_hat[0:6] = 0.0
                P[0:6, :] = 0.0
                P[:, 0:6] = 0.0
                P[0:6, 0:6] += np.diag((np.array([5.0, 5.0, 8.0]) * glv.sec).tolist() + [0.01, 0.01, 0.01]) ** 2

            yaw_abs_err[run, k] = abs(x_true[2]) / glv.sec

        final_stats.append(float(yaw_abs_err[run, -1]))

    mean_yaw = yaw_abs_err.mean(axis=0)
    p95_yaw = np.percentile(yaw_abs_err, 95, axis=0)
    below = np.where(mean_yaw <= 20.0)[0]
    t20_mean = None if len(below) == 0 else float((below[0] + 1) * ts)
    return {
        'mean_final_yaw_err_arcsec': float(np.mean(final_stats)),
        'p95_final_yaw_err_arcsec': float(np.percentile(final_stats, 95)),
        'mean_time_to_20arcsec_s': t20_mean,
        'mean_yaw_curve_arcsec': mean_yaw.tolist(),
        'p95_yaw_curve_arcsec': p95_yaw.tolist(),
        'time_s': (np.arange(n) + 1).astype(float).tolist(),
    }


def candidate_sequences():
    # hand-crafted physically meaningful dual-axis modulation families
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
    # add reversed-path companions
    cands = []
    for s in seeds:
        cands.append(s)
        cands.append(tuple(reversed(s)))
    # unique preserve order
    uniq = []
    seen = set()
    for s in cands:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq


def main():
    out_dir = Path('/root/.openclaw/workspace/tmp/alignment_strategy_sweep')
    out_dir.mkdir(parents=True, exist_ok=True)

    att0 = np.array([1.0, -91.0, -91.0]) * glv.deg
    results = []
    cands = candidate_sequences()
    print(f'Starting focused sweep over {len(cands)} candidate dual-axis sequences...')

    for i_seq, seq in enumerate(cands, start=1):
        print(f'  evaluating candidate {i_seq}/{len(cands)}: {seq}')
        res = covariance_score(seq, att0, SEARCH_TS)
        results.append(res)

    print(f'Focused sweep done. valid candidates={len(results)}')
    results.sort(key=lambda x: x['score'])
    top = results[:12]
    print('Top 5 covariance-ranked strategies after focused sweep:')
    for i, r in enumerate(top[:5], start=1):
        print(f"  #{i}: {r['sequence']} | yaw_sigma={r['final_yaw_sigma_arcsec']:.2f} arcsec | t20={r['time_to_20arcsec_s']} s | score={r['score']:.4e}")

    validated = []
    print('Starting Monte Carlo validation for top strategies...')
    for i_val, res in enumerate(top[:3], start=1):
        print(f'  validating top-{i_val}: {res["sequence"]}')
        mc = monte_carlo_validate(res, att0, n_runs=20, seed=42 + i_val)
        merged = {k: v for k, v in res.items() if k != 'sigma_hist_arcsec'}
        merged['monte_carlo'] = mc
        validated.append(merged)

    # enrich best result with sigma curve for plotting
    best_cov = results[0]
    best = next(v for v in validated if tuple(v['sequence']) == tuple(best_cov['sequence']))

    with open(out_dir / 'strategy_rankings.json', 'w', encoding='utf-8') as f:
        json.dump({'top12': [
            {
                'sequence': r['sequence'],
                'final_yaw_sigma_arcsec': r['final_yaw_sigma_arcsec'],
                'final_roll_sigma_arcsec': r['final_roll_sigma_arcsec'],
                'final_pitch_sigma_arcsec': r['final_pitch_sigma_arcsec'],
                'time_to_20arcsec_s': r['time_to_20arcsec_s'],
                'score': r['score'],
                'total_time_s': r['total_time_s'],
            } for r in top
        ], 'validated_top3': validated}, f, ensure_ascii=False, indent=2)

    # markdown summary
    lines = []
    lines.append('# Alignment Strategy Sweep Summary\n')
    lines.append(f'- coarse static: {COARSE_STATIC_S}s')
    lines.append(f'- action count: {TOTAL_ACTIONS}')
    lines.append(f'- per action: rotate {ROT_S}s + pre-align {PRE_ALIGN_S}s + post-static {POST_STATIC_S}s')
    lines.append(f'- total time budget: {COARSE_STATIC_S + TOTAL_ACTIONS * (ROT_S + PRE_ALIGN_S + POST_STATIC_S):.1f}s\n')
    lines.append('## Top 5 covariance-ranked strategies')
    for i, r in enumerate(top[:5], start=1):
        lines.append(f"{i}. {' -> '.join(r['sequence'])} | final yaw σ={r['final_yaw_sigma_arcsec']:.2f}\" | 20\" crossing={r['time_to_20arcsec_s']} s | score={r['score']:.4e}")
    lines.append('\n## Monte Carlo validation (top 3)')
    for i, r in enumerate(validated, start=1):
        mc = r['monte_carlo']
        lines.append(
            f"{i}. {' -> '.join(r['sequence'])} | mean final yaw err={mc['mean_final_yaw_err_arcsec']:.2f}\" | "
            f"p95 final yaw err={mc['p95_final_yaw_err_arcsec']:.2f}\" | mean 20\" crossing={mc['mean_time_to_20arcsec_s']} s"
        )
    lines.append('\n## Recommended strategy')
    lines.append(f"- sequence: {' -> '.join(best['sequence'])}")
    lines.append(f"- final yaw sigma: {best['final_yaw_sigma_arcsec']:.2f} arcsec")
    lines.append(f"- MC mean final yaw error: {best['monte_carlo']['mean_final_yaw_err_arcsec']:.2f} arcsec")
    lines.append(f"- MC p95 final yaw error: {best['monte_carlo']['p95_final_yaw_err_arcsec']:.2f} arcsec")
    lines.append(f"- MC mean 20 arcsec crossing: {best['monte_carlo']['mean_time_to_20arcsec_s']} s")

    (out_dir / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
