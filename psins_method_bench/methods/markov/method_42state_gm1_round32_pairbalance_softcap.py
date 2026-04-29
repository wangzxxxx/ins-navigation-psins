from __future__ import annotations

import sys
import types

if 'matplotlib' not in sys.modules:
    matplotlib_stub = types.ModuleType('matplotlib')
    pyplot_stub = types.ModuleType('matplotlib.pyplot')
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules['matplotlib'] = matplotlib_stub
    sys.modules['matplotlib.pyplot'] = pyplot_stub
if 'seaborn' not in sys.modules:
    sys.modules['seaborn'] = types.ModuleType('seaborn')

from common_markov import TMP_PSINS, emit_result, load_module, summarize_result

SOURCE = 'test_calibration_markov_pruned.py'
METHOD = '42-state GM1 round32 pair-balanced lever trust with cross-pair soft cap'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round32_pairbalance_softcap'


def _build_dataset(mod):
    ts = 0.01
    att0 = mod.np.array([1.0, -91.0, -91.0]) * mod.glv.deg
    pos0 = mod.posset(34.0, 0.0, 0.0)
    paras = mod.np.array([
        [1, 0, 1, 0, 90, 9, 70, 70], [2, 0, 1, 0, 90, 9, 20, 20], [3, 0, 1, 0, 90, 9, 20, 20],
        [4, 0, 1, 0, -90, 9, 20, 20], [5, 0, 1, 0, -90, 9, 20, 20], [6, 0, 1, 0, -90, 9, 20, 20],
        [7, 0, 0, 1, 90, 9, 20, 20], [8, 1, 0, 0, 90, 9, 20, 20], [9, 1, 0, 0, 90, 9, 20, 20],
        [10, 1, 0, 0, 90, 9, 20, 20], [11, -1, 0, 0, 90, 9, 20, 20], [12, -1, 0, 0, 90, 9, 20, 20],
        [13, -1, 0, 0, 90, 9, 20, 20], [14, 0, 0, 1, 90, 9, 20, 20], [15, 0, 0, 1, 90, 9, 20, 20],
        [16, 0, 0, -1, 90, 9, 20, 20], [17, 0, 0, -1, 90, 9, 20, 20], [18, 0, 0, -1, 90, 9, 20, 20],
    ], dtype=float)
    paras[:, 4] = paras[:, 4] * mod.glv.deg
    att = mod.attrottt(att0, paras, ts)
    imu, _ = mod.avp2imu(att, pos0)
    clbt_truth = mod.get_default_clbt()
    imu_clean = mod.imuclbt(imu, clbt_truth)
    bi_g = 0.002 * mod.glv.dph
    bi_a = 5.0 * mod.glv.ug
    tau_g = 300.0
    tau_a = 300.0
    imu_noisy = mod.imuadderr_full(
        imu_clean, ts,
        arw=0.005 * mod.glv.dpsh, vrw=5.0 * mod.glv.ugpsHz,
        bi_g=bi_g, tau_g=tau_g,
        bi_a=bi_a, tau_a=tau_a, seed=42,
    )
    return ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a


def _copy_clbt(mod, clbt):
    return {
        'Kg': mod.np.array(clbt['Kg'], dtype=float).copy(),
        'Ka': mod.np.array(clbt['Ka'], dtype=float).copy(),
        'Ka2': mod.np.array(clbt['Ka2'], dtype=float).copy(),
        'eb': mod.np.array(clbt['eb'], dtype=float).copy(),
        'db': mod.np.array(clbt['db'], dtype=float).copy(),
        'rx': mod.np.array(clbt['rx'], dtype=float).copy(),
        'ry': mod.np.array(clbt['ry'], dtype=float).copy(),
    }


def _static_window_bounds(mod, imu1, ts):
    frq2 = int(1 / ts / 2) - 1
    windows = []
    in_static = False
    start = None
    for k in range(frq2, len(imu1) - frq2, 2 * frq2):
        ww = mod.np.mean(imu1[k-frq2:k+frq2+1, 0:3], axis=0)
        is_static = mod.np.linalg.norm(ww) / ts < 20 * mod.glv.dph
        if is_static and not in_static:
            in_static = True
            start = max(0, k - frq2)
        elif (not is_static) and in_static:
            in_static = False
            windows.append((start, min(len(imu1), k + frq2)))
            start = None
    if in_static and start is not None:
        windows.append((start, len(imu1)))
    return windows


def _apply_strong_only(mod, imu_s, clbt, ts):
    res = mod.np.copy(imu_s)
    for i in range(len(res)):
        res[i, 0:3] = clbt['Kg'] @ res[i, 0:3] - clbt['eb'] * ts
        res[i, 3:6] = clbt['Ka'] @ res[i, 3:6] - clbt['db'] * ts
    return res


def _round32_refine(mod, imu1, pos0, ts, strong_clbt):
    eth = mod.Earth(pos0)
    wnie = mod.glv.wie * mod.np.array([0, mod.math.cos(pos0[0]), mod.math.sin(pos0[0])])
    gn = mod.np.array([0, 0, -eth.g])
    Cba = mod.np.eye(3)
    refined = _copy_clbt(mod, strong_clbt)
    windows = _static_window_bounds(mod, imu1, ts)
    if not windows:
        return refined, {'n_windows': 0, 'used_windows': 0}

    imu_corr = _apply_strong_only(mod, imu1, strong_clbt, ts)
    blocks = []
    innov_vals = []
    cond_before = []
    quality_vals = []

    for s, e in windows:
        seg = imu_corr[s:e, :]
        if len(seg) < 80:
            continue
        qnb = mod.alignsb(seg, pos0)[3]
        vn = mod.np.zeros(3)
        fb_sq_sum = mod.np.zeros(3)
        ss_sum = mod.np.zeros((3, 6))
        dotwf = mod.imudot(seg, 5.0)
        for k in range(len(seg)):
            wm = seg[k, 0:3]
            vm = seg[k, 3:6]
            dwb = dotwf[k, 0:3]
            fb = vm / ts
            SS = mod.imulvS(wm / ts, dwb, Cba)
            fb_sq_sum += fb**2
            ss_sum += SS[:, 0:6]
            qnb = mod.qupdt2(qnb, wm, wnie * ts)
            fn0 = mod.qmulv(qnb, fb)
            vn = vn + (mod.rotv(-wnie * ts / 2, fn0) + gn) * ts
        mean_fb_sq = fb_sq_sum / max(len(seg), 1)
        mean_ss = ss_sum / max(len(seg), 1)
        ywin = -vn / max(len(seg) * ts, 1e-12)
        Awin = mod.np.hstack((mod.np.diag(mean_fb_sq), mean_ss))

        coln = mod.np.linalg.norm(Awin, axis=0) + 1e-18
        A_scaled = Awin / coln[None, :]
        sv = mod.np.linalg.svd(A_scaled, compute_uv=False)
        smax = float(mod.np.max(sv)) if len(sv) else 0.0
        smin = float(mod.np.min(sv)) if len(sv) else 0.0
        cond = smax / max(smin, 1e-15) if smax > 0 else 1e15
        quality = smin / max(smax, 1e-15)
        innov = float(mod.np.linalg.norm(ywin))
        motion = float(mod.np.linalg.norm(mean_ss, ord='fro'))

        blocks.append((mod.np.diag(mean_fb_sq), mean_ss, ywin, len(seg), cond, quality, innov, motion))
        innov_vals.append(innov)
        cond_before.append(cond)
        quality_vals.append(quality)

    if not blocks:
        return refined, {'n_windows': len(windows), 'used_windows': 0}

    innov_arr = mod.np.asarray(innov_vals, dtype=float)
    innov_med = float(mod.np.median(innov_arr))
    innov_mad = float(mod.np.median(mod.np.abs(innov_arr - innov_med)) + 1e-12)
    innov_cut = innov_med + 1.8 * innov_mad

    lever_rows, lever_ys, weights = [], [], []
    ka2_diag_rows = []
    trimmed = 0
    for Dwin, SSwin, ywin, seg_len, cond, quality, innov, motion in blocks:
        cond_soft = 1.0 / (1.0 + (cond / 2.5e11) ** 0.65)
        quality_soft = 0.10 + 2.20 * quality
        motion_soft = 0.35 + 0.06 * motion
        innov_soft = 1.0 / (1.0 + (innov / max(innov_med + 2.5 * innov_mad, 1e-9)) ** 2)
        if innov > innov_cut:
            innov_soft *= 0.35
            trimmed += 1
        weight = max(seg_len * cond_soft * quality_soft * motion_soft * innov_soft, 1e-6)
        lever_rows.append(SSwin)
        lever_ys.append(ywin)
        weights.append(weight)
        ka2_diag_rows.append(mod.np.diag(Dwin).copy())

    A = mod.np.vstack(lever_rows)
    b = mod.np.hstack(lever_ys)
    row_w = mod.np.repeat(mod.np.sqrt(mod.np.asarray(weights, dtype=float)), 3)
    Aw = A * row_w[:, None]
    bw = b * row_w

    col_scale = mod.np.sqrt(mod.np.sum(Aw * Aw, axis=0) + 1e-18)
    col_scale = mod.np.maximum(col_scale, mod.np.array([0.08, 0.08, 0.08, 0.08, 0.08, 0.08]))
    As = Aw / col_scale[None, :]

    prior = -mod.np.concatenate((strong_clbt['rx'], strong_clbt['ry']))
    prior_s = prior * col_scale
    reg_sigma = mod.np.array([0.00385, 0.0027, 0.0027, 0.00385, 0.0027, 0.0027])
    ridge = mod.np.diag((1.0 / reg_sigma) ** 2)

    U, S, Vt = mod.np.linalg.svd(As, full_matrices=False)
    smax = float(S[0]) if len(S) else 0.0
    trunc_floor = max(0.036 * smax, 1e-9)
    Sinv = mod.np.array([1.0 / s if s >= trunc_floor else 0.0 for s in S])
    theta_pinv = Vt.T @ (Sinv * (U.T @ bw)) if len(S) else prior_s.copy()

    lhs = As.T @ As + 1.78 * ridge
    rhs = As.T @ bw + 1.78 * ridge @ prior_s
    try:
        theta_map = mod.np.linalg.solve(lhs, rhs)
    except mod.np.linalg.LinAlgError:
        theta_map, *_ = mod.np.linalg.lstsq(lhs, rhs, rcond=None)

    theta_s = 0.83 * theta_map + 0.17 * theta_pinv
    theta = mod.np.asarray(theta_s / col_scale).reshape(-1)
    proposal_lever = mod.np.clip(-theta, -0.08, 0.08)

    lever_prior = mod.np.concatenate((strong_clbt['rx'], strong_clbt['ry']))
    raw_lever_delta = mod.np.clip(proposal_lever - lever_prior, -0.0024, 0.0024)

    gram = As.T @ As
    diag_gram = mod.np.diag(gram)
    coupling = mod.np.sum(mod.np.abs(gram), axis=1) + 1e-18
    diag_ratio = diag_gram / coupling
    grad = mod.np.abs(As.T @ bw)
    score = diag_ratio * grad / mod.np.sqrt(diag_gram + 1e-18)
    lever_order = mod.np.argsort(score)[::-1]
    lead = int(lever_order[0]) if len(lever_order) else None
    runner = int(lever_order[1]) if len(lever_order) > 1 else None
    lead_ratio = float(score[lead] / max(score[runner], 1e-18)) if lead is not None and runner is not None else None

    lever_trust = mod.np.zeros(6)
    if lead is not None:
        lever_trust[lead] = 0.438
    if runner is not None and score[runner] / max(score[lead], 1e-18) > 0.12:
        lever_trust[runner] = min(0.118, 0.16 * score[runner] / max(score[lead], 1e-18))
    for idx in lever_order[2:]:
        if diag_ratio[idx] < 0.16:
            continue
        if score[idx] / max(score[lead], 1e-18) < 0.045:
            continue
        lever_trust[idx] = min(0.028, 0.05 * score[idx] / max(score[lead], 1e-18))
        break

    pair_defs = [(1, 5), (2, 3)]
    pair_balance = []
    cross_pair_soft_cap = []
    balanced_trust = lever_trust.copy()
    balanced_delta = raw_lever_delta.copy()
    for i, j in pair_defs:
        pair_score = float(score[i] + score[j])
        if pair_score <= 0:
            pair_balance.append({'pair': [int(i), int(j)], 'pair_score': 0.0, 'active': False})
            continue
        closeness = float(min(score[i], score[j]) / max(max(score[i], score[j]), 1e-18))
        balance = float(mod.np.clip(0.5 + 0.22 * (score[i] - score[j]) / pair_score, 0.34, 0.66))
        support = float(mod.np.clip(0.58 + 0.42 * closeness, 0.58, 1.0))
        pair_budget = float(min(0.438, (lever_trust[i] + lever_trust[j]) + 0.24 * support))
        active = closeness > 0.28 and min(diag_ratio[i], diag_ratio[j]) > 0.07
        if active:
            balanced_trust[i] = pair_budget * balance
            balanced_trust[j] = pair_budget * (1.0 - balance)
            cap = float((0.00100 + 0.00055 * closeness) * (0.90 + 0.25 * support))
            pair_raw = balanced_trust[[i, j]] * raw_lever_delta[[i, j]]
            pair_clip = mod.np.clip(pair_raw, -cap, cap)
            dom = 0 if abs(pair_clip[0]) >= abs(pair_clip[1]) else 1
            oth = 1 - dom
            dom_mag = abs(pair_clip[dom])
            oth_mag = abs(pair_clip[oth])
            if dom_mag > 1.35 * oth_mag + 1e-12:
                pair_clip[dom] = mod.np.sign(pair_clip[dom]) * min(dom_mag, 1.35 * oth_mag + 0.00018)
            balanced_delta[i], balanced_delta[j] = pair_clip[0], pair_clip[1]
            cross_pair_soft_cap.append({
                'pair': [int(i), int(j)],
                'cap': cap,
                'closeness': closeness,
                'support': support,
                'active': True,
                'pre_delta': [float((lever_trust[i] * raw_lever_delta[i])), float((lever_trust[j] * raw_lever_delta[j]))],
                'post_delta': [float(pair_clip[0]), float(pair_clip[1])],
            })
        else:
            balanced_delta[i] = balanced_trust[i] * raw_lever_delta[i]
            balanced_delta[j] = balanced_trust[j] * raw_lever_delta[j]
            cross_pair_soft_cap.append({'pair': [int(i), int(j)], 'active': False, 'closeness': closeness, 'support': support})
        pair_balance.append({
            'pair': [int(i), int(j)],
            'active': bool(active),
            'score': [float(score[i]), float(score[j])],
            'diag_ratio': [float(diag_ratio[i]), float(diag_ratio[j])],
            'closeness': closeness,
            'budget': pair_budget,
            'split': [float(balanced_trust[i]), float(balanced_trust[j])],
        })

    untouched = [idx for idx in range(6) if all(idx not in p for p in pair_defs)]
    for idx in untouched:
        balanced_delta[idx] = balanced_trust[idx] * raw_lever_delta[idx]

    lever = mod.np.clip(lever_prior + balanced_delta, -0.08, 0.08)
    refined['rx'] = lever[0:3]
    refined['ry'] = lever[3:6]

    lever_effect_rows = []
    for SSwin in lever_rows:
        lever_effect_rows.append(SSwin @ (-lever))
    lever_effect = mod.np.vstack(lever_effect_rows)
    lever_residual = mod.np.vstack(lever_ys) - lever_effect

    ka2_prior = mod.np.asarray(strong_clbt['Ka2'], dtype=float)
    ka2_prior_theta = -ka2_prior
    ka2_axis_trust = mod.np.array([0.030, 0.016, 0.012])
    ka2_axis_clip = mod.np.array([0.85, 0.45, 0.32]) * mod.glv.ugpg2
    ka2_score = mod.np.zeros(3)
    ka2_obs = mod.np.zeros(3)
    ka2_raw_delta = mod.np.zeros(3)

    weight_arr = mod.np.asarray(weights, dtype=float)
    for axis in range(3):
        a_axis = mod.np.asarray([row[axis] for row in ka2_diag_rows], dtype=float)
        y_axis = lever_residual[:, axis]
        wa = weight_arr * (0.70 + 0.60 * mod.np.clip(a_axis / (mod.np.median(a_axis) + 1e-18), 0.0, 2.0))
        sum_a2 = float(mod.np.sum(wa * (a_axis ** 2)))
        sum_ay = float(mod.np.sum(wa * a_axis * y_axis))
        sigma = [40.0, 80.0, 120.0][axis] * mod.glv.ugpg2
        theta_hat = (sum_ay + (1.0 / sigma**2) * ka2_prior_theta[axis]) / max(sum_a2 + (1.0 / sigma**2), 1e-18)
        proposal = float(mod.np.clip(-theta_hat, -35 * mod.glv.ugpg2, 35 * mod.glv.ugpg2))
        raw_delta = float(mod.np.clip(proposal - ka2_prior[axis], -ka2_axis_clip[axis], ka2_axis_clip[axis]))
        ka2_raw_delta[axis] = raw_delta
        ka2_obs[axis] = sum_a2
        ka2_score[axis] = abs(sum_ay) / max(sum_a2**0.5, 1e-18)

    refined['Ka2'] = ka2_prior + ka2_axis_trust * ka2_raw_delta

    return refined, {
        'n_windows': len(windows),
        'used_windows': len(blocks),
        'trimmed_windows': int(trimmed),
        'cond_median': float(mod.np.median(cond_before)) if cond_before else None,
        'cond_worst': float(mod.np.max(cond_before)) if cond_before else None,
        'quality_median': float(mod.np.median(quality_vals)) if quality_vals else None,
        'innov_median': innov_med,
        'innov_mad': innov_mad,
        'innov_cut': innov_cut,
        'trunc_floor': trunc_floor,
        'weight_sum': float(mod.np.sum(weights)),
        'lever_score_order': [int(i) for i in lever_order],
        'lever_score': [float(x) for x in score],
        'lever_diag_ratio': [float(x) for x in diag_ratio],
        'lead_ratio': lead_ratio,
        'lever_trust_pre_balance': [float(x) for x in lever_trust],
        'lever_trust': [float(x) for x in balanced_trust],
        'lever_raw_delta': [float(x) for x in raw_lever_delta],
        'lever_applied_delta': [float(x) for x in balanced_delta],
        'pair_balance': pair_balance,
        'cross_pair_soft_cap': cross_pair_soft_cap,
        'ka2_bypass_trust': [float(x) for x in ka2_axis_trust],
        'ka2_bypass_clip_ugpg2': [float(x / mod.glv.ugpg2) for x in ka2_axis_clip],
        'ka2_bypass_obs': [float(x) for x in ka2_obs],
        'ka2_bypass_score': [float(x) for x in ka2_score],
        'ka2_bypass_raw_delta_ugpg2': [float(x / mod.glv.ugpg2) for x in ka2_raw_delta],
    }


def run_method():
    mod = load_module('markov_pruned_42_round32_pairbalance_softcap', str(TMP_PSINS / SOURCE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = _build_dataset(mod)
    base_res = mod.run_calibration(
        imu_noisy, pos0, ts, n_states=42,
        bi_g=bi_g, tau_g=tau_g, bi_a=bi_a, tau_a=tau_a,
        label='42-GM1-R32-BASE'
    )
    refined_clbt, extra = _round32_refine(mod, imu_noisy, pos0, ts, base_res[0])
    return refined_clbt, base_res[1], base_res[2], base_res[3], {
        'iter_bounds': base_res[4],
        **extra,
    }


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
