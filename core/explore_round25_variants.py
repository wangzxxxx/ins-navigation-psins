import importlib.util
import sys
from pathlib import Path
import numpy as np

ROOT = Path('/root/.openclaw/workspace')
METHOD24 = ROOT / 'psins_method_bench' / 'methods' / 'markov' / 'method_42state_gm1_round24_softcond_dualtrust.py'
SRC = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'


def load(path, name):
    path = Path(path)
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

m24 = load(METHOD24, 'm24')
src = load(SRC, 'src25')
mod = src


def compute_errors(clbt):
    clbt_truth = src.get_default_clbt()
    dKg_truth = clbt_truth['Kg'] - np.eye(3)
    dKa_truth = clbt_truth['Ka'] - np.eye(3)
    params = [
        ('eb_x',   clbt_truth['eb'][0] / src.glv.dph,  lambda c: (-c['eb'][0]) / src.glv.dph), ('eb_y', clbt_truth['eb'][1] / src.glv.dph, lambda c: (-c['eb'][1]) / src.glv.dph), ('eb_z', clbt_truth['eb'][2] / src.glv.dph, lambda c: (-c['eb'][2]) / src.glv.dph),
        ('db_x',   clbt_truth['db'][0] / src.glv.ug,   lambda c: (-c['db'][0]) / src.glv.ug), ('db_y', clbt_truth['db'][1] / src.glv.ug, lambda c: (-c['db'][1]) / src.glv.ug), ('db_z', clbt_truth['db'][2] / src.glv.ug, lambda c: (-c['db'][2]) / src.glv.ug),
        ('dKg_xx', dKg_truth[0,0] / src.glv.ppm,       lambda c: (-(c['Kg']-np.eye(3))[0,0]) / src.glv.ppm), ('dKg_yx', dKg_truth[1,0] / src.glv.sec, lambda c: (-(c['Kg']-np.eye(3))[1,0]) / src.glv.sec), ('dKg_zx', dKg_truth[2,0] / src.glv.sec, lambda c: (-(c['Kg']-np.eye(3))[2,0]) / src.glv.sec),
        ('dKg_xy', dKg_truth[0,1] / src.glv.sec,       lambda c: (-(c['Kg']-np.eye(3))[0,1]) / src.glv.sec), ('dKg_yy', dKg_truth[1,1] / src.glv.ppm, lambda c: (-(c['Kg']-np.eye(3))[1,1]) / src.glv.ppm), ('dKg_zy', dKg_truth[2,1] / src.glv.sec, lambda c: (-(c['Kg']-np.eye(3))[2,1]) / src.glv.sec),
        ('dKg_xz', dKg_truth[0,2] / src.glv.sec,       lambda c: (-(c['Kg']-np.eye(3))[0,2]) / src.glv.sec), ('dKg_yz', dKg_truth[1,2] / src.glv.sec, lambda c: (-(c['Kg']-np.eye(3))[1,2]) / src.glv.sec), ('dKg_zz', dKg_truth[2,2] / src.glv.ppm, lambda c: (-(c['Kg']-np.eye(3))[2,2]) / src.glv.ppm),
        ('dKa_xx', dKa_truth[0,0] / src.glv.ppm,       lambda c: (-(c['Ka']-np.eye(3))[0,0]) / src.glv.ppm), ('dKa_xy', dKa_truth[0,1] / src.glv.sec, lambda c: (-(c['Ka']-np.eye(3))[0,1]) / src.glv.sec), ('dKa_xz', dKa_truth[0,2] / src.glv.sec, lambda c: (-(c['Ka']-np.eye(3))[0,2]) / src.glv.sec),
        ('dKa_yy', dKa_truth[1,1] / src.glv.ppm,       lambda c: (-(c['Ka']-np.eye(3))[1,1]) / src.glv.ppm), ('dKa_yz', dKa_truth[1,2] / src.glv.sec, lambda c: (-(c['Ka']-np.eye(3))[1,2]) / src.glv.sec), ('dKa_zz', dKa_truth[2,2] / src.glv.ppm, lambda c: (-(c['Ka']-np.eye(3))[2,2]) / src.glv.ppm),
        ('Ka2_x',  clbt_truth['Ka2'][0] / src.glv.ugpg2, lambda c: (-c['Ka2'][0]) / src.glv.ugpg2), ('Ka2_y', clbt_truth['Ka2'][1] / src.glv.ugpg2, lambda c: (-c['Ka2'][1]) / src.glv.ugpg2), ('Ka2_z', clbt_truth['Ka2'][2] / src.glv.ugpg2, lambda c: (-c['Ka2'][2]) / src.glv.ugpg2),
        ('rx_x', clbt_truth['rx'][0], lambda c: -c['rx'][0]), ('rx_y', clbt_truth['rx'][1], lambda c: -c['rx'][1]), ('rx_z', clbt_truth['rx'][2], lambda c: -c['rx'][2]),
        ('ry_x', clbt_truth['ry'][0], lambda c: -c['ry'][0]), ('ry_y', clbt_truth['ry'][1], lambda c: -c['ry'][1]), ('ry_z', clbt_truth['ry'][2], lambda c: -c['ry'][2]),
    ]
    rows = []
    pcts = []
    for name, truth, fn in params:
        est = float(fn(clbt))
        abs_err = abs(truth - est)
        pct_err = abs_err / abs(truth) * 100.0 if abs(truth) > 1e-15 else None
        rows.append({'param': name, 'truth': float(truth), 'estimate': est, 'pct_error': pct_err})
        if pct_err is not None:
            pcts.append(pct_err)
    return rows, float(np.mean(pcts)), float(np.median(pcts)), float(np.max(pcts))


ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = m24._build_dataset(mod)
base_res = mod.run_calibration(imu_noisy, pos0, ts, n_states=42, bi_g=bi_g, tau_g=tau_g, bi_a=bi_a, tau_a=tau_a, label='42-GM1-R25-EXP')
strong_clbt = base_res[0]
eth = mod.Earth(pos0)
wnie = mod.glv.wie * mod.np.array([0, mod.math.cos(pos0[0]), mod.math.sin(pos0[0])])
gn = mod.np.array([0, 0, -eth.g])
Cba = mod.np.eye(3)
windows = m24._static_window_bounds(mod, imu_noisy, ts)
imu_corr = m24._apply_strong_only(mod, imu_noisy, strong_clbt, ts)
lever_rows=[]; lever_targets=[]; weights=[]; cond_before=[]; quality_vals=[]
ka2_theta = -mod.np.asarray(strong_clbt['Ka2'], dtype=float).reshape(3)
for s,e in windows:
    seg = imu_corr[s:e,:]
    if len(seg) < 80:
        continue
    qnb = mod.alignsb(seg, pos0)[3]
    vn = mod.np.zeros(3)
    fb_sq_sum = mod.np.zeros(3)
    ss_sum = mod.np.zeros((3,6))
    dotwf = mod.imudot(seg, 5.0)
    for k in range(len(seg)):
        wm = seg[k,0:3]; vm = seg[k,3:6]; dwb = dotwf[k,0:3]
        fb = vm / ts
        SS = mod.imulvS(wm / ts, dwb, Cba)
        fb_sq_sum += fb**2
        ss_sum += SS[:,0:6]
        qnb = mod.qupdt2(qnb, wm, wnie * ts)
        fn0 = mod.qmulv(qnb, fb)
        vn = vn + (mod.rotv(-wnie * ts / 2, fn0) + gn) * ts
    mean_fb_sq = fb_sq_sum / max(len(seg),1)
    mean_ss = ss_sum / max(len(seg),1)
    A_lev = mean_ss
    b = -vn / max(len(seg) * ts, 1e-12) - mod.np.diag(mean_fb_sq) @ ka2_theta
    coln = mod.np.linalg.norm(A_lev, axis=0) + 1e-18
    A_scaled = A_lev / coln[None,:]
    sv = mod.np.linalg.svd(A_scaled, compute_uv=False)
    smax = float(mod.np.max(sv)) if len(sv) else 0.0
    smin = float(mod.np.min(sv)) if len(sv) else 0.0
    cond = smax / max(smin, 1e-15) if smax > 0 else 1e15
    cond_before.append(cond)
    quality = smin / max(smax,1e-15)
    quality_vals.append(quality)
    motion = mod.np.linalg.norm(mean_ss, ord='fro')
    cond_soft = 1.0 / (1.0 + (cond / 2.5e11) ** 0.65)
    quality_soft = 0.10 + 2.20 * quality
    motion_soft = 0.35 + 0.06 * motion
    weight = max(len(seg) * cond_soft * quality_soft * motion_soft, 1e-6)
    lever_rows.append(A_lev); lever_targets.append(b); weights.append(weight)
A = mod.np.vstack(lever_rows); b = mod.np.hstack(lever_targets)
row_w = mod.np.repeat(mod.np.sqrt(mod.np.asarray(weights, dtype=float)), 3)
Aw = A * row_w[:,None]; bw = b * row_w
col_scale = mod.np.sqrt(mod.np.sum(Aw*Aw, axis=0) + 1e-18)
col_scale = mod.np.maximum(col_scale, mod.np.array([0.08]*6))
As = Aw / col_scale[None,:]
prior = -mod.np.concatenate((strong_clbt['rx'], strong_clbt['ry']))
prior_lever = mod.np.concatenate((strong_clbt['rx'], strong_clbt['ry']))
prior_s = prior * col_scale
gram = As.T @ As
diag_gram = mod.np.diag(gram)
coupling = mod.np.sum(mod.np.abs(gram), axis=1) + 1e-18
diag_ratio = diag_gram / coupling
grad = mod.np.abs(As.T @ bw)
score = diag_ratio * grad / mod.np.sqrt(diag_gram + 1e-18)
order = mod.np.argsort(score)[::-1]
print('base_order', order.tolist(), 'score', score.tolist(), 'diag_ratio', diag_ratio.tolist(), flush=True)


def variant(name, delta_sigma, ridge_scale, map_mix, trunc_ratio, clip, trust_mode):
    ridge = mod.np.diag((1.0 / np.array(delta_sigma)) ** 2)
    U,S,Vt = mod.np.linalg.svd(As, full_matrices=False)
    smax = float(S[0]) if len(S) else 0.0
    trunc_floor = max(trunc_ratio*smax, 1e-9)
    Sinv = mod.np.array([1.0/s if s >= trunc_floor else 0.0 for s in S])
    theta_pinv = Vt.T @ (Sinv * (U.T @ bw)) if len(S) else prior_s.copy()
    lhs = As.T @ As + ridge_scale * ridge
    rhs = As.T @ bw + ridge_scale * ridge @ prior_s
    theta_map = mod.np.linalg.solve(lhs, rhs)
    theta_s = map_mix * theta_map + (1.0-map_mix) * theta_pinv
    theta = theta_s / col_scale
    proposal = -mod.np.asarray(theta).reshape(-1)
    raw_delta = np.clip(proposal - prior_lever, -clip, clip)
    if trust_mode == 'dual_ranked':
        trust = np.zeros(6)
        trust[order[0]] = 0.44
        trust[order[1]] = 0.16
    elif trust_mode == 'triple_ranked':
        trust = np.zeros(6)
        trust[order[0]] = 0.34
        trust[order[1]] = 0.16
        trust[order[2]] = 0.08
    elif trust_mode == 'diag_mix':
        norm = score / max(score[order[0]], 1e-18)
        trust = np.clip(0.015 + 0.26 * diag_ratio * np.sqrt(norm), 0.0, 0.30)
    elif trust_mode == 'pair_gate':
        trust = np.zeros(6)
        trust[order[0]] = 0.40
        trust[order[1]] = 0.14 if score[order[1]]/max(score[order[0]],1e-18) > 0.15 else 0.0
        trust[order[2]] = 0.06 if diag_ratio[order[2]] > 0.12 else 0.0
    else:
        raise ValueError(trust_mode)
    lever = np.clip(prior_lever + trust * raw_delta, -0.08, 0.08)
    refined = m24._copy_clbt(mod, strong_clbt)
    refined['rx'] = lever[:3]
    refined['ry'] = lever[3:]
    rows, mean, med, worst = compute_errors(refined)
    lever_rows_out = {r['param']: r['estimate'] for r in rows if r['param'].startswith('r')}
    print({'name': name, 'mean': mean, 'median': med, 'worst': worst, 'trust': trust.tolist(), 'lever': lever_rows_out}, flush=True)

variant('v1_dual_ranked_tighter', [0.0036,0.0024,0.0024,0.0036,0.0024,0.0024], 1.9, 0.80, 0.040, 0.0022, 'dual_ranked')
variant('v2_triple_ranked', [0.0038,0.0026,0.0026,0.0038,0.0026,0.0026], 1.8, 0.78, 0.040, 0.0020, 'triple_ranked')
variant('v3_diag_mix', [0.0038,0.0026,0.0026,0.0038,0.0026,0.0026], 1.8, 0.78, 0.040, 0.0022, 'diag_mix')
variant('v4_pair_gate', [0.0037,0.0025,0.0025,0.0037,0.0025,0.0025], 1.7, 0.80, 0.038, 0.0022, 'pair_gate')
