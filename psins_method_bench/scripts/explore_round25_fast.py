import importlib.util, sys
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

m24 = load(METHOD24, 'm24f')
src = load(SRC, 'srcf')
mod = src

ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = m24._build_dataset(mod)
base_res = mod.run_calibration(imu_noisy, pos0, ts, n_states=42, bi_g=bi_g, tau_g=tau_g, bi_a=bi_a, tau_a=tau_a, label='42-GM1-R25-FAST')
strong_clbt = base_res[0]
windows = m24._static_window_bounds(mod, imu_noisy, ts)
imu_corr = m24._apply_strong_only(mod, imu_noisy, strong_clbt, ts)
eth = mod.Earth(pos0)
wnie = mod.glv.wie * mod.np.array([0, mod.math.cos(pos0[0]), mod.math.sin(pos0[0])])
gn = mod.np.array([0, 0, -eth.g])
Cba = mod.np.eye(3)
ka2_theta = -mod.np.asarray(strong_clbt['Ka2'], dtype=float).reshape(3)
lever_rows=[]; lever_targets=[]; weights=[]
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
    mean_fb_sq = fb_sq_sum / len(seg)
    mean_ss = ss_sum / len(seg)
    A_lev = mean_ss
    b = -vn / (len(seg) * ts) - mod.np.diag(mean_fb_sq) @ ka2_theta
    coln = mod.np.linalg.norm(A_lev, axis=0) + 1e-18
    A_scaled = A_lev / coln[None,:]
    sv = mod.np.linalg.svd(A_scaled, compute_uv=False)
    smax = float(mod.np.max(sv)) if len(sv) else 0.0
    smin = float(mod.np.min(sv)) if len(sv) else 0.0
    cond = smax / max(smin, 1e-15) if smax > 0 else 1e15
    quality = smin / max(smax,1e-15)
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
prior_s = prior * col_scale
gram = As.T @ As
diag_gram = np.diag(gram)
coupling = np.sum(np.abs(gram), axis=1) + 1e-18
diag_ratio = diag_gram / coupling
grad = np.abs(As.T @ bw)
score = diag_ratio * grad / np.sqrt(diag_gram + 1e-18)
order = np.argsort(score)[::-1]
print('score_order', order.tolist(), 'score', score.tolist(), 'diag_ratio', diag_ratio.tolist(), flush=True)

clbt_truth = src.get_default_clbt()
dKg_truth = clbt_truth['Kg'] - np.eye(3)
dKa_truth = clbt_truth['Ka'] - np.eye(3)
params = [
 ('eb_x',clbt_truth['eb'][0] / src.glv.dph,lambda c:(-c['eb'][0]) / src.glv.dph),('eb_y',clbt_truth['eb'][1] / src.glv.dph,lambda c:(-c['eb'][1]) / src.glv.dph),('eb_z',clbt_truth['eb'][2] / src.glv.dph,lambda c:(-c['eb'][2]) / src.glv.dph),
 ('db_x',clbt_truth['db'][0] / src.glv.ug,lambda c:(-c['db'][0]) / src.glv.ug),('db_y',clbt_truth['db'][1] / src.glv.ug,lambda c:(-c['db'][1]) / src.glv.ug),('db_z',clbt_truth['db'][2] / src.glv.ug,lambda c:(-c['db'][2]) / src.glv.ug),
 ('dKg_xx',dKg_truth[0,0] / src.glv.ppm,lambda c:(-(c['Kg']-np.eye(3))[0,0]) / src.glv.ppm),('dKg_yx',dKg_truth[1,0] / src.glv.sec,lambda c:(-(c['Kg']-np.eye(3))[1,0]) / src.glv.sec),('dKg_zx',dKg_truth[2,0] / src.glv.sec,lambda c:(-(c['Kg']-np.eye(3))[2,0]) / src.glv.sec),
 ('dKg_xy',dKg_truth[0,1] / src.glv.sec,lambda c:(-(c['Kg']-np.eye(3))[0,1]) / src.glv.sec),('dKg_yy',dKg_truth[1,1] / src.glv.ppm,lambda c:(-(c['Kg']-np.eye(3))[1,1]) / src.glv.ppm),('dKg_zy',dKg_truth[2,1] / src.glv.sec,lambda c:(-(c['Kg']-np.eye(3))[2,1]) / src.glv.sec),
 ('dKg_xz',dKg_truth[0,2] / src.glv.sec,lambda c:(-(c['Kg']-np.eye(3))[0,2]) / src.glv.sec),('dKg_yz',dKg_truth[1,2] / src.glv.sec,lambda c:(-(c['Kg']-np.eye(3))[1,2]) / src.glv.sec),('dKg_zz',dKg_truth[2,2] / src.glv.ppm,lambda c:(-(c['Kg']-np.eye(3))[2,2]) / src.glv.ppm),
 ('dKa_xx',dKa_truth[0,0] / src.glv.ppm,lambda c:(-(c['Ka']-np.eye(3))[0,0]) / src.glv.ppm),('dKa_xy',dKa_truth[0,1] / src.glv.sec,lambda c:(-(c['Ka']-np.eye(3))[0,1]) / src.glv.sec),('dKa_xz',dKa_truth[0,2] / src.glv.sec,lambda c:(-(c['Ka']-np.eye(3))[0,2]) / src.glv.sec),
 ('dKa_yy',dKa_truth[1,1] / src.glv.ppm,lambda c:(-(c['Ka']-np.eye(3))[1,1]) / src.glv.ppm),('dKa_yz',dKa_truth[1,2] / src.glv.sec,lambda c:(-(c['Ka']-np.eye(3))[1,2]) / src.glv.sec),('dKa_zz',dKa_truth[2,2] / src.glv.ppm,lambda c:(-(c['Ka']-np.eye(3))[2,2]) / src.glv.ppm),
 ('Ka2_x',clbt_truth['Ka2'][0] / src.glv.ugpg2,lambda c:(-c['Ka2'][0]) / src.glv.ugpg2),('Ka2_y',clbt_truth['Ka2'][1] / src.glv.ugpg2,lambda c:(-c['Ka2'][1]) / src.glv.ugpg2),('Ka2_z',clbt_truth['Ka2'][2] / src.glv.ugpg2,lambda c:(-c['Ka2'][2]) / src.glv.ugpg2),
 ('rx_x',clbt_truth['rx'][0],lambda c:-c['rx'][0]),('rx_y',clbt_truth['rx'][1],lambda c:-c['rx'][1]),('rx_z',clbt_truth['rx'][2],lambda c:-c['rx'][2]),
 ('ry_x',clbt_truth['ry'][0],lambda c:-c['ry'][0]),('ry_y',clbt_truth['ry'][1],lambda c:-c['ry'][1]),('ry_z',clbt_truth['ry'][2],lambda c:-c['ry'][2]),
]

def metrics(clbt):
    p=[]; named={}
    for n,t,fn in params:
        est=float(fn(clbt)); pct=abs(t-est)/abs(t)*100.0
        p.append(pct); named[n]=est
    return float(np.mean(p)), float(np.median(p)), float(np.max(p)), named

variants = [
    {'name':'top2_a','sigma':[0.0038,0.0026,0.0026,0.0038,0.0026,0.0026],'ridge':1.8,'mix':(0.78,0.22),'clip':0.0022,'trust':'top2','t1':0.54,'t2':0.10},
    {'name':'top2_b','sigma':[0.0038,0.0026,0.0026,0.0038,0.0026,0.0026],'ridge':1.8,'mix':(0.76,0.24),'clip':0.0022,'trust':'top2','t1':0.48,'t2':0.16},
    {'name':'top2_c','sigma':[0.0040,0.0030,0.0030,0.0040,0.0030,0.0030],'ridge':1.8,'mix':(0.82,0.18),'clip':0.0024,'trust':'top2','t1':0.47,'t2':0.14},
    {'name':'diagmix_a','sigma':[0.0038,0.0026,0.0026,0.0038,0.0026,0.0026],'ridge':1.6,'mix':(0.76,0.24),'clip':0.0022,'trust':'diagmix','base':0.01,'gain':0.34,'cap':0.22},
    {'name':'diagmix_b','sigma':[0.0038,0.0026,0.0026,0.0038,0.0026,0.0026],'ridge':1.8,'mix':(0.78,0.22),'clip':0.0020,'trust':'diagmix','base':0.0,'gain':0.28,'cap':0.20},
]
for cfg in variants:
    delta_sigma = np.array(cfg['sigma'])
    ridge = np.diag((1.0 / delta_sigma) ** 2)
    U,S,Vt = np.linalg.svd(As, full_matrices=False)
    smax = float(S[0]) if len(S) else 0.0
    trunc_floor = max(0.04*smax, 1e-9)
    Sinv = np.array([1.0/s if s >= trunc_floor else 0.0 for s in S])
    theta_pinv = Vt.T @ (Sinv * (U.T @ bw)) if len(S) else prior_s.copy()
    lhs = As.T @ As + cfg['ridge'] * ridge
    rhs = As.T @ bw + cfg['ridge'] * ridge @ prior_s
    theta_map = np.linalg.solve(lhs, rhs)
    a,bm = cfg['mix']
    theta = (a * theta_map + bm * theta_pinv) / col_scale
    proposal = -np.asarray(theta).reshape(-1)
    prior_lever = np.concatenate((strong_clbt['rx'], strong_clbt['ry']))
    raw_delta = np.clip(proposal - prior_lever, -cfg['clip'], cfg['clip'])
    if cfg['trust']=='top2':
        trust = np.zeros(6)
        trust[order[0]] = cfg['t1']
        trust[order[1]] = cfg['t2']
    else:
        norm = score / max(score[order[0]], 1e-18)
        trust = np.clip(cfg['base'] + cfg['gain'] * diag_ratio * np.sqrt(norm), 0.0, cfg['cap'])
    lever = np.clip(prior_lever + trust * raw_delta, -0.08, 0.08)
    refined = m24._copy_clbt(mod, strong_clbt)
    refined['rx'] = lever[:3]
    refined['ry'] = lever[3:]
    mean, med, worst, named = metrics(refined)
    print({
        'name': cfg['name'], 'mean': mean, 'med': med, 'worst': worst,
        'trust': trust.tolist(),
        'rx_y': named['rx_y'], 'ry_z': named['ry_z'], 'rx_x': named['rx_x'], 'ry_x': named['ry_x']
    }, flush=True)
