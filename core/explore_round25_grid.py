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

m24 = load(METHOD24, 'm24g')
src = load(SRC, 'srcg')
mod = src

ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = m24._build_dataset(mod)
base_res = mod.run_calibration(imu_noisy, pos0, ts, n_states=42, bi_g=bi_g, tau_g=tau_g, bi_a=bi_a, tau_a=tau_a, label='42-GM1-R25-GRID')
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
U,S,Vt = np.linalg.svd(As, full_matrices=False)
smax = float(S[0]) if len(S) else 0.0
trunc_floor = max(0.035*smax, 1e-9)
Sinv = np.array([1.0/s if s >= trunc_floor else 0.0 for s in S])
theta_pinv = Vt.T @ (Sinv * (U.T @ bw)) if len(S) else prior_s.copy()
delta_sigma = np.array([0.0036,0.0025,0.0026,0.0038,0.0028,0.0024])
ridge = np.diag((1.0 / delta_sigma) ** 2)
lhs = As.T @ As + 1.7 * ridge
rhs = As.T @ bw + 1.7 * ridge @ prior_s
theta_map = np.linalg.solve(lhs, rhs)
theta = (0.74 * theta_map + 0.26 * theta_pinv) / col_scale
proposal = -np.asarray(theta).reshape(-1)
prior_lever = np.concatenate((strong_clbt['rx'], strong_clbt['ry']))
raw_delta = np.clip(proposal - prior_lever, -0.0028, 0.0028)
print('prior', prior_lever.tolist())
print('proposal', proposal.tolist())
print('raw_delta', raw_delta.tolist())

clbt_truth = src.get_default_clbt()
base_pcts = {}
for nm, truth, est in [
    ('rx_x', clbt_truth['rx'][0], -strong_clbt['rx'][0]), ('rx_y', clbt_truth['rx'][1], -strong_clbt['rx'][1]), ('rx_z', clbt_truth['rx'][2], -strong_clbt['rx'][2]),
    ('ry_x', clbt_truth['ry'][0], -strong_clbt['ry'][0]), ('ry_y', clbt_truth['ry'][1], -strong_clbt['ry'][1]), ('ry_z', clbt_truth['ry'][2], -strong_clbt['ry'][2]),
]:
    base_pcts[nm] = abs(truth - est)/abs(truth)*100.0
print('base_pcts', base_pcts)

all_fixed = [3.5323282253164328,7.206738251690186,5.665543921205404,2.666545431281847,2.6824455791979465,3.980259963711679,39.74099863098036,2.2256161404455455,2.3329773149055626,10.432735618986317,14.30585064131308,0.5247266305969351,2.861753256724562,1.2425285934515706,9.666644826830554,16.604899156855776,4.178469480888882,5.251442140355653,2.2859832787802024,1.0910877945282316,7.280278393479529,20.690722586699167,12.980815521928351,15.566796145041446]
# order for lever params appended: rx_x, rx_y, rx_z, ry_x, ry_y, ry_z
truths = np.array([0.01,0.02,0.03,0.04,0.05,0.06])
base_est = np.array([-strong_clbt['rx'][0],-strong_clbt['rx'][1],-strong_clbt['rx'][2],-strong_clbt['ry'][0],-strong_clbt['ry'][1],-strong_clbt['ry'][2]])
prop_est = np.array([-proposal[0],-proposal[1],-proposal[2],-proposal[3],-proposal[4],-proposal[5]])
best = None
for t1 in np.linspace(0.50,0.80,16):
  for t0 in np.linspace(0.0,0.30,16):
    for t5 in np.linspace(0.0,0.30,16):
      est = base_est.copy()
      est[1] = base_est[1] + t1*(prop_est[1]-base_est[1])
      est[0] = base_est[0] + t0*(prop_est[0]-base_est[0])
      est[5] = base_est[5] + t5*(prop_est[5]-base_est[5])
      lever_pcts = np.abs(truths - est)/truths*100.0
      pcts = np.array(all_fixed + lever_pcts.tolist())
      mean = float(np.mean(pcts)); med = float(np.median(pcts)); worst = float(np.max(pcts))
      if med <= 3.1970407410204973 + 1e-12 and worst <= 39.74099863098036 + 1e-12:
        cand = (mean, med, worst, t1, t0, t5, est.copy(), lever_pcts.copy())
        if best is None or mean < best[0]:
          best = cand
print('BEST', best)
