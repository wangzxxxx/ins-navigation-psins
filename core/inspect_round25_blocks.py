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
m24 = load(METHOD24, 'm24i'); src = load(SRC, 'srci'); mod=src

ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = m24._build_dataset(mod)
base_res = mod.run_calibration(imu_noisy, pos0, ts, n_states=42, bi_g=bi_g, tau_g=tau_g, bi_a=bi_a, tau_a=tau_a, label='42-GM1-R25-INSPECT')
strong = base_res[0]
windows = m24._static_window_bounds(mod, imu_noisy, ts)
imu_corr = m24._apply_strong_only(mod, imu_noisy, strong, ts)
eth = mod.Earth(pos0); wnie = mod.glv.wie * mod.np.array([0, mod.math.cos(pos0[0]), mod.math.sin(pos0[0])]); gn = mod.np.array([0,0,-eth.g]); Cba = mod.np.eye(3)
ka2_theta = -mod.np.asarray(strong['Ka2'], dtype=float).reshape(3)
lever_rows=[]; lever_targets=[]; weights=[]
for s,e in windows:
    seg = imu_corr[s:e,:]
    if len(seg) < 80: continue
    qnb = mod.alignsb(seg, pos0)[3]; vn = mod.np.zeros(3); fb_sq_sum = mod.np.zeros(3); ss_sum = mod.np.zeros((3,6)); dotwf = mod.imudot(seg, 5.0)
    for k in range(len(seg)):
        wm = seg[k,0:3]; vm = seg[k,3:6]; dwb = dotwf[k,0:3]
        fb = vm / ts; SS = mod.imulvS(wm / ts, dwb, Cba)
        fb_sq_sum += fb**2; ss_sum += SS[:,0:6]
        qnb = mod.qupdt2(qnb, wm, wnie*ts); fn0 = mod.qmulv(qnb, fb); vn = vn + (mod.rotv(-wnie*ts/2, fn0) + gn) * ts
    mean_fb_sq = fb_sq_sum/len(seg); mean_ss = ss_sum/len(seg)
    A_lev = mean_ss; b = -vn/(len(seg)*ts) - mod.np.diag(mean_fb_sq) @ ka2_theta
    coln = mod.np.linalg.norm(A_lev, axis=0) + 1e-18
    A_scaled = A_lev / coln[None,:]
    sv = mod.np.linalg.svd(A_scaled, compute_uv=False); smax = float(mod.np.max(sv)); smin = float(mod.np.min(sv)); cond = smax/max(smin,1e-15)
    quality = smin/max(smax,1e-15); motion = mod.np.linalg.norm(mean_ss, ord='fro')
    weight = max(len(seg) * (1.0/(1.0+(cond/2.5e11)**0.65)) * (0.10+2.20*quality) * (0.35+0.06*motion), 1e-6)
    lever_rows.append(A_lev); lever_targets.append(b); weights.append(weight)
A = np.vstack(lever_rows); b = np.hstack(lever_targets)
row_w = np.repeat(np.sqrt(np.asarray(weights)), 3)
Aw = A*row_w[:,None]; bw = b*row_w
col_scale = np.sqrt(np.sum(Aw*Aw, axis=0)+1e-18); col_scale = np.maximum(col_scale, np.array([0.08]*6)); As = Aw/col_scale[None,:]
prior = -np.concatenate((strong['rx'], strong['ry']))
print('prior est', (-prior).tolist())
for cols in ([0,3],[1,4],[2,5],[0,1,3],[1,2,4,5],[0,2,3,5],[0,1,2,3,4,5]):
    cols=list(cols)
    Ablk=As[:,cols]
    lhs=Ablk.T@Ablk + np.diag(np.ones(len(cols))*10.0)
    rhs=Ablk.T@bw + np.diag(np.ones(len(cols))*10.0) @ (prior[cols]*col_scale[cols])
    sol=np.linalg.solve(lhs,rhs)/col_scale[cols]
    est=-sol
    print('cols',cols,'est',est.tolist())
