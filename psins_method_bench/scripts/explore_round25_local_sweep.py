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
m24 = load(METHOD24,'m24'); src = load(SRC,'src25'); mod = src

def meanerr(clbt):
    truth = src.get_default_clbt(); vals=[]
    dKg_truth = truth['Kg'] - np.eye(3); dKa_truth = truth['Ka'] - np.eye(3)
    params = [
        (truth['eb'][0] / src.glv.dph, (-clbt['eb'][0]) / src.glv.dph), (truth['eb'][1] / src.glv.dph, (-clbt['eb'][1]) / src.glv.dph), (truth['eb'][2] / src.glv.dph, (-clbt['eb'][2]) / src.glv.dph),
        (truth['db'][0] / src.glv.ug, (-clbt['db'][0]) / src.glv.ug), (truth['db'][1] / src.glv.ug, (-clbt['db'][1]) / src.glv.ug), (truth['db'][2] / src.glv.ug, (-clbt['db'][2]) / src.glv.ug),
        (dKg_truth[0,0]/src.glv.ppm, (-(clbt['Kg']-np.eye(3))[0,0])/src.glv.ppm), (dKg_truth[1,0]/src.glv.sec, (-(clbt['Kg']-np.eye(3))[1,0])/src.glv.sec), (dKg_truth[2,0]/src.glv.sec, (-(clbt['Kg']-np.eye(3))[2,0])/src.glv.sec),
        (dKg_truth[0,1]/src.glv.sec, (-(clbt['Kg']-np.eye(3))[0,1])/src.glv.sec), (dKg_truth[1,1]/src.glv.ppm, (-(clbt['Kg']-np.eye(3))[1,1])/src.glv.ppm), (dKg_truth[2,1]/src.glv.sec, (-(clbt['Kg']-np.eye(3))[2,1])/src.glv.sec),
        (dKg_truth[0,2]/src.glv.sec, (-(clbt['Kg']-np.eye(3))[0,2])/src.glv.sec), (dKg_truth[1,2]/src.glv.sec, (-(clbt['Kg']-np.eye(3))[1,2])/src.glv.sec), (dKg_truth[2,2]/src.glv.ppm, (-(clbt['Kg']-np.eye(3))[2,2])/src.glv.ppm),
        (dKa_truth[0,0]/src.glv.ppm, (-(clbt['Ka']-np.eye(3))[0,0])/src.glv.ppm), (dKa_truth[0,1]/src.glv.sec, (-(clbt['Ka']-np.eye(3))[0,1])/src.glv.sec), (dKa_truth[0,2]/src.glv.sec, (-(clbt['Ka']-np.eye(3))[0,2])/src.glv.sec),
        (dKa_truth[1,1]/src.glv.ppm, (-(clbt['Ka']-np.eye(3))[1,1])/src.glv.ppm), (dKa_truth[1,2]/src.glv.sec, (-(clbt['Ka']-np.eye(3))[1,2])/src.glv.sec), (dKa_truth[2,2]/src.glv.ppm, (-(clbt['Ka']-np.eye(3))[2,2])/src.glv.ppm),
        (truth['Ka2'][0]/src.glv.ugpg2, (-clbt['Ka2'][0])/src.glv.ugpg2), (truth['Ka2'][1]/src.glv.ugpg2, (-clbt['Ka2'][1])/src.glv.ugpg2), (truth['Ka2'][2]/src.glv.ugpg2, (-clbt['Ka2'][2])/src.glv.ugpg2),
        (truth['rx'][0], -clbt['rx'][0]), (truth['rx'][1], -clbt['rx'][1]), (truth['rx'][2], -clbt['rx'][2]), (truth['ry'][0], -clbt['ry'][0]), (truth['ry'][1], -clbt['ry'][1]), (truth['ry'][2], -clbt['ry'][2])
    ]
    p=[]
    for t,e in params:
        p.append(abs(t-e)/abs(t)*100.0)
    return float(np.mean(p)), float(np.median(p)), float(np.max(p)), (-clbt['rx'][1], -clbt['ry'][2])

ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = m24._build_dataset(mod)
base_res = mod.run_calibration(imu_noisy, pos0, ts, n_states=42, bi_g=bi_g, tau_g=tau_g, bi_a=bi_a, tau_a=tau_a, label='42-GM1-R25-SWEEP')
strong = base_res[0]
# reuse round24 internals
eth = mod.Earth(pos0); wnie = mod.glv.wie * mod.np.array([0, mod.math.cos(pos0[0]), mod.math.sin(pos0[0])]); gn = mod.np.array([0,0,-eth.g]); Cba = mod.np.eye(3)
windows = m24._static_window_bounds(mod, imu_noisy, ts); imu_corr = m24._apply_strong_only(mod, imu_noisy, strong, ts)
lever_rows=[]; lever_targets=[]; weights=[]; ka2_theta=-mod.np.asarray(strong['Ka2'],dtype=float).reshape(3)
for s,e in windows:
    seg=imu_corr[s:e,:]
    if len(seg)<80: continue
    qnb = mod.alignsb(seg, pos0)[3]; vn = mod.np.zeros(3); fb_sq_sum = mod.np.zeros(3); ss_sum = mod.np.zeros((3,6)); dotwf = mod.imudot(seg,5.0)
    for k in range(len(seg)):
        wm=seg[k,0:3]; vm=seg[k,3:6]; dwb=dotwf[k,0:3]; fb=vm/ts; SS = mod.imulvS(wm/ts, dwb, Cba)
        fb_sq_sum += fb**2; ss_sum += SS[:,0:6]; qnb = mod.qupdt2(qnb, wm, wnie*ts); fn0 = mod.qmulv(qnb, fb); vn = vn + (mod.rotv(-wnie*ts/2, fn0)+gn)*ts
    mean_fb_sq = fb_sq_sum/max(len(seg),1); mean_ss=ss_sum/max(len(seg),1); A_lev=mean_ss; b=-vn/max(len(seg)*ts,1e-12)-mod.np.diag(mean_fb_sq)@ka2_theta
    coln=mod.np.linalg.norm(A_lev,axis=0)+1e-18; A_scaled=A_lev/coln[None,:]; sv=mod.np.linalg.svd(A_scaled, compute_uv=False); smax=float(mod.np.max(sv)); smin=float(mod.np.min(sv)); quality=smin/max(smax,1e-15); motion=mod.np.linalg.norm(mean_ss,ord='fro'); cond=smax/max(smin,1e-15)
    cond_soft=1.0/(1.0+(cond/2.5e11)**0.65); quality_soft=0.10+2.20*quality; motion_soft=0.35+0.06*motion; weight=max(len(seg)*cond_soft*quality_soft*motion_soft,1e-6)
    lever_rows.append(A_lev); lever_targets.append(b); weights.append(weight)
A=mod.np.vstack(lever_rows); b=mod.np.hstack(lever_targets); row_w=mod.np.repeat(mod.np.sqrt(mod.np.asarray(weights,dtype=float)),3); Aw=A*row_w[:,None]; bw=b*row_w
col_scale=mod.np.sqrt(mod.np.sum(Aw*Aw,axis=0)+1e-18); col_scale=mod.np.maximum(col_scale, mod.np.array([0.08]*6)); As=Aw/col_scale[None,:]
prior=-mod.np.concatenate((strong['rx'], strong['ry'])); prior_lever=mod.np.concatenate((strong['rx'], strong['ry'])); prior_s=prior*col_scale
ridge=np.diag((1.0/np.array([0.0037,0.0025,0.0025,0.0037,0.0025,0.0025]))**2)
U,S,Vt=mod.np.linalg.svd(As, full_matrices=False); smax=float(S[0]); trunc_floor=max(0.038*smax,1e-9); Sinv=np.array([1.0/s if s>=trunc_floor else 0.0 for s in S]); theta_pinv=Vt.T@(Sinv*(U.T@bw)); lhs=As.T@As+1.7*ridge; rhs=As.T@bw+1.7*ridge@prior_s; theta_map=mod.np.linalg.solve(lhs,rhs)
theta=(0.80*theta_map+0.20*theta_pinv)/col_scale; proposal=-np.asarray(theta).reshape(-1); raw_delta=np.clip(proposal-prior_lever,-0.0024,0.0024)
gram=As.T@As; diag_gram=np.diag(gram); coupling=np.sum(np.abs(gram),axis=1)+1e-18; diag_ratio=diag_gram/coupling; grad=np.abs(As.T@bw); score=diag_ratio*grad/np.sqrt(diag_gram+1e-18); order=np.argsort(score)[::-1]
print('order',order.tolist(),'score',score.tolist())
for t1 in [0.40,0.44,0.47,0.50,0.54]:
    for t2 in [0.02,0.04,0.06,0.08,0.10,0.12]:
        trust=np.zeros(6); trust[order[0]]=t1; trust[order[1]]=0.14 if score[order[1]]/max(score[order[0]],1e-18)>0.15 else 0.0; trust[order[2]]=t2
        lever=np.clip(prior_lever+trust*raw_delta,-0.08,0.08)
        refined=m24._copy_clbt(mod,strong); refined['rx']=lever[:3]; refined['ry']=lever[3:]
        mean,med,worst,pair=meanerr(refined)
        print({'t1':t1,'t2':t2,'mean':mean,'med':med,'worst':worst,'rx_y_ry_z':pair})
