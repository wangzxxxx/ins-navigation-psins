"""
test_calibration_correlation_decay.py
--------------------------------------
Selective Correlation Decay (SCD) with LLM Phase Control.

Targets the EXACT mechanism of chronic noise poisoning:
the cross-correlations P[12:36, 0:12] that act as a pipeline
for ZUPT noise to leak from velocity states into calibration parameters.

During stationary periods: P_new = D @ P @ D  where D[12:36] = alpha < 1
This shrinks the noise pipeline while preserving positive semi-definiteness.
During rotation: Phi matrix naturally rebuilds the correlations.

LLM controls: alpha (decay rate), transition_duration (grace period after rotation).
"""
import numpy as np
import sys
import os
import math
import re
import matplotlib.pyplot as plt
import seaborn
from dotenv import load_dotenv
from openai import OpenAI

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from psins_py.nav_utils import glv, posset, Earth
from psins_py.imu_utils import attrottt, avp2imu, imuclbt, imudot, cnscl, imulvS
from psins_py.kf_utils import kfupdate, alignsb, nnts
from psins_py.math_utils import q2mat, qmulv, qupdt2, rotv, askew

# =====================================================================
# LLM Init
# =====================================================================
load_dotenv()
try:
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model_name = os.environ.get("OPENAI_MODEL_NAME", "gpt-4o")
    provider_id = os.environ.get("MODEL_PROVIDER_ID", "azure_openai")
    if api_key and base_url:
        client = OpenAI(api_key=api_key, base_url=base_url,
                        default_headers={"X-Model-Provider-Id": provider_id})
        print(f"[LLM] Client initialized: model={model_name}")
    else:
        client = None
except Exception as e:
    client = None

STATE_LABELS_36 = (
    ['phi_x', 'phi_y', 'phi_z'] + ['dv_x', 'dv_y', 'dv_z'] +
    ['eb_x', 'eb_y', 'eb_z'] + ['db_x', 'db_y', 'db_z'] +
    ['Kg_xx', 'Kg_yx', 'Kg_zx', 'Kg_xy', 'Kg_yy', 'Kg_zy', 'Kg_xz', 'Kg_yz', 'Kg_zz'] +
    ['Ka_xx', 'Ka_xy', 'Ka_xz', 'Ka_yy', 'Ka_yz', 'Ka_zz'] +
    ['Ka2_x', 'Ka2_y', 'Ka2_z'] +
    ['rx_x', 'rx_y', 'rx_z'] + ['ry_x', 'ry_y', 'ry_z']
)


# =====================================================================
# LLM SCD Advisor
# =====================================================================
def get_llm_scd_params(iteration, prev_stats=None):
    """
    LLM decides the correlation decay parameters for the upcoming iteration.
    Returns:
      alpha_sf:   decay factor for scale factors/misalign (indices 12-35), per ZUPT step
      alpha_bias: decay factor for biases (indices 6-11), per ZUPT step
      transition_duration: seconds after rotation to SKIP decay (allow signal through)
    """
    # TRUE ISOTROPIC: uniform alpha=0.98 for ALL cross-blocks, no LLM
    return {'alpha_sf': 0.98, 'alpha_bias': 0.98, 'transition_duration': 2.0}

    stats_str = ""
    if prev_stats:
        stats_str = f"""
Previous iteration results:
  - Scale Factor volatility (std over last 20%): {prev_stats['sf_vol']:.4e}
  - Bias volatility: {prev_stats['bias_vol']:.4e}
  - Number of ZUPT updates: {prev_stats['n_zupt']}
  - Number of SCD applications: {prev_stats['n_scd']}
"""

    system_prompt = """You are tuning a Selective Correlation Decay (SCD) mechanism for IMU calibration.

MECHANISM: After each ZUPT measurement update during stationary periods, the P matrix is scaled:
  P_new = D @ P @ D, where D is a diagonal matrix.
  D[0:6] = 1.0 (navigation states: never decay, keep ZUPT working)
  D[6:12] = alpha_bias (biases: usually 1.0 = no decay, or very mild)
  D[12:36] = alpha_sf (scale factors/misalignments: the main target)

This shrinks:
  - Cross-correlations P[12:36, 0:6]: the "noise pipeline" from velocity to calibration
  - Auto-correlations P[12:36, 12:36]: calibration uncertainty (by alpha^2)

During ROTATION, the Phi matrix naturally rebuilds these correlations.

PARAMETERS:
1. ALPHA_SF (0.90 to 1.0): Per-step decay for scale factors. 
   0.98 = moderate (decay by ~33% over 20 ZUPT steps = 4 seconds stationary)
   0.95 = aggressive (decay by ~64% over 20 steps)
   0.99 = gentle (decay by ~18% over 20 steps)
   1.0 = no decay (standard KF)
2. ALPHA_BIAS (0.95 to 1.0): Per-step decay for biases. Usually keep at 1.0 since biases are well-observable.
3. TRANSITION_DURATION (1.0 to 5.0 seconds): Grace period after rotation stops. No decay during this period to allow the filter to absorb genuine post-rotation signal.

STRATEGY:
- Iteration 1: Start moderate (alpha_sf=0.98) to assess effect
- Iterations 2-3: Adjust based on whether volatility decreased
- Iterations 4-5: Fine-tune

OUTPUT FORMAT:
REASONING: [analysis]
ALPHA_SF: [value]
ALPHA_BIAS: [value]
TRANSITION_DURATION: [seconds]
"""

    user_prompt = f"This is iteration {iteration}/5.\n{stats_str}\nWhat SCD parameters should I use?"

    messages = [
        {"role": "system", "content": system_prompt.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]

    try:
        response = client.chat.completions.create(model=model_name, messages=messages)
        content = response.choices[0].message.content.strip()
        print(f"\n{'='*60}\n[LLM SCD Advisor - Iter {iteration}]\n{content}\n{'='*60}\n")

        asf = 0.98; ab = 1.0; td = 2.0
        m = re.search(r'ALPHA_SF:\s*([0-9.]+)', content)
        if m: asf = float(m.group(1))
        m = re.search(r'ALPHA_BIAS:\s*([0-9.]+)', content)
        if m: ab = float(m.group(1))
        m = re.search(r'TRANSITION_DURATION:\s*([0-9.]+)', content)
        if m: td = float(m.group(1))

        return {
            'alpha_sf': np.clip(asf, 0.85, 1.0),
            'alpha_bias': np.clip(ab, 0.90, 1.0),
            'transition_duration': np.clip(td, 0.5, 10.0)
        }
    except Exception as e:
        print(f"  [LLM] Error: {e}")
        return {'alpha_sf': 0.98, 'alpha_bias': 1.0, 'transition_duration': 2.0}


# =====================================================================
# Standard Building Blocks
# =====================================================================
def imuadderr_full(imu_in, ts, arw=0.0, vrw=0.0, bi_g=0.0, tau_g=3600.0, bi_a=0.0, tau_a=3600.0):
    np.random.seed(42)
    imu = np.copy(imu_in); m = imu.shape[0]; sts = math.sqrt(ts)
    if arw > 0: imu[:, 0:3] += arw * sts * np.random.randn(m, 3)
    if vrw > 0: imu[:, 3:6] += vrw * sts * np.random.randn(m, 3)
    if bi_g > 0 and tau_g > 0:
        c = math.exp(-ts/tau_g); sw = bi_g*math.sqrt(2*ts/tau_g); b = np.zeros(3)
        for k in range(m): b = c*b + sw*np.random.randn(3); imu[k, 0:3] += b*ts
    if bi_a > 0 and tau_a > 0:
        c = math.exp(-ts/tau_a); sw = bi_a*math.sqrt(2*ts/tau_a); b = np.zeros(3)
        for k in range(m): b = c*b + sw*np.random.randn(3); imu[k, 3:6] += b*ts
    return imu

def get_default_clbt():
    Kg = np.eye(3) - np.diag([10.,20.,30.])*glv.ppm + np.array([[0,10,20],[30,0,40],[50,60,0]])*glv.sec
    Ka = np.eye(3) - np.diag([10.,20.,30.])*glv.ppm + np.array([[0,10,20],[0,0,40],[0,0,0]])*glv.sec
    return {'sf':np.ones(6),'Kg':Kg,'Ka':Ka,'eb':np.array([.1,.2,.3])*glv.dph,
            'db':np.array([100,200,300])*glv.ug,'Ka2':np.array([10,20,30])*glv.ugpg2,
            'rx':np.array([1,2,3])/100.,'ry':np.array([4,5,6])/100.}

def Ka_from_upper(x):
    d=np.zeros((3,3)); d[0,0]=x[0];d[0,1]=x[1];d[0,2]=x[2];d[1,1]=x[3];d[1,2]=x[4];d[2,2]=x[5]; return d

def clbtkfinit_36(nts):
    n=36; kf={'nts':nts,'n':n,'m':3}
    qv=np.zeros(n); qv[0:3]=0.01*glv.dpsh; qv[3:6]=100*glv.ugpsHz
    kf['Qk']=np.diag(qv)**2*nts; kf['Rk']=np.diag([.001,.001,.001])**2
    pv=np.zeros(n)
    pv[0:3]=np.array([.1,.1,1.])*glv.deg; pv[3:6]=1.; pv[6:9]=.1*glv.dph; pv[9:12]=1.*glv.mg
    pv[12:15]=[100*glv.ppm,100*glv.sec,100*glv.sec]
    pv[15:18]=[100*glv.sec,100*glv.ppm,100*glv.sec]
    pv[18:21]=[100*glv.sec,100*glv.sec,100*glv.ppm]
    pv[21]=100*glv.ppm;pv[22]=100*glv.sec;pv[23]=100*glv.sec
    pv[24]=100*glv.ppm;pv[25]=100*glv.sec;pv[26]=100*glv.ppm
    pv[27:30]=100*glv.ugpg2; pv[30:33]=.1; pv[33:36]=.1
    kf['Pxk']=np.diag(pv)**2
    Hk=np.zeros((3,n));Hk[:,3:6]=np.eye(3)
    kf['Hk']=Hk;kf['xk']=np.zeros(n);kf['I']=np.eye(n)
    return kf

def getFt_36(fb, wb, Cnb, wnie, SS):
    n=36; wX=askew(wnie); fX=askew(Cnb@fb)
    fx,fy,fz=fb[0],fb[1],fb[2]; wx,wy,wz=wb[0],wb[1],wb[2]
    CDf2=Cnb@np.diag(fb**2)
    Ca=np.zeros((3,6))
    Ca[:,0]=Cnb[:,0]*fx;Ca[:,1]=Cnb[:,0]*fy;Ca[:,2]=Cnb[:,0]*fz
    Ca[:,3]=Cnb[:,1]*fy;Ca[:,4]=Cnb[:,1]*fz;Ca[:,5]=Cnb[:,2]*fz
    Ft=np.zeros((n,n))
    Ft[0:3,0:3]=-wX;Ft[0:3,6:9]=-Cnb
    Ft[0:3,12:15]=-wx*Cnb;Ft[0:3,15:18]=-wy*Cnb;Ft[0:3,18:21]=-wz*Cnb
    Ft[3:6,0:3]=fX;Ft[3:6,9:12]=Cnb;Ft[3:6,21:27]=Ca;Ft[3:6,27:30]=CDf2
    Ft[3:6,30:36]=Cnb@SS[:,0:6]
    return Ft

def clbtkffeedback_pruned(kf, clbt):
    xk=kf['xk']; dKg=xk[12:21].reshape(3,3).T
    clbt['Kg']=(np.eye(3)-dKg)@clbt['Kg']
    dKa=Ka_from_upper(xk[21:27]); clbt['Ka']=(np.eye(3)-dKa)@clbt['Ka']
    clbt['Ka2']+=xk[27:30];clbt['eb']+=xk[6:9];clbt['db']+=xk[9:12]
    clbt['rx']+=xk[30:33];clbt['ry']+=xk[33:36]
    return clbt


# =====================================================================
# Main Calibration Engine
# =====================================================================
def run_calibration(imu1, pos0, ts, scd_mode=False, label=""):
    eth=Earth(pos0)
    wnie=glv.wie*np.array([0,math.cos(pos0[0]),math.sin(pos0[0])])
    gn=np.array([0,0,-eth.g]); Cba=np.eye(3)
    nn,_,nts,_=nnts(2,ts); frq2=int(1/ts/2)-1

    k=frq2
    for k in range(frq2,min(5*60*2*frq2,len(imu1)),2*frq2):
        ww=np.mean(imu1[k-frq2:k+frq2+1,0:3],axis=0)
        if np.linalg.norm(ww)/ts>20*glv.dph: break
    kstatic=k-3*frq2

    clbt={'Kg':np.eye(3),'Ka':np.eye(3),'Ka2':np.zeros(3),
          'eb':np.zeros(3),'db':np.zeros(3),'rx':np.zeros(3),'ry':np.zeros(3)}
    length=len(imu1); dotwf=imudot(imu1,5.0)
    iterations=5; P_trace,X_trace,iter_bounds=[],[],[]
    prev_stats = None

    def apply_clbt(imu_s, c):
        res=np.copy(imu_s)
        for i in range(len(res)):
            res[i,0:3]=c['Kg']@res[i,0:3]-c['eb']*ts
            res[i,3:6]=c['Ka']@res[i,3:6]-c['db']*ts
        return res

    for it in range(iterations):
        # Get SCD params from LLM
        scd_params = None
        if scd_mode:
            scd_params = get_llm_scd_params(it + 1, prev_stats)
            print(f"  [{label}] Iter {it+1}/{iterations} | alpha_sf={scd_params['alpha_sf']:.3f}, "
                  f"alpha_bias={scd_params['alpha_bias']:.3f}, trans_dur={scd_params['transition_duration']:.1f}s")
        else:
            print(f"  [{label}] Iter {it+1}/{iterations}")

        kf = clbtkfinit_36(nts)
        if it == iterations - 1:
            kf['Pxk'] = kf['Pxk'] * 100
            kf['Pxk'][:, 2] = 0; kf['Pxk'][2, :] = 0
            kf['xk'] = np.zeros(36)

        imu_align = apply_clbt(imu1[frq2:kstatic, :], clbt)
        _, _, _, qnb = alignsb(imu_align, pos0)
        vn = np.zeros(3); t1s = 0.0

        was_rotating = False
        time_since_rot_stop = 999.0
        scd_applied_this_phase = False  # Only apply SCD ONCE per stationary phase
        n_zupt = 0; n_scd = 0

        for k in range(2*frq2, length-frq2, nn):
            k1=k+nn-1
            wm=imu1[k:k1+1,0:3]; vm=imu1[k:k1+1,3:6]
            dwb=np.mean(dotwf[k:k1+1,0:3],axis=0)
            phim,dvbm=cnscl(np.hstack((wm,vm)))
            phim=clbt['Kg']@phim-clbt['eb']*nts
            dvbm=clbt['Ka']@dvbm-clbt['db']*nts
            wb=phim/nts; fb=dvbm/nts

            SS=imulvS(wb,dwb,Cba)
            fL=SS[:,0:6]@np.concatenate((clbt['rx'],clbt['ry']))
            fn=qmulv(qnb,fb-clbt['Ka2']*(fb**2)-fL)
            vn=vn+(rotv(-wnie*nts/2,fn)+gn)*nts
            qnb=qupdt2(qnb,phim,wnie*nts)

            t1s+=nts
            Ft=getFt_36(fb,wb,q2mat(qnb),wnie,SS)
            kf['Phikk_1']=np.eye(36)+Ft*nts
            kf=kfupdate(kf,TimeMeasBoth='T')

            if t1s>(0.2-ts/2):
                t1s=0.0
                ww=np.mean(imu1[k-frq2:k+frq2+1,0:3],axis=0)
                is_static=np.linalg.norm(ww)/ts<20*glv.dph

                if not is_static:
                    was_rotating=True; time_since_rot_stop=0.0
                    scd_applied_this_phase = False  # Reset for next stationary phase
                else:
                    if was_rotating:
                        was_rotating=False; time_since_rot_stop=0.0
                    else:
                        time_since_rot_stop+=0.2

                if is_static:
                    n_zupt += 1
                    # Standard ZUPT measurement update (untouched!)
                    kf = kfupdate(kf, yk=vn, TimeMeasBoth='M')

                    # === SCD: Apply ONCE per stationary period ===
                    # Fire when transition grace period ends, but only once
                    if scd_mode and scd_params and not scd_applied_this_phase:
                        if time_since_rot_stop >= scd_params['transition_duration']:
                            a_sf = scd_params['alpha_sf']
                            a_b  = scd_params['alpha_bias']
                            P = kf['Pxk']
                            # Decay nav↔calib cross-correlations
                            P[0:6, 12:36] *= a_sf;  P[12:36, 0:6] *= a_sf
                            # Decay nav↔bias cross-correlations (mild)
                            P[0:6, 6:12] *= a_b;    P[6:12, 0:6] *= a_b
                            # Decay bias↔calib cross-correlations
                            P[6:12, 12:36] *= a_sf;  P[12:36, 6:12] *= a_sf
                            scd_applied_this_phase = True
                            n_scd += 1

                P_trace.append(np.diag(kf['Pxk']))
                X_trace.append(np.copy(kf['xk']))

        if scd_mode and n_zupt > 0:
            X_arr = np.array(X_trace[-max(100, len(X_trace)//5):])
            prev_stats = {
                'sf_vol': float(np.mean(np.std(X_arr[:, 12:30], axis=0))),
                'bias_vol': float(np.mean(np.std(X_arr[:, 6:12], axis=0))),
                'n_zupt': n_zupt, 'n_scd': n_scd,
            }
            print(f"    Applied SCD to {n_scd}/{n_zupt} ZUPTs")

        if it != iterations - 1:
            clbt = clbtkffeedback_pruned(kf, clbt)
        iter_bounds.append(len(P_trace))

    return clbt, kf, np.array(P_trace), np.array(X_trace), iter_bounds


# =====================================================================
# Main
# =====================================================================
def main():
    ts=0.01; att0=np.array([1.,-91.,-91.])*glv.deg; pos0=posset(34.,0.,0.)
    paras=np.array([
        [1,0,1,0,90,9,70,70],[2,0,1,0,90,9,20,20],[3,0,1,0,90,9,20,20],
        [4,0,1,0,-90,9,20,20],[5,0,1,0,-90,9,20,20],[6,0,1,0,-90,9,20,20],
        [7,0,0,1,90,9,20,20],[8,1,0,0,90,9,20,20],[9,1,0,0,90,9,20,20],
        [10,1,0,0,90,9,20,20],[11,-1,0,0,90,9,20,20],[12,-1,0,0,90,9,20,20],
        [13,-1,0,0,90,9,20,20],[14,0,0,1,90,9,20,20],[15,0,0,1,90,9,20,20],
        [16,0,0,-1,90,9,20,20],[17,0,0,-1,90,9,20,20],[18,0,0,-1,90,9,20,20],
    ],dtype=float)
    paras[:,4]*=glv.deg
    ARW=0.005*glv.dpsh; VRW=5.*glv.ugpsHz; BI_G=0.002*glv.dph; BI_A=5.*glv.ug
    TAU_G=300.; TAU_A=300.

    print("Generating IMU trajectory...")
    att=attrottt(att0,paras,ts); imu,_=avp2imu(att,pos0)
    clbt_truth=get_default_clbt(); imu_clean=imuclbt(imu,clbt_truth)
    imu_noisy=imuadderr_full(imu_clean,ts,arw=ARW,vrw=VRW,bi_g=BI_G,tau_g=TAU_G,bi_a=BI_A,tau_a=TAU_A)

    print("\n[A] Clean (5 iter)...")
    res_clean=run_calibration(imu_clean,pos0,ts,scd_mode=False,label="Clean")
    print("\n[B] Noisy (Standard KF, 5 iter)...")
    res_noisy=run_calibration(imu_noisy,pos0,ts,scd_mode=False,label="Noisy KF")
    print("\n[C] Noisy + SCD (5 iter, LLM alpha)...")
    res_scd=run_calibration(imu_noisy,pos0,ts,scd_mode=True,label="SCD")

    # ── Accuracy Comparison ──
    cA,cB,cS=res_clean[0],res_noisy[0],res_scd[0]
    dKg=clbt_truth['Kg']-np.eye(3); dKa=clbt_truth['Ka']-np.eye(3)
    params=[
        ("eb_x",clbt_truth['eb'][0],lambda c:-c['eb'][0]),("eb_y",clbt_truth['eb'][1],lambda c:-c['eb'][1]),
        ("eb_z",clbt_truth['eb'][2],lambda c:-c['eb'][2]),
        ("db_x",clbt_truth['db'][0],lambda c:-c['db'][0]),("db_y",clbt_truth['db'][1],lambda c:-c['db'][1]),
        ("db_z",clbt_truth['db'][2],lambda c:-c['db'][2]),
        ("Kg_xx",dKg[0,0],lambda c:-(c['Kg']-np.eye(3))[0,0]),("Kg_yx",dKg[1,0],lambda c:-(c['Kg']-np.eye(3))[1,0]),
        ("Kg_zx",dKg[2,0],lambda c:-(c['Kg']-np.eye(3))[2,0]),("Kg_xy",dKg[0,1],lambda c:-(c['Kg']-np.eye(3))[0,1]),
        ("Kg_yy",dKg[1,1],lambda c:-(c['Kg']-np.eye(3))[1,1]),("Kg_zy",dKg[2,1],lambda c:-(c['Kg']-np.eye(3))[2,1]),
        ("Kg_xz",dKg[0,2],lambda c:-(c['Kg']-np.eye(3))[0,2]),("Kg_yz",dKg[1,2],lambda c:-(c['Kg']-np.eye(3))[1,2]),
        ("Kg_zz",dKg[2,2],lambda c:-(c['Kg']-np.eye(3))[2,2]),
        ("Ka_xx",dKa[0,0],lambda c:-(c['Ka']-np.eye(3))[0,0]),("Ka_xy",dKa[0,1],lambda c:-(c['Ka']-np.eye(3))[0,1]),
        ("Ka_xz",dKa[0,2],lambda c:-(c['Ka']-np.eye(3))[0,2]),("Ka_yy",dKa[1,1],lambda c:-(c['Ka']-np.eye(3))[1,1]),
        ("Ka_yz",dKa[1,2],lambda c:-(c['Ka']-np.eye(3))[1,2]),("Ka_zz",dKa[2,2],lambda c:-(c['Ka']-np.eye(3))[2,2]),
        ("Ka2_x",clbt_truth['Ka2'][0],lambda c:-c['Ka2'][0]),("Ka2_y",clbt_truth['Ka2'][1],lambda c:-c['Ka2'][1]),
        ("Ka2_z",clbt_truth['Ka2'][2],lambda c:-c['Ka2'][2]),
        ("rx_x",clbt_truth['rx'][0],lambda c:-c['rx'][0]),("rx_y",clbt_truth['rx'][1],lambda c:-c['rx'][1]),
        ("rx_z",clbt_truth['rx'][2],lambda c:-c['rx'][2]),
        ("ry_x",clbt_truth['ry'][0],lambda c:-c['ry'][0]),("ry_y",clbt_truth['ry'][1],lambda c:-c['ry'][1]),
        ("ry_z",clbt_truth['ry'][2],lambda c:-c['ry'][2]),
    ]

    print("\n"+"="*150)
    print("CALIBRATION ACCURACY — Selective Correlation Decay (5 iterations each)")
    print("="*150)
    hdr=f"{'Param':<10}{'Truth':>14}{'Clean':>14}{'Err%':>8}{'Noisy':>14}{'Err%':>8}{'SCD':>14}{'Err%':>8}  {'Win':>8}"
    print(hdr); print("-"*150)
    wins={'Noisy':0,'SCD':0}
    for nm,tr,ge in params:
        a,b,s=ge(cA),ge(cB),ge(cS)
        if abs(tr)>1e-15: ea,eb_,es=abs(tr-a)/abs(tr)*100,abs(tr-b)/abs(tr)*100,abs(tr-s)/abs(tr)*100
        else: ea,eb_,es=abs(a)*1e6,abs(b)*1e6,abs(s)*1e6
        w="SCD" if es<eb_ else ("Noisy" if eb_<es else "Tie")
        if w=="SCD": wins['SCD']+=1
        elif w=="Noisy": wins['Noisy']+=1
        print(f"{nm:<10}{tr:>+14.6e}{a:>+14.6e}{ea:>7.2f}%{b:>+14.6e}{eb_:>7.2f}%{s:>+14.6e}{es:>7.2f}%  {w:>8}")
    print("="*150)
    print(f"\nSCOREBOARD:  SCD wins: {wins['SCD']}/30  |  Noisy KF wins: {wins['Noisy']}/30  |  Ties: {30-wins['SCD']-wins['Noisy']}/30")
    print("="*150)

    # ── Plots ──
    PC,XC=res_clean[2],res_clean[3]; PN,XN=res_noisy[2],res_noisy[3]; PS,XS=res_scd[2],res_scd[3]
    out_dir=os.path.dirname(os.path.abspath(__file__))
    ml=min(len(PC),len(PN),len(PS)); ta=np.arange(ml)*(2*ts)
    for i in range(6,36):
        nm=STATE_LABELS_36[i]; fig,axes=plt.subplots(2,1,figsize=(14,8))
        axes[0].plot(ta,np.sqrt(PC[:ml,i]),'b',label='Clean',alpha=.8)
        axes[0].plot(ta,np.sqrt(PN[:ml,i]),'r--',label='Noisy KF',lw=2)
        axes[0].plot(ta,np.sqrt(PS[:ml,i]),'g',label='SCD',lw=2,alpha=.7)
        for b in res_scd[4][:-1]:
            if b<ml: axes[0].axvline(x=b*(2*ts),color='gray',ls=':',lw=1.5,alpha=.8)
        axes[0].set_title(f'[{i:02d}] {nm} — Uncertainty');axes[0].legend();axes[0].grid(True)
        axes[1].plot(ta,XC[:ml,i],'b',label='Clean',alpha=.8)
        axes[1].plot(ta,XN[:ml,i],'r--',label='Noisy KF',lw=2)
        axes[1].plot(ta,XS[:ml,i],'g',label='SCD',lw=2,alpha=.7)
        for b in res_scd[4][:-1]:
            if b<ml: axes[1].axvline(x=b*(2*ts),color='gray',ls=':',lw=1.5,alpha=.8)
        axes[1].set_title(f'[{i:02d}] {nm} — State Estimate');axes[1].legend();axes[1].grid(True)
        plt.tight_layout(); plt.savefig(os.path.join(out_dir,f"{i:02d}_{nm}.svg"),format='svg'); plt.close()
    print(f"\nDone! Plots saved to {out_dir}")

if __name__=="__main__":
    main()
