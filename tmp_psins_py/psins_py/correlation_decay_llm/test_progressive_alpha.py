"""
test_progressive_alpha.py
-------------------------
Iteration-Progressive Alpha: alpha varies across iterations.

Insight: In early iterations, calibration estimates are rough and
cross-correlations carry valuable structural information.
In later iterations, estimates are refined, and remaining
cross-correlation is more likely noise contamination.

Strategy: Start with NO decay (alpha=1.0) in iter 1 to let the filter
freely explore, then progressively INCREASE the decay in later iterations.

Also test the inverse: start aggressive, end gentle.
"""
import numpy as np
import sys, os, math
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from psins_py.nav_utils import glv, posset, Earth
from psins_py.imu_utils import attrottt, avp2imu, imuclbt, imudot, cnscl, imulvS
from psins_py.kf_utils import kfupdate, alignsb, nnts
from psins_py.math_utils import q2mat, qmulv, qupdt2, rotv, askew

# =====================================================================
# Building Blocks (same as sweep script)
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
# Calibration Engine with per-iteration alpha
# =====================================================================
def run_calibration(imu1, pos0, ts, alpha_schedule=None, trans_dur=2.0):
    """
    alpha_schedule: list of 5 alpha values, one per iteration.
    If None, standard KF (no SCD).
    """
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

    def apply_clbt(imu_s, c):
        res=np.copy(imu_s)
        for i in range(len(res)):
            res[i,0:3]=c['Kg']@res[i,0:3]-c['eb']*ts
            res[i,3:6]=c['Ka']@res[i,3:6]-c['db']*ts
        return res

    for it in range(5):
        alpha = alpha_schedule[it] if alpha_schedule else None

        kf = clbtkfinit_36(nts)
        if it == 4:
            kf['Pxk'] = kf['Pxk'] * 100
            kf['Pxk'][:, 2] = 0; kf['Pxk'][2, :] = 0
            kf['xk'] = np.zeros(36)

        imu_align = apply_clbt(imu1[frq2:kstatic, :], clbt)
        _, _, _, qnb = alignsb(imu_align, pos0)
        vn = np.zeros(3); t1s = 0.0

        was_rotating = False
        time_since_rot_stop = 999.0
        scd_applied = False

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
                    scd_applied = False
                else:
                    if was_rotating:
                        was_rotating=False; time_since_rot_stop=0.0
                    else:
                        time_since_rot_stop+=0.2

                if is_static:
                    kf = kfupdate(kf, yk=vn, TimeMeasBoth='M')

                    if alpha is not None and alpha < 1.0 and not scd_applied:
                        if time_since_rot_stop >= trans_dur:
                            P = kf['Pxk']
                            P[0:6, 12:36] *= alpha;  P[12:36, 0:6] *= alpha
                            P[6:12, 12:36] *= alpha;  P[12:36, 6:12] *= alpha
                            scd_applied = True

        if it != 4:
            clbt = clbtkffeedback_pruned(kf, clbt)

    return clbt

# =====================================================================
# Scoring
# =====================================================================
def score(clbt, clbt_truth, noisy_clbt):
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
    wins = 0; total_err = 0
    for nm,tr,ge in params:
        b_val = ge(noisy_clbt); s_val = ge(clbt)
        if abs(tr) > 1e-15:
            eb_ = abs(tr - b_val) / abs(tr) * 100
            es  = abs(tr - s_val) / abs(tr) * 100
        else:
            eb_ = abs(b_val) * 1e6; es = abs(s_val) * 1e6
        if es < eb_: wins += 1
        total_err += es
    return wins, total_err / 30

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

    print("\n[Baseline] Noisy Standard KF...")
    noisy_clbt = run_calibration(imu_noisy, pos0, ts)

    # =====================================================================
    # Test various progressive schedules
    # =====================================================================
    schedules = {
        "Constant 0.99 (control)":       [0.99,  0.99,  0.99,  0.99,  0.99],
        "Aggressive->Gentle":            [0.95,  0.97,  0.98,  0.99,  0.995],
        "Gentle->Aggressive":            [0.995, 0.99,  0.98,  0.97,  0.95],
        "No SCD early, 0.99 late":       [1.0,   1.0,   0.99,  0.99,  0.99],
        "0.99 early, No SCD late":       [0.99,  0.99,  0.99,  1.0,   1.0],
        "Only iter 1-3 SCD":             [0.99,  0.99,  0.99,  1.0,   1.0],
        "Only iter 3-5 SCD":             [1.0,   1.0,   0.99,  0.99,  0.99],
        "Only last iter SCD":            [1.0,   1.0,   1.0,   1.0,   0.99],
        "Only first iter SCD":           [0.99,  1.0,   1.0,   1.0,   1.0],
        "Ramp 0.97->0.995":             [0.97,  0.976, 0.983, 0.989, 0.995],
        "Ramp 0.995->0.97":             [0.995, 0.989, 0.983, 0.976, 0.97],
        "V-shape 0.99-0.97-0.99":       [0.99,  0.98,  0.97,  0.98,  0.99],
        "Inverted-V 0.97-0.99-0.97":    [0.97,  0.98,  0.99,  0.98,  0.97],
    }

    print("\n" + "="*80)
    print("PROGRESSIVE ALPHA SCHEDULE SWEEP (trans_dur=2.0s)")
    print("="*80)

    results = []
    for name, sched in schedules.items():
        print(f"\n  {name:40s} {sched} ...", end="", flush=True)
        scd_clbt = run_calibration(imu_noisy, pos0, ts, alpha_schedule=sched, trans_dur=2.0)
        wins, avg_err = score(scd_clbt, clbt_truth, noisy_clbt)
        print(f"  WINS: {wins}/30  (avg err: {avg_err:.2f}%)")
        results.append((name, sched, wins, avg_err))

    # Sort by wins then by avg_err
    results.sort(key=lambda x: (-x[2], x[3]))

    print("\n" + "="*80)
    print("RANKING")
    print("="*80)
    print(f"  {'#':>3}  {'Schedule':40s}  {'Wins':>6}  {'Avg Err%':>10}")
    print(f"  {'-'*65}")
    for i, (name, sched, wins, avg_err) in enumerate(results):
        marker = " <<<" if wins >= 21 else ""
        print(f"  {i+1:>3}  {name:40s}  {wins:>4}/30  {avg_err:>9.2f}%{marker}")

if __name__=="__main__":
    main()
