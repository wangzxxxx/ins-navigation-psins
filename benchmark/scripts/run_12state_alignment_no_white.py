"""
12-state 对准实验（无白噪声，仅确定性偏置+刻度+安装误差）：
- 实验A：12-state @ 单位置静止对准
- 实验B：12-state @ 双位置对准

12-state：phi(3) + dv(3) + eb(3) + db(3)
"""
from __future__ import annotations
import sys, os, types, time, json, math, datetime

for _m in ['matplotlib','matplotlib.pyplot','seaborn']:
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(ROOT, 'psins_method_bench', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, os.path.join(ROOT, 'tmp_psins_py'))

from psins_py.nav_utils import glv, posset, Earth
from psins_py.kf_utils import alignsb
from psins_py.imu_utils import avp2imu, imuclbt, imudot, cnscl
from psins_py.math_utils import q2mat, qmulv, qupdt2, rotv, m2att, a2qua, askew
import numpy as np

SEED = 42
TS = 0.01
TOTAL_S = 1200.0

def get_default_clbt():
    """Match test_calibration_markov_pruned.py get_default_clbt."""
    Kg = np.eye(3) - np.diag([10,20,30])*glv.ppm + \
         np.array([[0,10,20],[30,0,40],[50,60,0]])*glv.sec
    Ka = np.eye(3) - np.diag([10,20,30])*glv.ppm + \
         np.array([[0,10,20],[0,0,40],[0,0,0]])*glv.sec
    return {
        'Kg': Kg, 'Ka': Ka,
        'eb': np.array([0.1,0.2,0.3])*glv.dph,
        'db': np.array([100,200,300])*glv.ug,
        'Ka2': np.array([10,20,30])*glv.ugpg2,
        'rx': np.array([1,2,3])/100,
        'ry': np.array([4,5,6])/100,
    }

def traj_static(ts, dur):
    n = int(dur/ts) + 1
    att = np.zeros((n, 4))
    att[:, 3] = np.arange(1, n+1)*ts
    return att

def traj_two_pos(ts, dur):
    n = int(dur/ts) + 1
    att = np.zeros((n, 4))
    att[:, 3] = np.arange(1, n+1)*ts
    dt=ts; n1=int(600/dt); n2=n1+int(180/dt); n3=n2+int(420/dt)
    for i in range(min(n1, n)): att[i,0:3]=[0,0,0]
    nr=int(180/dt)
    for i in range(nr):
        if n1+i<n: att[n1+i,2]=((i+1)/nr)*180*glv.deg
    for i in range(n2, min(n3, n)): att[i,2]=180*glv.deg
    return att

def make_imu(att, pos0, clbt):
    imu_ideal, _ = avp2imu(att, pos0)
    return imuclbt(imu_ideal, clbt)

def align_12state(imu, pos0, ts, clbt, label=''):
    """12-state KF alignment.
    
    State: x = [phi(3), dv(3), eb(3), db(3)] = 12
      phi = misalignment error (n-frame, rad)
      dv = velocity error (n-frame, m/s)
      eb = gyro bias (b-frame, rad/s)
      db = accel bias (b-frame, m/s^2)
    
    Standard INS error model:
      d(phi)/dt = -wnie×phi + C_nb*eb
      d(dv)/dt  = fn×phi + C_nb*db
      d(eb)/dt  = 0 (constant bias)
      d(db)/dt  = 0 (constant bias)
    
    Measurement: vn = 0 (during quasi-static periods)
    """
    print(f"  {label}: 12-state KF...", end=" ", flush=True)
    t0=time.time()

    eth=Earth(pos0); lat=pos0[0]
    gn=np.array([0,0,-eth.g])
    wnie=glv.wie*np.array([0,math.cos(lat),math.sin(lat)])
    frq2=int(1/ts/2)-1

    # ── Coarse alignment ──
    imu_align = imu[frq2:frq2+201, :]
    att_co_rad, _, _, qnb_co = alignsb(imu_align, pos0)
    att_co_deg = att_co_rad * 180/math.pi

    qnb = qnb_co.copy()
    vn = np.zeros(3)

    print(f"coarse: {att_co_deg[0]:.4f}/{att_co_deg[1]:.4f}/{att_co_deg[2]:.4f} deg  ", end="")

    # ── 12-state KF ──
    nx=12; xv=np.zeros(nx)

    # Initial covariance (large uncertainties)
    P=np.zeros((nx,nx))
    P[0:3,0:3]=np.diag([1.0,1.0,5.0])*(glv.deg)**2   # coarse alignment gives ~deg-level
    P[3:6,3:6]=np.diag([0.1,0.1,0.1])**2              # small after coarse alignment
    P[6:9,6:9]=np.diag([10,10,10])*(glv.dph)**2        # very loose gyro bias prior
    P[9:12,9:12]=np.diag([1000,1000,1000])*(glv.ug)**2 # very loose accel bias prior

    # Process noise (small for deterministic case)
    Q=np.zeros((nx,nx))
    Q[0:3,0:3]=np.diag([1e-12,1e-12,1e-10])
    Q[3:6,3:6]=np.diag([1e-14,1e-14,1e-12])

    # Measurement noise
    R=np.eye(3)*1e-6
    H=np.zeros((3,nx)); H[:,3:6]=np.eye(3)  # vn measurement

    nn=2; nlen=len(imu)
    meas=0; maxv=0.0

    nnts_v=nn*ts

    for k in range(2*frq2, nlen-frq2, nn):
        k1=k+nn-1

        # IMU coning/sculling
        wm=imu[k:k1+1,0:3]; vm=imu[k:k1+1,3:6]
        phim,dvbm=cnscl(np.hstack((wm,vm)))
        wb=phim/nnts_v; fb=dvbm/nnts_v

        # Navigation update
        fn=qmulv(qnb,fb)
        vn=vn+(rotv(-wnie*nnts_v/2,fn)+gn)*nnts_v
        qnb=qupdt2(qnb,phim,wnie*nnts_v)

        # ── Build F matrix (continuous-time INS error equations) ──
        C_nb=q2mat(qnb)
        F=np.zeros((nx,nx))

        # d(phi)/dt = -wnie×phi + C_nb*eb
        # Note: phi is the attitude ERROR in the navigation frame
        # The -wnie×phi term comes from Earth rotation in n-frame
        # C_nb*eb maps body-frame gyro bias to n-frame attitude rate
        F[0:3,0:3]=-askew(wnie)
        F[0:3,6:9]=C_nb  # eb (b-frame) → phi (n-frame)

        # d(dv)/dt = fn×phi + C_nb*db
        # fn×phi: velocity error from attitude misalignment
        # C_nb*db: velocity error from body-frame accel bias
        fnv=C_nb@fb
        F[3:6,0:3]=askew(fnv)
        F[3:6,9:12]=C_nb  # db (b-frame) → dv (n-frame)

        # d(eb)/dt=0, d(db)/dt=0
        # (constant biases = random walk with zero process noise)

        # Discretize: Phi = I + F*dt
        Phi=np.eye(nx)+F*nnts_v

        # Time update (state)
        xv=Phi@xv

        # Time update (covariance)
        P=Phi@P@Phi.T+Q
        P=0.5*(P+P.T)

        # Check if quasi-static (rotation rate low → vn should be ~0)
        ws=imu[max(0,k-frq2):min(k+frq2+1,nlen),0:3].mean(0)
        rr=np.linalg.norm(ws)/ts

        if rr < 20*glv.dph:  # ~0.1 deg/s threshold
            # Zero-velocity measurement
            zk=vn - H@xv  # innovation
            S=H@P@H.T+R  # H @ P @ H' + R

            try:
                K=P@H.T@np.linalg.inv(S)
                xv=xv+K@zk
                P=(np.eye(nx)-K@H)@P
                P=0.5*(P+P.T)
                # Add small process noise after measurement to prevent divergence
                P[0:3,0:3]+=np.diag([1e-12,1e-12,1e-10])
                P[3:6,3:6]+=np.diag([1e-14,1e-14,1e-12])
            except np.linalg.LinAlgError:
                pass
            meas+=1

        maxv=max(maxv,np.linalg.norm(vn))

    elapsed=time.time()-t0
    phi=xv[0:3]; eb_est=xv[6:9]; db_est=xv[9:12]
    phi_sec=np.abs(phi)/glv.sec

    eb_err=np.abs(eb_est-clbt['eb'])/glv.dph
    db_err=np.abs(db_est-clbt['db'])/glv.ug

    print(f"fine:{elapsed:.1f}s meas={meas} phi=[{phi_sec[0]:.1f},{phi_sec[1]:.1f},{phi_sec[2]:.1f}]\"", flush=True)

    return {
        'runtime_s':elapsed,
        'coarse_att_deg':[round(float(x),4) for x in att_co_deg],
        'attitude_error_arcsec':{
            'roll':round(float(phi_sec[0]),3),
            'pitch':round(float(phi_sec[1]),3),
            'yaw':round(float(phi_sec[2]),3),
            'norm':round(float(np.linalg.norm(phi)/glv.sec),3),
        },
        'bias_estimate':{
            'eb':[round(float(x),4) for x in eb_est/glv.dph],
            'db':[round(float(x),2) for x in db_est/glv.ug],
        },
        'bias_error':{
            'eb_err':[round(float(x),4) for x in eb_err],
            'db_err':[round(float(x),2) for x in db_err],
        },
        'vn_updates':meas,
        'max_vel':round(float(maxv),6),
    }

def main():
    print("="*55)
    print("12-state 对准（无白噪声，仅确定性偏置+刻度误差）")
    print("="*55)
    clbt=get_default_clbt()
    print(f"偏置: eb=[{clbt['eb'][0]/glv.dph},{clbt['eb'][1]/glv.dph},{clbt['eb'][2]/glv.dph}] d/h")
    print(f"      db=[{clbt['db'][0]/glv.ug},{clbt['db'][1]/glv.ug},{clbt['db'][2]/glv.ug}] ug")
    print(f"白噪声: 全部去除 | 时长={TOTAL_S}s | dt={TS}\n")

    pos0=posset(34,0,0)

    # A: single position
    print("--- A: 单位置静止 ---")
    att_a=traj_static(TS, TOTAL_S)
    imu_a=make_imu(att_a, pos0, clbt)
    print(f"  IMU: {imu_a.shape}")
    ra=align_12state(imu_a, pos0, TS, clbt, 'A')
    print()

    # B: two position
    print("--- B: 双位置 ---")
    att_b=traj_two_pos(TS, TOTAL_S)
    imu_b=make_imu(att_b, pos0, clbt)
    print(f"  IMU: {imu_b.shape}")
    rb=align_12state(imu_b, pos0, TS, clbt, 'B')

    print(f"\n{'='*55}\n结果:")
    for lb,r in [('A 单置',ra),('B 双位',rb)]:
        a2=r['attitude_error_arcsec']
        b=r['bias_estimate']; be=r['bias_error']
        print(f"\n【{lb}】 粗对准: {r['coarse_att_deg']} deg")
        print(f"  phi误差(角秒): roll={a2['roll']} pitch={a2['pitch']} yaw={a2['yaw']} norm={a2['norm']}")
        print(f"  eb估计(d/h): [{b['eb'][0]}, {b['eb'][1]}, {b['eb'][2]}]  误差=[{be['eb_err'][0]}, {be['eb_err'][1]}, {be['eb_err'][2]}]")
        print(f"  db估计(ug):  [{b['db'][0]}, {b['db'][1]}, {b['db'][2]}]  误差=[{be['db_err'][0]}, {be['db_err'][1]}, {be['db_err'][2]}]")
        print(f"  零速更新={r['vn_updates']}  max_v={r['max_vel']}m/s")

    print(f"\n双位 vs 单位:")
    for k in ['roll','pitch','yaw','norm']:
        va=ra['attitude_error_arcsec'][k]; vb=rb['attitude_error_arcsec'][k]
        d=va-vb; bet='双位' if d>0 else '单置'
        print(f"  {k}: {va}\" -> {vb}\" ({d:+}\" → {bet})")

    now=datetime.datetime.now().strftime('%Y-%m-%d')
    out=os.path.join(RESULTS_DIR,f'12state_no_white_{now}.json')
    with open(out,'w',encoding='utf-8') as f:
        json.dump({'A':ra, 'B':rb},f,ensure_ascii=False,indent=2)
    print(f"\n保存: {out}")

if __name__=='__main__':
    main()
