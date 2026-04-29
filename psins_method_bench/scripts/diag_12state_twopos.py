"""
诊断双位置 12-state KF：分段检查姿态误差和偏置估计的变化。
"""
from __future__ import annotations
import sys, os, types, math, json

for _m in ['matplotlib','matplotlib.pyplot','seaborn']:
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'tmp_psins_py'))

from psins_py.nav_utils import glv, posset, Earth
from psins_py.kf_utils import alignsb
from psins_py.imu_utils import avp2imu, imuclbt, cnscl, imudot
from psins_py.math_utils import q2mat, qmulv, qupdt2, rotv, m2att, a2qua, askew
import numpy as np

TS = 0.01; TOTAL_S = 1200.0

def get_clbt():
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

def traj_two_pos(ts, dur):
    n = int(dur/ts)+1
    att = np.zeros((n, 4))
    att[:, 3] = np.arange(1,n+1)*ts
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

def diag_12state_twopos(imu, pos0, ts, clbt):
    """Run 12-state KF on two-position data, logging results at key points."""
    eth=Earth(pos0); lat=pos0[0]
    gn=np.array([0,0,-eth.g])
    wnie=glv.wie*np.array([0,math.cos(lat),math.sin(lat)])
    frq2=int(1/ts/2)-1

    # Coarse alignment
    imu_align = imu[frq2:frq2+201,:]
    att_rad, _, _, qnb = alignsb(imu_align, pos0)
    print(f"Coarse: {att_rad*180/math.pi}")

    vn = np.zeros(3)
    nx=12; xv=np.zeros(nx)
    P=np.zeros((nx,nx))
    P[0:3,0:3]=np.diag([1,1,5])*(glv.deg)**2
    P[3:6,3:6]=np.diag([0.1,0.1,0.1])**2
    P[6:9,6:9]=np.diag([10,10,10])*(glv.dph)**2
    P[9:12,9:12]=np.diag([1000,1000,1000])*(glv.ug)**2
    Q=np.zeros((nx,nx))
    Q[0:3,0:3]=np.diag([1e-12,1e-12,1e-10])
    Q[3:6,3:6]=np.diag([1e-14,1e-14,1e-12])
    R=np.eye(3)*1e-6
    H=np.zeros((3,nx)); H[:,3:6]=np.eye(3)

    nn=2; nlen=len(imu)
    meas=0; maxv=0.0
    nnts_v=nn*ts
    
    # Log points: 0s, 300s, 500s, 550s, 600s, 650s, 700s, 750s, 780s, 900s, 1200s
    log_times = [0, 300, 500, 550, 600, 650, 700, 750, 780, 900, 1200]
    log_data = []
    
    for k in range(2*frq2, nlen-frq2, nn):
        k1=k+nn-1
        cur_time = (k+nn)*ts
        
        wm=imu[k:k1+1,0:3]; vm=imu[k:k1+1,3:6]
        phim,dvbm=cnscl(np.hstack((wm,vm)))
        fn=qmulv(qnb,dvbm/nnts_v)
        vn=vn+(rotv(-wnie*nnts_v/2,fn)+gn)*nnts_v
        qnb=qupdt2(qnb,phim,wnie*nnts_v)
        
        C_nb=q2mat(qnb)
        F=np.zeros((nx,nx))
        F[0:3,0:3]=-askew(wnie)
        F[0:3,6:9]=C_nb
        fnv=C_nb@(dvbm/nnts_v)
        F[3:6,0:3]=askew(fnv)
        F[3:6,9:12]=C_nb
        
        Phi=np.eye(nx)+F*nnts_v
        xv=Phi@xv
        P=Phi@P@Phi.T+Q
        P=0.5*(P+P.T)
        
        ws=imu[max(0,k-frq2):min(k+frq2+1,nlen),0:3].mean(0)
        rr=np.linalg.norm(ws)/ts
        
        if rr < 20*glv.dph:
            zk=vn-H@xv
            S=H@P@H.T+R
            try:
                K=P@H.T@np.linalg.inv(S)
                xv+=K@zk
                P=(np.eye(nx)-K@H)@P
                P=0.5*(P+P.T)
                P[0:3,0:3]+=np.diag([1e-12,1e-12,1e-10])
                P[3:6,3:6]+=np.diag([1e-14,1e-14,1e-12])
            except: pass
            meas+=1
        maxv=max(maxv,np.linalg.norm(vn))
        
        # Log at key points
        if cur_time >= log_times and any(abs(cur_time - lt) < nnts_v for lt in log_times if lt > 0):
            log_times = [t for t in log_times if abs(cur_time - t) >= nnts_v]
            phi_sec = np.abs(xv[0:3])/glv.sec
            eb = xv[6:9]/glv.dph
            db = xv[9:12]/glv.ug
            phase = "pre-rot" if cur_time < 600 else ("during-rot" if cur_time < 780 else "post-rot")
            print(f"  t={cur_time:.0f}s ({phase}): phi=[{phi_sec[0]:.1f},{phi_sec[1]:.1f},{phi_sec[2]:.1f}]\" "
                  f"eb=[{eb[0]:.4f},{eb[1]:.4f},{eb[2]:.4f}]d/h "
                  f"db=[{db[0]:.1f},{db[1]:.1f},{db[2]:.1f}]ug "
                  f"vn=[{vn[0]:.4f},{vn[1]:.4f},{vn[2]:.4f}] rr/wnie={rr/glv.dph:.2f}")
    
    phi_final = np.abs(xv[0:3])/glv.sec
    print(f"\nFinal: phi=[{phi_final[0]:.1f},{phi_final[1]:.1f},{phi_final[2]:.1f}]\" norm={np.linalg.norm(phi_final):.1f}\"")
    print(f"max_v={maxv:.4f} meas={meas}")

# Run diagnostics
print("=== 双位置 12-state KF 分段诊断 ===\n")
clbt = get_clbt()
pos0 = posset(34,0,0)
att = traj_two_pos(TS, TOTAL_S)
imu = make_imu(att, pos0, clbt)
diag_12state_twopos(imu, pos0, TS, clbt)
