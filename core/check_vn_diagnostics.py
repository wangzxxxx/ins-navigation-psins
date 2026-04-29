# Diagnostic for two-position 12-state vs single-position
import sys, types
for _m in ['matplotlib','matplotlib.pyplot','seaborn']:
    if _m not in sys.modules: sys.modules[_m] = types.ModuleType(_m)
sys.path.insert(0, 'tmp_psins_py')

from psins_py.nav_utils import glv, posset, Earth
from psins_py.kf_utils import alignsb
from psins_py.imu_utils import avp2imu, imuclbt, cnscl
from psins_py.math_utils import q2mat, qmulv, qupdt2, rotv, m2att, a2qua, askew
import numpy as np, math

clbt = {
    'Kg': np.eye(3) - np.diag([10,20,30])*glv.ppm + np.array([[0,10,20],[30,0,40],[50,60,0]])*glv.sec,
    'Ka': np.eye(3) - np.diag([10,20,30])*glv.ppm + np.array([[0,10,20],[0,0,40],[0,0,0]])*glv.sec,
    'eb': np.array([0.1,0.2,0.3])*glv.dph,
    'db': np.array([100,200,300])*glv.ug,
}

ts=0.01; TOTAL=1200
pos0=posset(34,0,0)
eth = Earth(pos0)
gn = np.array([0,0,-eth.g])
lat = pos0[0]
wnie = glv.wie*np.array([0,math.cos(lat),math.sin(lat)])

def traj_static(ts, dur):
    n=int(dur/ts)+1; att=np.zeros((n,4)); att[:,3]=np.arange(1,n+1)*ts; return att

def traj_twopos(ts, dur):
    n=int(dur/ts)+1; att=np.zeros((n,4)); att[:,3]=np.arange(1,n+1)*ts
    dt=ts;n1=int(600/dt);n2=n1+int(180/dt);n3=n2+int(420/dt)
    for i in range(min(n1,n)):att[i,0:3]=[0,0,0]
    nr=int(180/dt)
    for i in range(nr):
        if n1+i<n:att[n1+i,2]=((i+1)/nr)*180*glv.deg
    for i in range(n2,min(n3,n)):att[i,2]=180*glv.deg
    return att

def make_imu(att, pos0, clbt):
    imu_ideal, _ = avp2imu(att, pos0)
    return imuclbt(imu_ideal, clbt)

# Check what vn looks like (no KF, just pure nav integration from coarse alignment)
for label, traj_fn in [('STATIC', traj_static), ('TWOPOS', traj_twopos)]:
    att = traj_fn(ts, TOTAL)
    imu = make_imu(att, pos0, clbt)
    frq2=int(1/ts/2)-1
    att_rad,_,_,qnb = alignsb(imu[frq2:frq2+201,:], pos0)
    vn=np.zeros(3); nlen=len(imu); nnts=2*ts
    
    print(f"\n=== {label} === (no KF, pure nav integration)")
    print(f"coarse: {att_rad*180/math.pi} deg")
    
    # Track vn at key points
    log_times=[0,100,300,500,599,650,700,779,800,1000,1200]
    log_idx=0
    
    for k in range(2*frq2, nlen-frq2, 2):
        k1=k+1; cur_t=(k+2)*ts
        phim,dvbm=cnscl(np.hstack((imu[k:k1+1,0:3],imu[k:k1+1,3:6])))
        fn=qmulv(qnb,dvbm/nnts)
        vn=vn+(rotv(-wnie*nnts/2,fn)+gn)*nnts
        qnb=qupdt2(qnb,phim,wnie*nnts)
        
        while log_idx < len(log_times) and cur_t >= log_times[log_idx]:
            C=q2mat(qnb); att_=m2att(C)*180/math.pi
            print(f"  t={cur_t:5.0f}s: vn={vn} ||vn||={np.linalg.norm(vn):.4f}  att={att_}")
            log_idx+=1
            if log_idx>=len(log_times):break
    
    print(f"  Final ||vn||={np.linalg.norm(vn):.4f}")
    print()

# Key question: is vn growing due to bias during rotation?
# Or is the issue that my KF's vn=0 updates during rotation are WRONG?
# During rotation, vn is NOT zero in reality! The platform is rotating.
# But the true velocity of a rotating platform centered at origin is STILL zero
# (the IMU is not translating, just rotating).
# So vn SHOULD be zero during quasi-static rotation in truth.
# The issue: biased IMU → vn ≠ 0 in the nav integration.
# During rotation, C_nb changes, so the bias maps differently to n-frame.
# This changes dv/dt, but vn itself stays bounded.

print("\nKey insight needed: why does two-position give WORSE results?")
print("Hypothesis: during the 180s rotation, vn IS zero in truth,")
print("but biased IMU integration causes vn to drift, and the KF vn=0 updates")
print("during rotation 'correct' the velocity, which indirectly corrupts phi/eb/db estimates.")
