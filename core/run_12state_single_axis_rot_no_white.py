"""
12-state KF 单轴旋转对准实验（无白噪声）

方法：
1. 初始静止段（0-60s）：单位置静止对准
2. 旋转段（60-1140s）：Z轴连续旋转 10°/s，仅做状态传播，不做vn=0更新
   - 但旋转时，惯性偏置在不同C_nb下被"调制"到不同方向
   - 旋转段结束后，积累的运动信息会帮助分离eb/db分量
3. 最终静止段（1140-1200s）：再做静止对准

关键：旋转段期间不做vn=0更新！只在静止段更新。
"""
from __future__ import annotations
import sys, os, types, time, json, math

for _m in ['matplotlib','matplotlib.pyplot','seaborn']:
    if _m not in sys.modules:
        sys.modules[_m] = types.ModuleType(_m)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'tmp_psins_py'))

from psins_py.nav_utils import glv, posset, Earth
from psins_py.kf_utils import alignsb
from psins_py.imu_utils import avp2imu, imuclbt, cnscl
from psins_py.math_utils import q2mat, qmulv, qupdt2, rotv, m2att, a2qua, askew
import numpy as np

TS = 0.01
TOTAL_S = 1200.0
ROT_RATE_DEG_S = 10.0
ROT_START_S = 60.0
ROT_END_S = 1140.0

def get_default_clbt():
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

def traj_single_axis_rot(ts, dur, rate_deg_s=10.0, rot_start=60.0, rot_end=1140.0):
    """单轴旋转：先静止 → Z轴旋转 → 再静止"""
    n = int(dur/ts) + 1
    att = np.zeros((n, 4))
    att[:, 3] = np.arange(1, n+1) * ts
    
    for i in range(n):
        t = (i+1)*ts
        if t <= rot_start or t >= rot_end:
            # Static: attitude = 0
            att[i, 0:3] = [0, 0, 0]
        else:
            # Rotating: continuous yaw
            rot_time = t - rot_start
            angle_deg = rate_deg_s * rot_time
            att[i, 2] = angle_deg % 360.0 * glv.deg
    
    return att

def make_imu(att, pos0, clbt):
    imu_ideal, _ = avp2imu(att, pos0)
    return imuclbt(imu_ideal, clbt)

def align_12state_rotating(imu, pos0, ts, clbt, rot_start_s, rot_end_s, label=''):
    """12-state KF with selective vn=0 updates during rotation.
    
    Only applies vn=0 updates during static phases, NOT during rotation.
    """
    print(f"  {label}: 12-state KF (selective ZUPT)...", end=" ", flush=True)
    t0=time.time()

    eth=Earth(pos0); lat=pos0[0]
    gn=np.array([0,0,-eth.g])
    wnie=glv.wie*np.array([0,math.cos(lat),math.sin(lat)])
    frq2=int(1/ts/2)-1

    # Coarse alignment from initial static segment
    imu_align = imu[frq2:frq2+201, :]
    att_co_rad, _, _, qnb_co = alignsb(imu_align, pos0)
    att_co_deg = att_co_rad * 180/math.pi

    qnb = qnb_co.copy()
    vn = np.zeros(3)

    print(f"coarse: {att_co_deg[0]:.4f}/{att_co_deg[1]:.4f}/{att_co_deg[2]:.4f} deg  ", end="")

    # 12-state KF
    nx=12; xv=np.zeros(nx)

    P=np.zeros((nx,nx))
    P[0:3,0:3]=np.diag([1.0,1.0,5.0])*(glv.deg)**2
    P[3:6,3:6]=np.diag([0.1,0.1,0.1])**2
    P[6:9,6:9]=np.diag([10,10,10])*(glv.dph)**2
    P[9:12,9:12]=np.diag([1000,1000,1000])*(glv.ug)**2

    Q=np.zeros((nx,nx))
    Q[0:3,0:3]=np.diag([1e-12,1e-12,1e-10])
    Q[3:6,3:6]=np.diag([1e-14,1e-14,1e-12])

    R=np.eye(3)*1e-6
    H=np.zeros((3,nx)); H[:,3:6]=np.eye(3)

    nn=2; nlen=len(imu)
    meas=0; meas_rot=0; maxv=0.0
    nnts_v=nn*ts

    for k in range(2*frq2, nlen-frq2, nn):
        k1=k+nn-1

        # IMU coning/sculling
        wm=imu[k:k1+1,0:3]; vm=imu[k:k1+1,3:6]
        phim,dvbm=cnscl(np.hstack((wm,vm)))

        # Navigation update
        fn=qmulv(qnb,dvbm/nnts_v)
        vn=vn+(rotv(-wnie*nnts_v/2,fn)+gn)*nnts_v
        qnb=qupdt2(qnb,phim,wnie*nnts_v)

        # Build F matrix
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

        # Determine if we should do vn=0 update
        cur_t = (k+nn)*ts
        in_rotation = rot_start_s <= cur_t < rot_end_s
        
        if in_rotation:
            # During rotation: skip vn=0 update
            meas_rot += 1
            continue

        # Check quasi-static for static phases
        ws=imu[max(0,k-frq2):min(k+frq2+1,nlen),0:3].mean(0)
        rr=np.linalg.norm(ws)/ts
        
        if rr < 20*glv.dph:
            zk=vn - H@xv
            S=H@P@H.T+R
            try:
                K=P@H.T@np.linalg.inv(S)
                xv=xv+K@zk
                P=(np.eye(nx)-K@H)@P
                P=0.5*(P+P.T)
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

    print(f"fine:{elapsed:.1f}s static_updates={meas} rot_skip={meas_rot} phi=[{phi_sec[0]:.1f},{phi_sec[1]:.1f},{phi_sec[2]:.1f}]\"", flush=True)

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
        'rot_skip':meas_rot,
        'max_vel':round(float(maxv),6),
    }

def align_12state_static(imu, pos0, ts, clbt, label=''):
    """Standard 12-state KF (no rotation, all vn=0 updates)."""
    print(f"  {label}: 12-state KF (static)...", end=" ", flush=True)
    t0=time.time()

    eth=Earth(pos0); lat=pos0[0]
    gn=np.array([0,0,-eth.g])
    wnie=glv.wie*np.array([0,math.cos(lat),math.sin(lat)])
    frq2=int(1/ts/2)-1

    imu_align = imu[frq2:frq2+201, :]
    att_co_rad, _, _, qnb_co = alignsb(imu_align, pos0)
    att_co_deg = att_co_rad * 180/math.pi

    qnb = qnb_co.copy()
    vn = np.zeros(3)
    print(f"coarse: {att_co_deg[0]:.4f}/{att_co_deg[1]:.4f}/{att_co_deg[2]:.4f} deg  ", end="")

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

    for k in range(2*frq2, nlen-frq2, nn):
        k1=k+nn-1
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
                xv=xv+K@zk
                P=(np.eye(nx)-K@H)@P
                P=0.5*(P+P.T)
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
        'bias_estimate':{'eb':[round(float(x),4) for x in eb_est/glv.dph],'db':[round(float(x),2) for x in db_est/glv.ug]},
        'bias_error':{'eb_err':[round(float(x),4) for x in eb_err],'db_err':[round(float(x),2) for x in db_err]},
        'vn_updates':meas,
        'max_vel':round(float(maxv),6),
    }

def main():
    print("="*55)
    print("12-state 单轴旋转对准（无白噪声）")
    print("="*55)
    clbt=get_default_clbt()
    print(f"偏置: eb=[{clbt['eb'][0]/glv.dph},{clbt['eb'][1]/glv.dph},{clbt['eb'][2]/glv.dph}] d/h")
    print(f"      db=[{clbt['db'][0]/glv.ug},{clbt['db'][1]/glv.ug},{clbt['db'][2]/glv.ug}] ug\n")

    pos0=posset(34,0,0)

    # A: 1200s pure static
    print("--- A: 单位置静止 (1200s) ---")
    n=int(TOTAL_S/TS)+1
    att_a=np.zeros((n,4)); att_a[:,3]=np.arange(1,n+1)*TS
    imu_a=make_imu(att_a,pos0,clbt)
    ra=align_12state_static(imu_a,pos0,TS,clbt,'A')
    print()

    # B: 60s static → 1080s rotation → 60s static
    print("--- B: 单轴旋转 Z 10°/s (60s静止→1080s旋转→60s静止) ---")
    att_b=traj_single_axis_rot(TS,TOTAL_S,rate_deg_s=10.0,rot_start=ROT_START_S,rot_end=ROT_END_S)
    imu_b=make_imu(att_b,pos0,clbt)
    rb=align_12state_rotating(imu_b,pos0,TS,clbt,ROT_START_S,ROT_END_S,'B')
    print()

    # C: 60s static → 1080s rotation → 60s static, but WITH vn=0 during rotation
    # (This shows what happens if we naively apply vn=0 during rotation)
    print("--- C: 单轴旋转 + 全量 vn=0更新 (错误方法) ---")
    att_c=att_b.copy()
    imu_c=imu_b.copy()
    rc=align_12state_static(imu_c,pos0,TS,clbt,'C_naive')
    print()

    # Summary
    print(f"{'='*55}")
    print("结果对比:")
    for lb,r in [('A 静态',ra),('B 旋转选择性',rb),('C 旋转全量',rc)]:
        a2=r['attitude_error_arcsec']
        b=r['bias_estimate']; be=r['bias_error']
        print(f"\n{lb}:")
        print(f"  phi误差: roll={a2['roll']} pitch={a2['pitch']} yaw={a2['yaw']} → norm={a2['norm']}\"")
        print(f"  eb估计: [{b['eb'][0]},{b['eb'][1]},{b['eb'][2]}] d/h  误差: [{be['eb_err'][0]},{be['eb_err'][1]},{be['eb_err'][2]}]")
        print(f"  db估计: [{b['db'][0]},{b['db'][1]},{b['db'][2]}] ug  误差: [{be['db_err'][0]},{be['db_err'][1]},{be['db_err'][2]}]")
        print(f"  vn更新={r['vn_updates']} max_v={r['max_vel']:.2f}")

    now='2026-04-20'
    import datetime
    now=datetime.datetime.now().strftime('%Y-%m-%d')
    out=os.path.join(ROOT,'psins_method_bench','results',f'12state_single_axis_rot_{now}.json')
    os.makedirs(os.path.dirname(out),exist_ok=True)
    with open(out,'w',encoding='utf-8') as f:
        json.dump({'A':ra,'B':rb,'C_naive':rc},f,ensure_ascii=False,indent=2)
    print(f"\n保存: {out}")

if __name__=='__main__':
    main()
