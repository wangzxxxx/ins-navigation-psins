import numpy as np
import math
from scipy import signal
from .math_utils import a2mat, m2att, rv2m, m2rv, rv2q, q2mat, a2qua, qmulv, qupdt2, askew, cros, rotv, normv
from .nav_utils import glv, Earth

def ar1coefs(ts, tau):
    '''
    AR(1) filter coefficient calculation.
    '''
    tau = np.asarray(tau).flatten()
    a2 = np.exp(-ts / tau)
    b = 1 - a2
    a = np.column_stack((np.ones_like(a2), -a2))
    return b, a

def attrottt(att0, rotparas, ts):
    '''
    Attitude data simulation rotated by turntable.
    '''
    att0 = np.asarray(att0).flatten()
    rotparas = np.atleast_2d(rotparas)
    
    if rotparas.shape[1] == 6:
        rotparas = np.concatenate((rotparas, np.full((rotparas.shape[0], 1), 10.0)), axis=1)
    if rotparas.shape[1] == 7:
        rotparas = np.concatenate((rotparas, rotparas[:, 6:7]), axis=1)
        
    b, a = ar1coefs(ts, 10*ts)
    att = [att0.copy()]
    
    for k in range(rotparas.shape[0]):
        rpi = rotparas[k, :]
        U = rpi[1:4]
        U = U / np.linalg.norm(U)
        angle = rpi[4]
        T, T0, T1 = rpi[5], rpi[6], rpi[7]
        
        wi = angle / math.floor(T / ts)
        angles = np.concatenate((
            np.zeros(math.floor(T0 / ts)),
            np.arange(1, math.floor(T / ts) + 1) * wi,
            np.full(math.floor(T1 / ts), angle)
        ))
        
        # scipy.signal.filtfilt requires at least padlen filtering sequence len
        padlen = min(3 * max(len(a[0]), len(b)), len(angles) - 1)
        if padlen > 0:
            angles = signal.filtfilt(b, a[0], angles, padlen=padlen)
            
        len_angles = len(angles)
        atti = np.zeros((len_angles, 3))
        Cnb0 = a2mat(att[-1])
        
        for kk in range(len_angles):
            # Cnb0 * rv2m(U * angles[kk])
            atti[kk, :] = m2att(Cnb0 @ rv2m(U * angles[kk]))
            
        att.extend(atti)
        
    att = np.array(att)
    time_col = np.arange(1, att.shape[0] + 1) * ts
    return np.column_stack((att, time_col))

def avp2imu(avp, pos0=None):
    '''
    Simulate SIMU sensor outputs from attitude, velocity & position profile.
    '''
    avp = np.atleast_2d(avp)
    if pos0 is not None:
        pos0 = np.asarray(pos0).flatten()
        # avp = [att, pos0, time]
        length = avp.shape[0]
        avp_full = np.zeros((length, 10))
        avp_full[:, 0:3] = avp[:, 0:3]
        avp_full[:, 3:6] = 0.0
        avp_full[:, 6:9] = np.tile(pos0, (length, 1))
        avp_full[:, 9] = avp[:, -1]
        avp = avp_full
        
    length = avp.shape[0]
    ts = avp[1, 9] - avp[0, 9]
    ts2 = ts / 2
    
    Cbn_1 = a2mat(avp[0, 0:3]).T
    vn_1 = avp[0, 3:6]
    pos_1 = avp[0, 6:9]
    wm_1 = np.zeros(3)
    vm_1 = np.zeros(3)
    
    imu = np.zeros((length, 6))
    
    for k in range(1, length):
        Cnb = a2mat(avp[k, 0:3])
        vn = avp[k, 3:6]
        pos = avp[k, 6:9]
        
        eth = Earth((pos_1 + pos) / 2, (vn_1 + vn) / 2)
        
        phim = m2rv(Cbn_1 @ rv2m(eth.wnin * ts) @ Cnb)
        wm = np.linalg.inv(np.eye(3) + askew(wm_1 / 12)) @ phim
        
        dvbm = Cbn_1 @ qmulv(rv2q(eth.wnin * ts2), vn - vn_1 - eth.gcc * ts)
        vm = np.linalg.inv(np.eye(3) + askew(wm / 2 + wm_1 / 12)) @ (dvbm - cros(vm_1, wm) / 12)
        
        imu[k, :] = np.concatenate((wm, vm))
        
        Cbn_1 = Cnb.T
        vn_1 = vn
        pos_1 = pos
        wm_1 = wm
        vm_1 = vm
        
    imu_out = np.column_stack((imu[1:, :], avp[1:, 9]))
    avp0 = avp[0, 0:9]
    return imu_out, avp0

def conepolyn(wm):
    return np.zeros(3)

def scullpolyn(wm, vm):
    return np.zeros(3)

def cnscl(imu):
    '''
    Coning & sculling compensation for 1-sample.
    '''
    imu = np.atleast_2d(imu)
    wm = imu[:, 0:3]
    vm = imu[:, 3:6] if imu.shape[1] >= 6 else None
    
    n = imu.shape[0]
    if n == 1:
        phim = np.copy(wm[0])
        dvbm = np.copy(vm[0]) if vm is not None else np.zeros(3)
        return phim, dvbm
        
    wmm = np.sum(wm, axis=0)
    cm = np.zeros(3)  # Simplified compensation for optimal
    if n > 1:
        cm = np.dot(glv.cs[n-2, 0:n-1], wm[0:n-1, :])
    dphim = cros(cm, wm[-1, :])
    phim = wmm + dphim
    
    if vm is not None:
        vmm = np.sum(vm, axis=0)
        sm = np.zeros(3)
        if n > 1:
            sm = np.dot(glv.cs[n-2, 0:n-1], vm[0:n-1, :])
        scullm = cros(cm, vm[-1, :]) + cros(sm, wm[-1, :])
        rotm = 0.5 * cros(wmm, vmm)
        dvbm = vmm + rotm + scullm
    else:
        dvbm = np.zeros(3)
        
    return phim, dvbm

def imudot(imu, passband):
    '''
    Compute the angular acceleration & jerk from SIMU data.
    '''
    ts = imu[1, -1] - imu[0, -1]
    b = signal.firwin(11, passband * ts, pass_zero='lowpass')  # order 10
    a = 1.0
    
    dotwf_diff = np.vstack((imu[0:1, 0:6], imu[:, 0:6]))
    dotwf_diff = np.diff(dotwf_diff, axis=0)
    
    padlen = min(3 * len(b), dotwf_diff.shape[0] - 1)
    if padlen > 0:
        dotwf_filt = signal.filtfilt(b, a, dotwf_diff, axis=0, padlen=padlen) / (ts**2)
    else:
        dotwf_filt = dotwf_diff / (ts**2)
    
    return np.column_stack((dotwf_filt, imu[:, -1]))

def imulvS(wb, dotwb, Cba=None):
    '''
    Inner lever arm calculation matrix
    '''
    if Cba is None:
        Cba = np.eye(3)
    U = np.linalg.inv(Cba.T)
    V1, V2, V3 = Cba[:, 0], Cba[:, 1], Cba[:, 2]
    
    Q11, Q12, Q13 = U[0,0]*V1, U[0,1]*V2, U[0,2]*V3
    Q21, Q22, Q23 = U[1,0]*V1, U[1,1]*V2, U[1,2]*V3
    Q31, Q32, Q33 = U[2,0]*V1, U[2,1]*V2, U[2,2]*V3
    
    W = askew(dotwb) + askew(wb) @ askew(wb)
    
    SS = np.vstack((
        np.hstack((Q11[:, None] * W, Q12[:, None] * W, Q13[:, None] * W)),
        np.hstack((Q21[:, None] * W, Q22[:, None] * W, Q23[:, None] * W)),
        np.hstack((Q31[:, None] * W, Q32[:, None] * W, Q33[:, None] * W))
    ))
    # Note: numpy broadcasting above creates 3x3 per Qij*W slice. Actually, since U(i,j) is a scalar and Vj is 3x1 array...
    # Let me re-implement Qij correctly
    SS = np.zeros((3, 9))
    Q11, Q12, Q13 = U[0,0]*V1, U[0,1]*V2, U[0,2]*V3
    Q21, Q22, Q23 = U[1,0]*V1, U[1,1]*V2, U[1,2]*V3
    Q31, Q32, Q33 = U[2,0]*V1, U[2,1]*V2, U[2,2]*V3
    
    # Each Q is 3x1. W is 3x3. Q * W where * is matrix multiplication? 
    # In MATLAB: Q11*W means scalar*vector*matrix... Wait. U(1,1) is scalar. V1 is row vector (1x3). So Q11 is 1x3.
    # W is 3x3. So Q11*W is 1x3 matrix mult 3x3 -> 1x3!
    # Ah! V1 = Cba(:,1)' in matlab is row!
    V1 = Cba[:, 0]
    V2 = Cba[:, 1]
    V3 = Cba[:, 2]
    
    Q11 = U[0,0] * V1.T @ W
    Q12 = U[0,1] * V2.T @ W
    Q13 = U[0,2] * V3.T @ W
    Q21 = U[1,0] * V1.T @ W
    Q22 = U[1,1] * V2.T @ W
    Q23 = U[1,2] * V3.T @ W
    Q31 = U[2,0] * V1.T @ W
    Q32 = U[2,1] * V2.T @ W
    Q33 = U[2,2] * V3.T @ W
    
    SS[0, :] = np.concatenate((Q11, Q12, Q13))
    SS[1, :] = np.concatenate((Q21, Q22, Q23))
    SS[2, :] = np.concatenate((Q31, Q32, Q33))
    return SS

def imuadderr(imu, imuerr):
    '''
    SIMU adding errors simulation
    '''
    imu = np.copy(imu)
    ts = imu[1, -1] - imu[0, -1]
    
    if 'rx' in imuerr:
        wb = imu[:, 0:3] / ts
        dotwf = imudot(imu, 5.0)
        
        rx, ry, rz = imuerr['rx'], imuerr['ry'], imuerr['rz']
        fL = np.zeros((imu.shape[0], 3))
        fL[:, 0] = (-wb[:, 1]**2 - wb[:, 2]**2)*rx[0] + (wb[:, 0]*wb[:, 1] - dotwf[:, 2])*rx[1] + (wb[:, 0]*wb[:, 2] + dotwf[:, 1])*rx[2]
        fL[:, 1] = (wb[:, 0]*wb[:, 1] + dotwf[:, 2])*ry[0] + (-wb[:, 0]**2 - wb[:, 2]**2)*ry[1] + (wb[:, 1]*wb[:, 2] - dotwf[:, 0])*ry[2]
        fL[:, 2] = (wb[:, 0]*wb[:, 2] - dotwf[:, 1])*rz[0] + (wb[:, 1]*wb[:, 2] + dotwf[:, 0])*rz[1] + (-wb[:, 0]**2 - wb[:, 1]**2)*rz[2]
        
        imu[:, 3:6] += fL * ts
        
    m, _ = imu.shape
    sts = math.sqrt(ts)
    
    eb = imuerr.get('eb', np.zeros(3))
    db = imuerr.get('db', np.zeros(3))
    web = imuerr.get('web', np.zeros(3))
    wdb = imuerr.get('wdb', np.zeros(3))
    
    drift = np.zeros((m, 6))
    for i in range(3):
        drift[:, i] = ts*eb[i] + sts*web[i]*np.random.randn(m)
        drift[:, i+3] = ts*db[i] + sts*wdb[i]*np.random.randn(m)
        
    if 'Ka2' in imuerr:
        Ka2 = imuerr['Ka2']
        imu[:, 3] += Ka2[0] / ts * (imu[:, 3]**2)
        imu[:, 4] += Ka2[1] / ts * (imu[:, 4]**2)
        imu[:, 5] += Ka2[2] / ts * (imu[:, 5]**2)
        
    if 'dKg' in imuerr:
        Kg = np.eye(3) + imuerr['dKg']
        Ka = np.eye(3) + imuerr['dKa']
        imu[:, 0:3] = imu[:, 0:3] @ Kg.T
        imu[:, 3:6] = imu[:, 3:6] @ Ka.T
        
    imu[:, 0:6] += drift
    return imu

def dict_get_or_default(d, key, val):
    return d.get(key, val)

def imuclbt(imu, clbt=None):
    '''
    IMU calibration like imu_del_err.
    '''
    imu = np.copy(imu)
    ts = imu[1, -1] - imu[0, -1]
    
    if clbt is None:
        clbt = {
            'sf': np.ones(6),
            'Kg': np.eye(3),
            'eb': np.zeros(3),
            'Ka': np.eye(3),
            'db': np.zeros(3),
            'Ka2': np.zeros(3),
            'rx': np.zeros(3), 'ry': np.zeros(3), 'rz': np.zeros(3),
            'tGA': 0.0
        }
        
    sf = dict_get_or_default(clbt, 'sf', np.ones(6))
    Kg = dict_get_or_default(clbt, 'Kg', np.eye(3))
    Ka = dict_get_or_default(clbt, 'Ka', np.eye(3))
    eb = dict_get_or_default(clbt, 'eb', np.zeros(3))
    db = dict_get_or_default(clbt, 'db', np.zeros(3))
    Ka2 = dict_get_or_default(clbt, 'Ka2', np.zeros(3))
    rx = dict_get_or_default(clbt, 'rx', np.zeros(3))
    ry = dict_get_or_default(clbt, 'ry', np.zeros(3))
    rz = dict_get_or_default(clbt, 'rz', np.zeros(3))
    tGA = dict_get_or_default(clbt, 'tGA', 0.0)
    
    for k in range(6):
        imu[:, k] *= sf[k]
        
    imuerr = {
        'dKg': Kg - np.eye(3),
        'dKa': Ka - np.eye(3),
        'eb': -eb,
        'db': -db,
        'Ka2': -Ka2
    }
    
    imu = imuadderr(imu, imuerr)
    
    if np.linalg.norm(rx) > 0 or np.linalg.norm(ry) > 0 or np.linalg.norm(rz) > 0 or abs(tGA) > 0:
        for k in range(1, imu.shape[0]):
            wb = imu[k, 0:3] / ts
            fb = imu[k, 3:6] / ts
            dwb = (imu[k, 0:3] - imu[k-1, 0:3]) / (ts * ts)
            
            SS = imulvS(wb, dwb)
            fL = SS @ np.concatenate((rx, ry, rz))
            
            fb = fb - fL - tGA * cros(wb, fb)
            imu[k, 3:6] = fb * ts
            
    return imu
