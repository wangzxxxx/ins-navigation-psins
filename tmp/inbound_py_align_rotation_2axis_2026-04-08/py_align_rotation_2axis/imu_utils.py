import numpy as np
import math
from scipy import signal
from .glv_utils import glv
from .math_utils import a2mat, m2att, rv2m, m2rv, rv2q, askew, cros, qupdt2, qmulv

def ar1coefs(ts, tau):
    tau = np.asarray(tau).flatten()
    a2 = np.exp(-ts / tau)
    b = 1 - a2
    a = np.column_stack((np.ones_like(a2), -a2))
    return b, a

def attrottt(att0, rotparas, ts):
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
        
        padlen = min(3 * max(len(a[0]), len(b)), len(angles) - 1)
        if padlen > 0:
            angles = signal.filtfilt(b, a[0], angles, padlen=padlen)
            
        len_angles = len(angles)
        atti = np.zeros((len_angles, 3))
        Cnb0 = a2mat(att[-1])
        
        for kk in range(len_angles):
            atti[kk, :] = m2att(Cnb0 @ rv2m(U * angles[kk]))
            
        att.extend(atti)
        
    att = np.array(att)
    time_col = np.arange(1, att.shape[0] + 1) * ts
    return np.column_stack((att, time_col))

from .nav_utils import Earth

def avp2imu(avp, pos0=None):
    avp = np.atleast_2d(avp)
    if pos0 is not None:
        pos0 = np.asarray(pos0).flatten()
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

def setdiag(mat, val):
    m = np.copy(mat)
    np.fill_diagonal(m, val)
    return m

def imuerrset(eb=0.01, db=100.0, web=0.001, wdb=1.0):
    eb_arr = np.array([eb]*3) if np.isscalar(eb) else np.asarray(eb)
    db_arr = np.array([db]*3) if np.isscalar(db) else np.asarray(db)
    web_arr = np.array([web]*3) if np.isscalar(web) else np.asarray(web)
    wdb_arr = np.array([wdb]*3) if np.isscalar(wdb) else np.asarray(wdb)
    return {
        'eb': eb_arr * glv.dph,
        'db': db_arr * glv.ug,
        'web': web_arr * glv.dpsh,
        'wdb': wdb_arr * glv.ugpsHz,
        'dKg': np.zeros((3, 3)),
        'dKa': np.zeros((3, 3)),
        'Ka2': np.zeros(3)
    }

def imuadderr(imu, imuerr):
    imu = np.copy(imu)
    ts = imu[1, -1] - imu[0, -1]
    m_len = imu.shape[0]
    sts = math.sqrt(ts)
    
    eb = imuerr.get('eb', np.zeros(3))
    db = imuerr.get('db', np.zeros(3))
    web = imuerr.get('web', np.zeros(3))
    wdb = imuerr.get('wdb', np.zeros(3))
    
    drift = np.zeros((m_len, 6))
    for i in range(3):
        drift[:, i] = ts*eb[i] + sts*web[i]*np.random.randn(m_len)
        drift[:, i+3] = ts*db[i] + sts*wdb[i]*np.random.randn(m_len)
        
    if 'Ka2' in imuerr:
        Ka2 = imuerr['Ka2']
        if np.any(Ka2):
            imu[:, 3] += Ka2[0] / ts * (imu[:, 3]**2)
            imu[:, 4] += Ka2[1] / ts * (imu[:, 4]**2)
            imu[:, 5] += Ka2[2] / ts * (imu[:, 5]**2)
            
    if 'dKg' in imuerr:
        Kg = np.eye(3) + imuerr['dKg']
        imu[:, 0:3] = imu[:, 0:3] @ Kg.T
    if 'dKa' in imuerr:    
        Ka = np.eye(3) + imuerr['dKa']
        imu[:, 3:6] = imu[:, 3:6] @ Ka.T
        
    imu[:, 0:6] += drift
    return imu

def imuplot(imu):
    import matplotlib.pyplot as plt
    t = imu[:, -1]
    plt.figure(figsize=(10, 6))
    plt.subplot(211)
    plt.plot(t, imu[:, 0:3] / glv.dps)
    plt.title('IMU output')
    plt.ylabel('Gyro (dps)')
    plt.subplot(212)
    plt.plot(t, imu[:, 3:6] / glv.g0)
    plt.ylabel('Acc (g)')
    plt.xlabel('Time (s)')
    plt.show()
