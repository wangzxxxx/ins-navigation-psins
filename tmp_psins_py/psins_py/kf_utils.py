import numpy as np
from psins_py.math_utils import askew, a2qua, q2mat, a2mat, m2att
from psins_py.nav_utils import glv, Earth

def nnts(nn, ts):
    return nn, ts, nn*ts, 1

def alignsb(imu, pos):
    ts = imu[1, -1] - imu[0, -1]
    wbib = np.mean(imu[:, 0:3], axis=0) / ts
    fbsf = np.mean(imu[:, 3:6], axis=0) / ts
    
    eth = Earth(pos)
    gn = eth.gn
    wnie = eth.wnie
    
    # Coarse alignment dv2atti
    vn1 = gn
    vn2 = wnie
    vb1 = -fbsf
    vb2 = wbib
    
    vntmp1 = np.cross(vn1, vn2)
    vntmp2 = np.cross(vntmp1, vn1)
    
    vbtmp1 = np.cross(vb1, vb2)
    vbtmp2 = np.cross(vbtmp1, vb1)
    
    Vn = np.vstack([vn1/np.linalg.norm(vn1), vntmp1/np.linalg.norm(vntmp1), vntmp2/np.linalg.norm(vntmp2)]).T
    Vb = np.vstack([vb1/np.linalg.norm(vb1), vbtmp1/np.linalg.norm(vbtmp1), vbtmp2/np.linalg.norm(vbtmp2)]).T
    
    Cnb = Vn @ Vb.T
    att = m2att(Cnb)
    qnb = a2qua(att)
    
    return att, None, None, qnb

def lvS(Cba, wb, dotwb):
    U = np.linalg.inv(Cba.T)
    V1, V2, V3 = Cba[:,0], Cba[:,1], Cba[:,2]
    Q11, Q12, Q13 = U[0,0]*V1, U[0,1]*V2, U[0,2]*V3
    Q21, Q22, Q23 = U[1,0]*V1, U[1,1]*V2, U[1,2]*V3
    Q31, Q32, Q33 = U[2,0]*V1, U[2,1]*V2, U[2,2]*V3
    
    W = askew(dotwb) + askew(wb) @ askew(wb)
    
    return np.vstack([
        np.hstack([Q11@W, Q12@W, Q13@W]),
        np.hstack([Q21@W, Q22@W, Q23@W]),
        np.hstack([Q31@W, Q32@W, Q33@W])
    ])

def getFt(fb, wb, Cnb, wnie, SS):
    o33 = np.zeros((3, 3))
    o31 = np.zeros((3, 1))
    
    wX = askew(wnie)
    fX = askew(Cnb @ fb)
    
    wx, wy, wz = wb[0], wb[1], wb[2]
    fx, fy, fz = fb[0], fb[1], fb[2]
    
    CDf2 = Cnb @ np.diag(fb**2)
    CwXf = Cnb @ np.cross(wb, fb)
    CwXf = CwXf.reshape(3, 1)
    
    row1 = np.hstack([-wX, o33, -Cnb, o33, -wx*Cnb, -wy*Cnb, -wz*Cnb, o33, o33, o33, o33, o33, o33, o33, o31])
    row2 = np.hstack([fX, o33, o33, Cnb, o33, o33, o33, fx*Cnb, fy*Cnb, fz*Cnb, CDf2, Cnb@SS, CwXf])
    row3 = np.zeros((37, 43))
    
    return np.vstack([row1, row2, row3])

def clbtkfinit(ts):
    kf = {}
    kf['nts'] = ts
    kf['n'] = 43
    kf['m'] = 3
    kf['I'] = np.eye(kf['n'])
    
    qvec = np.zeros(43)
    qvec[0:3] = 0.01 * glv.dpsh
    qvec[3:6] = 100 * glv.ugpsHz
    kf['Qk'] = np.diag(qvec)**2 * ts
    
    kf['Rk'] = np.diag([0.001, 0.001, 0.001])**2
    
    pvec = np.zeros(43)
    pvec[0:3] = np.array([0.1, 0.1, 1.0]) * glv.deg
    pvec[3:6] = 1.0
    pvec[6:9] = 0.1 * glv.dph
    pvec[9:12] = 1.0 * glv.mg
    pvec[12:15] = [100*glv.ppm, 100*glv.sec, 100*glv.sec]
    pvec[15:18] = [100*glv.sec, 100*glv.ppm, 100*glv.sec]
    pvec[18:21] = [100*glv.sec, 100*glv.sec, 100*glv.ppm]
    pvec[21:24] = [100*glv.ppm, 100*glv.sec, 100*glv.sec]
    pvec[24:27] = [0, 100*glv.ppm, 100*glv.sec]
    pvec[27:30] = [0, 0, 100*glv.ppm]
    pvec[30:33] = 100 * glv.ugpg2
    pvec[33:36] = 0.1
    pvec[36:39] = 0.1
    pvec[39:42] = 0.0
    pvec[42] = 0.01
    kf['Pxk'] = np.diag(pvec)**2
    
    Hk = np.zeros((3, 43))
    Hk[:, 3:6] = np.eye(3)
    kf['Hk'] = Hk
    
    kf['xk'] = np.zeros(43)
    
    return kf

def kfupdate(kf, yk=None, TimeMeasBoth=None):
    if TimeMeasBoth is None:
        if yk is None:
            TimeMeasBoth = 'T'
        else:
            TimeMeasBoth = 'B'
            
    if TimeMeasBoth == 'T':
        kf['xk'] = kf['Phikk_1'] @ kf['xk']
        kf['Pxk'] = kf['Phikk_1'] @ kf['Pxk'] @ kf['Phikk_1'].T + kf['Qk']
        return kf
        
    if TimeMeasBoth == 'B':
        kf['xk'] = kf['Phikk_1'] @ kf['xk']
        kf['Pxk'] = kf['Phikk_1'] @ kf['Pxk'] @ kf['Phikk_1'].T + kf['Qk']
        
    Pxykk_1 = kf['Pxk'] @ kf['Hk'].T
    Py0 = kf['Hk'] @ Pxykk_1
    ykk_1 = kf['Hk'] @ kf['xk']
    kf['rk'] = yk - ykk_1
    
    Pykk_1 = Py0 + kf['Rk']
    
    try:
        Kk = Pxykk_1 @ np.linalg.inv(Pykk_1)
    except np.linalg.LinAlgError:
        Kk = Pxykk_1 @ np.linalg.pinv(Pykk_1)
        
    kf['Kk'] = Kk
    kf['xk'] = kf['xk'] + Kk @ kf['rk']
    kf['Pxk'] = kf['Pxk'] - Kk @ Pykk_1 @ Kk.T
    kf['Pxk'] = (kf['Pxk'] + kf['Pxk'].T) * 0.5
    
    return kf

def clbtkffeedback(kf, clbt):
    clbt['Kg'] = (np.eye(3) - kf['xk'][12:21].reshape(3,3).T) @ clbt['Kg']
    clbt['Ka'] = (np.eye(3) - kf['xk'][21:30].reshape(3,3).T) @ clbt['Ka']
    clbt['Ka2'] = clbt['Ka2'] + kf['xk'][30:33]
    clbt['eb'] = clbt['eb'] + kf['xk'][6:9]
    clbt['db'] = clbt['db'] + kf['xk'][9:12]
    clbt['rx'] = clbt['rx'] + kf['xk'][33:36]
    clbt['ry'] = clbt['ry'] + kf['xk'][36:39]
    clbt['rz'] = clbt['rz'] + kf['xk'][39:42]
    clbt['tGA'] = clbt['tGA'] + kf['xk'][42]
    return clbt

