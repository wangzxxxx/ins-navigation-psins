import numpy as np

def askew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ], dtype=float)

def q2mat(q):
    q11, q12, q13, q14 = q[0]*q[0], q[0]*q[1], q[0]*q[2], q[0]*q[3]
    q22, q23, q24 = q[1]*q[1], q[1]*q[2], q[1]*q[3]
    q33, q34 = q[2]*q[2], q[2]*q[3]
    q44 = q[3]*q[3]
    return np.array([
        [q11+q22-q33-q44, 2*(q23-q14), 2*(q24+q13)],
        [2*(q23+q14), q11-q22+q33-q44, 2*(q34-q12)],
        [2*(q24-q13), 2*(q34+q12), q11-q22-q33+q44]
    ])

def q2att(q):
    return m2att(q2mat(q))

def m2att(C):
    pitch = np.arcsin(np.clip(C[2,1], -1.0, 1.0))
    roll = np.arctan2(-C[2,0], C[2,2])
    yaw = np.arctan2(-C[0,1], C[1,1])
    return np.array([pitch, roll, yaw])

def a2mat(att):
    si, sj, sk = np.sin(att)
    ci, cj, ck = np.cos(att)
    return np.array([
        [cj*ck-si*sj*sk, -ci*sk, sj*ck+si*cj*sk],
        [cj*sk+si*sj*ck, ci*ck, sj*sk-si*cj*ck],
        [-ci*sj, si, ci*cj]
    ])

def a2qua(att):
    pitch, roll, yaw = att[0]/2, att[1]/2, att[2]/2
    sp, sr, sy = np.sin(pitch), np.sin(roll), np.sin(yaw)
    cp, cr, cy = np.cos(pitch), np.cos(roll), np.cos(yaw)
    return np.array([
        cp*cr*cy - sp*sr*sy,
        sp*cr*cy - cp*sr*sy,
        cp*sr*cy + sp*cr*sy,
        cp*cr*sy + sp*sr*cy
    ])

def qmul(q1, q2):
    return np.array([
        q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3],
        q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2],
        q1[0]*q2[2] + q1[2]*q2[0] + q1[3]*q2[1] - q1[1]*q2[3],
        q1[0]*q2[3] + q1[3]*q2[0] + q1[1]*q2[2] - q1[2]*q2[1]
    ])

def rv2q(rv):
    n2 = rv[0]**2 + rv[1]**2 + rv[2]**2
    if n2 < 1.0e-8:
        q0 = 1 - n2*(1/8 - n2/384)
        s = 1/2 - n2*(1/48 - n2/3840)
    else:
        n = np.sqrt(n2)
        n_2 = n/2
        q0 = np.cos(n_2)
        s = np.sin(n_2) / n
    return np.array([q0, s*rv[0], s*rv[1], s*rv[2]])

def qupdt2(q, rv_ib, rv_in):
    q = qmul(q, rv2q(rv_ib))
    if np.linalg.norm(rv_in) > 1e-10:
        q = qmul(rv2q(-rv_in), q)
    nn = np.linalg.norm(q)
    return q / nn

def rotv(rv_qua_mat, v):
    if isinstance(rv_qua_mat, np.ndarray) and rv_qua_mat.shape == (3, 3):
        C = rv_qua_mat
    elif len(rv_qua_mat) == 3:
        C = rv2m(rv_qua_mat)
    elif len(rv_qua_mat) == 4:
        C = q2mat(rv_qua_mat)
    else:
        C = rv_qua_mat
    return C @ v

def qdelphi(qpb, phi):
    """ qnb = qdelphi(qpb, phi) => qnb = qmul(rv2q(phi), qpb) """
    return qmul(rv2q(phi), qpb)

def qaddphi(qnb, phi):
    """ qpb = qaddphi(qnb, phi) => qpb = qmul(rv2q(-phi), qnb) """
    return qmul(rv2q(-phi), qnb)

def qq2phi(qpb, qnb):
    """ phi = qq2phi(qpb, qnb) => phi = q2rv(qmul(qnb, qconj(qpb))) 
        Wait, PSINS qq2phi.m : 
        qnp = qmul(qnb, qconj(qpb));
        phi = q2rv(qnp);
    """
    qpb_conj = np.array([qpb[0], -qpb[1], -qpb[2], -qpb[3]])
    dq = qmul(qnb, qpb_conj)
    return q2rv(dq)

def rv2m(rv):
    xx, yy, zz = rv[0]*rv[0], rv[1]*rv[1], rv[2]*rv[2]
    n2 = xx + yy + zz
    if n2 < 1.e-8:
        a = 1 - n2*(1/6 - n2/120)
        b = 0.5 - n2*(1/24 - n2/720)
    else:
        n = np.sqrt(n2)
        a = np.sin(n)/n
        b = (1 - np.cos(n))/n2
    arvx, arvy, arvz = a*rv[0], a*rv[1], a*rv[2]
    bxx, bxy, bxz = b*xx, b*rv[0]*rv[1], b*rv[0]*rv[2]
    byy, byz, bzz = b*yy, b*rv[1]*rv[2], b*zz
    return np.array([
        [1 - byy - bzz, -arvz + bxy, arvy + bxz],
        [arvz + bxy, 1 - bxx - bzz, -arvx + byz],
        [-arvy + bxz, arvx + byz, 1 - bxx - byy]
    ])

def q2rv(q):
    n2 = q[1]*q[1] + q[2]*q[2] + q[3]*q[3]
    if n2 < 1e-8:
        return 2 * np.array([q[1], q[2], q[3]])
    n = np.sqrt(n2)
    s = 2 * np.arccos(np.clip(q[0], -1.0, 1.0)) / n
    return s * np.array([q[1], q[2], q[3]])

def m2qua(C):
    q = np.zeros(4)
    q[0] = 0.5 * np.sqrt(max(0.0, 1.0 + C[0,0] + C[1,1] + C[2,2]))
    if q[0] > 1e-6:
        q[1] = (C[2,1] - C[1,2]) / (4 * q[0])
        q[2] = (C[0,2] - C[2,0]) / (4 * q[0])
        q[3] = (C[1,0] - C[0,1]) / (4 * q[0])
    else:
        q[1] = np.sqrt(max(0.0, -C[1,1] - C[2,2] + 1.0)) / 2
        q[2] = np.sqrt(max(0.0, -C[0,0] - C[2,2] + 1.0)) / 2
        q[3] = np.sqrt(max(0.0, -C[0,0] - C[1,1] + 1.0)) / 2
    return q / np.linalg.norm(q)

def m2rv(m):
    rv = np.array([m[2,1]-m[1,2], m[0,2]-m[2,0], m[1,0]-m[0,1]])
    phi = np.arccos(np.clip((m[0,0]+m[1,1]+m[2,2]-1)/2, -1.0, 1.0))
    if phi > np.pi - 1e-3:
        return q2rv(m2qua(m))
    elif phi < 1e-10:
        return 0.5 * rv
    else:
        return rv * phi / (2 * np.sin(phi))

def rotv(rv_qua_mat, v):
    if isinstance(rv_qua_mat, np.ndarray) and rv_qua_mat.shape == (3, 3):
        C = rv_qua_mat
    elif len(rv_qua_mat) == 3:
        C = rv2m(rv_qua_mat)
    elif len(rv_qua_mat) == 4:
        C = q2mat(rv_qua_mat)
    else:
        C = rv_qua_mat
    return C @ v

def qmulv(q, v):
    return rotv(q, v)

def normv(v):
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def cros(a, b):
    return np.cross(a, b)

def cnscl(imu):
    """ Simplified single sample coning & sculling algorithm """
    wm = imu[:, :3] if imu.ndim > 1 else imu[:3]
    vm = imu[:, 3:6] if imu.ndim > 1 else imu[3:6]
    
    n = imu.shape[0] if imu.ndim > 1 else 1
    if n == 1:
        phim = np.copy(wm) if imu.ndim > 1 else wm
        dvbm = np.copy(vm) if imu.ndim > 1 else vm
        return phim, dvbm
        
    wmm = np.sum(wm, axis=0)
    cm = np.zeros(3)
    if n > 1:
        cm = wm[0] * 0  # not optimal, but for nn=2
    dphim = cros(cm, wm[-1])
    phim = wmm + dphim
    
    vmm = np.sum(vm, axis=0)
    sm = np.zeros(3)
    scullm = cros(cm, vm[-1]) + cros(sm, wm[-1])
    rotm = 0.5 * cros(wmm, vmm)
    dvbm = vmm + rotm + scullm
    return phim, dvbm
