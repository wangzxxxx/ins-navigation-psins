import numpy as np
from .glv_utils import glv
from .math_utils import rv2m, a2qua, q2mat, askew, qupdt2, qdelphi, q2att, cnscl
from .nav_utils import Earth

def alignvn_dar_12state(imu, qnb_in, pos, phi0, imuerr, wvn, max_iter=2, isfig=0):
    imu = np.copy(imu)
    phi0 = np.asarray(phi0).flatten()
    wvn = np.asarray(wvn).flatten()
    if len(wvn) == 1:
        wvn = np.array([wvn[0], wvn[0], wvn[0]])
        
    ts = imu[1, -1] - imu[0, -1]
    nn = 2
    nts = nn * ts
    
    if len(qnb_in) == 1:
        qnb0 = a2qua(np.zeros(3)) # Simplified for aligni0, normally would do coarse align
    elif len(qnb_in) == 3:
        qnb0 = a2qua(qnb_in)
    else:
        qnb0 = qnb_in.copy()
        
    len_imu = (imu.shape[0] // nn) * nn
    eth = Earth(pos)
    Cnn = rv2m(-eth.wnie * nts / 2)
    
    web = imuerr.get('web', np.ones(3)*0.001*glv.dpsh).flatten()
    wdb = imuerr.get('wdb', np.ones(3)*1.0*glv.ugpsHz).flatten()
    eb = imuerr.get('eb', np.ones(3)*0.01*glv.dph).flatten()
    db = imuerr.get('db', np.ones(3)*100*glv.ug).flatten()
    
    Qk = np.diag(np.concatenate((web, wdb, np.zeros(6))))**2 * nts
    Rk = np.diag(wvn)**2 / nts
    
    init_eb_P = np.maximum(eb, 0.1 * glv.dph)
    init_db_P = np.maximum(db, 1000 * glv.ug)
    
    Ft = np.zeros((12, 12))
    Ft[0:3, 0:3] = askew(-eth.wnie)
    Phikk_1_base = np.eye(12) + Ft * nts
    Hk = np.zeros((3, 12))
    Hk[0:3, 3:6] = np.eye(3)
    
    att0 = None
    attk = None
    xkpk = None
    
    print(f"Starting 12-state iterative alignment (Total Iter: {max_iter})...")
    
    for iter_idx in range(1, max_iter + 1):
        print(f"  -> Iteration {iter_idx} / {max_iter} ... ", end="")
        
        Pxk = np.diag(np.concatenate((phi0, [1, 1, 1], init_eb_P, init_db_P)))**2
        xk = np.zeros(12)
        
        vn = np.zeros(3)
        qnb = qnb0.copy()
        
        attk_list = []
        xkpk_list = []
        
        for k in range(0, len_imu - nn + 1, nn):
            wvm = imu[k:k+nn, 0:6]
            t = imu[k+nn-1, -1]
            phim, dvbm = cnscl(wvm)
            
            Cnb = q2mat(qnb)
            dvn = Cnn @ Cnb @ dvbm
            vn = vn + dvn + eth.gn * nts
            qnb = qupdt2(qnb, phim, eth.wnin * nts)
            
            Cnbts = Cnb * nts
            Phikk_1 = Phikk_1_base.copy()
            Phikk_1[3:6, 0:3] = askew(dvn)
            Phikk_1[3:6, 9:12] = Cnbts
            Phikk_1[0:3, 6:9] = -Cnbts
            
            # KF Time Update
            xk = Phikk_1 @ xk
            Pxk = Phikk_1 @ Pxk @ Phikk_1.T + Qk
            Pxk = (Pxk + Pxk.T) / 2
            
            # KF Measurement Update (Zk = vn - 0 = vn)
            Zk = vn
            Innov = Zk - Hk @ xk
            S = Hk @ Pxk @ Hk.T + Rk
            Kk = Pxk @ Hk.T @ np.linalg.inv(S)
            
            xk = xk + Kk @ Innov
            Pxk = (np.eye(12) - Kk @ Hk) @ Pxk
            Pxk = (Pxk + Pxk.T) / 2
            
            # Feedback
            qnb = qdelphi(qnb, 0.91 * xk[0:3])
            xk[0:3] = 0.09 * xk[0:3]
            
            vn = vn - 0.91 * xk[3:6]
            xk[3:6] = 0.09 * xk[3:6]
            
            attk_list.append(np.concatenate((q2att(qnb), vn, [t])))
            xkpk_list.append(np.concatenate((xk, np.diag(Pxk), [t])))
            
        attk = np.array(attk_list)
        xkpk = np.array(xkpk_list)
        att0 = attk[-1, 0:3]
        
        if iter_idx < max_iter:
            est_eb = xk[6:9]
            est_db = xk[9:12]
            print(f"Correction Bias -> eb: [{np.linalg.norm(est_eb)/glv.dph:.4f}] dph, db: [{np.linalg.norm(est_db)/glv.ug:.1f}] ug")
            
            imu[:, 0:3] = imu[:, 0:3] - est_eb * ts
            imu[:, 3:6] = imu[:, 3:6] - est_db * ts
        else:
            print()
            
    if isfig:
        import matplotlib.pyplot as plt
        t = attk[:, -1]
        plt.figure(figsize=(12, 8))
        plt.subplot(221)
        plt.plot(t, attk[:, 0:2] / glv.deg)
        plt.title('Attitude Error (Pitch/Roll)')
        plt.subplot(222)
        plt.plot(t, attk[:, 2] / glv.deg)
        plt.title('Heading')
        plt.subplot(223)
        plt.plot(t, xkpk[:, 6:9] / glv.dph)
        plt.title('Gyro Bias Est')
        plt.subplot(224)
        plt.plot(t, xkpk[:, 9:12] / glv.ug)
        plt.title('Acc Bias Est')
        plt.tight_layout()
        plt.show()
        
    return att0, attk, xkpk


def alignvn_dar_12state_multiiter(imu, qnb_in, pos, phi0, imuerr, wvn, max_iter=2, isfig=0):
    """
    Same as alignvn_dar_12state but returns a list of xkpk arrays (one per iteration)
    for detailed multi-iteration convergence analysis.
    """
    imu = np.copy(imu)
    phi0 = np.asarray(phi0).flatten()
    wvn = np.asarray(wvn).flatten()
    if len(wvn) == 1:
        wvn = np.array([wvn[0], wvn[0], wvn[0]])

    ts = imu[1, -1] - imu[0, -1]
    nn = 2
    nts = nn * ts

    if len(qnb_in) == 1:
        qnb0 = a2qua(np.zeros(3))
    elif len(qnb_in) == 3:
        qnb0 = a2qua(qnb_in)
    else:
        qnb0 = qnb_in.copy()

    len_imu = (imu.shape[0] // nn) * nn
    eth = Earth(pos)
    Cnn = rv2m(-eth.wnie * nts / 2)

    web = imuerr.get('web', np.ones(3)*0.001*glv.dpsh).flatten()
    wdb = imuerr.get('wdb', np.ones(3)*1.0*glv.ugpsHz).flatten()
    eb = imuerr.get('eb', np.ones(3)*0.01*glv.dph).flatten()
    db = imuerr.get('db', np.ones(3)*100*glv.ug).flatten()

    Qk = np.diag(np.concatenate((web, wdb, np.zeros(6))))**2 * nts
    Rk = np.diag(wvn)**2 / nts

    init_eb_P = np.maximum(eb, 0.1 * glv.dph)
    init_db_P = np.maximum(db, 1000 * glv.ug)

    Ft = np.zeros((12, 12))
    Ft[0:3, 0:3] = askew(-eth.wnie)
    Phikk_1_base = np.eye(12) + Ft * nts
    Hk = np.zeros((3, 12))
    Hk[0:3, 3:6] = np.eye(3)

    att0 = None
    xkpk_all_iters = []

    print(f"Starting multi-iter 12-state alignment (Total Iter: {max_iter})...")

    for iter_idx in range(1, max_iter + 1):
        print(f"  -> Iteration {iter_idx} / {max_iter} ... ", end="")

        Pxk = np.diag(np.concatenate((phi0, [1, 1, 1], init_eb_P, init_db_P)))**2
        xk = np.zeros(12)
        vn = np.zeros(3)
        qnb = qnb0.copy()
        xkpk_list = []

        for k in range(0, len_imu - nn + 1, nn):
            wvm = imu[k:k+nn, 0:6]
            t = imu[k+nn-1, -1]
            phim, dvbm = cnscl(wvm)

            Cnb = q2mat(qnb)
            dvn = Cnn @ Cnb @ dvbm
            vn = vn + dvn + eth.gn * nts
            qnb = qupdt2(qnb, phim, eth.wnin * nts)

            Cnbts = Cnb * nts
            Phikk_1 = Phikk_1_base.copy()
            Phikk_1[3:6, 0:3] = askew(dvn)
            Phikk_1[3:6, 9:12] = Cnbts
            Phikk_1[0:3, 6:9] = -Cnbts

            xk = Phikk_1 @ xk
            Pxk = Phikk_1 @ Pxk @ Phikk_1.T + Qk
            Pxk = (Pxk + Pxk.T) / 2

            Zk = vn
            Innov = Zk - Hk @ xk
            S = Hk @ Pxk @ Hk.T + Rk
            Kk = Pxk @ Hk.T @ np.linalg.inv(S)

            xk = xk + Kk @ Innov
            Pxk = (np.eye(12) - Kk @ Hk) @ Pxk
            Pxk = (Pxk + Pxk.T) / 2

            qnb = qdelphi(qnb, 0.91 * xk[0:3])
            xk[0:3] = 0.09 * xk[0:3]
            vn = vn - 0.91 * xk[3:6]
            xk[3:6] = 0.09 * xk[3:6]

            xkpk_list.append(np.concatenate((xk, np.diag(Pxk), [t])))

        xkpk_arr = np.array(xkpk_list)
        xkpk_all_iters.append(xkpk_arr)
        att0 = q2att(qnb)

        if iter_idx < max_iter:
            est_eb = xk[6:9]
            est_db = xk[9:12]
            print(f"eb: [{np.linalg.norm(est_eb)/glv.dph:.4f}] dph, db: [{np.linalg.norm(est_db)/glv.ug:.1f}] ug")
            imu[:, 0:3] = imu[:, 0:3] - est_eb * ts
            imu[:, 3:6] = imu[:, 3:6] - est_db * ts
        else:
            print()

    return att0, xkpk_all_iters
