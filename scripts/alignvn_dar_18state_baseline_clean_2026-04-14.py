#!/usr/bin/env python3
"""Clean dual-axis 18-state DAR alignment baseline.

State layout:
    x = [phi(3), dv(3), eb(3), db(3), ng(3), xa(3)]

Notes:
- This is the "baseline18" / "18-state dual-axis alignment" clean script.
- It keeps the same dual-axis DAR path and the same injected truth-side
  diagonal dKg/dKa = 30 ppm mismatch used in the earlier baseline experiments.
- kg/ka are NOT estimated here. They are only injected into IMU truth via imuerr.
- The raw attitude-error order returned by qq2phi(...) in this script is:
      [roll, pitch, yaw]
  This matches the historical baseline code path where the 2nd component is the
  one discussed as the stable ~35 arcsec pitch bias.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
PSINS_ROOT = WORKSPACE / 'tmp_psins_py'
if str(PSINS_ROOT) not in sys.path:
    sys.path.insert(0, str(PSINS_ROOT))

from psins_py.imu_utils import attrottt, avp2imu, cnscl, imuadderr  # noqa: E402
from psins_py.kf_utils import kfupdate  # noqa: E402
from psins_py.math_utils import a2qua, askew, m2rv, q2att, q2mat, qmul, qupdt2, rv2m, rv2q  # noqa: E402
from psins_py.nav_utils import Earth, glv, posset  # noqa: E402


@dataclass
class IterationLog18:
    iteration: int
    final_att_deg: list[float]
    att_err_pry_arcsec: list[float]
    pitch_arcsec: float
    roll_arcsec: float
    yaw_arcsec: float
    yaw_abs_arcsec: float
    att_err_norm_arcsec: float
    est_eb_dph: list[float]
    est_db_ug: list[float]
    est_ng_dph: list[float]
    est_xa_ug: list[float]


@dataclass
class RunSummary:
    seed: int
    max_iter: int
    wash_scale: float
    carry_att_seed: bool
    total_time_s: float
    rot_paras: list[list[float]]
    phi0_deg: list[float]
    truth_final_att_deg: list[float]
    final_att_deg: list[float]
    final_att_err_pry_arcsec: list[float]
    pitch_arcsec: float
    roll_arcsec: float
    yaw_arcsec: float
    yaw_abs_arcsec: float
    att_err_norm_arcsec: float
    best_iteration_by_yaw: dict[str, Any]
    best_iteration_by_norm: dict[str, Any]
    iter_logs: list[dict[str, Any]]


def left_quat_update(q: np.ndarray, phi: np.ndarray) -> np.ndarray:
    qq = qmul(rv2q(phi), q)
    return qq / np.linalg.norm(qq)


# Keep the same sign convention as the historical DAR baseline scripts.
def qaddphi(q: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return left_quat_update(q, phi)


def qdelphi(q: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return left_quat_update(q, phi)


# If q1 = qaddphi(q2, phi), then qq2phi(q1, q2) ~= phi.
def qq2phi(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    return m2rv(q2mat(q1) @ q2mat(q2).T)


def build_rot_paras() -> np.ndarray:
    """Continuous dual-axis DAR path used by the baseline experiments."""
    return np.array([
        [1, 0, 0, 1, 720 * glv.deg, 30, 0, 0],
        [1, 0, 0, 1, -720 * glv.deg, 30, 0, 0],
        [1, 0, 0, 1, 0 * glv.deg, 10, 0, 0],
        [2, 0, 1, 0, 90 * glv.deg, 10, 0, 0],
        [3, 1, 0, 0, 720 * glv.deg, 30, 0, 0],
        [3, 1, 0, 0, -720 * glv.deg, 30, 0, 0],
        [3, 1, 0, 0, 0 * glv.deg, 10, 0, 0],
        [4, 0, 1, 0, 90 * glv.deg, 10, 0, 0],
        [5, 0, 0, -1, 720 * glv.deg, 30, 0, 0],
        [5, 0, 0, -1, -720 * glv.deg, 30, 0, 0],
        [5, 0, 0, -1, 0 * glv.deg, 10, 0, 0],
        [6, 0, 1, 0, 90 * glv.deg, 10, 0, 0],
        [7, 0, 1, 0, 90 * glv.deg, 10, 0, 0],
        [7, 0, 0, 1, 0 * glv.deg, 100, 0, 0],
    ], dtype=float)


def build_imuerr() -> dict[str, np.ndarray]:
    """Baseline IMU error setting, including truth-side diagonal dKg/dKa."""
    return {
        'eb': np.full(3, 0.01 * glv.dph),
        'db': np.full(3, 100.0 * glv.ug),
        'web': np.full(3, 0.0001 * glv.dpsh),
        'wdb': np.full(3, 1.0 * glv.ugpsHz),
        'dKg': np.diag(np.full(3, 30.0 * glv.ppm)),
        'dKa': np.diag(np.full(3, 30.0 * glv.ppm)),
    }


def avnkfinit_18_accel_colored(
    nts: float,
    pos: np.ndarray,
    phi0: np.ndarray,
    imuerr: dict[str, np.ndarray],
    wvn: np.ndarray,
    ng_sigma: np.ndarray,
    tau_g_s: np.ndarray,
    xa_sigma: np.ndarray,
    tau_a_s: np.ndarray,
) -> dict[str, Any]:
    """18-state KF init for x = [phi, dv, eb, db, ng, xa]."""
    eth = Earth(pos)
    web = np.asarray(imuerr['web']).reshape(3)
    wdb = np.asarray(imuerr['wdb']).reshape(3)
    eb = np.asarray(imuerr['eb']).reshape(3)
    db = np.asarray(imuerr['db']).reshape(3)
    ng_sigma = np.asarray(ng_sigma).reshape(3)
    tau_g_s = np.asarray(tau_g_s).reshape(3)
    xa_sigma = np.asarray(xa_sigma).reshape(3)
    tau_a_s = np.asarray(tau_a_s).reshape(3)

    init_eb_p = np.maximum(eb, 0.1 * glv.dph)
    init_db_p = np.maximum(db, 1000.0 * glv.ug)
    init_xa_p = np.maximum(xa_sigma, 5.0 * glv.ug)

    fg = np.exp(-nts / tau_g_s)
    fa = np.exp(-nts / tau_a_s)
    q_ng = ng_sigma * np.sqrt(np.maximum(1.0 - fg ** 2, 0.0))
    q_xa = xa_sigma * np.sqrt(np.maximum(1.0 - fa ** 2, 0.0))

    qk = np.zeros((18, 18))
    qk[0:3, 0:3] = np.diag(web ** 2 * nts)
    qk[3:6, 3:6] = np.diag(wdb ** 2 * nts)
    qk[12:15, 12:15] = np.diag(q_ng ** 2)
    qk[15:18, 15:18] = np.diag(q_xa ** 2)

    ft = np.zeros((18, 18))
    ft[0:3, 0:3] = askew(-eth.wnie)
    phikk_1 = np.eye(18) + ft * nts
    phikk_1[12:15, 12:15] = np.diag(fg)
    phikk_1[15:18, 15:18] = np.diag(fa)

    return {
        'n': 18,
        'm': 3,
        'nts': nts,
        'Qk': qk,
        'Rk': np.diag(wvn.reshape(3)) ** 2 / nts,
        'Pxk': np.diag(np.r_[phi0, np.ones(3), init_eb_p, init_db_p, ng_sigma, init_xa_p]) ** 2,
        'Phikk_1': phikk_1,
        'Hk': np.hstack([np.zeros((3, 3)), np.eye(3), np.zeros((3, 12))]),
        'xk': np.zeros(18),
        'fg': fg,
        'fa': fa,
    }


def alignvn_18state_iter(
    imu: np.ndarray,
    qnb: np.ndarray,
    pos: np.ndarray,
    phi0: np.ndarray,
    imuerr: dict[str, np.ndarray],
    wvn: np.ndarray,
    max_iter: int,
    truth_att: np.ndarray,
    ng_sigma: np.ndarray,
    tau_g_s: np.ndarray,
    xa_sigma: np.ndarray,
    tau_a_s: np.ndarray,
    wash_scale: float = 0.5,
    carry_att_seed: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[IterationLog18]]:
    """Iterative dual-axis 18-state baseline alignment."""
    imu_corr = imu.copy()

    nn = 2
    ts = float(imu_corr[1, -1] - imu_corr[0, -1])
    nts = nn * ts
    qnb_seed = a2qua(qnb) if len(qnb) == 3 else np.asarray(qnb).reshape(4)

    length = (len(imu_corr) // nn) * nn
    imu_corr = imu_corr[:length]

    eth = Earth(pos)
    cnn = rv2m(-eth.wnie * nts / 2)

    final_attk = None
    final_xkpk = None
    iter_logs: list[IterationLog18] = []

    for iteration in range(1, max_iter + 1):
        kf = avnkfinit_18_accel_colored(nts, pos, phi0, imuerr, wvn, ng_sigma, tau_g_s, xa_sigma, tau_a_s)
        vn = np.zeros(3)
        qnbi = qnb_seed.copy()

        attk_rows: list[np.ndarray] = []
        xkpk_rows: list[np.ndarray] = []

        for k in range(0, length, nn):
            wvm = imu_corr[k:k + nn, 0:6]
            t = float(imu_corr[k + nn - 1, -1])
            phim, dvbm = cnscl(wvm)

            cnb = q2mat(qnbi)
            dvn = cnn @ cnb @ dvbm
            vn = vn + dvn + eth.gn * nts
            qnbi = qupdt2(qnbi, phim, eth.wnin * nts)

            phi_k = kf['Phikk_1'].copy()
            cnbts = cnb * nts
            phi_k[3:6, 0:3] = askew(dvn)
            phi_k[3:6, 9:12] = cnbts
            phi_k[3:6, 15:18] = cnbts
            phi_k[0:3, 6:9] = -cnbts
            phi_k[0:3, 12:15] = -cnbts
            phi_k[12:15, 12:15] = np.diag(kf['fg'])
            phi_k[15:18, 15:18] = np.diag(kf['fa'])
            kf['Phikk_1'] = phi_k

            kf = kfupdate(kf, vn)

            qnbi = qdelphi(qnbi, 0.91 * kf['xk'][0:3])
            kf['xk'][0:3] *= 0.09
            vn = vn - 0.91 * kf['xk'][3:6]
            kf['xk'][3:6] *= 0.09

            attk_rows.append(np.r_[q2att(qnbi), vn, t])
            xkpk_rows.append(np.r_[kf['xk'], np.diag(kf['Pxk']), t])

        attk = np.asarray(attk_rows)
        xkpk = np.asarray(xkpk_rows)
        final_attk = attk
        final_xkpk = xkpk

        est_att = attk[-1, 0:3]
        att_err_pry_arcsec = qq2phi(a2qua(est_att), a2qua(truth_att)) / glv.sec
        roll_arcsec = float(att_err_pry_arcsec[0])
        pitch_arcsec = float(att_err_pry_arcsec[1])
        yaw_arcsec = float(att_err_pry_arcsec[2])
        iter_logs.append(IterationLog18(
            iteration=iteration,
            final_att_deg=(est_att / glv.deg).tolist(),
            att_err_pry_arcsec=att_err_pry_arcsec.tolist(),
            pitch_arcsec=pitch_arcsec,
            roll_arcsec=roll_arcsec,
            yaw_arcsec=yaw_arcsec,
            yaw_abs_arcsec=abs(yaw_arcsec),
            att_err_norm_arcsec=float(np.linalg.norm(att_err_pry_arcsec)),
            est_eb_dph=(kf['xk'][6:9] / glv.dph).tolist(),
            est_db_ug=(kf['xk'][9:12] / glv.ug).tolist(),
            est_ng_dph=(kf['xk'][12:15] / glv.dph).tolist(),
            est_xa_ug=(kf['xk'][15:18] / glv.ug).tolist(),
        ))

        if iteration < max_iter:
            if carry_att_seed:
                qnb_seed = qnbi.copy()
            imu_corr[:, 0:3] -= wash_scale * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= wash_scale * kf['xk'][9:12] * ts

    assert final_attk is not None and final_xkpk is not None
    return final_attk[-1, 0:3], final_attk, final_xkpk, iter_logs


def run_case(seed: int, max_iter: int, wash_scale: float, carry_att_seed: bool) -> RunSummary:
    np.random.seed(seed)

    ts = 0.01
    att0 = np.array([0.0, 0.0, 0.0])
    pos0 = posset(34, 116, 480, isdeg=1)
    rot_paras = build_rot_paras()
    att_truth = attrottt(att0, rot_paras, ts)
    imu, _ = avp2imu(att_truth, pos0)

    imuerr = build_imuerr()
    imu_noisy = imuadderr(imu, imuerr)

    phi0 = np.array([0.1, 0.1, 0.5]) * glv.deg
    att0_guess = q2att(qaddphi(a2qua(att0), phi0))
    wvn = np.array([0.01, 0.01, 0.01])

    att_aligned, _attk, _xkpk, iter_logs = alignvn_18state_iter(
        imu=imu_noisy,
        qnb=att0_guess,
        pos=pos0,
        phi0=phi0,
        imuerr=imuerr,
        wvn=wvn,
        max_iter=max_iter,
        truth_att=att_truth[-1, 0:3],
        ng_sigma=np.array([0.05, 0.05, 0.05]) * glv.dph,
        tau_g_s=np.array([300.0, 300.0, 300.0]),
        xa_sigma=np.array([0.01, 0.01, 0.01]) * glv.ug,
        tau_a_s=np.array([100.0, 100.0, 100.0]),
        wash_scale=wash_scale,
        carry_att_seed=carry_att_seed,
    )

    att_err_pry_arcsec = qq2phi(a2qua(att_aligned), a2qua(att_truth[-1, 0:3])) / glv.sec
    roll_arcsec = float(att_err_pry_arcsec[0])
    pitch_arcsec = float(att_err_pry_arcsec[1])
    yaw_arcsec = float(att_err_pry_arcsec[2])
    best_iteration_by_yaw = min(iter_logs, key=lambda x: x.yaw_abs_arcsec)
    best_iteration_by_norm = min(iter_logs, key=lambda x: x.att_err_norm_arcsec)

    return RunSummary(
        seed=seed,
        max_iter=max_iter,
        wash_scale=wash_scale,
        carry_att_seed=carry_att_seed,
        total_time_s=float(att_truth[-1, -1]),
        rot_paras=rot_paras.tolist(),
        phi0_deg=(phi0 / glv.deg).tolist(),
        truth_final_att_deg=(att_truth[-1, 0:3] / glv.deg).tolist(),
        final_att_deg=(att_aligned / glv.deg).tolist(),
        final_att_err_pry_arcsec=att_err_pry_arcsec.tolist(),
        pitch_arcsec=pitch_arcsec,
        roll_arcsec=roll_arcsec,
        yaw_arcsec=yaw_arcsec,
        yaw_abs_arcsec=abs(yaw_arcsec),
        att_err_norm_arcsec=float(np.linalg.norm(att_err_pry_arcsec)),
        best_iteration_by_yaw=asdict(best_iteration_by_yaw),
        best_iteration_by_norm=asdict(best_iteration_by_norm),
        iter_logs=[asdict(x) for x in iter_logs],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Clean dual-axis 18-state DAR alignment baseline')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for imuadderr')
    parser.add_argument('--max-iter', type=int, default=5, help='Outer iterative alignment count')
    parser.add_argument('--wash-scale', type=float, default=0.5, help='Bias wash scale between outer iterations')
    parser.add_argument('--no-carry-att-seed', action='store_true', help='Disable carrying final attitude seed to next iteration')
    parser.add_argument('--json-out', type=str, default='', help='Optional JSON output path')
    args = parser.parse_args()

    summary = run_case(
        seed=args.seed,
        max_iter=args.max_iter,
        wash_scale=args.wash_scale,
        carry_att_seed=not args.no_carry_att_seed,
    )

    payload = asdict(summary)
    print(json.dumps(payload, indent=2, ensure_ascii=False))

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f'\n[ok] wrote {out_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
