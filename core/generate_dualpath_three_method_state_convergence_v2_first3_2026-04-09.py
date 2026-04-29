#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import types
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np

# Stub unavailable plotting deps before loading PSINS modules.
if 'matplotlib' not in sys.modules:
    matplotlib_stub = types.ModuleType('matplotlib')
    pyplot_stub = types.ModuleType('matplotlib.pyplot')
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules['matplotlib'] = matplotlib_stub
    sys.modules['matplotlib.pyplot'] = pyplot_stub
if 'seaborn' not in sys.modules:
    sys.modules['seaborn'] = types.ModuleType('seaborn')

WORKSPACE = Path('/root/.openclaw/workspace')
PSINS_ROOT = WORKSPACE / 'tmp_psins_py'
if str(PSINS_ROOT) not in sys.path:
    sys.path.insert(0, str(PSINS_ROOT))

BASE12_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
H24_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_hybrid24_staged_py_2026-03-30.py'
PURE_SCD_PATH = WORKSPACE / 'scripts' / 'compare_ch4_pure_scd_vs_freeze_2026-04-03.py'
OUT_DIR = WORKSPACE / 'tmp' / 'psins_dualpath_three_method_state_convergence_v2_first3_2026-04-09'
OUT_JSON = OUT_DIR / 'summary.json'
TRACE_DT_S = 0.2
OUTER_ITERS = 3
ROT_GATE_DPS = 5.0
MAX_POINTS = 1200
LINE_OPACITY = 0.56
LINE_WIDTH = 1.5
LEGEND_LINE_WIDTH = 2.8
TS = 0.01
WASH_SCALE = 0.5
SCALE_WASH_SCALE = 0.5
WVN = np.array([0.01, 0.01, 0.01])
PHI_DEG = np.array([0.1, 0.1, 0.5])
SVG_ONLY = True

COLORS = {
    'g2_scaleonly_rotation': '#1769aa',
    'g3_markov_rotation': '#d9822b',
    'g4_scd_rotation': '#c92a2a',
}
LINE_DASHES = {
    'g2_scaleonly_rotation': None,
    'g3_markov_rotation': '10 6',
    'g4_scd_rotation': '3 5',
}
GROUP_LABELS = {
    'g2_scaleonly_rotation': 'G1baseline',
    'g3_markov_rotation': 'G2MarkovMarkov',
    'g4_scd_rotation': 'G3Markov+LLM+SCD',
}

_BASE12 = None
_H24 = None
_PURE = None


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_base12():
    global _BASE12
    if _BASE12 is None:
        _BASE12 = load_module('dualpath_plot_base12_20260409', BASE12_PATH)
    return _BASE12


def load_h24():
    global _H24
    if _H24 is None:
        _H24 = load_module('dualpath_plot_h24_20260409', H24_PATH)
    return _H24


def load_pure():
    global _PURE
    if _PURE is None:
        _PURE = load_module('dualpath_plot_pure_20260409', PURE_SCD_PATH)
    return _PURE


def state_meta18(glv):
    return [
        {'label': 'phi_x', 'unit': 'deg', 'scale': 1.0 / glv.deg},
        {'label': 'phi_y', 'unit': 'deg', 'scale': 1.0 / glv.deg},
        {'label': 'phi_z', 'unit': 'deg', 'scale': 1.0 / glv.deg},
        {'label': 'dv_x', 'unit': 'm/s', 'scale': 1.0},
        {'label': 'dv_y', 'unit': 'm/s', 'scale': 1.0},
        {'label': 'dv_z', 'unit': 'm/s', 'scale': 1.0},
        {'label': 'eb_x', 'unit': 'dph', 'scale': 1.0 / glv.dph},
        {'label': 'eb_y', 'unit': 'dph', 'scale': 1.0 / glv.dph},
        {'label': 'eb_z', 'unit': 'dph', 'scale': 1.0 / glv.dph},
        {'label': 'db_x', 'unit': 'ug', 'scale': 1.0 / glv.ug},
        {'label': 'db_y', 'unit': 'ug', 'scale': 1.0 / glv.ug},
        {'label': 'db_z', 'unit': 'ug', 'scale': 1.0 / glv.ug},
        {'label': 'dKg_x', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
        {'label': 'dKg_y', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
        {'label': 'dKg_z', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
        {'label': 'dKa_x', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
        {'label': 'dKa_y', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
        {'label': 'dKa_z', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
    ]


def state_meta24(glv):
    return [
        {'label': 'phi_x', 'unit': 'deg', 'scale': 1.0 / glv.deg},
        {'label': 'phi_y', 'unit': 'deg', 'scale': 1.0 / glv.deg},
        {'label': 'phi_z', 'unit': 'deg', 'scale': 1.0 / glv.deg},
        {'label': 'dv_x', 'unit': 'm/s', 'scale': 1.0},
        {'label': 'dv_y', 'unit': 'm/s', 'scale': 1.0},
        {'label': 'dv_z', 'unit': 'm/s', 'scale': 1.0},
        {'label': 'eb_x', 'unit': 'dph', 'scale': 1.0 / glv.dph},
        {'label': 'eb_y', 'unit': 'dph', 'scale': 1.0 / glv.dph},
        {'label': 'eb_z', 'unit': 'dph', 'scale': 1.0 / glv.dph},
        {'label': 'db_x', 'unit': 'ug', 'scale': 1.0 / glv.ug},
        {'label': 'db_y', 'unit': 'ug', 'scale': 1.0 / glv.ug},
        {'label': 'db_z', 'unit': 'ug', 'scale': 1.0 / glv.ug},
        {'label': 'ng_x', 'unit': 'dph', 'scale': 1.0 / glv.dph},
        {'label': 'ng_y', 'unit': 'dph', 'scale': 1.0 / glv.dph},
        {'label': 'ng_z', 'unit': 'dph', 'scale': 1.0 / glv.dph},
        {'label': 'xa_x', 'unit': 'ug', 'scale': 1.0 / glv.ug},
        {'label': 'xa_y', 'unit': 'ug', 'scale': 1.0 / glv.ug},
        {'label': 'xa_z', 'unit': 'ug', 'scale': 1.0 / glv.ug},
        {'label': 'dKg_x', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
        {'label': 'dKg_y', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
        {'label': 'dKg_z', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
        {'label': 'dKa_x', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
        {'label': 'dKa_y', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
        {'label': 'dKa_z', 'unit': 'ppm', 'scale': 1.0 / glv.ppm},
    ]


def state_by_label(meta):
    return {item['label']: {**item, 'index': i} for i, item in enumerate(meta)}


def build_shared_dual_dataset():
    base12 = load_base12()
    h24 = load_h24()
    acc18 = h24.load_acc18()
    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    att0 = np.array([0.0, 0.0, 0.0])
    rot_paras = base12.build_rot_paras()
    att_truth = acc18.attrottt(att0, rot_paras, TS)
    imu, _ = acc18.avp2imu(att_truth, pos0)
    imuerr = base12.build_imuerr()
    imu_noisy = acc18.imuadderr(imu, imuerr)
    phi = PHI_DEG * acc18.glv.deg
    att0_guess = acc18.q2att(base12.qaddphi(acc18.a2qua(att0), phi))
    return {
        'pos0': pos0,
        'att0_deg': [0.0, 0.0, 0.0],
        'rot_paras': rot_paras,
        'duration_s': float(att_truth[-1, 3]),
        'truth_att': att_truth[-1, 0:3].copy(),
        'imu_noisy': imu_noisy,
        'imuerr': imuerr,
        'phi': phi,
        'att0_guess': att0_guess,
    }


def trace_scale18_first3(shared):
    h24 = load_h24()
    acc18 = h24.load_acc18()
    glv = acc18.glv
    imuerr = shared['imuerr']
    imu_corr = shared['imu_noisy'].copy()
    nn = 2
    ts = float(imu_corr[1, -1] - imu_corr[0, -1])
    nts = nn * ts
    qnb_seed = acc18.a2qua(shared['att0_guess']) if len(shared['att0_guess']) == 3 else np.asarray(shared['att0_guess']).reshape(4)
    length = (len(imu_corr) // nn) * nn
    imu_corr = imu_corr[:length]

    eth = acc18.Earth(shared['pos0'])
    cnn = acc18.rv2m(-eth.wnie * nts / 2)
    rot_gate_rad = ROT_GATE_DPS * glv.deg

    web = np.asarray(imuerr['web']).reshape(3)
    wdb = np.asarray(imuerr['wdb']).reshape(3)
    eb = np.asarray(imuerr['eb']).reshape(3)
    db = np.asarray(imuerr['db']).reshape(3)
    init_eb_p = np.maximum(eb, 0.1 * glv.dph)
    init_db_p = np.maximum(db, 1000 * glv.ug)
    init_scale_p = np.full(3, 100.0 * glv.ppm)
    qk = np.zeros((18, 18))
    qk[0:3, 0:3] = np.diag(web**2 * nts)
    qk[3:6, 3:6] = np.diag(wdb**2 * nts)
    ft = np.zeros((18, 18))
    ft[0:3, 0:3] = acc18.askew(-eth.wnie)
    phikk_1 = np.eye(18) + ft * nts
    hk = np.hstack([np.zeros((3, 3)), np.eye(3), np.zeros((3, 12))])

    p_all, x_all, iter_bounds = [], [], []
    last_saved_global_t = -1e9
    for iteration in range(1, OUTER_ITERS + 1):
        kf = {
            'Phikk_1': phikk_1.copy(),
            'Qk': qk,
            'Rk': np.diag(WVN.reshape(3)) ** 2 / nts,
            'Pxk': np.diag(np.r_[shared['phi'], np.ones(3), init_eb_p, init_db_p, init_scale_p, init_scale_p]) ** 2,
            'Hk': hk.copy(),
            'xk': np.zeros(18),
        }
        vn = np.zeros(3)
        qnbi = qnb_seed.copy()
        elapsed_s = 0.0

        for k in range(0, length, nn):
            wvm = imu_corr[k:k + nn, 0:6]
            phim, dvbm = acc18.cnscl(wvm)
            cnb = acc18.q2mat(qnbi)
            dvn = cnn @ cnb @ dvbm
            vn = vn + dvn + eth.gn * nts
            qnbi = acc18.qupdt2(qnbi, phim, eth.wnin * nts)

            phi_k = kf['Phikk_1'].copy()
            cnbts = cnb * nts
            phi_k[3:6, 0:3] = acc18.askew(dvn)
            phi_k[3:6, 9:12] = cnbts
            phi_k[0:3, 6:9] = -cnbts
            high_rot = np.max(np.abs(phim / nts)) > rot_gate_rad
            if high_rot:
                phi_k[0:3, 12:15] = -cnb @ np.diag(phim[0:3])
                phi_k[3:6, 15:18] = cnb @ np.diag(dvbm[0:3])
            else:
                phi_k[0:3, 12:15] = 0.0
                phi_k[3:6, 15:18] = 0.0

            kf['Phikk_1'] = phi_k
            kf = acc18.kfupdate(kf, vn)
            qnbi = acc18.qdelphi(qnbi, 0.91 * kf['xk'][0:3])
            kf['xk'][0:3] *= 0.09
            vn = vn - 0.91 * kf['xk'][3:6]
            kf['xk'][3:6] *= 0.09

            elapsed_s += nts
            current_global_t = elapsed_s + (iteration - 1) * shared['duration_s']
            if not p_all or current_global_t - last_saved_global_t >= TRACE_DT_S - 1e-12:
                p_all.append(np.diag(kf['Pxk']).copy())
                x_all.append(np.copy(kf['xk']))
                last_saved_global_t = current_global_t

        iter_bounds.append(len(p_all))
        if iteration < OUTER_ITERS:
            qnb_seed = qnbi.copy()
            imu_corr[:, 0:3] -= WASH_SCALE * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= WASH_SCALE * kf['xk'][9:12] * ts
            imu_corr = h24.apply_scale_wash(imu_corr, kf['xk'][12:15], kf['xk'][15:18], SCALE_WASH_SCALE)

    return {
        'group_key': 'g2_scaleonly_rotation',
        'label': GROUP_LABELS['g2_scaleonly_rotation'],
        'n_states': 18,
        'state_by_label': state_by_label(state_meta18(glv)),
        'x_trace': np.array(x_all),
        'p_trace': np.array(p_all),
        'iter_bounds': iter_bounds,
    }


def trace_plain24_first3(shared):
    h24 = load_h24()
    acc18 = h24.load_acc18()
    glv = acc18.glv
    imu_corr = shared['imu_noisy'].copy()
    nn = 2
    ts = float(imu_corr[1, -1] - imu_corr[0, -1])
    nts = nn * ts
    qnb_seed = acc18.a2qua(shared['att0_guess']) if len(shared['att0_guess']) == 3 else np.asarray(shared['att0_guess']).reshape(4)
    length = (len(imu_corr) // nn) * nn
    imu_corr = imu_corr[:length]

    eth = acc18.Earth(shared['pos0'])
    cnn = acc18.rv2m(-eth.wnie * nts / 2)
    rot_gate_rad = ROT_GATE_DPS * glv.deg

    p_all, x_all, iter_bounds = [], [], []
    last_saved_global_t = -1e9
    for iteration in range(1, OUTER_ITERS + 1):
        kf = h24.avnkfinit_24(
            nts, shared['pos0'], shared['phi'], shared['imuerr'], WVN.copy(),
            np.array([0.05, 0.05, 0.05]) * glv.dph,
            np.array([300.0, 300.0, 300.0]),
            np.array([0.01, 0.01, 0.01]) * glv.ug,
            np.array([100.0, 100.0, 100.0]),
            enable_scale_states=True,
        )
        vn = np.zeros(3)
        qnbi = qnb_seed.copy()
        elapsed_s = 0.0

        for k in range(0, length, nn):
            wvm = imu_corr[k:k + nn, 0:6]
            phim, dvbm = acc18.cnscl(wvm)
            cnb = acc18.q2mat(qnbi)
            dvn = cnn @ cnb @ dvbm
            vn = vn + dvn + eth.gn * nts
            qnbi = acc18.qupdt2(qnbi, phim, eth.wnin * nts)

            phi_k = kf['Phikk_1'].copy()
            cnbts = cnb * nts
            phi_k[3:6, 0:3] = acc18.askew(dvn)
            phi_k[3:6, 9:12] = cnbts
            phi_k[3:6, 15:18] = cnbts
            phi_k[0:3, 6:9] = -cnbts
            phi_k[0:3, 12:15] = -cnbts
            phi_k[12:15, 12:15] = np.diag(kf['fg'])
            phi_k[15:18, 15:18] = np.diag(kf['fa'])
            high_rot = np.max(np.abs(phim / nts)) > rot_gate_rad
            if high_rot:
                phi_k[0:3, 18:21] = -cnb @ np.diag(phim[0:3])
                phi_k[3:6, 21:24] = cnb @ np.diag(dvbm[0:3])
            else:
                phi_k[0:3, 18:21] = 0.0
                phi_k[3:6, 21:24] = 0.0
            kf['Phikk_1'] = phi_k
            kf = acc18.kfupdate(kf, vn)

            qnbi = acc18.qdelphi(qnbi, 0.91 * kf['xk'][0:3])
            kf['xk'][0:3] *= 0.09
            vn = vn - 0.91 * kf['xk'][3:6]
            kf['xk'][3:6] *= 0.09

            elapsed_s += nts
            current_global_t = elapsed_s + (iteration - 1) * shared['duration_s']
            if not p_all or current_global_t - last_saved_global_t >= TRACE_DT_S - 1e-12:
                p_all.append(np.diag(kf['Pxk']).copy())
                x_all.append(np.copy(kf['xk']))
                last_saved_global_t = current_global_t

        iter_bounds.append(len(p_all))
        if iteration < OUTER_ITERS:
            qnb_seed = qnbi.copy()
            imu_corr[:, 0:3] -= WASH_SCALE * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= WASH_SCALE * kf['xk'][9:12] * ts
            imu_corr = h24.apply_scale_wash(imu_corr, kf['xk'][18:21], kf['xk'][21:24], SCALE_WASH_SCALE)

    return {
        'group_key': 'g3_markov_rotation',
        'label': GROUP_LABELS['g3_markov_rotation'],
        'n_states': 24,
        'state_by_label': state_by_label(state_meta24(glv)),
        'x_trace': np.array(x_all),
        'p_trace': np.array(p_all),
        'iter_bounds': iter_bounds,
    }


def trace_purescd24_first3(shared):
    base12 = load_base12()
    h24 = load_h24()
    pure = load_pure()
    acc18 = h24.load_acc18()
    glv = acc18.glv
    imu_corr = shared['imu_noisy'].copy()
    nn = 2
    ts = float(imu_corr[1, -1] - imu_corr[0, -1])
    nts = nn * ts
    qnb_seed = acc18.a2qua(shared['att0_guess']) if len(shared['att0_guess']) == 3 else np.asarray(shared['att0_guess']).reshape(4)
    length = (len(imu_corr) // nn) * nn
    imu_corr = imu_corr[:length]

    eth = acc18.Earth(shared['pos0'])
    cnn = acc18.rv2m(-eth.wnie * nts / 2)
    rot_gate_rad = ROT_GATE_DPS * glv.deg
    scd_cfg = pure.SCDConfig(enabled=True, alpha=0.995, transition_duration_s=2.0, apply_after_release_iter=1, note='hard_a995_td2_i1')

    p_all, x_all, iter_bounds = [], [], []
    last_saved_global_t = -1e9
    for iteration in range(1, OUTER_ITERS + 1):
        kf = h24.avnkfinit_24(
            nts, shared['pos0'], shared['phi'], shared['imuerr'], WVN.copy(),
            np.array([0.05, 0.05, 0.05]) * glv.dph,
            np.array([300.0, 300.0, 300.0]),
            np.array([0.01, 0.01, 0.01]) * glv.ug,
            np.array([100.0, 100.0, 100.0]),
            enable_scale_states=True,
        )
        vn = np.zeros(3)
        qnbi = qnb_seed.copy()
        time_since_rot_stop = 0.0
        scd_applied_this_phase = False
        elapsed_s = 0.0

        for k in range(0, length, nn):
            wvm = imu_corr[k:k + nn, 0:6]
            phim, dvbm = acc18.cnscl(wvm)
            cnb = acc18.q2mat(qnbi)
            dvn = cnn @ cnb @ dvbm
            vn = vn + dvn + eth.gn * nts
            qnbi = acc18.qupdt2(qnbi, phim, eth.wnin * nts)

            phi_k = kf['Phikk_1'].copy()
            cnbts = cnb * nts
            phi_k[3:6, 0:3] = acc18.askew(dvn)
            phi_k[3:6, 9:12] = cnbts
            phi_k[3:6, 15:18] = cnbts
            phi_k[0:3, 6:9] = -cnbts
            phi_k[0:3, 12:15] = -cnbts
            phi_k[12:15, 12:15] = np.diag(kf['fg'])
            phi_k[15:18, 15:18] = np.diag(kf['fa'])
            high_rot = np.max(np.abs(phim / nts)) > rot_gate_rad
            if high_rot:
                phi_k[0:3, 18:21] = -cnb @ np.diag(phim[0:3])
                phi_k[3:6, 21:24] = cnb @ np.diag(dvbm[0:3])
                time_since_rot_stop = 0.0
                scd_applied_this_phase = False
            else:
                phi_k[0:3, 18:21] = 0.0
                phi_k[3:6, 21:24] = 0.0
                time_since_rot_stop += nts
            kf['Phikk_1'] = phi_k
            kf = acc18.kfupdate(kf, vn)

            qnbi = acc18.qdelphi(qnbi, 0.91 * kf['xk'][0:3])
            kf['xk'][0:3] *= 0.09
            vn = vn - 0.91 * kf['xk'][3:6]
            kf['xk'][3:6] *= 0.09

            if scd_cfg.enabled and iteration >= scd_cfg.apply_after_release_iter and (not high_rot):
                if (time_since_rot_stop >= scd_cfg.transition_duration_s) and (not scd_applied_this_phase):
                    kf = pure.apply_scd_once(kf, scd_cfg)
                    scd_applied_this_phase = True

            elapsed_s += nts
            current_global_t = elapsed_s + (iteration - 1) * shared['duration_s']
            if not p_all or current_global_t - last_saved_global_t >= TRACE_DT_S - 1e-12:
                p_all.append(np.diag(kf['Pxk']).copy())
                x_all.append(np.copy(kf['xk']))
                last_saved_global_t = current_global_t

        iter_bounds.append(len(p_all))
        if iteration < OUTER_ITERS:
            qnb_seed = qnbi.copy()
            imu_corr[:, 0:3] -= WASH_SCALE * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= WASH_SCALE * kf['xk'][9:12] * ts
            imu_corr = h24.apply_scale_wash(imu_corr, kf['xk'][18:21], kf['xk'][21:24], SCALE_WASH_SCALE)

    return {
        'group_key': 'g4_scd_rotation',
        'label': GROUP_LABELS['g4_scd_rotation'],
        'n_states': 24,
        'state_by_label': state_by_label(state_meta24(glv)),
        'x_trace': np.array(x_all),
        'p_trace': np.array(p_all),
        'iter_bounds': iter_bounds,
    }


def line_style(group_key: str, legend: bool = False) -> str:
    color = COLORS.get(group_key, '#000')
    dash = LINE_DASHES.get(group_key)
    width = LEGEND_LINE_WIDTH if legend else LINE_WIDTH
    opacity = 1.0 if legend else LINE_OPACITY
    attrs = [
        f'stroke="{color}"',
        f'stroke-width="{width}"',
        f'stroke-opacity="{opacity}"',
        'stroke-linecap="round"',
        'stroke-linejoin="round"',
    ]
    if dash:
        attrs.append(f'stroke-dasharray="{dash}"')
    return ' '.join(attrs)


def fmt_y(v: float) -> str:
    if abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 1e-3):
        return f'{v:.2e}'
    return f'{v:.4f}'


def fmt_time_tick(v: float) -> str:
    return str(int(round(v)))


def nice_ticks(vmin: float, vmax: float, n: int = 5):
    if math.isclose(vmin, vmax, rel_tol=0.0, abs_tol=1e-15):
        delta = 1.0 if abs(vmin) < 1e-12 else abs(vmin) * 0.1
        vmin -= delta
        vmax += delta
    return [vmin + (vmax - vmin) * i / (n - 1) for i in range(n)]


def downsample(x, ys, max_points=MAX_POINTS):
    n = len(x)
    if n <= max_points:
        return x, ys
    idx = [round(i * (n - 1) / (max_points - 1)) for i in range(max_points)]
    dedup = []
    seen = set()
    for i in idx:
        if i not in seen:
            seen.add(i)
            dedup.append(i)
    return [x[i] for i in dedup], [[arr[i] for i in dedup] for arr in ys]


def polyline(xs, ys):
    return ' '.join(f'{x:.2f},{y:.2f}' for x, y in zip(xs, ys))


def percentile(seq, q):
    if not seq:
        return 0.0
    arr = sorted(seq)
    if len(arr) == 1:
        return arr[0]
    pos = (len(arr) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    frac = pos - lo
    return arr[lo] * (1 - frac) + arr[hi] * frac


def robust_range(values, low_q=0.01, high_q=0.99, min_pad_frac=0.08):
    if not values:
        return -1.0, 1.0
    lo = percentile(values, low_q)
    hi = percentile(values, high_q)
    full_lo = min(values)
    full_hi = max(values)
    if full_lo < lo:
        lo = min(lo, full_lo)
    if full_hi > hi:
        hi = max(hi, full_hi)
    if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-15):
        delta = 1.0 if abs(lo) < 1e-12 else abs(lo) * 0.1
        lo -= delta
        hi += delta
    pad = max((hi - lo) * min_pad_frac, 1e-12)
    return lo - pad, hi + pad


def log_ticks(vmin, vmax, n=5):
    if vmin <= 0:
        vmin = min(v for v in [vmax * 1e-6, 1e-12] if v > 0)
    if vmax <= 0:
        vmax = 1.0
    lo = math.log10(vmin)
    hi = math.log10(vmax)
    if math.isclose(lo, hi, abs_tol=1e-12):
        lo -= 1.0
        hi += 1.0
    return [10 ** (lo + (hi - lo) * i / (n - 1)) for i in range(n)]


def map_linear(v, vmin, vmax, y0, y1):
    if math.isclose(vmin, vmax, abs_tol=1e-15):
        return (y0 + y1) / 2
    return y1 - (v - vmin) / (vmax - vmin) * (y1 - y0)


def map_log(v, vmin, vmax, y0, y1):
    v = max(v, vmin)
    lo = math.log10(vmin)
    hi = math.log10(vmax)
    if math.isclose(lo, hi, abs_tol=1e-15):
        return (y0 + y1) / 2
    return y1 - (math.log10(v) - lo) / (hi - lo) * (y1 - y0)


def draw_panel(parts, rect, title, x_ticks, y_ticks, x_mapper, y_mapper, x_label, y_label, zero_y=None, iter_bounds=None):
    x0, y0, x1, y1 = rect
    parts.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="white" stroke="#d0d7de"/>')
    parts.append(f'<text x="{x0 + 8}" y="{y0 + 20}" class="panel-title">{escape(title)}</text>')
    plot_top = y0 + 28
    plot_bottom = y1 - 36
    plot_left = x0 + 78
    plot_right = x1 - 18
    for yt in y_ticks:
        yy = y_mapper(yt)
        parts.append(f'<line x1="{plot_left}" y1="{yy:.2f}" x2="{plot_right}" y2="{yy:.2f}" class="grid"/>')
        parts.append(f'<text x="{plot_left - 12}" y="{yy + 4:.2f}" class="tick" text-anchor="end">{escape(fmt_y(yt))}</text>')
    for xt in x_ticks:
        xx = x_mapper(xt)
        parts.append(f'<line x1="{xx:.2f}" y1="{plot_top}" x2="{xx:.2f}" y2="{plot_bottom}" class="grid"/>')
        parts.append(f'<text x="{xx:.2f}" y="{plot_bottom + 18}" class="tick" text-anchor="middle">{escape(fmt_time_tick(xt))}</text>')
    if zero_y is not None:
        parts.append(f'<line x1="{plot_left}" y1="{zero_y:.2f}" x2="{plot_right}" y2="{zero_y:.2f}" class="zero"/>')
    if iter_bounds:
        for bound in iter_bounds[:-1]:
            xx = x_mapper(bound)
            parts.append(f'<line x1="{xx:.2f}" y1="{plot_top}" x2="{xx:.2f}" y2="{plot_bottom}" class="iter"/>')
    parts.append(f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" class="axis"/>')
    parts.append(f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" class="axis"/>')
    parts.append(f'<text x="{(plot_left + plot_right)/2:.2f}" y="{y1 - 8}" class="small" text-anchor="middle">{escape(x_label)}</text>')
    y_label_x = x0 + 8
    y_label_y = (plot_top + plot_bottom) / 2
    parts.append(f'<text x="{y_label_x}" y="{y_label_y:.2f}" class="small" transform="rotate(-90 {y_label_x} {y_label_y:.2f})" text-anchor="middle">{escape(y_label)}</text>')


def render_state_preview(out_svg: Path, state_label: str, unit: str, series_list):
    width = 1500
    height = 470
    gap = 18
    panel_w = 464
    panel_h = 320
    origin_x = 30
    origin_y = 40
    panels = [
        (origin_x, origin_y, origin_x + panel_w, origin_y + panel_h),
        (origin_x + panel_w + gap, origin_y, origin_x + 2 * panel_w + gap, origin_y + panel_h),
        (origin_x + 2 * (panel_w + gap), origin_y, origin_x + 3 * panel_w + 2 * gap, origin_y + panel_h),
    ]

    all_x = [x for s in series_list for x in s['x']]
    x_min, x_max = min(all_x), max(all_x)
    x_ticks = nice_ticks(x_min, x_max, 6)
    iter_bounds = series_list[0]['iter_bounds_s'] if series_list else []
    if len(iter_bounds) >= 2:
        round3_start = iter_bounds[-2]
        tail_start = max(round3_start, x_max - 100.0)
    else:
        tail_start = max(x_min, x_max - 100.0)
    tail_end = x_max
    tail_ticks = nice_ticks(tail_start, tail_end, 5)

    est_all = [v for s in series_list for v in s['est']]
    full_ymin, full_ymax = robust_range(est_all, low_q=0.0, high_q=1.0, min_pad_frac=0.08)
    tail_vals = [v for s in series_list for x, v in zip(s['x'], s['est']) if x >= tail_start]
    tail_ymin, tail_ymax = robust_range(tail_vals, low_q=0.0, high_q=1.0, min_pad_frac=0.12)
    sigma_all = [max(v, 1e-12) for s in series_list for v in s['sigma']]
    sigma_min = min(sigma_all)
    sigma_max = max(sigma_all)
    sigma_ticks = log_ticks(sigma_min, sigma_max, 5)

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<style>text{font-family:Arial,Helvetica,sans-serif;} .title{font-size:22px;font-weight:bold;} .panel-title{font-size:15px;font-weight:bold;fill:#223;} .tick{font-size:11px;fill:#555;} .small{font-size:12px;fill:#444;} .axis{stroke:#333;stroke-width:1;} .grid{stroke:#e5e7eb;stroke-width:1;} .zero{stroke:#9aa0a6;stroke-width:1;stroke-dasharray:4 4;} .iter{stroke:#94a3b8;stroke-width:1;stroke-dasharray:6 4;} .legend{font-size:12px;fill:#222;}</style>')
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    parts.append(f'<text x="18" y="26" class="title">{escape(state_label)}</text>')

    def make_mappers(rect, xmin, xmax, ymin, ymax, log_y=False):
        x0, y0, x1, y1 = rect
        plot_top = y0 + 28
        plot_bottom = y1 - 36
        plot_left = x0 + 78
        plot_right = x1 - 18
        def mx(x):
            if math.isclose(xmin, xmax, abs_tol=1e-15):
                return (plot_left + plot_right) / 2
            return plot_left + (x - xmin) / (xmax - xmin) * (plot_right - plot_left)
        def my(y):
            return map_log(y, ymin, ymax, plot_top, plot_bottom) if log_y else map_linear(y, ymin, ymax, plot_top, plot_bottom)
        return mx, my

    mx1, my1 = make_mappers(panels[0], x_min, x_max, full_ymin, full_ymax, log_y=False)
    zero1 = my1(0.0) if full_ymin <= 0 <= full_ymax else None
    draw_panel(parts, panels[0], 'Estimate · first 3 rounds', x_ticks, nice_ticks(full_ymin, full_ymax, 5), mx1, my1, 'time (s)', unit, zero_y=zero1, iter_bounds=iter_bounds)

    mx2, my2 = make_mappers(panels[1], tail_start, tail_end, tail_ymin, tail_ymax, log_y=False)
    zero2 = my2(0.0) if tail_ymin <= 0 <= tail_ymax else None
    tail_iter = [b for b in iter_bounds if tail_start <= b <= tail_end]
    draw_panel(parts, panels[1], 'Estimate · round-3 last 100 s', tail_ticks, nice_ticks(tail_ymin, tail_ymax, 5), mx2, my2, 'time (s)', unit, zero_y=zero2, iter_bounds=tail_iter)

    mx3, my3 = make_mappers(panels[2], x_min, x_max, sigma_min, sigma_max, log_y=True)
    draw_panel(parts, panels[2], 'σ half-width · separate panel', x_ticks, sigma_ticks, mx3, my3, 'time (s)', f'σ ({unit})', zero_y=None, iter_bounds=iter_bounds)

    for series in series_list:
        style = line_style(series['group_key'])
        x1s, [y1s] = downsample(series['x'], [series['est']])
        pts1 = polyline([mx1(v) for v in x1s], [my1(v) for v in y1s])
        parts.append(f'<polyline points="{pts1}" fill="none" {style}/>')

        tail_x = [x for x in series['x'] if x >= tail_start]
        tail_y = [y for x, y in zip(series['x'], series['est']) if x >= tail_start]
        tail_x_ds, [tail_y_ds] = downsample(tail_x, [tail_y], max_points=800)
        pts2 = polyline([mx2(v) for v in tail_x_ds], [my2(v) for v in tail_y_ds])
        parts.append(f'<polyline points="{pts2}" fill="none" {style}/>')

        x3s, [y3s] = downsample(series['x'], [series['sigma']])
        pts3 = polyline([mx3(v) for v in x3s], [my3(max(v, sigma_min)) for v in y3s])
        parts.append(f'<polyline points="{pts3}" fill="none" {style}/>')

    legend_x = 22
    legend_y = 382
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="980" height="42" fill="white" stroke="#d0d7de"/>')
    for i, series in enumerate(series_list):
        x = legend_x + 18 + i * 310
        y = legend_y + 22
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" {line_style(series["group_key"], legend=True)}/>')
        parts.append(f'<text x="{x + 38}" y="{y + 4}" class="legend">{escape(series["label"])}</text>')

    parts.append('</svg>')
    out_svg.write_text('\n'.join(parts), encoding='utf-8')


def extract_series(group, state_label):
    info = group['state_by_label'].get(state_label)
    if info is None:
        return None
    idx = info['index']
    x_vals = [i * TRACE_DT_S for i in range(len(group['x_trace']))]
    est = [float(v[idx]) * info['scale'] for v in group['x_trace']]
    sigma = [math.sqrt(abs(float(v[idx]))) * info['scale'] for v in group['p_trace']]
    return {
        'group_key': group['group_key'],
        'label': group['label'],
        'state_label': state_label,
        'unit': info['unit'],
        'x': x_vals,
        'est': est,
        'sigma': sigma,
        'iter_bounds_s': [b * TRACE_DT_S for b in group['iter_bounds']],
    }


def discover_state_labels(groups):
    seen = set()
    labels = []
    for group in groups:
        for label in group['state_by_label'].keys():
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shared = build_shared_dual_dataset()
    groups = [
        trace_scale18_first3(shared),
        trace_plain24_first3(shared),
        trace_purescd24_first3(shared),
    ]

    state_labels = discover_state_labels(groups)
    summary = {
        'task': 'dualpath_three_method_state_convergence_v2_first3_2026_04_09',
        'style_note': 'same v2 first-3-round style as prior single-axis package; middle panel shows round-3 last 100 s',
        'path_note': 'old Chapter-4 dual-axis rotation strategy build_rot_paras()',
        'labels_only_note': 'legend labels were rewritten per user request; underlying methods remain scale-only / Markov / pure-SCD',
        'display_labels': GROUP_LABELS,
        'output_dir': str(OUT_DIR),
        'plots': [],
    }

    for state_label in state_labels:
        series_list = []
        unit = ''
        for group in groups:
            s = extract_series(group, state_label)
            if s is not None:
                series_list.append(s)
                unit = s['unit']
        svg_path = OUT_DIR / f'{state_label}_v2.svg'
        render_state_preview(svg_path, state_label, unit, series_list)
        summary['plots'].append({
            'state_label': state_label,
            'unit': unit,
            'svg': str(svg_path),
            'groups_present': [s['group_key'] for s in series_list],
        })

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
