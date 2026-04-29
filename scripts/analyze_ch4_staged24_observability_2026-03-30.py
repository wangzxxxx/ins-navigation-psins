#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
PSINS_ROOT = WORKSPACE / 'tmp_psins_py'
if str(PSINS_ROOT) not in sys.path:
    sys.path.insert(0, str(PSINS_ROOT))

ACC18_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_accel_colored_py_2026-03-30.py'
BASE12_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_12state_py_iterfix_2026-03-30.py'
OUT_DIR = WORKSPACE / 'tmp' / 'alignment_strategy_sweep'
OUT_JSON = OUT_DIR / 'ch4_staged24_observability_2026-03-30.json'
OUT_MD = OUT_DIR / 'ch4_staged24_observability_2026-03-30.md'
OUT_FIG = OUT_DIR / 'fig_ch4_staged24_observability_2026-03-30.svg'


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


acc18 = load_module('obs_acc18_20260330', ACC18_PATH)
base12 = load_module('obs_base12_20260330', BASE12_PATH)
glv = acc18.glv

STATE_NAMES = [
    'phi_E', 'phi_N', 'phi_U',
    'dV_E', 'dV_N', 'dV_U',
    'eb_x', 'eb_y', 'eb_z',
    'db_x', 'db_y', 'db_z',
    'ng_x', 'ng_y', 'ng_z',
    'xa_x', 'xa_y', 'xa_z',
    'kg_x', 'kg_y', 'kg_z',
    'ka_x', 'ka_y', 'ka_z',
]

FAMILY_MAP = {
    'phi': list(range(0, 3)),
    'dV': list(range(3, 6)),
    'eb': list(range(6, 9)),
    'db': list(range(9, 12)),
    'ng': list(range(12, 15)),
    'xa': list(range(15, 18)),
    'kg': list(range(18, 21)),
    'ka': list(range(21, 24)),
}


def characteristic_scales() -> np.ndarray:
    imuerr = base12.build_imuerr()
    phi0 = np.array([0.1, 0.1, 0.5]) * glv.deg
    init_eb = np.maximum(np.asarray(imuerr['eb']).reshape(3), 0.1 * glv.dph)
    init_db = np.maximum(np.asarray(imuerr['db']).reshape(3), 1000 * glv.ug)
    ng_sigma = np.array([0.05, 0.05, 0.05]) * glv.dph
    xa_sigma = np.maximum(np.array([0.01, 0.01, 0.01]) * glv.ug, 5.0 * glv.ug)
    scale_sigma = np.full(3, 100.0 * glv.ppm)
    return np.r_[phi0, np.ones(3), init_eb, init_db, ng_sigma, xa_sigma, scale_sigma, scale_sigma]


def build_nominal_data():
    ts = 0.01
    att0 = np.array([0.0, 0.0, 0.0])
    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    rot_paras = base12.build_rot_paras()
    att_truth = acc18.attrottt(att0, rot_paras, ts)
    imu, _ = acc18.avp2imu(att_truth, pos0)
    return ts, pos0, att_truth, imu


def build_phi_k(cnb: np.ndarray, dvn: np.ndarray, phim: np.ndarray, dvbm: np.ndarray, nts: float,
                fg: np.ndarray, fa: np.ndarray, scale_active: bool, high_rot: bool) -> np.ndarray:
    phi_k = np.eye(24)
    phi_k[0:3, 0:3] += acc18.askew(-acc18.Earth(pos0).wnie) * nts
    cnbts = cnb * nts
    phi_k[3:6, 0:3] = acc18.askew(dvn)
    phi_k[3:6, 3:6] = np.eye(3)
    phi_k[3:6, 9:12] = cnbts
    phi_k[3:6, 15:18] = cnbts
    phi_k[0:3, 6:9] = -cnbts
    phi_k[0:3, 12:15] = -cnbts
    phi_k[12:15, 12:15] = np.diag(fg)
    phi_k[15:18, 15:18] = np.diag(fa)
    if scale_active and high_rot:
        phi_k[0:3, 18:21] = -cnb @ np.diag(phim[0:3])
        phi_k[3:6, 21:24] = cnb @ np.diag(dvbm[0:3])
    return phi_k


def compute_observability(stage: str) -> dict:
    ts, pos0_local, att_truth, imu = build_nominal_data()
    nn = 2
    nts = nn * ts
    eth = acc18.Earth(pos0_local)
    cnn = acc18.rv2m(-eth.wnie * nts / 2)
    scale_vec = characteristic_scales()
    S = np.diag(scale_vec)
    Sinv = np.diag(1.0 / scale_vec)
    R = np.diag(np.array([0.01, 0.01, 0.01])) ** 2 / nts
    Rinv = np.linalg.inv(R)
    H = np.hstack([np.zeros((3, 3)), np.eye(3), np.zeros((3, 18))])
    Htilde = H @ S
    tau_g = np.array([300.0, 300.0, 300.0])
    tau_a = np.array([100.0, 100.0, 100.0])
    fg = np.exp(-nts / tau_g)
    fa = np.exp(-nts / tau_a)
    rot_gate_rad = 5.0 * glv.deg

    W = np.zeros((24, 24))
    Psi = np.eye(24)

    length = (len(imu) // nn) * nn
    for k in range(0, length, nn):
        wvm = imu[k:k + nn, 0:6]
        phim, dvbm = acc18.cnscl(wvm)
        att_k = att_truth[k, 0:3]
        cnb = acc18.q2mat(acc18.a2qua(att_k))
        dvn = cnn @ cnb @ dvbm

        scale_active = (stage == 'stage2')
        high_rot = np.max(np.abs(phim / nts)) > rot_gate_rad
        phi_k = build_phi_k(cnb, dvn, phim, dvbm, nts, fg, fa, scale_active=scale_active, high_rot=high_rot)
        Phi_tilde = Sinv @ phi_k @ S
        Psi = Phi_tilde @ Psi
        W += Psi.T @ Htilde.T @ Rinv @ Htilde @ Psi

    diagW = np.diag(W)
    return {
        'stage': stage,
        'diag': diagW.tolist(),
    }


def normalize_scores(diag1: np.ndarray, diag2: np.ndarray):
    eps = 1e-30
    logs = np.log10(np.r_[diag1 + eps, diag2 + eps])
    lo = np.min(logs)
    hi = np.max(logs)
    if hi - lo < 1e-12:
        s1 = np.ones_like(diag1)
        s2 = np.ones_like(diag2)
    else:
        s1 = (np.log10(diag1 + eps) - lo) / (hi - lo)
        s2 = (np.log10(diag2 + eps) - lo) / (hi - lo)
    return s1, s2, {'log_min': float(lo), 'log_max': float(hi)}


def qualitative(score: float) -> str:
    if score >= 0.75:
        return '强'
    if score >= 0.45:
        return '中'
    return '弱'


def make_figure(stage1_scores: np.ndarray, stage2_scores: np.ndarray):
    x = np.arange(len(STATE_NAMES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.bar(x - w / 2, stage1_scores, width=w, label='阶段 I：冻结 dKg/dKa')
    ax.bar(x + w / 2, stage2_scores, width=w, label='阶段 II：释放并门控 dKg/dKa')
    ax.set_ylabel('normalized observability index')
    ax.set_xticks(x)
    ax.set_xticklabels(STATE_NAMES, rotation=60, ha='right', fontsize=8)
    ax.set_title('State-wise observability under staged24 strategy')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_FIG, format='svg')
    plt.close(fig)


def main():
    global pos0
    ts, pos0, att_truth, imu = build_nominal_data()
    stage1 = compute_observability('stage1')
    stage2 = compute_observability('stage2')
    diag1 = np.array(stage1['diag'])
    diag2 = np.array(stage2['diag'])
    s1, s2, meta = normalize_scores(diag1, diag2)

    families = {}
    for family, idxs in FAMILY_MAP.items():
        families[family] = {
            'stage1_mean': float(np.mean(s1[idxs])),
            'stage2_mean': float(np.mean(s2[idxs])),
        }

    per_state = []
    for i, name in enumerate(STATE_NAMES):
        per_state.append({
            'state': name,
            'stage1_score': float(s1[i]),
            'stage2_score': float(s2[i]),
            'gain': float(s2[i] - s1[i]),
            'stage1_level': qualitative(float(s1[i])),
            'stage2_level': qualitative(float(s2[i])),
        })

    make_figure(s1, s2)

    summary_notes = {
        'stage1': '阶段 I 中主姿态、速度和等效常值/随机误差状态为主要可观对象，kg/ka 基本被冻结。',
        'stage2': '阶段 II 在保持前述主状态可观性的同时，使对角 scale-factor 状态获得有效观测通道。',
        'core_judgement': 'staged24 的关键不是简单增加状态，而是把 scale states 的可观测性建立在前期姿态锁定之后。',
    }

    payload = {
        'method': 'staged24_observability_analysis',
        'trajectory_note': 'same DAR path and timing family as current staged24 experiments; nominal truth / no MC noise',
        'normalization': 'normalized by characteristic state scales consistent with filter initialization, then log-compressed to [0,1]',
        'state_names': STATE_NAMES,
        'per_state': per_state,
        'family_summary': families,
        'summary_notes': summary_notes,
        'figure': str(OUT_FIG),
        'meta': meta,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    lines = [
        '# Staged24 observability analysis',
        '',
        '- Method: finite-horizon normalized observability Gramian under the current staged24 trajectory',
        '- Stage I: freeze `dKg/dKa`',
        '- Stage II: release `dKg/dKa` with high-rotation gating',
        '',
        '## Family summary',
        '',
        '| family | stage I mean score | stage II mean score |',
        '|---|---:|---:|',
    ]
    for family, item in families.items():
        lines.append(f"| {family} | {item['stage1_mean']:.3f} | {item['stage2_mean']:.3f} |")
    lines.extend([
        '',
        '## Notes',
        f"- {summary_notes['stage1']}",
        f"- {summary_notes['stage2']}",
        f"- {summary_notes['core_judgement']}",
        '',
        f'- figure: `{OUT_FIG}`',
    ])
    OUT_MD.write_text('\n'.join(lines) + '\n')
    print(json.dumps({
        'out_json': str(OUT_JSON),
        'out_md': str(OUT_MD),
        'out_fig': str(OUT_FIG),
        'family_summary': families,
        'top_state_gains': sorted(per_state, key=lambda x: x['gain'], reverse=True)[:8],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
