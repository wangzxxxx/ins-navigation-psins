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

OBS_SCRIPT = WORKSPACE / 'scripts' / 'analyze_ch4_staged24_observability_2026-03-30.py'
OBS_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'ch4_staged24_observability_2026-03-30.json'
OUT_PNG = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'fig_ch4_staged24_freeze_rationale_pub_2026-03-31.png'
OUT_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'ch4_staged24_freeze_rationale_2026-03-31.json'

PRIMARY_FAMILIES = ['phi', 'dV', 'eb', 'db', 'ng', 'xa']
SCALE_FAMILIES = ['kg', 'ka']


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = load_module('obs_ch4_freeze_rationale_20260331', OBS_SCRIPT)


def compute_W(mode: str) -> np.ndarray:
    ts, pos0_local, att_truth, imu = mod.build_nominal_data()
    mod.pos0 = pos0_local
    nn = 2
    nts = nn * ts
    eth = mod.acc18.Earth(pos0_local)
    cnn = mod.acc18.rv2m(-eth.wnie * nts / 2)
    scale_vec = mod.characteristic_scales()
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
    rot_gate_rad = 5.0 * mod.glv.deg

    W = np.zeros((24, 24))
    Psi = np.eye(24)
    length = (len(imu) // nn) * nn
    for k in range(0, length, nn):
        wvm = imu[k:k + nn, 0:6]
        phim, dvbm = mod.acc18.cnscl(wvm)
        att_k = att_truth[k, 0:3]
        cnb = mod.acc18.q2mat(mod.acc18.a2qua(att_k))
        dvn = cnn @ cnb @ dvbm
        if mode == 'freeze':
            scale_active = False
            high_rot = False
        elif mode == 'gated_release':
            scale_active = True
            high_rot = np.max(np.abs(phim / nts)) > rot_gate_rad
        else:
            raise ValueError(mode)
        phi_k = mod.build_phi_k(cnb, dvn, phim, dvbm, nts, fg, fa, scale_active=scale_active, high_rot=high_rot)
        Phi_tilde = Sinv @ phi_k @ S
        Psi = Phi_tilde @ Psi
        W += Psi.T @ Htilde.T @ Rinv @ Htilde @ Psi
    return W


def family_means_from_scores(scores: np.ndarray, family_map: dict[str, list[int]]) -> dict[str, float]:
    out = {}
    for fam, idxs in family_map.items():
        out[fam] = float(np.mean(scores[idxs]))
    return out


def main():
    with open(OBS_JSON, 'r', encoding='utf-8') as f:
        obs_payload = json.load(f)

    scale_direct = {
        'kg': obs_payload['family_summary']['kg']['stage2_mean'],
        'ka': obs_payload['family_summary']['ka']['stage2_mean'],
    }

    W_freeze = compute_W('freeze')
    W_release = compute_W('gated_release')

    Wxx = W_release[:18, :18]
    Wxz = W_release[:18, 18:]
    Wzz = W_release[18:, 18:]
    W_cond = Wxx - Wxz @ np.linalg.inv(Wzz) @ Wxz.T

    diag_stage1 = np.diag(W_freeze[:18, :18])
    diag_stage2 = np.diag(W_cond)
    all_logs = np.log10(np.r_[diag_stage1, diag_stage2] + 1e-30)
    lo = float(np.min(all_logs))
    hi = float(np.max(all_logs))

    def score(diag: np.ndarray) -> np.ndarray:
        return (np.log10(diag + 1e-30) - lo) / (hi - lo)

    s1 = score(diag_stage1)
    s2 = score(diag_stage2)

    family_map = {
        'phi': [0, 1, 2],
        'dV': [3, 4, 5],
        'eb': [6, 7, 8],
        'db': [9, 10, 11],
        'ng': [12, 13, 14],
        'xa': [15, 16, 17],
    }
    primary_stage1 = family_means_from_scores(s1, family_map)
    primary_stage2 = family_means_from_scores(s2, family_map)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2))
    plt.rcParams.update({'font.size': 12})

    # left panel: primary-state conditional observability
    ax = axes[0]
    x = np.arange(len(PRIMARY_FAMILIES))
    w = 0.34
    y1 = np.array([primary_stage1[f] for f in PRIMARY_FAMILIES])
    y2 = np.array([primary_stage2[f] for f in PRIMARY_FAMILIES])
    ax.bar(x - w / 2, y1, width=w, label='Stage I: freeze scale states')
    ax.bar(x + w / 2, y2, width=w, label='Stage II: conditional info after release')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(PRIMARY_FAMILIES)
    ax.set_ylabel('normalized conditional observability index')
    ax.set_title('Primary-state information after accounting for competition')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(frameon=False, loc='upper right')

    # right panel: scale-state activation
    ax = axes[1]
    x = np.arange(len(SCALE_FAMILIES))
    y1 = np.zeros(len(SCALE_FAMILIES))
    y2 = np.array([scale_direct[f] for f in SCALE_FAMILIES])
    ax.bar(x - w / 2, y1, width=w, label='Stage I: frozen')
    ax.bar(x + w / 2, y2, width=w, label='Stage II: released + gated')
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(SCALE_FAMILIES)
    ax.set_ylabel('normalized activation index')
    ax.set_title('Scale-state activation in the high-rotation stage')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(frameon=False, loc='upper left')

    fig.suptitle('Why freeze first and release later in staged 24-state estimation')
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=240, bbox_inches='tight')
    plt.close(fig)

    payload = {
        'primary_stage1': primary_stage1,
        'primary_stage2_conditional': primary_stage2,
        'scale_stage2_direct': scale_direct,
        'cross_block_norm': float(np.linalg.norm(Wxz)),
        'old_block_condition_freeze': float(np.linalg.cond(W_freeze[:18, :18])),
        'old_block_condition_after_conditional_release': float(np.linalg.cond(W_cond)),
        'figure': str(OUT_PNG),
        'note': 'Primary states are evaluated by conditional observability (Schur complement) after accounting for jointly estimated scale states; scale states use direct activation index.',
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
