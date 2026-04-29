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
OUT_PNG = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'fig_ch4_staged24_freeze_rationale_merged_pub_2026-03-31.png'
OUT_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'ch4_staged24_freeze_rationale_merged_2026-03-31.json'


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


mod = load_module('obs_ch4_freeze_rationale_merged_20260331', OBS_SCRIPT)


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


def main():
    families = ['phi', 'dV', 'eb', 'db', 'ng', 'xa', 'kg', 'ka']
    primary_map = {
        'phi': [0, 1, 2],
        'dV': [3, 4, 5],
        'eb': [6, 7, 8],
        'db': [9, 10, 11],
        'ng': [12, 13, 14],
        'xa': [15, 16, 17],
    }

    W_freeze = compute_W('freeze')
    W_release = compute_W('gated_release')
    Wxx = W_release[:18, :18]
    Wxz = W_release[:18, 18:]
    Wzz = W_release[18:, 18:]
    W_cond = Wxx - Wxz @ np.linalg.inv(Wzz) @ Wxz.T

    diag_stage1_primary = np.diag(W_freeze[:18, :18])
    diag_stage2_primary = np.diag(W_cond)
    diag_stage2_scale = np.diag(Wzz)

    all_vals = np.r_[diag_stage1_primary, diag_stage2_primary, diag_stage2_scale]
    logs = np.log10(all_vals + 1e-30)
    lo, hi = float(np.min(logs)), float(np.max(logs))

    def norm(diag: np.ndarray) -> np.ndarray:
        return (np.log10(diag + 1e-30) - lo) / (hi - lo)

    s1_primary = norm(diag_stage1_primary)
    s2_primary = norm(diag_stage2_primary)
    s2_scale = norm(diag_stage2_scale)

    stage1 = []
    stage2 = []
    for fam in families:
        if fam in primary_map:
            idx = primary_map[fam]
            stage1.append(float(np.mean(s1_primary[idx])))
            stage2.append(float(np.mean(s2_primary[idx])))
        elif fam == 'kg':
            stage1.append(0.0)
            stage2.append(float(np.mean(s2_scale[0:3])))
        elif fam == 'ka':
            stage1.append(0.0)
            stage2.append(float(np.mean(s2_scale[3:6])))

    x = np.arange(len(families))
    w = 0.34
    fig, ax = plt.subplots(figsize=(12.4, 5.6))
    ax.bar(x - w / 2, stage1, width=w, label='Stage I: freeze dKg/dKa')
    ax.bar(x + w / 2, stage2, width=w, label='Stage II: release + high-rotation gate')
    ax.axvline(5.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
    ax.text(2.75, 1.035, 'primary states', ha='center', va='bottom', fontsize=10)
    ax.text(6.5, 1.035, 'scale states', ha='center', va='bottom', fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.set_ylabel('normalized index')
    ax.set_title('Why freezing scale states first is beneficial in staged 24-state estimation')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(frameon=False, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=240, bbox_inches='tight')
    plt.close(fig)

    payload = {
        'families': families,
        'stage1': stage1,
        'stage2': stage2,
        'cross_block_norm': float(np.linalg.norm(Wxz)),
        'primary_condition_number_stage1': float(np.linalg.cond(W_freeze[:18, :18])),
        'primary_condition_number_stage2_conditional': float(np.linalg.cond(W_cond)),
        'note': 'Primary families use conditional observability after Schur-complement elimination of scale states; kg/ka use released-stage activation, all unified into one normalized axis for visualization.',
        'figure': str(OUT_PNG),
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
