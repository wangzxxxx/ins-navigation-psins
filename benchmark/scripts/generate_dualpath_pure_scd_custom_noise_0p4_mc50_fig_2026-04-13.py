#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
RESULT_JSON = WORKSPACE / 'psins_method_bench' / 'results' / 'dualpath_pure_scd_custom_noise_0p4_mc50_2026-04-13.json'
OUT_DIR = WORKSPACE / 'tmp' / 'psins_repeatability'
OUT_PNG = OUT_DIR / 'fig_dualpath_pure_scd_custom_noise_0p4_mc50_2026-04-13.png'
OUT_SVG = OUT_DIR / 'fig_dualpath_pure_scd_custom_noise_0p4_mc50_2026-04-13.svg'


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def make_figure(payload: dict):
    rows = payload['per_seed']
    roll_signed = np.array([row['final_roll_arcsec'] for row in rows], dtype=float)
    pitch_signed = np.array([row['final_pitch_arcsec'] for row in rows], dtype=float)
    yaw_abs = np.array([row['final_yaw_abs_arcsec'] for row in rows], dtype=float)
    norm_abs = np.array([row['final_att_err_norm_arcsec'] for row in rows], dtype=float)

    roll_sorted = np.sort(roll_signed)
    pitch_sorted = np.sort(pitch_signed)
    yaw_sorted = np.sort(yaw_abs)
    norm_sorted = np.sort(norm_abs)

    idx_roll = np.arange(1, len(roll_sorted) + 1)
    idx_pitch = np.arange(1, len(pitch_sorted) + 1)
    idx_yaw = np.arange(1, len(yaw_sorted) + 1)
    idx_norm = np.arange(1, len(norm_sorted) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(11.2, 8.2), dpi=180)
    axes = axes.ravel()

    axes[0].plot(idx_roll, roll_sorted, marker='o', linewidth=1.15, markersize=3.2, color='#1f77b4')
    axes[0].axhline(0.0, color='k', linewidth=0.8, linestyle='--')
    axes[0].set_xlabel('sorted sample index')
    axes[0].set_ylabel('roll signed error / arcsec')
    axes[0].set_title('pure-SCD24 MC50 roll stability (noise ×0.4)')
    axes[0].grid(True, linestyle='--', alpha=0.35)

    axes[1].plot(idx_pitch, pitch_sorted, marker='o', linewidth=1.15, markersize=3.2, color='#1f77b4')
    axes[1].axhline(0.0, color='k', linewidth=0.8, linestyle='--')
    axes[1].set_xlabel('sorted sample index')
    axes[1].set_ylabel('pitch signed error / arcsec')
    axes[1].set_title('pure-SCD24 MC50 pitch stability (noise ×0.4)')
    axes[1].grid(True, linestyle='--', alpha=0.35)

    axes[2].plot(idx_yaw, yaw_sorted, marker='o', linewidth=1.15, markersize=3.2, color='#1f77b4')
    axes[2].axhline(20.0, color='#c44e52', linewidth=0.9, linestyle='--', label='20 arcsec target')
    axes[2].set_xlabel('sorted sample index')
    axes[2].set_ylabel('yaw absolute error / arcsec')
    axes[2].set_title('pure-SCD24 MC50 yaw tail (noise ×0.4)')
    axes[2].grid(True, linestyle='--', alpha=0.35)
    axes[2].legend(frameon=False, loc='upper left')

    axes[3].plot(idx_norm, norm_sorted, marker='o', linewidth=1.15, markersize=3.2, color='#1f77b4')
    axes[3].axhline(20.0, color='#c44e52', linewidth=0.9, linestyle='--', label='20 arcsec target')
    axes[3].set_xlabel('sorted sample index')
    axes[3].set_ylabel('attitude error norm / arcsec')
    axes[3].set_title('pure-SCD24 MC50 norm distribution (noise ×0.4)')
    axes[3].grid(True, linestyle='--', alpha=0.35)
    axes[3].legend(frameon=False, loc='upper left')

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, format='png', bbox_inches='tight')
    fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    payload = load_json(RESULT_JSON)
    make_figure(payload)
    print(json.dumps({
        'json': str(RESULT_JSON),
        'png': str(OUT_PNG),
        'svg': str(OUT_SVG),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
