#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
RESULT_JSON = WORKSPACE / 'psins_method_bench' / 'results' / 'dualpath_pure_scd_custom_noise_0p3_mc50_2026-04-13.json'
OUT_DIR = WORKSPACE / 'tmp' / 'psins_repeatability'
OUT_PNG = OUT_DIR / 'fig_dualpath_pure_scd_custom_noise_0p3_sigma_gallery_2026-04-13.png'
OUT_SVG = OUT_DIR / 'fig_dualpath_pure_scd_custom_noise_0p3_sigma_gallery_2026-04-13.svg'


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main() -> None:
    payload = load_json(RESULT_JSON)
    rows = payload['per_seed']
    yaw = np.array([row['final_yaw_arcsec'] for row in rows], dtype=float)
    roll = np.array([row['final_roll_arcsec'] for row in rows], dtype=float)
    pitch = np.array([row['final_pitch_arcsec'] for row in rows], dtype=float)
    norm = np.array([row['final_att_err_norm_arcsec'] for row in rows], dtype=float)

    mu_yaw = float(yaw.mean())
    sigma_yaw = float(yaw.std(ddof=1)) if len(yaw) > 1 else 0.0
    median_yaw = float(np.median(yaw))

    labels = ['roll', 'pitch', 'yaw', 'norm']
    means = np.array([roll.mean(), pitch.mean(), yaw.mean(), norm.mean()], dtype=float)
    sigmas = np.array([
        roll.std(ddof=1),
        pitch.std(ddof=1),
        yaw.std(ddof=1),
        norm.std(ddof=1),
    ], dtype=float)
    colors = ['#4C78A8', '#F58518', '#54A24B', '#B279A2']

    idx = np.arange(1, len(yaw) + 1)
    yaw_sorted = np.sort(yaw)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.4, 8.2), dpi=180)
    axes = axes.ravel()

    # 1) Histogram + mu±sigma
    bins = max(10, min(18, len(yaw) // 3))
    axes[0].hist(yaw, bins=bins, color='#54A24B', alpha=0.75, edgecolor='white')
    axes[0].axvline(mu_yaw, color='black', linewidth=1.2, label='mean')
    axes[0].axvline(mu_yaw - sigma_yaw, color='#C44E52', linestyle='--', linewidth=1.0, label='±1σ')
    axes[0].axvline(mu_yaw + sigma_yaw, color='#C44E52', linestyle='--', linewidth=1.0)
    axes[0].axvline(median_yaw, color='#4C78A8', linestyle=':', linewidth=1.0, label='median')
    axes[0].axvspan(mu_yaw - sigma_yaw, mu_yaw + sigma_yaw, color='#C44E52', alpha=0.10)
    axes[0].set_title('A. Histogram + μ±1σ (yaw signed error)')
    axes[0].set_xlabel('arcsec')
    axes[0].set_ylabel('count')
    axes[0].grid(True, linestyle='--', alpha=0.25)
    axes[0].legend(frameon=False)

    # 2) Strip / jitter + mu±sigma band
    jitter = np.linspace(-0.12, 0.12, len(yaw))
    axes[1].scatter(1 + jitter, yaw, s=18, color='#54A24B', alpha=0.72)
    axes[1].axhspan(mu_yaw - sigma_yaw, mu_yaw + sigma_yaw, color='#C44E52', alpha=0.12, label='μ±1σ band')
    axes[1].axhline(mu_yaw, color='black', linewidth=1.2, label='mean')
    axes[1].axhline(median_yaw, color='#4C78A8', linestyle=':', linewidth=1.0, label='median')
    axes[1].set_xlim(0.7, 1.3)
    axes[1].set_xticks([1])
    axes[1].set_xticklabels(['MC50 yaw'])
    axes[1].set_ylabel('arcsec')
    axes[1].set_title('B. Sample strip + μ±1σ band')
    axes[1].grid(True, axis='y', linestyle='--', alpha=0.25)
    axes[1].legend(frameon=False)

    # 3) Sorted samples + mu±sigma band
    axes[2].plot(idx, yaw_sorted, marker='o', linewidth=1.1, markersize=3.0, color='#54A24B')
    axes[2].axhline(mu_yaw, color='black', linewidth=1.2, label='mean')
    axes[2].axhline(mu_yaw - sigma_yaw, color='#C44E52', linestyle='--', linewidth=1.0, label='±1σ')
    axes[2].axhline(mu_yaw + sigma_yaw, color='#C44E52', linestyle='--', linewidth=1.0)
    axes[2].fill_between(idx, mu_yaw - sigma_yaw, mu_yaw + sigma_yaw, color='#C44E52', alpha=0.10)
    axes[2].set_xlabel('sorted sample index')
    axes[2].set_ylabel('yaw signed error / arcsec')
    axes[2].set_title('C. Sorted samples + μ±1σ band')
    axes[2].grid(True, linestyle='--', alpha=0.25)
    axes[2].legend(frameon=False)

    # 4) Errorbar summary across axes / norm
    x = np.arange(len(labels))
    axes[3].errorbar(x, means, yerr=sigmas, fmt='o', color='black', ecolor='#C44E52', elinewidth=1.5, capsize=6)
    for xi, yi, c in zip(x, means, colors):
        axes[3].scatter([xi], [yi], s=60, color=c, zorder=3)
    axes[3].axhline(0.0, color='gray', linestyle='--', linewidth=0.8)
    axes[3].set_xticks(x)
    axes[3].set_xticklabels(labels)
    axes[3].set_ylabel('mean ± 1σ / arcsec')
    axes[3].set_title('D. Compact errorbar summary (best for comparison)')
    axes[3].grid(True, axis='y', linestyle='--', alpha=0.25)

    fig.suptitle('Which plot best shows 1σ? 0.3× noise, pure-SCD24, MC50 example', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_PNG, format='png', bbox_inches='tight')
    fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
    plt.close(fig)

    print(json.dumps({
        'json': str(RESULT_JSON),
        'png': str(OUT_PNG),
        'svg': str(OUT_SVG),
        'yaw_mean': mu_yaw,
        'yaw_sigma': sigma_yaw,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
