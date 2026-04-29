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
OUT_SVG = OUT_DIR / 'fig_dualpath_pure_scd_custom_noise_0p3_sigma_histA_2026-04-13.svg'
OUT_PNG = OUT_DIR / 'fig_dualpath_pure_scd_custom_noise_0p3_sigma_histA_2026-04-13.png'


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main() -> None:
    payload = load_json(RESULT_JSON)
    rows = payload['per_seed']
    yaw = np.array([row['final_yaw_arcsec'] for row in rows], dtype=float)
    mu = float(yaw.mean())
    sigma = float(yaw.std(ddof=1)) if len(yaw) > 1 else 0.0
    median = float(np.median(yaw))
    bins = max(10, min(18, len(yaw) // 3))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.2, 4.8), dpi=180)
    ax.hist(yaw, bins=bins, color='#54A24B', alpha=0.75, edgecolor='white')
    ax.axvline(mu, color='black', linewidth=1.2, label='mean')
    ax.axvline(mu - sigma, color='#C44E52', linestyle='--', linewidth=1.0, label='±1σ')
    ax.axvline(mu + sigma, color='#C44E52', linestyle='--', linewidth=1.0)
    ax.axvline(median, color='#4C78A8', linestyle=':', linewidth=1.0, label='median')
    ax.axvspan(mu - sigma, mu + sigma, color='#C44E52', alpha=0.10)
    ax.set_title('Histogram + μ±1σ (yaw signed error, noise ×0.3)')
    ax.set_xlabel('arcsec')
    ax.set_ylabel('count')
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.03,
        0.95,
        f'μ={mu:.3f}"\nσ={sigma:.3f}"\nmed={median:.3f}"',
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=9,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.78),
    )
    fig.tight_layout()
    fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
    fig.savefig(OUT_PNG, format='png', bbox_inches='tight')
    plt.close(fig)

    print(json.dumps({
        'svg': str(OUT_SVG),
        'png': str(OUT_PNG),
        'mu': mu,
        'sigma': sigma,
        'median': median,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
