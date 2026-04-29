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
OUT_PNG = OUT_DIR / 'fig_dualpath_pure_scd_custom_noise_0p3_mc50_sigma_2026-04-13.png'
OUT_SVG = OUT_DIR / 'fig_dualpath_pure_scd_custom_noise_0p3_mc50_sigma_2026-04-13.svg'

AXIS_INFO = [
    ('roll', 'final_roll_arcsec', '#4C78A8'),
    ('pitch', 'final_pitch_arcsec', '#F58518'),
    ('yaw', 'final_yaw_arcsec', '#54A24B'),
]


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def add_hist(ax, values: np.ndarray, name: str, color: str):
    mu = float(values.mean())
    sigma = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    bins = max(10, min(18, len(values) // 3))
    ax.hist(values, bins=bins, color=color, alpha=0.72, edgecolor='white')
    ax.axvline(mu, color='black', linestyle='-', linewidth=1.2, label='mean')
    ax.axvline(mu - sigma, color='#C44E52', linestyle='--', linewidth=1.0, label='±1σ')
    ax.axvline(mu + sigma, color='#C44E52', linestyle='--', linewidth=1.0)
    ax.axvspan(mu - sigma, mu + sigma, color='#C44E52', alpha=0.10)
    ax.set_title(f'{name} terminal signed error')
    ax.set_xlabel('arcsec')
    ax.set_ylabel('count')
    ax.grid(True, linestyle='--', alpha=0.25)
    ax.text(
        0.03,
        0.95,
        f'μ={mu:.3f}"\nσ={sigma:.3f}"',
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=9,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.75),
    )


def make_figure(payload: dict):
    rows = payload['per_seed']
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), dpi=180)
    axes = axes.ravel()

    for ax, (name, key, color) in zip(axes[:3], AXIS_INFO):
        values = np.array([row[key] for row in rows], dtype=float)
        add_hist(ax, values, name, color)

    norm_values = np.array([row['final_att_err_norm_arcsec'] for row in rows], dtype=float)
    norm_mu = float(norm_values.mean())
    norm_sigma = float(norm_values.std(ddof=1)) if len(norm_values) > 1 else 0.0
    norm_bins = max(10, min(18, len(norm_values) // 3))
    axes[3].hist(norm_values, bins=norm_bins, color='#72B7B2', alpha=0.75, edgecolor='white')
    axes[3].axvline(norm_mu, color='black', linestyle='-', linewidth=1.2, label='mean')
    axes[3].axvline(norm_mu - norm_sigma, color='#C44E52', linestyle='--', linewidth=1.0, label='±1σ')
    axes[3].axvline(norm_mu + norm_sigma, color='#C44E52', linestyle='--', linewidth=1.0)
    axes[3].axvspan(max(0.0, norm_mu - norm_sigma), norm_mu + norm_sigma, color='#C44E52', alpha=0.10)
    axes[3].set_title('norm terminal error distribution')
    axes[3].set_xlabel('arcsec')
    axes[3].set_ylabel('count')
    axes[3].grid(True, linestyle='--', alpha=0.25)
    axes[3].legend(frameon=False)
    axes[3].text(
        0.03,
        0.95,
        f'μ={norm_mu:.3f}"\nσ={norm_sigma:.3f}"',
        transform=axes[3].transAxes,
        va='top',
        ha='left',
        fontsize=9,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.75),
    )

    fig.suptitle('Dual-axis pure-SCD24 @ noise ×0.3: MC50 signed-error and norm distributions', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
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
