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
OUT_PNG = OUT_DIR / 'fig_dualpath_pure_scd_custom_noise_0p3_mc50_violin_2026-04-13.png'
OUT_SVG = OUT_DIR / 'fig_dualpath_pure_scd_custom_noise_0p3_mc50_violin_2026-04-13.svg'

PANEL_SPECS = [
    ('roll', 'final_roll_arcsec', '#4C78A8', 'signed error / arcsec'),
    ('pitch', 'final_pitch_arcsec', '#F58518', 'signed error / arcsec'),
    ('yaw', 'final_yaw_arcsec', '#54A24B', 'signed error / arcsec'),
    ('norm', 'final_att_err_norm_arcsec', '#B279A2', 'error norm / arcsec'),
]


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def make_panel(ax, values: np.ndarray, title: str, color: str, ylabel: str, nonnegative: bool = False):
    violin = ax.violinplot([values], positions=[1], widths=0.75, showmeans=False, showmedians=False, showextrema=False)
    for body in violin['bodies']:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.42)

    mu = float(values.mean())
    sigma = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    median = float(np.median(values))

    ax.scatter([1], [mu], color='black', s=28, zorder=3, label='mean')
    ax.scatter([1], [median], color='#C44E52', s=28, zorder=3, label='median')
    ax.errorbar([1], [mu], yerr=[sigma], fmt='none', ecolor='black', elinewidth=1.2, capsize=5, zorder=2)
    ax.axhline(0.0, color='gray', linestyle='--', linewidth=0.8)
    if nonnegative:
        ax.axhline(20.0, color='#C44E52', linestyle='--', linewidth=0.9, alpha=0.9)
        ax.set_ylim(bottom=0.0)

    jitter = np.linspace(-0.08, 0.08, len(values))
    ax.scatter(1 + jitter, values, s=12, color=color, alpha=0.65, zorder=3)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks([1])
    ax.set_xticklabels(['MC50'])
    ax.grid(True, axis='y', linestyle='--', alpha=0.25)
    ax.text(
        0.04,
        0.96,
        f'μ={mu:.3f}"\nσ={sigma:.3f}"\nmed={median:.3f}"',
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=8.8,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.78),
    )


def main() -> None:
    payload = load_json(RESULT_JSON)
    rows = payload['per_seed']

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.0), dpi=180)
    axes = axes.ravel()

    for ax, (name, key, color, ylabel) in zip(axes, PANEL_SPECS):
        values = np.array([row[key] for row in rows], dtype=float)
        make_panel(ax, values, f'{name} terminal distribution', color, ylabel, nonnegative=(name == 'norm'))

    fig.suptitle('Dual-axis pure-SCD24 @ noise ×0.3: MC50 violin preview', fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT_PNG, format='png', bbox_inches='tight')
    fig.savefig(OUT_SVG, format='svg', bbox_inches='tight')
    plt.close(fig)

    print(json.dumps({
        'json': str(RESULT_JSON),
        'png': str(OUT_PNG),
        'svg': str(OUT_SVG),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
