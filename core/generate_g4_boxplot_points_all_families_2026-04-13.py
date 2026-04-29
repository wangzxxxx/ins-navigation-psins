#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
SUMMARY_JSON = WORKSPACE / 'tmp' / 'psins_repeatability' / 'g4_custom_noise_mc20_2026-04-13' / 'g4_custom_noise_mc20_summary.json'
OUT_DIR = WORKSPACE / 'tmp' / 'psins_repeatability' / 'g4_custom_noise_mc20_2026-04-13' / 'boxplot_points_all_families'

FAMILY_SPECS = [
    {
        'family_key': 'eb',
        'title': 'Gyro bias eb',
        'unit': 'dph',
        'params': ['eb_x', 'eb_y', 'eb_z'],
    },
    {
        'family_key': 'db',
        'title': 'Accelerometer bias db',
        'unit': 'ug',
        'params': ['db_x', 'db_y', 'db_z'],
    },
    {
        'family_key': 'dKg_diag',
        'title': 'Gyro scale diagonal dKg',
        'unit': 'ppm',
        'params': ['dKg_xx', 'dKg_yy', 'dKg_zz'],
    },
    {
        'family_key': 'dKg_offdiag',
        'title': 'Gyro off-diagonal / mounting dKg',
        'unit': 'sec',
        'params': ['dKg_yx', 'dKg_zx', 'dKg_xy', 'dKg_zy', 'dKg_xz', 'dKg_yz'],
    },
    {
        'family_key': 'dKa_diag',
        'title': 'Accel scale diagonal dKa',
        'unit': 'ppm',
        'params': ['dKa_xx', 'dKa_yy', 'dKa_zz'],
    },
    {
        'family_key': 'dKa_offdiag',
        'title': 'Accel off-diagonal / mounting dKa',
        'unit': 'sec',
        'params': ['dKa_xy', 'dKa_xz', 'dKa_yz'],
    },
    {
        'family_key': 'Ka2',
        'title': 'Second-order term Ka2',
        'unit': 'ug/g²',
        'params': ['Ka2_x', 'Ka2_y', 'Ka2_z'],
    },
    {
        'family_key': 'rx',
        'title': 'Lever-arm coeff rx',
        'unit': 'sec',
        'params': ['rx_x', 'rx_y', 'rx_z'],
    },
    {
        'family_key': 'ry',
        'title': 'Lever-arm coeff ry',
        'unit': 'sec',
        'params': ['ry_x', 'ry_y', 'ry_z'],
    },
]

PALETTE = ['#4C78A8', '#F58518', '#54A24B', '#72B7B2', '#B279A2', '#E45756', '#9D755D', '#BAB0AC']


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def add_stats_box(ax, values: np.ndarray):
    mu = float(values.mean())
    sigma = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    ax.text(
        0.04,
        0.96,
        f'μ={mu:.3f}\nσ={sigma:.3f}',
        transform=ax.transAxes,
        va='top',
        ha='left',
        fontsize=8.5,
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.82),
    )


def family_panel_values(rows, family_key: str, key: str):
    if key == 'norm':
        return np.asarray([row['families'][family_key]['norm_est_display'] for row in rows], dtype=float)
    return np.asarray([row['families'][family_key]['components'][key]['est_display'] for row in rows], dtype=float)


def render_family(rows, family_spec: dict):
    family_key = family_spec['family_key']
    panels = list(family_spec['params']) + ['norm']
    n_panels = len(panels)
    ncols = 2 if n_panels <= 4 else 4
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.8 * nrows), dpi=180)
    axes = np.atleast_1d(axes).ravel()

    for idx, panel_name in enumerate(panels):
        ax = axes[idx]
        color = PALETTE[idx % len(PALETTE)]
        values = family_panel_values(rows, family_key, panel_name)
        bp = ax.boxplot(values, vert=True, widths=0.35, patch_artist=True, showfliers=False)
        for box in bp['boxes']:
            box.set(facecolor=color, alpha=0.25, edgecolor=color)
        for med in bp['medians']:
            med.set(color='black', linewidth=1.2)
        for whisker in bp['whiskers']:
            whisker.set(color=color, linewidth=1.1)
        for cap in bp['caps']:
            cap.set(color=color, linewidth=1.1)
        jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(1 + jitter, values, color=color, alpha=0.75, s=18, zorder=3)
        add_stats_box(ax, values)
        ax.set_title('norm' if panel_name == 'norm' else panel_name)
        ax.set_ylabel(family_spec['unit'])
        ax.set_xticks([1])
        ax.set_xticklabels(['box+points'])
        ax.grid(True, axis='y', linestyle='--', alpha=0.20)

    for ax in axes[n_panels:]:
        ax.axis('off')

    fig.suptitle(f"{family_spec['title']} · boxplot + raw points", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = OUT_DIR / f"{family_key}_boxplot_points.png"
    svg_path = OUT_DIR / f"{family_key}_boxplot_points.svg"
    fig.savefig(png_path, format='png', bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    return {'png': str(png_path), 'svg': str(svg_path)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = load_json(SUMMARY_JSON)
    rows = payload['per_seed']
    outputs = {}
    for spec in FAMILY_SPECS:
        outputs[spec['family_key']] = render_family(rows, spec)
    result = {
        'style': 'boxplot_points',
        'outputs': outputs,
    }
    with open(OUT_DIR / 'result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
