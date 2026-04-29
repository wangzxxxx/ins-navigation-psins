#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
CSV_PATH = Path('/root/.openclaw/media/inbound/repeatability_20position_hybrid_real22_direct100_runs---2ce819b6-9fb7-4342-85bd-a6197eaba714.csv')
OUT_DIR = WORKSPACE / 'tmp' / 'psins_repeatability' / 'real22_direct100_boxplot_points_rms_2026-04-13'

FAMILY_SPECS = [
    {
        'family_key': 'eb',
        'title': 'Gyro bias eb',
        'unit': 'dph',
        'components': ['eb_x', 'eb_y', 'eb_z'],
    },
    {
        'family_key': 'db',
        'title': 'Accelerometer bias db',
        'unit': 'ug',
        'components': ['db_x', 'db_y', 'db_z'],
    },
    {
        'family_key': 'dKg_diag',
        'title': 'Gyro scale diagonal dKg',
        'unit': 'ppm',
        'components': ['dKg_xx', 'dKg_yy', 'dKg_zz'],
    },
    {
        'family_key': 'dKg_offdiag',
        'title': 'Gyro off-diagonal / mounting dKg',
        'unit': 'sec',
        'components': ['dKg_yx', 'dKg_zx', 'dKg_xy', 'dKg_zy', 'dKg_xz', 'dKg_yz'],
    },
    {
        'family_key': 'dKa_diag',
        'title': 'Accel scale diagonal dKa',
        'unit': 'ppm',
        'components': ['dKa_xx', 'dKa_yy', 'dKa_zz'],
    },
    {
        'family_key': 'dKa_offdiag',
        'title': 'Accel off-diagonal / mounting dKa',
        'unit': 'sec',
        'components': ['dKa_xy', 'dKa_xz', 'dKa_yz'],
    },
    {
        'family_key': 'Ka2',
        'title': 'Second-order term Ka2',
        'unit': 'ug/g²',
        'components': ['Ka2_x', 'Ka2_y', 'Ka2_z'],
    },
    {
        'family_key': 'rx',
        'title': 'Lever-arm coeff rx',
        'unit': 'sec',
        'components': ['rx_x', 'rx_y', 'rx_z'],
    },
    {
        'family_key': 'ry',
        'title': 'Lever-arm coeff ry',
        'unit': 'sec',
        'components': ['ry_x', 'ry_y', 'ry_z'],
    },
]

PALETTE = ['#4C78A8', '#F58518', '#54A24B', '#72B7B2', '#B279A2', '#E45756', '#9D755D', '#BAB0AC']


def load_rows(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    parsed = []
    for row in rows:
        out = {}
        for k, v in row.items():
            if k in {'run_id', 'calib_id', 'timestamp', 'segment_id', 'note'}:
                out[k] = v
            elif k == 'success':
                out[k] = str(v).lower() == 'true'
            elif k == 'seed':
                out[k] = int(v)
            else:
                out[k] = float(v) if v not in (None, '', 'null', 'None') else None
        parsed.append(out)
    return parsed


def sample_std(values: np.ndarray) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else 0.0


def add_stats_box(ax, values: np.ndarray):
    mu = float(values.mean())
    sigma = sample_std(values)
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


def render_family(rows, spec: dict):
    panels = list(spec['components']) + ['rms']
    n_panels = len(panels)
    ncols = 2 if n_panels <= 4 else 4
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 3.8 * nrows), dpi=180)
    axes = np.atleast_1d(axes).ravel()

    panel_stats = {}
    for idx, panel in enumerate(panels):
        ax = axes[idx]
        color = PALETTE[idx % len(PALETTE)]
        if panel == 'rms':
            values = np.asarray([
                float(np.sqrt(np.mean([row[c] ** 2 for c in spec['components'] if row[c] is not None])))
                for row in rows
            ], dtype=float)
        else:
            values = np.asarray([row[panel] for row in rows if row[panel] is not None], dtype=float)
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
        ax.set_title(panel)
        ax.set_ylabel(spec['unit'])
        ax.set_xticks([1])
        ax.set_xticklabels(['box+points'])
        ax.grid(True, axis='y', linestyle='--', alpha=0.20)
        panel_stats[panel] = {
            'mean': float(values.mean()),
            'std': sample_std(values),
            'median': float(np.median(values)),
            'min': float(values.min()),
            'max': float(values.max()),
            'n': int(len(values)),
            'unit': spec['unit'],
        }

    for ax in axes[n_panels:]:
        ax.axis('off')

    fig.suptitle(f"{spec['title']} · real22 direct100 · boxplot + raw points (RMS)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    png_path = OUT_DIR / f"{spec['family_key']}_real22_boxplot_points_rms.png"
    svg_path = OUT_DIR / f"{spec['family_key']}_real22_boxplot_points_rms.svg"
    fig.savefig(png_path, format='png', bbox_inches='tight')
    fig.savefig(svg_path, format='svg', bbox_inches='tight')
    plt.close(fig)
    return {'png': str(png_path), 'svg': str(svg_path), 'stats': panel_stats}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_rows(CSV_PATH)
    outputs = {}
    for spec in FAMILY_SPECS:
        outputs[spec['family_key']] = render_family(rows, spec)

    overall = {
        'mean_abs_pct_error': {
            'mean': float(np.mean([r['mean_abs_pct_error'] for r in rows])),
            'std': sample_std(np.asarray([r['mean_abs_pct_error'] for r in rows], dtype=float)),
        },
        'median_abs_pct_error': {
            'mean': float(np.mean([r['median_abs_pct_error'] for r in rows])),
            'std': sample_std(np.asarray([r['median_abs_pct_error'] for r in rows], dtype=float)),
        },
        'max_abs_pct_error': {
            'mean': float(np.mean([r['max_abs_pct_error'] for r in rows])),
            'std': sample_std(np.asarray([r['max_abs_pct_error'] for r in rows], dtype=float)),
        },
    }
    result = {
        'csv_path': str(CSV_PATH),
        'n_runs': len(rows),
        'aggregate_panel': 'rms',
        'outputs': outputs,
        'overall': overall,
    }
    with open(OUT_DIR / 'result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
