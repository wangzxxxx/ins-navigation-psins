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
OUT_DIR = WORKSPACE / 'tmp' / 'psins_repeatability' / 'g4_custom_noise_mc20_2026-04-13' / 'style_previews'

FAMILY_KEY = 'eb'
FAMILY_TITLE = 'Gyro bias eb'
PANELS = [
    ('eb_x', 'x-axis', '#4C78A8'),
    ('eb_y', 'y-axis', '#F58518'),
    ('eb_z', 'z-axis', '#54A24B'),
    ('norm', 'norm', '#B279A2'),
]


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_panel_values(rows, key: str):
    if key == 'norm':
        return np.asarray([row['families'][FAMILY_KEY]['norm_est_display'] for row in rows], dtype=float)
    return np.asarray([row['families'][FAMILY_KEY]['components'][key]['est_display'] for row in rows], dtype=float)


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
    return mu, sigma


def fig_hist(rows):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), dpi=180)
    axes = axes.ravel()
    for ax, (key, title, color) in zip(axes, PANELS):
        values = get_panel_values(rows, key)
        bins = min(10, max(6, len(values) // 2 if len(values) >= 8 else len(values)))
        ax.hist(values, bins=bins, color=color, alpha=0.70, edgecolor='white', linewidth=1.0)
        mu, sigma = add_stats_box(ax, values)
        ax.axvline(mu, color='black', linewidth=1.2)
        if sigma > 0:
            ax.axvline(mu - sigma, color='#C44E52', linestyle='--', linewidth=1.0)
            ax.axvline(mu + sigma, color='#C44E52', linestyle='--', linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel('count')
        ax.grid(True, linestyle='--', alpha=0.20)
    fig.suptitle('Style A · Histogram panels', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_box(rows):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), dpi=180)
    axes = axes.ravel()
    for ax, (key, title, color) in zip(axes, PANELS):
        values = get_panel_values(rows, key)
        bp = ax.boxplot(values, vert=True, widths=0.45, patch_artist=True, showfliers=True)
        for box in bp['boxes']:
            box.set(facecolor=color, alpha=0.45, edgecolor=color)
        for med in bp['medians']:
            med.set(color='black', linewidth=1.2)
        for whisker in bp['whiskers']:
            whisker.set(color=color, linewidth=1.1)
        for cap in bp['caps']:
            cap.set(color=color, linewidth=1.1)
        add_stats_box(ax, values)
        ax.set_title(title)
        ax.set_ylabel('value')
        ax.set_xticks([1])
        ax.set_xticklabels(['box'])
        ax.grid(True, axis='y', linestyle='--', alpha=0.20)
    fig.suptitle('Style B · Boxplot panels', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_box_strip(rows):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), dpi=180)
    axes = axes.ravel()
    for ax, (key, title, color) in zip(axes, PANELS):
        values = get_panel_values(rows, key)
        bp = ax.boxplot(values, vert=True, widths=0.35, patch_artist=True, showfliers=False)
        for box in bp['boxes']:
            box.set(facecolor=color, alpha=0.25, edgecolor=color)
        for med in bp['medians']:
            med.set(color='black', linewidth=1.2)
        jitter = np.linspace(-0.10, 0.10, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(1 + jitter, values, color=color, alpha=0.75, s=18, zorder=3)
        add_stats_box(ax, values)
        ax.set_title(title)
        ax.set_ylabel('value')
        ax.set_xticks([1])
        ax.set_xticklabels(['box+points'])
        ax.grid(True, axis='y', linestyle='--', alpha=0.20)
    fig.suptitle('Style C · Boxplot + raw points', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_violin_strip(rows):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), dpi=180)
    axes = axes.ravel()
    for ax, (key, title, color) in zip(axes, PANELS):
        values = get_panel_values(rows, key)
        vp = ax.violinplot([values], positions=[1], widths=0.65, showmeans=False, showmedians=False, showextrema=False)
        for body in vp['bodies']:
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.30)
        jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
        ax.scatter(1 + jitter, values, color=color, alpha=0.72, s=16, zorder=3)
        add_stats_box(ax, values)
        ax.set_title(title)
        ax.set_ylabel('value')
        ax.set_xticks([1])
        ax.set_xticklabels(['violin+points'])
        ax.grid(True, axis='y', linestyle='--', alpha=0.20)
    fig.suptitle('Style D · Violin + raw points', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def fig_ecdf(rows):
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), dpi=180)
    axes = axes.ravel()
    for ax, (key, title, color) in zip(axes, PANELS):
        values = np.sort(get_panel_values(rows, key))
        y = np.arange(1, len(values) + 1) / len(values)
        mu, sigma = add_stats_box(ax, values)
        ax.step(values, y, where='post', color=color, linewidth=1.8)
        ax.scatter(values, y, color=color, s=12, alpha=0.70)
        ax.axvline(mu, color='black', linewidth=1.2)
        if sigma > 0:
            ax.axvline(mu - sigma, color='#C44E52', linestyle='--', linewidth=1.0)
            ax.axvline(mu + sigma, color='#C44E52', linestyle='--', linewidth=1.0)
        ax.set_title(title)
        ax.set_ylabel('ECDF')
        ax.set_xlabel('value')
        ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle='--', alpha=0.20)
    fig.suptitle('Style E · ECDF panels', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def save_fig(fig, stem: str):
    png = OUT_DIR / f'{stem}.png'
    svg = OUT_DIR / f'{stem}.svg'
    fig.savefig(png, format='png', bbox_inches='tight')
    fig.savefig(svg, format='svg', bbox_inches='tight')
    plt.close(fig)
    return {'png': str(png), 'svg': str(svg)}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = load_json(SUMMARY_JSON)
    rows = payload['per_seed']

    outputs = {
        'style_A_histogram': save_fig(fig_hist(rows), 'style_A_histogram'),
        'style_B_boxplot': save_fig(fig_box(rows), 'style_B_boxplot'),
        'style_C_boxplot_points': save_fig(fig_box_strip(rows), 'style_C_boxplot_points'),
        'style_D_violin_points': save_fig(fig_violin_strip(rows), 'style_D_violin_points'),
        'style_E_ecdf': save_fig(fig_ecdf(rows), 'style_E_ecdf'),
    }

    result = {
        'family_key': FAMILY_KEY,
        'family_title': FAMILY_TITLE,
        'outputs': outputs,
    }
    with open(OUT_DIR / 'result.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
