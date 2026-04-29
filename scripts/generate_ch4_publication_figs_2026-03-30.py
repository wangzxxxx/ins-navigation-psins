#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
OUT_DIR = WORKSPACE / 'tmp' / 'alignment_strategy_sweep'
OUT_DIR.mkdir(parents=True, exist_ok=True)

STAGE2_PATH = OUT_DIR / 'stage2_results_2026-03-29.json'
BASELINE_PATH = OUT_DIR / 'alignvn_dar_hybrid24_pitch_repair_probe_2026-03-30.json'
STAGED5_PATH = OUT_DIR / 'alignvn_dar_hybrid24_staged_result_2026-03-30.json'
MC50_PATH = OUT_DIR / 'alignvn_dar_hybrid24_staged_mc50_result_2026-03-30.json'

FIG01 = OUT_DIR / 'fig_ch4_rotation_modulation_principle_2026-03-30.png'
FIG02 = OUT_DIR / 'fig_ch4_top3_mc_compare_pub_2026-03-30.png'
FIG03 = OUT_DIR / 'fig_ch4_top3_mean_curve_pub_2026-03-30.png'
FIG04 = OUT_DIR / 'fig_ch4_timing_recheck_pub_2026-03-30.png'
FIG05 = OUT_DIR / 'fig_ch4_staged24_observability_pub_2026-03-30.png'
FIG07 = OUT_DIR / 'fig_ch4_model_evolution_compare_pub_2026-03-30.png'
FIG08 = OUT_DIR / 'fig_ch4_plain24_vs_staged24_iter_curve_pub_2026-03-30.png'
FIG09 = OUT_DIR / 'fig_ch4_staged24_mc50_distribution_pub_2026-03-30.png'

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 15,
})


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_fig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=240, bbox_inches='tight')
    plt.close(fig)


def make_fig01_modulation_principle():
    t = np.linspace(0, 2 * np.pi, 400)
    eps_x = 1.0
    eps_y = 0.6
    static_x = np.full_like(t, eps_x)
    static_y = np.full_like(t, eps_y)
    rot_x = eps_x * np.cos(t) - eps_y * np.sin(t)
    rot_y = eps_x * np.sin(t) + eps_y * np.cos(t)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    ax = axes[0]
    ax.plot(t, static_x, linewidth=2.2, label=r'$\varepsilon_x$ (constant)')
    ax.plot(t, static_y, linewidth=2.2, label=r'$\varepsilon_y$ (constant)')
    ax.set_title('Static frame: constant sensor bias')
    ax.set_xlabel('modulation phase / rad')
    ax.set_ylabel('projected bias (normalized)')
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(frameon=False, loc='upper right')

    ax = axes[1]
    ax.plot(t, rot_x, linewidth=2.2, label=r'$\varepsilon_x^n(t)$')
    ax.plot(t, rot_y, linewidth=2.2, label=r'$\varepsilon_y^n(t)$')
    ax.axhline(0.0, color='k', linewidth=1.0, linestyle='--')
    ax.fill_between(t, rot_x, 0, alpha=0.12)
    ax.set_title('Rotated frame: zero-mean periodic projection')
    ax.set_xlabel('modulation phase / rad')
    ax.set_ylabel('projected bias (normalized)')
    ax.set_xticks([0, np.pi, 2 * np.pi])
    ax.set_xticklabels(['0', r'$\pi$', r'$2\pi$'])
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(frameon=False, loc='upper right')

    fig.suptitle('Principle of rotation modulation for constant-bias suppression')
    save_fig(fig, FIG01)


def make_fig02_top3_bars(stage2: dict):
    top3 = stage2['validated_top3']
    labels = ['\n'.join(['→'.join(item['sequence'][:3]), '→'.join(item['sequence'][3:])]) for item in top3]
    cov_sigma = np.array([item['final_yaw_sigma_arcsec'] for item in top3])
    mc_mean = np.array([item['monte_carlo']['mean_final_yaw_err_arcsec'] for item in top3])
    mc_p95 = np.array([item['monte_carlo']['p95_final_yaw_err_arcsec'] for item in top3])

    x = np.arange(len(labels))
    w = 0.24
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.bar(x - w, cov_sigma, width=w, label='covariance final yaw σ')
    ax.bar(x, mc_mean, width=w, label='MC mean final yaw error')
    ax.bar(x + w, mc_p95, width=w, label='MC p95 final yaw error')
    ax.set_xlabel('candidate dual-axis sequence')
    ax.set_ylabel('yaw metric / arcsec')
    ax.set_title('Top-3 candidate sequences: covariance vs Monte Carlo statistics')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(frameon=False, ncol=3, loc='upper right')
    save_fig(fig, FIG02)


def make_fig03_top3_curves(stage2: dict):
    top3 = stage2['validated_top3']
    fig, ax = plt.subplots(figsize=(12.0, 5.6))
    colors = ['#4e79a7', '#f28e2b', '#59a14f']
    for c, item in zip(colors, top3):
        t = np.array(item['monte_carlo']['time_s'])
        y = np.array(item['monte_carlo']['mean_yaw_curve_arcsec'])
        label = '→'.join(item['sequence'])
        ax.plot(t, y, linewidth=2.0, color=c, label=label)
    ax.axhline(20.0, color='r', linestyle='--', linewidth=1.0, label='20 arcsec target')
    ax.set_xlabel('alignment time / s')
    ax.set_ylabel('MC mean yaw absolute error / arcsec')
    ax.set_title('Mean yaw convergence curves of the top-3 candidate sequences')
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(frameon=False, loc='upper right')
    save_fig(fig, FIG03)


def make_fig04_timing_recheck():
    labels = [
        'Z+→Y+→Z-→Y-→Z+→Y-\n10.5/2.0/32.5',
        'Y+→Y-→Z+→Z-→Y+→Z-\n9.0/3.0/33.0',
        'Y+→Y-→Z+→Z-→Y+→Z-\n7.5/1.0/36.5',
    ]
    mean_vals = np.array([550.97, 524.39, 628.53])
    p95_vals = np.array([935.97, 1023.64, 977.21])
    x = np.arange(len(labels))
    w = 0.32

    fig, ax = plt.subplots(figsize=(12.0, 5.8))
    ax.bar(x - w / 2, mean_vals, width=w, label='MC mean final yaw error')
    ax.bar(x + w / 2, p95_vals, width=w, label='MC p95 final yaw error')
    ax.set_xlabel('family and local timing allocation (rotate/pre/post, s)')
    ax.set_ylabel('yaw metric / arcsec')
    ax.set_title('Local timing re-check of the two leading sequence families')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(frameon=False, loc='upper right')
    save_fig(fig, FIG04)


def make_fig05_observability():
    families = ['phi', 'dV', 'eb', 'db', 'ng', 'xa', 'kg', 'ka']
    stage1 = np.array([0.975, 0.971, 0.874, 0.961, 0.850, 0.816, 0.000, 0.000])
    stage2 = np.array([0.975, 0.971, 0.874, 0.961, 0.850, 0.816, 0.888, 0.873])
    x = np.arange(len(families))
    w = 0.34

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.bar(x - w / 2, stage1, width=w, label='Stage I: freeze dKg/dKa')
    ax.bar(x + w / 2, stage2, width=w, label='Stage II: release + gate dKg/dKa')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('state family')
    ax.set_ylabel('normalized observability index')
    ax.set_title('Normalized observability index under the staged 24-state strategy')
    ax.set_xticks(x)
    ax.set_xticklabels(families)
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(frameon=False, loc='upper left')
    save_fig(fig, FIG05)


def make_fig07_model_evolution(baseline: dict, staged5: dict):
    items = [
        ('18-state\niter=1', baseline['baseline18_iter1']['pitch_mean_abs'], baseline['baseline18_iter1']['yaw_abs_mean'], baseline['baseline18_iter1']['norm_mean']),
        ('18-state\niter=5', baseline['baseline18_iter5']['pitch_mean_abs'], baseline['baseline18_iter5']['yaw_abs_mean'], baseline['baseline18_iter5']['norm_mean']),
        ('hybrid24\niter=1', baseline['hybrid24_iter1']['pitch_mean_abs'], baseline['hybrid24_iter1']['yaw_abs_mean'], baseline['hybrid24_iter1']['norm_mean']),
        ('staged24\niter=5', staged5['staged24_iter5']['statistics']['mean_abs_arcsec'][1], staged5['staged24_iter5']['statistics']['yaw_abs_mean_arcsec'], staged5['staged24_iter5']['statistics']['norm_mean_arcsec']),
    ]
    labels = [x[0] for x in items]
    pitch = np.array([x[1] for x in items])
    yaw = np.array([x[2] for x in items])
    norm = np.array([x[3] for x in items])
    x = np.arange(len(labels))
    w = 0.24

    fig, ax = plt.subplots(figsize=(11.2, 5.4))
    ax.bar(x - w, pitch, width=w, label='pitch mean |error|')
    ax.bar(x, yaw, width=w, label='yaw mean |error|')
    ax.bar(x + w, norm, width=w, label='mean norm error')
    ax.set_ylabel('error / arcsec')
    ax.set_xlabel('model configuration')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('Representative statistics during model evolution under the same DAR truth')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(frameon=False, ncol=3, loc='upper right')
    save_fig(fig, FIG07)


def make_fig08_iter_curves(staged5: dict):
    plain_logs = staged5['plain24_iter5']['per_seed'][0]['iter_logs']
    staged_logs = staged5['staged24_iter5']['per_seed'][0]['iter_logs']
    it_plain = [x['iteration'] for x in plain_logs]
    it_staged = [x['iteration'] for x in staged_logs]
    pitch_plain = [x['att_err_arcsec'][1] for x in plain_logs]
    pitch_staged = [x['att_err_arcsec'][1] for x in staged_logs]
    yaw_plain = [abs(x['att_err_arcsec'][2]) for x in plain_logs]
    yaw_staged = [abs(x['att_err_arcsec'][2]) for x in staged_logs]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    axes[0].plot(it_plain, pitch_plain, marker='o', linewidth=2.0, label='plain24')
    axes[0].plot(it_staged, pitch_staged, marker='s', linewidth=2.0, label='staged24')
    axes[0].axhline(0.0, color='k', linewidth=1.0, linestyle='--')
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('pitch signed error / arcsec')
    axes[0].set_title('Seed0 pitch evolution')
    axes[0].grid(True, linestyle='--', alpha=0.35)
    axes[0].legend(frameon=False)

    axes[1].plot(it_plain, yaw_plain, marker='o', linewidth=2.0, label='plain24')
    axes[1].plot(it_staged, yaw_staged, marker='s', linewidth=2.0, label='staged24')
    axes[1].axhline(20.0, color='r', linewidth=1.0, linestyle='--', label='20 arcsec target')
    axes[1].set_xlabel('iteration')
    axes[1].set_ylabel('yaw absolute error / arcsec')
    axes[1].set_title('Seed0 yaw evolution')
    axes[1].grid(True, linestyle='--', alpha=0.35)
    axes[1].legend(frameon=False)

    save_fig(fig, FIG08)


def make_fig09_mc50_distribution(mc50: dict):
    per_seed = mc50['per_seed']
    yaw_abs = np.array([row['final_yaw_abs_arcsec'] for row in per_seed])
    pitch = np.array([row['final_att_err_arcsec'][1] for row in per_seed])

    yaw_sorted = np.sort(yaw_abs)
    pitch_sorted = np.sort(pitch)
    idx_yaw = np.arange(1, len(yaw_sorted) + 1)
    idx_pitch = np.arange(1, len(pitch_sorted) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    axes[0].plot(idx_yaw, yaw_sorted, marker='o', linewidth=1.6, markersize=4)
    axes[0].axhline(20.0, color='r', linewidth=1.0, linestyle='--', label='20 arcsec target')
    axes[0].set_xlabel('sorted sample index')
    axes[0].set_ylabel('yaw absolute error / arcsec')
    axes[0].set_title('staged24 MC50 yaw tail')
    axes[0].grid(True, linestyle='--', alpha=0.35)
    axes[0].legend(frameon=False)

    axes[1].plot(idx_pitch, pitch_sorted, marker='o', linewidth=1.6, markersize=4)
    axes[1].axhline(0.0, color='k', linewidth=1.0, linestyle='--')
    axes[1].set_xlabel('sorted sample index')
    axes[1].set_ylabel('pitch signed error / arcsec')
    axes[1].set_title('staged24 MC50 pitch stability')
    axes[1].grid(True, linestyle='--', alpha=0.35)

    save_fig(fig, FIG09)


def main():
    stage2 = load_json(STAGE2_PATH)
    baseline = load_json(BASELINE_PATH)
    staged5 = load_json(STAGED5_PATH)
    mc50 = load_json(MC50_PATH)

    make_fig01_modulation_principle()
    make_fig02_top3_bars(stage2)
    make_fig03_top3_curves(stage2)
    make_fig04_timing_recheck()
    make_fig05_observability()
    make_fig07_model_evolution(baseline, staged5)
    make_fig08_iter_curves(staged5)
    make_fig09_mc50_distribution(mc50)

    print(json.dumps({
        'fig01': str(FIG01),
        'fig02': str(FIG02),
        'fig03': str(FIG03),
        'fig04': str(FIG04),
        'fig05': str(FIG05),
        'fig07': str(FIG07),
        'fig08': str(FIG08),
        'fig09': str(FIG09),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
