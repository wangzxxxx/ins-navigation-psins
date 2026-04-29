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
BASELINE_PATH = OUT_DIR / 'alignvn_dar_hybrid24_pitch_repair_probe_2026-03-30.json'
STAGED5_PATH = OUT_DIR / 'alignvn_dar_hybrid24_staged_result_2026-03-30.json'
MC50_PATH = OUT_DIR / 'alignvn_dar_hybrid24_staged_mc50_result_2026-03-30.json'

FIG1 = OUT_DIR / 'fig_ch4_model_evolution_compare_2026-03-30.svg'
FIG2 = OUT_DIR / 'fig_ch4_plain24_vs_staged24_iter_curve_2026-03-30.svg'
FIG3 = OUT_DIR / 'fig_ch4_staged24_mc50_distribution_2026-03-30.svg'


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def make_fig1(baseline: dict, staged5: dict):
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

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.bar(x - w, pitch, width=w, label='pitch mean |err|')
    ax.bar(x, yaw, width=w, label='yaw mean |err|')
    ax.bar(x + w, norm, width=w, label='norm mean')
    ax.set_ylabel('arcsec')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title('Model evolution under the same DAR truth (representative statistics)')
    ax.grid(axis='y', linestyle='--', alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG1, format='svg')
    plt.close(fig)


def make_fig2(staged5: dict):
    plain_logs = staged5['plain24_iter5']['per_seed'][0]['iter_logs']
    staged_logs = staged5['staged24_iter5']['per_seed'][0]['iter_logs']
    it_plain = [x['iteration'] for x in plain_logs]
    it_staged = [x['iteration'] for x in staged_logs]
    pitch_plain = [x['att_err_arcsec'][1] for x in plain_logs]
    pitch_staged = [x['att_err_arcsec'][1] for x in staged_logs]
    yaw_plain = [abs(x['att_err_arcsec'][2]) for x in plain_logs]
    yaw_staged = [abs(x['att_err_arcsec'][2]) for x in staged_logs]

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.2))
    axes[0].plot(it_plain, pitch_plain, marker='o', label='plain24')
    axes[0].plot(it_staged, pitch_staged, marker='s', label='staged24')
    axes[0].axhline(0.0, color='k', linewidth=0.8, linestyle='--')
    axes[0].set_xlabel('iteration')
    axes[0].set_ylabel('pitch signed error / arcsec')
    axes[0].set_title('Seed0 pitch evolution')
    axes[0].grid(True, linestyle='--', alpha=0.35)
    axes[0].legend(frameon=False)

    axes[1].plot(it_plain, yaw_plain, marker='o', label='plain24')
    axes[1].plot(it_staged, yaw_staged, marker='s', label='staged24')
    axes[1].axhline(20.0, color='r', linewidth=0.8, linestyle='--', label='20 arcsec')
    axes[1].set_xlabel('iteration')
    axes[1].set_ylabel('yaw absolute error / arcsec')
    axes[1].set_title('Seed0 yaw evolution')
    axes[1].grid(True, linestyle='--', alpha=0.35)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(FIG2, format='svg')
    plt.close(fig)


def make_fig3(mc50: dict):
    per_seed = mc50['per_seed']
    yaw_abs = np.array([row['final_yaw_abs_arcsec'] for row in per_seed])
    pitch = np.array([row['final_att_err_arcsec'][1] for row in per_seed])

    yaw_sorted = np.sort(yaw_abs)
    pitch_sorted = np.sort(pitch)
    idx_yaw = np.arange(1, len(yaw_sorted) + 1)
    idx_pitch = np.arange(1, len(pitch_sorted) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].plot(idx_yaw, yaw_sorted, marker='o', linewidth=1.2, markersize=3)
    axes[0].axhline(20.0, color='r', linewidth=0.8, linestyle='--', label='20 arcsec')
    axes[0].set_xlabel('sorted sample index')
    axes[0].set_ylabel('yaw absolute error / arcsec')
    axes[0].set_title('staged24 MC50 yaw tail')
    axes[0].grid(True, linestyle='--', alpha=0.35)
    axes[0].legend(frameon=False)

    axes[1].plot(idx_pitch, pitch_sorted, marker='o', linewidth=1.2, markersize=3)
    axes[1].axhline(0.0, color='k', linewidth=0.8, linestyle='--')
    axes[1].set_xlabel('sorted sample index')
    axes[1].set_ylabel('pitch signed error / arcsec')
    axes[1].set_title('staged24 MC50 pitch stability')
    axes[1].grid(True, linestyle='--', alpha=0.35)

    fig.tight_layout()
    fig.savefig(FIG3, format='svg')
    plt.close(fig)


def main():
    baseline = load_json(BASELINE_PATH)
    staged5 = load_json(STAGED5_PATH)
    mc50 = load_json(MC50_PATH)
    make_fig1(baseline, staged5)
    make_fig2(staged5)
    make_fig3(mc50)
    print(json.dumps({
        'fig1': str(FIG1),
        'fig2': str(FIG2),
        'fig3': str(FIG3),
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
