#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
IN_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'alignvn_dar_hybrid24_staged_result_2026-03-30.json'
OUT_PNG = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'fig_ch4_fullround_convergence_pub_2026-03-31.png'
OUT_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'ch4_fullround_convergence_2026-03-31.json'


def mean_metrics_over_iters(per_seed: list[dict]) -> dict[str, list[float]]:
    pitch_mean_abs, yaw_mean_abs, norm_mean = [], [], []
    for it in range(1, 6):
        pitch_vals, yaw_vals, norm_vals = [], [], []
        for row in per_seed:
            log = next(x for x in row['iter_logs'] if x['iteration'] == it)
            e = np.array(log['att_err_arcsec'], dtype=float)
            pitch_vals.append(abs(e[1]))
            yaw_vals.append(abs(e[2]))
            norm_vals.append(log['att_err_norm_arcsec'])
        pitch_mean_abs.append(float(np.mean(pitch_vals)))
        yaw_mean_abs.append(float(np.mean(yaw_vals)))
        norm_mean.append(float(np.mean(norm_vals)))
    return {
        'pitch_mean_abs': pitch_mean_abs,
        'yaw_mean_abs': yaw_mean_abs,
        'norm_mean': norm_mean,
    }


def main():
    data = json.loads(IN_JSON.read_text(encoding='utf-8'))
    plain = mean_metrics_over_iters(data['plain24_iter5']['per_seed'])
    staged = mean_metrics_over_iters(data['staged24_iter5']['per_seed'])
    baseline = data['baseline_reference']['baseline18_iter5']
    iters = np.arange(1, 6)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.6))
    panels = [
        ('Pitch mean |error|', 'pitch_mean_abs', baseline['pitch_mean_abs']),
        ('Yaw mean |error|', 'yaw_mean_abs', baseline['yaw_abs_mean']),
        ('Mean attitude error norm', 'norm_mean', baseline['norm_mean']),
    ]

    for ax, (title, key, base) in zip(axes, panels):
        ax.axvspan(0.5, 1.5, color='#d9d9d9', alpha=0.25)
        ax.axvline(1.5, color='gray', linestyle='--', linewidth=1.0)
        ax.plot(iters, plain[key], marker='o', linewidth=2.0, label='plain24')
        ax.plot(iters, staged[key], marker='s', linewidth=2.0, label='staged24')
        ax.axhline(base, color='k', linestyle=':', linewidth=1.0, label='18-state iter5' if key == 'pitch_mean_abs' else None)
        ax.set_title(title)
        ax.set_xlabel('outer iteration')
        ax.set_ylabel('arcsec')
        ax.set_xticks(iters)
        ax.grid(True, linestyle='--', alpha=0.35)
        ymax = max(max(plain[key]), max(staged[key]), base)
        ax.text(1.0, ymax * 1.02, 'freeze', ha='center', va='bottom', fontsize=9)
        ax.text(3.5, ymax * 1.02, 'release + gate', ha='center', va='bottom', fontsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, [x for x in labels if x], frameon=False, loc='upper right')
    fig.suptitle('Full-round convergence under the same 5-seed setting (staged release begins at iter=2)')
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=240, bbox_inches='tight')
    plt.close(fig)

    payload = {
        'plain24': plain,
        'staged24': staged,
        'baseline18_iter5': {
            'pitch_mean_abs': baseline['pitch_mean_abs'],
            'yaw_abs_mean': baseline['yaw_abs_mean'],
            'norm_mean': baseline['norm_mean'],
        },
        'release_begins_at_iter': 2,
        'figure': str(OUT_PNG),
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
