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
OUT_PNG = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'fig_ch4_iter_tradeoff_pub_2026-03-31.png'
OUT_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'ch4_iter_tradeoff_2026-03-31.json'


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
    base = data['baseline_reference']
    plain = mean_metrics_over_iters(data['plain24_iter5']['per_seed'])
    staged = mean_metrics_over_iters(data['staged24_iter5']['per_seed'])
    iters = np.arange(1, 6)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))
    titles = [
        ('Pitch mean |error|', 'pitch_mean_abs', base['baseline18_iter5']['pitch_mean_abs']),
        ('Yaw mean |error|', 'yaw_mean_abs', base['baseline18_iter5']['yaw_abs_mean']),
        ('Mean attitude error norm', 'norm_mean', base['baseline18_iter5']['norm_mean']),
    ]

    for ax, (title, key, baseline_val) in zip(axes, titles):
        ax.plot(iters, plain[key], marker='o', linewidth=2.0, label='plain24')
        ax.plot(iters, staged[key], marker='s', linewidth=2.0, label='staged24')
        ax.axhline(baseline_val, color='gray', linestyle='--', linewidth=1.0, label='18-state iter5' if key == 'pitch_mean_abs' else None)
        ax.set_title(title)
        ax.set_xlabel('outer iteration')
        ax.set_ylabel('arcsec')
        ax.set_xticks(iters)
        ax.grid(True, linestyle='--', alpha=0.35)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, [l for l in labels if l], frameon=False)
    fig.suptitle('Iteration-level trade-off of plain24 and staged24 under the same 5-seed setting')
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=240, bbox_inches='tight')
    plt.close(fig)

    payload = {
        'plain24': plain,
        'staged24': staged,
        'baseline18_iter5': {
            'pitch_mean_abs': base['baseline18_iter5']['pitch_mean_abs'],
            'yaw_abs_mean': base['baseline18_iter5']['yaw_abs_mean'],
            'norm_mean': base['baseline18_iter5']['norm_mean'],
        },
        'figure': str(OUT_PNG),
        'note': 'This figure compares iteration-level mean metrics within the same plain24/staged24 experiment family; the earlier minimal hybrid24 pitch-repair probe is intentionally excluded from this figure because it is a separate objective-specific probe.',
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
