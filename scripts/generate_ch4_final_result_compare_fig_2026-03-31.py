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
OUT_PNG = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'fig_ch4_final_result_compare_pub_2026-03-31.png'
OUT_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'ch4_final_result_compare_2026-03-31.json'


def main():
    data = json.loads(IN_JSON.read_text(encoding='utf-8'))
    plain = data['plain24_iter5']
    staged = data['staged24_iter5']

    def extract(cfg: dict):
        s = cfg['statistics']
        per = cfg['per_seed']
        yaw_max = max(row['final_yaw_abs_arcsec'] for row in per)
        return {
            'roll_mean_abs': float(s['mean_abs_arcsec'][0]),
            'pitch_mean_abs': float(s['mean_abs_arcsec'][1]),
            'yaw_mean_abs': float(s['yaw_abs_mean_arcsec']),
            'norm_mean': float(s['norm_mean_arcsec']),
            'yaw_median_abs': float(s['yaw_abs_median_arcsec']),
            'yaw_max_abs': float(yaw_max),
        }

    p = extract(plain)
    s = extract(staged)

    labels1 = ['roll\nmean|.|', 'pitch\nmean|.|', 'yaw\nmean|.|', 'norm\nmean']
    labels2 = ['yaw\nmedian|.|', 'yaw\nmax|.|']
    vals1_plain = [p['roll_mean_abs'], p['pitch_mean_abs'], p['yaw_mean_abs'], p['norm_mean']]
    vals1_staged = [s['roll_mean_abs'], s['pitch_mean_abs'], s['yaw_mean_abs'], s['norm_mean']]
    vals2_plain = [p['yaw_median_abs'], p['yaw_max_abs']]
    vals2_staged = [s['yaw_median_abs'], s['yaw_max_abs']]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), gridspec_kw={'width_ratios': [2.3, 1.2]})
    w = 0.34

    x = np.arange(len(labels1))
    axes[0].bar(x - w/2, vals1_plain, width=w, label='plain24 iter5')
    axes[0].bar(x + w/2, vals1_staged, width=w, label='staged24 iter5')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels1)
    axes[0].set_ylabel('arcsec')
    axes[0].set_title('Final alignment result (5-seed mean metrics)')
    axes[0].grid(axis='y', linestyle='--', alpha=0.35)
    axes[0].legend(frameon=False, loc='upper left')

    x2 = np.arange(len(labels2))
    axes[1].bar(x2 - w/2, vals2_plain, width=w, label='plain24 iter5')
    axes[1].bar(x2 + w/2, vals2_staged, width=w, label='staged24 iter5')
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(labels2)
    axes[1].set_ylabel('arcsec')
    axes[1].set_title('Yaw tail comparison')
    axes[1].grid(axis='y', linestyle='--', alpha=0.35)

    fig.suptitle('Plain vs staged: final alignment trade-off at iter=5')
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=240, bbox_inches='tight')
    plt.close(fig)

    payload = {
        'plain24_iter5': p,
        'staged24_iter5': s,
        'figure': str(OUT_PNG),
        'note': 'This figure compares final end-of-alignment metrics only. Plain is better on pitch mean absolute error; staged is better on yaw and overall norm.',
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
