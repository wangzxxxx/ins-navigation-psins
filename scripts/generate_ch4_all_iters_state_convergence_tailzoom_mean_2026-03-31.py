#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WORKSPACE = Path('/root/.openclaw/workspace')
IN_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'ch4_all_iters_state_convergence_2026-03-31.json'
OUT_PNG = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'fig_ch4_all_iters_state_convergence_tailzoom_mean_pub_2026-03-31.png'
OUT_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'ch4_all_iters_state_convergence_tailzoom_mean_2026-03-31.json'
CUT_S = 260.0


def gate_spans(rows):
    spans = []
    start = None
    prev = None
    for row in rows:
        t = row['time_s']
        flag = row['high_rot']
        if flag and start is None:
            start = t
        if (not flag) and start is not None:
            spans.append((start, prev if prev is not None else t))
            start = None
        prev = t
    if start is not None and prev is not None:
        spans.append((start, prev))
    return spans


def filtered(arr):
    return [r for r in arr if r['time_s'] >= CUT_S]


def main():
    obj = json.loads(IN_JSON.read_text(encoding='utf-8'))
    plain = {k: filtered(v) for k, v in obj['plain24_mean'].items()}
    staged = {k: filtered(v) for k, v in obj['staged24_mean'].items()}

    roll_max = max(max(r['roll_abs_arcsec'] for r in arr) for arr in list(plain.values()) + list(staged.values()))
    pitch_min = min(min(r['pitch_signed_arcsec'] for r in arr) for arr in list(plain.values()) + list(staged.values()))
    pitch_max = max(max(r['pitch_signed_arcsec'] for r in arr) for arr in list(plain.values()) + list(staged.values()))
    yaw_max = max(max(r['yaw_abs_arcsec'] for r in arr) for arr in list(plain.values()) + list(staged.values()))

    fig, axes = plt.subplots(3, 5, figsize=(18, 9))
    row_keys = ['roll_abs_arcsec', 'pitch_signed_arcsec', 'yaw_abs_arcsec']
    ylabels = ['roll |error| / arcsec', 'pitch error / arcsec', 'yaw |error| / arcsec']

    for col, it in enumerate(['1', '2', '3', '4', '5']):
        gate = gate_spans(staged[it])
        for ridx, (key, ylabel) in enumerate(zip(row_keys, ylabels)):
            ax = axes[ridx, col]
            for a, b in gate:
                ax.axvspan(a, b, color='#d9d9d9', alpha=0.20)
            ax.plot([r['time_s'] for r in plain[it]], [r[key] for r in plain[it]], linewidth=2.0, label='plain24 (5-seed mean)')
            ax.plot([r['time_s'] for r in staged[it]], [r[key] for r in staged[it]], linewidth=2.0, label='staged24 (5-seed mean)')
            ax.grid(True, linestyle='--', alpha=0.3)
            ax.set_xlim(CUT_S, plain[it][-1]['time_s'])
            if ridx == 1:
                ax.axhline(0.0, color='k', linestyle='--', linewidth=0.8)
                ax.set_ylim(pitch_min * 1.08, pitch_max * 1.08)
            elif ridx == 0:
                ax.set_ylim(0.0, roll_max * 1.10)
            else:
                ax.set_ylim(0.0, yaw_max * 1.10)
            if col == 0:
                ax.set_ylabel(ylabel)
            if ridx == 2:
                ax.set_xlabel('alignment time / s')
            if ridx == 0:
                ax.set_title(f'iter={it} (5-seed mean, tail zoom)')
            if ridx == 0 and col == 4:
                ax.legend(frameon=False, loc='upper right')

    fig.suptitle('Tail-zoomed convergence using the same 5-seed mean setting as Table 1 (t ≥ 260 s)')
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=220, bbox_inches='tight')
    plt.close(fig)

    payload = {
        'source': str(IN_JSON),
        'figure': str(OUT_PNG),
        'cut_s': CUT_S,
        'plain24_tailzoom_mean': plain,
        'staged24_tailzoom_mean': staged,
        'note': 'This figure uses the same 5-seed mean setting as Table 1, so it can be read directly against the statistical result table.',
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'figure': str(OUT_PNG), 'cut_s': CUT_S}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
