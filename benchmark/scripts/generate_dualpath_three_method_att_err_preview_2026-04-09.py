#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from xml.sax.saxutils import escape

WORKSPACE = Path('/root/.openclaw/workspace')
BASE_SCRIPT = WORKSPACE / 'psins_method_bench' / 'scripts' / 'generate_dualpath_three_method_state_convergence_v2_first3_2026-04-09.py'
OUT_DIR = WORKSPACE / 'tmp' / 'psins_dualpath_three_method_att_err_preview_2026-04-09'
OUT_JSON = OUT_DIR / 'summary.json'
TRACE_DT_S = 0.2
OUTER_ITERS = 3
MAX_POINTS = 1200
LINE_OPACITY = 0.56
LINE_WIDTH = 1.8
LEGEND_LINE_WIDTH = 2.8


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


bp = load_module('dualpath_baseplot_20260409', BASE_SCRIPT)


def downsample(x, ys, max_points=MAX_POINTS):
    n = len(x)
    if n <= max_points:
        return x, ys
    idx = [round(i * (n - 1) / (max_points - 1)) for i in range(max_points)]
    dedup = []
    seen = set()
    for i in idx:
        if i not in seen:
            seen.add(i)
            dedup.append(i)
    return [x[i] for i in dedup], [[arr[i] for i in dedup] for arr in ys]


def polyline(xs, ys):
    return ' '.join(f'{x:.2f},{y:.2f}' for x, y in zip(xs, ys))


def fmt_y(v: float) -> str:
    if abs(v) >= 1000 or (abs(v) > 0 and abs(v) < 1e-3):
        return f'{v:.2e}'
    return f'{v:.4f}'


def fmt_time_tick(v: float) -> str:
    return str(int(round(v)))


def nice_ticks(vmin: float, vmax: float, n: int = 5):
    if math.isclose(vmin, vmax, rel_tol=0.0, abs_tol=1e-15):
        delta = 1.0 if abs(vmin) < 1e-12 else abs(vmin) * 0.1
        vmin -= delta
        vmax += delta
    return [vmin + (vmax - vmin) * i / (n - 1) for i in range(n)]


def percentile(seq, q):
    if not seq:
        return 0.0
    arr = sorted(seq)
    if len(arr) == 1:
        return arr[0]
    pos = (len(arr) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    frac = pos - lo
    return arr[lo] * (1 - frac) + arr[hi] * frac


def robust_range(values, low_q=0.0, high_q=1.0, min_pad_frac=0.08):
    if not values:
        return -1.0, 1.0
    lo = percentile(values, low_q)
    hi = percentile(values, high_q)
    full_lo = min(values)
    full_hi = max(values)
    lo = min(lo, full_lo)
    hi = max(hi, full_hi)
    if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-15):
        delta = 1.0 if abs(lo) < 1e-12 else abs(lo) * 0.1
        lo -= delta
        hi += delta
    pad = max((hi - lo) * min_pad_frac, 1e-12)
    return lo - pad, hi + pad


def draw_panel(parts, rect, title, x_ticks, y_ticks, x_mapper, y_mapper, x_label, y_label, zero_y=None, iter_bounds=None):
    x0, y0, x1, y1 = rect
    parts.append(f'<rect x="{x0}" y="{y0}" width="{x1 - x0}" height="{y1 - y0}" fill="white" stroke="#d0d7de"/>')
    parts.append(f'<text x="{x0 + 8}" y="{y0 + 20}" class="panel-title">{escape(title)}</text>')
    plot_top = y0 + 28
    plot_bottom = y1 - 36
    plot_left = x0 + 78
    plot_right = x1 - 18
    for yt in y_ticks:
        yy = y_mapper(yt)
        parts.append(f'<line x1="{plot_left}" y1="{yy:.2f}" x2="{plot_right}" y2="{yy:.2f}" class="grid"/>')
        parts.append(f'<text x="{plot_left - 12}" y="{yy + 4:.2f}" class="tick" text-anchor="end">{escape(fmt_y(yt))}</text>')
    for xt in x_ticks:
        xx = x_mapper(xt)
        parts.append(f'<line x1="{xx:.2f}" y1="{plot_top}" x2="{xx:.2f}" y2="{plot_bottom}" class="grid"/>')
        parts.append(f'<text x="{xx:.2f}" y="{plot_bottom + 18}" class="tick" text-anchor="middle">{escape(fmt_time_tick(xt))}</text>')
    if zero_y is not None:
        parts.append(f'<line x1="{plot_left}" y1="{zero_y:.2f}" x2="{plot_right}" y2="{zero_y:.2f}" class="zero"/>')
    if iter_bounds:
        for bound in iter_bounds[:-1]:
            xx = x_mapper(bound)
            parts.append(f'<line x1="{xx:.2f}" y1="{plot_top}" x2="{xx:.2f}" y2="{plot_bottom}" class="iter"/>')
    parts.append(f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" class="axis"/>')
    parts.append(f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" class="axis"/>')
    parts.append(f'<text x="{(plot_left + plot_right)/2:.2f}" y="{y1 - 8}" class="small" text-anchor="middle">{escape(x_label)}</text>')
    y_label_x = x0 + 8
    y_label_y = (plot_top + plot_bottom) / 2
    parts.append(f'<text x="{y_label_x}" y="{y_label_y:.2f}" class="small" transform="rotate(-90 {y_label_x} {y_label_y:.2f})" text-anchor="middle">{escape(y_label)}</text>')


def render_axis_preview(out_svg: Path, out_png: Path, axis_label: str, series_list):
    width = 1500
    height = 470
    gap = 18
    panel_w = 464
    panel_h = 320
    origin_x = 30
    origin_y = 40
    panels = [
        (origin_x, origin_y, origin_x + panel_w, origin_y + panel_h),
        (origin_x + panel_w + gap, origin_y, origin_x + 2 * panel_w + gap, origin_y + panel_h),
    ]

    all_x = [x for s in series_list for x in s['x']]
    x_min, x_max = min(all_x), max(all_x)
    x_ticks = nice_ticks(x_min, x_max, 6)
    iter_bounds = series_list[0]['iter_bounds_s'] if series_list else []
    if len(iter_bounds) >= 2:
        round3_start = iter_bounds[-2]
        tail_start = max(round3_start, x_max - 100.0)
    else:
        tail_start = max(x_min, x_max - 100.0)
    tail_end = x_max
    tail_ticks = nice_ticks(tail_start, tail_end, 5)

    est_all = [v for s in series_list for v in s['y']]
    full_ymin, full_ymax = robust_range(est_all, low_q=0.0, high_q=1.0, min_pad_frac=0.08)
    tail_vals = [v for s in series_list for x, v in zip(s['x'], s['y']) if x >= tail_start]
    if axis_label == 'att_err_y':
        lo = percentile(tail_vals, 0.01)
        hi = percentile(tail_vals, 0.99)
        if math.isclose(lo, hi, rel_tol=0.0, abs_tol=1e-15):
            delta = 1.0 if abs(lo) < 1e-12 else abs(lo) * 0.1
            lo -= delta
            hi += delta
        pad = max((hi - lo) * 0.12, 1e-12)
        tail_ymin, tail_ymax = lo - pad, hi + pad
    else:
        tail_ymin, tail_ymax = robust_range(tail_vals, low_q=0.0, high_q=1.0, min_pad_frac=0.12)

    parts = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">')
    parts.append('<style>text{font-family:Arial,Helvetica,sans-serif;} .title{font-size:22px;font-weight:bold;} .panel-title{font-size:15px;font-weight:bold;fill:#223;} .tick{font-size:11px;fill:#555;} .small{font-size:12px;fill:#444;} .axis{stroke:#333;stroke-width:1;} .grid{stroke:#e5e7eb;stroke-width:1;} .zero{stroke:#9aa0a6;stroke-width:1;stroke-dasharray:4 4;} .iter{stroke:#94a3b8;stroke-width:1;stroke-dasharray:6 4;} .legend{font-size:12px;fill:#222;}</style>')
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>')
    parts.append(f'<text x="18" y="26" class="title">{escape(axis_label)}</text>')

    def make_mappers(rect, xmin, xmax, ymin, ymax):
        x0, y0, x1, y1 = rect
        plot_top = y0 + 28
        plot_bottom = y1 - 36
        plot_left = x0 + 78
        plot_right = x1 - 18
        def mx(x):
            if math.isclose(xmin, xmax, abs_tol=1e-15):
                return (plot_left + plot_right) / 2
            return plot_left + (x - xmin) / (xmax - xmin) * (plot_right - plot_left)
        def my(y):
            if math.isclose(ymin, ymax, abs_tol=1e-15):
                return (plot_top + plot_bottom) / 2
            return plot_bottom - (y - ymin) / (ymax - ymin) * (plot_bottom - plot_top)
        return mx, my

    mx1, my1 = make_mappers(panels[0], x_min, x_max, full_ymin, full_ymax)
    zero1 = my1(0.0) if full_ymin <= 0 <= full_ymax else None
    draw_panel(parts, panels[0], 'Attitude error · first 3 rounds', x_ticks, nice_ticks(full_ymin, full_ymax, 5), mx1, my1, 'time (s)', 'arcsec', zero_y=zero1, iter_bounds=iter_bounds)

    mx2, my2 = make_mappers(panels[1], tail_start, tail_end, tail_ymin, tail_ymax)
    zero2 = my2(0.0) if tail_ymin <= 0 <= tail_ymax else None
    tail_iter = [b for b in iter_bounds if tail_start <= b <= tail_end]
    draw_panel(parts, panels[1], 'Attitude error · round-3 last 100 s', tail_ticks, nice_ticks(tail_ymin, tail_ymax, 5), mx2, my2, 'time (s)', 'arcsec', zero_y=zero2, iter_bounds=tail_iter)

    for series in series_list:
        style = bp.line_style(series['group_key'])
        xs1, [ys1] = downsample(series['x'], [series['y']])
        parts.append(f'<polyline points="{polyline([mx1(v) for v in xs1], [my1(v) for v in ys1])}" fill="none" {style}/>')
        tail_x = [x for x in series['x'] if x >= tail_start]
        tail_y = [y for x, y in zip(series['x'], series['y']) if x >= tail_start]
        xs2, [ys2] = downsample(tail_x, [tail_y], max_points=800)
        parts.append(f'<polyline points="{polyline([mx2(v) for v in xs2], [my2(v) for v in ys2])}" fill="none" {style}/>')

    legend_x = 22
    legend_y = 382
    parts.append(f'<rect x="{legend_x}" y="{legend_y}" width="980" height="42" fill="white" stroke="#d0d7de"/>')
    for i, series in enumerate(series_list):
        x = legend_x + 18 + i * 310
        y = legend_y + 22
        parts.append(f'<line x1="{x}" y1="{y}" x2="{x + 28}" y2="{y}" {bp.line_style(series["group_key"], legend=True)}/>')
        parts.append(f'<text x="{x + 38}" y="{y + 4}" class="legend">{escape(series["label"])}</text>')

    parts.append('</svg>')
    out_svg.write_text('\n'.join(parts), encoding='utf-8')
    os.system(f'ffmpeg -y -loglevel error -i "{out_svg}" -frames:v 1 "{out_png}"')


def sample_if_needed(records, global_t, att_err_arcsec, traces):
    if not traces['x'] or global_t - traces['last_saved_t'] >= TRACE_DT_S - 1e-12:
        traces['x'].append(global_t)
        for i, key in enumerate(['att_err_x', 'att_err_y', 'att_err_z']):
            traces[key].append(float(att_err_arcsec[i]))
        traces['last_saved_t'] = global_t


def run_scale18_att_err(shared):
    base12 = bp.load_base12()
    h24 = bp.load_h24()
    acc18 = h24.load_acc18()
    glv = acc18.glv
    att_truth = acc18.attrottt([0.0, 0.0, 0.0], shared['rot_paras'], bp.TS)

    imuerr = shared['imuerr']
    imu_corr = shared['imu_noisy'].copy()
    nn = 2
    ts = float(imu_corr[1, -1] - imu_corr[0, -1])
    nts = nn * ts
    qnb_seed = acc18.a2qua(shared['att0_guess']) if len(shared['att0_guess']) == 3 else shared['att0_guess'].copy()
    length = (len(imu_corr) // nn) * nn
    imu_corr = imu_corr[:length]

    eth = acc18.Earth(shared['pos0'])
    cnn = acc18.rv2m(-eth.wnie * nts / 2)
    rot_gate_rad = bp.ROT_GATE_DPS * glv.deg

    web = imuerr['web'].reshape(3)
    wdb = imuerr['wdb'].reshape(3)
    eb = imuerr['eb'].reshape(3)
    db = imuerr['db'].reshape(3)
    init_eb_p = bp.np.maximum(eb, 0.1 * glv.dph)
    init_db_p = bp.np.maximum(db, 1000 * glv.ug)
    init_scale_p = bp.np.full(3, 100.0 * glv.ppm)
    qk = bp.np.zeros((18, 18))
    qk[0:3, 0:3] = bp.np.diag(web**2 * nts)
    qk[3:6, 3:6] = bp.np.diag(wdb**2 * nts)
    ft = bp.np.zeros((18, 18))
    ft[0:3, 0:3] = acc18.askew(-eth.wnie)
    phikk_1 = bp.np.eye(18) + ft * nts
    hk = bp.np.hstack([bp.np.zeros((3, 3)), bp.np.eye(3), bp.np.zeros((3, 12))])

    traces = {'group_key': 'g2_scaleonly_rotation', 'label': bp.GROUP_LABELS['g2_scaleonly_rotation'], 'x': [], 'att_err_x': [], 'att_err_y': [], 'att_err_z': [], 'iter_bounds_s': [], 'last_saved_t': -1e9}
    for iteration in range(1, OUTER_ITERS + 1):
        kf = {
            'Phikk_1': phikk_1.copy(),
            'Qk': qk,
            'Rk': bp.np.diag(bp.WVN.reshape(3)) ** 2 / nts,
            'Pxk': bp.np.diag(bp.np.r_[shared['phi'], bp.np.ones(3), init_eb_p, init_db_p, init_scale_p, init_scale_p]) ** 2,
            'Hk': hk.copy(),
            'xk': bp.np.zeros(18),
        }
        vn = bp.np.zeros(3)
        qnbi = qnb_seed.copy()
        elapsed_s = 0.0
        for k in range(0, length, nn):
            wvm = imu_corr[k:k + nn, 0:6]
            phim, dvbm = acc18.cnscl(wvm)
            cnb = acc18.q2mat(qnbi)
            dvn = cnn @ cnb @ dvbm
            vn = vn + dvn + eth.gn * nts
            qnbi = acc18.qupdt2(qnbi, phim, eth.wnin * nts)
            phi_k = kf['Phikk_1'].copy()
            cnbts = cnb * nts
            phi_k[3:6, 0:3] = acc18.askew(dvn)
            phi_k[3:6, 9:12] = cnbts
            phi_k[0:3, 6:9] = -cnbts
            high_rot = bp.np.max(bp.np.abs(phim / nts)) > rot_gate_rad
            if high_rot:
                phi_k[0:3, 12:15] = -cnb @ bp.np.diag(phim[0:3])
                phi_k[3:6, 15:18] = cnb @ bp.np.diag(dvbm[0:3])
            else:
                phi_k[0:3, 12:15] = 0.0
                phi_k[3:6, 15:18] = 0.0
            kf['Phikk_1'] = phi_k
            kf = acc18.kfupdate(kf, vn)
            qnbi = acc18.qdelphi(qnbi, 0.91 * kf['xk'][0:3])
            kf['xk'][0:3] *= 0.09
            vn = vn - 0.91 * kf['xk'][3:6]
            kf['xk'][3:6] *= 0.09
            elapsed_s += nts
            global_t = elapsed_s + (iteration - 1) * shared['duration_s']
            truth_att_k = att_truth[min(k + nn - 1, len(att_truth) - 1), 0:3]
            att_err = acc18.qq2phi(acc18.a2qua(acc18.q2att(qnbi)), acc18.a2qua(truth_att_k)) / glv.sec
            sample_if_needed(None, global_t, att_err, traces)
        traces['iter_bounds_s'].append(traces['x'][-1])
        if iteration < OUTER_ITERS:
            qnb_seed = qnbi.copy()
            imu_corr[:, 0:3] -= bp.WASH_SCALE * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= bp.WASH_SCALE * kf['xk'][9:12] * ts
            imu_corr = h24.apply_scale_wash(imu_corr, kf['xk'][12:15], kf['xk'][15:18], bp.SCALE_WASH_SCALE)
    traces.pop('last_saved_t', None)
    return traces


def run_plain24_att_err(shared):
    h24 = bp.load_h24()
    acc18 = h24.load_acc18()
    glv = acc18.glv
    att_truth = acc18.attrottt([0.0, 0.0, 0.0], shared['rot_paras'], bp.TS)

    imu_corr = shared['imu_noisy'].copy()
    nn = 2
    ts = float(imu_corr[1, -1] - imu_corr[0, -1])
    nts = nn * ts
    qnb_seed = acc18.a2qua(shared['att0_guess']) if len(shared['att0_guess']) == 3 else shared['att0_guess'].copy()
    length = (len(imu_corr) // nn) * nn
    imu_corr = imu_corr[:length]
    eth = acc18.Earth(shared['pos0'])
    cnn = acc18.rv2m(-eth.wnie * nts / 2)
    rot_gate_rad = bp.ROT_GATE_DPS * glv.deg

    traces = {'group_key': 'g3_markov_rotation', 'label': bp.GROUP_LABELS['g3_markov_rotation'], 'x': [], 'att_err_x': [], 'att_err_y': [], 'att_err_z': [], 'iter_bounds_s': [], 'last_saved_t': -1e9}
    for iteration in range(1, OUTER_ITERS + 1):
        kf = h24.avnkfinit_24(
            nts, shared['pos0'], shared['phi'], shared['imuerr'], bp.WVN.copy(),
            bp.np.array([0.05, 0.05, 0.05]) * glv.dph,
            bp.np.array([300.0, 300.0, 300.0]),
            bp.np.array([0.01, 0.01, 0.01]) * glv.ug,
            bp.np.array([100.0, 100.0, 100.0]),
            enable_scale_states=True,
        )
        vn = bp.np.zeros(3)
        qnbi = qnb_seed.copy()
        elapsed_s = 0.0
        for k in range(0, length, nn):
            wvm = imu_corr[k:k + nn, 0:6]
            phim, dvbm = acc18.cnscl(wvm)
            cnb = acc18.q2mat(qnbi)
            dvn = cnn @ cnb @ dvbm
            vn = vn + dvn + eth.gn * nts
            qnbi = acc18.qupdt2(qnbi, phim, eth.wnin * nts)
            phi_k = kf['Phikk_1'].copy()
            cnbts = cnb * nts
            phi_k[3:6, 0:3] = acc18.askew(dvn)
            phi_k[3:6, 9:12] = cnbts
            phi_k[3:6, 15:18] = cnbts
            phi_k[0:3, 6:9] = -cnbts
            phi_k[0:3, 12:15] = -cnbts
            phi_k[12:15, 12:15] = bp.np.diag(kf['fg'])
            phi_k[15:18, 15:18] = bp.np.diag(kf['fa'])
            high_rot = bp.np.max(bp.np.abs(phim / nts)) > rot_gate_rad
            if high_rot:
                phi_k[0:3, 18:21] = -cnb @ bp.np.diag(phim[0:3])
                phi_k[3:6, 21:24] = cnb @ bp.np.diag(dvbm[0:3])
            else:
                phi_k[0:3, 18:21] = 0.0
                phi_k[3:6, 21:24] = 0.0
            kf['Phikk_1'] = phi_k
            kf = acc18.kfupdate(kf, vn)
            qnbi = acc18.qdelphi(qnbi, 0.91 * kf['xk'][0:3])
            kf['xk'][0:3] *= 0.09
            vn = vn - 0.91 * kf['xk'][3:6]
            kf['xk'][3:6] *= 0.09
            elapsed_s += nts
            global_t = elapsed_s + (iteration - 1) * shared['duration_s']
            truth_att_k = att_truth[min(k + nn - 1, len(att_truth) - 1), 0:3]
            att_err = acc18.qq2phi(acc18.a2qua(acc18.q2att(qnbi)), acc18.a2qua(truth_att_k)) / glv.sec
            sample_if_needed(None, global_t, att_err, traces)
        traces['iter_bounds_s'].append(traces['x'][-1])
        if iteration < OUTER_ITERS:
            qnb_seed = qnbi.copy()
            imu_corr[:, 0:3] -= bp.WASH_SCALE * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= bp.WASH_SCALE * kf['xk'][9:12] * ts
            imu_corr = h24.apply_scale_wash(imu_corr, kf['xk'][18:21], kf['xk'][21:24], bp.SCALE_WASH_SCALE)
    traces.pop('last_saved_t', None)
    return traces


def run_purescd_att_err(shared):
    h24 = bp.load_h24()
    pure = bp.load_pure()
    acc18 = h24.load_acc18()
    glv = acc18.glv
    att_truth = acc18.attrottt([0.0, 0.0, 0.0], shared['rot_paras'], bp.TS)

    imu_corr = shared['imu_noisy'].copy()
    nn = 2
    ts = float(imu_corr[1, -1] - imu_corr[0, -1])
    nts = nn * ts
    qnb_seed = acc18.a2qua(shared['att0_guess']) if len(shared['att0_guess']) == 3 else shared['att0_guess'].copy()
    length = (len(imu_corr) // nn) * nn
    imu_corr = imu_corr[:length]
    eth = acc18.Earth(shared['pos0'])
    cnn = acc18.rv2m(-eth.wnie * nts / 2)
    rot_gate_rad = bp.ROT_GATE_DPS * glv.deg
    scd_cfg = pure.SCDConfig(enabled=True, alpha=0.995, transition_duration_s=2.0, apply_after_release_iter=1, note='hard_a995_td2_i1')

    traces = {'group_key': 'g4_scd_rotation', 'label': bp.GROUP_LABELS['g4_scd_rotation'], 'x': [], 'att_err_x': [], 'att_err_y': [], 'att_err_z': [], 'iter_bounds_s': [], 'last_saved_t': -1e9}
    for iteration in range(1, OUTER_ITERS + 1):
        kf = h24.avnkfinit_24(
            nts, shared['pos0'], shared['phi'], shared['imuerr'], bp.WVN.copy(),
            bp.np.array([0.05, 0.05, 0.05]) * glv.dph,
            bp.np.array([300.0, 300.0, 300.0]),
            bp.np.array([0.01, 0.01, 0.01]) * glv.ug,
            bp.np.array([100.0, 100.0, 100.0]),
            enable_scale_states=True,
        )
        vn = bp.np.zeros(3)
        qnbi = qnb_seed.copy()
        time_since_rot_stop = 0.0
        scd_applied_this_phase = False
        elapsed_s = 0.0
        for k in range(0, length, nn):
            wvm = imu_corr[k:k + nn, 0:6]
            phim, dvbm = acc18.cnscl(wvm)
            cnb = acc18.q2mat(qnbi)
            dvn = cnn @ cnb @ dvbm
            vn = vn + dvn + eth.gn * nts
            qnbi = acc18.qupdt2(qnbi, phim, eth.wnin * nts)
            phi_k = kf['Phikk_1'].copy()
            cnbts = cnb * nts
            phi_k[3:6, 0:3] = acc18.askew(dvn)
            phi_k[3:6, 9:12] = cnbts
            phi_k[3:6, 15:18] = cnbts
            phi_k[0:3, 6:9] = -cnbts
            phi_k[0:3, 12:15] = -cnbts
            phi_k[12:15, 12:15] = bp.np.diag(kf['fg'])
            phi_k[15:18, 15:18] = bp.np.diag(kf['fa'])
            high_rot = bp.np.max(bp.np.abs(phim / nts)) > rot_gate_rad
            if high_rot:
                phi_k[0:3, 18:21] = -cnb @ bp.np.diag(phim[0:3])
                phi_k[3:6, 21:24] = cnb @ bp.np.diag(dvbm[0:3])
                time_since_rot_stop = 0.0
                scd_applied_this_phase = False
            else:
                phi_k[0:3, 18:21] = 0.0
                phi_k[3:6, 21:24] = 0.0
                time_since_rot_stop += nts
            kf['Phikk_1'] = phi_k
            kf = acc18.kfupdate(kf, vn)
            qnbi = acc18.qdelphi(qnbi, 0.91 * kf['xk'][0:3])
            kf['xk'][0:3] *= 0.09
            vn = vn - 0.91 * kf['xk'][3:6]
            kf['xk'][3:6] *= 0.09
            if scd_cfg.enabled and iteration >= scd_cfg.apply_after_release_iter and (not high_rot):
                if (time_since_rot_stop >= scd_cfg.transition_duration_s) and (not scd_applied_this_phase):
                    kf = pure.apply_scd_once(kf, scd_cfg)
                    scd_applied_this_phase = True
            elapsed_s += nts
            global_t = elapsed_s + (iteration - 1) * shared['duration_s']
            truth_att_k = att_truth[min(k + nn - 1, len(att_truth) - 1), 0:3]
            att_err = acc18.qq2phi(acc18.a2qua(acc18.q2att(qnbi)), acc18.a2qua(truth_att_k)) / glv.sec
            sample_if_needed(None, global_t, att_err, traces)
        traces['iter_bounds_s'].append(traces['x'][-1])
        if iteration < OUTER_ITERS:
            qnb_seed = qnbi.copy()
            imu_corr[:, 0:3] -= bp.WASH_SCALE * kf['xk'][6:9] * ts
            imu_corr[:, 3:6] -= bp.WASH_SCALE * kf['xk'][9:12] * ts
            imu_corr = h24.apply_scale_wash(imu_corr, kf['xk'][18:21], kf['xk'][21:24], bp.SCALE_WASH_SCALE)
    traces.pop('last_saved_t', None)
    return traces


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shared = bp.build_shared_dual_dataset()
    groups = [run_scale18_att_err(shared), run_plain24_att_err(shared), run_purescd_att_err(shared)]
    axes = ['att_err_x', 'att_err_y', 'att_err_z']
    summary = {
        'task': 'dualpath_three_method_att_err_preview_2026_04_09',
        'path_note': 'old Chapter-4 dual-axis rotation strategy build_rot_paras()',
        'display_labels': bp.GROUP_LABELS,
        'output_dir': str(OUT_DIR),
        'plots': [],
    }
    for axis in axes:
        series_list = [{'group_key': g['group_key'], 'label': g['label'], 'x': g['x'], 'y': g[axis], 'iter_bounds_s': g['iter_bounds_s']} for g in groups]
        svg_path = OUT_DIR / f'{axis}_preview.svg'
        png_path = OUT_DIR / f'{axis}_preview.png'
        render_axis_preview(svg_path, png_path, axis, series_list)
        summary['plots'].append({'axis': axis, 'svg': str(svg_path), 'png': str(png_path)})
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
