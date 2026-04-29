#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

WORKSPACE = Path('/root/.openclaw/workspace')
PSINS_ROOT = WORKSPACE / 'tmp_psins_py'
if str(PSINS_ROOT) not in sys.path:
    sys.path.insert(0, str(PSINS_ROOT))

H24_PATH = WORKSPACE / 'scripts' / 'alignvn_dar_hybrid24_staged_py_2026-03-30.py'
LLM_SCD_JSON = WORKSPACE / 'tmp' / 'alignment_strategy_sweep' / 'ch4_llm_scd_only_alignment_2026-04-03.json'
OUT_DIR = WORKSPACE / 'tmp' / 'ch4_llm_scd_rewrite_2026-04-06'
OUT_JSON = OUT_DIR / 'ch4_plain24_llm_scd_observability_2026-04-06.json'
OUT_MD = OUT_DIR / 'ch4_plain24_llm_scd_observability_2026-04-06.md'
OUT_FIG = OUT_DIR / 'fig_ch4_plain24_llm_scd_observability_2026-04-06.svg'

STATE_NAMES = [
    'phi_E', 'phi_N', 'phi_U',
    'dV_E', 'dV_N', 'dV_U',
    'eb_x', 'eb_y', 'eb_z',
    'db_x', 'db_y', 'db_z',
    'ng_x', 'ng_y', 'ng_z',
    'xa_x', 'xa_y', 'xa_z',
    'kg_x', 'kg_y', 'kg_z',
    'ka_x', 'ka_y', 'ka_z',
]

FAMILY_MAP = {
    'phi': list(range(0, 3)),
    'dV': list(range(3, 6)),
    'eb': list(range(6, 9)),
    'db': list(range(9, 12)),
    'ng': list(range(12, 15)),
    'xa': list(range(15, 18)),
    'kg': list(range(18, 21)),
    'ka': list(range(21, 24)),
}
PRIMARY_FAMILIES = ['phi', 'dV', 'eb', 'db', 'ng', 'xa']
SCALE_FAMILIES = ['kg', 'ka']
PRIMARY_IDX = list(range(18))
SCALE_IDX = list(range(18, 24))


def load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'failed to load module from {path}')
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


h24 = load_module('h24_plain24_llm_scd_obs_20260406', H24_PATH)
acc18 = h24.load_acc18()
base12 = h24.load_base12()
glv = acc18.glv


def characteristic_scales() -> np.ndarray:
    imuerr = base12.build_imuerr()
    phi0 = np.array([0.1, 0.1, 0.5]) * glv.deg
    init_eb = np.maximum(np.asarray(imuerr['eb']).reshape(3), 0.1 * glv.dph)
    init_db = np.maximum(np.asarray(imuerr['db']).reshape(3), 1000 * glv.ug)
    ng_sigma = np.array([0.05, 0.05, 0.05]) * glv.dph
    xa_sigma = np.maximum(np.array([0.01, 0.01, 0.01]) * glv.ug, 5.0 * glv.ug)
    scale_sigma = np.full(3, 100.0 * glv.ppm)
    return np.r_[phi0, np.ones(3), init_eb, init_db, ng_sigma, xa_sigma, scale_sigma, scale_sigma]


def build_nominal_data() -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    ts = 0.01
    att0 = np.array([0.0, 0.0, 0.0])
    pos0 = acc18.posset(34, 116, 480, isdeg=1)
    rot_paras = base12.build_rot_paras()
    att_truth = acc18.attrottt(att0, rot_paras, ts)
    imu, _ = acc18.avp2imu(att_truth, pos0)
    return ts, pos0, att_truth, imu


def build_phi_k(
    pos0: np.ndarray,
    cnb: np.ndarray,
    dvn: np.ndarray,
    phim: np.ndarray,
    dvbm: np.ndarray,
    nts: float,
    fg: np.ndarray,
    fa: np.ndarray,
    high_rot: bool,
) -> np.ndarray:
    phi_k = np.eye(24)
    phi_k[0:3, 0:3] += acc18.askew(-acc18.Earth(pos0).wnie) * nts
    cnbts = cnb * nts
    phi_k[3:6, 0:3] = acc18.askew(dvn)
    phi_k[3:6, 9:12] = cnbts
    phi_k[3:6, 15:18] = cnbts
    phi_k[0:3, 6:9] = -cnbts
    phi_k[0:3, 12:15] = -cnbts
    phi_k[12:15, 12:15] = np.diag(fg)
    phi_k[15:18, 15:18] = np.diag(fa)
    if high_rot:
        phi_k[0:3, 18:21] = -cnb @ np.diag(phim[0:3])
        phi_k[3:6, 21:24] = cnb @ np.diag(dvbm[0:3])
    return phi_k


def normalize_scores(diag: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    eps = 1e-30
    logs = np.log10(diag + eps)
    lo = float(np.min(logs))
    hi = float(np.max(logs))
    if hi - lo < 1e-12:
        return np.ones_like(diag), {'log10_min': lo, 'log10_max': hi}
    return (logs - lo) / (hi - lo), {'log10_min': lo, 'log10_max': hi}


def normalize_pair(diag_a: np.ndarray, diag_b: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    eps = 1e-30
    logs = np.log10(np.r_[diag_a + eps, diag_b + eps])
    lo = float(np.min(logs))
    hi = float(np.max(logs))
    if hi - lo < 1e-12:
        return np.ones_like(diag_a), np.ones_like(diag_b), {'log10_min': lo, 'log10_max': hi}
    sa = (np.log10(diag_a + eps) - lo) / (hi - lo)
    sb = (np.log10(diag_b + eps) - lo) / (hi - lo)
    return sa, sb, {'log10_min': lo, 'log10_max': hi}


def qualitative(score: float | None) -> str | None:
    if score is None:
        return None
    if score >= 0.75:
        return '强'
    if score >= 0.45:
        return '中'
    return '弱'


def loss_level(loss: float | None) -> str | None:
    if loss is None:
        return None
    if loss >= 0.20:
        return '高'
    if loss >= 0.10:
        return '中'
    return '低'


def family_mean(scores: np.ndarray, family: str) -> float:
    idx = FAMILY_MAP[family]
    return float(np.mean(scores[idx]))


def fmt_float(x: float | None, digits: int = 3) -> str:
    if x is None:
        return '—'
    return f'{x:.{digits}f}'


def compute_current_formal_observability() -> dict[str, Any]:
    ts, pos0, att_truth, imu = build_nominal_data()
    nn = 2
    nts = nn * ts
    eth = acc18.Earth(pos0)
    cnn = acc18.rv2m(-eth.wnie * nts / 2)
    S = np.diag(characteristic_scales())
    Sinv = np.diag(1.0 / np.diag(S))
    R = np.diag(np.array([0.01, 0.01, 0.01])) ** 2 / nts
    Rinv = np.linalg.inv(R)
    H = np.hstack([np.zeros((3, 3)), np.eye(3), np.zeros((3, 18))])
    Htilde = H @ S
    fg = np.exp(-nts / np.array([300.0, 300.0, 300.0]))
    fa = np.exp(-nts / np.array([100.0, 100.0, 100.0]))
    rot_gate_rad = 5.0 * glv.deg

    W = np.zeros((24, 24))
    Psi = np.eye(24)
    high_rot_count = 0
    length = (len(imu) // nn) * nn
    for k in range(0, length, nn):
        wvm = imu[k:k + nn, 0:6]
        phim, dvbm = acc18.cnscl(wvm)
        att_k = att_truth[k, 0:3]
        cnb = acc18.q2mat(acc18.a2qua(att_k))
        dvn = cnn @ cnb @ dvbm
        high_rot = bool(np.max(np.abs(phim / nts)) > rot_gate_rad)
        high_rot_count += int(high_rot)
        phi_k = build_phi_k(pos0, cnb, dvn, phim, dvbm, nts, fg, fa, high_rot)
        Phi_tilde = Sinv @ phi_k @ S
        Psi = Phi_tilde @ Psi
        W += Psi.T @ Htilde.T @ Rinv @ Htilde @ Psi

    diag = np.diag(W)
    direct_scores, direct_meta = normalize_scores(diag)

    Wxx = W[np.ix_(PRIMARY_IDX, PRIMARY_IDX)]
    Wxz = W[np.ix_(PRIMARY_IDX, SCALE_IDX)]
    Wzz = W[np.ix_(SCALE_IDX, SCALE_IDX)]
    W_cond = Wxx - Wxz @ np.linalg.inv(Wzz) @ Wxz.T
    diag_xx = np.diag(Wxx)
    diag_cond = np.diag(W_cond)
    direct_primary_scores, conditional_primary_scores, primary_meta = normalize_pair(diag_xx, diag_cond)
    competition_loss = direct_primary_scores - conditional_primary_scores

    primary_scale_cross = float(np.linalg.norm(Wxz))
    family_cross_share = {}
    for family in PRIMARY_FAMILIES:
        block = W[np.ix_(FAMILY_MAP[family], SCALE_IDX)]
        family_cross_share[family] = float(np.linalg.norm(block) / primary_scale_cross) if primary_scale_cross > 0 else 0.0

    best_candidate = None
    if LLM_SCD_JSON.exists():
        payload = json.loads(LLM_SCD_JSON.read_text(encoding='utf-8'))
        best_name = payload.get('rankings', {}).get('best_by_yaw_mean')
        for item in payload.get('candidates', []):
            if item.get('candidate', {}).get('name') == best_name:
                best_candidate = item
                break

    target_scope_share = None
    if best_candidate is not None:
        row_indices = best_candidate['candidate']['row_indices']
        col_indices = best_candidate['candidate']['col_indices']
        target_block = W[np.ix_(row_indices, col_indices)]
        target_scope_share = float(np.linalg.norm(target_block) / primary_scale_cross) if primary_scale_cross > 0 else 0.0
    else:
        row_indices = None
        col_indices = None

    family_summary: dict[str, dict[str, Any]] = {}
    for family in FAMILY_MAP:
        info = {
            'formal_direct_score': family_mean(direct_scores, family),
            'formal_direct_level': qualitative(family_mean(direct_scores, family)),
        }
        if family in PRIMARY_FAMILIES:
            idx = FAMILY_MAP[family]
            info['primary_direct_score'] = float(np.mean(direct_primary_scores[idx]))
            info['conditional_primary_score'] = float(np.mean(conditional_primary_scores[idx]))
            info['competition_loss'] = float(np.mean(competition_loss[idx]))
            info['competition_loss_level'] = loss_level(info['competition_loss'])
            info['cross_to_scale_share'] = family_cross_share[family]
        else:
            info['primary_direct_score'] = None
            info['conditional_primary_score'] = None
            info['competition_loss'] = None
            info['competition_loss_level'] = None
            info['cross_to_scale_share'] = None
        family_summary[family] = info

    per_state = []
    for i, name in enumerate(STATE_NAMES):
        family = next(key for key, idxs in FAMILY_MAP.items() if i in idxs)
        item: dict[str, Any] = {
            'state': name,
            'family': family,
            'formal_direct_diag': float(diag[i]),
            'formal_direct_log10diag': float(np.log10(diag[i] + 1e-30)),
            'formal_direct_score': float(direct_scores[i]),
            'formal_direct_level': qualitative(float(direct_scores[i])),
        }
        if i < 18:
            item['primary_direct_diag'] = float(diag_xx[i])
            item['primary_direct_score'] = float(direct_primary_scores[i])
            item['primary_direct_level'] = qualitative(float(direct_primary_scores[i]))
            item['conditional_primary_diag'] = float(diag_cond[i])
            item['conditional_primary_score'] = float(conditional_primary_scores[i])
            item['conditional_primary_level'] = qualitative(float(conditional_primary_scores[i]))
            item['competition_loss'] = float(competition_loss[i])
            item['competition_loss_level'] = loss_level(float(competition_loss[i]))
        else:
            item['primary_direct_diag'] = None
            item['primary_direct_score'] = None
            item['primary_direct_level'] = None
            item['conditional_primary_diag'] = None
            item['conditional_primary_score'] = None
            item['conditional_primary_level'] = None
            item['competition_loss'] = None
            item['competition_loss_level'] = None
        per_state.append(item)

    top_direct = sorted(per_state, key=lambda x: x['formal_direct_score'], reverse=True)[:8]
    weakest_direct = sorted(per_state, key=lambda x: x['formal_direct_score'])[:8]
    top_competition_loss = sorted(
        [x for x in per_state if x['competition_loss'] is not None],
        key=lambda x: x['competition_loss'],
        reverse=True,
    )[:8]

    trajectory_stats = {
        'sample_period_s': ts,
        'two_sample_period_s': nts,
        'total_time_s': float(len(imu) * ts),
        'total_two_sample_steps': int(length // nn),
        'high_rotation_steps': int(high_rot_count),
        'high_rotation_time_s': float(high_rot_count * nts),
        'high_rotation_fraction': float(high_rot_count / (length // nn)),
    }

    current_method_note = (
        'Formal observability is evaluated on the current DAR/plain24/Markov 24-state skeleton with scale states '
        'present throughout and scale-to-navigation coupling activated only during high-rotation segments. '
        'SCD itself is not part of the formal observability definition because it only rescales covariance cross-blocks; '
        'therefore a separate competition-aware conditional view is reported for the primary states.'
    )

    llm_result_note = None
    if best_candidate is not None:
        ref_plain = payload.get('reference', {}).get('plain24', {})
        s = best_candidate.get('statistics', {})
        if ref_plain and s:
            llm_result_note = {
                'candidate': best_candidate['candidate']['name'],
                'pitch_mean_abs_arcsec': float(s['pitch_mean_abs_arcsec']),
                'yaw_abs_mean_arcsec': float(s['yaw_abs_mean_arcsec']),
                'norm_mean_arcsec': float(s['norm_mean_arcsec']),
                'plain24_yaw_abs_mean_arcsec': float(ref_plain['yaw_abs_mean_arcsec']),
                'yaw_gain_arcsec': float(ref_plain['yaw_abs_mean_arcsec'] - s['yaw_abs_mean_arcsec']),
                'yaw_gain_ratio': float((ref_plain['yaw_abs_mean_arcsec'] - s['yaw_abs_mean_arcsec']) / ref_plain['yaw_abs_mean_arcsec']),
            }

    return {
        'method': 'plain24_llm_scd_observability_analysis',
        'context': {
            'trajectory': 'current Chapter 4 dual-axis DAR rotation schedule',
            'filter': 'plain24 / Markov DAR skeleton with kg/ka present from iter1',
            'llm_scd_role': 'LLM chooses constrained SCD patches; SCD is covariance scheduling rather than a state-transition/measurement change.',
            'current_method_note': current_method_note,
        },
        'trajectory_stats': trajectory_stats,
        'formal_direct_meta': direct_meta,
        'conditional_primary_meta': primary_meta,
        'family_summary': family_summary,
        'per_state': per_state,
        'competition': {
            'primary_scale_crossblock_fro_norm': primary_scale_cross,
            'family_cross_to_scale_share': family_cross_share,
            'top_competition_loss_states': top_competition_loss,
            'best_llm_candidate_scope': {
                'name': best_candidate['candidate']['name'] if best_candidate else None,
                'row_indices': row_indices,
                'col_indices': col_indices,
                'scope_share_of_primary_scale_crossblock': target_scope_share,
                'note': 'Share is measured by Frobenius norm of the targeted cross-block over the whole primary↔scale cross-block.',
            },
        },
        'best_llm_result_note': llm_result_note,
        'top_direct_states': top_direct,
        'weakest_direct_states': weakest_direct,
        'summary_notes': {
            'strongest_families': 'phi, dV, db are the strongest formal families under the current DAR/plain24 excitation; kg is partially activated, ka is uneven, ng/xa remain weak.',
            'competition_view': 'After conditioning on jointly estimated scale states, dV and db remain the cleanest primary families, while phi/eb lose noticeably more normalized information.',
            'llm_link': 'The current best LLM+SCD scope targets attitude+bias ↔ scale, which covers the dominant share of the primary↔scale cross-block energy in this nominal analysis.',
        },
        'figure': str(OUT_FIG),
    }


# ---------- simple SVG figure ----------

def svg_text(x: float, y: float, text: str, size: int = 18, weight: str = 'normal', anchor: str = 'start', fill: str = '#111827') -> str:
    text = (
        text.replace('&', '&amp;')
        .replace('<', '&lt;')
        .replace('>', '&gt;')
    )
    return f"<text x='{x:.1f}' y='{y:.1f}' font-size='{size}' font-weight='{weight}' text-anchor='{anchor}' fill='{fill}' font-family='Arial, Helvetica, sans-serif'>{text}</text>"


def bar_panel_svg(
    x0: float,
    y0: float,
    w: float,
    h: float,
    labels: list[str],
    series: list[tuple[str, list[float], str]],
    title: str,
    highlight_labels: set[str] | None = None,
) -> str:
    highlight_labels = highlight_labels or set()
    parts: list[str] = []
    pad_l, pad_r, pad_t, pad_b = 62.0, 18.0, 42.0, 86.0
    px0 = x0 + pad_l
    py0 = y0 + pad_t
    pw = w - pad_l - pad_r
    ph = h - pad_t - pad_b

    parts.append(f"<rect x='{x0:.1f}' y='{y0:.1f}' width='{w:.1f}' height='{h:.1f}' rx='16' fill='#ffffff' stroke='#d1d5db'/>")
    parts.append(svg_text(x0 + 18, y0 + 28, title, size=22, weight='bold'))

    # grid + y labels
    for i in range(6):
        frac = i / 5
        y = py0 + ph * (1 - frac)
        parts.append(f"<line x1='{px0:.1f}' y1='{y:.1f}' x2='{px0 + pw:.1f}' y2='{y:.1f}' stroke='#e5e7eb' stroke-dasharray='4 4'/>")
        parts.append(svg_text(px0 - 10, y + 5, f'{frac:.1f}', size=14, anchor='end', fill='#6b7280'))

    parts.append(f"<line x1='{px0:.1f}' y1='{py0:.1f}' x2='{px0:.1f}' y2='{py0 + ph:.1f}' stroke='#111827' stroke-width='1.2'/>")
    parts.append(f"<line x1='{px0:.1f}' y1='{py0 + ph:.1f}' x2='{px0 + pw:.1f}' y2='{py0 + ph:.1f}' stroke='#111827' stroke-width='1.2'/>")

    n = len(labels)
    g_w = pw / max(n, 1)
    inner_w = g_w * 0.76
    bar_w = inner_w / max(len(series), 1)

    for j, label in enumerate(labels):
        gx = px0 + j * g_w + (g_w - inner_w) / 2
        if label in highlight_labels:
            parts.append(f"<rect x='{gx - 6:.1f}' y='{py0 + 8:.1f}' width='{inner_w + 12:.1f}' height='{ph - 8:.1f}' fill='#fff7ed' opacity='0.85'/>")
        for k, (_, values, color) in enumerate(series):
            val = max(0.0, min(1.0, float(values[j])))
            bh = ph * val
            bx = gx + k * bar_w
            by = py0 + ph - bh
            parts.append(f"<rect x='{bx:.1f}' y='{by:.1f}' width='{bar_w - 2:.1f}' height='{bh:.1f}' fill='{color}' rx='4'/>")
        parts.append(svg_text(gx + inner_w / 2, py0 + ph + 28, label, size=15, anchor='middle', fill='#374151'))

    # legend
    lx = x0 + 18
    ly = y0 + h - 30
    for i, (name, _, color) in enumerate(series):
        xx = lx + i * 210
        parts.append(f"<rect x='{xx:.1f}' y='{ly - 12:.1f}' width='18' height='18' fill='{color}' rx='3'/>")
        parts.append(svg_text(xx + 26, ly + 2, name, size=15, fill='#374151'))

    return '\n'.join(parts)


def make_svg(payload: dict[str, Any]) -> None:
    width, height = 1600, 900
    bg = "<rect x='0' y='0' width='1600' height='900' fill='#f8fafc'/>"

    family_labels = list(FAMILY_MAP.keys())
    family_scores = [payload['family_summary'][fam]['formal_direct_score'] for fam in family_labels]

    primary_labels = PRIMARY_FAMILIES
    primary_direct = [payload['family_summary'][fam]['formal_direct_score'] for fam in primary_labels]
    primary_cond = [payload['family_summary'][fam]['conditional_primary_score'] for fam in primary_labels]

    target_scope = payload['competition']['best_llm_candidate_scope']
    scope_name = target_scope['name'] or 'N/A'
    share = target_scope['scope_share_of_primary_scale_crossblock']
    share_txt = 'N/A' if share is None else f'{share * 100:.1f}%'
    yaw_txt = ''
    if payload.get('best_llm_result_note'):
        note = payload['best_llm_result_note']
        yaw_txt = f"best LLM+SCD yaw: {note['plain24_yaw_abs_mean_arcsec']:.3f} -> {note['yaw_abs_mean_arcsec']:.3f} arcsec"

    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        bg,
        svg_text(48, 54, 'Current Chapter-4 observability under DAR / plain24 / LLM+SCD context', size=30, weight='bold'),
        svg_text(48, 84, 'Formal view: finite-horizon normalized observability Gramian on the current 24-state DAR skeleton', size=18, fill='#374151'),
        svg_text(48, 110, 'Competition-aware view: primary-state conditional information after accounting for jointly estimated scale states', size=18, fill='#374151'),
        bar_panel_svg(
            40, 145, 735, 620,
            family_labels,
            [('formal direct', family_scores, '#2563eb')],
            'A. Formal family-level observability',
            highlight_labels={'phi', 'dV', 'db', 'kg'},
        ),
        bar_panel_svg(
            820, 145, 740, 620,
            primary_labels,
            [('formal direct', primary_direct, '#60a5fa'), ('conditional primary', primary_cond, '#f97316')],
            'B. Primary-state competition after accounting for scale states',
            highlight_labels={'phi', 'eb', 'db'},
        ),
        f"<rect x='40' y='790' width='1520' height='82' rx='14' fill='#ffffff' stroke='#d1d5db'/>",
        svg_text(60, 822, f"Current best LLM+SCD scope: {scope_name}; targeted block share of primary↔scale cross-block = {share_txt}", size=20, weight='bold', fill='#9a3412'),
        svg_text(60, 850, 'Interpretation: SCD does not change formal observability; it trims covariance competition, mainly on attitude+bias ↔ scale channels.', size=18, fill='#374151'),
    ]
    if yaw_txt:
        parts.append(svg_text(1540, 850, yaw_txt, size=18, anchor='end', fill='#374151'))
    parts.append('</svg>')
    OUT_FIG.write_text('\n'.join(parts), encoding='utf-8')


def make_markdown(payload: dict[str, Any]) -> None:
    fam = payload['family_summary']
    lines = [
        '# Current DAR / plain24 observability analysis (LLM+SCD context)',
        '',
        '- **Formal object**: finite-horizon normalized observability Gramian on the current Chapter-4 DAR/plain24/Markov 24-state skeleton.',
        '- **Important note**: SCD is **not** part of formal observability here; it only suppresses covariance cross-blocks. Therefore the formal scores come from the same 24-state dynamics, and a separate competition-aware conditional view is reported for the primary states.',
        f"- **Trajectory span**: {payload['trajectory_stats']['total_time_s']:.1f} s, with {payload['trajectory_stats']['high_rotation_time_s']:.1f} s high-rotation excitation ({payload['trajectory_stats']['high_rotation_fraction'] * 100:.2f}%).",
        '',
        '## 1) Family-level scores',
        '',
        '| family | formal direct score | level | conditional primary score | competition loss | note |',
        '|---|---:|---|---:|---:|---|',
    ]

    family_notes = {
        'phi': '姿态失准角主族，formal 很强，但与 scale 存在明显竞争。',
        'dV': '速度量测直达族，formal 与 conditional 都较稳。',
        'eb': '陀螺常值偏置族，formal 中等，竞争后明显变弱。',
        'db': '加计常值偏置族，formal 很强，竞争后仍保持较好。',
        'ng': '陀螺 GM 有色噪声族，formal 弱。',
        'xa': '加计有色噪声族，formal 最弱。',
        'kg': 'gyro scale 只给 formal direct，主要由高转速段激活。',
        'ka': 'accel scale 只给 formal direct，轴向不均衡明显。',
    }
    for family in FAMILY_MAP:
        item = fam[family]
        lines.append(
            f"| {family} | {item['formal_direct_score']:.3f} | {item['formal_direct_level']} | {fmt_float(item['conditional_primary_score'])} | {fmt_float(item['competition_loss'])} | {family_notes[family]} |"
        )

    lines.extend([
        '',
        '## 2) State-level scores (all 24 states)',
        '',
        '| state | family | formal direct score | level | conditional primary score | competition loss | log10(diag W) |',
        '|---|---|---:|---|---:|---:|---:|',
    ])
    for row in payload['per_state']:
        lines.append(
            f"| {row['state']} | {row['family']} | {row['formal_direct_score']:.3f} | {row['formal_direct_level']} | {fmt_float(row['conditional_primary_score'])} | {fmt_float(row['competition_loss'])} | {row['formal_direct_log10diag']:.3f} |"
        )

    lines.extend([
        '',
        '## 3) Key numeric findings',
        '',
        '### Strongest formal states',
        '',
        '| state | direct score | family |',
        '|---|---:|---|',
    ])
    for row in payload['top_direct_states']:
        lines.append(f"| {row['state']} | {row['formal_direct_score']:.3f} | {row['family']} |")

    lines.extend([
        '',
        '### Weakest formal states',
        '',
        '| state | direct score | family |',
        '|---|---:|---|',
    ])
    for row in payload['weakest_direct_states']:
        lines.append(f"| {row['state']} | {row['formal_direct_score']:.3f} | {row['family']} |")

    lines.extend([
        '',
        '### Largest competition losses among primary states',
        '',
        '| state | conditional score | competition loss | family |',
        '|---|---:|---:|---|',
    ])
    for row in payload['competition']['top_competition_loss_states']:
        lines.append(f"| {row['state']} | {row['conditional_primary_score']:.3f} | {row['competition_loss']:.3f} | {row['family']} |")

    lines.extend([
        '',
        '## 4) Interpretation for the current Chapter-4 strategy',
        '',
        f"1. **Formal observability strongest families**: phi={fam['phi']['formal_direct_score']:.3f}, dV={fam['dV']['formal_direct_score']:.3f}, db={fam['db']['formal_direct_score']:.3f}. This means the current DAR trajectory mainly locks attitude/velocity/constant-bias channels first.",
        f"2. **Scale states are partially but not uniformly observable**: kg={fam['kg']['formal_direct_score']:.3f}, ka={fam['ka']['formal_direct_score']:.3f}; at state level `kg_y={payload['per_state'][19]['formal_direct_score']:.3f}` is strong, while `ka_y={payload['per_state'][22]['formal_direct_score']:.3f}` is essentially the weakest state in the full set.",
        f"3. **Colored states stay weak**: ng={fam['ng']['formal_direct_score']:.3f}, xa={fam['xa']['formal_direct_score']:.3f}. Under the current 5-minute DAR alignment horizon, colored states are not the main carriers of clean observable information.",
        f"4. **Competition-aware primary view**: dV keeps the best conditional score ({fam['dV']['conditional_primary_score']:.3f}), db also remains relatively robust ({fam['db']['conditional_primary_score']:.3f}), while phi drops from strong formal excitation to a lower conditional score ({fam['phi']['conditional_primary_score']:.3f}). Among actionable navigation states, eb loses the most normalized information ({fam['eb']['competition_loss']:.3f}).",
    ])

    scope = payload['competition']['best_llm_candidate_scope']
    if scope['name'] is not None:
        share = scope['scope_share_of_primary_scale_crossblock']
        lines.append(
            f"5. **Why the current LLM+SCD target makes sense**: the best current candidate `{scope['name']}` suppresses `attitude+bias ↔ scale` cross-covariance. In this nominal observability/competition proxy, that scope covers about **{share * 100:.1f}%** of the full primary↔scale cross-block Frobenius norm, so it is targeting the dominant competitive channel instead of broad random surgery."
        )
    if payload.get('best_llm_result_note'):
        note = payload['best_llm_result_note']
        lines.append(
            f"6. **Consistency with current alignment results**: relative to plain24, the current best LLM+SCD candidate reduces mean yaw error from **{note['plain24_yaw_abs_mean_arcsec']:.3f}\"** to **{note['yaw_abs_mean_arcsec']:.3f}\"**, i.e. **{note['yaw_gain_arcsec']:.3f}\" ({note['yaw_gain_ratio'] * 100:.2f}%)**. This is consistent with the competition-aware view: SCD is helping information allocation, not changing the formal observability structure."
        )

    lines.extend([
        '',
        '## 5) Output artifact',
        '',
        f"- figure: `{OUT_FIG}`",
        f"- json: `{OUT_JSON}`",
        f"- markdown: `{OUT_MD}`",
    ])

    OUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = compute_current_formal_observability()
    make_svg(payload)
    payload['figure'] = str(OUT_FIG)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    make_markdown(payload)
    print(json.dumps({
        'out_json': str(OUT_JSON),
        'out_md': str(OUT_MD),
        'out_fig': str(OUT_FIG),
        'family_summary': payload['family_summary'],
        'best_llm_candidate_scope': payload['competition']['best_llm_candidate_scope'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
