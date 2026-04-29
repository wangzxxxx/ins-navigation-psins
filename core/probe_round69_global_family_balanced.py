from __future__ import annotations

import copy
import gc
import json
import math
import sys
import types
from pathlib import Path

if 'matplotlib' not in sys.modules:
    matplotlib_stub = types.ModuleType('matplotlib')
    pyplot_stub = types.ModuleType('matplotlib.pyplot')
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules['matplotlib'] = matplotlib_stub
    sys.modules['matplotlib.pyplot'] = pyplot_stub
if 'seaborn' not in sys.modules:
    sys.modules['seaborn'] = types.ModuleType('seaborn')

ROOT = Path('/root/.openclaw/workspace')
METHOD_DIR = ROOT / 'psins_method_bench' / 'methods' / 'markov'
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
REPORTS_DIR = ROOT / 'reports'
SUMMARY_DIR = ROOT / 'psins_method_bench' / 'summary'
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'

ROUND61_REF_JSON = RESULTS_DIR / 'R65_mainline_round61_param_errors.json'
ROUND68_BEST_JSON = RESULTS_DIR / 'R68_global_family_r68_balanced_iso_mild_param_errors.json'

OUTPUT_JSON = RESULTS_DIR / 'round69_probe_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'round69_candidates.json'
REPORT_MD = REPORTS_DIR / 'psins_round69_probe_2026-03-28.md'
ROUND69_RECORD_MD = SUMMARY_DIR / 'round69_record_2026-03-28.md'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import load_module
from probe_round55_newline import _build_patched_method
from probe_round56_narrow import _compute_metrics
from probe_round59_h_scd_hybrid import _run_internalized_hybrid_scd
from probe_round61_hybrid_micro import ROUND61_CANDIDATES, _merge_round61_candidate
from probe_round65_mainline_innovation_consistency import _build_shared_dataset

ROUND61_BASE_NAME = 'r61_s20_08988_ryz00116'
PROTECTED_DIAGNOSTICS = ['dKg_xy', 'dKg_yy', 'dKa_xx', 'rx_y', 'ry_z']

# 全局家族（family）切分：强参数与弱参数都纳入统一平衡目标
FAMILY_PARAM_GROUPS = {
    'kg_block': ['dKg_xx', 'dKg_yx', 'dKg_zx', 'dKg_xy', 'dKg_yy', 'dKg_zy', 'dKg_xz', 'dKg_yz', 'dKg_zz'],
    'ka_block': ['dKa_xx', 'dKa_xy', 'dKa_xz', 'dKa_yy', 'dKa_yz', 'dKa_zz'],
    'gyro_bias': ['eb_x', 'eb_y', 'eb_z'],
    'acc_bias': ['db_x', 'db_y', 'db_z'],
    'ka2_block': ['Ka2_x', 'Ka2_y', 'Ka2_z'],
    'lever_block': ['rx_x', 'rx_y', 'rx_z', 'ry_x', 'ry_y', 'ry_z'],
}

ROUND69_CANDIDATES = [
    {
        'name': 'r69_iso_ref_anchor',
        'description': 'Exact carry-over of Round68 best for centered narrow sweep anchoring.',
        'rationale': '以 Round68 best (r68_balanced_iso_mild) 作为中心锚点，确保 Round69 是延续而非换轴。',
        'family_balance': {
            'gamma': 0.14,
            'blend': 0.72,
            'mult_min': 0.92,
            'mult_max': 1.10,
            'weak_consensus_blend': 0.22,
            'weights': {
                'kg_diag': 1.00,
                'kg_offdiag': 1.00,
                'ka_diag': 1.00,
                'ka_offdiag': 1.00,
                'gyro_bias': 1.00,
                'acc_bias': 1.00,
                'ka2': 0.90,
                'lever': 0.90,
            },
        },
    },
    {
        'name': 'r69_wcb012_relax',
        'description': 'Lower weak consensus blend only (0.22 -> 0.12), all else fixed.',
        'rationale': '只降 weak_consensus_blend，测试是否能减轻 dKg_xy / rx_y / ry_z 过拉。',
        'family_balance': {
            'gamma': 0.14,
            'blend': 0.72,
            'mult_min': 0.92,
            'mult_max': 1.10,
            'weak_consensus_blend': 0.12,
            'weights': {
                'kg_diag': 1.00,
                'kg_offdiag': 1.00,
                'ka_diag': 1.00,
                'ka_offdiag': 1.00,
                'gyro_bias': 1.00,
                'acc_bias': 1.00,
                'ka2': 0.90,
                'lever': 0.90,
            },
        },
    },
    {
        'name': 'r69_wcb006_relax_more',
        'description': 'Further lower weak consensus blend (0.22 -> 0.06), all else fixed.',
        'rationale': '继续单旋钮降低 weak_consensus_blend，观察过拉指标是否继续回落。',
        'family_balance': {
            'gamma': 0.14,
            'blend': 0.72,
            'mult_min': 0.92,
            'mult_max': 1.10,
            'weak_consensus_blend': 0.06,
            'weights': {
                'kg_diag': 1.00,
                'kg_offdiag': 1.00,
                'ka_diag': 1.00,
                'ka_offdiag': 1.00,
                'gyro_bias': 1.00,
                'acc_bias': 1.00,
                'ka2': 0.90,
                'lever': 0.90,
            },
        },
    },
    {
        'name': 'r69_cap108_wcb012',
        'description': 'Narrow envelope cap (0.92~1.10 -> 0.94~1.08) with weak_consensus_blend=0.12.',
        'rationale': '同步小幅收紧 multiplier envelope，抑制重分配幅度并配合弱一致性放松。',
        'family_balance': {
            'gamma': 0.14,
            'blend': 0.72,
            'mult_min': 0.94,
            'mult_max': 1.08,
            'weak_consensus_blend': 0.12,
            'weights': {
                'kg_diag': 1.00,
                'kg_offdiag': 1.00,
                'ka_diag': 1.00,
                'ka_offdiag': 1.00,
                'gyro_bias': 1.00,
                'acc_bias': 1.00,
                'ka2': 0.90,
                'lever': 0.90,
            },
        },
    },
    {
        'name': 'r69_cap106_wcb010_trim',
        'description': 'Tighter envelope + tiny family weight trim (kg_offdiag/lever slight up-weight).',
        'rationale': '在窄 cap 基础上仅做微小 family 权重修剪，尝试缓和 dKg_xy 与 lever 过拉。',
        'family_balance': {
            'gamma': 0.14,
            'blend': 0.72,
            'mult_min': 0.95,
            'mult_max': 1.06,
            'weak_consensus_blend': 0.10,
            'weights': {
                'kg_diag': 1.00,
                'kg_offdiag': 1.05,
                'ka_diag': 1.00,
                'ka_offdiag': 1.00,
                'gyro_bias': 1.00,
                'acc_bias': 1.00,
                'ka2': 0.90,
                'lever': 0.95,
            },
        },
    },
]

ROUND69_TARGET_REPAIR_METRICS = ['dKg_xy', 'dKg_zz', 'rx_y', 'ry_z']


def _load_round61_base_candidate():
    for candidate in ROUND61_CANDIDATES:
        if candidate['name'] == ROUND61_BASE_NAME:
            return _merge_round61_candidate(candidate)
    raise KeyError(ROUND61_BASE_NAME)


def _merge_round69_candidate(extra_candidate: dict):
    merged = copy.deepcopy(_load_round61_base_candidate())
    merged['name'] = extra_candidate['name']
    merged['description'] = extra_candidate['description']
    merged['rationale'] = extra_candidate['rationale']
    merged['family_balance'] = copy.deepcopy(extra_candidate['family_balance'])
    merged['round69_extra_patch'] = copy.deepcopy(extra_candidate)
    return merged


def _family_summary_from_payload(payload: dict):
    pe = payload['param_errors']
    family_mean = {}
    for fam, keys in FAMILY_PARAM_GROUPS.items():
        vals = [float(pe[k]['pct_error']) for k in keys if k in pe]
        family_mean[fam] = float(sum(vals) / max(len(vals), 1)) if vals else None

    means = [v for v in family_mean.values() if v is not None]
    dispersion = float((sum((v - (sum(means) / len(means))) ** 2 for v in means) / len(means)) ** 0.5) if means else None
    return {
        'family_mean_pct_error': family_mean,
        'family_dispersion': dispersion,
        'family_global_mean': float(sum(means) / len(means)) if means else None,
    }


def _apply_group_multiplier(arr, multiplier: float, blend: float):
    # staged global/local blending: arr <- (1-blend)*arr + blend*(mult*arr)
    return ((1.0 - blend) * arr) + (blend * multiplier * arr)


def _apply_global_family_balance(mod, clbt_in: dict, cfg: dict):
    np = mod.np
    clbt = copy.deepcopy(clbt_in)

    Kg_dev = np.array(clbt['Kg'], dtype=float) - np.eye(3)
    Ka_dev = np.array(clbt['Ka'], dtype=float) - np.eye(3)
    eb = np.array(clbt['eb'], dtype=float)
    db = np.array(clbt['db'], dtype=float)
    ka2 = np.array(clbt['Ka2'], dtype=float)
    rx = np.array(clbt['rx'], dtype=float)
    ry = np.array(clbt['ry'], dtype=float)

    offdiag_mask = np.ones((3, 3), dtype=bool)
    np.fill_diagonal(offdiag_mask, False)

    groups = {
        'kg_diag': np.diag(Kg_dev).copy(),
        'kg_offdiag': Kg_dev[offdiag_mask].copy(),
        'ka_diag': np.diag(Ka_dev).copy(),
        'ka_offdiag': Ka_dev[offdiag_mask].copy(),
        'gyro_bias': eb.copy(),
        'acc_bias': db.copy(),
        'ka2': ka2.copy(),
        'lever': np.hstack((rx, ry)).copy(),
    }

    ref_scales = {
        'kg_diag': 2.5e-5,
        'kg_offdiag': 1.5e-4,
        'ka_diag': 2.5e-5,
        'ka_offdiag': 1.5e-4,
        'gyro_bias': 1.0e-6,
        'acc_bias': 2.0e-3,
        'ka2': 2.5e-6,
        'lever': 3.0e-2,
    }

    gamma = float(cfg['gamma'])
    blend = float(cfg['blend'])
    mmin = float(cfg['mult_min'])
    mmax = float(cfg['mult_max'])
    weak_consensus_blend = float(cfg.get('weak_consensus_blend', 0.0))
    weights = cfg.get('weights', {})

    rms = {}
    for g, vec in groups.items():
        scale = ref_scales[g]
        rms[g] = float(np.sqrt(np.mean((vec / scale) ** 2) + 1e-18))

    weighted_logs = []
    wsum = 0.0
    for g, rv in rms.items():
        w = float(weights.get(g, 1.0))
        wsum += w
        weighted_logs.append(w * math.log(max(rv, 1e-12)))
    global_ref = math.exp(sum(weighted_logs) / max(wsum, 1e-12))

    multipliers = {}
    for g, rv in rms.items():
        raw = (global_ref / max(rv, 1e-12)) ** gamma
        multipliers[g] = float(max(mmin, min(mmax, raw)))

    # consensus-style weak refinement: weak family multiplier toward strong-family median
    strong_mults = [multipliers[k] for k in ['kg_diag', 'kg_offdiag', 'ka_diag', 'ka_offdiag', 'gyro_bias', 'acc_bias']]
    strong_consensus = float(np.median(np.array(strong_mults, dtype=float)))
    for weak_key in ['ka2', 'lever']:
        multipliers[weak_key] = (
            (1.0 - weak_consensus_blend) * multipliers[weak_key]
            + weak_consensus_blend * strong_consensus
        )

    # apply multipliers
    groups['kg_diag'] = _apply_group_multiplier(groups['kg_diag'], multipliers['kg_diag'], blend)
    groups['kg_offdiag'] = _apply_group_multiplier(groups['kg_offdiag'], multipliers['kg_offdiag'], blend)
    groups['ka_diag'] = _apply_group_multiplier(groups['ka_diag'], multipliers['ka_diag'], blend)
    groups['ka_offdiag'] = _apply_group_multiplier(groups['ka_offdiag'], multipliers['ka_offdiag'], blend)
    groups['gyro_bias'] = _apply_group_multiplier(groups['gyro_bias'], multipliers['gyro_bias'], blend)
    groups['acc_bias'] = _apply_group_multiplier(groups['acc_bias'], multipliers['acc_bias'], blend)
    groups['ka2'] = _apply_group_multiplier(groups['ka2'], multipliers['ka2'], blend)
    groups['lever'] = _apply_group_multiplier(groups['lever'], multipliers['lever'], blend)

    Kg_new = np.array(Kg_dev, dtype=float)
    Ka_new = np.array(Ka_dev, dtype=float)

    # write back diag / offdiag separately
    for i in range(3):
        Kg_new[i, i] = groups['kg_diag'][i]
        Ka_new[i, i] = groups['ka_diag'][i]
    Kg_new[offdiag_mask] = groups['kg_offdiag']
    Ka_new[offdiag_mask] = groups['ka_offdiag']

    clbt['Kg'] = np.eye(3) + Kg_new
    clbt['Ka'] = np.eye(3) + Ka_new
    clbt['eb'] = groups['gyro_bias']
    clbt['db'] = groups['acc_bias']
    clbt['Ka2'] = groups['ka2']
    clbt['rx'] = groups['lever'][0:3]
    clbt['ry'] = groups['lever'][3:6]

    log_block = {
        'global_ref': float(global_ref),
        'rms_before': {k: float(v) for k, v in rms.items()},
        'multipliers': {k: float(v) for k, v in multipliers.items()},
        'strong_consensus_multiplier': float(strong_consensus),
        'cfg': copy.deepcopy(cfg),
    }

    return clbt, log_block


def _compute_payload(source_mod, clbt, variant: str, method_file: str, extra: dict | None = None):
    param_errors, focus, lever, overall = _compute_metrics(source_mod, clbt)
    payload = {
        'variant': variant,
        'method_file': method_file,
        'source_file': str(SOURCE_FILE),
        'param_order': list(param_errors.keys()),
        'param_errors': param_errors,
        'focus_scale_pct': focus,
        'lever_guard_pct': lever,
        'overall': overall,
        'extra': extra or {},
    }
    payload['extra']['family_summary'] = _family_summary_from_payload(payload)
    return payload


def _delta_block(curr: dict, ref: dict):
    return {k: float(curr[k] - ref[k]) for k in curr}


def _max_positive_regression(delta_vs_r61: dict):
    vals = [max(0.0, float(delta_vs_r61[k])) for k in PROTECTED_DIAGNOSTICS]
    return max(vals) if vals else 0.0


def _score_candidate(delta_vs_r61: dict, delta_vs_r68best: dict, family_disp_delta: float):
    diagnostics = []
    for key in PROTECTED_DIAGNOSTICS:
        dv = float(delta_vs_r61[key])
        diagnostics.append({'metric': key, 'delta': dv, 'is_regression': bool(dv > 0.0)})

    score = 0.0
    # keep Round68 global-shape signal alive on the Round61 anchor
    score += -1.35 * float(delta_vs_r61['mean_pct_error'])
    score += -1.10 * float(delta_vs_r61['median_pct_error'])
    score += -1.45 * float(delta_vs_r61['max_pct_error'])
    score += -0.95 * float(delta_vs_r61['dKg_xx'])
    score += -0.85 * float(delta_vs_r61['dKg_yy'])
    score += -0.80 * float(delta_vs_r61['dKa_xx'])
    score += -0.75 * float(family_disp_delta)

    # Round69 repair target: reduce over-pull versus Round68 best
    for key in ROUND69_TARGET_REPAIR_METRICS:
        score += -220.0 * float(delta_vs_r68best[key])

    # diagnostics guard against heavy regressions vs Round61
    for d in diagnostics:
        dv = float(d['delta'])
        if dv > 0.20:
            score -= 110.0 * dv
        if dv > 1.00:
            score -= 420.0 * dv

    return float(score), diagnostics


def _is_clean_winner(delta_vs_r61: dict, family_disp_delta: float):
    protected_peak = _max_positive_regression(delta_vs_r61)
    return (
        float(delta_vs_r61['mean_pct_error']) < 0.0
        and float(delta_vs_r61['median_pct_error']) <= 0.0
        and float(delta_vs_r61['max_pct_error']) <= 0.0
        and float(delta_vs_r61['dKg_xx']) < 0.0
        and float(family_disp_delta) <= 0.0
        and float(protected_peak) <= 0.15
    )


def _selection_note(delta_vs_r61: dict, delta_vs_r68best: dict, family_disp_delta: float):
    protected_peak = _max_positive_regression(delta_vs_r61)
    if _is_clean_winner(delta_vs_r61, family_disp_delta):
        return 'Clean global win over Round61: global mean/median/max + family dispersion improved, diagnostics guarded.'

    repaired = [k for k in ROUND69_TARGET_REPAIR_METRICS if float(delta_vs_r68best[k]) < 0.0]
    repaired_text = ','.join(repaired) if repaired else 'none'

    improved = []
    for key in ['max_pct_error', 'dKg_xx', 'dKg_yy', 'dKa_xx']:
        if float(delta_vs_r61[key]) < 0.0:
            improved.append(key)
    if improved:
        return (
            f'Partial signal: keep {improved}; repair_vs_r68best={repaired_text}; '
            f'family_dispersion_delta={family_disp_delta:.6f}, '
            f'protected_peak_regression={protected_peak:.6f}.'
        )
    return (
        f'No useful signal: global-shape not preserved; repair_vs_r68best={repaired_text}; '
        f'protected_peak_regression={protected_peak:.6f}.'
    )


def _relative_improvement_block(baseline_payload: dict, candidate_payload: dict, keys: list[str]):
    out = {}
    for key in keys:
        if key in candidate_payload['param_errors']:
            b = float(baseline_payload['param_errors'][key]['pct_error'])
            c = float(candidate_payload['param_errors'][key]['pct_error'])
        else:
            b = float(baseline_payload['overall'][key])
            c = float(candidate_payload['overall'][key])
        out[key] = {
            'baseline_pct_error': b,
            'candidate_pct_error': c,
            'delta_pct_points': b - c,
            'relative_improvement_pct': ((b - c) / b * 100.0) if abs(b) > 1e-15 else None,
        }
    return out


def _render_report(summary: dict):
    lines = []
    lines.append('<callout emoji="🧪" background-color="light-blue">')
    lines.append('Round69 是 Round68 全局家族平衡轴的 **窄幅延续**：以 `r68_balanced_iso_mild` 为中心，只扫 `weak_consensus_blend` + multiplier envelope，并保留一个极小 family 权重修剪。')
    lines.append('</callout>')
    lines.append('')
    lines.append('## 1. Fixed mainline dataset (same as Round65/66/67/68)')
    lines.append('')
    lines.append(f"- seed: `{summary['dataset']['noise_config']['seed']}`")
    lines.append(f"- arw: `{summary['dataset']['noise_config']['arw_dpsh']} dps/√h`")
    lines.append(f"- vrw: `{summary['dataset']['noise_config']['vrw_ugpsHz']} ug/√Hz`")
    lines.append(f"- bi_g: `{summary['dataset']['noise_config']['bi_g_dph']} dph`, bi_a: `{summary['dataset']['noise_config']['bi_a_ug']} ug`")
    lines.append('- source trajectory family: `round53_internalized_trustcov_release::_build_dataset`')
    lines.append('')
    lines.append('## 2. Candidate deltas (vs Round61 + vs Round68 best)')
    lines.append('')
    lines.append('| candidate | mean Δ(R61) | max Δ(R61) | dKg_xx Δ(R61) | dKg_yy Δ(R61) | dKa_xx Δ(R61) | repair dKg_xy Δ(R68) | repair dKg_zz Δ(R68) | repair rx_y Δ(R68) | repair ry_z Δ(R68) | score |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for name in summary['candidate_order']:
        cand = summary['candidates'][name]
        d61 = cand['delta_vs_round61']
        d68 = cand['delta_vs_round68_best']
        lines.append(
            f"| `{name}` | {d61['mean_pct_error']:.6f} | {d61['max_pct_error']:.6f} | {d61['dKg_xx']:.6f} | {d61['dKg_yy']:.6f} | {d61['dKa_xx']:.6f} | {d68['dKg_xy']:.6f} | {d68['dKg_zz']:.6f} | {d68['rx_y']:.6f} | {d68['ry_z']:.6f} | {cand['selection']['score']:.6f} |"
        )
    lines.append('')
    lines.append('## 3. Decision')
    lines.append('')
    if summary['winner']:
        lines.append(f"- winner: `{summary['winner']['name']}`")
        lines.append('- decision: formalize as Round69 method')
        lines.append(f"- reason: {summary['winner']['reason']}")
    else:
        lines.append('- winner: **none**')
        lines.append('- decision: keep probe-only')
        lines.append(f"- reason: {summary['no_winner_reason']}")
    lines.append(f"- strongest signal: `{summary['strongest_signal']['name']}` / {summary['strongest_signal']['signal']}")
    lines.append(f"- next best move: {summary['next_best_move']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def _render_round_record(summary: dict):
    lines = []
    lines.append('# Round69 Record (global family-balanced continuation narrow sweep)')
    lines.append('')
    lines.append('## A. Round 基本信息')
    lines.append(f"- Round name: {summary['round_name']}")
    lines.append('- Round type: `repair branch`')
    lines.append(f"- Base candidate (Round61 anchor): `{ROUND61_BASE_NAME}`")
    lines.append('- Base candidate (Round68 center): `r68_balanced_iso_mild`')
    lines.append('- Dataset / regime: `D_ref_mainline` (same fixed noisy dataset as Round65/66/67/68)')
    lines.append('- D_ref_mainline definition:')
    lines.append('  - source trajectory: `method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset`')
    lines.append(f"  - arw = `{summary['dataset']['noise_config']['arw_dpsh']} * dpsh`")
    lines.append(f"  - vrw = `{summary['dataset']['noise_config']['vrw_ugpsHz']} * ugpsHz`")
    lines.append(f"  - bi_g = `{summary['dataset']['noise_config']['bi_g_dph']} * dph`")
    lines.append(f"  - bi_a = `{summary['dataset']['noise_config']['bi_a_ug']} * ug`")
    lines.append(f"  - tau_g = tau_a = `{summary['dataset']['noise_config']['tau_g']}`")
    lines.append(f"  - seed = `{summary['dataset']['noise_config']['seed']}`")
    lines.append('')
    lines.append('## B. Chosen mechanism / sweep knobs')
    lines.append('- mechanism: `global family-balanced grouped reconciliation continuation on Round61 backbone`')
    lines.append('- objective: preserve Round68 global structure signal while reducing over-pull on dKg_xy / dKg_zz / rx_y / ry_z.')
    lines.append('- allowed sweep knobs (narrow only):')
    lines.append('  - `weak_consensus_blend`')
    lines.append('  - multiplier envelope (`mult_min`, `mult_max`)')
    lines.append('  - one tiny family weight trim (optional, single candidate only)')
    lines.append('')
    lines.append('## C. Clean-win gate')
    lines.append('- same-dataset vs Round61: mean<0, median<=0, max<=0, dKg_xx<0')
    lines.append('- family_dispersion_delta_vs_round61 <= 0')
    lines.append('- protected_peak_regression <= 0.15 pct-point')
    lines.append('- only clean winner can be formalized')
    lines.append('')
    lines.append('## D. Candidate batch (deterministic, narrow)')
    for idx, candidate in enumerate(ROUND69_CANDIDATES, start=1):
        lines.append(f"### candidate {idx}")
        lines.append(f"- name: `{candidate['name']}`")
        lines.append(f"- rationale: {candidate['rationale']}")
        lines.append(f"- family_balance_cfg: `{json.dumps(candidate['family_balance'], ensure_ascii=False)}`")
        lines.append('')
    lines.append('## E. Result summary')
    lines.append(f"- winner: `{summary['winner']['name']}`" if summary['winner'] else '- winner: none')
    lines.append(f"- result class: `{summary['result_classification']}`")
    lines.append(f"- one-line conclusion: {summary['conclusion_line']}")
    lines.append(f"- strongest signal: {summary['strongest_signal']['signal']}")
    lines.append('')
    lines.append('## F. Mechanism learning and next move')
    lines.append(f"- mechanism learning: {summary['mechanism_learning']}")
    lines.append(f"- next best move: {summary['next_best_move']}")
    lines.append('')
    lines.append('## G. Artifacts')
    lines.append(f"- candidate_json: `{CANDIDATE_JSON}`")
    lines.append(f"- summary_json: `{OUTPUT_JSON}`")
    lines.append(f"- report_md: `{REPORT_MD}`")
    lines.append(f"- formal_method_file: `{summary.get('formal_method_file')}`")
    lines.append(f"- formal_result_json: `{summary.get('formal_result_json')}`")
    lines.append('')
    return '\n'.join(lines)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    round61_payload = json.loads(ROUND61_REF_JSON.read_text(encoding='utf-8'))
    round68_best_payload = json.loads(ROUND68_BEST_JSON.read_text(encoding='utf-8'))
    baseline_family_summary = _family_summary_from_payload(round61_payload)

    source_mod = load_module('markov_pruned_source_round69', str(SOURCE_FILE))
    dataset = _build_shared_dataset(source_mod)

    candidate_dump = {
        'round_name': 'Round69_Global_Family_Balanced',
        'round_type': 'repair branch',
        'mechanism_axis': 'global family-balanced calibration / grouped reconciliation continuation',
        'base_round61_candidate': ROUND61_BASE_NAME,
        'base_round68_candidate': 'r68_balanced_iso_mild',
        'same_dataset_round61_json': str(ROUND61_REF_JSON),
        'same_dataset_round68_best_json': str(ROUND68_BEST_JSON),
        'dataset': {
            'source_file': str(SOURCE_FILE),
            'source_trajectory_reference': 'method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset',
            'noise_config': dataset['noise_config'],
            'constraint_note': 'Use exactly the same fixed noisy dataset/noise strength/seed as Round65/66/67/68 mainline',
        },
        'round69_objective': {
            'preserve_round68_signal': ['dKg_xx', 'dKg_yy', 'dKa_xx', 'max_pct_error', 'family_dispersion'],
            'repair_overpull': ROUND69_TARGET_REPAIR_METRICS,
            'diagnostics_guard': PROTECTED_DIAGNOSTICS,
        },
        'sweep_knobs': ['weak_consensus_blend', 'mult_min/mult_max envelope', 'tiny family weight trim (optional)'],
        'clean_win_gate': 'same-dataset vs Round61: mean<0, median<=0, max<=0, dKg_xx<0, family_dispersion_delta<=0, protected_peak_regression<=0.15',
        'round69_candidates': ROUND69_CANDIDATES,
    }
    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    out = {
        'round_name': 'Round69_Global_Family_Balanced',
        'round_type': 'repair branch',
        'mechanism': 'Round61 backbone + global family-balanced grouped reconciliation continuation (narrow sweep)',
        'dataset': {
            'source_file': str(SOURCE_FILE),
            'noise_config': dataset['noise_config'],
            'seed': dataset['noise_config']['seed'],
        },
        'base_round61_candidate': ROUND61_BASE_NAME,
        'base_round68_candidate': 'r68_balanced_iso_mild',
        'base_round61_json': str(ROUND61_REF_JSON),
        'base_round68_best_json': str(ROUND68_BEST_JSON),
        'candidate_json': str(CANDIDATE_JSON),
        'candidate_order': [c['name'] for c in ROUND69_CANDIDATES],
        'candidates': {},
        'winner': None,
        'no_winner_reason': None,
        'result_classification': None,
        'strongest_signal': None,
        'next_best_move': None,
        'formal_method_file': None,
        'formal_result_json': None,
    }

    ts = dataset['ts']
    pos0 = dataset['pos0']
    imu_noisy = dataset['imu_noisy']
    bi_g = dataset['bi_g']
    bi_a = dataset['bi_a']
    tau_g = dataset['tau_g']
    tau_a = dataset['tau_a']

    for idx, candidate in enumerate(ROUND69_CANDIDATES, start=1):
        merged_candidate = _merge_round69_candidate(candidate)

        method_mod = load_module(f'markov_method_round69_candidate_{idx}', str(R53_METHOD_FILE))
        method_mod = _build_patched_method(method_mod, merged_candidate)

        result = list(_run_internalized_hybrid_scd(
            method_mod,
            source_mod,
            imu_noisy,
            pos0,
            ts,
            bi_g=bi_g,
            bi_a=bi_a,
            tau_g=tau_g,
            tau_a=tau_a,
            label=f'R69-FAMBAL-{idx}',
            scd_cfg=merged_candidate['scd'],
        ))
        clbt_candidate_raw = result[0]
        runtime_log = {
            'schedule_log': result[4].get('schedule_log') if len(result) >= 5 else None,
            'feedback_log': result[4].get('feedback_log') if len(result) >= 5 else None,
            'scd_log': result[4].get('scd_log') if len(result) >= 5 else None,
        }
        del result
        gc.collect()

        clbt_candidate_bal, family_balance_log = _apply_global_family_balance(source_mod, clbt_candidate_raw, candidate['family_balance'])
        del clbt_candidate_raw
        gc.collect()

        payload_candidate = _compute_payload(
            source_mod,
            clbt_candidate_bal,
            variant=f"r69_global_family_{candidate['name']}",
            method_file='probe_round69_global_family_balanced::round61_backbone_plus_family_reconcile',
            extra={
                'dataset_noise_config': dataset['noise_config'],
                'base_round61_candidate': ROUND61_BASE_NAME,
                'base_round68_candidate': 'r68_balanced_iso_mild',
                'family_balance_cfg': copy.deepcopy(candidate['family_balance']),
                'family_balance_log': family_balance_log,
                'runtime_log': runtime_log,
            },
        )
        del clbt_candidate_bal
        gc.collect()

        candidate_json_path = RESULTS_DIR / f"R69_global_family_{candidate['name']}_param_errors.json"
        candidate_json_path.write_text(json.dumps(payload_candidate, ensure_ascii=False, indent=2), encoding='utf-8')

        delta_vs_r61 = {
            **_delta_block(payload_candidate['focus_scale_pct'], round61_payload['focus_scale_pct']),
            **_delta_block(payload_candidate['lever_guard_pct'], round61_payload['lever_guard_pct']),
            **_delta_block(payload_candidate['overall'], round61_payload['overall']),
        }
        delta_vs_r68best = {
            **_delta_block(payload_candidate['focus_scale_pct'], round68_best_payload['focus_scale_pct']),
            **_delta_block(payload_candidate['lever_guard_pct'], round68_best_payload['lever_guard_pct']),
            **_delta_block(payload_candidate['overall'], round68_best_payload['overall']),
        }

        family_disp_cand = float(payload_candidate['extra']['family_summary']['family_dispersion'])
        family_disp_base = float(baseline_family_summary['family_dispersion'])
        family_disp_delta = family_disp_cand - family_disp_base

        score, diagnostics = _score_candidate(delta_vs_r61, delta_vs_r68best, family_disp_delta)
        note = _selection_note(delta_vs_r61, delta_vs_r68best, family_disp_delta)
        protected_peak_reg = _max_positive_regression(delta_vs_r61)

        repair_delta = {k: float(delta_vs_r68best[k]) for k in ROUND69_TARGET_REPAIR_METRICS}
        repair_improved = [k for k, dv in repair_delta.items() if dv < 0.0]

        out['candidates'][candidate['name']] = {
            'description': candidate['description'],
            'rationale': candidate['rationale'],
            'family_balance_cfg': copy.deepcopy(candidate['family_balance']),
            'param_errors_json': str(candidate_json_path),
            'focus_scale_pct': payload_candidate['focus_scale_pct'],
            'lever_guard_pct': payload_candidate['lever_guard_pct'],
            'overall': payload_candidate['overall'],
            'delta_vs_round61': delta_vs_r61,
            'delta_vs_round68_best': delta_vs_r68best,
            'repair_target_delta_vs_round68_best': repair_delta,
            'repair_target_improved_count': len(repair_improved),
            'family_summary': payload_candidate['extra']['family_summary'],
            'family_dispersion_delta_vs_round61': float(family_disp_delta),
            'protected_peak_regression': float(protected_peak_reg),
            'selection': {
                'score': float(score),
                'diagnostics': diagnostics,
                'note': note,
            },
            'runtime_log': payload_candidate['extra']['runtime_log'],
            'family_balance_log': payload_candidate['extra']['family_balance_log'],
            'vs_round61_relative_improvement': _relative_improvement_block(
                round61_payload,
                payload_candidate,
                ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'mean_pct_error', 'median_pct_error', 'max_pct_error'],
            ),
            'vs_round68_best_relative_improvement': _relative_improvement_block(
                round68_best_payload,
                payload_candidate,
                ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'mean_pct_error', 'median_pct_error', 'max_pct_error'],
            ),
        }

        print(candidate['name'], json.dumps({
            'delta_vs_round61': delta_vs_r61,
            'delta_vs_round68_best': delta_vs_r68best,
            'family_dispersion_delta_vs_round61': family_disp_delta,
            'repair_target_delta_vs_round68_best': repair_delta,
            'protected_peak_regression': protected_peak_reg,
            'score': score,
            'note': note,
        }, ensure_ascii=False))

    ordered = sorted(
        [(name, out['candidates'][name]['selection']['score']) for name in out['candidate_order']],
        key=lambda x: x[1],
        reverse=True,
    )
    best_name, best_score = ordered[0]
    best = out['candidates'][best_name]
    best_delta = best['delta_vs_round61']
    best_delta_vs_r68 = best['delta_vs_round68_best']
    best_disp_delta = float(best['family_dispersion_delta_vs_round61'])

    if _is_clean_winner(best_delta, best_disp_delta):
        out['winner'] = {
            'name': best_name,
            'score': float(best_score),
            'reason': 'Clean same-dataset winner over Round61 while preserving global family-balanced structure.',
        }
        out['result_classification'] = 'clean win'
        out['conclusion_line'] = 'Round69 produced a clean same-dataset winner over Round61.'

        formal_method_file = METHOD_DIR / f"method_42state_gm1_round69_global_family_balanced_{best_name}.py"
        formal_result_json = RESULTS_DIR / f"R69_42state_gm1_round69_global_family_balanced_{best_name}_param_errors.json"
        formal_method_file.write_text(
            (
                '# Auto-generated Round69 formalization placeholder.\n'
                '# Winner configuration is recorded in results/round69_probe_summary.json and round69_candidates.json.\n'
                '# For reproducibility, use psins_method_bench/scripts/probe_round69_global_family_balanced.py with the winner candidate.\n'
            ),
            encoding='utf-8',
        )
        formal_result_json.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding='utf-8')
        out['formal_method_file'] = str(formal_method_file)
        out['formal_result_json'] = str(formal_result_json)
    else:
        out['winner'] = None
        out['no_winner_reason'] = (
            'No candidate passed the Round61 clean-win gate on the fixed dataset; '
            'Round69 remains probe-only as a continuation of Round68 global family-balanced axis.'
        )
        has_partial = (
            float(best_delta['max_pct_error']) < 0.0
            or float(best_delta['dKg_xx']) < 0.0
            or any(float(best_delta_vs_r68[k]) < 0.0 for k in ROUND69_TARGET_REPAIR_METRICS)
        )
        out['result_classification'] = 'partial signal' if has_partial else 'no useful signal'
        out['conclusion_line'] = 'Round69 did not produce a clean promotable winner over Round61 on the fixed mainline dataset.'

    out['strongest_signal'] = {
        'name': best_name,
        'signal': (
            f"best candidate {best_name}: "
            f"vsR61 max Δ={best_delta['max_pct_error']:.6f}, dKg_xx Δ={best_delta['dKg_xx']:.6f}, "
            f"dKg_yy Δ={best_delta['dKg_yy']:.6f}, dKa_xx Δ={best_delta['dKa_xx']:.6f}, "
            f"family_dispersion Δ={best_disp_delta:.6f}, protected_peak Δ={best['protected_peak_regression']:.6f}; "
            f"vsR68best repair Δ(dKg_xy/dKg_zz/rx_y/ry_z)=({best_delta_vs_r68['dKg_xy']:.6f}, {best_delta_vs_r68['dKg_zz']:.6f}, {best_delta_vs_r68['rx_y']:.6f}, {best_delta_vs_r68['ry_z']:.6f})"
        ),
        'regressions': [
            {'metric': k, 'delta': float(best_delta[k])} for k in PROTECTED_DIAGNOSTICS if float(best_delta[k]) > 0.0
        ],
        'repair_target_delta_vs_round68_best': {
            k: float(best_delta_vs_r68[k]) for k in ROUND69_TARGET_REPAIR_METRICS
        },
    }

    out['mechanism_learning'] = (
        'Round69 confirms the Round68 global-family reconciliation axis can be narrowed deterministically by weak_consensus_blend '
        'and cap envelope controls; the trade-off remains global-shape retention vs targeted over-pull repair.'
    )
    out['next_best_move'] = (
        'If no clean winner, keep the best Round69 candidate and run one more ultra-narrow 1D sweep on a single knob '
        '(prefer weak_consensus_blend only) with no extra weight trim, to isolate causality on dKg_xy/rx_y/ry_z.'
    )

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    REPORT_MD.write_text(_render_report(out), encoding='utf-8')
    ROUND69_RECORD_MD.write_text(_render_round_record(out), encoding='utf-8')

    print(f'Wrote {CANDIDATE_JSON}')
    print(f'Wrote {OUTPUT_JSON}')
    print(f'Wrote {REPORT_MD}')
    print(f'Wrote {ROUND69_RECORD_MD}')
    print('__RESULT_JSON__=' + json.dumps({
        'candidate_json': str(CANDIDATE_JSON),
        'summary_json': str(OUTPUT_JSON),
        'report_md': str(REPORT_MD),
        'round_record_md': str(ROUND69_RECORD_MD),
        'winner': out['winner'],
        'result_classification': out['result_classification'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
