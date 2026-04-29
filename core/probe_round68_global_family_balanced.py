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

OUTPUT_JSON = RESULTS_DIR / 'round68_probe_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'round68_candidates.json'
REPORT_MD = REPORTS_DIR / 'psins_round68_probe_2026-03-28.md'
ROUND68_RECORD_MD = SUMMARY_DIR / 'round68_record_2026-03-28.md'

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

ROUND68_CANDIDATES = [
    {
        'name': 'r68_balanced_iso_mild',
        'description': 'Round61 backbone + mild family-balanced isotropic reconciliation across all parameter families.',
        'rationale': '最小改动验证：先看统一 family 归一化目标是否能在不破坏主干的前提下提升全局误差形态。',
        'family_balance': {
            'gamma': 0.14,
            'blend': 0.72,
            'mult_min': 0.92,
            'mult_max': 1.10,
            'weak_consensus_blend': 0.22,
            'weights': {
                'kg_diag': 1.0,
                'kg_offdiag': 1.0,
                'ka_diag': 1.0,
                'ka_offdiag': 1.0,
                'gyro_bias': 1.0,
                'acc_bias': 1.0,
                'ka2': 0.9,
                'lever': 0.9,
            },
        },
    },
    {
        'name': 'r68_balanced_strong_anchor',
        'description': 'Family-balanced reconciliation with stronger anchor on strong families (Kg/Ka/bias), weak families gently co-moved.',
        'rationale': '把全局目标优先放在强可观参数家族，避免“只修弱参数外观”造成的伪全局改进。',
        'family_balance': {
            'gamma': 0.18,
            'blend': 0.78,
            'mult_min': 0.90,
            'mult_max': 1.12,
            'weak_consensus_blend': 0.35,
            'weights': {
                'kg_diag': 1.15,
                'kg_offdiag': 1.15,
                'ka_diag': 1.10,
                'ka_offdiag': 1.10,
                'gyro_bias': 1.05,
                'acc_bias': 1.05,
                'ka2': 0.75,
                'lever': 0.70,
            },
        },
    },
    {
        'name': 'r68_balanced_weak_consensus',
        'description': 'Family-balanced reconciliation with consensus-style weak-family freeze/refine coupling.',
        'rationale': '把 weak families（Ka2/lever）与强家族一致性绑定，测试“全局协调 + 冻结弱扰动”是否更稳。',
        'family_balance': {
            'gamma': 0.20,
            'blend': 0.80,
            'mult_min': 0.88,
            'mult_max': 1.14,
            'weak_consensus_blend': 0.55,
            'weights': {
                'kg_diag': 1.10,
                'kg_offdiag': 1.05,
                'ka_diag': 1.10,
                'ka_offdiag': 1.05,
                'gyro_bias': 1.00,
                'acc_bias': 1.00,
                'ka2': 0.85,
                'lever': 0.80,
            },
        },
    },
    {
        'name': 'r68_balanced_tightcap',
        'description': 'Two-stage family reconciliation with tighter multipliers to emphasize deterministic global shape control.',
        'rationale': '以更紧的乘子上限约束全局重分配，验证“窄幅但系统化”的 family 归一是否更可靠。',
        'family_balance': {
            'gamma': 0.16,
            'blend': 0.68,
            'mult_min': 0.95,
            'mult_max': 1.06,
            'weak_consensus_blend': 0.28,
            'weights': {
                'kg_diag': 1.00,
                'kg_offdiag': 1.00,
                'ka_diag': 1.00,
                'ka_offdiag': 1.00,
                'gyro_bias': 0.95,
                'acc_bias': 0.95,
                'ka2': 0.85,
                'lever': 0.85,
            },
        },
    },
]


def _load_round61_base_candidate():
    for candidate in ROUND61_CANDIDATES:
        if candidate['name'] == ROUND61_BASE_NAME:
            return _merge_round61_candidate(candidate)
    raise KeyError(ROUND61_BASE_NAME)


def _merge_round68_candidate(extra_candidate: dict):
    merged = copy.deepcopy(_load_round61_base_candidate())
    merged['name'] = extra_candidate['name']
    merged['description'] = extra_candidate['description']
    merged['rationale'] = extra_candidate['rationale']
    merged['family_balance'] = copy.deepcopy(extra_candidate['family_balance'])
    merged['round68_extra_patch'] = copy.deepcopy(extra_candidate)
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


def _score_candidate(delta_vs_r61: dict, family_disp_delta: float):
    diagnostics = []
    for key in PROTECTED_DIAGNOSTICS:
        dv = float(delta_vs_r61[key])
        diagnostics.append({'metric': key, 'delta': dv, 'is_regression': bool(dv > 0.0)})

    score = 0.0
    # global objective first
    score += -2.00 * float(delta_vs_r61['mean_pct_error'])
    score += -1.60 * float(delta_vs_r61['median_pct_error'])
    score += -1.85 * float(delta_vs_r61['max_pct_error'])
    score += -0.90 * float(delta_vs_r61['dKg_xx'])
    score += -0.60 * float(delta_vs_r61['dKg_zz'])
    score += -0.70 * float(family_disp_delta)

    # diagnostics as guard: soft + hard tiers
    for d in diagnostics:
        dv = float(d['delta'])
        if dv > 0.20:
            score -= 120.0 * dv
        if dv > 1.00:
            score -= 500.0 * dv

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


def _selection_note(delta_vs_r61: dict, family_disp_delta: float):
    protected_peak = _max_positive_regression(delta_vs_r61)
    if _is_clean_winner(delta_vs_r61, family_disp_delta):
        return 'Clean global win over Round61: global mean/median/max + family dispersion improved, diagnostics guarded.'

    improved = []
    for key in ['mean_pct_error', 'median_pct_error', 'max_pct_error', 'dKg_xx', 'dKg_zz']:
        if float(delta_vs_r61[key]) < 0.0:
            improved.append(key)
    if improved:
        return (
            f'Partial global signal on {improved}; '
            f'family_dispersion_delta={family_disp_delta:.6f}, '
            f'protected_peak_regression={protected_peak:.6f}.'
        )
    return (
        'No useful global signal: global objective not improved and/or diagnostics guard violated.'
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
    lines.append('<callout emoji="🌐" background-color="light-blue">')
    lines.append('Round68 把主语切到 **global family-balanced calibration / grouped reconciliation**：Round61 作为稳定 backbone，新增机制只做全家族协调，不做局部参数补丁故事。')
    lines.append('</callout>')
    lines.append('')
    lines.append('## 1. Fixed mainline dataset (same as Round65/66/67)')
    lines.append('')
    lines.append(f"- seed: `{summary['dataset']['noise_config']['seed']}`")
    lines.append(f"- arw: `{summary['dataset']['noise_config']['arw_dpsh']} dps/√h`")
    lines.append(f"- vrw: `{summary['dataset']['noise_config']['vrw_ugpsHz']} ug/√Hz`")
    lines.append(f"- bi_g: `{summary['dataset']['noise_config']['bi_g_dph']} dph`, bi_a: `{summary['dataset']['noise_config']['bi_a_ug']} ug`")
    lines.append('- source trajectory family: `round53_internalized_trustcov_release::_build_dataset`')
    lines.append('')
    lines.append('## 2. Round68 candidates vs Round61 (global objective + diagnostics)')
    lines.append('')
    lines.append('| candidate | mean Δ | median Δ | max Δ | dKg_xx Δ | dKg_zz Δ | family_disp Δ | protected_peak Δ | score | note |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for name in summary['candidate_order']:
        cand = summary['candidates'][name]
        d = cand['delta_vs_round61']
        lines.append(
            f"| `{name}` | {d['mean_pct_error']:.6f} | {d['median_pct_error']:.6f} | {d['max_pct_error']:.6f} | {d['dKg_xx']:.6f} | {d['dKg_zz']:.6f} | {cand['family_dispersion_delta_vs_round61']:.6f} | {cand['protected_peak_regression']:.6f} | {cand['selection']['score']:.6f} | {cand['selection']['note']} |"
        )
    lines.append('')
    lines.append('## 3. Decision')
    lines.append('')
    if summary['winner']:
        lines.append(f"- winner: `{summary['winner']['name']}`")
        lines.append('- decision: formalize as Round68 method')
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
    lines.append('# Round68 Record (global family-balanced calibration probe)')
    lines.append('')
    lines.append('## A. Round 基本信息')
    lines.append(f"- Round name: {summary['round_name']}")
    lines.append('- Round type: `new mechanism probe`')
    lines.append(f"- Base candidate: `{ROUND61_BASE_NAME}`")
    lines.append('- Dataset / regime: `D_ref_mainline` (same fixed noisy dataset as Round65/66/67)')
    lines.append('- D_ref_mainline definition:')
    lines.append('  - source trajectory: `method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset`')
    lines.append(f"  - arw = `{summary['dataset']['noise_config']['arw_dpsh']} * dpsh`")
    lines.append(f"  - vrw = `{summary['dataset']['noise_config']['vrw_ugpsHz']} * ugpsHz`")
    lines.append(f"  - bi_g = `{summary['dataset']['noise_config']['bi_g_dph']} * dph`")
    lines.append(f"  - bi_a = `{summary['dataset']['noise_config']['bi_a_ug']} * ug`")
    lines.append(f"  - tau_g = tau_a = `{summary['dataset']['noise_config']['tau_g']}`")
    lines.append(f"  - seed = `{summary['dataset']['noise_config']['seed']}`")
    lines.append('')
    lines.append('## B. Chosen mechanism / global objective framing')
    lines.append('- mechanism: `global family-balanced grouped reconciliation on top of Round61 backbone`')
    lines.append('- framing: calibration quality is a global multi-family objective, not a local patch objective.')
    lines.append('- objective priority: global mean / median / max + family-dispersion reduction.')
    lines.append('- protected diagnostics (guard only, not sole objective): dKg_xy / dKg_yy / dKa_xx / rx_y / ry_z.')
    lines.append('')
    lines.append('## C. Clean-win gate')
    lines.append('- mean<0, median<=0, max<=0, dKg_xx<0')
    lines.append('- family_dispersion_delta_vs_round61 <= 0')
    lines.append('- protected_peak_regression <= 0.15 pct-point')
    lines.append('- only clean winner can be formalized')
    lines.append('')
    lines.append('## D. Candidate batch (deterministic, narrow)')
    for idx, candidate in enumerate(ROUND68_CANDIDATES, start=1):
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
    baseline_family_summary = _family_summary_from_payload(round61_payload)

    source_mod = load_module('markov_pruned_source_round68', str(SOURCE_FILE))
    dataset = _build_shared_dataset(source_mod)

    candidate_dump = {
        'round_name': 'Round68_Global_Family_Balanced',
        'round_type': 'new mechanism probe',
        'mechanism_axis': 'global family-balanced calibration / grouped reconciliation on top of Round61 backbone',
        'base_round61_candidate': ROUND61_BASE_NAME,
        'same_dataset_round61_json': str(ROUND61_REF_JSON),
        'dataset': {
            'source_file': str(SOURCE_FILE),
            'source_trajectory_reference': 'method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset',
            'noise_config': dataset['noise_config'],
            'constraint_note': 'Use exactly the same fixed noisy dataset/noise strength/seed as Round65/66/67 mainline',
        },
        'global_objective': {
            'primary': ['mean_pct_error', 'median_pct_error', 'max_pct_error', 'dKg_xx'],
            'family_balance_metric': 'family_dispersion',
            'diagnostics_guard': PROTECTED_DIAGNOSTICS,
        },
        'clean_win_gate': 'mean<0, median<=0, max<=0, dKg_xx<0, family_dispersion_delta<=0, protected_peak_regression<=0.15',
        'round68_candidates': ROUND68_CANDIDATES,
    }
    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    out = {
        'round_name': 'Round68_Global_Family_Balanced',
        'round_type': 'new mechanism probe',
        'mechanism': 'Round61 backbone + global family-balanced grouped reconciliation',
        'dataset': {
            'source_file': str(SOURCE_FILE),
            'noise_config': dataset['noise_config'],
            'seed': dataset['noise_config']['seed'],
        },
        'base_round61_candidate': ROUND61_BASE_NAME,
        'base_round61_json': str(ROUND61_REF_JSON),
        'candidate_json': str(CANDIDATE_JSON),
        'candidate_order': [c['name'] for c in ROUND68_CANDIDATES],
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

    for idx, candidate in enumerate(ROUND68_CANDIDATES, start=1):
        merged_candidate = _merge_round68_candidate(candidate)

        method_mod = load_module(f'markov_method_round68_candidate_{idx}', str(R53_METHOD_FILE))
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
            label=f'R68-FAMBAL-{idx}',
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
            variant=f"r68_global_family_{candidate['name']}",
            method_file='probe_round68_global_family_balanced::round61_backbone_plus_family_reconcile',
            extra={
                'dataset_noise_config': dataset['noise_config'],
                'base_round61_candidate': ROUND61_BASE_NAME,
                'family_balance_cfg': copy.deepcopy(candidate['family_balance']),
                'family_balance_log': family_balance_log,
                'runtime_log': runtime_log,
            },
        )
        del clbt_candidate_bal
        gc.collect()

        candidate_json_path = RESULTS_DIR / f"R68_global_family_{candidate['name']}_param_errors.json"
        candidate_json_path.write_text(json.dumps(payload_candidate, ensure_ascii=False, indent=2), encoding='utf-8')

        delta_vs_r61 = {
            **_delta_block(payload_candidate['focus_scale_pct'], round61_payload['focus_scale_pct']),
            **_delta_block(payload_candidate['lever_guard_pct'], round61_payload['lever_guard_pct']),
            **_delta_block(payload_candidate['overall'], round61_payload['overall']),
        }

        family_disp_cand = float(payload_candidate['extra']['family_summary']['family_dispersion'])
        family_disp_base = float(baseline_family_summary['family_dispersion'])
        family_disp_delta = family_disp_cand - family_disp_base

        score, diagnostics = _score_candidate(delta_vs_r61, family_disp_delta)
        note = _selection_note(delta_vs_r61, family_disp_delta)
        protected_peak_reg = _max_positive_regression(delta_vs_r61)

        out['candidates'][candidate['name']] = {
            'description': candidate['description'],
            'rationale': candidate['rationale'],
            'family_balance_cfg': copy.deepcopy(candidate['family_balance']),
            'param_errors_json': str(candidate_json_path),
            'focus_scale_pct': payload_candidate['focus_scale_pct'],
            'lever_guard_pct': payload_candidate['lever_guard_pct'],
            'overall': payload_candidate['overall'],
            'delta_vs_round61': delta_vs_r61,
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
        }

        print(candidate['name'], json.dumps({
            'delta_vs_round61': delta_vs_r61,
            'family_dispersion_delta_vs_round61': family_disp_delta,
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
    best_disp_delta = float(best['family_dispersion_delta_vs_round61'])

    if _is_clean_winner(best_delta, best_disp_delta):
        out['winner'] = {
            'name': best_name,
            'score': float(best_score),
            'reason': 'Clean global same-dataset winner over Round61 under family-balanced grouped reconciliation.',
        }
        out['result_classification'] = 'clean win'
        out['conclusion_line'] = 'Round68 produced a clean same-dataset global winner over Round61.'

        formal_method_file = METHOD_DIR / f"method_42state_gm1_round68_global_family_balanced_{best_name}.py"
        formal_result_json = RESULTS_DIR / f"R68_42state_gm1_round68_global_family_balanced_{best_name}_param_errors.json"
        formal_method_file.write_text(
            (
                '# Auto-generated Round68 formalization placeholder.\n'
                '# Winner configuration is recorded in results/round68_probe_summary.json and round68_candidates.json.\n'
                '# For reproducibility, use psins_method_bench/scripts/probe_round68_global_family_balanced.py with the winner candidate.\n'
            ),
            encoding='utf-8',
        )
        formal_result_json.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding='utf-8')
        out['formal_method_file'] = str(formal_method_file)
        out['formal_result_json'] = str(formal_result_json)
    else:
        out['winner'] = None
        out['no_winner_reason'] = (
            'No candidate passed the Round61 clean-win gate under the same fixed dataset '
            'for the Round68 global family-balanced reconciliation axis.'
        )
        if float(best_delta['mean_pct_error']) < 0.0 or float(best_delta['median_pct_error']) < 0.0 or float(best_delta['max_pct_error']) < 0.0:
            out['result_classification'] = 'partial signal'
        else:
            out['result_classification'] = 'no useful signal'
        out['conclusion_line'] = 'Round68 did not produce a clean promotable winner over Round61 on the fixed mainline dataset.'

    out['strongest_signal'] = {
        'name': best_name,
        'signal': (
            f"best candidate {best_name}: "
            f"mean Δ={best_delta['mean_pct_error']:.6f}, median Δ={best_delta['median_pct_error']:.6f}, "
            f"max Δ={best_delta['max_pct_error']:.6f}, dKg_xx Δ={best_delta['dKg_xx']:.6f}, "
            f"family_dispersion Δ={best_disp_delta:.6f}, protected_peak Δ={best['protected_peak_regression']:.6f}"
        ),
        'regressions': [
            {'metric': k, 'delta': float(best_delta[k])} for k in PROTECTED_DIAGNOSTICS if float(best_delta[k]) > 0.0
        ],
    }

    out['mechanism_learning'] = (
        'Round68 keeps Round61 as a stable estimation backbone and moves adaptation to a post-estimation '
        'global family-balance reconciliation layer, which is globally motivated and interpretable by family-level RMS multipliers.'
    )
    out['next_best_move'] = (
        'Fix the best Round68 candidate and run a one-knob narrow sweep on weak_consensus_blend or multiplier cap only, '
        'to see whether protected_peak regression can be pushed below the clean gate without losing global mean/median/max.'
    )

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    REPORT_MD.write_text(_render_report(out), encoding='utf-8')
    ROUND68_RECORD_MD.write_text(_render_round_record(out), encoding='utf-8')

    print(f'Wrote {CANDIDATE_JSON}')
    print(f'Wrote {OUTPUT_JSON}')
    print(f'Wrote {REPORT_MD}')
    print(f'Wrote {ROUND68_RECORD_MD}')
    print('__RESULT_JSON__=' + json.dumps({
        'candidate_json': str(CANDIDATE_JSON),
        'summary_json': str(OUTPUT_JSON),
        'report_md': str(REPORT_MD),
        'round_record_md': str(ROUND68_RECORD_MD),
        'winner': out['winner'],
        'result_classification': out['result_classification'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
