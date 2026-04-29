from __future__ import annotations

import copy
import gc
import json
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

OUTPUT_JSON = RESULTS_DIR / 'round66_probe_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'round66_candidates.json'
REPORT_MD = REPORTS_DIR / 'psins_round66_probe_2026-03-28.md'
ROUND66_RECORD_MD = SUMMARY_DIR / 'round66_record_2026-03-28.md'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import load_module
from probe_round55_newline import _build_patched_method
from probe_round56_narrow import _compute_metrics
from probe_round61_hybrid_micro import ROUND61_CANDIDATES, _merge_round61_candidate
from probe_round65_mainline_innovation_consistency import _build_shared_dataset
from probe_round65b_dualgate_repair import _run_internalized_hybrid_scd_dualgate


ROUND61_BASE_NAME = 'r61_s20_08988_ryz00116'

HARD_PROTECTED_KEYS = ['dKg_xy', 'dKg_yy', 'dKa_xx', 'rx_y', 'ry_z']
TARGET_KEYS = ['mean_pct_error', 'max_pct_error', 'dKg_xx', 'dKg_zz']

NOOP_FEEDBACK_CHANNEL = {
    'target_nis': 1.0,
    'ema_beta': 0.08,
    'slope': 1.0,
    'gate_floor': 1.0,
    'warmup_static_meas': 0,
    'power': 1.0,
    'apply_floor': 1.0,
}

ROUND66_CANDIDATES = [
    {
        'name': 'r66_scd_xxzz_consis_mild_f90',
        'description': 'Consistency-adaptive SCD only on xx/zz with conservative floor, keeping Round61 feedback fully frozen.',
        'rationale': 'Most conservative new-direction probe: only allow tiny SCD modulation on xx/zz while preserving Round61 feedback body unchanged.',
        'scd_channel': {
            'target_nis': 1.0,
            'ema_beta': 0.10,
            'slope': 1.00,
            'gate_floor': 0.90,
            'warmup_static_meas': 8,
            'power': 1.00,
            'apply_floor': 0.90,
        },
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.9990,
            'transition_duration': 2.0,
            'target': 'xxzz_pair',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    },
    {
        'name': 'r66_scd_xxzz_consis_bal_f82',
        'description': 'Balanced xx/zz-only adaptive SCD with moderate floor and slope.',
        'rationale': 'Test whether moderate SCD-only adaptation can recover dKg_zz without leaking regressions to yy/Ka_xx-protected paths.',
        'scd_channel': {
            'target_nis': 1.0,
            'ema_beta': 0.09,
            'slope': 1.45,
            'gate_floor': 0.82,
            'warmup_static_meas': 8,
            'power': 1.20,
            'apply_floor': 0.82,
        },
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.9989,
            'transition_duration': 2.0,
            'target': 'xxzz_pair',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    },
    {
        'name': 'r66_scd_xxzz_consis_slowema_f85',
        'description': 'xx/zz-only adaptive SCD with slower consistency EMA and higher floor.',
        'rationale': 'Reduce gate jitter to keep mechanism deterministic/interpretable and avoid overreacting to transient innovation fluctuations.',
        'scd_channel': {
            'target_nis': 1.0,
            'ema_beta': 0.05,
            'slope': 1.35,
            'gate_floor': 0.85,
            'warmup_static_meas': 10,
            'power': 1.15,
            'apply_floor': 0.85,
        },
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.9988,
            'transition_duration': 2.0,
            'target': 'xxzz_pair',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    },
    {
        'name': 'r66_scd_xxzz_consis_push_f75',
        'description': 'More aggressive xx/zz-only adaptive SCD gate to stress-test the narrow mechanism limit.',
        'rationale': 'Probe upper bound of narrow SCD-only adaptation strength to check whether a stronger xx/zz correction can beat Round61 cleanly.',
        'scd_channel': {
            'target_nis': 1.0,
            'ema_beta': 0.10,
            'slope': 1.80,
            'gate_floor': 0.75,
            'warmup_static_meas': 8,
            'power': 1.35,
            'apply_floor': 0.75,
        },
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.9987,
            'transition_duration': 2.0,
            'target': 'xxzz_pair',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    },
]


def _load_round61_base_candidate():
    for candidate in ROUND61_CANDIDATES:
        if candidate['name'] == ROUND61_BASE_NAME:
            return _merge_round61_candidate(candidate)
    raise KeyError(ROUND61_BASE_NAME)


def _merge_round66_candidate(extra_candidate: dict):
    merged = copy.deepcopy(_load_round61_base_candidate())
    merged['name'] = extra_candidate['name']
    merged['description'] = extra_candidate['description']
    merged['rationale'] = extra_candidate['rationale']
    merged['scd'] = copy.deepcopy(extra_candidate['scd'])
    merged['feedback_channel'] = copy.deepcopy(NOOP_FEEDBACK_CHANNEL)
    merged['scd_channel'] = copy.deepcopy(extra_candidate['scd_channel'])
    merged['round66_extra_patch'] = copy.deepcopy(extra_candidate)
    return merged


def _compute_payload(source_mod, clbt, variant: str, method_file: str, extra: dict | None = None):
    param_errors, focus, lever, overall = _compute_metrics(source_mod, clbt)
    return {
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


def _delta_block(curr: dict, ref: dict):
    return {k: float(curr[k] - ref[k]) for k in curr}


def _score_candidate(delta_vs_r61: dict):
    penalties = []
    for key in HARD_PROTECTED_KEYS:
        value = float(delta_vs_r61[key])
        if value > 1e-9:
            penalties.append({'metric': key, 'delta': value})

    score = 0.0
    score += -1.25 * float(delta_vs_r61['mean_pct_error'])
    score += -1.00 * float(delta_vs_r61['max_pct_error'])
    score += -0.80 * float(delta_vs_r61['dKg_xx'])
    score += -0.55 * float(delta_vs_r61['dKg_zz'])
    score += -0.35 * float(delta_vs_r61['median_pct_error'])
    score += -0.20 * float(delta_vs_r61['dKg_xy'])
    score += -0.25 * float(delta_vs_r61['dKg_yy'])
    score += -0.20 * float(delta_vs_r61['dKa_xx'])
    score += -0.20 * float(delta_vs_r61['rx_y'])

    for p in penalties:
        score -= 1000.0 * float(p['delta'])

    return float(score), penalties


def _is_clean_winner(delta_vs_r61: dict, penalties: list[dict]):
    if penalties:
        return False
    return (
        float(delta_vs_r61['mean_pct_error']) < 0.0
        and float(delta_vs_r61['max_pct_error']) <= 0.0
        and float(delta_vs_r61['dKg_xx']) < 0.0
    )


def _selection_note(delta_vs_r61: dict, penalties: list[dict]):
    if _is_clean_winner(delta_vs_r61, penalties):
        return 'Clean same-dataset win over Round61 with frozen feedback and xx/zz-only adaptive SCD.'

    if penalties:
        return f'Protected regression detected vs Round61: {penalties}'

    improved = [k for k in TARGET_KEYS if float(delta_vs_r61[k]) < 0.0]
    if improved:
        return f'Partial narrow-SCD signal on {improved}, but clean-win gate (mean/max/dKg_xx + no protected regressions) not met.'
    return 'No useful same-dataset signal over Round61.'


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
    lines.append('Round66 是 **new mechanism probe**：冻结 Round61 feedback 主体，只在 SCD 的 xx/zz 子路径上施加 consistency-adaptive 控制。')
    lines.append('</callout>')
    lines.append('')
    lines.append('## 1. Fixed mainline dataset (same as Round65)')
    lines.append('')
    lines.append(f"- seed: `{summary['dataset']['noise_config']['seed']}`")
    lines.append(f"- arw: `{summary['dataset']['noise_config']['arw_dpsh']} dps/√h`")
    lines.append(f"- vrw: `{summary['dataset']['noise_config']['vrw_ugpsHz']} ug/√Hz`")
    lines.append(f"- bi_g: `{summary['dataset']['noise_config']['bi_g_dph']} dph`, bi_a: `{summary['dataset']['noise_config']['bi_a_ug']} ug`")
    lines.append('- source trajectory family: `round53_internalized_trustcov_release::_build_dataset`')
    lines.append('')
    lines.append('## 2. Round66 candidates vs Round61')
    lines.append('')
    lines.append('| candidate | dKg_xx Δ | dKg_xy Δ | dKg_yy Δ | dKg_zz Δ | dKa_xx Δ | rx_y Δ | ry_z Δ | mean Δ | median Δ | max Δ | score | note |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for name in summary['candidate_order']:
        cand = summary['candidates'][name]
        d = cand['delta_vs_round61']
        lines.append(
            f"| `{name}` | {d['dKg_xx']:.6f} | {d['dKg_xy']:.6f} | {d['dKg_yy']:.6f} | {d['dKg_zz']:.6f} | {d['dKa_xx']:.6f} | {d['rx_y']:.6f} | {d['ry_z']:.6f} | {d['mean_pct_error']:.6f} | {d['median_pct_error']:.6f} | {d['max_pct_error']:.6f} | {cand['selection']['score']:.6f} | {cand['selection']['note']} |"
        )
    lines.append('')
    lines.append('## 3. Decision')
    lines.append('')
    if summary['winner']:
        lines.append(f"- winner: `{summary['winner']['name']}`")
        lines.append('- decision: formalize as Round66 method')
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
    lines.append('# Round66 Record (new mechanism probe)')
    lines.append('')
    lines.append('## A. Round 基本信息')
    lines.append(f"- Round name: {summary['round_name']}")
    lines.append('- Round type: `new mechanism probe`')
    lines.append(f"- Base candidate: `{ROUND61_BASE_NAME}`")
    lines.append('- Dataset / regime: `D_ref_mainline` (same fixed noisy dataset as Round65)')
    lines.append('- D_ref_mainline definition:')
    lines.append('  - source trajectory: `method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset`')
    lines.append(f"  - arw = `{summary['dataset']['noise_config']['arw_dpsh']} * dpsh`")
    lines.append(f"  - vrw = `{summary['dataset']['noise_config']['vrw_ugpsHz']} * ugpsHz`")
    lines.append(f"  - bi_g = `{summary['dataset']['noise_config']['bi_g_dph']} * dph`")
    lines.append(f"  - bi_a = `{summary['dataset']['noise_config']['bi_a_ug']} * ug`")
    lines.append(f"  - tau_g = tau_a = `{summary['dataset']['noise_config']['tau_g']}`")
    lines.append(f"  - seed = `{summary['dataset']['noise_config']['seed']}`")
    lines.append('')
    lines.append('## B. 本轮目标')
    lines.append('- This round is a cleaner new-direction probe, not a Round65-B split-gate repair continuation.')
    lines.append('- Freeze Round61 feedback body as much as possible (feedback gate fixed no-op at 1.0).')
    lines.append('- Apply consistency-adaptive control ONLY to SCD xx/zz subpath (iter2 once-per-phase SCD target=`xxzz_pair`).')
    lines.append('')
    lines.append('## C. Allowed knobs')
    lines.append('- knob group 1: SCD consistency gate stats (EMA/slope/floor/warmup/power)')
    lines.append('- knob group 2: SCD alpha base around Round61 neighborhood')
    lines.append('- locked/no-change: Round61 feedback route, yy/Ka_xx/lever feedback protections')
    lines.append('')
    lines.append('## D. Protected metrics and clean-win gate')
    lines.append('- hard-protected metrics: dKg_xy / dKg_yy / dKa_xx / rx_y / ry_z')
    lines.append('- clean-win gate vs Round61: mean<0, max<=0, dKg_xx<0 and hard-protected no regression')
    lines.append('- formalize gate: only clean winner can be promoted')
    lines.append('')
    lines.append('## E. Candidate design (3-5 max)')
    for idx, candidate in enumerate(ROUND66_CANDIDATES, start=1):
        lines.append(f'### candidate {idx}')
        lines.append(f"- name: `{candidate['name']}`")
        lines.append(f"- rationale: {candidate['rationale']}")
        lines.append(f"- scd_channel: `{json.dumps(candidate['scd_channel'], ensure_ascii=False)}`")
        lines.append(f"- scd: `{json.dumps(candidate['scd'], ensure_ascii=False)}`")
        lines.append('')
    lines.append('## F. Result summary')
    lines.append(f"- winner: `{summary['winner']['name']}`" if summary['winner'] else '- winner: none')
    lines.append(f"- result class: `{summary['result_classification']}`")
    lines.append(f"- one-line conclusion: {summary['conclusion_line']}")
    lines.append(f"- strongest signal: {summary['strongest_signal']['signal']}")
    lines.append('')
    lines.append('## G. Mechanism learning and next move')
    lines.append(f"- mechanism learning: {summary['mechanism_learning']}")
    lines.append(f"- next best move: {summary['next_best_move']}")
    lines.append('')
    lines.append('## H. Artifacts')
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

    source_mod = load_module('markov_pruned_source_round66', str(SOURCE_FILE))
    dataset = _build_shared_dataset(source_mod)

    candidate_dump = {
        'round_name': 'Round66_SCD_XXZZ_ConsistencyOnly',
        'round_type': 'new mechanism probe',
        'mechanism': 'Freeze Round61 feedback path; apply innovation-consistency adaptation only to SCD xx/zz subpath',
        'base_round61_candidate': ROUND61_BASE_NAME,
        'same_dataset_round61_json': str(ROUND61_REF_JSON),
        'dataset': {
            'source_file': str(SOURCE_FILE),
            'source_trajectory_reference': 'method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset',
            'noise_config': dataset['noise_config'],
            'constraint_note': 'Use exactly the same fixed noisy dataset/noise strength/seed as Round65 mainline',
        },
        'protected_metrics': HARD_PROTECTED_KEYS,
        'clean_win_gate': 'mean<0, max<=0, dKg_xx<0 and no hard-protected regression vs Round61',
        'feedback_freeze': NOOP_FEEDBACK_CHANNEL,
        'round66_candidates': ROUND66_CANDIDATES,
    }
    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    out = {
        'round_name': 'Round66_SCD_XXZZ_ConsistencyOnly',
        'round_type': 'new mechanism probe',
        'mechanism': 'Round61 frozen feedback + consistency-adaptive SCD-only on xx/zz subpath',
        'dataset': {
            'source_file': str(SOURCE_FILE),
            'noise_config': dataset['noise_config'],
            'seed': dataset['noise_config']['seed'],
        },
        'base_round61_candidate': ROUND61_BASE_NAME,
        'base_round61_json': str(ROUND61_REF_JSON),
        'candidate_json': str(CANDIDATE_JSON),
        'candidate_order': [c['name'] for c in ROUND66_CANDIDATES],
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

    for idx, candidate in enumerate(ROUND66_CANDIDATES, start=1):
        merged_candidate = _merge_round66_candidate(candidate)

        method_mod = load_module(f'markov_method_round66_candidate_{idx}', str(R53_METHOD_FILE))
        method_mod = _build_patched_method(method_mod, merged_candidate)

        result = list(_run_internalized_hybrid_scd_dualgate(
            method_mod,
            source_mod,
            imu_noisy,
            pos0,
            ts,
            bi_g=bi_g,
            bi_a=bi_a,
            tau_g=tau_g,
            tau_a=tau_a,
            label=f'R66-SCDONLY-{idx}',
            scd_cfg=merged_candidate['scd'],
            feedback_cfg=merged_candidate['feedback_channel'],
            scd_gate_cfg=merged_candidate['scd_channel'],
        ))
        clbt_candidate = result[0]
        runtime_log = {
            'schedule_log': result[4].get('schedule_log') if len(result) >= 5 else None,
            'feedback_log': result[4].get('feedback_log') if len(result) >= 5 else None,
            'scd_log': result[4].get('scd_log') if len(result) >= 5 else None,
            'dual_gate_log': result[4].get('dual_gate_log') if len(result) >= 5 else None,
        }
        del result
        gc.collect()

        payload_candidate = _compute_payload(
            source_mod,
            clbt_candidate,
            variant=f"r66_scdonly_{candidate['name']}",
            method_file='probe_round66_scd_xxzz_only::scd_only_consistency_adaptive',
            extra={
                'dataset_noise_config': dataset['noise_config'],
                'base_round61_candidate': ROUND61_BASE_NAME,
                'feedback_channel_frozen': copy.deepcopy(NOOP_FEEDBACK_CHANNEL),
                'scd_channel': copy.deepcopy(candidate['scd_channel']),
                'scd_cfg': copy.deepcopy(candidate['scd']),
                'runtime_log': runtime_log,
            },
        )
        del clbt_candidate
        gc.collect()

        candidate_json_path = RESULTS_DIR / f"R66_scdonly_{candidate['name']}_param_errors.json"
        candidate_json_path.write_text(json.dumps(payload_candidate, ensure_ascii=False, indent=2), encoding='utf-8')

        delta_vs_r61 = {
            **_delta_block(payload_candidate['focus_scale_pct'], round61_payload['focus_scale_pct']),
            **_delta_block(payload_candidate['lever_guard_pct'], round61_payload['lever_guard_pct']),
            **_delta_block(payload_candidate['overall'], round61_payload['overall']),
        }

        score, penalties = _score_candidate(delta_vs_r61)
        note = _selection_note(delta_vs_r61, penalties)

        out['candidates'][candidate['name']] = {
            'description': candidate['description'],
            'rationale': candidate['rationale'],
            'feedback_channel': copy.deepcopy(NOOP_FEEDBACK_CHANNEL),
            'scd_channel': copy.deepcopy(candidate['scd_channel']),
            'scd_cfg': copy.deepcopy(candidate['scd']),
            'param_errors_json': str(candidate_json_path),
            'focus_scale_pct': payload_candidate['focus_scale_pct'],
            'lever_guard_pct': payload_candidate['lever_guard_pct'],
            'overall': payload_candidate['overall'],
            'delta_vs_round61': delta_vs_r61,
            'selection': {
                'score': float(score),
                'penalties': penalties,
                'note': note,
            },
            'runtime_log': payload_candidate['extra']['runtime_log'],
            'vs_round61_relative_improvement': _relative_improvement_block(
                round61_payload,
                payload_candidate,
                ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'mean_pct_error', 'median_pct_error', 'max_pct_error'],
            ),
        }

        print(candidate['name'], json.dumps({
            'delta_vs_round61': delta_vs_r61,
            'score': score,
            'penalties': penalties,
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
    best_penalties = best['selection']['penalties']

    if _is_clean_winner(best_delta, best_penalties):
        out['winner'] = {
            'name': best_name,
            'score': float(best_score),
            'reason': 'Clean same-dataset winner over Round61 under frozen-feedback + xx/zz-only adaptive SCD mechanism.',
        }
        out['result_classification'] = 'clean win'
        out['conclusion_line'] = 'Round66 produced a clean same-dataset winner over Round61.'

        formal_method_file = METHOD_DIR / f"method_42state_gm1_round66_scd_xxzz_only_{best_name}.py"
        formal_result_json = RESULTS_DIR / f"R66_42state_gm1_round66_scd_xxzz_only_{best_name}_param_errors.json"
        formal_method_file.write_text(
            (
                '# Auto-generated Round66 formalization placeholder.\n'
                '# Winner configuration is recorded in results/round66_probe_summary.json and round66_candidates.json.\n'
                '# For reproducibility, use psins_method_bench/scripts/probe_round66_scd_xxzz_only.py with the winner candidate.\n'
            ),
            encoding='utf-8',
        )
        formal_result_json.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding='utf-8')
        out['formal_method_file'] = str(formal_method_file)
        out['formal_result_json'] = str(formal_result_json)
    else:
        out['winner'] = None
        out['no_winner_reason'] = 'No candidate passed the same-dataset Round61 clean-win gate under frozen-feedback + xx/zz-only adaptive SCD constraints.'
        improved_targets = [k for k in TARGET_KEYS if float(best_delta[k]) < 0.0]
        if best_penalties:
            out['result_classification'] = 'no useful signal'
        else:
            out['result_classification'] = 'partial signal' if improved_targets else 'no useful signal'
        out['conclusion_line'] = 'Round66 did not produce a clean promotable winner over Round61 on the fixed mainline dataset.'

    out['strongest_signal'] = {
        'name': best_name,
        'signal': (
            f"best candidate {best_name}: "
            f"dKg_xx Δ={best_delta['dKg_xx']:.6f}, dKg_zz Δ={best_delta['dKg_zz']:.6f}, "
            f"mean Δ={best_delta['mean_pct_error']:.6f}, max Δ={best_delta['max_pct_error']:.6f}"
        ),
        'regressions': str(best_penalties),
    }

    out['mechanism_learning'] = (
        'Freezing Round61 feedback while adapting only xx/zz SCD keeps causality interpretable, '
        'but current settings did not yet deliver a clean no-regression same-dataset win over Round61.'
    )
    out['next_best_move'] = (
        'Keep feedback fully frozen; run an even tighter one-knob SCD-only alpha ladder around the best Round66 candidate '
        '(single-floor/single-power perturbation) and verify if dKg_xx/mean can improve without yy/Ka_xx/rx_y leakage.'
    )

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    REPORT_MD.write_text(_render_report(out), encoding='utf-8')
    ROUND66_RECORD_MD.write_text(_render_round_record(out), encoding='utf-8')

    print(f'Wrote {CANDIDATE_JSON}')
    print(f'Wrote {OUTPUT_JSON}')
    print(f'Wrote {REPORT_MD}')
    print(f'Wrote {ROUND66_RECORD_MD}')
    print('__RESULT_JSON__=' + json.dumps({
        'candidate_json': str(CANDIDATE_JSON),
        'summary_json': str(OUTPUT_JSON),
        'report_md': str(REPORT_MD),
        'round_record_md': str(ROUND66_RECORD_MD),
        'winner': out['winner'],
        'result_classification': out['result_classification'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
