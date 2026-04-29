from __future__ import annotations

import copy
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
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'
COMPUTE_R61_FILE = SCRIPTS_DIR / 'compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors.py'
ROUND62_SUMMARY_JSON = RESULTS_DIR / 'round62_alpha_guard_probe_summary.json'
ROUND63A_SUMMARY_JSON = RESULTS_DIR / 'round63a_scd_gating_probe_summary.json'
OUTPUT_JSON = RESULTS_DIR / 'round63b_combo_probe_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'round63b_combo_candidates.json'
REPORT_MD = REPORTS_DIR / 'psins_round63b_combo_probe_2026-03-28.md'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import load_module
from probe_round62_alpha_guard import (  # noqa: E402
    ROUND62_CANDIDATES,
    _build_round62_method,
    _compute_payload,
    _merge_round62_candidate,
    build_shared_dataset,
    make_suffix,
)
from probe_round63a_scd_gating import (  # noqa: E402
    _resolve_round63a_scd,
    _run_internalized_hybrid_scd_round63a,
)

BASE_R62_WINNER_NAME = 'r62_ultralow_guard_soft_neutral'
BASE_R63A_WINNER_NAME = 'r63a_ul_scd_floor_off'
PROBE_SCALES = [1.0, 0.10, 0.08, 0.05, 0.03]

ROUND63B_CANDIDATES = [
    {
        'name': 'r63b_combo_floor_off_r62soft',
        'description': 'Exact Round62 winner plus the Round63-A floor-off rule: below 0.08x disable the iter2 once-per-phase SCD entirely.',
        'rationale': 'Direct combination baseline. Tests whether Round62’s ultra-low alpha-guard is already enough to absorb the SCD-off tradeoff.',
        'round62_guard': {
            'apply_policy_names': ['iter2_commit'],
            'start_noise_scale': 0.08,
            'full_noise_scale': 0.03,
            'power': 2.0,
            'state_alpha_pull': {16: 1.0, 21: 1.0},
            'rx_y_post_pull': 1.0,
        },
        'round63a_scd_guard': {
            'active_below_noise_scale': 0.08,
            'scd_override': {
                'apply_policy_names': [],
                'target_name': 'disabled',
            },
        },
    },
    {
        'name': 'r63b_combo_floor_off_003only',
        'description': 'Keep the Round62 winner intact through 0.05x, and only at 0.03x disable the iter2 once-per-phase SCD.',
        'rationale': 'Weaker floor-off variant. If 0.05x is still in the sweet spot, this localizes the SCD ablation to the deepest ultra-low regime only.',
        'round62_guard': {
            'apply_policy_names': ['iter2_commit'],
            'start_noise_scale': 0.08,
            'full_noise_scale': 0.03,
            'power': 2.0,
            'state_alpha_pull': {16: 1.0, 21: 1.0},
            'rx_y_post_pull': 1.0,
        },
        'round63a_scd_guard': {
            'active_below_noise_scale': 0.05,
            'scd_override': {
                'apply_policy_names': [],
                'target_name': 'disabled',
            },
        },
    },
    {
        'name': 'r63b_combo_alpha09995_r62soft',
        'description': 'Round62 winner plus a milder ultra-low SCD alpha: below 0.08x change once-per-phase SCD alpha from 0.999 to 0.9995.',
        'rationale': 'Alpha-weaken variant from Round63-A. Keeps the SCD body alive, but softens the cut enough that Round62 guard may handle the remaining ultra-low overshoot.',
        'round62_guard': {
            'apply_policy_names': ['iter2_commit'],
            'start_noise_scale': 0.08,
            'full_noise_scale': 0.03,
            'power': 2.0,
            'state_alpha_pull': {16: 1.0, 21: 1.0},
            'rx_y_post_pull': 1.0,
        },
        'round63a_scd_guard': {
            'active_below_noise_scale': 0.08,
            'scd_override': {
                'alpha': 0.9995,
            },
        },
    },
    {
        'name': 'r63b_combo_core3_r62soft',
        'description': 'Round62 winner plus a smaller ultra-low SCD target: below 0.08x keep SCD only on the Round61 backbone trio dKg_xx / dKg_xy / dKg_zz.',
        'rationale': 'Target-weaken variant from Round63-A. Keeps the SCD idea only on the three states that still looked structurally helpful, while Round62 guard handles yy / Ka_xx-side overshoot.',
        'round62_guard': {
            'apply_policy_names': ['iter2_commit'],
            'start_noise_scale': 0.08,
            'full_noise_scale': 0.03,
            'power': 2.0,
            'state_alpha_pull': {16: 1.0, 21: 1.0},
            'rx_y_post_pull': 1.0,
        },
        'round63a_scd_guard': {
            'active_below_noise_scale': 0.08,
            'scd_override': {
                'target': 'custom',
                'target_name': 'selected_core3',
                'target_indices': [12, 15, 20],
            },
        },
    },
]


def _baseline_payload_path(noise_scale: float) -> Path:
    return RESULTS_DIR / f'M_markov_42state_gm1_shared_{make_suffix(noise_scale)}_param_errors.json'


def _round61_payload_path(noise_scale: float) -> Path:
    return RESULTS_DIR / f'R61_42state_gm1_round61_h_scd_state20_microtight_commit_shared_{make_suffix(noise_scale)}_param_errors.json'


def _compare_payload_path(noise_scale: float) -> Path:
    return RESULTS_DIR / f'compare_baseline_vs_round61_shared_{make_suffix(noise_scale)}.json'


def _param_specs(source_mod):
    compute_r61_mod = load_module('compute_r61_for_r63b_combo_specs', str(COMPUTE_R61_FILE))
    return compute_r61_mod._param_specs(source_mod)


def _build_scale_delta(cand_focus, cand_lever, cand_overall, ref_focus, ref_lever, ref_overall):
    return {
        'dKg_xx': float(cand_focus['dKg_xx'] - ref_focus['dKg_xx']),
        'dKg_xy': float(cand_focus['dKg_xy'] - ref_focus['dKg_xy']),
        'dKg_yy': float(cand_focus['dKg_yy'] - ref_focus['dKg_yy']),
        'dKg_zz': float(cand_focus['dKg_zz'] - ref_focus['dKg_zz']),
        'dKa_xx': float(cand_focus['dKa_xx'] - ref_focus['dKa_xx']),
        'rx_y': float(cand_lever['rx_y'] - ref_lever['rx_y']),
        'ry_z': float(cand_lever['ry_z'] - ref_lever['ry_z']),
        'mean_pct_error': float(cand_overall['mean_pct_error'] - ref_overall['mean_pct_error']),
        'median_pct_error': float(cand_overall['median_pct_error'] - ref_overall['median_pct_error']),
        'max_pct_error': float(cand_overall['max_pct_error'] - ref_overall['max_pct_error']),
    }


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


def _sorted_policy_patch(iter_patches: dict):
    out = {}
    for iter_idx, patch in sorted(iter_patches.items()):
        out[str(iter_idx + 1)] = {
            key: {str(k): float(v) for k, v in value.items()} if isinstance(value, dict) else value
            for key, value in patch.items()
        }
    return out


def _merge_round63b_candidate(extra_candidate: dict):
    base_round62 = _merge_round62_candidate(next(c for c in ROUND62_CANDIDATES if c['name'] == BASE_R62_WINNER_NAME))
    base_round62['name'] = extra_candidate['name']
    base_round62['description'] = extra_candidate['description']
    base_round62['rationale'] = extra_candidate['rationale']
    base_round62['round62_guard'] = copy.deepcopy(extra_candidate['round62_guard'])
    base_round62['round63a_scd_guard'] = copy.deepcopy(extra_candidate['round63a_scd_guard'])
    base_round62['round63b_extra_patch'] = copy.deepcopy(extra_candidate)
    return base_round62


def _run_candidate_at_scale(candidate: dict, noise_scale: float, idx: int):
    suffix = make_suffix(noise_scale)
    source_mod = load_module(f'markov_pruned_source_r63b_combo_{candidate["name"]}_{suffix}', str(SOURCE_FILE))
    params = _param_specs(source_mod)
    dataset = build_shared_dataset(source_mod, noise_scale)
    merged_candidate = _merge_round63b_candidate(candidate)
    resolved_scd, scd_meta = _resolve_round63a_scd(merged_candidate, noise_scale)
    base_method_mod = load_module(f'markov_method_r63b_combo_base_{candidate["name"]}_{suffix}', str(R53_METHOD_FILE))
    method_mod = _build_round62_method(base_method_mod, merged_candidate, noise_scale)

    result = list(_run_internalized_hybrid_scd_round63a(
        method_mod,
        source_mod,
        dataset['imu_noisy'],
        dataset['pos0'],
        dataset['ts'],
        bi_g=dataset['bi_g'],
        bi_a=dataset['bi_a'],
        tau_g=dataset['tau_g'],
        tau_a=dataset['tau_a'],
        label=f'42-GM1-R63B-{idx}-{suffix.upper()}',
        scd_cfg=resolved_scd,
    ))
    extra = result[4] if len(result) >= 5 and isinstance(result[4], dict) else {}
    extra = dict(extra)
    extra.update({
        'noise_scale': float(noise_scale),
        'noise_config': dataset['noise_config'],
        'comparison_mode': 'shared_dataset_apples_to_apples',
        'round63b_selected_candidate': merged_candidate['name'],
        'round63b_base_round62_winner': BASE_R62_WINNER_NAME,
        'round63b_base_round63a_winner': BASE_R63A_WINNER_NAME,
        'round63b_round62_guard': copy.deepcopy(merged_candidate.get('round62_guard', {})),
        'round63b_round63a_scd_guard': copy.deepcopy(merged_candidate.get('round63a_scd_guard', {})),
        'round63b_resolved_scd': copy.deepcopy(resolved_scd),
        'round63b_scd_guard_meta': copy.deepcopy(scd_meta),
        'round63b_note': 'Round63-B keeps the Round62 ultra-low alpha-guard and probes only a tiny ultra-low SCD-side combination branch.',
    })
    payload = _compute_payload(
        source_mod,
        result[0],
        params,
        variant=f'42state_gm1_round63b_combo_{merged_candidate["name"]}_shared_{suffix}',
        method_file=f'probe_round63b_combo::{merged_candidate["name"]}',
        extra=extra,
    )
    return merged_candidate, payload


def _score_candidate(candidate_scales: dict):
    penalties = []
    protect_scale_keys = ['noise1x', 'noise0p1', 'noise0p08']
    protect_metrics = ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'mean_pct_error', 'median_pct_error', 'max_pct_error']
    score_vs_round62 = 0.0
    for scale_key in protect_scale_keys:
        cand = candidate_scales[scale_key]['delta_vs_round62_winner']
        for metric_name in protect_metrics:
            value = float(cand[metric_name])
            if value > 1e-9:
                penalties.append({'scale': scale_key, 'metric': metric_name, 'delta': value})
                score_vs_round62 -= 1000.0 * value

    weights = {
        'noise0p05': {'dKg_yy': 1.5, 'dKa_xx': 1.3, 'rx_y': 1.2, 'mean_pct_error': 0.8, 'median_pct_error': 0.7},
        'noise0p03': {'dKg_yy': 2.3, 'dKa_xx': 1.8, 'rx_y': 1.5, 'mean_pct_error': 1.0, 'median_pct_error': 0.9},
    }
    score_vs_round63a = 0.0
    for scale_key, scale_weights in weights.items():
        d62 = candidate_scales[scale_key]['delta_vs_round62_winner']
        d63 = candidate_scales[scale_key]['delta_vs_round63a_winner']
        for metric_name, weight in scale_weights.items():
            score_vs_round62 += weight * (-float(d62[metric_name]))
            score_vs_round63a += weight * (-float(d63[metric_name]))
    return float(score_vs_round62), float(score_vs_round63a), penalties


def _selection_note(candidate_scales: dict, score_vs_round62: float, score_vs_round63a: float, penalties: list[dict]):
    if penalties:
        return f'Protected-scale regression detected: {penalties}'

    d05_r62 = candidate_scales['noise0p05']['delta_vs_round62_winner']
    d03_r62 = candidate_scales['noise0p03']['delta_vs_round62_winner']
    d05_r63 = candidate_scales['noise0p05']['delta_vs_round63a_winner']
    d03_r63 = candidate_scales['noise0p03']['delta_vs_round63a_winner']

    if score_vs_round62 > 0.0 and score_vs_round63a > 0.0:
        return 'Clean combo win: weighted ultra-low target set improves versus both Round62 and Round63-A.'
    if score_vs_round62 > 0.0 and (d03_r62['dKa_xx'] < 0 or d05_r62['dKa_xx'] < 0):
        return 'Beats Round62 on part of the ultra-low target set, but does not yet cleanly dominate Round63-A.'
    if d03_r62['dKa_xx'] < 0 or d05_r62['dKa_xx'] < 0:
        return 'Partial combo signal: Ka_xx improves against Round62, but yy / rx_y / overall do not combine cleanly.'
    return 'No useful combination signal: alpha-guard + this SCD rule does not beat the current Round62 / Round63-A tradeoff.'


def _render_report(summary: dict):
    lines = []
    lines.append('<callout emoji="🧪" background-color="light-blue">')
    lines.append('Round63-B：在 **Round62 ultra-low alpha-guard winner** 上，叠加一个极小的 **Round63-A ultra-low SCD** 组合分叉。目标不是重跑纯 SCD，而是检查两条 ultra-low 修复线能否真正叠加。')
    lines.append('</callout>')
    lines.append('')
    lines.append('## 1. Probe 设置')
    lines.append('')
    lines.append(f'- Round62 winner base: `{summary["round62_reference"]["winner_name"]}`')
    lines.append(f'- Round63-A branch reference: `{summary["round63a_reference"]["winner_name"]}`')
    lines.append('- Shared-noise probe scales: `1.00x, 0.10x, 0.08x, 0.05x, 0.03x`')
    lines.append('- Batch size: **4 deterministic combo candidates**')
    lines.append('')
    lines.append('## 2. 候选摘要')
    lines.append('')
    lines.append('| candidate | 0.05x ΔvsR62 (yy/Ka_xx/rx_y/mean/median) | 0.03x ΔvsR62 (yy/Ka_xx/rx_y/mean/median) | score vs R62 | score vs R63A | note |')
    lines.append('|---|---|---|---:|---:|---|')
    for name in summary['candidate_order']:
        cand = summary['candidates'][name]
        s05 = cand['scales']['noise0p05']['delta_vs_round62_winner']
        s03 = cand['scales']['noise0p03']['delta_vs_round62_winner']
        lines.append(
            f"| `{name}` | {s05['dKg_yy']:.4f} / {s05['dKa_xx']:.4f} / {s05['rx_y']:.4f} / {s05['mean_pct_error']:.4f} / {s05['median_pct_error']:.4f} | {s03['dKg_yy']:.4f} / {s03['dKa_xx']:.4f} / {s03['rx_y']:.4f} / {s03['mean_pct_error']:.4f} / {s03['median_pct_error']:.4f} | {cand['selection']['score_vs_round62']:.4f} | {cand['selection']['score_vs_round63a']:.4f} | {cand['selection']['note']} |"
        )
    lines.append('')
    lines.append('## 3. Winner / status')
    lines.append('')
    if summary['winner']:
        lines.append(f'- Clear winner: `{summary["winner"]["name"]}`')
        lines.append(f'- Reason: {summary["winner"]["reason"]}')
    else:
        best = summary.get('best_candidate')
        if best:
            lines.append(f'- Best candidate so far: `{best["name"]}`')
            lines.append(f'- Best-candidate note: {best["reason"]}')
        lines.append('- Clear winner: **none yet**')
        lines.append(f'- Reason: {summary["no_winner_reason"]}')
    lines.append('')
    lines.append('## 4. Candidate notes')
    lines.append('')
    for name in summary['candidate_order']:
        cand = summary['candidates'][name]
        lines.append(f'### `{name}`')
        lines.append(f'- Description: {cand["description"]}')
        lines.append(f'- Rationale: {cand["rationale"]}')
        lines.append(f'- Round62 guard: `{json.dumps(cand["round62_guard"], ensure_ascii=False)}`')
        lines.append(f'- Ultra-low SCD guard: `{json.dumps(cand["round63a_scd_guard"], ensure_ascii=False)}`')
        lines.append(f'- Selection note: {cand["selection"]["note"]}')
        lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    round62_summary = json.loads(ROUND62_SUMMARY_JSON.read_text(encoding='utf-8'))
    round63a_summary = json.loads(ROUND63A_SUMMARY_JSON.read_text(encoding='utf-8'))
    round62_winner = round62_summary.get('winner')
    round63a_branch_winner = round63a_summary.get('branch_winner') or round63a_summary.get('winner')
    if not round62_winner or not round63a_branch_winner:
        raise RuntimeError('Missing Round62 or Round63-A winner reference')

    round62_winner_name = round62_winner['name']
    round63a_winner_name = round63a_branch_winner['name']
    round62_ref_scales = round62_summary['candidates'][round62_winner_name]['scales']
    round63a_ref_scales = round63a_summary['candidates'][round63a_winner_name]['scales']

    candidate_dump = {
        'round62_reference_winner': round62_winner_name,
        'round63a_reference_winner': round63a_winner_name,
        'probe_scales': PROBE_SCALES,
        'round63b_candidates': ROUND63B_CANDIDATES,
    }
    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    baseline_payloads = {}
    round61_payloads = {}
    round61_compare = {}
    for scale in PROBE_SCALES:
        suffix = make_suffix(scale)
        baseline_payloads[suffix] = json.loads(_baseline_payload_path(scale).read_text(encoding='utf-8'))
        round61_payloads[suffix] = json.loads(_round61_payload_path(scale).read_text(encoding='utf-8'))
        round61_compare[suffix] = json.loads(_compare_payload_path(scale).read_text(encoding='utf-8'))

    out = {
        'probe_scales': PROBE_SCALES,
        'candidate_order': [c['name'] for c in ROUND63B_CANDIDATES],
        'baseline_paths': {make_suffix(s): str(_baseline_payload_path(s)) for s in PROBE_SCALES},
        'round61_paths': {make_suffix(s): str(_round61_payload_path(s)) for s in PROBE_SCALES},
        'round62_reference': {
            'summary_json': str(ROUND62_SUMMARY_JSON),
            'winner_name': round62_winner_name,
            'winner_reason': round62_winner.get('reason'),
        },
        'round63a_reference': {
            'summary_json': str(ROUND63A_SUMMARY_JSON),
            'winner_name': round63a_winner_name,
            'winner_reason': round63a_branch_winner.get('reason'),
        },
        'candidate_json': str(CANDIDATE_JSON),
        'candidates': {},
        'winner': None,
        'best_candidate': None,
        'no_winner_reason': None,
    }

    for idx, candidate in enumerate(ROUND63B_CANDIDATES, start=1):
        merged_candidate = _merge_round63b_candidate(candidate)
        candidate_scales = {}
        for scale in PROBE_SCALES:
            suffix = make_suffix(scale)
            merged_candidate, payload = _run_candidate_at_scale(candidate, scale, idx)
            baseline_payload = baseline_payloads[suffix]
            round61_payload = round61_payloads[suffix]
            compare_payload = round61_compare[suffix]
            round62_payload = round62_ref_scales[suffix]
            round63a_payload = round63a_ref_scales[suffix]

            focus = payload['focus_scale_pct']
            lever = payload['lever_guard_pct']
            overall = payload['overall']
            delta_vs_round61 = _build_scale_delta(
                focus,
                lever,
                overall,
                round61_payload['focus_scale_pct'],
                round61_payload['lever_guard_pct'],
                round61_payload['overall'],
            )
            delta_vs_round62 = _build_scale_delta(
                focus,
                lever,
                overall,
                round62_payload['focus'],
                round62_payload['lever'],
                round62_payload['overall'],
            )
            delta_vs_round63a = _build_scale_delta(
                focus,
                lever,
                overall,
                round63a_payload['focus'],
                round63a_payload['lever'],
                round63a_payload['overall'],
            )
            candidate_vs_baseline = _relative_improvement_block(
                baseline_payload,
                payload,
                ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'mean_pct_error', 'median_pct_error', 'max_pct_error'],
            )
            candidate_scales[suffix] = {
                'noise_scale': float(scale),
                'focus': focus,
                'lever': lever,
                'overall': overall,
                'delta_vs_round61': delta_vs_round61,
                'delta_vs_round62_winner': delta_vs_round62,
                'delta_vs_round63a_winner': delta_vs_round63a,
                'candidate_vs_baseline': candidate_vs_baseline,
                'round61_vs_baseline_reference': {
                    'dKg_xx': compare_payload['all_params']['dKg_xx'],
                    'dKg_xy': compare_payload['all_params']['dKg_xy'],
                    'dKg_yy': compare_payload['all_params']['dKg_yy'],
                    'dKg_zz': compare_payload['all_params']['dKg_zz'],
                    'dKa_xx': compare_payload['all_params']['dKa_xx'],
                    'rx_y': compare_payload['all_params']['rx_y'],
                    'ry_z': compare_payload['all_params']['ry_z'],
                    'mean_pct_error': compare_payload['overall']['mean_pct_error'],
                    'median_pct_error': compare_payload['overall']['median_pct_error'],
                    'max_pct_error': compare_payload['overall']['max_pct_error'],
                },
                'round62_winner_reference': {
                    'focus': round62_payload['focus'],
                    'lever': round62_payload['lever'],
                    'overall': round62_payload['overall'],
                },
                'round63a_winner_reference': {
                    'focus': round63a_payload['focus'],
                    'lever': round63a_payload['lever'],
                    'overall': round63a_payload['overall'],
                },
                'extra': {
                    'resolved_scd': payload['extra'].get('round63b_resolved_scd'),
                    'scd_guard_meta': payload['extra'].get('round63b_scd_guard_meta'),
                    'feedback_log': payload['extra'].get('feedback_log'),
                    'scd_log': payload['extra'].get('scd_log'),
                    'schedule_log': payload['extra'].get('schedule_log'),
                },
            }

        score_vs_round62, score_vs_round63a, penalties = _score_candidate(candidate_scales)
        note = _selection_note(candidate_scales, score_vs_round62, score_vs_round63a, penalties)
        out['candidates'][candidate['name']] = {
            'description': candidate['description'],
            'rationale': candidate['rationale'],
            'base_round62_winner': round62_winner_name,
            'base_round63a_winner': round63a_winner_name,
            'policy_patch': _sorted_policy_patch(merged_candidate.get('iter_patches', {})),
            'round62_guard': copy.deepcopy(candidate.get('round62_guard', {})),
            'round63a_scd_guard': copy.deepcopy(candidate.get('round63a_scd_guard', {})),
            'round63b_extra_patch': copy.deepcopy(candidate),
            'scd': copy.deepcopy(merged_candidate['scd']),
            'post_rx_y_mult': float(merged_candidate.get('post_rx_y_mult', 1.0)),
            'post_ry_z_mult': float(merged_candidate.get('post_ry_z_mult', 1.0)),
            'scales': candidate_scales,
            'selection': {
                'score_vs_round62': float(score_vs_round62),
                'score_vs_round63a': float(score_vs_round63a),
                'penalties': penalties,
                'note': note,
            },
        }

        print(candidate['name'], json.dumps({
            'score_vs_round62': score_vs_round62,
            'score_vs_round63a': score_vs_round63a,
            'note': note,
            'noise0p05_delta_vs_round62': candidate_scales['noise0p05']['delta_vs_round62_winner'],
            'noise0p03_delta_vs_round62': candidate_scales['noise0p03']['delta_vs_round62_winner'],
        }, ensure_ascii=False))

    ordered = sorted(
        [
            (
                name,
                out['candidates'][name]['selection']['score_vs_round62'],
                out['candidates'][name]['selection']['score_vs_round63a'],
            )
            for name in out['candidate_order']
        ],
        key=lambda x: (x[1], x[2]),
        reverse=True,
    )
    best_name, best_score_vs_round62, best_score_vs_round63a = ordered[0]
    best_candidate = out['candidates'][best_name]
    out['best_candidate'] = {
        'name': best_name,
        'score_vs_round62': float(best_score_vs_round62),
        'score_vs_round63a': float(best_score_vs_round63a),
        'reason': best_candidate['selection']['note'],
    }

    if (
        (not best_candidate['selection']['penalties'])
        and best_score_vs_round62 > 0.02
        and best_score_vs_round63a > 0.02
    ):
        out['winner'] = {
            'name': best_name,
            'score_vs_round62': float(best_score_vs_round62),
            'score_vs_round63a': float(best_score_vs_round63a),
            'reason': 'Best protected no-regression combo score, and the weighted ultra-low target set improves versus both Round62 and Round63-A.',
        }
    else:
        out['winner'] = None
        out['no_winner_reason'] = 'No candidate achieved a clean protected-scale combo win over both Round62 and Round63-A; keep Round62 as mainline and treat Round63-B as an exploratory branch only.'

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    REPORT_MD.write_text(_render_report(out), encoding='utf-8')
    print(f'Wrote {OUTPUT_JSON}')
    print(f'Wrote {REPORT_MD}')
    print('__RESULT_JSON__=' + json.dumps({
        'output_json': str(OUTPUT_JSON),
        'candidate_json': str(CANDIDATE_JSON),
        'report_md': str(REPORT_MD),
        'winner': out['winner'],
        'best_candidate': out['best_candidate'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
