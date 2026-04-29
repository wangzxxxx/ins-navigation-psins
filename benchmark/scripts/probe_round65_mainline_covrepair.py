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
BASELINE_JSON = RESULTS_DIR / 'A_noisy_baseline_param_errors.json'
R61_JSON = RESULTS_DIR / 'R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors.json'
R64_JSON = RESULTS_DIR / 'round64_mainline_trustcov_probe_summary.json'
OUTPUT_JSON = RESULTS_DIR / 'round65_mainline_covrepair_probe_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'round65_mainline_covrepair_candidates.json'
REPORT_MD = REPORTS_DIR / 'psins_round65_mainline_covrepair_probe_2026-03-28.md'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import load_module
from probe_round55_newline import _build_patched_method
from probe_round56_narrow import _compute_metrics
from probe_round59_h_scd_hybrid import _run_internalized_hybrid_scd
from probe_round61_hybrid_micro import ROUND61_CANDIDATES, _merge_round61_candidate

BASE_R61_CANDIDATE_NAME = 'r61_s20_08988_ryz00116'
BASE_R64_CANDIDATE_NAME = 'r64_cov_sched_static_meas_plus'

ROUND64_BASE_PATCH = {
    'name': BASE_R64_CANDIDATE_NAME,
    'description': 'Iter2 covariance/measurement schedule rebalance: slightly more selected-state process mobility and slightly stronger static/late measurement weight.',
    'rationale': 'Orthogonal mainline test: leave trust-map structure mostly intact and probe a tiny Q/R schedule rebalance to improve calibration on 1x.',
    'iter_patches': {
        1: {
            'selected_q_static_scale': 0.795,
            'selected_q_dynamic_scale': 1.010,
            'selected_q_late_mult': 1.300,
            'static_r_scale': 1.020,
            'dynamic_r_scale': 0.998,
            'late_r_mult': 0.992,
            'late_release_frac': 0.575,
        },
    },
}

ROUND65_CANDIDATES = [
    {
        'name': 'r65_cov_sched_relax_late',
        'description': 'Keep the Round64 schedule direction, but soften the late-stage mobility/measurement push just enough to see if yy and lever settle back.',
        'rationale': 'Pure covariance-schedule repair: back off the Round64 late push before adding any extra per-state or lever surgery.',
        'iter_patches': {
            1: {
                'selected_q_dynamic_scale': 1.008,
                'selected_q_late_mult': 1.296,
                'static_r_scale': 1.019,
                'dynamic_r_scale': 0.999,
                'late_r_mult': 0.993,
                'late_release_frac': 0.576,
            },
        },
    },
    {
        'name': 'r65_cov_sched_rx_postguard',
        'description': 'Hold the exact Round64 schedule patch and only retune the post rx_y confirmation from 1.00047 to 1.00052.',
        'rationale': 'Direct lever repair test: Round64 looked under-pulled on rx_y by about one micro-step, so first check whether the lever regression is mostly a post-confirmation calibration issue.',
        'post_rx_y_mult': 1.00052,
    },
    {
        'name': 'r65_cov_sched_state16_microguard',
        'description': 'Hold the exact Round64 schedule patch and add only a tiny iter2 state16 alpha protection bump.',
        'rationale': 'Direct yy repair test: if the Round64 regression is concentrated on dKg_yy, a minimal state16-only protection should be the most interpretable surgical counter-move.',
        'iter_patches': {
            1: {
                'state_alpha_mult': {16: 1.009},
            },
        },
    },
    {
        'name': 'r65_cov_sched_relax_late_rxguard',
        'description': 'Combine the softened late schedule with the tiny rx_y confirmation repair.',
        'rationale': 'Two-knob repair branch: let the schedule back off slightly while also restoring the lever side with the smallest deterministic post guard.',
        'iter_patches': {
            1: {
                'selected_q_dynamic_scale': 1.008,
                'selected_q_late_mult': 1.296,
                'static_r_scale': 1.019,
                'dynamic_r_scale': 0.999,
                'late_r_mult': 0.993,
                'late_release_frac': 0.576,
            },
        },
        'post_rx_y_mult': 1.00052,
    },
    {
        'name': 'r65_cov_sched_full_repair_narrow',
        'description': 'Narrow full repair: slightly less selected-state mobility than Round64, a slightly softer late measurement push, plus the micro yy/rx guard pair.',
        'rationale': 'Most complete narrow repair attempt around the Round64 direction while still avoiding any broad trust-map reshaping or SCD cadence changes.',
        'iter_patches': {
            1: {
                'selected_q_static_scale': 0.793,
                'selected_q_dynamic_scale': 1.006,
                'selected_q_late_mult': 1.294,
                'static_r_scale': 1.018,
                'dynamic_r_scale': 0.999,
                'late_r_mult': 0.994,
                'late_release_frac': 0.577,
                'state_alpha_mult': {16: 1.009},
            },
        },
        'post_rx_y_mult': 1.00052,
    },
]


def _load_r61_base_candidate():
    for candidate in ROUND61_CANDIDATES:
        if candidate['name'] == BASE_R61_CANDIDATE_NAME:
            return _merge_round61_candidate(candidate)
    raise KeyError(BASE_R61_CANDIDATE_NAME)


def _apply_candidate_patch(base: dict, patch_candidate: dict):
    merged = copy.deepcopy(base)
    if patch_candidate.get('name') is not None:
        merged['name'] = patch_candidate['name']
    if patch_candidate.get('description') is not None:
        merged['description'] = patch_candidate['description']
    if patch_candidate.get('rationale') is not None:
        merged['rationale'] = patch_candidate['rationale']

    if patch_candidate.get('scd_patch'):
        merged.setdefault('scd', {})
        merged['scd'].update(copy.deepcopy(patch_candidate['scd_patch']))

    merged_patches = copy.deepcopy(merged.get('iter_patches', {}))
    for iter_idx, patch in patch_candidate.get('iter_patches', {}).items():
        dst = merged_patches.setdefault(iter_idx, {})
        for key, value in patch.items():
            if isinstance(value, dict):
                current = copy.deepcopy(dst.get(key, {}))
                current.update(copy.deepcopy(value))
                dst[key] = current
            else:
                dst[key] = copy.deepcopy(value)
    merged['iter_patches'] = merged_patches

    if patch_candidate.get('post_rx_y_mult') is not None:
        merged['post_rx_y_mult'] = float(patch_candidate['post_rx_y_mult'])
    if patch_candidate.get('post_ry_z_mult') is not None:
        merged['post_ry_z_mult'] = float(patch_candidate['post_ry_z_mult'])
    return merged


def _merge_round65_candidate(extra_candidate: dict):
    merged = _apply_candidate_patch(_load_r61_base_candidate(), ROUND64_BASE_PATCH)
    merged = _apply_candidate_patch(merged, extra_candidate)
    merged['round64_base_patch'] = copy.deepcopy(ROUND64_BASE_PATCH)
    merged['round65_extra_patch'] = copy.deepcopy(extra_candidate)
    return merged


def _delta_block(curr: dict, ref: dict):
    return {k: float(curr[k] - ref[k]) for k in curr}


def _sorted_policy_patch(iter_patches: dict):
    out = {}
    for iter_idx, patch in sorted(iter_patches.items()):
        out[str(iter_idx + 1)] = {
            key: {str(k): float(v) for k, v in value.items()} if isinstance(value, dict) else value
            for key, value in patch.items()
        }
    return out


def _run_candidate(candidate: dict, idx: int):
    merged_candidate = _merge_round65_candidate(candidate)
    method_mod = load_module(f'markov_method_round65_probe_{idx}', str(R53_METHOD_FILE))
    method_mod = _build_patched_method(method_mod, merged_candidate)
    method_mod.METHOD = f"42-state GM1 round65 cov repair probe {merged_candidate['name']}"
    method_mod.VARIANT = f"42state_gm1_round65_covrepair_probe_{merged_candidate['name']}"

    source_mod = load_module(f'markov_pruned_source_for_round65_probe_{idx}', str(SOURCE_FILE))
    ts, pos0, imu_noisy, bi_g, bi_a, tau_g, tau_a = method_mod._build_dataset(source_mod)
    result = _run_internalized_hybrid_scd(
        method_mod,
        source_mod,
        imu_noisy,
        pos0,
        ts,
        bi_g=bi_g,
        bi_a=bi_a,
        tau_g=tau_g,
        tau_a=tau_a,
        label=f'42-GM1-R65-{idx}',
        scd_cfg=merged_candidate['scd'],
    )
    clbt = result[0]
    extra = result[4] if len(result) >= 5 else {}
    _, focus, lever, overall = _compute_metrics(source_mod, clbt)
    return merged_candidate, result, focus, lever, overall, extra


def _score_candidate(delta_vs_r61: dict, delta_vs_r64: dict):
    protected_metrics = ['dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'median_pct_error']
    penalties = []
    for key in protected_metrics:
        value = float(delta_vs_r61[key])
        if value > 1e-9:
            penalties.append({'metric': key, 'delta': value})

    repair_credit = 0.0
    repair_credit += 700.0 * max(-float(delta_vs_r64['dKg_yy']), 0.0)
    repair_credit += 700.0 * max(-float(delta_vs_r64['rx_y']), 0.0)
    repair_credit += 220.0 * max(-float(delta_vs_r64['mean_pct_error']), 0.0)
    repair_credit += 220.0 * max(-float(delta_vs_r64['max_pct_error']), 0.0)

    mainline_credit = 0.0
    mainline_credit += 180.0 * max(-float(delta_vs_r61['mean_pct_error']), 0.0)
    mainline_credit += 150.0 * max(-float(delta_vs_r61['max_pct_error']), 0.0)
    mainline_credit += 120.0 * max(-float(delta_vs_r61['dKg_xx']), 0.0)

    gate_penalty = sum(1000.0 * float(p['delta']) for p in penalties)
    return float(repair_credit + mainline_credit - gate_penalty), penalties


def _selection_note(delta_vs_r61: dict, delta_vs_r64: dict, penalties: list[dict]):
    repaired_yy = float(delta_vs_r64['dKg_yy']) < 0.0
    repaired_rx = float(delta_vs_r64['rx_y']) < 0.0

    if (not penalties) and (repaired_yy or repaired_rx) and (
        float(delta_vs_r61['mean_pct_error']) < 0.0
        or float(delta_vs_r61['max_pct_error']) < 0.0
        or float(delta_vs_r61['dKg_xx']) < 0.0
    ):
        return 'Clean repair win: repairs the Round64 target pair without introducing new protected regression versus Round61.'
    if repaired_yy and repaired_rx and penalties:
        return f'Repairs both Round64 target regressions, but still not a no-regression Round61 replacement: {penalties}'
    if repaired_yy or repaired_rx:
        repaired = []
        if repaired_yy:
            repaired.append('dKg_yy')
        if repaired_rx:
            repaired.append('rx_y')
        if penalties:
            return f"Partial repair signal: fixes {', '.join(repaired)} versus Round64, but protected regression remains versus Round61: {penalties}"
        return f"Partial repair signal: fixes {', '.join(repaired)} versus Round64, but the full Round61 gain set does not improve together."
    if penalties:
        return f'No useful repair signal; protected regression remains versus Round61: {penalties}'
    return 'Near-neutral branch, but it does not clearly repair the Round64 target pair.'


def _render_report(summary: dict):
    lines = []
    lines.append('<callout emoji="🧪" background-color="light-blue">')
    lines.append('Round65：沿着 **Round64 covariance-schedule** 方向做 repair-style 小批次，只修 `dKg_yy / rx_y`，保持 Round61 的 SCD cadence、trust-map 主体和同噪声同 seed 约束不变。')
    lines.append('</callout>')
    lines.append('')
    lines.append('## 1. Probe 设置')
    lines.append('')
    lines.append(f'- Standard KF baseline (same noisy seed=42) overall mean/max: `{summary["baseline_noisy_overall"]["mean_pct_error"]:.6f}` / `{summary["baseline_noisy_overall"]["max_pct_error"]:.6f}`')
    lines.append(f'- Mainline base candidate: `{summary["base_round61_candidate"]}`')
    lines.append(f'- Repair reference candidate: `{summary["base_round64_candidate"]}`')
    lines.append('- Dataset / regime: default 1x noisy baseline, deterministic seed=42')
    lines.append('- Batch size: **5 deterministic narrow repair candidates**')
    lines.append('- Guard rails: keep Round61 once-per-phase SCD (`alpha=0.999`, `transition_duration=2.0`, `iter2_commit`) unchanged; do not reopen broad trust-map reshaping')
    lines.append('')
    lines.append('## 2. Anchor metrics')
    lines.append('')
    lines.append(f'- Round61 overall mean / median / max: `{summary["baseline_r61"]["overall"]["mean_pct_error"]:.6f}` / `{summary["baseline_r61"]["overall"]["median_pct_error"]:.6f}` / `{summary["baseline_r61"]["overall"]["max_pct_error"]:.6f}`')
    lines.append(f'- Round64 repair base overall mean / median / max: `{summary["baseline_r64"]["overall"]["mean_pct_error"]:.6f}` / `{summary["baseline_r64"]["overall"]["median_pct_error"]:.6f}` / `{summary["baseline_r64"]["overall"]["max_pct_error"]:.6f}`')
    lines.append(f'- Round64 target regressions vs Round61: `dKg_yy {summary["baseline_r64_delta_vs_r61"]["dKg_yy"]:.6f}`, `rx_y {summary["baseline_r64_delta_vs_r61"]["rx_y"]:.6f}`, `mean {summary["baseline_r64_delta_vs_r61"]["mean_pct_error"]:.6f}`')
    lines.append('')
    lines.append('## 3. Candidate summary')
    lines.append('')
    lines.append('| candidate | yy repair vs R64 | rx_y repair vs R64 | mean repair vs R64 | max repair vs R64 | yy ΔvsR61 | rx_y ΔvsR61 | mean ΔvsR61 | max ΔvsR61 | score | note |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for name in summary['candidate_order']:
        cand = summary['candidates'][name]
        d64 = cand['delta_vs_r64']
        d61 = cand['delta_vs_r61']
        lines.append(
            f"| `{name}` | {d64['dKg_yy']:.6f} | {d64['rx_y']:.6f} | {d64['mean_pct_error']:.6f} | {d64['max_pct_error']:.6f} | {d61['dKg_yy']:.6f} | {d61['rx_y']:.6f} | {d61['mean_pct_error']:.6f} | {d61['max_pct_error']:.6f} | {cand['selection']['score']:.6f} | {cand['selection']['note']} |"
        )
    lines.append('')
    lines.append('## 4. Winner / status')
    lines.append('')
    lines.append(f"- Best repair candidate by score: `{summary['best_candidate']['name']}`")
    lines.append(f"- Best candidate note: {summary['best_candidate']['note']}")
    if summary['winner']:
        lines.append(f"- Mainline decision: promote `{summary['winner']['name']}`")
        lines.append(f"- Promotion reason: {summary['winner']['reason']}")
    else:
        lines.append('- Mainline decision: **do not change Round61**')
        lines.append(f"- Reason: {summary['no_winner_reason']}")
    lines.append('')
    lines.append('## 5. Mechanism readout')
    lines.append('')
    lines.append(f"- Best candidate vs Round64 target pair: `dKg_yy {summary['best_candidate']['delta_vs_r64']['dKg_yy']:.6f}`, `rx_y {summary['best_candidate']['delta_vs_r64']['rx_y']:.6f}`")
    lines.append(f"- Best candidate vs Round61 gate: `mean {summary['best_candidate']['delta_vs_r61']['mean_pct_error']:.6f}`, `max {summary['best_candidate']['delta_vs_r61']['max_pct_error']:.6f}`, `dKg_yy {summary['best_candidate']['delta_vs_r61']['dKg_yy']:.6f}`, `rx_y {summary['best_candidate']['delta_vs_r61']['rx_y']:.6f}`")
    lines.append('- Interpretation: if the batch repairs only rx_y or only dKg_yy but cannot clear the Round61 protected gate, treat it as a repair signal rather than a promotion candidate.')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    baseline_payload = json.loads(BASELINE_JSON.read_text(encoding='utf-8'))
    r61_payload = json.loads(R61_JSON.read_text(encoding='utf-8'))
    r64_payload = json.loads(R64_JSON.read_text(encoding='utf-8'))
    r64_base = r64_payload['candidates'][BASE_R64_CANDIDATE_NAME]

    candidate_dump = {
        'standard_kf_baseline_json': str(BASELINE_JSON),
        'base_round61_candidate': BASE_R61_CANDIDATE_NAME,
        'base_round64_candidate': BASE_R64_CANDIDATE_NAME,
        'round64_base_patch': ROUND64_BASE_PATCH,
        'round65_candidates': ROUND65_CANDIDATES,
        'direction': 'repair-style Round64 covariance-schedule follow-up under the same 1x noisy seed=42 benchmark',
    }
    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    baseline_noisy_overall = {
        'mean_pct_error': float(baseline_payload['summary']['overall_mean_pct_error']),
        'median_pct_error': float(baseline_payload['summary']['overall_median_pct_error']),
        'max_pct_error': float(baseline_payload['summary']['worst_param_pct_error']),
    }

    baseline_r64_delta_vs_r61 = {
        **_delta_block(r64_base['focus'], r61_payload['focus_scale_pct']),
        **_delta_block(r64_base['lever'], r61_payload['lever_guard_pct']),
        **_delta_block(r64_base['overall'], r61_payload['overall']),
    }

    out = {
        'standard_kf_baseline_json': str(BASELINE_JSON),
        'candidate_json': str(CANDIDATE_JSON),
        'base_round61_candidate': BASE_R61_CANDIDATE_NAME,
        'base_round64_candidate': BASE_R64_CANDIDATE_NAME,
        'dataset': 'default_1x_main_same_noisy_seed42',
        'baseline_noisy_overall': baseline_noisy_overall,
        'baseline_r61': {
            'focus': r61_payload['focus_scale_pct'],
            'lever': r61_payload['lever_guard_pct'],
            'overall': r61_payload['overall'],
        },
        'baseline_r64': {
            'focus': r64_base['focus'],
            'lever': r64_base['lever'],
            'overall': r64_base['overall'],
        },
        'baseline_r64_delta_vs_r61': baseline_r64_delta_vs_r61,
        'candidate_order': [c['name'] for c in ROUND65_CANDIDATES],
        'candidates': {},
        'best_candidate': None,
        'winner': None,
        'no_winner_reason': None,
    }

    for idx, candidate in enumerate(ROUND65_CANDIDATES, start=1):
        merged_candidate, result, focus, lever, overall, extra = _run_candidate(candidate, idx)
        delta_vs_r61 = {
            **_delta_block(focus, r61_payload['focus_scale_pct']),
            **_delta_block(lever, r61_payload['lever_guard_pct']),
            **_delta_block(overall, r61_payload['overall']),
        }
        delta_vs_r64 = {
            **_delta_block(focus, r64_base['focus']),
            **_delta_block(lever, r64_base['lever']),
            **_delta_block(overall, r64_base['overall']),
        }
        score, penalties = _score_candidate(delta_vs_r61, delta_vs_r64)
        note = _selection_note(delta_vs_r61, delta_vs_r64, penalties)

        out['candidates'][merged_candidate['name']] = {
            'description': merged_candidate['description'],
            'rationale': merged_candidate['rationale'],
            'base_round61_candidate': BASE_R61_CANDIDATE_NAME,
            'base_round64_candidate': BASE_R64_CANDIDATE_NAME,
            'policy_patch': _sorted_policy_patch(merged_candidate.get('iter_patches', {})),
            'round64_base_patch': copy.deepcopy(ROUND64_BASE_PATCH),
            'round65_extra_patch': copy.deepcopy(candidate),
            'scd': copy.deepcopy(merged_candidate['scd']),
            'post_rx_y_mult': float(merged_candidate.get('post_rx_y_mult', 1.0)),
            'post_ry_z_mult': float(merged_candidate.get('post_ry_z_mult', 1.0)),
            'focus': focus,
            'lever': lever,
            'overall': overall,
            'delta_vs_r61': delta_vs_r61,
            'delta_vs_r64': delta_vs_r64,
            'selection': {
                'score': float(score),
                'penalties': penalties,
                'note': note,
            },
            'extra': {
                'schedule_log': extra.get('schedule_log'),
                'feedback_log': extra.get('feedback_log'),
                'scd_log': extra.get('scd_log'),
            },
        }

        print(merged_candidate['name'], json.dumps({
            'delta_vs_r64': delta_vs_r64,
            'delta_vs_r61': delta_vs_r61,
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
    best_delta61 = best['delta_vs_r61']
    best_penalties = best['selection']['penalties']
    out['best_candidate'] = {
        'name': best_name,
        'score': float(best_score),
        'note': best['selection']['note'],
        'delta_vs_r64': best['delta_vs_r64'],
        'delta_vs_r61': best['delta_vs_r61'],
    }

    if (not best_penalties) and (float(best_delta61['mean_pct_error']) < 0.0 or float(best_delta61['max_pct_error']) < 0.0 or float(best_delta61['dKg_xx']) < 0.0):
        out['winner'] = {
            'name': best_name,
            'score': float(best_score),
            'reason': 'Best repair candidate cleared the Round61 protected gate while keeping the Round64 covariance-schedule direction under the same noisy seed=42 benchmark.',
        }
    else:
        out['winner'] = None
        out['no_winner_reason'] = 'No Round65 repair candidate converted the Round64 covariance-schedule signal into a clean no-regression Round61 replacement.'

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
