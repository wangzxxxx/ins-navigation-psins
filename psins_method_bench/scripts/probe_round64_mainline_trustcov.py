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
R61_JSON = RESULTS_DIR / 'R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors.json'
OUTPUT_JSON = RESULTS_DIR / 'round64_mainline_trustcov_probe_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'round64_mainline_trustcov_candidates.json'
REPORT_MD = REPORTS_DIR / 'psins_round64_mainline_trustcov_probe_2026-03-28.md'

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

ROUND64_CANDIDATES = [
    {
        'name': 'r64_trust_scorelean_softcov',
        'description': 'Iter2 trust-map rebalance: shift alpha mapping toward score signal (less covariance-dominant), keep SCD and per-state alpha multipliers unchanged.',
        'rationale': 'New 1x-mainline direction: instead of ultra-low gating, retune trust-map shape so selected-state feedback is decided more by x/sqrt(P) and less by pure covariance shrink.',
        'iter_patches': {
            1: {
                'trust_mix': 0.64,
                'trust_score_soft': 1.95,
                'trust_cov_soft': 0.52,
                'selected_alpha_floor': 0.958,
                'selected_alpha_span': 0.126,
            },
        },
    },
    {
        'name': 'r64_trust_span_up',
        'description': 'Iter2 trust-map contrast increase: slightly lower floor and larger span while softening covariance contribution.',
        'rationale': 'Checks whether a wider trust-map dynamic range can improve dKg_xx/max without touching the Round61 SCD cadence or ultra-low branch logic.',
        'iter_patches': {
            1: {
                'trust_mix': 0.62,
                'trust_score_soft': 2.00,
                'trust_cov_soft': 0.50,
                'selected_alpha_floor': 0.954,
                'selected_alpha_span': 0.132,
            },
        },
    },
    {
        'name': 'r64_cov_sched_static_meas_plus',
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
    },
    {
        'name': 'r64_combo_trust_cov_narrow',
        'description': 'Narrow combo: mild trust-map score-lean + mild iter2 Q/R schedule rebalance in one deterministic patch.',
        'rationale': 'Carefully combined route to test whether trust-map and schedule adjustments can cohere on 1x without the Round62/63 ultra-low gate path.',
        'iter_patches': {
            1: {
                'trust_mix': 0.62,
                'trust_score_soft': 1.98,
                'trust_cov_soft': 0.50,
                'selected_alpha_floor': 0.956,
                'selected_alpha_span': 0.128,
                'selected_q_static_scale': 0.790,
                'selected_q_dynamic_scale': 1.006,
                'selected_q_late_mult': 1.292,
                'static_r_scale': 1.022,
                'late_r_mult': 0.993,
                'late_release_frac': 0.576,
            },
        },
    },
]


def _load_r61_base_candidate():
    for candidate in ROUND61_CANDIDATES:
        if candidate['name'] == BASE_R61_CANDIDATE_NAME:
            return _merge_round61_candidate(candidate)
    raise KeyError(BASE_R61_CANDIDATE_NAME)


def _merge_round64_candidate(extra_candidate: dict):
    merged = copy.deepcopy(_load_r61_base_candidate())
    merged['name'] = extra_candidate['name']
    merged['description'] = extra_candidate['description']
    merged['rationale'] = extra_candidate['rationale']

    merged_patches = copy.deepcopy(merged.get('iter_patches', {}))
    for iter_idx, patch in extra_candidate.get('iter_patches', {}).items():
        dst = merged_patches.setdefault(iter_idx, {})
        for key, value in patch.items():
            if isinstance(value, dict):
                current = copy.deepcopy(dst.get(key, {}))
                current.update(copy.deepcopy(value))
                dst[key] = current
            else:
                dst[key] = copy.deepcopy(value)
    merged['iter_patches'] = merged_patches

    if extra_candidate.get('post_rx_y_mult') is not None:
        merged['post_rx_y_mult'] = float(extra_candidate['post_rx_y_mult'])
    if extra_candidate.get('post_ry_z_mult') is not None:
        merged['post_ry_z_mult'] = float(extra_candidate['post_ry_z_mult'])

    merged['round64_extra_patch'] = copy.deepcopy(extra_candidate)
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
    merged_candidate = _merge_round64_candidate(candidate)
    method_mod = load_module(f'markov_method_round64_probe_{idx}', str(R53_METHOD_FILE))
    method_mod = _build_patched_method(method_mod, merged_candidate)
    method_mod.METHOD = f"42-state GM1 round64 mainline trust/cov probe {merged_candidate['name']}"
    method_mod.VARIANT = f"42state_gm1_round64_mainline_probe_{merged_candidate['name']}"

    source_mod = load_module(f'markov_pruned_source_for_round64_probe_{idx}', str(SOURCE_FILE))
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
        label=f'42-GM1-R64-{idx}',
        scd_cfg=merged_candidate['scd'],
    )
    clbt = result[0]
    extra = result[4] if len(result) >= 5 else {}
    _, focus, lever, overall = _compute_metrics(source_mod, clbt)
    return merged_candidate, result, focus, lever, overall, extra


def _score_candidate(delta_vs_r61: dict):
    penalties = []
    protected_metrics = ['dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'median_pct_error']
    for key in protected_metrics:
        value = float(delta_vs_r61[key])
        if value > 1e-9:
            penalties.append({'metric': key, 'delta': value})

    score = 0.0
    score += -1.20 * float(delta_vs_r61['mean_pct_error'])
    score += -1.00 * float(delta_vs_r61['max_pct_error'])
    score += -0.80 * float(delta_vs_r61['dKg_xx'])
    score += -0.30 * float(delta_vs_r61['median_pct_error'])

    for p in penalties:
        score -= 1000.0 * float(p['delta'])

    return float(score), penalties


def _selection_note(delta_vs_r61: dict, penalties: list[dict]):
    if penalties:
        return f'Protected regression detected: {penalties}'

    if delta_vs_r61['mean_pct_error'] < 0 and delta_vs_r61['max_pct_error'] <= 0:
        return 'Clean 1x win: improves mean while holding/improving max and protected metrics.'
    if delta_vs_r61['max_pct_error'] < 0 and delta_vs_r61['mean_pct_error'] <= 0:
        return 'Clean 1x win: improves max while holding/improving mean and protected metrics.'
    if delta_vs_r61['mean_pct_error'] < 0 or delta_vs_r61['max_pct_error'] < 0 or delta_vs_r61['dKg_xx'] < 0:
        return 'Partial 1x gain signal, but mean/max/protected metrics do not improve together strongly.'
    return 'No useful 1x gain signal over Round61.'


def _render_report(summary: dict):
    lines = []
    lines.append('<callout emoji="🧪" background-color="light-blue">')
    lines.append('Round64：从 Round61 主干出发，做 **mainline trust-map / covariance-schedule rebalance**，不走 Round62/63 的 ultra-low SCD gating 修复线。')
    lines.append('</callout>')
    lines.append('')
    lines.append('## 1. Probe 设置')
    lines.append('')
    lines.append(f'- Base candidate: `{summary["base_round61_candidate"]}`')
    lines.append('- Dataset: default 1x (deterministic seed=42)')
    lines.append('- Batch size: **4 deterministic narrow candidates**')
    lines.append('- Mainline guard: keep Round61 SCD cadence (`once_per_phase`, `alpha=0.999`, `iter2_commit`) unchanged')
    lines.append('')
    lines.append('## 2. 候选摘要（相对 Round61）')
    lines.append('')
    lines.append('| candidate | dKg_xx Δ | dKg_xy Δ | dKg_yy Δ | dKg_zz Δ | dKa_xx Δ | rx_y Δ | ry_z Δ | mean Δ | median Δ | max Δ | score | note |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for name in summary['candidate_order']:
        cand = summary['candidates'][name]
        d = cand['delta_vs_r61']
        lines.append(
            f"| `{name}` | {d['dKg_xx']:.6f} | {d['dKg_xy']:.6f} | {d['dKg_yy']:.6f} | {d['dKg_zz']:.6f} | {d['dKa_xx']:.6f} | {d['rx_y']:.6f} | {d['ry_z']:.6f} | {d['mean_pct_error']:.6f} | {d['median_pct_error']:.6f} | {d['max_pct_error']:.6f} | {cand['selection']['score']:.6f} | {cand['selection']['note']} |"
        )
    lines.append('')
    lines.append('## 3. Winner / status')
    lines.append('')
    if summary['winner']:
        lines.append(f"- Winner: `{summary['winner']['name']}`")
        lines.append(f"- Reason: {summary['winner']['reason']}")
    else:
        lines.append('- Winner: **none**')
        lines.append(f"- Reason: {summary['no_winner_reason']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    r61_payload = json.loads(R61_JSON.read_text(encoding='utf-8'))

    candidate_dump = {
        'base_round61_candidate': BASE_R61_CANDIDATE_NAME,
        'round64_candidates': ROUND64_CANDIDATES,
        'direction': 'mainline trust-map / covariance-schedule rebalance at 1x',
    }
    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    out = {
        'base_round61_candidate': BASE_R61_CANDIDATE_NAME,
        'dataset': 'default_1x_main',
        'candidate_order': [c['name'] for c in ROUND64_CANDIDATES],
        'baseline_r61': {
            'focus': r61_payload['focus_scale_pct'],
            'lever': r61_payload['lever_guard_pct'],
            'overall': r61_payload['overall'],
        },
        'candidate_json': str(CANDIDATE_JSON),
        'candidates': {},
        'winner': None,
        'no_winner_reason': None,
    }

    for idx, candidate in enumerate(ROUND64_CANDIDATES, start=1):
        merged_candidate, result, focus, lever, overall, extra = _run_candidate(candidate, idx)
        delta_vs_r61 = {
            **_delta_block(focus, r61_payload['focus_scale_pct']),
            **_delta_block(lever, r61_payload['lever_guard_pct']),
            **_delta_block(overall, r61_payload['overall']),
        }
        score, penalties = _score_candidate(delta_vs_r61)
        note = _selection_note(delta_vs_r61, penalties)

        out['candidates'][merged_candidate['name']] = {
            'description': merged_candidate['description'],
            'rationale': merged_candidate['rationale'],
            'base_round61_candidate': BASE_R61_CANDIDATE_NAME,
            'policy_patch': _sorted_policy_patch(merged_candidate.get('iter_patches', {})),
            'round64_extra_patch': copy.deepcopy(candidate),
            'scd': copy.deepcopy(merged_candidate['scd']),
            'post_rx_y_mult': float(merged_candidate.get('post_rx_y_mult', 1.0)),
            'post_ry_z_mult': float(merged_candidate.get('post_ry_z_mult', 1.0)),
            'focus': focus,
            'lever': lever,
            'overall': overall,
            'delta_vs_r61': delta_vs_r61,
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
    best_delta = best['delta_vs_r61']
    best_penalties = best['selection']['penalties']

    if (not best_penalties) and best_score > 0.01 and (
        best_delta['mean_pct_error'] < 0 or best_delta['max_pct_error'] < 0 or best_delta['dKg_xx'] < 0
    ):
        out['winner'] = {
            'name': best_name,
            'score': float(best_score),
            'reason': 'Best no-regression 1x mainline trust/cov candidate versus Round61 on weighted mean/max/dKg_xx score.',
        }
    else:
        out['winner'] = None
        out['no_winner_reason'] = 'No Round64 candidate produced a clean no-regression 1x improvement over Round61 on the main weighted target set.'

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    REPORT_MD.write_text(_render_report(out), encoding='utf-8')
    print(f'Wrote {OUTPUT_JSON}')
    print(f'Wrote {REPORT_MD}')
    print('__RESULT_JSON__=' + json.dumps({
        'output_json': str(OUTPUT_JSON),
        'candidate_json': str(CANDIDATE_JSON),
        'report_md': str(REPORT_MD),
        'winner': out['winner'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
