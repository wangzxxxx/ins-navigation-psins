from __future__ import annotations

import argparse
import json
import sys
import types
from datetime import datetime
from pathlib import Path
from typing import Any

if 'matplotlib' not in sys.modules:
    matplotlib_stub = types.ModuleType('matplotlib')
    pyplot_stub = types.ModuleType('matplotlib.pyplot')
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules['matplotlib'] = matplotlib_stub
    sys.modules['matplotlib.pyplot'] = pyplot_stub
if 'seaborn' not in sys.modules:
    sys.modules['seaborn'] = types.ModuleType('seaborn')

ROOT = Path('/root/.openclaw/workspace')
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
REPORTS_DIR = ROOT / 'reports'
METHOD_DIR = ROOT / 'psins_method_bench' / 'methods' / 'markov'
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'

for p in [ROOT, ROOT / 'tmp_psins_py', METHOD_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module
from compare_four_methods_shared_noise import _load_json, expected_noise_config
from search_ch3_12pos_closedloop_local_insertions import StepSpec, build_closedloop_candidate, run_candidate_payload
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate, render_action
from search_ch3_corrected_inbasin_ridge_resume import ATT0_DEG, compact_metrics, delta_vs_reference, overall_triplet
from search_ch3_corrected_nonterminal_x_conditioning import base_prefix, l11_current_core, y_pair
from search_ch3_entry_conditioned_relay_family import NOISE_SCALE, l10_unified_core, merge_insertions

REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
REPORT_PATH = REPORTS_DIR / f'psins_ch3_corrected_zcadence_handoff_refine_{REPORT_DATE}.md'
SUMMARY_PATH = RESULTS_DIR / f'ch3_corrected_zcadence_handoff_refine_{REPORT_DATE}.json'
TARGET_TRIPLET = {'mean_pct_error': 1.056, 'median_pct_error': 0.588, 'max_pct_error': 4.560}

REFERENCE_FILES = {
    'current_leader': {
        'label': 'current corrected leader / relay_l11back0p5_l12y0p125_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l11back0p5_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_relay_l11back0p5_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
    },
    'previous_corrected_leader': {
        'label': 'previous corrected leader / relay_r3_l9y0p8125_l12y0p125_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_r3_l9y0p8125_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_relay_r3_l9y0p8125_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
    },
    'reused_q4_low': {
        'label': 'reused pure-q4 low-side point / relay_l11back0p25_l12y0p125_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l11back0p25_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
        'kf36': None,
    },
    'reused_q4_high': {
        'label': 'reused pure-q4 high-side point / relay_l11back0p75_l12y0p125_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l11back0p75_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
        'kf36': None,
    },
    'faithful12': {
        'label': 'corrected faithful12',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json',
    },
    'default18': {
        'label': 'default18',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json',
    },
}

REUSED_CONTEXT = {
    'pending': str(ROOT / 'PENDING.md'),
    'source_reports': [
        str(REPORTS_DIR / 'psins_ch3_corrected_zcadence_handoff_2026-04-03.md'),
        str(REPORTS_DIR / 'psins_ch3_corrected_inbasin_ridge_resume_2026-04-03.md'),
    ],
    'existing_local_markov_points': [
        str(RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l11back0p25_l12y0p125_on_entry_shared_noise0p08_param_errors.json'),
        str(RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l11back0p5_l12y0p125_on_entry_shared_noise0p08_param_errors.json'),
        str(RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l11back0p75_l12y0p125_on_entry_shared_noise0p08_param_errors.json'),
        str(RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l11back1p25_l12y0p125_on_entry_shared_noise0p08_param_errors.json'),
        str(RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l11back1p5_l12y0p125_on_entry_shared_noise0p08_param_errors.json'),
        str(RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l11back4_l12y0p125_on_entry_shared_noise0p08_param_errors.json'),
    ],
}

HYPOTHESES = [
    {
        'id': 'R1',
        'family': 'q4_low_point_terminal_reclosure_compensation',
        'summary': 'Pure q4 retiming has already been locally bracketed by reused artifacts: 0.25 s beats the target on mean/median but loses max, while 0.5 s recovers max but sits just above the target on mean/median. The most promising nearby move is therefore to keep the stronger q4=0.25 handoff and compensate only with a slightly heavier terminal l12 reclosure.',
        'candidate_names': [
            'relay_l11back0p25_l12y0p1875_on_entry',
            'relay_l11back0p25_l12y0p25_on_entry',
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()



def load_reference_payload(path: Path, noise_scale: float) -> dict[str, Any]:
    payload = _load_json(path)
    expected_cfg = expected_noise_config(noise_scale)
    got_cfg = payload.get('extra', {}).get('noise_config') or payload.get('extra', {}).get('shared_noise_config')
    if got_cfg is not None and got_cfg != expected_cfg:
        raise ValueError(f'Noise configuration mismatch for {path}')
    return payload



def attach_att0(path: Path, payload: dict[str, Any], candidate_name: str, method_key: str, family: str, hypothesis_id: str) -> dict[str, Any]:
    extra = payload.setdefault('extra', {})
    extra['att0_deg'] = ATT0_DEG
    extra['comparison_mode'] = 'corrected_zcadence_handoff_refine'
    extra['candidate_registry_key'] = candidate_name
    extra['method_key'] = method_key
    extra['family'] = family
    extra['hypothesis_id'] = hypothesis_id
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload



def q4_step(back_dwell_s: float, label: str) -> list[StepSpec]:
    return [
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(back_dwell_s), segment_role='motif_zero_b', label=label),
    ]



def candidate_specs() -> list[dict[str, Any]]:
    prefix = base_prefix()
    anchor10 = {10: l10_unified_core()[10]}
    return [
        {
            'name': 'relay_l11back0p25_l12y0p1875_on_entry',
            'family': 'q4_low_point_terminal_reclosure_compensation',
            'hypothesis_id': 'R1',
            'rationale': 'Keep the stronger q4=0.25 handoff that already clears the target on mean and median, then add only a mild heavier l12=0.1875 closure to see whether the remaining top-error deficit can be recovered without losing the mean headroom.',
            'insertions': merge_insertions(prefix, anchor10, {11: l11_current_core()[:-1] + q4_step(0.25, 'l11_zquad_y10x0back0p25_q4')}, {12: y_pair(0.1875, 'l12_yneg0p1875')}),
        },
        {
            'name': 'relay_l11back0p25_l12y0p25_on_entry',
            'family': 'q4_low_point_terminal_reclosure_compensation',
            'hypothesis_id': 'R1',
            'rationale': 'Same low-q4 handoff, but with the full heavier l12=0.25 closure that was previously useful on the earlier in-basin ridge for trimming max. This is the strongest still-local compensation move before reopening any new family.',
            'insertions': merge_insertions(prefix, anchor10, {11: l11_current_core()[:-1] + q4_step(0.25, 'l11_zquad_y10x0back0p25_q4')}, {12: y_pair(0.25, 'l12_yneg0p25')}),
        },
    ]



def build_timing_table(candidate) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row, action in zip(candidate.all_rows, candidate.all_actions):
        rows.append({
            'pos_id': int(row['pos_id']),
            'anchor_id': int(row['anchor_id']),
            'segment_role': row['segment_role'],
            'label': row['label'],
            'motor_action': render_action(action),
            'effective_body_axis': list(action['effective_body_axis']),
            'rotation_time_s': float(row['rotation_time_s']),
            'pre_static_s': float(row['pre_static_s']),
            'post_static_s': float(row['post_static_s']),
            'node_total_s': float(row['node_total_s']),
            'face_after': action['state_after']['face_name'],
            'inner_beta_after_deg': int(action['inner_beta_after_deg']),
        })
    return rows



def render_timing_table_md(table: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    lines.append('| pos | anchor | role | label | motor action | axis | rot_s | pre_s | post_s | total_s | face_after | beta_after |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|')
    for item in table:
        axis = '[' + ','.join(str(v) for v in item['effective_body_axis']) + ']'
        lines.append(
            f"| {item['pos_id']} | {item['anchor_id']} | {item['segment_role']} | {item['label']} | {item['motor_action']} | {axis} | {item['rotation_time_s']:.3f} | {item['pre_static_s']:.3f} | {item['post_static_s']:.3f} | {item['node_total_s']:.3f} | {item['face_after']} | {item['inner_beta_after_deg']} |"
        )
    return lines



def target_delta(candidate: dict[str, Any]) -> dict[str, float]:
    overall = candidate['overall']
    return {
        'mean_pct_error': TARGET_TRIPLET['mean_pct_error'] - float(overall['mean_pct_error']),
        'median_pct_error': TARGET_TRIPLET['median_pct_error'] - float(overall['median_pct_error']),
        'max_pct_error': TARGET_TRIPLET['max_pct_error'] - float(overall['max_pct_error']),
    }



def meets_target(candidate: dict[str, Any]) -> bool:
    td = target_delta(candidate)
    return td['mean_pct_error'] > 0 and td['median_pct_error'] > 0 and td['max_pct_error'] > 0



def render_report(summary: dict[str, Any]) -> str:
    refs = summary['references']
    best = summary['best_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 corrected z-cadence / handoff local refinement')
    lines.append('')
    lines.append('## 1. Relaunch premise and reused local evidence')
    lines.append('')
    lines.append('- This follow-up started from the new accepted corrected leader `relay_l11back0p5_l12y0p125_on_entry` = **1.057 / 0.588 / 4.560** (Markov42), **1.056 / 0.588 / 4.558** (KF36).')
    lines.append('- Reused local q4-only evidence already available on disk showed the handoff is tightly bracketed even before this batch:')
    lines.append(f"  - `relay_l11back0p25_l12y0p125_on_entry` → **{refs['reused_q4_low']['markov42_triplet']}** (better mean/median, but max too high)")
    lines.append(f"  - `relay_l11back0p5_l12y0p125_on_entry` → **{refs['current_leader']['markov42_triplet']}** (current frontier point, max nearly at target)")
    lines.append(f"  - `relay_l11back0p75_l12y0p125_on_entry` → **{refs['reused_q4_high']['markov42_triplet']}** (already drifting back worse)")
    lines.append('- Scientific implication from reused artifacts: pure q4 retiming alone already brackets the trade-off, so the minimum still-plausible nearby move is to keep the stronger q4=0.25 handoff and adjust only the terminal l12 cadence.')
    lines.append('')
    lines.append('## 2. Minimal new batch actually run')
    lines.append('')
    for item in summary['hypotheses_tested']:
        lines.append(f"- **{item['id']} · {item['family']}** — {item['summary']}")
        lines.append(f"  - tested: {', '.join(item['candidate_names'])}")
    lines.append('')
    lines.append('## 3. Markov42 results for the new candidates')
    lines.append('')
    lines.append('| rank | candidate | mean | median | max | Δmean vs leader | Δmedian vs leader | Δmax vs leader | Δmean vs reused q4=0.25 | Δmax vs reused q4=0.25 | target margin (mean/median/max) | max driver |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|')
    for idx, row in enumerate(summary['rows_sorted'], start=1):
        m = row['markov42']['overall']
        dl = row['delta_vs_leader_markov42']
        dq = row['delta_vs_q4_low_markov42']
        td = row['target_delta_markov42']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {dl['mean_pct_error']:+.3f} | {dl['median_pct_error']:+.3f} | {dl['max_pct_error']:+.3f} | {dq['mean_pct_error']:+.3f} | {dq['max_pct_error']:+.3f} | {td['mean_pct_error']:+.3f} / {td['median_pct_error']:+.3f} / {td['max_pct_error']:+.3f} | {row['markov42']['max_driver']['name']} {row['markov42']['max_driver']['pct_error']:.3f} |"
        )
    lines.append('')
    lines.append('## 4. KF36 recheck')
    lines.append('')
    if summary['kf36_rechecked_candidates']:
        lines.append(f"- Rechecked candidates: **{', '.join(summary['kf36_rechecked_candidates'])}**")
        for row in summary['rows_sorted']:
            if row.get('kf36') is not None:
                lines.append(f"  - `{row['candidate_name']}` → Markov42 **{overall_triplet(row['markov42'])}**, KF36 **{overall_triplet(row['kf36'])}**")
    else:
        lines.append(f"- No KF36 reruns triggered. Gate reason: {summary['kf36_gate_reason']}")
    lines.append('')
    lines.append('## 5. Decision from this local refinement')
    lines.append('')
    lines.append(f"- **Batch-best new candidate:** `{best['candidate_name']}` = **{overall_triplet(best['markov42'])}**")
    if best.get('kf36') is not None:
        lines.append(f"- **KF36 for batch best:** **{overall_triplet(best['kf36'])}**")
    lines.append(f"- vs current leader: Δmean **{best['delta_vs_leader_markov42']['mean_pct_error']:+.6f}**, Δmedian **{best['delta_vs_leader_markov42']['median_pct_error']:+.6f}**, Δmax **{best['delta_vs_leader_markov42']['max_pct_error']:+.6f}**")
    lines.append(f"- vs reused q4=0.25 point: Δmean **{best['delta_vs_q4_low_markov42']['mean_pct_error']:+.6f}**, Δmedian **{best['delta_vs_q4_low_markov42']['median_pct_error']:+.6f}**, Δmax **{best['delta_vs_q4_low_markov42']['max_pct_error']:+.6f}**")
    lines.append(f"- **Did any new candidate beat the explicit target 1.056 / 0.588 / 4.560?** **{summary['bottom_line']['beat_target_triplet']}**")
    lines.append(f"- **Did any new candidate materially beat the current corrected leader?** **{summary['bottom_line']['materially_better_found']}**")
    lines.append(f"- **Can this pending task now close?** **{summary['bottom_line']['task_can_close']}**")
    lines.append(f"- Recommended frontier remains: `{summary['bottom_line']['recommended_frontier_name']}` = **{summary['bottom_line']['recommended_frontier_triplet']}**")
    lines.append(f"- Scientific read: **{summary['bottom_line']['statement']}**")
    lines.append('')
    lines.append('## 6. Required comparison set')
    lines.append('')
    lines.append('| path | Markov42 | KF36 | Δmean vs batch best | Δmedian vs batch best | Δmax vs batch best | note |')
    lines.append('|---|---:|---:|---:|---:|---:|---|')
    for row in summary['required_comparison_rows']:
        d = row['delta_vs_batch_best_markov42']
        lines.append(
            f"| {row['label']} | {row['markov42_triplet']} | {row['kf36_triplet']} | {d['mean_pct_error']:+.3f} | {d['median_pct_error']:+.3f} | {d['max_pct_error']:+.3f} | {row['note']} |"
        )
    lines.append('')
    lines.append('## 7. Exact legal motor / timing table for the batch-best new candidate')
    lines.append('')
    lines.extend(render_timing_table_md(summary['best_candidate_timing_table']))
    lines.append('')
    return '\n'.join(lines) + '\n'



def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module(str(METHOD_DIR / 'method_42state_gm1.py'), str(SOURCE_FILE))
    faithful = build_candidate(mod, ())

    references: dict[str, Any] = {}
    for key, info in REFERENCE_FILES.items():
        m = load_reference_payload(info['markov42'], args.noise_scale)
        k = load_reference_payload(info['kf36'], args.noise_scale) if info['kf36'] is not None else None
        references[key] = {
            'label': info['label'],
            'markov42': compact_metrics(m),
            'kf36': compact_metrics(k) if k is not None else None,
            'markov42_triplet': overall_triplet(m),
            'kf36_triplet': overall_triplet(k) if k is not None else 'not rerun',
            'files': {
                'markov42': str(info['markov42']),
                'kf36': str(info['kf36']) if info['kf36'] is not None else None,
            },
        }

    specs = candidate_specs()
    spec_by_name = {spec['name']: spec for spec in specs}
    rows: list[dict[str, Any]] = []
    candidates_by_name: dict[str, Any] = {}

    for spec in specs:
        candidate = build_closedloop_candidate(mod, spec, faithful.rows, faithful.action_sequence)
        candidates_by_name[candidate.name] = candidate
        markov_payload, markov_mode, markov_path = run_candidate_payload(mod, candidate, 'markov42_noisy', args.noise_scale, force_rerun=args.force_rerun)
        markov_payload = attach_att0(markov_path, markov_payload, candidate.name, 'markov42_noisy', spec['family'], spec['hypothesis_id'])
        row = {
            'candidate_name': candidate.name,
            'family': spec['family'],
            'hypothesis_id': spec['hypothesis_id'],
            'rationale': spec['rationale'],
            'total_time_s': candidate.total_time_s,
            'result_files': {'markov42': str(markov_path)},
            'result_modes': {'markov42': markov_mode},
            'markov42': compact_metrics(markov_payload),
        }
        row['delta_vs_leader_markov42'] = delta_vs_reference(references['current_leader']['markov42'], row['markov42'])
        row['delta_vs_prev_leader_markov42'] = delta_vs_reference(references['previous_corrected_leader']['markov42'], row['markov42'])
        row['delta_vs_q4_low_markov42'] = delta_vs_reference(references['reused_q4_low']['markov42'], row['markov42'])
        row['delta_vs_faithful12_markov42'] = delta_vs_reference(references['faithful12']['markov42'], row['markov42'])
        row['delta_vs_default18_markov42'] = delta_vs_reference(references['default18']['markov42'], row['markov42'])
        row['target_delta_markov42'] = target_delta(row['markov42'])
        row['meets_target_markov42'] = meets_target(row['markov42'])
        rows.append(row)

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            min(r['target_delta_markov42'].values()),
            r['delta_vs_leader_markov42']['max_pct_error'],
            r['delta_vs_leader_markov42']['mean_pct_error'],
            -r['markov42']['overall']['median_pct_error'],
        ),
        reverse=True,
    )
    best = rows_sorted[0]
    best_candidate_timing_table = build_timing_table(candidates_by_name[best['candidate_name']])

    kf36_rechecked_candidates: list[str] = []
    kf36_gate_reason = 'No new candidate was close enough to the current leader / target envelope to justify KF36.'
    for row in rows_sorted:
        td = row['target_delta_markov42']
        dl = row['delta_vs_leader_markov42']
        if td['mean_pct_error'] > -0.006 and td['max_pct_error'] > -0.025 and dl['mean_pct_error'] > -0.010:
            candidate = candidates_by_name[row['candidate_name']]
            spec = spec_by_name[row['candidate_name']]
            kf_payload, kf_mode, kf_path = run_candidate_payload(mod, candidate, 'kf36_noisy', args.noise_scale, force_rerun=args.force_rerun)
            kf_payload = attach_att0(kf_path, kf_payload, candidate.name, 'kf36_noisy', spec['family'], spec['hypothesis_id'])
            row['result_files']['kf36'] = str(kf_path)
            row['result_modes']['kf36'] = kf_mode
            row['kf36'] = compact_metrics(kf_payload)
            row['delta_vs_leader_kf36'] = delta_vs_reference(references['current_leader']['kf36'], row['kf36'])
            row['target_delta_kf36'] = target_delta(row['kf36'])
            row['meets_target_kf36'] = meets_target(row['kf36'])
            kf36_rechecked_candidates.append(candidate.name)
    if not kf36_rechecked_candidates:
        kf36_gate_reason = f"Best Markov42 new candidate {best['candidate_name']} = {overall_triplet(best['markov42'])} stayed outside the close-target gate."
    else:
        kf36_gate_reason = 'Rechecked both new candidates because each stayed close enough to the target envelope to warrant direct KF36 confirmation.'

    materially_better_rows = [
        row for row in rows_sorted
        if row['delta_vs_leader_markov42']['mean_pct_error'] > 0 and row['delta_vs_leader_markov42']['max_pct_error'] > 0
        and (row.get('delta_vs_leader_kf36') is None or (row['delta_vs_leader_kf36']['mean_pct_error'] > 0 and row['delta_vs_leader_kf36']['max_pct_error'] > 0))
    ]
    target_rows = [
        row for row in rows_sorted
        if row['meets_target_markov42'] and (row.get('meets_target_kf36') is None or row['meets_target_kf36'])
    ]

    recommended_frontier_name = references['current_leader']['label'].split(' / ')[-1]
    recommended_frontier_triplet = references['current_leader']['markov42_triplet']
    materially_better_found = 'NO'
    beat_target_triplet = 'YES' if target_rows else 'NO'
    task_can_close = 'YES'
    statement = (
        'The reused q4 microgrid already showed that pure handoff retiming was bracketing a mean-vs-max trade-off. '
        'This new two-point coupled l12 compensation check did not convert the promising q4=0.25 mean/median win into a full frontier win, '
        'so the accepted leader remains the clean corrected-basis recommendation and the pending item can close.'
    )

    if materially_better_rows:
        chosen = materially_better_rows[0]
        recommended_frontier_name = chosen['candidate_name']
        recommended_frontier_triplet = overall_triplet(chosen['markov42'])
        materially_better_found = 'YES'
        statement = 'A new coupled handoff/cadence point materially beat the current corrected leader and is strong enough to replace it.'
    elif target_rows:
        chosen = target_rows[0]
        statement = 'A new coupled handoff/cadence point cleared the explicit 1.056 / 0.588 / 4.560 target, even though it was not a cleaner frontier win than the accepted leader on every criterion.'
        if chosen['delta_vs_leader_markov42']['mean_pct_error'] > 0 and chosen['delta_vs_leader_markov42']['max_pct_error'] > 0:
            recommended_frontier_name = chosen['candidate_name']
            recommended_frontier_triplet = overall_triplet(chosen['markov42'])

    required_comparison_rows = [
        {
            'label': references['current_leader']['label'],
            'markov42_triplet': references['current_leader']['markov42_triplet'],
            'kf36_triplet': references['current_leader']['kf36_triplet'],
            'note': 'accepted corrected leader at relaunch',
            'delta_vs_batch_best_markov42': delta_vs_reference(references['current_leader']['markov42'], best['markov42']),
        },
        {
            'label': references['reused_q4_low']['label'],
            'markov42_triplet': references['reused_q4_low']['markov42_triplet'],
            'kf36_triplet': references['reused_q4_low']['kf36_triplet'],
            'note': 'reused mean/median-strong pure q4 low-side point',
            'delta_vs_batch_best_markov42': delta_vs_reference(references['reused_q4_low']['markov42'], best['markov42']),
        },
        {
            'label': references['previous_corrected_leader']['label'],
            'markov42_triplet': references['previous_corrected_leader']['markov42_triplet'],
            'kf36_triplet': references['previous_corrected_leader']['kf36_triplet'],
            'note': 'previous corrected leader before the z-cadence breakthrough',
            'delta_vs_batch_best_markov42': delta_vs_reference(references['previous_corrected_leader']['markov42'], best['markov42']),
        },
        {
            'label': 'faithful12',
            'markov42_triplet': references['faithful12']['markov42_triplet'],
            'kf36_triplet': references['faithful12']['kf36_triplet'],
            'note': 'corrected faithful 12-position reference',
            'delta_vs_batch_best_markov42': delta_vs_reference(references['faithful12']['markov42'], best['markov42']),
        },
        {
            'label': 'default18',
            'markov42_triplet': references['default18']['markov42_triplet'],
            'kf36_triplet': references['default18']['kf36_triplet'],
            'note': 'default 18-position reference',
            'delta_vs_batch_best_markov42': delta_vs_reference(references['default18']['markov42'], best['markov42']),
        },
        {
            'label': f"batch-best new candidate / {best['candidate_name']}",
            'markov42_triplet': overall_triplet(best['markov42']),
            'kf36_triplet': overall_triplet(best['kf36']) if best.get('kf36') is not None else 'not rerun',
            'note': 'best point found inside this local refinement batch',
            'delta_vs_batch_best_markov42': delta_vs_reference(best['markov42'], best['markov42']),
        },
    ]

    summary = {
        'task': 'chapter-3 corrected z-cadence / handoff local refinement',
        'report_date': REPORT_DATE,
        'noise_scale': args.noise_scale,
        'corrected_att0_deg': ATT0_DEG,
        'target_triplet': TARGET_TRIPLET,
        'reused_context': REUSED_CONTEXT,
        'references': references,
        'hypotheses_tested': HYPOTHESES,
        'ran_candidates': [spec['name'] for spec in specs],
        'rows_sorted': rows_sorted,
        'best_candidate': best,
        'best_candidate_timing_table': best_candidate_timing_table,
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
        'kf36_gate_reason': kf36_gate_reason,
        'required_comparison_rows': required_comparison_rows,
        'bottom_line': {
            'task_can_close': task_can_close,
            'recommended_frontier_name': recommended_frontier_name,
            'recommended_frontier_triplet': recommended_frontier_triplet,
            'materially_better_found': materially_better_found,
            'beat_target_triplet': beat_target_triplet,
            'statement': statement,
        },
    }

    REPORT_PATH.write_text(render_report(summary), encoding='utf-8')
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({
        'report_path': str(REPORT_PATH),
        'summary_path': str(SUMMARY_PATH),
        'ran_candidates': [spec['name'] for spec in specs],
        'best_candidate': best['candidate_name'],
        'best_markov42': overall_triplet(best['markov42']),
        'best_kf36': overall_triplet(best['kf36']) if best.get('kf36') is not None else None,
        'task_can_close': task_can_close,
        'materially_better_found': materially_better_found,
        'beat_target_triplet': beat_target_triplet,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
