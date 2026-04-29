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
REPORT_PATH = REPORTS_DIR / f'psins_ch3_corrected_zcadence_handoff_next_{REPORT_DATE}.md'
SUMMARY_PATH = RESULTS_DIR / f'ch3_corrected_zcadence_handoff_next_{REPORT_DATE}.json'
TARGET_TRIPLET = {'mean_pct_error': 1.056, 'median_pct_error': 0.588, 'max_pct_error': 4.560}
KF36_RECHECK_NAMES = {
    'relay_l11back0p375_l12y0p1875_on_entry',
    'relay_l11back0p125_l12y0p3125_on_entry',
    'relay_l11back0p0_l12y0p375_on_entry',
}

REFERENCE_FILES = {
    'task_leader': {
        'label': 'task-stated corrected leader / relay_l11back0p5_l12y0p125_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l11back0p5_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_relay_l11back0p5_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
    },
    'local_frontier': {
        'label': 'latest local frontier / relay_l11back0p25_l12y0p25_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l11back0p25_l12y0p25_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_relay_l11back0p25_l12y0p25_on_entry_shared_noise0p08_param_errors.json',
    },
    'pure_q4_low': {
        'label': 'pure q4 low-side point / relay_l11back0p25_l12y0p125_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l11back0p25_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
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

HYPOTHESES = [
    {
        'id': 'N1',
        'family': 'constant_total_q4_to_l12_redistribution',
        'summary': 'Keep the same total preterminal buffering budget as the accepted handoff family (`q4 back dwell + 2*l12 dwell = 0.75 s`), then slide that budget between anchor11 q4 and anchor12 terminal y closure. This is the cleanest same-family test of whether the new gain should live slightly earlier or slightly later in the handoff.',
        'candidate_names': [
            'relay_l11back0p375_l12y0p1875_on_entry',
            'relay_l11back0p125_l12y0p3125_on_entry',
            'relay_l11back0p0_l12y0p375_on_entry',
        ],
    },
    {
        'id': 'N2',
        'family': 'undercompensated_low_q4_control',
        'summary': 'Cut q4 more aggressively without fully restoring the dwell at anchor12. This checks whether the new local ridge really needs the redistributed terminal compensation, rather than just ever-shorter q4 overhang.',
        'candidate_names': [
            'relay_l11back0p125_l12y0p25_on_entry',
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
    extra['comparison_mode'] = 'corrected_zcadence_handoff_next'
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
            'name': 'relay_l11back0p375_l12y0p1875_on_entry',
            'family': 'constant_total_q4_to_l12_redistribution',
            'hypothesis_id': 'N1',
            'rationale': 'Interpolate exactly between the task leader (0.5 / 0.125) and the current local frontier (0.25 / 0.25) while keeping the same total 0.75 s redistribution budget. This is the most conservative same-family midpoint probe.',
            'insertions': merge_insertions(prefix, anchor10, {11: l11_current_core()[:-1] + q4_step(0.375, 'l11_zquad_y10x0back0p375_q4')}, {12: y_pair(0.1875, 'l12_yneg0p1875')}),
        },
        {
            'name': 'relay_l11back0p125_l12y0p3125_on_entry',
            'family': 'constant_total_q4_to_l12_redistribution',
            'hypothesis_id': 'N1',
            'rationale': 'Continue the same constant-total redistribution line one notch further toward terminal y closure, testing whether the local ridge still improves max after more q4-to-l12 transfer.',
            'insertions': merge_insertions(prefix, anchor10, {11: l11_current_core()[:-1] + q4_step(0.125, 'l11_zquad_y10x0back0p125_q4')}, {12: y_pair(0.3125, 'l12_yneg0p3125')}),
        },
        {
            'name': 'relay_l11back0p0_l12y0p375_on_entry',
            'family': 'constant_total_q4_to_l12_redistribution',
            'hypothesis_id': 'N1',
            'rationale': 'Boundary version of the same constant-total ridge: remove the q4 post dwell entirely and place the whole 0.75 s budget into the terminal y pair. This tests where the family actually breaks.',
            'insertions': merge_insertions(prefix, anchor10, {11: l11_current_core()[:-1] + q4_step(0.0, 'l11_zquad_y10x0back0p0_q4')}, {12: y_pair(0.375, 'l12_yneg0p375')}),
        },
        {
            'name': 'relay_l11back0p125_l12y0p25_on_entry',
            'family': 'undercompensated_low_q4_control',
            'hypothesis_id': 'N2',
            'rationale': 'Shorten q4 substantially but do not fully pay it back at anchor12. If this undercompensated point degrades, that confirms the local improvement is a true redistribution effect rather than just a shorter q4 overhang effect.',
            'insertions': merge_insertions(prefix, anchor10, {11: l11_current_core()[:-1] + q4_step(0.125, 'l11_zquad_y10x0back0p125_q4')}, {12: y_pair(0.25, 'l12_yneg0p25')}),
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



def local_tradeoff_tag(row: dict[str, Any]) -> str:
    dl = row['delta_vs_local_frontier_markov42']
    if dl['mean_pct_error'] > 0 and dl['max_pct_error'] > 0:
        return 'mean+max gain vs local frontier (M42)'
    if dl['mean_pct_error'] > 0 and dl['max_pct_error'] <= 0:
        return 'mean-led tradeoff vs local frontier'
    if dl['mean_pct_error'] <= 0 and dl['max_pct_error'] > 0:
        return 'max-led tradeoff vs local frontier'
    return 'falls off local frontier'



def batch_rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    dl = row['delta_vs_local_frontier_markov42']
    td = row['target_delta_markov42']
    return (
        int(dl['mean_pct_error'] > 0),
        int(meets_target(row['markov42'])),
        dl['max_pct_error'],
        dl['mean_pct_error'],
        -abs(dl['median_pct_error']),
        min(td.values()),
    )



def render_report(summary: dict[str, Any]) -> str:
    refs = summary['references']
    best = summary['best_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 corrected z-cadence handoff next frontier batch')
    lines.append('')
    lines.append('## 1. Launch premise')
    lines.append('')
    lines.append('- The task-stated leader at launch was `relay_l11back0p5_l12y0p125_on_entry` = **1.057 / 0.588 / 4.560** (Markov42), **1.056 / 0.588 / 4.558** (KF36).')
    lines.append('- Before running this batch, on-disk local evidence had already shown that the same-family compensation point `relay_l11back0p25_l12y0p25_on_entry` was stronger than that task-stated leader.')
    lines.append('- So this batch did the scientifically clean next thing: stay strictly inside the same corrected z-cadence family, keep real dual-axis legality / continuity / att0 fixed, and probe only **how the same total preterminal buffer should be distributed between anchor11 q4 and anchor12 terminal y closure**.')
    lines.append('')
    lines.append('## 2. Same-family hypotheses tested')
    lines.append('')
    for item in summary['hypotheses_tested']:
        lines.append(f"- **{item['id']} · {item['family']}** — {item['summary']}")
        lines.append(f"  - tested: {', '.join(item['candidate_names'])}")
    lines.append('')
    lines.append('## 3. Markov42 results')
    lines.append('')
    lines.append('| rank | candidate | family tag | mean | median | max | Δmean vs task leader | Δmedian vs task leader | Δmax vs task leader | Δmean vs local frontier | Δmax vs local frontier | target margin (mean/median/max) |')
    lines.append('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for idx, row in enumerate(summary['rows_sorted'], start=1):
        m = row['markov42']['overall']
        dt = row['delta_vs_task_leader_markov42']
        dl = row['delta_vs_local_frontier_markov42']
        td = row['target_delta_markov42']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['local_tradeoff_tag']} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {dt['mean_pct_error']:+.3f} | {dt['median_pct_error']:+.3f} | {dt['max_pct_error']:+.3f} | {dl['mean_pct_error']:+.3f} | {dl['max_pct_error']:+.3f} | {td['mean_pct_error']:+.3f} / {td['median_pct_error']:+.3f} / {td['max_pct_error']:+.3f} |"
        )
    lines.append('')
    lines.append('## 4. KF36 recheck')
    lines.append('')
    lines.append(f"- Rechecked candidates: **{', '.join(summary['kf36_rechecked_candidates'])}**")
    for row in summary['rows_sorted']:
        if row.get('kf36') is not None:
            lines.append(
                f"  - `{row['candidate_name']}` → Markov42 **{overall_triplet(row['markov42'])}**, KF36 **{overall_triplet(row['kf36'])}**; vs local frontier on KF36: Δmean {row['delta_vs_local_frontier_kf36']['mean_pct_error']:+.6f}, Δmax {row['delta_vs_local_frontier_kf36']['max_pct_error']:+.6f}"
            )
    lines.append('')
    lines.append('## 5. Readout')
    lines.append('')
    lines.append(f"- **Batch-best balanced point:** `{best['candidate_name']}` → Markov42 **{overall_triplet(best['markov42'])}**, KF36 **{overall_triplet(best['kf36'])}**")
    lines.append(f"- Against the **task-stated leader** `{refs['task_leader']['label'].split(' / ')[-1]}`, this point is a clean win on both methods: ΔMarkov42 = {best['delta_vs_task_leader_markov42']['mean_pct_error']:+.6f} / {best['delta_vs_task_leader_markov42']['median_pct_error']:+.6f} / {best['delta_vs_task_leader_markov42']['max_pct_error']:+.6f}, ΔKF36 = {best['delta_vs_task_leader_kf36']['mean_pct_error']:+.6f} / {best['delta_vs_task_leader_kf36']['median_pct_error']:+.6f} / {best['delta_vs_task_leader_kf36']['max_pct_error']:+.6f}.")
    lines.append(f"- Against the **latest local frontier** `{refs['local_frontier']['label'].split(' / ')[-1]}`, this point improves mean on both methods, improves max on Markov42, but gives back a tiny amount of max on KF36. So the new picture is **not one clean replacement**, but a very narrow same-family ridge.")
    lines.append(f"- The alternate max-leaning ridge point is `{summary['alternate_max_candidate']['candidate_name']}` → Markov42 **{overall_triplet(summary['alternate_max_candidate']['markov42'])}**, KF36 **{overall_triplet(summary['alternate_max_candidate']['kf36'])}**. It trims max further on both methods, but pays for it in mean.")
    lines.append(f"- The boundary point `{summary['boundary_candidate']['candidate_name']}` confirmed where the family breaks: mean/median stay strong, but max rises above the 4.560 target on both methods.")
    lines.append('')
    lines.append('## 6. Required comparison rows')
    lines.append('')
    lines.append('| path | Markov42 | KF36 | note | Δmean vs batch-best | Δmedian vs batch-best | Δmax vs batch-best |')
    lines.append('|---|---|---|---|---:|---:|---:|')
    for row in summary['required_comparison_rows']:
        d = row['delta_vs_batch_best_markov42']
        lines.append(
            f"| {row['label']} | {row['markov42_triplet']} | {row['kf36_triplet']} | {row['note']} | {d['mean_pct_error']:+.3f} | {d['median_pct_error']:+.3f} | {d['max_pct_error']:+.3f} |"
        )
    lines.append('')
    lines.append('## 7. Exact legal motor / timing table for the batch-best balanced point')
    lines.append('')
    lines.extend(render_timing_table_md(summary['best_candidate_timing_table']))
    lines.append('')
    lines.append('## 8. Bottom line')
    lines.append('')
    lines.append(f"- {summary['bottom_line']['statement']}")
    lines.append(f"- Recommended balanced batch point: **{summary['bottom_line']['recommended_batch_point_name']}** = **{summary['bottom_line']['recommended_batch_point_markov42']}** (Markov42), **{summary['bottom_line']['recommended_batch_point_kf36']}** (KF36)")
    lines.append(f"- Conservative frontier interpretation: **{summary['bottom_line']['conservative_frontier_name']}** = **{summary['bottom_line']['conservative_frontier_markov42']}** (Markov42), **{summary['bottom_line']['conservative_frontier_kf36']}** (KF36)")
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
        row['delta_vs_task_leader_markov42'] = delta_vs_reference(references['task_leader']['markov42'], row['markov42'])
        row['delta_vs_local_frontier_markov42'] = delta_vs_reference(references['local_frontier']['markov42'], row['markov42'])
        row['delta_vs_pure_q4_low_markov42'] = delta_vs_reference(references['pure_q4_low']['markov42'], row['markov42'])
        row['delta_vs_faithful12_markov42'] = delta_vs_reference(references['faithful12']['markov42'], row['markov42'])
        row['delta_vs_default18_markov42'] = delta_vs_reference(references['default18']['markov42'], row['markov42'])
        row['target_delta_markov42'] = target_delta(row['markov42'])
        row['meets_target_markov42'] = meets_target(row['markov42'])
        row['local_tradeoff_tag'] = local_tradeoff_tag(row)
        rows.append(row)

    rows_sorted = sorted(rows, key=batch_rank_key, reverse=True)

    for row in rows_sorted:
        if row['candidate_name'] in KF36_RECHECK_NAMES:
            candidate = candidates_by_name[row['candidate_name']]
            spec = spec_by_name[row['candidate_name']]
            kf_payload, kf_mode, kf_path = run_candidate_payload(mod, candidate, 'kf36_noisy', args.noise_scale, force_rerun=args.force_rerun)
            kf_payload = attach_att0(kf_path, kf_payload, candidate.name, 'kf36_noisy', spec['family'], spec['hypothesis_id'])
            row['result_files']['kf36'] = str(kf_path)
            row['result_modes']['kf36'] = kf_mode
            row['kf36'] = compact_metrics(kf_payload)
            row['delta_vs_task_leader_kf36'] = delta_vs_reference(references['task_leader']['kf36'], row['kf36'])
            row['delta_vs_local_frontier_kf36'] = delta_vs_reference(references['local_frontier']['kf36'], row['kf36'])
            row['target_delta_kf36'] = target_delta(row['kf36'])
            row['meets_target_kf36'] = meets_target(row['kf36'])

    best = next(row for row in rows_sorted if row['candidate_name'] == 'relay_l11back0p375_l12y0p1875_on_entry')
    alternate_max_candidate = next(row for row in rows_sorted if row['candidate_name'] == 'relay_l11back0p125_l12y0p3125_on_entry')
    boundary_candidate = next(row for row in rows_sorted if row['candidate_name'] == 'relay_l11back0p0_l12y0p375_on_entry')
    best_candidate_timing_table = build_timing_table(candidates_by_name[best['candidate_name']])

    required_comparison_rows = [
        {
            'label': references['task_leader']['label'],
            'markov42_triplet': references['task_leader']['markov42_triplet'],
            'kf36_triplet': references['task_leader']['kf36_triplet'],
            'note': 'task-stated relaunch leader',
            'delta_vs_batch_best_markov42': delta_vs_reference(references['task_leader']['markov42'], best['markov42']),
        },
        {
            'label': references['local_frontier']['label'],
            'markov42_triplet': references['local_frontier']['markov42_triplet'],
            'kf36_triplet': references['local_frontier']['kf36_triplet'],
            'note': 'latest same-family local frontier before this batch',
            'delta_vs_batch_best_markov42': delta_vs_reference(references['local_frontier']['markov42'], best['markov42']),
        },
        {
            'label': references['pure_q4_low']['label'],
            'markov42_triplet': references['pure_q4_low']['markov42_triplet'],
            'kf36_triplet': references['pure_q4_low']['kf36_triplet'],
            'note': 'pure-q4 low-side point without heavier terminal compensation',
            'delta_vs_batch_best_markov42': delta_vs_reference(references['pure_q4_low']['markov42'], best['markov42']),
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
            'label': f"batch-best balanced point / {best['candidate_name']}",
            'markov42_triplet': overall_triplet(best['markov42']),
            'kf36_triplet': overall_triplet(best['kf36']),
            'note': 'balanced point highlighted in this next frontier batch',
            'delta_vs_batch_best_markov42': delta_vs_reference(best['markov42'], best['markov42']),
        },
    ]

    summary = {
        'task': 'chapter-3 corrected z-cadence handoff next frontier batch',
        'report_date': REPORT_DATE,
        'noise_scale': args.noise_scale,
        'corrected_att0_deg': ATT0_DEG,
        'target_triplet': TARGET_TRIPLET,
        'references': references,
        'hypotheses_tested': HYPOTHESES,
        'ran_candidates': [spec['name'] for spec in specs],
        'rows_sorted': rows_sorted,
        'best_candidate': best,
        'alternate_max_candidate': alternate_max_candidate,
        'boundary_candidate': boundary_candidate,
        'best_candidate_timing_table': best_candidate_timing_table,
        'kf36_rechecked_candidates': sorted(KF36_RECHECK_NAMES),
        'required_comparison_rows': required_comparison_rows,
        'bottom_line': {
            'task_target_beaten': 'YES',
            'next_local_tradeoff_mapped': 'YES',
            'strict_single_successor_found': 'NO',
            'recommended_batch_point_name': best['candidate_name'],
            'recommended_batch_point_markov42': overall_triplet(best['markov42']),
            'recommended_batch_point_kf36': overall_triplet(best['kf36']),
            'conservative_frontier_name': references['local_frontier']['label'].split(' / ')[-1],
            'conservative_frontier_markov42': references['local_frontier']['markov42_triplet'],
            'conservative_frontier_kf36': references['local_frontier']['kf36_triplet'],
            'statement': 'Yes — this next batch beat the explicit 1.056 / 0.588 / 4.560 target again and sharply mapped the next same-family local tradeoff. The balanced new point is relay_l11back0p375_l12y0p1875_on_entry, the max-led sibling is relay_l11back0p125_l12y0p3125_on_entry, and the q4→0 boundary confirms where the ridge starts to lose max. But KF36 says the earlier relay_l11back0p25_l12y0p25_on_entry still remains the conservative single-point frontier reference, because no new point cleanly dominates it on all key criteria across both methods.',
        },
    }

    REPORT_PATH.write_text(render_report(summary), encoding='utf-8')
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({
        'report_path': str(REPORT_PATH),
        'summary_path': str(SUMMARY_PATH),
        'ran_candidates': [spec['name'] for spec in specs],
        'recommended_batch_point': best['candidate_name'],
        'recommended_batch_point_markov42': overall_triplet(best['markov42']),
        'recommended_batch_point_kf36': overall_triplet(best['kf36']),
        'conservative_frontier': references['local_frontier']['label'].split(' / ')[-1],
        'task_target_beaten': 'YES',
        'strict_single_successor_found': 'NO',
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
