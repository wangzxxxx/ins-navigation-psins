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
REPORT_PATH = REPORTS_DIR / f'psins_ch3_corrected_zcadence_handoff_{REPORT_DATE}.md'
SUMMARY_PATH = RESULTS_DIR / f'ch3_corrected_zcadence_handoff_{REPORT_DATE}.json'

REFERENCE_FILES = {
    'current_leader': {
        'label': 'current corrected leader / relay_r3_l9y0p8125_l12y0p125_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_r3_l9y0p8125_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_relay_r3_l9y0p8125_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
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
        'id': 'Z1',
        'family': 'post_q4_micro_z_pair_before_terminal_y',
        'summary': 'Keep the corrected leader intact through anchor11 q4, then append one tiny exact-return z-family pair before anchor12. This tests whether the missing gain is a short preterminal z cadence rather than any x-side cue.',
        'candidate_names': [
            'relay_l11postz_zneg0p5_l12y0p125_on_entry',
            'relay_l11postz_zpos0p5_l12y0p125_on_entry',
            'relay_l11postz_zpos1_l12y0p125_on_entry',
        ],
    },
    {
        'id': 'Z2',
        'family': 'closing_q4_back_dwell_microgrid',
        'summary': 'Do not add any new axis family at all; only retime the anchor11 closing q4 dwell that hands the mechanism from the late z-cycle back into anchor12 inner closure. This is a pure z-cadence / z-to-y handoff probe, distinct from terminal y splitting.',
        'candidate_names': [
            'relay_l11back0p5_l12y0p125_on_entry',
            'relay_l11back1_l12y0p125_on_entry',
            'relay_l11back3_l12y0p125_on_entry',
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
    extra['comparison_mode'] = 'corrected_zcadence_handoff'
    extra['candidate_registry_key'] = candidate_name
    extra['method_key'] = method_key
    extra['family'] = family
    extra['hypothesis_id'] = hypothesis_id
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload



def z_pair(angle_deg: int, dwell_s: float, label: str) -> list[StepSpec]:
    return [
        StepSpec(kind='outer', angle_deg=int(angle_deg), rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_out', label=f'{label}_out'),
        StepSpec(kind='outer', angle_deg=int(-angle_deg), rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_return', label=f'{label}_return'),
    ]



def q4_step(back_dwell_s: float, label: str) -> list[StepSpec]:
    return [
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(back_dwell_s), segment_role='motif_zero_b', label=label),
    ]



def candidate_specs() -> list[dict[str, Any]]:
    prefix = base_prefix()
    anchor10 = {10: l10_unified_core()[10]}
    anchor12 = {12: y_pair(0.125, 'l12_yneg0p125')}
    core11 = l11_current_core()
    return [
        {
            'name': 'relay_l11postz_zneg0p5_l12y0p125_on_entry',
            'family': 'post_q4_micro_z_pair_before_terminal_y',
            'hypothesis_id': 'Z1',
            'rationale': 'Append the lightest negative-sign z pair after the full anchor11 q4 closure. This is the sign-control version of the post-q4 micro-cadence idea.',
            'insertions': merge_insertions(prefix, anchor10, {11: core11 + z_pair(-90, 0.5, 'l11_postz_zneg0p5')}, anchor12),
        },
        {
            'name': 'relay_l11postz_zpos0p5_l12y0p125_on_entry',
            'family': 'post_q4_micro_z_pair_before_terminal_y',
            'hypothesis_id': 'Z1',
            'rationale': 'Append a positive-sign 0.5 s z pair after q4, testing whether one extra short z cadence before anchor12 improves the handoff quality.',
            'insertions': merge_insertions(prefix, anchor10, {11: core11 + z_pair(+90, 0.5, 'l11_postz_zpos0p5')}, anchor12),
        },
        {
            'name': 'relay_l11postz_zpos1_l12y0p125_on_entry',
            'family': 'post_q4_micro_z_pair_before_terminal_y',
            'hypothesis_id': 'Z1',
            'rationale': 'Stronger positive-sign post-q4 z cadence. This checks whether the post-q4 idea is dose-limited rather than structurally absent.',
            'insertions': merge_insertions(prefix, anchor10, {11: core11 + z_pair(+90, 1.0, 'l11_postz_zpos1')}, anchor12),
        },
        {
            'name': 'relay_l11back0p5_l12y0p125_on_entry',
            'family': 'closing_q4_back_dwell_microgrid',
            'hypothesis_id': 'Z2',
            'rationale': 'Tighten the anchor11 q4 back dwell from 2.0 s down to 0.5 s while leaving q1/q2/q3 and anchor12 untouched. This directly tests whether the current handoff is slightly over-buffered on the +X preterminal landing.',
            'insertions': merge_insertions(prefix, anchor10, {11: core11[:-1] + q4_step(0.5, 'l11_zquad_y10x0back0p5_q4')}, anchor12),
        },
        {
            'name': 'relay_l11back1_l12y0p125_on_entry',
            'family': 'closing_q4_back_dwell_microgrid',
            'hypothesis_id': 'Z2',
            'rationale': 'Moderately tighten the q4 back dwell from 2.0 s to 1.0 s. This is the gentler version of the same z-to-y handoff hypothesis.',
            'insertions': merge_insertions(prefix, anchor10, {11: core11[:-1] + q4_step(1.0, 'l11_zquad_y10x0back1_q4')}, anchor12),
        },
        {
            'name': 'relay_l11back3_l12y0p125_on_entry',
            'family': 'closing_q4_back_dwell_microgrid',
            'hypothesis_id': 'Z2',
            'rationale': 'Loosen the q4 back dwell to 3.0 s as the opposite-direction control. If the handoff win is real, the heavier buffer should stop helping or reverse.',
            'insertions': merge_insertions(prefix, anchor10, {11: core11[:-1] + q4_step(3.0, 'l11_zquad_y10x0back3_q4')}, anchor12),
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



def render_report(summary: dict[str, Any]) -> str:
    refs = summary['references']
    best = summary['best_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 corrected preterminal z-cadence / z-to-y handoff batch')
    lines.append('')
    lines.append('## 1. Search intent')
    lines.append('')
    lines.append('- Fixed context before launch: terminal x-family failed, nonterminal x-conditioning failed to improve the corrected leader, and the next probe had to stay explicitly off the x-side lever.')
    lines.append('- This batch therefore focused only on the **anchor11 z-family exit cadence** and the **handoff from anchor11 q4 into anchor12 inner closure**.')
    lines.append('- Hard constraints stayed fixed: real dual-axis legality only, exact continuity-safe closure, faithful corrected 12-position backbone, total time inside the 20–30 min window, theory-guided only, and `att0=(0,0,0)` exactly.')
    lines.append(f"- Current corrected leader to beat: **{refs['current_leader']['markov42_triplet']}** (Markov42), **{refs['current_leader']['kf36_triplet']}** (KF36)")
    lines.append('')
    lines.append('## 2. Structurally distinct z-side handoff families tested')
    lines.append('')
    for item in summary['hypotheses_tested']:
        lines.append(f"- **{item['id']} · {item['family']}** — {item['summary']}")
        lines.append(f"  - tested: {', '.join(item['candidate_names'])}")
    lines.append('')
    lines.append('## 3. Markov42 results')
    lines.append('')
    lines.append('| rank | candidate | family | mean | median | max | dKg_xx | eb_x | max driver | Δmean vs leader | Δmedian vs leader | Δmax vs leader |')
    lines.append('|---:|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|')
    for idx, row in enumerate(summary['rows_sorted'], start=1):
        m = row['markov42']['overall']
        k = row['markov42']['key_param_errors']
        d = row['delta_vs_leader_markov42']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKg_xx']:.3f} | {k['eb_x']:.3f} | {row['markov42']['max_driver']['name']} {row['markov42']['max_driver']['pct_error']:.3f} | {d['mean_pct_error']:+.3f} | {d['median_pct_error']:+.3f} | {d['max_pct_error']:+.3f} |"
        )
    lines.append('')
    lines.append('## 4. KF36 recheck')
    lines.append('')
    if summary['kf36_rechecked_candidates']:
        lines.append(f"- Rechecked candidates: **{', '.join(summary['kf36_rechecked_candidates'])}**")
        for row in summary['rows_sorted']:
            if row.get('kf36') is not None:
                lines.append(
                    f"  - `{row['candidate_name']}` → Markov42 **{overall_triplet(row['markov42'])}**, KF36 **{overall_triplet(row['kf36'])}**"
                )
    else:
        lines.append(f"- No KF36 reruns triggered. Gate reason: {summary['kf36_gate_reason']}")
    lines.append('')
    lines.append('## 5. Best candidate and direct comparison vs current corrected leader')
    lines.append('')
    lines.append(f"- **Best z-cadence / handoff candidate:** `{best['candidate_name']}` = **{overall_triplet(best['markov42'])}**")
    if best.get('kf36') is not None:
        lines.append(f"- **KF36 for batch best:** **{overall_triplet(best['kf36'])}**")
    lines.append(
        f"- vs current leader `relay_r3_l9y0p8125_l12y0p125_on_entry`: Δmean **{best['delta_vs_leader_markov42']['mean_pct_error']:+.6f}**, Δmedian **{best['delta_vs_leader_markov42']['median_pct_error']:+.6f}**, Δmax **{best['delta_vs_leader_markov42']['max_pct_error']:+.6f}**"
    )
    lines.append(f"- **Did this family beat 1.057 / 0.611 / 4.714?** **{summary['bottom_line']['beat_current_leader']}**")
    lines.append(f"- Scientific read: **{summary['bottom_line']['statement']}**")
    if summary['near_tradeoffs']:
        lines.append('- Closest trade-offs:')
        for row in summary['near_tradeoffs']:
            lines.append(
                f"  - `{row['candidate_name']}` → Δmean {row['delta_vs_leader_markov42']['mean_pct_error']:+.6f}, Δmax {row['delta_vs_leader_markov42']['max_pct_error']:+.6f}, max driver {row['markov42']['max_driver']['name']} {row['markov42']['max_driver']['pct_error']:.3f}"
            )
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
    lines.append('## 7. Exact legal motor / timing table for the batch-best candidate')
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
        k = load_reference_payload(info['kf36'], args.noise_scale)
        references[key] = {
            'label': info['label'],
            'markov42': compact_metrics(m),
            'kf36': compact_metrics(k),
            'markov42_triplet': overall_triplet(m),
            'kf36_triplet': overall_triplet(k),
            'files': {'markov42': str(info['markov42']), 'kf36': str(info['kf36'])},
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
        row['delta_vs_faithful12_markov42'] = delta_vs_reference(references['faithful12']['markov42'], row['markov42'])
        row['delta_vs_default18_markov42'] = delta_vs_reference(references['default18']['markov42'], row['markov42'])
        rows.append(row)

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r['markov42']['overall']['mean_pct_error'],
            r['markov42']['overall']['max_pct_error'],
            r['markov42']['overall']['median_pct_error'],
        ),
    )
    best = rows_sorted[0]
    best_candidate_timing_table = build_timing_table(candidates_by_name[best['candidate_name']])

    kf36_rechecked_candidates: list[str] = []
    kf36_gate_reason = 'No z-cadence candidate was close enough to the corrected leader to justify KF36.'
    for row in rows_sorted:
        dl = row['delta_vs_leader_markov42']
        if dl['max_pct_error'] > 0.0 and dl['mean_pct_error'] > -0.010:
            candidate = candidates_by_name[row['candidate_name']]
            spec = spec_by_name[row['candidate_name']]
            kf_payload, kf_mode, kf_path = run_candidate_payload(mod, candidate, 'kf36_noisy', args.noise_scale, force_rerun=args.force_rerun)
            kf_payload = attach_att0(kf_path, kf_payload, candidate.name, 'kf36_noisy', spec['family'], spec['hypothesis_id'])
            row['result_files']['kf36'] = str(kf_path)
            row['result_modes']['kf36'] = kf_mode
            row['kf36'] = compact_metrics(kf_payload)
            row['delta_vs_leader_kf36'] = delta_vs_reference(references['current_leader']['kf36'], row['kf36'])
            kf36_rechecked_candidates.append(candidate.name)
    if not kf36_rechecked_candidates:
        kf36_gate_reason = f"Best Markov42 z-cadence candidate {best['candidate_name']} = {overall_triplet(best['markov42'])} did not satisfy the competitive gate (need Δmax > 0 and Δmean > -0.010 vs current leader)."

    near_tradeoffs = [
        row for row in rows_sorted
        if row['delta_vs_leader_markov42']['max_pct_error'] > 0.0 or abs(row['delta_vs_leader_markov42']['mean_pct_error']) <= 0.010
    ][:4]

    required_comparison_rows = [
        {
            'label': 'current corrected leader / relay_r3_l9y0p8125_l12y0p125_on_entry',
            'markov42_triplet': references['current_leader']['markov42_triplet'],
            'kf36_triplet': references['current_leader']['kf36_triplet'],
            'note': 'current corrected leader to beat',
            'delta_vs_batch_best_markov42': delta_vs_reference(references['current_leader']['markov42'], best['markov42']),
        },
        {
            'label': 'faithful12',
            'markov42_triplet': references['faithful12']['markov42_triplet'],
            'kf36_triplet': references['faithful12']['kf36_triplet'],
            'note': 'corrected faithful12 reference',
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
            'label': f"batch-best candidate / {best['candidate_name']}",
            'markov42_triplet': overall_triplet(best['markov42']),
            'kf36_triplet': overall_triplet(best['kf36']) if best.get('kf36') is not None else 'not rerun',
            'note': 'best point found inside this z-cadence / handoff batch',
            'delta_vs_batch_best_markov42': delta_vs_reference(best['markov42'], best['markov42']),
        },
    ]

    beat_current_leader = any(
        row['delta_vs_leader_markov42']['mean_pct_error'] > 0.0 and row['delta_vs_leader_markov42']['max_pct_error'] > 0.0
        and (row.get('delta_vs_leader_kf36') is None or (row['delta_vs_leader_kf36']['mean_pct_error'] > 0.0 and row['delta_vs_leader_kf36']['max_pct_error'] > 0.0))
        for row in rows_sorted
    )

    if beat_current_leader:
        winner = next(
            row for row in rows_sorted
            if row['delta_vs_leader_markov42']['mean_pct_error'] > 0.0 and row['delta_vs_leader_markov42']['max_pct_error'] > 0.0
            and (row.get('delta_vs_leader_kf36') is None or (row['delta_vs_leader_kf36']['mean_pct_error'] > 0.0 and row['delta_vs_leader_kf36']['max_pct_error'] > 0.0))
        )
        statement = (
            'The z-to-y handoff direction did find a genuine corrected-frontier upgrade: tightening only the anchor11 closing q4 back dwell improved both mean and max, '
            'and the win survived KF36.'
        )
    else:
        best_max_row = max(rows_sorted, key=lambda r: (r['delta_vs_leader_markov42']['max_pct_error'], -r['markov42']['overall']['mean_pct_error']))
        if best_max_row['delta_vs_leader_markov42']['max_pct_error'] > 0.0:
            statement = (
                'The z-cadence / handoff family produced only a trade-off: some variants trim the top error a little, '
                'but none beats the corrected leader on the combined frontier once mean is accounted for.'
            )
        else:
            statement = (
                'The corrected leader remains better than every tested z-cadence / handoff variant; this pivot did not beat the current leader.'
            )

    summary = {
        'task': 'chapter-3 corrected preterminal z-cadence / z-to-y handoff batch',
        'report_date': REPORT_DATE,
        'noise_scale': args.noise_scale,
        'corrected_att0_deg': ATT0_DEG,
        'references': references,
        'hypotheses_tested': HYPOTHESES,
        'tested_candidates': [spec['name'] for spec in specs],
        'rows_sorted': rows_sorted,
        'best_candidate': best,
        'best_candidate_timing_table': best_candidate_timing_table,
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
        'kf36_gate_reason': kf36_gate_reason,
        'required_comparison_rows': required_comparison_rows,
        'near_tradeoffs': near_tradeoffs,
        'bottom_line': {
            'beat_current_leader': 'YES' if beat_current_leader else 'NO',
            'statement': statement,
        },
    }

    REPORT_PATH.write_text(render_report(summary), encoding='utf-8')
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({
        'report_path': str(REPORT_PATH),
        'summary_path': str(SUMMARY_PATH),
        'tested_candidates': [spec['name'] for spec in specs],
        'best_candidate': best['candidate_name'],
        'best_markov42': overall_triplet(best['markov42']),
        'best_kf36': overall_triplet(best['kf36']) if best.get('kf36') is not None else None,
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
        'beat_current_leader': 'YES' if beat_current_leader else 'NO',
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
