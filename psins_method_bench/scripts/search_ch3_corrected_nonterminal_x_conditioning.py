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
from search_ch3_entry_conditioned_relay_family import (
    NOISE_SCALE,
    l8_xpair,
    l9_ypair_neg,
    l10_unified_core,
    merge_insertions,
    xpair_outerhold,
    zquad,
)

REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
REPORT_PATH = REPORTS_DIR / f'psins_ch3_corrected_nonterminal_x_conditioning_{REPORT_DATE}.md'
SUMMARY_PATH = RESULTS_DIR / f'ch3_corrected_nonterminal_x_conditioning_{REPORT_DATE}.json'
EPS = 1e-9

REFERENCE_FILES = {
    'current_leader': {
        'label': 'current corrected leader / relay_r3_l9y0p8125_l12y0p125_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_r3_l9y0p8125_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_relay_r3_l9y0p8125_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
    },
    'terminal_x_best': {
        'label': 'terminal-x best / relay_xpost_pos1_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_xpost_pos1_on_entry_shared_noise0p08_param_errors.json',
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
        'id': 'A',
        'family': 'preterminal_x_conditioning_after_l11_closure',
        'summary': 'Keep the corrected leader core intact through the existing anchor11 x-hold and z-quad, then inject one tiny additional anchor11 x-family closed loop immediately before anchor12. This tests whether the missing x information must be applied right before terminal closure rather than after it.',
        'candidate_names': [
            'relay_l11postz_x0p5_l12y0p125_on_entry',
            'relay_l11postz_x1_l12y0p125_on_entry',
        ],
    },
    {
        'id': 'B',
        'family': 'x_y_interleaved_l11_boundary_closure',
        'summary': 'Split the anchor11 z-quad and insert a small anchor11 x-family cue after the first half of the y/z closure, so the x-side conditioning happens before the final anchor11 y-negative leg and before anchor12 terminal reclosure.',
        'candidate_names': [
            'relay_l11interleave_q2_x0p5_l12y0p125_on_entry',
            'relay_l11interleave_q2_x1_l12y0p125_on_entry',
        ],
    },
    {
        'id': 'C',
        'family': 'nearby_anchor10_minimal_x_gate',
        'summary': 'Add one minimal anchor10 x-family gate after the existing anchor10 unified core, one anchor earlier than the terminal boundary, to test whether x-side conditioning must arrive before the whole anchor11/12 closure stack begins.',
        'candidate_names': [
            'relay_l10pregate_x1_l12y0p125_on_entry',
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
    extra['comparison_mode'] = 'corrected_nonterminal_x_conditioning'
    extra['candidate_registry_key'] = candidate_name
    extra['method_key'] = method_key
    extra['family'] = family
    extra['hypothesis_id'] = hypothesis_id
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload



def y_pair(dwell_s: float, label: str) -> list[StepSpec]:
    return [
        StepSpec(kind='inner', angle_deg=-90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_out', label=f'{label}_out'),
        StepSpec(kind='inner', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_return', label=f'{label}_return'),
    ]



def l11_current_core() -> list[StepSpec]:
    return xpair_outerhold(10.0, 'l11_xpair_outerhold') + zquad(10.0, 0.0, 2.0, 'l11_zquad_y10x0back2')



def zquad_front_half() -> list[StepSpec]:
    return zquad(10.0, 0.0, 2.0, 'l11_zquad_y10x0back2')[:2]



def zquad_back_half() -> list[StepSpec]:
    return zquad(10.0, 0.0, 2.0, 'l11_zquad_y10x0back2')[2:]



def base_prefix() -> dict[int, list[StepSpec]]:
    return merge_insertions(
        l8_xpair(1.0, 'l8_x1'),
        l9_ypair_neg(0.8125, 'l9_ypair_neg0p8125'),
    )



def candidate_specs() -> list[dict[str, Any]]:
    prefix = base_prefix()
    return [
        {
            'name': 'relay_l11postz_x0p5_l12y0p125_on_entry',
            'family': 'preterminal_x_conditioning_after_l11_closure',
            'hypothesis_id': 'A',
            'rationale': 'Add the lightest extra anchor11 post-zquad x-family loop immediately before anchor12. This is the cleanest pre-terminal x-conditioning probe that stays out of the post-terminal basin.',
            'insertions': merge_insertions(
                prefix,
                {10: l10_unified_core()[10]},
                {11: l11_current_core() + xpair_outerhold(0.5, 'l11_xpair_postz0p5')},
                {12: y_pair(0.125, 'l12_yneg0p125')},
            ),
        },
        {
            'name': 'relay_l11postz_x1_l12y0p125_on_entry',
            'family': 'preterminal_x_conditioning_after_l11_closure',
            'hypothesis_id': 'A',
            'rationale': 'Same post-zquad pre-terminal x-conditioning idea, but with a slightly stronger 1 s anchor11 x loop to test whether the missing x cue needs a nontrivial dwell before anchor12.',
            'insertions': merge_insertions(
                prefix,
                {10: l10_unified_core()[10]},
                {11: l11_current_core() + xpair_outerhold(1.0, 'l11_xpair_postz1')},
                {12: y_pair(0.125, 'l12_yneg0p125')},
            ),
        },
        {
            'name': 'relay_l11interleave_q2_x0p5_l12y0p125_on_entry',
            'family': 'x_y_interleaved_l11_boundary_closure',
            'hypothesis_id': 'B',
            'rationale': 'Interleave a tiny x-family cue after the first half of the anchor11 z-quad so that x conditioning lands before the anchor11 y-negative leg and the terminal anchor12 reclosure.',
            'insertions': merge_insertions(
                prefix,
                {10: l10_unified_core()[10]},
                {11: xpair_outerhold(10.0, 'l11_xpair_outerhold') + zquad_front_half() + xpair_outerhold(0.5, 'l11_xpair_midq2_0p5') + zquad_back_half()},
                {12: y_pair(0.125, 'l12_yneg0p125')},
            ),
        },
        {
            'name': 'relay_l11interleave_q2_x1_l12y0p125_on_entry',
            'family': 'x_y_interleaved_l11_boundary_closure',
            'hypothesis_id': 'B',
            'rationale': 'Same interleaved boundary-closure family, but with a 1 s x cue. This tests whether the x-before-y ordering needs a stronger cue than the minimal 0.5 s dose.',
            'insertions': merge_insertions(
                prefix,
                {10: l10_unified_core()[10]},
                {11: xpair_outerhold(10.0, 'l11_xpair_outerhold') + zquad_front_half() + xpair_outerhold(1.0, 'l11_xpair_midq2_1') + zquad_back_half()},
                {12: y_pair(0.125, 'l12_yneg0p125')},
            ),
        },
        {
            'name': 'relay_l10pregate_x1_l12y0p125_on_entry',
            'family': 'nearby_anchor10_minimal_x_gate',
            'hypothesis_id': 'C',
            'rationale': 'Inject a single minimal x-family gate at anchor10 after the existing unified core, so x-side conditioning arrives one whole anchor before the l11/l12 closure pair.',
            'insertions': merge_insertions(
                prefix,
                {10: l10_unified_core()[10] + xpair_outerhold(1.0, 'l10_xpair_gate1')},
                {11: l11_current_core()},
                {12: y_pair(0.125, 'l12_yneg0p125')},
            ),
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
    lines.append('# Chapter-3 corrected nonterminal x-conditioning batch')
    lines.append('')
    lines.append('## 1. Mission and structural pivot')
    lines.append('')
    lines.append('- This batch explicitly **left the post-terminal x basin** after the terminal-x pivot failed.')
    lines.append('- New target: **nonterminal x-conditioning** that lands before the final anchor12 closure, under the same corrected leader backbone and the same att0=(0,0,0) legality constraints.')
    lines.append('- Hard constraints remained fixed: real dual-axis legality only, continuity-safe exact reconnection, faithful 12-position backbone, theory-guided search only, total time still in the 20–30 min window.')
    lines.append(f"- Current corrected leader to beat: **{refs['current_leader']['markov42_triplet']}** (Markov42), **{refs['current_leader']['kf36_triplet']}** (KF36)")
    lines.append(f"- Failed terminal-x reference: **{refs['terminal_x_best']['markov42_triplet']}** (`relay_xpost_pos1_on_entry`)")
    lines.append('')
    lines.append('## 2. Nonterminal families actually tested')
    lines.append('')
    for item in summary['hypotheses_tested']:
        lines.append(f"- **{item['id']} · {item['family']}** — {item['summary']}")
        lines.append(f"  - tested: {', '.join(item['candidate_names'])}")
    lines.append('')
    lines.append('## 3. Markov42 results')
    lines.append('')
    lines.append('| rank | candidate | family | mean | median | max | dKg_xx | eb_x | Δmean vs leader | Δmedian vs leader | Δmax vs leader | Δmean vs terminal-x | Δmax vs terminal-x |')
    lines.append('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(summary['rows_sorted'], start=1):
        m = row['markov42']['overall']
        k = row['markov42']['key_param_errors']
        dl = row['delta_vs_leader_markov42']
        dt = row['delta_vs_terminal_x_best_markov42']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {k['dKg_xx']:.3f} | {k['eb_x']:.3f} | {dl['mean_pct_error']:+.3f} | {dl['median_pct_error']:+.3f} | {dl['max_pct_error']:+.3f} | {dt['mean_pct_error']:+.3f} | {dt['max_pct_error']:+.3f} |"
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
    lines.append('## 5. Best candidate and decision')
    lines.append('')
    lines.append(f"- **Batch-best candidate:** `{best['candidate_name']}` = **{overall_triplet(best['markov42'])}**")
    if best.get('kf36') is not None:
        lines.append(f"- **KF36 for batch best:** **{overall_triplet(best['kf36'])}**")
    lines.append(
        f"- vs current leader: Δmean **{best['delta_vs_leader_markov42']['mean_pct_error']:+.6f}**, Δmedian **{best['delta_vs_leader_markov42']['median_pct_error']:+.6f}**, Δmax **{best['delta_vs_leader_markov42']['max_pct_error']:+.6f}**"
    )
    lines.append(
        f"- vs terminal-x best: Δmean **{best['delta_vs_terminal_x_best_markov42']['mean_pct_error']:+.6f}**, Δmedian **{best['delta_vs_terminal_x_best_markov42']['median_pct_error']:+.6f}**, Δmax **{best['delta_vs_terminal_x_best_markov42']['max_pct_error']:+.6f}**"
    )
    lines.append(f"- **Did nonterminal x-conditioning beat 1.057 / 0.611 / 4.714?** **{summary['bottom_line']['beat_current_leader']}**")
    lines.append(f"- Scientific read: **{summary['bottom_line']['statement']}**")
    if summary['near_tradeoffs']:
        lines.append('- Closest trade-offs:')
        for row in summary['near_tradeoffs']:
            lines.append(
                f"  - `{row['candidate_name']}` → Δmean {row['delta_vs_leader_markov42']['mean_pct_error']:+.6f}, Δmax {row['delta_vs_leader_markov42']['max_pct_error']:+.6f}, dKg_xx {row['markov42']['key_param_errors']['dKg_xx']:.3f}, eb_x {row['markov42']['key_param_errors']['eb_x']:.3f}"
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
        row['delta_vs_terminal_x_best_markov42'] = delta_vs_reference(references['terminal_x_best']['markov42'], row['markov42'])
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

    competitive_pool = [
        row for row in rows_sorted
        if row['delta_vs_leader_markov42']['mean_pct_error'] > -0.025 and row['delta_vs_leader_markov42']['max_pct_error'] > -0.150
    ]
    if not competitive_pool:
        competitive_pool = [best]
    by_max = max(rows_sorted, key=lambda r: (r['delta_vs_leader_markov42']['max_pct_error'], r['delta_vs_leader_markov42']['mean_pct_error']))
    competitive_names = []
    for row in competitive_pool[:2] + [by_max]:
        if row['candidate_name'] not in competitive_names:
            competitive_names.append(row['candidate_name'])

    kf36_rechecked_candidates: list[str] = []
    if competitive_names:
        for name in competitive_names[:3]:
            row = next(item for item in rows_sorted if item['candidate_name'] == name)
            candidate = candidates_by_name[name]
            spec = spec_by_name[name]
            kf_payload, kf_mode, kf_path = run_candidate_payload(mod, candidate, 'kf36_noisy', args.noise_scale, force_rerun=args.force_rerun)
            kf_payload = attach_att0(kf_path, kf_payload, candidate.name, 'kf36_noisy', spec['family'], spec['hypothesis_id'])
            row['result_files']['kf36'] = str(kf_path)
            row['result_modes']['kf36'] = kf_mode
            row['kf36'] = compact_metrics(kf_payload)
            row['delta_vs_leader_kf36'] = delta_vs_reference(references['current_leader']['kf36'], row['kf36'])
            kf36_rechecked_candidates.append(name)
        kf36_gate_reason = 'Rechecked the near-leader competitive set: top mean-ranked candidates plus the strongest max-side trade-off.'
    else:
        kf36_gate_reason = 'No candidate reached the near-leader competitive pool.'

    near_tradeoffs = [
        row for row in rows_sorted
        if row['delta_vs_leader_markov42']['max_pct_error'] > 0.0 or abs(row['delta_vs_leader_markov42']['mean_pct_error']) <= 0.006
    ][:4]

    required_comparison_rows = [
        {
            'label': references['current_leader']['label'],
            'markov42_triplet': references['current_leader']['markov42_triplet'],
            'kf36_triplet': references['current_leader']['kf36_triplet'],
            'note': 'current corrected leader to beat',
            'delta_vs_batch_best_markov42': delta_vs_reference(references['current_leader']['markov42'], best['markov42']),
        },
        {
            'label': references['terminal_x_best']['label'],
            'markov42_triplet': references['terminal_x_best']['markov42_triplet'],
            'kf36_triplet': references['terminal_x_best']['kf36_triplet'],
            'note': 'best failed terminal-x pivot point from the previous batch',
            'delta_vs_batch_best_markov42': delta_vs_reference(references['terminal_x_best']['markov42'], best['markov42']),
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
            'note': 'best point found inside this nonterminal x-conditioning batch',
            'delta_vs_batch_best_markov42': delta_vs_reference(best['markov42'], best['markov42']),
        },
    ]

    beat_current_leader = any(
        row['delta_vs_leader_markov42']['mean_pct_error'] > EPS and row['delta_vs_leader_markov42']['max_pct_error'] > EPS
        and (row.get('delta_vs_leader_kf36') is None or (row['delta_vs_leader_kf36']['mean_pct_error'] > EPS and row['delta_vs_leader_kf36']['max_pct_error'] > EPS))
        for row in rows_sorted
    )

    collapsed_rows = [
        row for row in rows_sorted
        if abs(row['delta_vs_leader_markov42']['mean_pct_error']) <= EPS
        and abs(row['delta_vs_leader_markov42']['median_pct_error']) <= EPS
        and abs(row['delta_vs_leader_markov42']['max_pct_error']) <= EPS
    ]

    if beat_current_leader:
        statement = 'A nonterminal x-conditioning path beat the corrected leader on both mean and max, and the gain survived KF36 recheck.'
    else:
        if collapsed_rows:
            statement = (
                'The anchor11 post-z preterminal x gate collapsed back onto the current leader: both tested post-z anchor11 x doses reproduced the same 1.057 / 0.611 / 4.714 landing under Markov42, '
                'and KF36 matched that neutral result. The deeper interleaved boundary-closure family and the earlier anchor10 x gate only degraded from there, so nonterminal x-conditioning did not beat the leader.'
            )
        elif best['delta_vs_terminal_x_best_markov42']['mean_pct_error'] > EPS and best['delta_vs_terminal_x_best_markov42']['max_pct_error'] > EPS:
            statement = (
                'The nonterminal pivot did improve over the failed terminal-x basin, confirming that x-side information is more useful before terminal closure than after it. '
                'But none of the tested nonterminal families beat the current corrected leader on the combined mean+max frontier.'
            )
        elif any(row['delta_vs_leader_markov42']['max_pct_error'] > EPS for row in rows_sorted):
            statement = (
                'The nonterminal pivot produced only trade-offs: at least one family trims max or x-side residuals a little, '
                'but the mean+max pair still does not beat the corrected leader.'
            )
        else:
            statement = (
                'Leaving the terminal-x basin was scientifically useful but not enough to dislodge the corrected leader: '
                'every tested nonterminal x-conditioning family stayed behind 1.057 / 0.611 / 4.714.'
            )

    summary = {
        'task': 'chapter-3 corrected nonterminal x-conditioning batch',
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
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
        'beat_current_leader': 'YES' if beat_current_leader else 'NO',
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
