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
from search_ch3_corrected_hidden_family_next4 import closed_pair
from search_ch3_entry_conditioned_relay_family import (
    NOISE_SCALE,
    l8_xpair,
    l9_ypair_neg,
    l10_unified_core,
    l11_y10x0back2_core,
    merge_insertions,
)

ATT0_DEG = [0.0, 0.0, 0.0]
REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
REPORT_PATH = REPORTS_DIR / f'psins_ch3_corrected_inbasin_ridge_resume_{REPORT_DATE}.md'
SUMMARY_PATH = RESULTS_DIR / f'ch3_corrected_inbasin_ridge_resume_{REPORT_DATE}.json'

REFERENCE_FILES = {
    'accepted_leader': {
        'label': 'accepted corrected leader / relay_r3_l9y0p8125_l12y0p125_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_r3_l9y0p8125_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_relay_r3_l9y0p8125_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
    },
    'max_oriented_neighbor': {
        'label': 'nearby max-oriented point / relay_r3_l9y0p8125_l12y0p25_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_r3_l9y0p8125_l12y0p25_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_relay_r3_l9y0p8125_l12y0p25_on_entry_shared_noise0p08_param_errors.json',
    },
    'batch2_incumbent': {
        'label': 'earlier accepted ridge base / relay_l9y0p75_l12y0p25_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l9y0p75_l12y0p25_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_relay_l9y0p75_l12y0p25_on_entry_shared_noise0p08_param_errors.json',
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
        str(REPORTS_DIR / 'psins_ch3_corrected_inbasin_coupling_resume_2026-04-02.md'),
        str(REPORTS_DIR / 'psins_ch3_corrected_frontier_relaunch_batch3_2026-04-02.md'),
        str(REPORTS_DIR / 'psins_ch3_corrected_frontier_next2_2026-04-02.md'),
    ],
    'source_summaries': [
        str(RESULTS_DIR / 'ch3_corrected_inbasin_coupling_resume_2026-04-02.json'),
        str(RESULTS_DIR / 'ch3_corrected_frontier_relaunch_batch3_2026-04-02.json'),
    ],
}

HYPOTHESES = [
    {
        'id': 'A',
        'family': 'fixed_l9_terminal_midpoint_compromise',
        'summary': 'Hold the accepted l9 = 0.8125 s ridge fixed and move the terminal l12 dwell halfway between the current mean-leaning leader (0.125 s) and the max-leaning neighbor (0.25 s).',
        'candidate_names': [
            'relay_r4_l9y0p8125_l12y0p1875_on_entry',
        ],
    },
    {
        'id': 'B',
        'family': 'higher_l9_with_light_terminal',
        'summary': 'Increase l9 only by one half-step to 0.84375 s while keeping the leader\'s lighter l12 = 0.125 s, to see whether dKg_xx suppression keeps improving before mean drifts.',
        'candidate_names': [
            'relay_r4_l9y0p84375_l12y0p125_on_entry',
        ],
    },
    {
        'id': 'C',
        'family': 'higher_l9_coupled_terminal_push',
        'summary': 'Couple the same small l9 increase with midpoint / heavier terminal closures. This tests whether the max-oriented direction can be extended without reopening any new family.',
        'candidate_names': [
            'relay_r4_l9y0p84375_l12y0p1875_on_entry',
            'relay_r4_l9y0p84375_l12y0p25_on_entry',
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



def max_param(payload: dict[str, Any]) -> dict[str, Any]:
    name, info = max(payload['param_errors'].items(), key=lambda kv: float(kv[1]['pct_error']))
    return {'name': name, 'pct_error': float(info['pct_error'])}



def compact_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        'overall': {
            'mean_pct_error': float(payload['overall']['mean_pct_error']),
            'median_pct_error': float(payload['overall']['median_pct_error']),
            'max_pct_error': float(payload['overall']['max_pct_error']),
        },
        'key_param_errors': {
            'dKg_xx': float(payload['param_errors']['dKg_xx']['pct_error']),
            'eb_x': float(payload['param_errors']['eb_x']['pct_error']),
            'Ka2_y': float(payload['param_errors']['Ka2_y']['pct_error']),
            'dKa_xx': float(payload['param_errors']['dKa_xx']['pct_error']),
            'dKg_yy': float(payload['param_errors']['dKg_yy']['pct_error']),
        },
        'max_driver': max_param(payload),
    }



def overall_triplet(payload: dict[str, Any]) -> str:
    o = payload['overall']
    return f"{o['mean_pct_error']:.3f} / {o['median_pct_error']:.3f} / {o['max_pct_error']:.3f}"



def delta_vs_reference(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        out[metric] = float(reference['overall'][metric]) - float(candidate['overall'][metric])
    return out



def attach_att0(path: Path, payload: dict[str, Any], candidate_name: str, method_key: str, family: str, hypothesis_id: str) -> dict[str, Any]:
    extra = payload.setdefault('extra', {})
    extra['att0_deg'] = ATT0_DEG
    extra['comparison_mode'] = 'corrected_inbasin_ridge_resume'
    extra['candidate_registry_key'] = candidate_name
    extra['method_key'] = method_key
    extra['family'] = family
    extra['hypothesis_id'] = hypothesis_id
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload



def l12_split_pair(out_dwell_s: float, ret_dwell_s: float, label: str) -> dict[int, list[StepSpec]]:
    return {
        12: [
            StepSpec(kind='inner', angle_deg=-90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(out_dwell_s), segment_role='motif_out', label=f'{label}_out'),
            StepSpec(kind='inner', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(ret_dwell_s), segment_role='motif_return', label=f'{label}_return'),
        ]
    }



def core_with(l9_dwell: float, l12_kind: str, out_dwell: float, ret_dwell: float | None = None) -> dict[int, list[StepSpec]]:
    parts = [
        l8_xpair(1.0, 'l8_x1'),
        l9_ypair_neg(l9_dwell, f'l9_ypair_neg{str(l9_dwell).replace("-", "neg").replace(".", "p")}'),
        l10_unified_core(),
        l11_y10x0back2_core(),
    ]
    if l12_kind == 'sym':
        parts.append(closed_pair(12, 'inner', -90, out_dwell, f'l12_yneg{str(out_dwell).replace("-", "neg").replace(".", "p")}'))
    elif l12_kind == 'split':
        if ret_dwell is None:
            raise ValueError('split l12 requires ret_dwell')
        parts.append(l12_split_pair(out_dwell, ret_dwell, f'l12split_{str(out_dwell).replace(".", "p")}_{str(ret_dwell).replace(".", "p")}'))
    else:
        raise KeyError(l12_kind)
    return merge_insertions(*parts)



def accepted_leader_spec() -> dict[str, Any]:
    return {
        'name': 'relay_r3_l9y0p8125_l12y0p125_on_entry',
        'family': 'accepted_reference',
        'hypothesis_id': 'ref',
        'rationale': 'Already accepted corrected leader from the prior in-basin coupling batch.',
        'insertions': core_with(0.8125, 'sym', 0.125),
    }



def candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            'name': 'relay_r4_l9y0p8125_l12y0p1875_on_entry',
            'family': 'fixed_l9_terminal_midpoint_compromise',
            'hypothesis_id': 'A',
            'rationale': 'Keep the accepted l9 ridge fixed and place l12 at the exact midpoint between the current leader (0.125 s) and the max-oriented neighbor (0.25 s).',
            'insertions': core_with(0.8125, 'sym', 0.1875),
        },
        {
            'name': 'relay_r4_l9y0p84375_l12y0p125_on_entry',
            'family': 'higher_l9_with_light_terminal',
            'hypothesis_id': 'B',
            'rationale': 'Micro-extend the accepted l9 gate upward by one half-step while keeping the lighter terminal closure unchanged, to test whether the current ridge is still climbing on max without sacrificing mean.',
            'insertions': core_with(0.84375, 'sym', 0.125),
        },
        {
            'name': 'relay_r4_l9y0p84375_l12y0p1875_on_entry',
            'family': 'higher_l9_coupled_terminal_push',
            'hypothesis_id': 'C',
            'rationale': 'Diagonal in-basin compromise: slightly stronger l9 plus midpoint l12, aimed at absorbing some max gain from the heavier terminal side with less mean tax than the full 0.25 closure.',
            'insertions': core_with(0.84375, 'sym', 0.1875),
        },
        {
            'name': 'relay_r4_l9y0p84375_l12y0p25_on_entry',
            'family': 'higher_l9_coupled_terminal_push',
            'hypothesis_id': 'C',
            'rationale': 'Most max-oriented admissible extension inside the same basin: raise l9 one half-step and keep the heavier 0.25 s terminal closure from the known max-leaning neighbor.',
            'insertions': core_with(0.84375, 'sym', 0.25),
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



def make_comparison_row(label: str, markov42: dict[str, Any], best_markov42: dict[str, Any], note: str, kf36: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        'label': label,
        'markov42_triplet': overall_triplet(markov42),
        'note': note,
        'delta_vs_batch_best_markov42': delta_vs_reference(markov42, best_markov42),
    }
    row['kf36_triplet'] = overall_triplet(kf36) if kf36 is not None else 'n/a'
    return row



def render_report(summary: dict[str, Any]) -> str:
    refs = summary['references']
    best = summary['best_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 corrected-basis in-basin ridge resume')
    lines.append('')
    lines.append('## 1. Relaunch context')
    lines.append('')
    lines.append('- No active PSINS run was alive at relaunch time; this batch is the minimal explicit restart requested by `PENDING.md`.')
    lines.append(f"- Accepted frontier at launch: `{refs['accepted_leader']['label'].split(' / ')[-1]}` = **{refs['accepted_leader']['markov42_triplet']}** (Markov42), **{refs['accepted_leader']['kf36_triplet']}** (KF36)")
    lines.append(f"- Nearby max-oriented trade-off already known: `{refs['max_oriented_neighbor']['label'].split(' / ')[-1]}` = **{refs['max_oriented_neighbor']['markov42_triplet']}** (Markov42), **{refs['max_oriented_neighbor']['kf36_triplet']}** (KF36)")
    lines.append('- Hard constraints remained fixed: **att0 = (0,0,0)**, real dual-axis legality only, same faithful 12-position backbone, no fresh-family reopening.')
    lines.append('- Reused prior evidence:')
    lines.append(f"  - `{REUSED_CONTEXT['pending']}`")
    for path in REUSED_CONTEXT['source_reports']:
        lines.append(f'  - `{path}`')
    lines.append('')
    lines.append('## 2. Minimal batch actually run')
    lines.append('')
    for item in summary['hypotheses_tested']:
        lines.append(f"- **{item['id']} · {item['family']}** — {item['summary']}")
        lines.append(f"  - tested: {', '.join(item['candidate_names'])}")
    lines.append('')
    lines.append('## 3. Markov42 local results')
    lines.append('')
    lines.append('| rank | candidate | family | mean | median | max | Δmean vs accepted | Δmedian vs accepted | Δmax vs accepted | Δmean vs max-point | Δmax vs max-point | max driver |')
    lines.append('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for idx, row in enumerate(summary['rows_sorted'], start=1):
        m = row['markov42']['overall']
        da = row['delta_vs_accepted_markov42']
        dm = row['delta_vs_max_neighbor_markov42']
        md = row['markov42']['max_driver']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {da['mean_pct_error']:+.3f} | {da['median_pct_error']:+.3f} | {da['max_pct_error']:+.3f} | {dm['mean_pct_error']:+.3f} | {dm['max_pct_error']:+.3f} | {md['name']} {md['pct_error']:.3f} |"
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
    lines.append('## 5. Decision from this batch')
    lines.append('')
    lines.append(f"- **Batch-best Markov42 candidate:** `{best['candidate_name']}` = **{overall_triplet(best['markov42'])}**")
    if best.get('kf36') is not None:
        lines.append(f"- **KF36 for batch best:** **{overall_triplet(best['kf36'])}**")
    lines.append(
        f"- vs accepted leader: Δmean **{best['delta_vs_accepted_markov42']['mean_pct_error']:+.6f}**, Δmedian **{best['delta_vs_accepted_markov42']['median_pct_error']:+.6f}**, Δmax **{best['delta_vs_accepted_markov42']['max_pct_error']:+.6f}**"
    )
    lines.append(
        f"- vs max-oriented neighbor: Δmean **{best['delta_vs_max_neighbor_markov42']['mean_pct_error']:+.6f}**, Δmedian **{best['delta_vs_max_neighbor_markov42']['median_pct_error']:+.6f}**, Δmax **{best['delta_vs_max_neighbor_markov42']['max_pct_error']:+.6f}**"
    )
    lines.append(f"- **Did this batch beat 1.057 / 0.611 / 4.714?** **{summary['bottom_line']['beat_current_leader']}**")
    lines.append(f"- **Closure recommendation for this pending item:** **{summary['bottom_line']['task_can_close']}**")
    lines.append(f"- **Recommended frontier after this batch:** `{summary['bottom_line']['recommended_frontier_name']}` = **{summary['bottom_line']['recommended_frontier_triplet']}**")
    lines.append(f"- **Materially better valid strategy beyond accepted leader found?** **{summary['bottom_line']['materially_better_found']}**")
    lines.append(f"- Scientific read: **{summary['bottom_line']['statement']}**")
    if summary['near_tradeoffs']:
        lines.append('- Useful trade-offs still worth noting:')
        for row in summary['near_tradeoffs']:
            lines.append(
                f"  - `{row['candidate_name']}` → Δmean {row['delta_vs_accepted_markov42']['mean_pct_error']:+.6f}, Δmax {row['delta_vs_accepted_markov42']['max_pct_error']:+.6f}, max driver {row['markov42']['max_driver']['name']} {row['markov42']['max_driver']['pct_error']:.3f}"
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

    leader_spec = accepted_leader_spec()
    leader_candidate = build_closedloop_candidate(mod, leader_spec, faithful.rows, faithful.action_sequence)

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
        row['delta_vs_accepted_markov42'] = delta_vs_reference(references['accepted_leader']['markov42'], row['markov42'])
        row['delta_vs_max_neighbor_markov42'] = delta_vs_reference(references['max_oriented_neighbor']['markov42'], row['markov42'])
        row['delta_vs_batch2_markov42'] = delta_vs_reference(references['batch2_incumbent']['markov42'], row['markov42'])
        row['delta_vs_faithful12_markov42'] = delta_vs_reference(references['faithful12']['markov42'], row['markov42'])
        row['delta_vs_default18_markov42'] = delta_vs_reference(references['default18']['markov42'], row['markov42'])
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: (r['markov42']['overall']['mean_pct_error'], r['markov42']['overall']['max_pct_error'], r['markov42']['overall']['median_pct_error']))
    best = rows_sorted[0]
    best_candidate_timing_table = build_timing_table(candidates_by_name[best['candidate_name']])

    kf36_rechecked_candidates: list[str] = []
    kf36_gate_reason = 'No candidate was close enough to the accepted leader to justify KF36.'
    for row in rows_sorted[:3]:
        da = row['delta_vs_accepted_markov42']
        if da['mean_pct_error'] > -0.015 and da['max_pct_error'] > -0.060:
            candidate = candidates_by_name[row['candidate_name']]
            spec = spec_by_name[row['candidate_name']]
            kf_payload, kf_mode, kf_path = run_candidate_payload(mod, candidate, 'kf36_noisy', args.noise_scale, force_rerun=args.force_rerun)
            kf_payload = attach_att0(kf_path, kf_payload, candidate.name, 'kf36_noisy', spec['family'], spec['hypothesis_id'])
            row['result_files']['kf36'] = str(kf_path)
            row['result_modes']['kf36'] = kf_mode
            row['kf36'] = compact_metrics(kf_payload)
            row['delta_vs_accepted_kf36'] = delta_vs_reference(references['accepted_leader']['kf36'], row['kf36'])
            row['delta_vs_max_neighbor_kf36'] = delta_vs_reference(references['max_oriented_neighbor']['kf36'], row['kf36'])
            kf36_rechecked_candidates.append(candidate.name)
    if not kf36_rechecked_candidates:
        kf36_gate_reason = f"Best Markov42 candidate {best['candidate_name']} = {overall_triplet(best['markov42'])} stayed outside the near-leader gate (need Δmean > -0.015 and Δmax > -0.060)."

    recommended_name = references['accepted_leader']['label'].split(' / ')[-1]
    recommended_triplet = references['accepted_leader']['markov42_triplet']
    recommended_timing_table = build_timing_table(leader_candidate)
    materially_better_found = 'NO'
    task_can_close = 'YES'
    statement = 'The local in-basin ridge remains centered on the accepted leader: this restart batch did not uncover a clearly better valid point beyond the already accepted corrected frontier.'

    accepted_improvers_markov = [
        row for row in rows_sorted
        if row['delta_vs_accepted_markov42']['mean_pct_error'] > 0 and row['delta_vs_accepted_markov42']['max_pct_error'] > 0
    ]
    accepted_improvers_kf = [
        row for row in accepted_improvers_markov
        if row.get('delta_vs_accepted_kf36') is not None and row['delta_vs_accepted_kf36']['mean_pct_error'] > 0 and row['delta_vs_accepted_kf36']['max_pct_error'] > 0
    ]

    if accepted_improvers_kf:
        chosen = accepted_improvers_kf[0]
        recommended_name = chosen['candidate_name']
        recommended_triplet = overall_triplet(chosen['markov42'])
        recommended_timing_table = build_timing_table(candidates_by_name[chosen['candidate_name']])
        materially_better_found = 'YES'
        statement = 'A genuinely better corrected-basis point survived both Markov42 and KF36 inside the same in-basin ridge, so the accepted leader should be updated to the new candidate.'
    else:
        best_tradeoff = min(rows_sorted, key=lambda r: (abs(r['delta_vs_accepted_markov42']['mean_pct_error']), -r['delta_vs_accepted_markov42']['max_pct_error']))
        if best_tradeoff['delta_vs_accepted_markov42']['max_pct_error'] > 0 and best_tradeoff['delta_vs_accepted_markov42']['mean_pct_error'] <= 0:
            statement = 'The restart batch only confirmed the same trade-off direction: some nearby points still shave max a little, but none beat the accepted leader on the mean+max pair with KF36 support. The accepted leader remains the clean recommendation.'

    near_tradeoffs = [
        row for row in rows_sorted
        if row['delta_vs_accepted_markov42']['max_pct_error'] > 0 or abs(row['delta_vs_accepted_markov42']['mean_pct_error']) <= 0.002
    ][:4]

    required_comparison_rows = [
        make_comparison_row(
            'current leader / relay_r3_l9y0p8125_l12y0p125_on_entry',
            references['accepted_leader']['markov42'],
            best['markov42'],
            'accepted corrected leader to beat',
            references['accepted_leader']['kf36'],
        ),
        make_comparison_row(
            'nearby max-oriented point / relay_r3_l9y0p8125_l12y0p25_on_entry',
            references['max_oriented_neighbor']['markov42'],
            best['markov42'],
            'same-basin max-leaning trade-off point',
            references['max_oriented_neighbor']['kf36'],
        ),
        make_comparison_row(
            'faithful12',
            references['faithful12']['markov42'],
            best['markov42'],
            'corrected faithful 12-position reference',
            references['faithful12']['kf36'],
        ),
        make_comparison_row(
            'default18',
            references['default18']['markov42'],
            best['markov42'],
            'default 18-position reference',
            references['default18']['kf36'],
        ),
        make_comparison_row(
            f"batch-best candidate / {best['candidate_name']}",
            best['markov42'],
            best['markov42'],
            'best point found inside this restart batch',
            best.get('kf36'),
        ),
    ]

    summary = {
        'task': 'chapter-3 corrected-basis in-basin ridge resume',
        'report_date': REPORT_DATE,
        'noise_scale': args.noise_scale,
        'corrected_att0_deg': ATT0_DEG,
        'reused_context': REUSED_CONTEXT,
        'references': references,
        'hypotheses_tested': HYPOTHESES,
        'tested_candidates': [spec['name'] for spec in specs],
        'rows_sorted': rows_sorted,
        'best_candidate': best,
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
        'kf36_gate_reason': kf36_gate_reason,
        'accepted_improvers_markov42': [row['candidate_name'] for row in accepted_improvers_markov],
        'accepted_improvers_kf36': [row['candidate_name'] for row in accepted_improvers_kf],
        'recommended_frontier_name': recommended_name,
        'recommended_frontier_triplet': recommended_triplet,
        'recommended_timing_table': recommended_timing_table,
        'best_candidate_timing_table': best_candidate_timing_table,
        'required_comparison_rows': required_comparison_rows,
        'near_tradeoffs': near_tradeoffs,
        'bottom_line': {
            'task_can_close': task_can_close,
            'recommended_frontier_name': recommended_name,
            'recommended_frontier_triplet': recommended_triplet,
            'materially_better_found': materially_better_found,
            'beat_current_leader': 'YES' if materially_better_found == 'YES' else 'NO',
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
        'recommended_frontier_name': recommended_name,
        'recommended_frontier_triplet': recommended_triplet,
        'materially_better_found': materially_better_found,
        'task_can_close': task_can_close,
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
