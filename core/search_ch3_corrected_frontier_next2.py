from __future__ import annotations

import argparse
import json
import sys
import types
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
from search_ch3_12pos_closedloop_local_insertions import build_closedloop_candidate, run_candidate_payload
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate, render_action
from search_ch3_corrected_hidden_family_next4 import closed_pair
from search_ch3_entry_conditioned_relay_family import (
    NOISE_SCALE,
    REPORT_DATE,
    StepSpec,
    l8_xpair,
    l9_ypair_neg,
    l10_unified_core,
    l11_y10x0back2_core,
    merge_insertions,
)

ATT0_DEG = [0.0, 0.0, 0.0]
REPORT_PATH = REPORTS_DIR / f'psins_ch3_corrected_frontier_next2_{REPORT_DATE}.md'
SUMMARY_PATH = RESULTS_DIR / f'ch3_corrected_frontier_next2_{REPORT_DATE}.json'

REFERENCE_FILES = {
    'leader': {
        'label': 'current leader / relay_l12y0p5_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l12y0p5_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_relay_l12y0p5_on_entry_shared_noise0p08_param_errors.json',
    },
    'companion': {
        'label': 'closest companion / entryrelay_l8x1_l9y0p75_unifiedcore',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_entryrelay_l8x1_l9y0p75_unifiedcore_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_entryrelay_l8x1_l9y0p75_unifiedcore_shared_noise0p08_param_errors.json',
    },
    'tribranch': {
        'label': 'recent tribranch / trihybrid_l7neg1_l9y0p75_l12y0p5_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_trihybrid_l7neg1_l9y0p75_l12y0p5_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_trihybrid_l7neg1_l9y0p75_l12y0p5_on_entry_shared_noise0p08_param_errors.json',
    },
    'faithful12': {
        'label': 'faithful12',
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
        'family': 'terminal_y_reclosure_micro_ridge_around_leader',
        'summary': 'Probe only the anchor-12 terminal-y micro-ridge around the current leader. No corridor grafting, no basis change, just lighter / heavier / directionally biased reclosure on the leader backbone.',
        'candidate_names': [
            'relay_l12y0p375_on_entry',
            'relay_l12y0p625_on_entry',
            'relay_l12split625_375_on_entry',
        ],
    },
    {
        'id': 'B',
        'family': 'tiny_companion_l9_balance',
        'summary': 'Probe only tiny l9 balance changes around the companion ridge, keeping the same l8 entry conditioner and unified core, with no extra corridor structure.',
        'candidate_names': [
            'entryrelay_l8x1_l9y0p6875_unifiedcore',
            'entryrelay_l8x1_l9y0p8125_unifiedcore',
        ],
    },
    {
        'id': 'C',
        'family': 'single_small_leader_companion_bridge',
        'summary': 'Exactly one structurally justified bridge between leader and companion: slightly soften the leader-side l9 while slightly lightening the terminal l12 reclosure. This is the only true hybrid in the batch.',
        'candidate_names': [
            'bridge_l9y0p9375_l12y0p375_on_entry',
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
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
            'dKa_yy': float(payload['param_errors']['dKa_yy']['pct_error']),
            'dKg_zz': float(payload['param_errors']['dKg_zz']['pct_error']),
            'Ka2_y': float(payload['param_errors']['Ka2_y']['pct_error']),
            'Ka2_z': float(payload['param_errors']['Ka2_z']['pct_error']),
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
    extra['comparison_mode'] = 'corrected_frontier_next2'
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



def leader_backbone(l12_kind: str | None = None, out_dwell: float | None = None, ret_dwell: float | None = None) -> dict[int, list[StepSpec]]:
    parts = [
        l8_xpair(1.0, 'l8_x1'),
        l9_ypair_neg(1.0, 'l9_ypair_neg1'),
        l10_unified_core(),
        l11_y10x0back2_core(),
    ]
    if l12_kind == 'sym':
        assert out_dwell is not None
        parts.append(closed_pair(12, 'inner', -90, float(out_dwell), f"l12_yneg{str(out_dwell).replace('-', 'neg').replace('.', 'p')}"))
    elif l12_kind == 'split':
        assert out_dwell is not None and ret_dwell is not None
        parts.append(l12_split_pair(float(out_dwell), float(ret_dwell), f"l12split_{str(out_dwell).replace('.', 'p')}_{str(ret_dwell).replace('.', 'p')}"))
    return merge_insertions(*parts)



def companion_backbone(l9_dwell: float) -> dict[int, list[StepSpec]]:
    return merge_insertions(
        l8_xpair(1.0, 'l8_x1'),
        l9_ypair_neg(float(l9_dwell), f"l9_ypair_neg{str(l9_dwell).replace('-', 'neg').replace('.', 'p')}"),
        l10_unified_core(),
        l11_y10x0back2_core(),
    )



def bridge_backbone(l9_dwell: float, l12_dwell: float) -> dict[int, list[StepSpec]]:
    return merge_insertions(
        l8_xpair(1.0, 'l8_x1'),
        l9_ypair_neg(float(l9_dwell), f"l9_ypair_neg{str(l9_dwell).replace('-', 'neg').replace('.', 'p')}"),
        l10_unified_core(),
        l11_y10x0back2_core(),
        closed_pair(12, 'inner', -90, float(l12_dwell), f"l12_yneg{str(l12_dwell).replace('-', 'neg').replace('.', 'p')}"),
    )



def candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            'name': 'relay_l12y0p375_on_entry',
            'family': 'terminal_y_reclosure_micro_ridge_around_leader',
            'hypothesis_id': 'A',
            'rationale': 'Leader-only lighter terminal reclosure: keep the leader backbone unchanged and reduce anchor-12 y dwell from 0.5 s to 0.375 s on each leg.',
            'insertions': leader_backbone('sym', 0.375),
        },
        {
            'name': 'relay_l12y0p625_on_entry',
            'family': 'terminal_y_reclosure_micro_ridge_around_leader',
            'hypothesis_id': 'A',
            'rationale': 'Leader-only heavier terminal reclosure: keep the leader backbone unchanged and raise anchor-12 y dwell from 0.5 s to 0.625 s on each leg.',
            'insertions': leader_backbone('sym', 0.625),
        },
        {
            'name': 'relay_l12split625_375_on_entry',
            'family': 'terminal_y_reclosure_micro_ridge_around_leader',
            'hypothesis_id': 'A',
            'rationale': 'Leader-only directional split with preserved total terminal dwell (1.0 s total): front-load more hold on the outgoing l12 leg (0.625 / 0.375) to test whether the useful terminal signal is concentrated before exact closure.',
            'insertions': leader_backbone('split', 0.625, 0.375),
        },
        {
            'name': 'entryrelay_l8x1_l9y0p6875_unifiedcore',
            'family': 'tiny_companion_l9_balance',
            'hypothesis_id': 'B',
            'rationale': 'Tiny companion-side softening below 0.75 s: keep the same companion structure and move only the l9 y gate to 0.6875 s.',
            'insertions': companion_backbone(0.6875),
        },
        {
            'name': 'entryrelay_l8x1_l9y0p8125_unifiedcore',
            'family': 'tiny_companion_l9_balance',
            'hypothesis_id': 'B',
            'rationale': 'Tiny companion-side tightening above 0.75 s: keep the same companion structure and move only the l9 y gate to 0.8125 s.',
            'insertions': companion_backbone(0.8125),
        },
        {
            'name': 'bridge_l9y0p9375_l12y0p375_on_entry',
            'family': 'single_small_leader_companion_bridge',
            'hypothesis_id': 'C',
            'rationale': 'Single tiny bridge between leader and companion: nudge l9 slightly toward the companion (1.0 -> 0.9375) while lightening terminal l12 from 0.5 -> 0.375, with no corridor grafting and no extra family changes.',
            'insertions': bridge_backbone(0.9375, 0.375),
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
    lines = []
    lines.append('| pos | anchor | role | label | motor action | axis | rot_s | pre_s | post_s | total_s | face_after | beta_after |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|')
    for item in table:
        axis = '[' + ','.join(str(v) for v in item['effective_body_axis']) + ']'
        lines.append(
            f"| {item['pos_id']} | {item['anchor_id']} | {item['segment_role']} | {item['label']} | {item['motor_action']} | {axis} | {item['rotation_time_s']:.3f} | {item['pre_static_s']:.3f} | {item['post_static_s']:.3f} | {item['node_total_s']:.3f} | {item['face_after']} | {item['inner_beta_after_deg']} |"
        )
    return lines



def failure_reason(row: dict[str, Any]) -> str:
    d = row['delta_vs_leader_markov42']
    driver = row['markov42']['max_driver']
    return (
        f"Δmean {d['mean_pct_error']:+.3f}; Δmedian {d['median_pct_error']:+.3f}; "
        f"Δmax {d['max_pct_error']:+.3f}; max driver={driver['name']} {driver['pct_error']:.3f}"
    )



def render_report(summary: dict[str, Any]) -> str:
    refs = summary['references']
    best = summary['best_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 corrected frontier next2')
    lines.append('')
    lines.append('## 1. Mission and fixed constraints')
    lines.append('')
    lines.append('- Mission: continue the corrected-basis frontier from the current **leader / companion ridge** only.')
    lines.append('- Hard constraints stayed fixed: real dual-axis legality only; continuity-safe / reconnectable execution; original 12-position backbone unchanged; theory-guided small moves only; att0 = **(0, 0, 0)** exactly.')
    lines.append(f"- Launch leader: **{refs['leader']['markov42_triplet']}** (`relay_l12y0p5_on_entry`)")
    lines.append(f"- Closest companion: **{refs['companion']['markov42_triplet']}** (`entryrelay_l8x1_l9y0p75_unifiedcore`)")
    lines.append(f"- Recent tribranch non-winner: **{refs['tribranch']['markov42_triplet']}** (`trihybrid_l7neg1_l9y0p75_l12y0p5_on_entry`)")
    lines.append('')
    lines.append('## 2. Small batch actually tested')
    lines.append('')
    for item in summary['hypotheses_tested']:
        lines.append(f"- **{item['id']} · {item['family']}** — {item['summary']}")
        lines.append(f"  - tested: {', '.join(item['candidate_names'])}")
    lines.append('')
    lines.append('## 3. Markov42 batch results')
    lines.append('')
    lines.append('| rank | candidate | family | mean | median | max | Δmean vs leader | Δmedian vs leader | Δmax vs leader | Δmean vs companion | Δmax vs companion | max driver |')
    lines.append('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for idx, row in enumerate(summary['rows_sorted'], start=1):
        m = row['markov42']['overall']
        d0 = row['delta_vs_leader_markov42']
        d1 = row['delta_vs_companion_markov42']
        md = row['markov42']['max_driver']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {d0['mean_pct_error']:+.3f} | {d0['median_pct_error']:+.3f} | {d0['max_pct_error']:+.3f} | {d1['mean_pct_error']:+.3f} | {d1['max_pct_error']:+.3f} | {md['name']} {md['pct_error']:.3f} |"
        )
    lines.append('')
    lines.append('## 4. Best candidate of this next batch')
    lines.append('')
    lines.append(f"- **Best candidate:** `{best['candidate_name']}`")
    lines.append(f"- Rationale: {best['rationale']}")
    lines.append(f"- **Markov42:** **{overall_triplet(best['markov42'])}**")
    if best.get('kf36') is not None:
        lines.append(f"- **KF36:** **{overall_triplet(best['kf36'])}**")
    lines.append(f"- vs leader: Δmean **{best['delta_vs_leader_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_leader_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_leader_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- vs companion: Δmean **{best['delta_vs_companion_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_companion_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_companion_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- max driver: **{best['markov42']['max_driver']['name']} = {best['markov42']['max_driver']['pct_error']:.3f}%**")
    lines.append('')
    lines.append('## 5. KF36 recheck for the best competitive candidate(s)')
    lines.append('')
    lines.append(f"- Rechecked: **{', '.join(summary['kf36_rechecked_candidates']) if summary['kf36_rechecked_candidates'] else 'none'}**")
    if summary['kf36_rechecked_candidates']:
        for row in summary['rows_sorted']:
            if row.get('kf36') is not None:
                lines.append(f"  - `{row['candidate_name']}` → Markov42 **{overall_triplet(row['markov42'])}**, KF36 **{overall_triplet(row['kf36'])}**")
    else:
        lines.append(f"- Gate reason: {summary['kf36_gate_reason']}")
    lines.append('')
    lines.append('## 6. Exact legal motor / timing table for the batch-best candidate')
    lines.append('')
    lines.extend(render_timing_table_md(summary['best_candidate_timing_table']))
    lines.append('')
    lines.append('## 7. Required comparison set')
    lines.append('')
    lines.append('| path | Markov42 | KF36 | Δmean vs batch best | Δmedian vs batch best | Δmax vs batch best |')
    lines.append('|---|---:|---:|---:|---:|---:|')
    for key in ['leader', 'companion', 'tribranch', 'faithful12', 'default18']:
        ref = refs[key]
        d = delta_vs_reference(best['markov42'], ref['markov42'])
        lines.append(
            f"| {ref['label']} | {ref['markov42_triplet']} | {ref['kf36_triplet']} | {d['mean_pct_error']:+.3f} | {d['median_pct_error']:+.3f} | {d['max_pct_error']:+.3f} |"
        )
    lines.append(f"| batch-best candidate | {overall_triplet(best['markov42'])} | {overall_triplet(best['kf36']) if best.get('kf36') is not None else 'not rerun'} | {0.0:+.3f} | {0.0:+.3f} | {0.0:+.3f} |")
    lines.append('')
    lines.append('## 8. Bottom line')
    lines.append('')
    lines.append(f"- **Did this next corrected-frontier batch push below 1.064 / 0.569 / 4.859?** **{summary['bottom_line']['beat_leader_triplet']}**")
    lines.append(f"- Batch-best candidate: **{summary['bottom_line']['batch_best_triplet']}** (`{summary['bottom_line']['batch_best_name']}`)")
    lines.append(f"- Scientific read: **{summary['bottom_line']['statement']}**")
    if summary['partial_tradeoff_candidates']:
        lines.append('- Closest trade-off points:')
        for row in summary['partial_tradeoff_candidates']:
            lines.append(f"  - `{row['candidate_name']}` → {failure_reason(row)}")
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
        markov_payload, _, markov_path = run_candidate_payload(mod, candidate, 'markov42_noisy', args.noise_scale, force_rerun=args.force_rerun)
        markov_payload = attach_att0(markov_path, markov_payload, candidate.name, 'markov42_noisy', spec['family'], spec['hypothesis_id'])
        row = {
            'candidate_name': candidate.name,
            'family': spec['family'],
            'hypothesis_id': spec['hypothesis_id'],
            'rationale': spec['rationale'],
            'total_time_s': candidate.total_time_s,
            'result_files': {'markov42': str(markov_path)},
            'markov42': compact_metrics(markov_payload),
        }
        row['delta_vs_leader_markov42'] = delta_vs_reference(references['leader']['markov42'], row['markov42'])
        row['delta_vs_companion_markov42'] = delta_vs_reference(references['companion']['markov42'], row['markov42'])
        row['delta_vs_tribranch_markov42'] = delta_vs_reference(references['tribranch']['markov42'], row['markov42'])
        row['delta_vs_faithful12_markov42'] = delta_vs_reference(references['faithful12']['markov42'], row['markov42'])
        row['delta_vs_default18_markov42'] = delta_vs_reference(references['default18']['markov42'], row['markov42'])
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: (r['markov42']['overall']['mean_pct_error'], r['markov42']['overall']['max_pct_error'], r['markov42']['overall']['median_pct_error']))
    best = rows_sorted[0]

    kf36_rechecked_candidates: list[str] = []
    kf36_gate_reason = 'No candidate was close enough to the leader ridge to justify KF36 recheck.'
    for row in rows_sorted[:3]:
        d = row['delta_vs_leader_markov42']
        if d['mean_pct_error'] > -0.03 and d['max_pct_error'] > -0.12:
            candidate = candidates_by_name[row['candidate_name']]
            spec = spec_by_name[row['candidate_name']]
            kf_payload, _, kf_path = run_candidate_payload(mod, candidate, 'kf36_noisy', args.noise_scale, force_rerun=args.force_rerun)
            kf_payload = attach_att0(kf_path, kf_payload, candidate.name, 'kf36_noisy', spec['family'], spec['hypothesis_id'])
            row['result_files']['kf36'] = str(kf_path)
            row['kf36'] = compact_metrics(kf_payload)
            row['delta_vs_leader_kf36'] = delta_vs_reference(references['leader']['kf36'], row['kf36'])
            row['delta_vs_companion_kf36'] = delta_vs_reference(references['companion']['kf36'], row['kf36'])
            kf36_rechecked_candidates.append(candidate.name)
    if not kf36_rechecked_candidates:
        kf36_gate_reason = f"Best Markov42 candidate {best['candidate_name']} = {overall_triplet(best['markov42'])} stayed outside the recheck gate."
    else:
        kf36_gate_reason = 'Rechecked the near-leader terminal micro-ridge candidates that stayed within the Markov42 gate.'

    best_candidate_obj = candidates_by_name[best['candidate_name']]
    best_timing_table = build_timing_table(best_candidate_obj)

    leader_overall = references['leader']['markov42']['overall']
    beat_leader_triplet = [
        row['candidate_name']
        for row in rows_sorted
        if row['markov42']['overall']['mean_pct_error'] < leader_overall['mean_pct_error']
        and row['markov42']['overall']['median_pct_error'] < leader_overall['median_pct_error']
        and row['markov42']['overall']['max_pct_error'] < leader_overall['max_pct_error']
    ]

    partial_tradeoff_candidates = [
        row for row in rows_sorted
        if row['candidate_name'] not in beat_leader_triplet
        and (
            row['delta_vs_leader_markov42']['mean_pct_error'] > 0
            or row['delta_vs_leader_markov42']['max_pct_error'] > 0
            or abs(row['delta_vs_leader_markov42']['median_pct_error']) <= 0.015
        )
    ][:4]

    if beat_leader_triplet:
        statement = 'The local leader/companion ridge still had headroom: at least one next-step probe beat the full leader triplet, so the corrected frontier was not yet saturated.'
        beat_flag = 'YES'
    else:
        statement = 'No probe in this next batch beat the full leader triplet. The ridge still shows partial trade-offs (some points help mean or max), but the combined leader basin around relay_l12y0p5_on_entry remains locally saturated under these continuity-safe micro-moves.'
        beat_flag = 'NO'

    summary = {
        'task': 'chapter-3 corrected frontier next2',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'corrected_att0_deg': ATT0_DEG,
        'references': references,
        'hypotheses_tested': HYPOTHESES,
        'tested_candidates': [spec['name'] for spec in specs],
        'rows_sorted': rows_sorted,
        'best_candidate': best,
        'best_candidate_timing_table': best_timing_table,
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
        'kf36_gate_reason': kf36_gate_reason,
        'beat_leader_triplet_candidates': beat_leader_triplet,
        'partial_tradeoff_candidates': partial_tradeoff_candidates,
        'bottom_line': {
            'beat_leader_triplet': beat_flag,
            'batch_best_name': best['candidate_name'],
            'batch_best_triplet': overall_triplet(best['markov42']),
            'statement': statement,
        },
    }

    REPORT_PATH.write_text(render_report(summary), encoding='utf-8')
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({
        'report_path': str(REPORT_PATH),
        'summary_path': str(SUMMARY_PATH),
        'batch_best_name': best['candidate_name'],
        'batch_best_triplet': overall_triplet(best['markov42']),
        'beat_leader_triplet': beat_flag,
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
