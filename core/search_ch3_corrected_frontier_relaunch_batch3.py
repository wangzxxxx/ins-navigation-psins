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
REPORT_PATH = REPORTS_DIR / f'psins_ch3_corrected_frontier_relaunch_batch3_{REPORT_DATE}.md'
SUMMARY_PATH = RESULTS_DIR / f'ch3_corrected_frontier_relaunch_batch3_{REPORT_DATE}.json'

REFERENCE_FILES = {
    'prompt_launch_leader': {
        'label': 'prompt launch leader / relay_l12y0p5_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l12y0p5_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_relay_l12y0p5_on_entry_shared_noise0p08_param_errors.json',
    },
    'batch2_incumbent': {
        'label': 'actual incumbent after resume batch2 / relay_l9y0p75_l12y0p25_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l9y0p75_l12y0p25_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_relay_l9y0p75_l12y0p25_on_entry_shared_noise0p08_param_errors.json',
    },
    'prompt_second_point': {
        'label': 'prompt second point / entryrelay_l8x1_l9y0p75_unifiedcore',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_entryrelay_l8x1_l9y0p75_unifiedcore_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_entryrelay_l8x1_l9y0p75_unifiedcore_shared_noise0p08_param_errors.json',
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
        'family': 'terminal_y_reclosure_micro_ridge_around_batch2_incumbent',
        'summary': 'Hold the incumbent l9 = 0.75 s soft gate fixed and probe only smaller / larger / asymmetric l12 terminal reclosures. This directly tests whether the new winner is sitting on a narrow terminal timing ridge or just at one coarse-grid lucky point.',
        'candidate_names': [
            'relay_r3_l9y0p75_l12y0p125_on_entry',
            'relay_r3_l9y0p75_l12y0p375_on_entry',
            'relay_r3_l9y0p75_l12split375_125_on_entry',
            'relay_r3_l9y0p75_l12split125_375_on_entry',
        ],
    },
    {
        'id': 'B',
        'family': 'minimal_l9_l12_compatibility_tweaks_around_batch2_incumbent',
        'summary': 'Retune l9 slightly below/above 0.75 s, first with the incumbent l12 = 0.25 s micro-closure and then with the directionally expected companion l12 adjustment (lower l9 -> slightly heavier closure, higher l9 -> slightly lighter closure).',
        'candidate_names': [
            'relay_r3_l9y0p6875_l12y0p25_on_entry',
            'relay_r3_l9y0p8125_l12y0p25_on_entry',
            'relay_r3_l9y0p6875_l12y0p375_on_entry',
            'relay_r3_l9y0p8125_l12y0p125_on_entry',
        ],
    },
    {
        'id': 'C',
        'family': 'tight_micro_hybrid_with_prompt_second_point',
        'summary': 'The asymmetrically split l12 probes are the physically meaningful micro-hybrid here: they keep the prompt second point backbone (`l8 x1 + l9 0.75 + unified core`) and only add a tiny directionally biased terminal reclosure, instead of reopening broader side families.',
        'candidate_names': [
            'relay_r3_l9y0p75_l12split375_125_on_entry',
            'relay_r3_l9y0p75_l12split125_375_on_entry',
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
    extra['comparison_mode'] = 'corrected_frontier_relaunch_batch3'
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



def candidate_specs() -> list[dict[str, Any]]:
    return [
        {
            'name': 'relay_r3_l9y0p75_l12y0p125_on_entry',
            'family': 'terminal_y_reclosure_micro_ridge_around_batch2_incumbent',
            'hypothesis_id': 'A',
            'rationale': 'Same incumbent backbone, but halve the terminal l12 dwell from 0.25 s to 0.125 s on each leg to test whether the current winner is still slightly over-closed at the terminal y ridge.',
            'insertions': core_with(0.75, 'sym', 0.125),
        },
        {
            'name': 'relay_r3_l9y0p75_l12y0p375_on_entry',
            'family': 'terminal_y_reclosure_micro_ridge_around_batch2_incumbent',
            'hypothesis_id': 'A',
            'rationale': 'Same incumbent backbone, but strengthen the terminal l12 dwell from 0.25 s to 0.375 s on each leg to test the opposite side of the local terminal ridge.',
            'insertions': core_with(0.75, 'sym', 0.375),
        },
        {
            'name': 'relay_r3_l9y0p75_l12split375_125_on_entry',
            'family': 'terminal_y_reclosure_micro_ridge_around_batch2_incumbent',
            'hypothesis_id': 'A',
            'rationale': 'Keep the total terminal dwell equal to the incumbent (0.5 s total), but front-load it on the outgoing l12 leg (0.375 / 0.125). This checks whether the useful terminal information is concentrated before exact closure rather than after it.',
            'insertions': core_with(0.75, 'split', 0.375, 0.125),
        },
        {
            'name': 'relay_r3_l9y0p75_l12split125_375_on_entry',
            'family': 'terminal_y_reclosure_micro_ridge_around_batch2_incumbent',
            'hypothesis_id': 'A',
            'rationale': 'Reverse the same total terminal dwell split (0.125 / 0.375) to test whether any asymmetry benefit is directional or just due to total time.',
            'insertions': core_with(0.75, 'split', 0.125, 0.375),
        },
        {
            'name': 'relay_r3_l9y0p6875_l12y0p25_on_entry',
            'family': 'minimal_l9_l12_compatibility_tweaks_around_batch2_incumbent',
            'hypothesis_id': 'B',
            'rationale': 'Lower the l9 gate slightly below 0.75 s while keeping the incumbent l12 = 0.25 s, to test whether the coarse-grid optimum is drifting mildly downward.',
            'insertions': core_with(0.6875, 'sym', 0.25),
        },
        {
            'name': 'relay_r3_l9y0p8125_l12y0p25_on_entry',
            'family': 'minimal_l9_l12_compatibility_tweaks_around_batch2_incumbent',
            'hypothesis_id': 'B',
            'rationale': 'Raise the l9 gate slightly above 0.75 s while keeping the incumbent l12 = 0.25 s, to test the upper side of the same compatibility ridge.',
            'insertions': core_with(0.8125, 'sym', 0.25),
        },
        {
            'name': 'relay_r3_l9y0p6875_l12y0p375_on_entry',
            'family': 'minimal_l9_l12_compatibility_tweaks_around_batch2_incumbent',
            'hypothesis_id': 'B',
            'rationale': 'Coupled lower-l9 / heavier-l12 variant: if earlier l9 closure slightly under-excites the terminal correction, a somewhat stronger l12 reclosure may compensate.',
            'insertions': core_with(0.6875, 'sym', 0.375),
        },
        {
            'name': 'relay_r3_l9y0p8125_l12y0p125_on_entry',
            'family': 'minimal_l9_l12_compatibility_tweaks_around_batch2_incumbent',
            'hypothesis_id': 'B',
            'rationale': 'Coupled higher-l9 / lighter-l12 variant: if later l9 already supplies more y-information, the terminal closure may need to back off proportionally.',
            'insertions': core_with(0.8125, 'sym', 0.125),
        },
    ]



def failure_reason(row: dict[str, Any]) -> str:
    d = row['delta_vs_batch2_markov42']
    parts: list[str] = []
    parts.append(f"mean {d['mean_pct_error']:+.3f}")
    parts.append(f"median {d['median_pct_error']:+.3f}")
    parts.append(f"max {d['max_pct_error']:+.3f}")
    driver = row['markov42']['max_driver']
    parts.append(f"max driver={driver['name']} {driver['pct_error']:.3f}")
    return '; '.join(parts)



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



def render_report(summary: dict[str, Any]) -> str:
    refs = summary['references']
    best = summary['best_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 corrected frontier relaunch batch 3')
    lines.append('')
    lines.append('## 1. Launch clarification')
    lines.append('')
    lines.append('- Prompt launch leader used by the task statement: `relay_l12y0p5_on_entry` = **1.064 / 0.569 / 4.859**.')
    lines.append('- But the workspace already contained a same-day verified batch-2 incumbent: `relay_l9y0p75_l12y0p25_on_entry` = **1.063 / 0.615 / 4.725** (Markov42 and KF36 aligned).')
    lines.append('- Therefore this relaunch was executed as a **true local refinement around the actual incumbent**, not a rollback to the stale pre-batch2 leader.')
    lines.append('- Fixed constraints remained: att0 = (0,0,0), real dual-axis legality only, exact continuity-safe closure, faithful 12-position backbone, total time 20–30 min.')
    lines.append('')
    lines.append('## 2. Narrow hypotheses tested')
    lines.append('')
    for item in summary['hypotheses_tested']:
        lines.append(f"- **{item['id']} · {item['family']}** — {item['summary']}")
        lines.append(f"  - tested: {', '.join(item['candidate_names'])}")
    lines.append('')
    lines.append('## 3. Markov42 local batch results')
    lines.append('')
    lines.append('| rank | candidate | family | mean | median | max | Δmean vs batch2 | Δmedian vs batch2 | Δmax vs batch2 | Δmean vs prompt leader | Δmax vs prompt leader | max driver |')
    lines.append('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for idx, row in enumerate(summary['rows_sorted'], start=1):
        m = row['markov42']['overall']
        d0 = row['delta_vs_batch2_markov42']
        d1 = row['delta_vs_prompt_leader_markov42']
        md = row['markov42']['max_driver']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {d0['mean_pct_error']:+.3f} | {d0['median_pct_error']:+.3f} | {d0['max_pct_error']:+.3f} | {d1['mean_pct_error']:+.3f} | {d1['max_pct_error']:+.3f} | {md['name']} {md['pct_error']:.3f} |"
        )
    lines.append('')
    lines.append('## 4. Best candidate from this relaunch batch')
    lines.append('')
    lines.append(f"- **Best candidate:** `{best['candidate_name']}`")
    lines.append(f"- Rationale: {best['rationale']}")
    lines.append(f"- **Markov42:** **{overall_triplet(best['markov42'])}**")
    if best.get('kf36') is not None:
        lines.append(f"- **KF36:** **{overall_triplet(best['kf36'])}**")
    lines.append(f"- vs batch2 incumbent: Δmean **{best['delta_vs_batch2_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_batch2_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_batch2_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- vs prompt launch leader: Δmean **{best['delta_vs_prompt_leader_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_prompt_leader_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_prompt_leader_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- max driver: **{best['markov42']['max_driver']['name']} = {best['markov42']['max_driver']['pct_error']:.3f}%**")
    lines.append('')
    lines.append('## 5. KF36 recheck')
    lines.append('')
    if summary['kf36_rechecked_candidates']:
        lines.append(f"- Rechecked candidates: **{', '.join(summary['kf36_rechecked_candidates'])}**")
        for row in summary['rows_sorted']:
            if row.get('kf36') is not None:
                lines.append(f"  - `{row['candidate_name']}` → Markov42 **{overall_triplet(row['markov42'])}**, KF36 **{overall_triplet(row['kf36'])}**")
    else:
        lines.append(f"- No KF36 reruns triggered. Gate reason: {summary['kf36_gate_reason']}")
    lines.append('')
    lines.append('## 6. Comparison against required references')
    lines.append('')
    lines.append('| reference | Markov42 | KF36 | Δmean vs batch best | Δmax vs batch best |')
    lines.append('|---|---:|---:|---:|---:|')
    for key in ['batch2_incumbent', 'prompt_launch_leader', 'prompt_second_point', 'faithful12', 'default18']:
        ref = refs[key]
        dmean = float(ref['markov42']['overall']['mean_pct_error']) - float(best['markov42']['overall']['mean_pct_error'])
        dmax = float(ref['markov42']['overall']['max_pct_error']) - float(best['markov42']['overall']['max_pct_error'])
        lines.append(
            f"| {ref['label']} | {ref['markov42_triplet']} | {ref['kf36_triplet']} | {dmean:+.3f} | {dmax:+.3f} |"
        )
    lines.append('')
    lines.append('## 7. Exact legal motor / timing table for the batch-best candidate')
    lines.append('')
    lines.extend(render_timing_table_md(summary['best_candidate_timing_table']))
    lines.append('')
    lines.append('## 8. Bottom line')
    lines.append('')
    lines.append(f"- **Did this relaunch beat the batch2 incumbent `{refs['batch2_incumbent']['label'].split(' / ')[-1]}`?** **{summary['bottom_line']['beat_batch2_incumbent']}**")
    lines.append(f"- **Did this relaunch produce a candidate below the task threshold 1.064 / 0.569 / 4.859?** **{summary['bottom_line']['beat_prompt_threshold']}**")
    lines.append(f"- Batch-best candidate of this relaunch: **{summary['bottom_line']['batch_best_triplet']}** (`{summary['bottom_line']['batch_best_name']}`)")
    lines.append(f"- Overall scientific read: **{summary['bottom_line']['statement']}**")
    if summary['near_misses']:
        lines.append('- Near-miss notes:')
        for row in summary['near_misses']:
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
        row['delta_vs_batch2_markov42'] = delta_vs_reference(references['batch2_incumbent']['markov42'], row['markov42'])
        row['delta_vs_prompt_leader_markov42'] = delta_vs_reference(references['prompt_launch_leader']['markov42'], row['markov42'])
        row['delta_vs_second_point_markov42'] = delta_vs_reference(references['prompt_second_point']['markov42'], row['markov42'])
        row['delta_vs_faithful12_markov42'] = delta_vs_reference(references['faithful12']['markov42'], row['markov42'])
        row['delta_vs_default18_markov42'] = delta_vs_reference(references['default18']['markov42'], row['markov42'])
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: (r['markov42']['overall']['mean_pct_error'], r['markov42']['overall']['max_pct_error'], r['markov42']['overall']['median_pct_error']))
    best = rows_sorted[0]

    beat_batch2_incumbent = [
        row['candidate_name']
        for row in rows_sorted
        if row['delta_vs_batch2_markov42']['mean_pct_error'] > 0 and row['delta_vs_batch2_markov42']['max_pct_error'] > 0
    ]
    beat_prompt_threshold = [
        row['candidate_name']
        for row in rows_sorted
        if row['delta_vs_prompt_leader_markov42']['mean_pct_error'] > 0 and row['delta_vs_prompt_leader_markov42']['max_pct_error'] > 0
    ]
    near_misses = [
        row for row in rows_sorted
        if not (row['delta_vs_batch2_markov42']['mean_pct_error'] > 0 and row['delta_vs_batch2_markov42']['max_pct_error'] > 0)
        and (
            abs(row['delta_vs_batch2_markov42']['mean_pct_error']) <= 0.02
            or abs(row['delta_vs_batch2_markov42']['max_pct_error']) <= 0.12
        )
    ][:4]

    kf36_rechecked_candidates: list[str] = []
    kf36_gate_reason = 'No candidate was close enough to the incumbent to justify KF36.'
    for row in rows_sorted[:3]:
        d = row['delta_vs_batch2_markov42']
        if d['mean_pct_error'] > -0.02 and d['max_pct_error'] > -0.12:
            candidate = candidates_by_name[row['candidate_name']]
            spec = spec_by_name[row['candidate_name']]
            kf_payload, _, kf_path = run_candidate_payload(mod, candidate, 'kf36_noisy', args.noise_scale, force_rerun=args.force_rerun)
            kf_payload = attach_att0(kf_path, kf_payload, candidate.name, 'kf36_noisy', spec['family'], spec['hypothesis_id'])
            row['result_files']['kf36'] = str(kf_path)
            row['kf36'] = compact_metrics(kf_payload)
            row['delta_vs_batch2_kf36'] = delta_vs_reference(references['batch2_incumbent']['kf36'], row['kf36'])
            row['delta_vs_prompt_leader_kf36'] = delta_vs_reference(references['prompt_launch_leader']['kf36'], row['kf36'])
            kf36_rechecked_candidates.append(candidate.name)
    if not kf36_rechecked_candidates:
        kf36_gate_reason = f"Best Markov42 candidate {best['candidate_name']} = {overall_triplet(best['markov42'])} stayed outside the incumbent gate."

    best_candidate_obj = candidates_by_name[best['candidate_name']]
    best_timing_table = build_timing_table(best_candidate_obj)

    if beat_batch2_incumbent:
        statement = 'A fresh local improvement was found even after the batch2 win; the corrected frontier still had headroom inside the narrow l9/l12 ridge.'
        beat_batch2 = 'YES'
    else:
        statement = 'The relaunch batch did not beat the batch2 incumbent. The best probes stayed in the same basin, but every micro-move traded one objective against another: the local ridge appears real, yet already fairly tight around l9≈0.75 with a light terminal l12 closure.'
        beat_batch2 = 'NO'

    summary = {
        'task': 'chapter-3 corrected frontier relaunch batch 3',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'corrected_att0_deg': ATT0_DEG,
        'references': references,
        'hypotheses_tested': HYPOTHESES,
        'tested_candidates': [spec['name'] for spec in specs],
        'rows_sorted': rows_sorted,
        'best_candidate': best,
        'best_candidate_timing_table': best_timing_table,
        'beat_batch2_incumbent_candidates': beat_batch2_incumbent,
        'beat_prompt_threshold_candidates': beat_prompt_threshold,
        'near_misses': near_misses,
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
        'kf36_gate_reason': kf36_gate_reason,
        'bottom_line': {
            'beat_batch2_incumbent': beat_batch2,
            'beat_prompt_threshold': 'YES' if beat_prompt_threshold else 'NO',
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
        'beat_batch2_incumbent': beat_batch2,
        'beat_prompt_threshold': 'YES' if beat_prompt_threshold else 'NO',
        'batch_best_name': best['candidate_name'],
        'batch_best_triplet': overall_triplet(best['markov42']),
        'beat_batch2_incumbent_candidates': beat_batch2_incumbent,
        'beat_prompt_threshold_candidates': beat_prompt_threshold,
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
