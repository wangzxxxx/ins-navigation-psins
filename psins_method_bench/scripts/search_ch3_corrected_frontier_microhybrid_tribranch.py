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
from search_ch3_12pos_closedloop_local_insertions import StepSpec, build_closedloop_candidate, run_candidate_payload
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate, render_action
from search_ch3_entry_conditioned_relay_family import (
    NOISE_SCALE,
    REPORT_DATE,
    l8_xpair,
    l9_ypair_neg,
    l10_unified_core,
    l11_y10x0back2_core,
    merge_insertions,
)

ATT0_DEG = [0.0, 0.0, 0.0]
REPORT_PATH = REPORTS_DIR / f'psins_ch3_corrected_frontier_microhybrid_tribranch_{REPORT_DATE}.md'
SUMMARY_PATH = RESULTS_DIR / f'ch3_corrected_frontier_microhybrid_tribranch_{REPORT_DATE}.json'

REFERENCE_FILES = {
    'leader': {
        'label': 'current corrected leader / relay_l12y0p5_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l12y0p5_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_relay_l12y0p5_on_entry_shared_noise0p08_param_errors.json',
    },
    'companion': {
        'label': 'closest companion / entryrelay_l8x1_l9y0p75_unifiedcore',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_entryrelay_l8x1_l9y0p75_unifiedcore_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_entryrelay_l8x1_l9y0p75_unifiedcore_shared_noise0p08_param_errors.json',
    },
    'splitridge': {
        'label': 'splitridge improved runner-up / splitridge_l7neg1p5_l12y1_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_splitridge_l7neg1p5_l12y1_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_splitridge_l7neg1p5_l12y1_on_entry_shared_noise0p08_param_errors.json',
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
        'id': 'H1',
        'family': 'leader_plus_splitridge_corridor_doseup',
        'summary': 'Keep the current leader backbone fixed and import only the splitridge corridor dose-up signal (anchor7 = -1.5 s) to see whether the saved splitridge median mechanism transfers when the leader terminal closure stays light.',
        'candidate_names': [
            'hybrid_l7neg1p5_l12y0p5_on_entry',
        ],
    },
    {
        'id': 'H2',
        'family': 'tribranch_microhybrid_on_companion_backbone',
        'summary': 'Fuse the three competitive branches in the narrowest way: companion backbone (l8 x1 + l9 y0.75 + unified core), plus a splitridge corridor conditioner, plus the leader terminal micro-closure.',
        'candidate_names': [
            'trihybrid_l7neg1_l9y0p75_l12y0p5_on_entry',
            'trihybrid_l7neg1p5_l9y0p75_l12y0p5_on_entry',
        ],
    },
    {
        'id': 'H3',
        'family': 'tribranch_terminal_pull_toward_splitridge',
        'summary': 'On the same tri-branch hybrid, increase only the terminal closure from 0.5 s to 0.75 s to test whether a small pull toward the splitridge terminal dose can recover median without collapsing mean/max.',
        'candidate_names': [
            'trihybrid_l7neg1p5_l9y0p75_l12y0p75_on_entry',
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
    extra['comparison_mode'] = 'corrected_frontier_microhybrid_tribranch'
    extra['candidate_registry_key'] = candidate_name
    extra['method_key'] = method_key
    extra['family'] = family
    extra['hypothesis_id'] = hypothesis_id
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload



def closed_pair(anchor: int, kind: str, angle_deg: int, dwell_s: float, label: str, rot_s: float = 5.0) -> dict[int, list[StepSpec]]:
    return {
        anchor: [
            StepSpec(kind=kind, angle_deg=angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_out', label=f'{label}_out'),
            StepSpec(kind=kind, angle_deg=-angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_return', label=f'{label}_return'),
        ]
    }



def candidate_specs() -> list[dict[str, Any]]:
    leader_base = merge_insertions(
        l8_xpair(1.0, 'l8_x1'),
        l9_ypair_neg(1.0, 'l9_ypair_neg1'),
        l10_unified_core(),
        l11_y10x0back2_core(),
    )
    companion_base = merge_insertions(
        l8_xpair(1.0, 'l8_x1'),
        l9_ypair_neg(0.75, 'l9_ypair_neg0p75'),
        l10_unified_core(),
        l11_y10x0back2_core(),
    )
    return [
        {
            'name': 'hybrid_l7neg1p5_l12y0p5_on_entry',
            'family': 'leader_plus_splitridge_corridor_doseup',
            'hypothesis_id': 'H1',
            'rationale': 'Direct corridor dose-up on the current leader: keep l9 = 1.0 and l12 = 0.5 fixed, raise only the imported splitridge corridor from the previously tested 1.0 to 1.5 s.',
            'insertions': merge_insertions(leader_base, closed_pair(7, 'outer', -90, 1.5, 'l7_xneg1p5'), closed_pair(12, 'inner', -90, 0.5, 'l12_yneg0p5')),
        },
        {
            'name': 'trihybrid_l7neg1_l9y0p75_l12y0p5_on_entry',
            'family': 'tribranch_microhybrid_on_companion_backbone',
            'hypothesis_id': 'H2',
            'rationale': 'Minimum tri-branch fusion: companion backbone plus a moderate splitridge corridor (1.0 s) plus the leader terminal closure (0.5 s).',
            'insertions': merge_insertions(companion_base, closed_pair(7, 'outer', -90, 1.0, 'l7_xneg1'), closed_pair(12, 'inner', -90, 0.5, 'l12_yneg0p5')),
        },
        {
            'name': 'trihybrid_l7neg1p5_l9y0p75_l12y0p5_on_entry',
            'family': 'tribranch_microhybrid_on_companion_backbone',
            'hypothesis_id': 'H2',
            'rationale': 'Stronger tri-branch fusion: same companion backbone and leader terminal closure, but import the best splitridge corridor dose (1.5 s).',
            'insertions': merge_insertions(companion_base, closed_pair(7, 'outer', -90, 1.5, 'l7_xneg1p5'), closed_pair(12, 'inner', -90, 0.5, 'l12_yneg0p5')),
        },
        {
            'name': 'trihybrid_l7neg1p5_l9y0p75_l12y0p75_on_entry',
            'family': 'tribranch_terminal_pull_toward_splitridge',
            'hypothesis_id': 'H3',
            'rationale': 'Same strong tri-branch fusion, but pull the terminal closure upward from 0.5 s to 0.75 s as a midpoint toward the splitridge terminal dose.',
            'insertions': merge_insertions(companion_base, closed_pair(7, 'outer', -90, 1.5, 'l7_xneg1p5'), closed_pair(12, 'inner', -90, 0.75, 'l12_yneg0p75')),
        },
    ]



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
    lines.append('# Chapter-3 corrected frontier micro-hybrid tri-branch batch')
    lines.append('')
    lines.append('## 1. Search target')
    lines.append('')
    lines.append('- Hard basis enforced: **att0 = (0, 0, 0)** only.')
    lines.append('- Hard legality enforced: **real dual-axis motor sequence only**, with exact closure before the faithful backbone resumes.')
    lines.append('- This batch deliberately stayed **inside the existing corrected frontier** rather than opening any fresh family.')
    lines.append(f"- Governing frontier at launch: leader **{refs['leader']['markov42_triplet']}**, companion **{refs['companion']['markov42_triplet']}**, splitridge runner-up **{refs['splitridge']['markov42_triplet']}**.")
    lines.append('- Design logic: spend one narrow batch on the only still-unspent tri-branch question — can the splitridge corridor help once it is grafted onto the stronger leader/companion backbone instead of the weaker splitbookend baseline?')
    lines.append('')
    lines.append('## 2. Micro-hybrid hypotheses tested')
    lines.append('')
    for item in summary['hypotheses_tested']:
        lines.append(f"- **{item['id']} · {item['family']}** — {item['summary']}")
        lines.append(f"  - tested: {', '.join(item['candidate_names'])}")
    lines.append('')
    lines.append('## 3. Markov42 landed batch')
    lines.append('')
    lines.append('| rank | candidate | family | mean | median | max | Δmean vs leader | Δmedian vs leader | Δmax vs leader | Δmean vs splitridge | Δmax vs splitridge | max driver |')
    lines.append('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for idx, row in enumerate(summary['rows_sorted'], start=1):
        m = row['markov42']['overall']
        dl = row['delta_vs_leader_markov42']
        ds = row['delta_vs_splitridge_markov42']
        driver = row['markov42']['max_driver']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {dl['mean_pct_error']:+.3f} | {dl['median_pct_error']:+.3f} | {dl['max_pct_error']:+.3f} | {ds['mean_pct_error']:+.3f} | {ds['max_pct_error']:+.3f} | {driver['name']} {driver['pct_error']:.3f} |"
        )
    lines.append('')
    lines.append('## 4. Best landing in this batch')
    lines.append('')
    lines.append(f"- **Batch-best candidate:** `{best['candidate_name']}`")
    lines.append(f"- Rationale: {best['rationale']}")
    lines.append(f"- **Markov42:** **{overall_triplet(best['markov42'])}**")
    if best.get('kf36') is not None:
        lines.append(f"- **KF36:** **{overall_triplet(best['kf36'])}**")
    lines.append(f"- vs leader: Δmean **{best['delta_vs_leader_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_leader_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_leader_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- vs companion: Δmean **{best['delta_vs_companion_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_companion_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_companion_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- vs splitridge runner-up: Δmean **{best['delta_vs_splitridge_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_splitridge_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_splitridge_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- max driver: **{best['markov42']['max_driver']['name']} = {best['markov42']['max_driver']['pct_error']:.3f}%**")
    lines.append('')
    lines.append('## 5. KF36 recheck gate')
    lines.append('')
    if summary['kf36_rechecked_candidates']:
        lines.append(f"- Rechecked candidates: **{', '.join(summary['kf36_rechecked_candidates'])}**")
        for row in summary['rows_sorted']:
            if row.get('kf36') is not None:
                lines.append(f"  - `{row['candidate_name']}` → Markov42 **{overall_triplet(row['markov42'])}**, KF36 **{overall_triplet(row['kf36'])}**")
    else:
        lines.append('- Rechecked candidates: **none**')
        lines.append(f"- Gate reason: {summary['kf36_gate_reason']}")
    lines.append('')
    lines.append('## 6. Exact legal motor / timing table for the batch-best candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for idx, (row, action, face) in enumerate(zip(best['all_rows'], best['all_actions'], best['all_faces']), start=1):
        lines.append(
            f"| {idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 7. Continuity proof for the batch-best candidate')
    lines.append('')
    for item in best['continuity_checks']:
        before = item['state_before_insertion']
        after = item['state_after_insertion']
        lines.append(f"- anchor {item['anchor_id']}: closure_ok = **{'yes' if item['closure_ok'] else 'no'}**")
        lines.append(f"  - before: beta={before['beta_deg']}°, outer_axis={before['outer_axis_body']}, face={before['face_name']}")
        lines.append(f"  - after : beta={after['beta_deg']}°, outer_axis={after['outer_axis_body']}, face={after['face_name']}")
        nxt = item.get('next_base_action_preview')
        if nxt is not None:
            lines.append(f"  - next base action remains legal as `{nxt['kind']}` {nxt['motor_angle_deg']:+d}° with effective axis {nxt['effective_body_axis']}")
        else:
            lines.append('  - insertion lands at the terminal end of the base path, so there is no further base action to resume.')
    lines.append('')
    lines.append('## 8. Bottom line')
    lines.append('')
    lines.append(f"- **Did any candidate beat the current leader on both mean and max?** **{summary['bottom_line']['beat_leader_on_mean_and_max']}**")
    lines.append(f"- **Did any candidate beat the companion on both mean and max?** **{summary['bottom_line']['beat_companion_on_mean_and_max']}**")
    lines.append(f"- **Did any candidate beat the splitridge runner-up on both mean and max?** **{summary['bottom_line']['beat_splitridge_on_mean_and_max']}**")
    lines.append(f"- Batch-best point: **{summary['bottom_line']['batch_best_triplet']}** (`{summary['bottom_line']['batch_best_name']}`)")
    lines.append(f"- Scientific read: **{summary['bottom_line']['statement']}**")
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
            'all_rows': candidate.all_rows,
            'all_actions': candidate.all_actions,
            'all_faces': candidate.all_faces,
            'continuity_checks': candidate.continuity_checks,
        }
        row['delta_vs_leader_markov42'] = delta_vs_reference(references['leader']['markov42'], row['markov42'])
        row['delta_vs_companion_markov42'] = delta_vs_reference(references['companion']['markov42'], row['markov42'])
        row['delta_vs_splitridge_markov42'] = delta_vs_reference(references['splitridge']['markov42'], row['markov42'])
        row['delta_vs_faithful12_markov42'] = delta_vs_reference(references['faithful12']['markov42'], row['markov42'])
        row['delta_vs_default18_markov42'] = delta_vs_reference(references['default18']['markov42'], row['markov42'])
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: (r['markov42']['overall']['mean_pct_error'], r['markov42']['overall']['max_pct_error'], r['markov42']['overall']['median_pct_error']))
    best = rows_sorted[0]

    beat_leader = [
        row['candidate_name']
        for row in rows_sorted
        if row['delta_vs_leader_markov42']['mean_pct_error'] > 0 and row['delta_vs_leader_markov42']['max_pct_error'] > 0
    ]
    beat_companion = [
        row['candidate_name']
        for row in rows_sorted
        if row['delta_vs_companion_markov42']['mean_pct_error'] > 0 and row['delta_vs_companion_markov42']['max_pct_error'] > 0
    ]
    beat_splitridge = [
        row['candidate_name']
        for row in rows_sorted
        if row['delta_vs_splitridge_markov42']['mean_pct_error'] > 0 and row['delta_vs_splitridge_markov42']['max_pct_error'] > 0
    ]

    near_misses = [
        row for row in rows_sorted
        if not (row['delta_vs_leader_markov42']['mean_pct_error'] > 0 and row['delta_vs_leader_markov42']['max_pct_error'] > 0)
        and (
            row['delta_vs_splitridge_markov42']['mean_pct_error'] > 0
            or row['delta_vs_companion_markov42']['max_pct_error'] > 0
            or abs(row['delta_vs_leader_markov42']['max_pct_error']) <= 0.25
        )
    ][:4]

    kf36_rechecked_candidates: list[str] = []
    kf36_gate_reason = 'No candidate was close enough to the current leader to justify KF36.'
    for row in rows_sorted[:2]:
        d = row['delta_vs_leader_markov42']
        ds = row['delta_vs_splitridge_markov42']
        should_rerun = (
            (d['mean_pct_error'] > -0.03 and d['max_pct_error'] > -0.25)
            or (ds['mean_pct_error'] > 0 and ds['max_pct_error'] > 0)
        )
        if should_rerun:
            candidate = candidates_by_name[row['candidate_name']]
            spec = spec_by_name[row['candidate_name']]
            kf_payload, _, kf_path = run_candidate_payload(mod, candidate, 'kf36_noisy', args.noise_scale, force_rerun=args.force_rerun)
            kf_payload = attach_att0(kf_path, kf_payload, candidate.name, 'kf36_noisy', spec['family'], spec['hypothesis_id'])
            row['result_files']['kf36'] = str(kf_path)
            row['kf36'] = compact_metrics(kf_payload)
            row['delta_vs_leader_kf36'] = delta_vs_reference(references['leader']['kf36'], row['kf36'])
            row['delta_vs_companion_kf36'] = delta_vs_reference(references['companion']['kf36'], row['kf36'])
            row['delta_vs_splitridge_kf36'] = delta_vs_reference(references['splitridge']['kf36'], row['kf36'])
            kf36_rechecked_candidates.append(candidate.name)
    if not kf36_rechecked_candidates:
        kf36_gate_reason = f"Best Markov42 candidate {best['candidate_name']} = {overall_triplet(best['markov42'])} stayed outside the near-leader gate."

    if beat_leader:
        statement = 'A new corrected-basis winner emerged directly from the accepted frontier: at least one tri-branch micro-hybrid lowered both mean and max below the current leader.'
    elif beat_companion or beat_splitridge:
        statement = 'No candidate dethroned the current leader, but the corridor graft was still informative: the leader/companion basin remained stronger on mean/max, while the splitridge corridor could not be transferred cleanly enough to create a new balanced winner.'
    else:
        statement = 'The frontier stayed stable under this narrow tri-branch probe. Importing the splitridge corridor onto the stronger leader/companion backbone consistently paid too much in mean/max, so the corridor appears branch-specific rather than a portable improvement lever.'

    summary = {
        'task': 'chapter-3 corrected frontier micro-hybrid tri-branch batch',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'corrected_att0_deg': ATT0_DEG,
        'references': references,
        'hypotheses_tested': HYPOTHESES,
        'tested_candidates': [spec['name'] for spec in specs],
        'rows_sorted': rows_sorted,
        'best_candidate': best,
        'beat_leader_candidates': beat_leader,
        'beat_companion_candidates': beat_companion,
        'beat_splitridge_candidates': beat_splitridge,
        'near_misses': near_misses,
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
        'kf36_gate_reason': kf36_gate_reason,
        'bottom_line': {
            'beat_leader_on_mean_and_max': 'YES' if beat_leader else 'NO',
            'beat_companion_on_mean_and_max': 'YES' if beat_companion else 'NO',
            'beat_splitridge_on_mean_and_max': 'YES' if beat_splitridge else 'NO',
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
        'beat_leader_candidates': beat_leader,
        'beat_companion_candidates': beat_companion,
        'beat_splitridge_candidates': beat_splitridge,
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
