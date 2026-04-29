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
from search_ch3_12pos_closedloop_local_insertions import StepSpec, build_closedloop_candidate, render_action, run_candidate_payload
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate
from search_ch3_entry_conditioned_relay_family import candidate_specs as entryrelay_candidate_specs

NOISE_SCALE = 0.08
REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
ATT0_DEG = [0.0, 0.0, 0.0]

REPORT_PATH = REPORTS_DIR / f'psins_ch3_corrected_frontier_ridge_splitbookend_{REPORT_DATE}.md'
SUMMARY_PATH = RESULTS_DIR / f'ch3_corrected_frontier_ridge_splitbookend_{REPORT_DATE}.json'

REFERENCE_FILES = {
    'entry_frontier': {
        'label': 'corrected incumbent entry-conditioned relay',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json',
    },
    'splitbookend_runnerup': {
        'label': 'corrected splitbookend runner-up',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_splitbookend_l7neg1_l12y1_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_splitbookend_l7neg1_l12y1_on_entry_shared_noise0p08_param_errors.json',
    },
    'xboundary_branch': {
        'label': 'corrected x-boundary side branch',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_xboundary_l2_pos1_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_xboundary_l2_pos1_on_entry_shared_noise0p08_param_errors.json',
    },
    'frontz_branch': {
        'label': 'corrected front-z side branch',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_frontz_l4_neg1_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_frontz_l4_neg1_on_entry_shared_noise0p08_param_errors.json',
    },
}

HYPOTHESES = [
    {
        'id': 'H1',
        'family': 'splitbookend_terminal_y_ridge_on_corrected_entry',
        'summary': 'Hold the successful anchor7 negative corridor bookend fixed and locally refine only the anchor12 terminal y reclosure dwell around the current y1 splitbookend runner-up, to test whether the strong median gain can be retained while recovering mean/max.',
        'selected': True,
        'tested': True,
        'candidate_names': [
            'splitridge_l7neg1_l12y0p5_on_entry',
            'splitridge_l7neg1_l12y0p75_on_entry',
            'splitridge_l7neg1_l12y1p25_on_entry',
        ],
    },
    {
        'id': 'H2',
        'family': 'splitbookend_corridor_dose_microcheck_on_corrected_entry',
        'summary': 'Hold the terminal y1 closure fixed and micro-perturb only the anchor7 negative corridor dwell to see whether the runner-up sits on a narrow or broad corridor-dose ridge.',
        'selected': True,
        'tested': True,
        'candidate_names': [
            'splitridge_l7neg0p5_l12y1_on_entry',
            'splitridge_l7neg1p5_l12y1_on_entry',
        ],
    },
]

SELECTED_FAMILIES = [
    'splitbookend terminal-y ridge on corrected entry backbone',
    'splitbookend corridor-dose microcheck on corrected entry backbone',
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
            'Ka2_x': float(payload['param_errors']['Ka2_x']['pct_error']),
        },
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
    extra['comparison_mode'] = 'corrected_frontier_ridge_splitbookend'
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



def merge_insertions(*dicts: dict[int, list[StepSpec]]) -> dict[int, list[StepSpec]]:
    out: dict[int, list[StepSpec]] = {}
    for d in dicts:
        for k, v in d.items():
            out.setdefault(k, []).extend(v)
    return out



def make_spec(name: str, family: str, hypothesis_id: str, rationale: str, insertions: dict[int, list[StepSpec]]) -> dict[str, Any]:
    return {
        'name': name,
        'family': family,
        'hypothesis_id': hypothesis_id,
        'rationale': rationale,
        'insertions': insertions,
    }



def candidate_specs() -> list[dict[str, Any]]:
    entry_spec = next(item for item in entryrelay_candidate_specs() if item['name'] == 'entryrelay_l8x1_l9y1_unifiedcore')
    base_insertions = {k: list(v) for k, v in entry_spec['insertions'].items()}
    return [
        make_spec(
            'splitridge_l7neg1_l12y0p5_on_entry',
            'splitbookend_terminal_y_ridge_on_corrected_entry',
            'H1',
            'Reduce only the anchor12 terminal y dwell from 1.0 s to 0.5 s while keeping the successful anchor7 negative corridor conditioner fixed.',
            merge_insertions(base_insertions, closed_pair(7, 'outer', -90, 1.0, 'l7_xneg1'), closed_pair(12, 'inner', -90, 0.5, 'l12_yneg0p5')),
        ),
        make_spec(
            'splitridge_l7neg1_l12y0p75_on_entry',
            'splitbookend_terminal_y_ridge_on_corrected_entry',
            'H1',
            'Midpoint terminal-y refinement between the current y1 runner-up and the lighter y0.5 closure.',
            merge_insertions(base_insertions, closed_pair(7, 'outer', -90, 1.0, 'l7_xneg1'), closed_pair(12, 'inner', -90, 0.75, 'l12_yneg0p75')),
        ),
        make_spec(
            'splitridge_l7neg1_l12y1p25_on_entry',
            'splitbookend_terminal_y_ridge_on_corrected_entry',
            'H1',
            'Slight terminal-y dose-up around the current runner-up, to verify whether y1 is already near the useful edge or still under-driven.',
            merge_insertions(base_insertions, closed_pair(7, 'outer', -90, 1.0, 'l7_xneg1'), closed_pair(12, 'inner', -90, 1.25, 'l12_yneg1p25')),
        ),
        make_spec(
            'splitridge_l7neg0p5_l12y1_on_entry',
            'splitbookend_corridor_dose_microcheck_on_corrected_entry',
            'H2',
            'Keep the terminal y1 closure fixed and lighten only the anchor7 negative corridor dwell to 0.5 s.',
            merge_insertions(base_insertions, closed_pair(7, 'outer', -90, 0.5, 'l7_xneg0p5'), closed_pair(12, 'inner', -90, 1.0, 'l12_yneg1')),
        ),
        make_spec(
            'splitridge_l7neg1p5_l12y1_on_entry',
            'splitbookend_corridor_dose_microcheck_on_corrected_entry',
            'H2',
            'Keep the terminal y1 closure fixed and slightly strengthen the anchor7 negative corridor dwell to 1.5 s.',
            merge_insertions(base_insertions, closed_pair(7, 'outer', -90, 1.5, 'l7_xneg1p5'), closed_pair(12, 'inner', -90, 1.0, 'l12_yneg1')),
        ),
    ]



def pick_family_best(rows: list[dict[str, Any]], family_name: str) -> dict[str, Any]:
    family_rows = [r for r in rows if r['family'] == family_name]
    return min(family_rows, key=lambda r: (r['markov42']['overall']['mean_pct_error'], r['markov42']['overall']['max_pct_error']))



def render_report(summary: dict[str, Any]) -> str:
    best = summary['best_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 corrected frontier ridge batch: splitbookend local refinement')
    lines.append('')
    lines.append('## 1. Search target')
    lines.append('')
    lines.append('- Hard basis enforced: **att0 = (0, 0, 0)** only.')
    lines.append('- This batch does **not** open another fresh family. It performs a local ridge refinement inside the already-competitive corrected splitbookend branch.')
    lines.append('- Focus: keep the corrected incumbent entry-relay backbone fixed, then perturb only the splitbookend corridor/terminal micro-doses around the runner-up `splitbookend_l7neg1_l12y1_on_entry`.')
    lines.append('- Corrected frontier to compare against:')
    lines.append(f"  - corrected incumbent entry-conditioned relay: **{summary['references']['entry_frontier']['markov42_triplet']}**")
    lines.append(f"  - corrected splitbookend runner-up: **{summary['references']['splitbookend_runnerup']['markov42_triplet']}**")
    lines.append(f"  - corrected x-boundary side branch: **{summary['references']['xboundary_branch']['markov42_triplet']}**")
    lines.append(f"  - corrected front-z side branch: **{summary['references']['frontz_branch']['markov42_triplet']}**")
    lines.append('')
    lines.append('## 2. Ridge hypotheses spent in this batch')
    lines.append('')
    for item in summary['hypotheses_considered']:
        flags = []
        if item['selected']:
            flags.append('selected')
        if item['tested']:
            flags.append('tested')
        flag_text = ', '.join(flags) if flags else 'listed only'
        lines.append(f"- **{item['id']} · {item['family']}** ({flag_text}) — {item['summary']}")
    lines.append('')
    lines.append('## 3. Corrected Markov42 landed batch')
    lines.append('')
    lines.append('| rank | candidate | family | mean | median | max | Δmean vs splitbookend | Δmedian vs splitbookend | Δmax vs splitbookend | Δmean vs entry | Δmax vs entry | Δmean vs frontz | Δmax vs frontz |')
    lines.append('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(summary['rows_sorted'], start=1):
        m = row['markov42']['overall']
        ds = row['delta_vs_splitbookend_markov42']
        de = row['delta_vs_entry_markov42']
        df = row['delta_vs_frontz_markov42']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {ds['mean_pct_error']:+.3f} | {ds['median_pct_error']:+.3f} | {ds['max_pct_error']:+.3f} | {de['mean_pct_error']:+.3f} | {de['max_pct_error']:+.3f} | {df['mean_pct_error']:+.3f} | {df['max_pct_error']:+.3f} |"
        )
    lines.append('')
    lines.append('## 4. Best corrected landing in this ridge batch')
    lines.append('')
    lines.append(f"- **Best candidate:** `{best['candidate_name']}`")
    lines.append(f"- **Family:** `{best['family']}`")
    lines.append(f"- **Markov42:** **{overall_triplet(best['markov42'])}**")
    if best.get('kf36') is not None:
        lines.append(f"- **KF36 recheck:** **{overall_triplet(best['kf36'])}**")
    lines.append(f"- vs corrected splitbookend runner-up: Δmean **{best['delta_vs_splitbookend_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_splitbookend_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_splitbookend_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- vs corrected incumbent entry relay: Δmean **{best['delta_vs_entry_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_entry_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_entry_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- vs corrected x-boundary side branch: Δmean **{best['delta_vs_xboundary_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_xboundary_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_xboundary_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- vs corrected front-z side branch: Δmean **{best['delta_vs_frontz_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_frontz_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_frontz_markov42']['max_pct_error']:+.3f}**")
    lines.append('')
    lines.append('## 5. Exact legal motor / timing table for the best candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for idx, (row, action, face) in enumerate(zip(best['all_rows'], best['all_actions'], best['all_faces']), start=1):
        lines.append(
            f"| {idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 6. Continuity proof for the best candidate')
    lines.append('')
    for item in best['continuity_checks']:
        before = item['state_before_insertion']
        after = item['state_after_insertion']
        nxt = item['next_base_action_preview']
        lines.append(f"- anchor {item['anchor_id']}: closure_ok = **{'yes' if item['closure_ok'] else 'no'}**")
        lines.append(f"  - before: beta={before['beta_deg']:.0f}°, outer_axis={before['outer_axis_body']}, face={before['face_name']}")
        lines.append(f"  - after : beta={after['beta_deg']:.0f}°, outer_axis={after['outer_axis_body']}, face={after['face_name']}")
        if nxt is not None:
            lines.append(f"  - next base action remains legal as `{nxt['kind']}` {nxt['motor_angle_deg']:+.0f}° with effective axis {nxt['effective_body_axis']}")
        else:
            lines.append("  - insertion lands at the terminal end of the base path, so there is no further base action to resume.")
    lines.append('')
    lines.append('## 7. KF36 recheck gate')
    lines.append('')
    if summary['kf36_rechecked_candidates']:
        lines.append(f"- Triggered candidates: **{', '.join(summary['kf36_rechecked_candidates'])}**")
        for row in summary['rows_sorted']:
            if row.get('kf36') is not None:
                lines.append(f"  - `{row['candidate_name']}` → Markov42 **{overall_triplet(row['markov42'])}**, KF36 **{overall_triplet(row['kf36'])}**")
    else:
        lines.append('- Triggered candidates: **none**')
        lines.append(f"- Gate result: {summary['kf36_gate_reason']}")
    lines.append('')
    lines.append('## 8. Bottom line')
    lines.append('')
    lines.append(f"- **Did this ridge batch beat the corrected incumbent 1.084 / 0.617 / 5.137 on mean and max?** **{summary['bottom_line']['beat_corrected_frontier']}**")
    lines.append(f"- **Did it produce a stronger balanced branch than splitbookend on mean and max?** **{summary['bottom_line']['beat_splitbookend_runnerup']}**")
    lines.append(f"- **Did it at least dominate the x-boundary side branch on mean and max?** **{summary['bottom_line']['beat_xboundary_branch']}**")
    lines.append(f"- Best new ridge landing was `{best['candidate_name']}` = **{overall_triplet(best['markov42'])}**" + (f" (KF36 **{overall_triplet(best['kf36'])}**)" if best.get('kf36') is not None else '') + '.')
    lines.append(f"- Scientific read: {summary['bottom_line']['statement']}")
    return '\n'.join(lines) + '\n'



def main() -> None:
    args = parse_args()
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
            'files': {
                'markov42': str(info['markov42']),
                'kf36': str(info['kf36']),
            },
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
            'result_files': {
                'markov42': str(markov_path),
            },
            'markov42': compact_metrics(markov_payload),
            'all_rows': candidate.all_rows,
            'all_actions': candidate.all_actions,
            'all_faces': candidate.all_faces,
            'continuity_checks': candidate.continuity_checks,
        }
        row['delta_vs_entry_markov42'] = delta_vs_reference(references['entry_frontier']['markov42'], row['markov42'])
        row['delta_vs_splitbookend_markov42'] = delta_vs_reference(references['splitbookend_runnerup']['markov42'], row['markov42'])
        row['delta_vs_xboundary_markov42'] = delta_vs_reference(references['xboundary_branch']['markov42'], row['markov42'])
        row['delta_vs_frontz_markov42'] = delta_vs_reference(references['frontz_branch']['markov42'], row['markov42'])
        rows.append(row)

    family_best_names = {
        pick_family_best(rows, 'splitbookend_terminal_y_ridge_on_corrected_entry')['candidate_name'],
        pick_family_best(rows, 'splitbookend_corridor_dose_microcheck_on_corrected_entry')['candidate_name'],
    }
    kf36_rechecked_candidates: list[str] = []
    kf36_gate_reason = 'No candidate reached the competitiveness gate.'
    for row in rows:
        markov = row['markov42']['overall']
        near_split = markov['max_pct_error'] <= references['splitbookend_runnerup']['markov42']['overall']['max_pct_error'] + 0.25 and markov['mean_pct_error'] <= references['splitbookend_runnerup']['markov42']['overall']['mean_pct_error'] + 0.10
        near_frontz = markov['max_pct_error'] <= references['frontz_branch']['markov42']['overall']['max_pct_error'] + 0.25 and markov['mean_pct_error'] <= references['frontz_branch']['markov42']['overall']['mean_pct_error'] + 0.10
        if row['candidate_name'] in family_best_names and (near_split or near_frontz):
            candidate = candidates_by_name[row['candidate_name']]
            spec = spec_by_name[row['candidate_name']]
            kf_payload, _, kf_path = run_candidate_payload(mod, candidate, 'kf36_noisy', args.noise_scale, force_rerun=args.force_rerun)
            kf_payload = attach_att0(kf_path, kf_payload, candidate.name, 'kf36_noisy', spec['family'], spec['hypothesis_id'])
            row['result_files']['kf36'] = str(kf_path)
            row['kf36'] = compact_metrics(kf_payload)
            row['delta_vs_entry_kf36'] = delta_vs_reference(references['entry_frontier']['kf36'], row['kf36'])
            row['delta_vs_splitbookend_kf36'] = delta_vs_reference(references['splitbookend_runnerup']['kf36'], row['kf36'])
            row['delta_vs_xboundary_kf36'] = delta_vs_reference(references['xboundary_branch']['kf36'], row['kf36'])
            row['delta_vs_frontz_kf36'] = delta_vs_reference(references['frontz_branch']['kf36'], row['kf36'])
            kf36_rechecked_candidates.append(candidate.name)
    if not kf36_rechecked_candidates:
        best_tmp = min(rows, key=lambda r: (r['markov42']['overall']['mean_pct_error'], r['markov42']['overall']['max_pct_error']))
        kf36_gate_reason = f"Best Markov42 candidate {best_tmp['candidate_name']} = {overall_triplet(best_tmp['markov42'])} still was not close enough to the splitbookend/front-z ridge zone to justify KF36."

    rows_sorted = sorted(rows, key=lambda r: (r['markov42']['overall']['mean_pct_error'], r['markov42']['overall']['max_pct_error'], r['markov42']['overall']['median_pct_error']))
    best = rows_sorted[0]

    beat_corrected_frontier = (
        best['markov42']['overall']['mean_pct_error'] < references['entry_frontier']['markov42']['overall']['mean_pct_error']
        and best['markov42']['overall']['max_pct_error'] < references['entry_frontier']['markov42']['overall']['max_pct_error']
    )
    beat_splitbookend_runnerup = (
        best['markov42']['overall']['mean_pct_error'] < references['splitbookend_runnerup']['markov42']['overall']['mean_pct_error']
        and best['markov42']['overall']['max_pct_error'] < references['splitbookend_runnerup']['markov42']['overall']['max_pct_error']
    )
    beat_xboundary_branch = (
        best['markov42']['overall']['mean_pct_error'] < references['xboundary_branch']['markov42']['overall']['mean_pct_error']
        and best['markov42']['overall']['max_pct_error'] < references['xboundary_branch']['markov42']['overall']['max_pct_error']
    )
    beats_frontz_branch = (
        best['markov42']['overall']['mean_pct_error'] < references['frontz_branch']['markov42']['overall']['mean_pct_error']
        and best['markov42']['overall']['max_pct_error'] < references['frontz_branch']['markov42']['overall']['max_pct_error']
    )

    family_best = {
        'splitbookend_terminal_y_ridge_on_corrected_entry': pick_family_best(rows, 'splitbookend_terminal_y_ridge_on_corrected_entry'),
        'splitbookend_corridor_dose_microcheck_on_corrected_entry': pick_family_best(rows, 'splitbookend_corridor_dose_microcheck_on_corrected_entry'),
    }

    if beat_corrected_frontier:
        statement = 'A new corrected-basis takeover was found inside the splitbookend ridge: the best local point beats the incumbent entry relay on both mean and max.'
    elif beat_splitbookend_runnerup and beats_frontz_branch:
        statement = 'The splitbookend ridge did not dethrone the incumbent, but it did produce a materially stronger balanced branch than the previous splitbookend/front-z side points.'
    elif beat_splitbookend_runnerup:
        statement = 'The splitbookend ridge improved the runner-up branch on mean and max, but not enough to overtake the incumbent or the best max-oriented front-z side branch.'
    else:
        statement = 'The splitbookend runner-up proved locally narrow under corrected att0: neither lighter/heavier terminal-y closure nor small corridor-dose shifts produced a better mean/max point than the saved splitbookend, and no candidate overtook the incumbent frontier.'

    summary = {
        'task': 'chapter-3 corrected frontier ridge splitbookend batch',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'corrected_att0_deg': ATT0_DEG,
        'hypotheses_considered': HYPOTHESES,
        'selected_families': SELECTED_FAMILIES,
        'tested_candidates': [spec['name'] for spec in specs],
        'references': references,
        'rows_sorted': rows_sorted,
        'family_best_candidates': family_best,
        'best_candidate': best,
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
        'kf36_gate_reason': kf36_gate_reason,
        'bottom_line': {
            'beat_corrected_frontier': 'YES' if beat_corrected_frontier else 'NO',
            'beat_splitbookend_runnerup': 'YES' if beat_splitbookend_runnerup else 'NO',
            'beat_xboundary_branch': 'YES' if beat_xboundary_branch else 'NO',
            'beat_frontz_branch': 'YES' if beats_frontz_branch else 'NO',
            'corrected_frontier_markov42': references['entry_frontier']['markov42_triplet'],
            'splitbookend_runnerup_markov42': references['splitbookend_runnerup']['markov42_triplet'],
            'xboundary_branch_markov42': references['xboundary_branch']['markov42_triplet'],
            'frontz_branch_markov42': references['frontz_branch']['markov42_triplet'],
            'best_candidate_markov42': overall_triplet(best['markov42']),
            'best_candidate_kf36': overall_triplet(best['kf36']) if best.get('kf36') is not None else None,
            'statement': statement,
        },
    }

    REPORT_PATH.write_text(render_report(summary), encoding='utf-8')
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({
        'report_path': str(REPORT_PATH),
        'summary_path': str(SUMMARY_PATH),
        'best_candidate': best['candidate_name'],
        'best_markov42': overall_triplet(best['markov42']),
        'best_kf36': overall_triplet(best['kf36']) if best.get('kf36') is not None else None,
        'beat_corrected_frontier': 'YES' if beat_corrected_frontier else 'NO',
        'beat_splitbookend_runnerup': 'YES' if beat_splitbookend_runnerup else 'NO',
        'beat_xboundary_branch': 'YES' if beat_xboundary_branch else 'NO',
        'beat_frontz_branch': 'YES' if beats_frontz_branch else 'NO',
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
