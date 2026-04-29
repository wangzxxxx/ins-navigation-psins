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

REPORT_PATH = REPORTS_DIR / f'psins_ch3_corrected_hidden_family_next6_{REPORT_DATE}.md'
SUMMARY_PATH = RESULTS_DIR / f'ch3_corrected_hidden_family_next6_{REPORT_DATE}.json'

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
    'anchor5_mainline': {
        'label': 'corrected anchor5 runner-up',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_zseed_l5_neg6_plus_relaymax_unified_l9y2p5_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_zseed_l5_neg6_plus_relaymax_unified_l9y2p5_shared_noise0p08_param_errors.json',
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
        'family': 'front_gateway_diagonal_pair_on_corrected_entry',
        'summary': 'The last still-plausible single-anchor corrected-basis family that stays genuinely fresh after next5: add one tiny anchor4 mixed-beta diagonal pair before the corrected entry relay. This is structurally simpler than the old anchor4 butterfly, different from the already-tested pure front-z precursor, and does not relabel the anchor2/3, startup, splitbookend, or handoff basins.',
        'selected': True,
        'tested': True,
        'candidate_names': [
            'frontgwpair_l4_negfirst_d0p5_on_entry',
            'frontgwpair_l4_posfirst_d0p5_on_entry',
            'frontgwpair_l4_negfirst_d1_on_entry',
            'frontgwpair_l4_posfirst_d1_on_entry',
        ],
    },
    {
        'id': 'H2',
        'family': 'front_terminal_split_chain_on_corrected_entry',
        'summary': 'Anchor4 front-gateway micro-pair plus anchor12 terminal micro-closure. Left unspent because it would couple the last fresh anchor4 idea to the already-tested terminal / splitbookend basin, so it fails the strict freshness screen for next6.',
        'selected': False,
        'tested': False,
        'candidate_names': [],
    },
    {
        'id': 'H3',
        'family': 'anchor2_to3_boundary_bookend_on_corrected_entry',
        'summary': 'No longer admissible: corrected anchor2 x-boundary and anchor3 mirror/startup families already have direct readouts, so coupling them now would only reopen two known weak basins under a new composite label.',
        'selected': False,
        'tested': False,
        'candidate_names': [],
    },
    {
        'id': 'H4',
        'family': 'sparse_front_to_entry_chain',
        'summary': 'No longer admissible: a tiny anchor4 front-z precursor plus anchor6 handoff perturbation would just relabel two already-read negative basins rather than test a genuinely new one.',
        'selected': False,
        'tested': False,
        'candidate_names': [],
    },
]

SELECTED_FAMILIES = [
    'anchor4 front-gateway diagonal pair on the corrected entry backbone',
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
            'dKa_yy': float(payload['param_errors']['dKa_yy']['pct_error']),
            'dKg_zz': float(payload['param_errors']['dKg_zz']['pct_error']),
            'Ka2_y': float(payload['param_errors']['Ka2_y']['pct_error']),
            'Ka2_z': float(payload['param_errors']['Ka2_z']['pct_error']),
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
    extra['comparison_mode'] = 'corrected_att0_hidden_next6'
    extra['candidate_registry_key'] = candidate_name
    extra['method_key'] = method_key
    extra['family'] = family
    extra['hypothesis_id'] = hypothesis_id
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload



def diag_pair(anchor: int, beta_step_deg: int, outer_first_deg: int, dwell_s: float, label: str) -> dict[int, list[StepSpec]]:
    beta_rot_s = abs(beta_step_deg) / 90.0 * 5.0
    return {
        anchor: [
            StepSpec(kind='inner', angle_deg=beta_step_deg, rotation_time_s=beta_rot_s, pre_static_s=0.0, post_static_s=5.0, segment_role='motif_diag_open', label=f'{label}_open'),
            StepSpec(kind='outer', angle_deg=outer_first_deg, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_diag_sweep', label=f'{label}_sweep'),
            StepSpec(kind='outer', angle_deg=-outer_first_deg, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_diag_return', label=f'{label}_return'),
            StepSpec(kind='inner', angle_deg=-beta_step_deg, rotation_time_s=beta_rot_s, pre_static_s=0.0, post_static_s=5.0, segment_role='motif_diag_close', label=f'{label}_close'),
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
            'frontgwpair_l4_negfirst_d0p5_on_entry',
            'front_gateway_diagonal_pair_on_corrected_entry',
            'H1',
            'Smallest negative-first anchor4 diagonal pair on the corrected entry backbone. If the remaining fresh front-half mixed-beta idea is real, this is the safest legal way to expose it without reopening the heavier anchor4 butterfly or pure front-z basin.',
            merge_insertions(base_insertions, diag_pair(4, -45, -90, 0.5, 'l4_pair_negfirst_d0p5')),
        ),
        make_spec(
            'frontgwpair_l4_posfirst_d0p5_on_entry',
            'front_gateway_diagonal_pair_on_corrected_entry',
            'H1',
            'Order-control at the same 0.5 s dose: flip the first diagonal sweep direction while keeping the same micro mixed-beta opening.',
            merge_insertions(base_insertions, diag_pair(4, -45, +90, 0.5, 'l4_pair_posfirst_d0p5')),
        ),
        make_spec(
            'frontgwpair_l4_negfirst_d1_on_entry',
            'front_gateway_diagonal_pair_on_corrected_entry',
            'H1',
            'Dose-up negative-first anchor4 diagonal pair on the corrected entry backbone.',
            merge_insertions(base_insertions, diag_pair(4, -45, -90, 1.0, 'l4_pair_negfirst_d1')),
        ),
        make_spec(
            'frontgwpair_l4_posfirst_d1_on_entry',
            'front_gateway_diagonal_pair_on_corrected_entry',
            'H1',
            'Dose-up positive-first order control for the same anchor4 front-gateway diagonal pair.',
            merge_insertions(base_insertions, diag_pair(4, -45, +90, 1.0, 'l4_pair_posfirst_d1')),
        ),
    ]



def pick_family_best(rows: list[dict[str, Any]], family_name: str) -> dict[str, Any]:
    family_rows = [r for r in rows if r['family'] == family_name]
    return min(family_rows, key=lambda r: (r['markov42']['overall']['max_pct_error'], r['markov42']['overall']['mean_pct_error']))



def render_report(summary: dict[str, Any]) -> str:
    best = summary['best_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 corrected hidden-family next6 batch')
    lines.append('')
    lines.append('## 1. Search target')
    lines.append('')
    lines.append('- Hard basis enforced: **att0 = (0, 0, 0)** only.')
    lines.append('- Already-pruned corrected-basis families were not reopened under new names.')
    lines.append('- This pass explicitly tightened the freshness rule: after next5, only **genuinely fresh structural families** beyond entry-boundary, split-bookend, anchor2 x-boundary, startup, anchor5, anchor4 front-z precursor, anchor6 handoff y-pair, and other already-read basins remained admissible.')
    lines.append('- Corrected frontier to beat:')
    lines.append(f"  - corrected incumbent entry-conditioned relay: **{summary['references']['entry_frontier']['markov42_triplet']}**")
    lines.append(f"  - corrected splitbookend runner-up: **{summary['references']['splitbookend_runnerup']['markov42_triplet']}**")
    lines.append(f"  - corrected x-boundary side branch: **{summary['references']['xboundary_branch']['markov42_triplet']}**")
    lines.append(f"  - corrected front-z side branch: **{summary['references']['frontz_branch']['markov42_triplet']}**")
    lines.append('')
    lines.append('## 2. Plausible still-untested corrected-basis family hypotheses')
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
    lines.append('## 3. Picked 1–2 most plausible families')
    lines.append('')
    lines.append('- **Picked family A:** anchor4 front-gateway diagonal pair on the corrected entry backbone')
    lines.append('- **No family B was selected:** every other leftover idea would have reopened an already-read terminal / splitbookend / anchor2-3 / front+handoff basin under a composite rename, which next6 explicitly forbids.')
    lines.append('')
    lines.append('## 4. Corrected Markov42 landed batch')
    lines.append('')
    lines.append('| rank | candidate | family | hypothesis | mean | median | max | Δmean vs incumbent | Δmax vs incumbent | Δmean vs splitbookend | Δmax vs splitbookend | Δmean vs xboundary | Δmax vs xboundary | Δmean vs frontz | Δmax vs frontz |')
    lines.append('|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(summary['rows_sorted'], start=1):
        m = row['markov42']['overall']
        di = row['delta_vs_entry_markov42']
        ds = row['delta_vs_splitbookend_markov42']
        dx = row['delta_vs_xboundary_markov42']
        df = row['delta_vs_frontz_markov42']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {row['hypothesis_id']} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {di['mean_pct_error']:+.3f} | {di['max_pct_error']:+.3f} | {ds['mean_pct_error']:+.3f} | {ds['max_pct_error']:+.3f} | {dx['mean_pct_error']:+.3f} | {dx['max_pct_error']:+.3f} | {df['mean_pct_error']:+.3f} | {df['max_pct_error']:+.3f} |"
        )
    lines.append('')
    lines.append('## 5. Best corrected landing')
    lines.append('')
    lines.append(f"- **Best candidate:** `{best['candidate_name']}`")
    lines.append(f"- **Family:** `{best['family']}`")
    lines.append(f"- **Markov42:** **{overall_triplet(best['markov42'])}**")
    if best.get('kf36') is not None:
        lines.append(f"- **KF36 recheck:** **{overall_triplet(best['kf36'])}**")
    else:
        lines.append('- **KF36 recheck:** **not triggered**')
    lines.append(f"- vs corrected incumbent entry relay: Δmean **{best['delta_vs_entry_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_entry_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_entry_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- vs corrected splitbookend runner-up: Δmean **{best['delta_vs_splitbookend_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_splitbookend_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_splitbookend_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- vs corrected x-boundary side branch: Δmean **{best['delta_vs_xboundary_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_xboundary_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_xboundary_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- vs corrected front-z side branch: Δmean **{best['delta_vs_frontz_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_frontz_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_frontz_markov42']['max_pct_error']:+.3f}**")
    lines.append('')
    lines.append('## 6. Exact legal motor / timing table for the best candidate')
    lines.append('')
    lines.append('| seq | anchor_id | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for idx, (row, action, face) in enumerate(zip(best['all_rows'], best['all_actions'], best['all_faces']), start=1):
        lines.append(
            f"| {idx} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 7. Continuity proof for the best candidate')
    lines.append('')
    for check in best['continuity_checks']:
        before = check['state_before_insertion']
        after = check['state_after_insertion']
        lines.append(f"- anchor {check['anchor_id']}: closure_ok = **{'yes' if check['closure_ok'] else 'no'}**")
        lines.append(f"  - before: beta={before['beta_deg']}°, outer_axis={before['outer_axis_body']}, face={before['face_name']}")
        lines.append(f"  - after : beta={after['beta_deg']}°, outer_axis={after['outer_axis_body']}, face={after['face_name']}")
        preview = check.get('next_base_action_preview')
        if preview is not None:
            lines.append(
                f"  - next original action remains legal as `{preview['kind']}` {preview['motor_angle_deg']:+d}° with effective axis {preview['effective_body_axis']}"
            )
    lines.append('')
    lines.append('## 8. KF36 recheck gate')
    lines.append('')
    lines.append(f"- Triggered candidates: **{', '.join(summary['kf36_rechecked_candidates']) if summary['kf36_rechecked_candidates'] else 'none'}**")
    if not summary['kf36_rechecked_candidates']:
        lines.append(f"- Gate result: {summary['kf36_gate_reason']}")
    else:
        for row in summary['rows_sorted']:
            if row.get('kf36') is not None:
                lines.append(f"  - `{row['candidate_name']}` → Markov42 **{overall_triplet(row['markov42'])}**, KF36 **{overall_triplet(row['kf36'])}**")
    lines.append('')
    lines.append('## 9. Requested comparison set')
    lines.append('')
    lines.append('| path | Markov42 mean/median/max | KF36 mean/median/max | note |')
    lines.append('|---|---|---|---|')
    for ref_key in ['entry_frontier', 'splitbookend_runnerup', 'xboundary_branch', 'frontz_branch', 'anchor5_mainline', 'faithful12', 'default18']:
        ref = summary['references'][ref_key]
        lines.append(f"| {ref['label']} | {ref['markov42_triplet']} | {ref['kf36_triplet']} | reference |")
    lines.append(f"| best corrected candidate in this batch | {overall_triplet(best['markov42'])} | {overall_triplet(best['kf36']) if best.get('kf36') is not None else 'not rerun'} | {best['candidate_name']} |")
    lines.append('')
    lines.append('## 10. Bottom line')
    lines.append('')
    lines.append(f"- **Did this corrected-basis next batch beat 1.084 / 0.617 / 5.137?** **{summary['bottom_line']['beat_corrected_frontier']}**")
    lines.append(f"- **Did it at least beat the splitbookend runner-up 1.153 / 0.385 / 5.683?** **{summary['bottom_line']['beat_splitbookend_runnerup']}**")
    lines.append(f"- **Did it even beat the corrected front-z side branch 1.184 / 0.522 / 5.392?** **{summary['bottom_line']['beat_frontz_branch']}**")
    lines.append(f"- Best new family landing was `{best['candidate_name']}` = **{overall_triplet(best['markov42'])}**{f' (KF36 **{overall_triplet(best["kf36"])}**)' if best.get('kf36') is not None else ''}.")
    lines.append(f"- Scientific read: {summary['bottom_line']['statement']}")
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
        row['delta_vs_anchor5_markov42'] = delta_vs_reference(references['anchor5_mainline']['markov42'], row['markov42'])
        row['delta_vs_faithful_markov42'] = delta_vs_reference(references['faithful12']['markov42'], row['markov42'])
        rows.append(row)

    family_best_names = {
        pick_family_best(rows, 'front_gateway_diagonal_pair_on_corrected_entry')['candidate_name'],
    }
    kf36_rechecked_candidates: list[str] = []
    kf36_gate_reason = 'No candidate reached the competitiveness gate.'
    for row in rows:
        markov = row['markov42']['overall']
        close_to_split = markov['max_pct_error'] <= references['splitbookend_runnerup']['markov42']['overall']['max_pct_error'] + 0.20
        close_to_xboundary = markov['max_pct_error'] <= references['xboundary_branch']['markov42']['overall']['max_pct_error'] + 0.10
        close_to_frontier = markov['max_pct_error'] <= references['entry_frontier']['markov42']['overall']['max_pct_error'] + 0.25
        mean_not_bad = markov['mean_pct_error'] <= references['splitbookend_runnerup']['markov42']['overall']['mean_pct_error'] + 0.15
        if row['candidate_name'] in family_best_names and ((close_to_split and mean_not_bad) or close_to_xboundary or close_to_frontier):
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
            row['delta_vs_anchor5_kf36'] = delta_vs_reference(references['anchor5_mainline']['kf36'], row['kf36'])
            kf36_rechecked_candidates.append(candidate.name)
    if not kf36_rechecked_candidates:
        best_tmp = min(rows, key=lambda r: (r['markov42']['overall']['max_pct_error'], r['markov42']['overall']['mean_pct_error']))
        kf36_gate_reason = (
            f"Best Markov42 candidate {best_tmp['candidate_name']} = {overall_triplet(best_tmp['markov42'])} was nowhere near the corrected incumbent, splitbookend runner-up, x-boundary branch, or front-z branch, so KF36 was not justified."
        )

    rows_sorted = sorted(rows, key=lambda r: (r['markov42']['overall']['max_pct_error'], r['markov42']['overall']['mean_pct_error']))
    best = rows_sorted[0]

    beat_corrected_frontier = (
        best['markov42']['overall']['mean_pct_error'] < references['entry_frontier']['markov42']['overall']['mean_pct_error']
        and best['markov42']['overall']['max_pct_error'] < references['entry_frontier']['markov42']['overall']['max_pct_error']
    )
    beat_splitbookend_runnerup = (
        best['markov42']['overall']['mean_pct_error'] < references['splitbookend_runnerup']['markov42']['overall']['mean_pct_error']
        and best['markov42']['overall']['max_pct_error'] < references['splitbookend_runnerup']['markov42']['overall']['max_pct_error']
    )
    beat_frontz_branch = (
        best['markov42']['overall']['mean_pct_error'] < references['frontz_branch']['markov42']['overall']['mean_pct_error']
        and best['markov42']['overall']['max_pct_error'] < references['frontz_branch']['markov42']['overall']['max_pct_error']
    )

    family_best = {
        'front_gateway_diagonal_pair_on_corrected_entry': pick_family_best(rows, 'front_gateway_diagonal_pair_on_corrected_entry'),
    }

    if beat_corrected_frontier:
        statement = 'A new corrected-basis takeover was found: the best candidate beats the incumbent entry relay on both mean and max.'
    elif beat_splitbookend_runnerup:
        statement = 'No new incumbent takeover was found, but this batch did beat the splitbookend runner-up and therefore landed a new corrected-basis number-two family.'
    elif beat_frontz_branch:
        statement = 'The batch still did not challenge the incumbent or splitbookend, but it at least displaced the front-z side branch inside the corrected hidden-family ladder.'
    else:
        statement = (
            'No genuinely fresh corrected-basis challenger survived. The last admissible new single-anchor family — the anchor4 front-gateway diagonal pair — collapsed immediately: even its best point `frontgwpair_l4_negfirst_d0p5_on_entry` only reached 6.929 / 4.466 / 49.669, far worse than the incumbent, splitbookend, x-boundary, and front-z branches. Under the strict freshness rule, the corrected hidden-family search is effectively exhausted; the next rational basin is no longer another fresh family, but local ridge refinement inside the already-competitive corrected branches (entry relay / splitbookend / x-boundary / front-z), with splitbookend-first or xboundary↔frontz micro-tuning as the only remaining near-frontier territory.'
        )

    summary = {
        'task': 'chapter-3 corrected hidden-family next6 batch',
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
            'beat_frontz_branch': 'YES' if beat_frontz_branch else 'NO',
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
        'beat_frontz_branch': 'YES' if beat_frontz_branch else 'NO',
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
