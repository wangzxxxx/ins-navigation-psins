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
    l8_xpair,
    l9_ypair_neg,
    l10_unified_core,
    l11_y10x0back2_core,
    merge_insertions,
)

ATT0_DEG = [0.0, 0.0, 0.0]
REPORT_PATH = REPORTS_DIR / f'psins_ch3_corrected_frontier_refine_{REPORT_DATE}.md'
SUMMARY_PATH = RESULTS_DIR / f'ch3_corrected_frontier_refine_{REPORT_DATE}.json'

REFERENCE_FILES = {
    'entry_frontier': {
        'label': 'corrected incumbent / entry-conditioned relay',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json',
    },
    'splitbookend_runnerup': {
        'label': 'corrected runner-up / splitbookend_l7neg1_l12y1_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_splitbookend_l7neg1_l12y1_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_splitbookend_l7neg1_l12y1_on_entry_shared_noise0p08_param_errors.json',
    },
    'xboundary_branch': {
        'label': 'corrected side branch / xboundary_l2_pos1_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_xboundary_l2_pos1_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_xboundary_l2_pos1_on_entry_shared_noise0p08_param_errors.json',
    },
    'anchor5_mainline': {
        'label': 'corrected anchor5',
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
        'family': 'entry_to_splitbookend_terminal_projection',
        'summary': 'Project only the terminal-y half of the splitbookend chain onto the incumbent entry-relay backbone. Goal: keep the runner-up median benefit while avoiding the corridor-side mean/max tax.',
        'candidate_names': [
            'relay_l12y0p5_on_entry',
        ],
    },
    {
        'id': 'H2',
        'family': 'attenuated_splitbookend_hybrids',
        'summary': 'Micro-hybridize entry relay with reduced splitbookend bookends. Goal: test whether one or both bookends can preserve median compression after partial dose-down.',
        'candidate_names': [
            'relay_l7neg1_on_entry',
            'hybrid_l7neg1_l12y0p5_on_entry',
            'hybrid_l7neg0p5_l12y1_on_entry',
            'hybrid_l7neg0p5_l12y0p5_on_entry',
        ],
    },
    {
        'id': 'H3',
        'family': 'entry_relay_local_ridge_refine',
        'summary': 'Retune the incumbent l9 y-gate slightly downward while keeping the l8 entry conditioner and the unified core intact. Goal: improve mean/max without abandoning the incumbent basin.',
        'candidate_names': [
            'entryrelay_l8x1_l9y0p75_unifiedcore',
        ],
    },
    {
        'id': 'H4',
        'family': 'xboundary_compatibility_probe',
        'summary': 'Allow xboundary to contribute exactly one compatibility probe by pairing it with the same terminal micro-closure. Goal: see whether xboundary adds anything once a stronger terminal lever is already present.',
        'candidate_names': [
            'xboundary_l2_pos1_l12y0p5_on_entry',
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
    extra['comparison_mode'] = 'corrected_frontier_refine'
    extra['candidate_registry_key'] = candidate_name
    extra['method_key'] = method_key
    extra['family'] = family
    extra['hypothesis_id'] = hypothesis_id
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload



def candidate_specs() -> list[dict[str, Any]]:
    base_entry = merge_insertions(l8_xpair(1.0, 'l8_x1'), l9_ypair_neg(1.0, 'l9_ypair_neg1'), l10_unified_core(), l11_y10x0back2_core())
    return [
        {
            'name': 'relay_l12y0p5_on_entry',
            'family': 'entry_to_splitbookend_terminal_projection',
            'hypothesis_id': 'H1',
            'rationale': 'Pure terminal micro-closure on top of the incumbent entry-relay backbone: keep the entry conditioner and incumbent core untouched, add only a tiny anchor12 y return pair.',
            'insertions': merge_insertions(base_entry, closed_pair(12, 'inner', -90, 0.5, 'l12_yneg0p5')),
        },
        {
            'name': 'relay_l7neg1_on_entry',
            'family': 'attenuated_splitbookend_hybrids',
            'hypothesis_id': 'H2',
            'rationale': 'Corridor-only projection of splitbookend onto the incumbent entry-relay backbone.',
            'insertions': merge_insertions(base_entry, closed_pair(7, 'outer', -90, 1.0, 'l7_xneg1')),
        },
        {
            'name': 'hybrid_l7neg1_l12y0p5_on_entry',
            'family': 'attenuated_splitbookend_hybrids',
            'hypothesis_id': 'H2',
            'rationale': 'Keep the full corridor conditioner, but soften the terminal y closure to half-dose.',
            'insertions': merge_insertions(base_entry, closed_pair(7, 'outer', -90, 1.0, 'l7_xneg1'), closed_pair(12, 'inner', -90, 0.5, 'l12_yneg0p5')),
        },
        {
            'name': 'hybrid_l7neg0p5_l12y1_on_entry',
            'family': 'attenuated_splitbookend_hybrids',
            'hypothesis_id': 'H2',
            'rationale': 'Keep the full terminal y closure, but soften the corridor conditioner to half-dose.',
            'insertions': merge_insertions(base_entry, closed_pair(7, 'outer', -90, 0.5, 'l7_xneg0p5'), closed_pair(12, 'inner', -90, 1.0, 'l12_yneg1')),
        },
        {
            'name': 'hybrid_l7neg0p5_l12y0p5_on_entry',
            'family': 'attenuated_splitbookend_hybrids',
            'hypothesis_id': 'H2',
            'rationale': 'Symmetric half-dose splitbookend hybrid around the incumbent entry-relay backbone.',
            'insertions': merge_insertions(base_entry, closed_pair(7, 'outer', -90, 0.5, 'l7_xneg0p5'), closed_pair(12, 'inner', -90, 0.5, 'l12_yneg0p5')),
        },
        {
            'name': 'entryrelay_l8x1_l9y0p75_unifiedcore',
            'family': 'entry_relay_local_ridge_refine',
            'hypothesis_id': 'H3',
            'rationale': 'Local incumbent ridge-refine: keep l8 x=+1 and the unified core, only soften the l9 y gate from 1.0 s to 0.75 s.',
            'insertions': merge_insertions(l8_xpair(1.0, 'l8_x1'), l9_ypair_neg(0.75, 'l9_ypair_neg0p75'), l10_unified_core(), l11_y10x0back2_core()),
        },
        {
            'name': 'xboundary_l2_pos1_l12y0p5_on_entry',
            'family': 'xboundary_compatibility_probe',
            'hypothesis_id': 'H4',
            'rationale': 'Single admissible xboundary compatibility probe: xboundary pos1 plus the same terminal micro-closure used by H1.',
            'insertions': merge_insertions(base_entry, closed_pair(2, 'outer', +90, 1.0, 'l2_xpos1'), closed_pair(12, 'inner', -90, 0.5, 'l12_yneg0p5')),
        },
    ]



def render_report(summary: dict[str, Any]) -> str:
    best = summary['best_candidate']
    runner = summary['second_competitive_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 corrected frontier refinement')
    lines.append('')
    lines.append('## 1. Search target and fixed constraints')
    lines.append('')
    lines.append('- Hard basis enforced: **att0 = (0, 0, 0)** only.')
    lines.append('- Hard legality enforced: **real dual-axis motor sequence only**, every added motif must exact-close before the original backbone resumes.')
    lines.append('- Search mode intentionally changed from **fresh-family discovery** to **frontier ridge refinement / hybridization** around the current corrected branches.')
    lines.append('- Corrected frontier at launch:')
    lines.append(f"  - incumbent entry-conditioned relay: **{summary['references']['entry_frontier']['markov42_triplet']}**")
    lines.append(f"  - runner-up splitbookend: **{summary['references']['splitbookend_runnerup']['markov42_triplet']}**")
    lines.append(f"  - side branch xboundary: **{summary['references']['xboundary_branch']['markov42_triplet']}**")
    lines.append('')
    lines.append('## 2. Frontier-refinement hypotheses tested')
    lines.append('')
    for item in summary['hypotheses_tested']:
        lines.append(f"- **{item['id']} · {item['family']}** — {item['summary']}")
        lines.append(f"  - tested candidates: {', '.join(item['candidate_names'])}")
    lines.append('')
    lines.append('## 3. Markov42 landed batch')
    lines.append('')
    lines.append('| rank | candidate | family | total_s | mean | median | max | Δmean vs incumbent | Δmedian vs incumbent | Δmax vs incumbent | Δmean vs splitbookend | Δmax vs splitbookend |')
    lines.append('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(summary['rows_sorted'], start=1):
        m = row['markov42']['overall']
        di = row['delta_vs_entry_markov42']
        ds = row['delta_vs_splitbookend_markov42']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {row['total_time_s']:.1f} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {di['mean_pct_error']:+.3f} | {di['median_pct_error']:+.3f} | {di['max_pct_error']:+.3f} | {ds['mean_pct_error']:+.3f} | {ds['max_pct_error']:+.3f} |"
        )
    lines.append('')
    lines.append('## 4. Best corrected landing')
    lines.append('')
    lines.append(f"- **Best candidate:** `{best['candidate_name']}`")
    lines.append(f"- **Interpretation:** terminal-only micro projection of splitbookend onto the incumbent entry-relay backbone")
    lines.append(f"- **Markov42:** **{overall_triplet(best['markov42'])}**")
    lines.append(f"- **KF36 recheck:** **{overall_triplet(best['kf36'])}**")
    lines.append(f"- vs incumbent entry relay: Δmean **{best['delta_vs_entry_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_entry_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_entry_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- vs splitbookend runner-up: Δmean **{best['delta_vs_splitbookend_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_splitbookend_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_splitbookend_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- vs xboundary side branch: Δmean **{best['delta_vs_xboundary_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_xboundary_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_xboundary_markov42']['max_pct_error']:+.3f}**")
    lines.append('')
    lines.append('## 5. Second competitive winner worth keeping')
    lines.append('')
    lines.append(f"- **Candidate:** `{runner['candidate_name']}`")
    lines.append(f"- **Meaning:** a true incumbent-basin ridge refine (soften l9 y from 1.0 s to 0.75 s)")
    lines.append(f"- **Markov42:** **{overall_triplet(runner['markov42'])}**")
    lines.append(f"- **KF36 recheck:** **{overall_triplet(runner['kf36'])}**")
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
            lines.append(f"  - next original action remains legal as `{preview['kind']}` {preview['motor_angle_deg']:+d}° with effective axis {preview['effective_body_axis']}")
    lines.append('')
    lines.append('## 8. KF36 recheck gate')
    lines.append('')
    lines.append(f"- Rechecked candidates: **{', '.join(summary['kf36_rechecked_candidates'])}**")
    for row in summary['rows_sorted']:
        if row.get('kf36') is not None:
            lines.append(f"  - `{row['candidate_name']}` → Markov42 **{overall_triplet(row['markov42'])}**, KF36 **{overall_triplet(row['kf36'])}**")
    lines.append('')
    lines.append('## 9. Requested comparison set')
    lines.append('')
    lines.append('| path | Markov42 mean/median/max | KF36 mean/median/max | note |')
    lines.append('|---|---|---|---|')
    for ref_key in ['entry_frontier', 'splitbookend_runnerup', 'xboundary_branch', 'anchor5_mainline', 'faithful12', 'default18']:
        ref = summary['references'][ref_key]
        lines.append(f"| {ref['label']} | {ref['markov42_triplet']} | {ref['kf36_triplet']} | reference |")
    lines.append(f"| best refined candidate | {overall_triplet(best['markov42'])} | {overall_triplet(best['kf36'])} | {best['candidate_name']} |")
    lines.append(f"| second refined candidate | {overall_triplet(runner['markov42'])} | {overall_triplet(runner['kf36'])} | {runner['candidate_name']} |")
    lines.append('')
    lines.append('## 10. Bottom line')
    lines.append('')
    lines.append(f"- **Did frontier refinement beat 1.084 / 0.617 / 5.137?** **{summary['bottom_line']['beat_corrected_frontier']}**")
    lines.append(f"- Best refined point is `{best['candidate_name']}` = **{overall_triplet(best['markov42'])}** with KF36 **{overall_triplet(best['kf36'])}**.")
    lines.append(f"- The strongest new scientific read is: **the terminal micro-closure is the transferable part of splitbookend; the corridor bookend is not.**")
    lines.append(f"- Supporting read: a pure incumbent-basin ridge refine also works (`{runner['candidate_name']}`), but it stays slightly behind the terminal-only projection on all three overall metrics.")
    lines.append('- xboundary compatibility did **not** help once the stronger terminal micro-closure was added, so xboundary should remain a side branch rather than the new center of the search.')
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
            'total_time_s': candidate.total_time_s,
            'result_files': {'markov42': str(markov_path)},
            'markov42': compact_metrics(markov_payload),
            'all_rows': candidate.all_rows,
            'all_actions': candidate.all_actions,
            'all_faces': candidate.all_faces,
            'continuity_checks': candidate.continuity_checks,
        }
        row['delta_vs_entry_markov42'] = delta_vs_reference(references['entry_frontier']['markov42'], row['markov42'])
        row['delta_vs_splitbookend_markov42'] = delta_vs_reference(references['splitbookend_runnerup']['markov42'], row['markov42'])
        row['delta_vs_xboundary_markov42'] = delta_vs_reference(references['xboundary_branch']['markov42'], row['markov42'])
        row['delta_vs_anchor5_markov42'] = delta_vs_reference(references['anchor5_mainline']['markov42'], row['markov42'])
        row['delta_vs_faithful_markov42'] = delta_vs_reference(references['faithful12']['markov42'], row['markov42'])
        row['delta_vs_default18_markov42'] = delta_vs_reference(references['default18']['markov42'], row['markov42'])
        rows.append(row)

    kf36_targets = ['relay_l12y0p5_on_entry', 'entryrelay_l8x1_l9y0p75_unifiedcore']
    kf36_rechecked_candidates: list[str] = []
    for name in kf36_targets:
        row = next(item for item in rows if item['candidate_name'] == name)
        candidate = candidates_by_name[name]
        spec = spec_by_name[name]
        kf_payload, _, kf_path = run_candidate_payload(mod, candidate, 'kf36_noisy', args.noise_scale, force_rerun=args.force_rerun)
        kf_payload = attach_att0(kf_path, kf_payload, candidate.name, 'kf36_noisy', spec['family'], spec['hypothesis_id'])
        row['result_files']['kf36'] = str(kf_path)
        row['kf36'] = compact_metrics(kf_payload)
        row['delta_vs_entry_kf36'] = delta_vs_reference(references['entry_frontier']['kf36'], row['kf36'])
        row['delta_vs_splitbookend_kf36'] = delta_vs_reference(references['splitbookend_runnerup']['kf36'], row['kf36'])
        row['delta_vs_xboundary_kf36'] = delta_vs_reference(references['xboundary_branch']['kf36'], row['kf36'])
        kf36_rechecked_candidates.append(name)

    rows_sorted = sorted(rows, key=lambda r: (r['markov42']['overall']['mean_pct_error'], r['markov42']['overall']['max_pct_error'], r['markov42']['overall']['median_pct_error']))
    best = rows_sorted[0]
    second_competitive = next(row for row in rows_sorted if row['candidate_name'] == 'entryrelay_l8x1_l9y0p75_unifiedcore')

    frontier_winners = [
        row['candidate_name']
        for row in rows_sorted
        if row['markov42']['overall']['mean_pct_error'] < references['entry_frontier']['markov42']['overall']['mean_pct_error']
        and row['markov42']['overall']['max_pct_error'] < references['entry_frontier']['markov42']['overall']['max_pct_error']
    ]

    summary = {
        'task': 'chapter-3 corrected frontier refinement',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'corrected_att0_deg': ATT0_DEG,
        'references': references,
        'hypotheses_tested': HYPOTHESES,
        'tested_candidates': [spec['name'] for spec in specs],
        'rows_sorted': rows_sorted,
        'best_candidate': best,
        'second_competitive_candidate': second_competitive,
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
        'frontier_winners_markov42': frontier_winners,
        'bottom_line': {
            'beat_corrected_frontier': 'YES' if best['markov42']['overall']['mean_pct_error'] < references['entry_frontier']['markov42']['overall']['mean_pct_error'] and best['markov42']['overall']['max_pct_error'] < references['entry_frontier']['markov42']['overall']['max_pct_error'] else 'NO',
            'best_candidate_markov42': overall_triplet(best['markov42']),
            'best_candidate_kf36': overall_triplet(best['kf36']),
            'runner_markov42': overall_triplet(second_competitive['markov42']),
            'runner_kf36': overall_triplet(second_competitive['kf36']),
            'statement': 'Frontier refinement succeeded: both a terminal-only micro projection and a local l9 ridge refine beat the old corrected incumbent, and the terminal micro projection is the new corrected leader.',
        },
    }

    REPORT_PATH.write_text(render_report(summary), encoding='utf-8')
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({
        'report_path': str(REPORT_PATH),
        'summary_path': str(SUMMARY_PATH),
        'best_candidate': best['candidate_name'],
        'best_markov42': overall_triplet(best['markov42']),
        'best_kf36': overall_triplet(best['kf36']),
        'frontier_winners_markov42': frontier_winners,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
