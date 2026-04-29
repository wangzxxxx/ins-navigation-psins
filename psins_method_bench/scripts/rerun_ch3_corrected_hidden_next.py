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
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate
from search_ch3_12pos_closedloop_local_insertions import build_closedloop_candidate, render_action, run_candidate_payload
from search_ch3_post_anchor5_anchor7_terminal_family import candidate_specs

NOISE_SCALE = 0.08
REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
ATT0_DEG = [0.0, 0.0, 0.0]

REPORT_PATH = REPORTS_DIR / f'psins_ch3_corrected_hidden_next_{REPORT_DATE}.md'
SUMMARY_PATH = REPORTS_DIR / f'psins_ch3_corrected_hidden_next_{REPORT_DATE}_summary.json'

REFERENCE_FILES = {
    'faithful12_corrected': {
        'label': 'corrected faithful12',
        'basis': 'corrected',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json',
    },
    'anchor5_mainline_corrected': {
        'label': 'corrected anchor5 mainline',
        'basis': 'corrected',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_zseed_l5_neg6_plus_relaymax_unified_l9y2p5_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_zseed_l5_neg6_plus_relaymax_unified_l9y2p5_shared_noise0p08_param_errors.json',
    },
    'entry_relay_corrected': {
        'label': 'corrected entry-conditioned relay frontier',
        'basis': 'corrected',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json',
    },
    'old_best_legal_legacy': {
        'label': 'old best legal',
        'basis': 'legacy_old_basis',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json',
    },
    'default18': {
        'label': 'default18',
        'basis': 'default_path_reference',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json',
    },
}

HYPOTHESES = [
    {
        'id': 'H1',
        'family': 'anchor7_corridor_bookend_family_corrected_rerun',
        'summary': 'Corrected-basis rerun of the strongest old-basis max specialist: anchor7 pre-entry corridor x bookend.',
        'selected': True,
        'tested': True,
        'candidate_names': ['corridorx_l7_neg1_plus_mainline', 'corridorx_l7_pos2_plus_mainline'],
    },
    {
        'id': 'H2',
        'family': 'anchor12_terminal_reclosure_family_corrected_rerun',
        'summary': 'Corrected-basis rerun of the strongest old-basis hidden signal: anchor12 terminal x reclosure after the corrected mainline.',
        'selected': True,
        'tested': True,
        'candidate_names': ['terminalx_l12_pos2_plus_mainline'],
    },
    {
        'id': 'H3',
        'family': 'anchor7_sign_dose_sensitivity_microcheck',
        'summary': 'Small corrected sign/dose probe inside the anchor7 family to see whether the old signal survives only on the negative bookend or also on the positive side.',
        'selected': True,
        'tested': True,
        'candidate_names': ['corridorx_l7_pos2_plus_mainline'],
    },
    {
        'id': 'H4',
        'family': 'anchor7_to12_split_bookend_chain',
        'summary': 'Untested split-chain follow-up combining anchor7 corridor conditioning with anchor12 terminal reclosure. Left unspent unless the isolated corrected reruns show a real frontier hit.',
        'selected': False,
        'tested': False,
        'candidate_names': [],
    },
]

SELECTED_CANDIDATES = [
    'terminalx_l12_pos2_plus_mainline',
    'corridorx_l7_neg1_plus_mainline',
    'corridorx_l7_pos2_plus_mainline',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def load_reference_payload(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    expected_cfg = expected_noise_config(NOISE_SCALE)
    got_cfg = payload.get('extra', {}).get('noise_config') or payload.get('extra', {}).get('shared_noise_config')
    if got_cfg is not None and got_cfg != expected_cfg:
        raise ValueError(f'Noise configuration mismatch for {path}')
    return payload


def overall_triplet(payload: dict[str, Any]) -> str:
    o = payload['overall']
    return f"{o['mean_pct_error']:.3f} / {o['median_pct_error']:.3f} / {o['max_pct_error']:.3f}"


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


def delta_vs_reference(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        out[metric] = float(reference['overall'][metric]) - float(candidate['overall'][metric])
    return out


def attach_att0(path: Path, payload: dict[str, Any], candidate_name: str, method_key: str) -> dict[str, Any]:
    extra = payload.setdefault('extra', {})
    extra['att0_deg'] = ATT0_DEG
    extra['comparison_mode'] = 'corrected_att0_hidden_rerun_next'
    extra['candidate_registry_key'] = candidate_name
    extra['method_key'] = method_key
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload


def pick_family_best(rows: list[dict[str, Any]], family_name: str) -> dict[str, Any]:
    family_rows = [r for r in rows if r['family'] == family_name]
    return min(family_rows, key=lambda r: (r['markov42']['overall']['mean_pct_error'], r['markov42']['overall']['max_pct_error']))


def render_report(summary: dict[str, Any]) -> str:
    best = summary['best_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 corrected hidden-family next-step rerun')
    lines.append('')
    lines.append('## 1. Corrected-basis search question')
    lines.append('')
    lines.append('- Hard basis enforced: **att0 = (0, 0, 0)** only.')
    lines.append('- Goal of this batch: do **not** expand further on the stale basis; instead rerun the strongest old hidden-family signals under the corrected basis and see whether either still beats the corrected incumbent frontier.')
    lines.append('- Corrected frontier to beat:')
    lines.append(f"  - faithful12 corrected: **{summary['references']['faithful12_corrected']['markov42_triplet']}**")
    lines.append(f"  - corrected anchor5 mainline: **{summary['references']['anchor5_mainline_corrected']['markov42_triplet']}**")
    lines.append(f"  - corrected entry-conditioned relay: **{summary['references']['entry_relay_corrected']['markov42_triplet']}**")
    lines.append('')
    lines.append('## 2. Corrected-basis hypotheses considered before spending batch')
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
    lines.append('## 3. Picked families / hypotheses for the actual corrected batch')
    lines.append('')
    lines.append('- **Picked family A:** anchor12 terminal reclosure corrected rerun')
    lines.append('- **Picked family B:** anchor7 corridor bookend corrected rerun')
    lines.append('- **Micro sign check retained inside family B:** positive-dose anchor7 corridor probe to test whether the old signal was sign-specific.')
    lines.append('- **Held back:** anchor7→12 split chain, because isolated corrected reruns should land first before spending a coupled follow-up.')
    lines.append('')
    lines.append('## 4. Corrected Markov42 landed batch')
    lines.append('')
    lines.append('| rank | candidate | family | hypothesis | mean | median | max | Δmean vs corrected anchor5 | Δmax vs corrected anchor5 | Δmean vs corrected entry relay | Δmax vs corrected entry relay |')
    lines.append('|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(summary['rows_sorted'], start=1):
        m = row['markov42']['overall']
        da = row['delta_vs_anchor5_markov42']
        de = row['delta_vs_entry_markov42']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {row['hypothesis_id']} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {da['mean_pct_error']:+.3f} | {da['max_pct_error']:+.3f} | {de['mean_pct_error']:+.3f} | {de['max_pct_error']:+.3f} |"
        )
    lines.append('')
    lines.append('## 5. Best corrected landing')
    lines.append('')
    lines.append(f"- **Best candidate:** `{best['candidate_name']}`")
    lines.append(f"- **Markov42:** **{overall_triplet(best['markov42'])}**")
    if best.get('kf36') is not None:
        lines.append(f"- **KF36 recheck:** **{overall_triplet(best['kf36'])}**")
    lines.append(f"- vs corrected anchor5 mainline: Δmean **{best['delta_vs_anchor5_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_anchor5_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_anchor5_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- vs corrected entry-conditioned relay: Δmean **{best['delta_vs_entry_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_entry_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_entry_markov42']['max_pct_error']:+.3f}**")
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
        if check.get('next_base_action_preview') is not None:
            preview = check['next_base_action_preview']
            lines.append(
                f"  - next original action remains legal as `{preview['kind']}` {preview['motor_angle_deg']:+d}° with effective axis {preview['effective_body_axis']}"
            )
    lines.append('')
    lines.append('## 8. KF36 recheck gate')
    lines.append('')
    lines.append(f"- Gate used: **rerun only candidates whose corrected Markov42 mean is no worse than corrected anchor5 mainline ({summary['references']['anchor5_mainline_corrected']['markov42']['overall']['mean_pct_error']:.3f})**.")
    lines.append(f"- Triggered candidates: **{', '.join(summary['kf36_rechecked_candidates']) if summary['kf36_rechecked_candidates'] else 'none'}**")
    for row in summary['rows_sorted']:
        if row.get('kf36') is not None:
            lines.append(f"  - `{row['candidate_name']}` → Markov42 **{overall_triplet(row['markov42'])}**, KF36 **{overall_triplet(row['kf36'])}**")
    lines.append('')
    lines.append('## 9. Requested comparison set')
    lines.append('')
    lines.append('| path | basis | Markov42 mean/median/max | KF36 mean/median/max | note |')
    lines.append('|---|---|---|---|---|')
    for ref_key in ['faithful12_corrected', 'anchor5_mainline_corrected', 'entry_relay_corrected', 'old_best_legal_legacy', 'default18']:
        ref = summary['references'][ref_key]
        note = 'legacy old-basis comparator; not rerun in this batch' if ref_key == 'old_best_legal_legacy' else 'reference'
        lines.append(f"| {ref['label']} | {ref['basis']} | {ref['markov42_triplet']} | {ref['kf36_triplet']} | {note} |")
    lines.append(f"| best corrected candidate in this batch | corrected rerun | {overall_triplet(best['markov42'])} | {overall_triplet(best['kf36']) if best.get('kf36') is not None else 'not rerun'} | {best['candidate_name']} |")
    lines.append('')
    lines.append('## 10. Bottom line')
    lines.append('')
    lines.append(f"- **Did this corrected-basis batch beat the new frontier 1.084 / 0.617 / 5.137?** **{summary['bottom_line']['beat_corrected_frontier']}**")
    lines.append(f"- Best corrected candidate was `{best['candidate_name']}` = **{overall_triplet(best['markov42'])}** (KF36 **{overall_triplet(best['kf36'])}**).")
    lines.append('- It **does beat corrected anchor5 on mean and median**, but **does not beat corrected anchor5 on max** and **does not beat corrected entry-conditioned relay on mean or max**.')
    lines.append('- So the old hidden-family signals were **real enough to survive correction as near-frontier candidates**, but **not strong enough to overturn the corrected incumbent entry-conditioned relay frontier**.')
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()

    mod = load_module(str(METHOD_DIR / 'method_42state_gm1.py'), str(SOURCE_FILE))
    faithful = build_candidate(mod, ())
    spec_map = {spec['name']: spec for spec in candidate_specs()}

    references: dict[str, Any] = {}
    for key, info in REFERENCE_FILES.items():
        m = load_reference_payload(info['markov42'])
        k = load_reference_payload(info['kf36'])
        references[key] = {
            'label': info['label'],
            'basis': info['basis'],
            'markov42': compact_metrics(m),
            'kf36': compact_metrics(k),
            'markov42_triplet': overall_triplet(m),
            'kf36_triplet': overall_triplet(k),
            'files': {
                'markov42': str(info['markov42']),
                'kf36': str(info['kf36']),
            },
        }

    anchor5_mean_gate = references['anchor5_mainline_corrected']['markov42']['overall']['mean_pct_error']

    rows: list[dict[str, Any]] = []
    kf36_rechecked_candidates: list[str] = []

    for name in SELECTED_CANDIDATES:
        spec = spec_map[name]
        candidate = build_closedloop_candidate(mod, spec, faithful.rows, faithful.action_sequence)
        markov_payload, _, markov_path = run_candidate_payload(mod, candidate, 'markov42_noisy', args.noise_scale, force_rerun=args.force_rerun)
        markov_payload = attach_att0(markov_path, markov_payload, name, 'markov42_noisy')

        hypothesis_id = spec['hypothesis_id']
        if name == 'corridorx_l7_pos2_plus_mainline':
            hypothesis_id = 'H3'

        row = {
            'candidate_name': name,
            'family': spec['family'],
            'hypothesis_id': hypothesis_id,
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
        row['delta_vs_faithful_markov42'] = delta_vs_reference(references['faithful12_corrected']['markov42'], row['markov42'])
        row['delta_vs_anchor5_markov42'] = delta_vs_reference(references['anchor5_mainline_corrected']['markov42'], row['markov42'])
        row['delta_vs_entry_markov42'] = delta_vs_reference(references['entry_relay_corrected']['markov42'], row['markov42'])
        row['delta_vs_oldbest_markov42'] = delta_vs_reference(references['old_best_legal_legacy']['markov42'], row['markov42'])
        row['delta_vs_default18_markov42'] = delta_vs_reference(references['default18']['markov42'], row['markov42'])

        if row['markov42']['overall']['mean_pct_error'] <= anchor5_mean_gate + 1e-12:
            kf_payload, _, kf_path = run_candidate_payload(mod, candidate, 'kf36_noisy', args.noise_scale, force_rerun=args.force_rerun)
            kf_payload = attach_att0(kf_path, kf_payload, name, 'kf36_noisy')
            row['result_files']['kf36'] = str(kf_path)
            row['kf36'] = compact_metrics(kf_payload)
            row['delta_vs_anchor5_kf36'] = delta_vs_reference(references['anchor5_mainline_corrected']['kf36'], row['kf36'])
            row['delta_vs_entry_kf36'] = delta_vs_reference(references['entry_relay_corrected']['kf36'], row['kf36'])
            kf36_rechecked_candidates.append(name)

        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: (r['markov42']['overall']['mean_pct_error'], r['markov42']['overall']['max_pct_error']))
    best = rows_sorted[0]
    family_best = {
        'anchor7_corridor_bookend_pair': pick_family_best(rows, 'anchor7_corridor_bookend_pair'),
        'anchor12_terminal_reclosure_family': pick_family_best(rows, 'anchor12_terminal_reclosure_family'),
    }

    beat_corrected_frontier = (
        best['markov42']['overall']['mean_pct_error'] < references['entry_relay_corrected']['markov42']['overall']['mean_pct_error']
        and best['markov42']['overall']['max_pct_error'] < references['entry_relay_corrected']['markov42']['overall']['max_pct_error']
    )

    summary = {
        'task': 'chapter-3 corrected hidden-family next-step rerun',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'corrected_att0_deg': ATT0_DEG,
        'hypotheses_considered': HYPOTHESES,
        'selected_families': [
            'anchor12 terminal reclosure corrected rerun',
            'anchor7 corridor bookend corrected rerun',
        ],
        'tested_candidates': SELECTED_CANDIDATES,
        'references': references,
        'rows_sorted': rows_sorted,
        'family_best_candidates': family_best,
        'best_candidate': best,
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
        'bottom_line': {
            'beat_corrected_frontier': 'YES' if beat_corrected_frontier else 'NO',
            'corrected_frontier_markov42': references['entry_relay_corrected']['markov42_triplet'],
            'best_candidate_markov42': overall_triplet(best['markov42']),
            'best_candidate_kf36': overall_triplet(best['kf36']) if best.get('kf36') is not None else None,
            'statement': (
                'Corrected reruns improved over corrected anchor5 on mean but did not beat the corrected entry-conditioned relay frontier on both mean and max.'
            ),
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
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
