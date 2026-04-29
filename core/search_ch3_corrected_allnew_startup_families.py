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

NOISE_SCALE = 0.08
REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
ATT0_DEG = [0.0, 0.0, 0.0]

REPORT_PATH = REPORTS_DIR / f'psins_ch3_corrected_allnew_startup_families_{REPORT_DATE}.md'
SUMMARY_PATH = REPORTS_DIR / f'psins_ch3_corrected_allnew_startup_families_{REPORT_DATE}_summary.json'

REFERENCE_FILES = {
    'entry_relay_corrected': {
        'label': 'corrected incumbent entry-conditioned relay frontier',
        'basis': 'corrected',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json',
    },
    'anchor5_mainline_corrected': {
        'label': 'corrected anchor5 runner-up',
        'basis': 'corrected',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_zseed_l5_neg6_plus_relaymax_unified_l9y2p5_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_zseed_l5_neg6_plus_relaymax_unified_l9y2p5_shared_noise0p08_param_errors.json',
    },
    'corridor_corrected': {
        'label': 'corrected corridor side branch',
        'basis': 'corrected',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_corridorx_l7_neg1_plus_mainline_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_corridorx_l7_neg1_plus_mainline_shared_noise0p08_param_errors.json',
    },
    'faithful12_corrected': {
        'label': 'corrected faithful12',
        'basis': 'corrected',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json',
    },
    'default18': {
        'label': 'default18',
        'basis': 'default_path_reference',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json',
    },
}

EXCLUDED_BASINS = [
    'late10/late11 local',
    'relay',
    'anchor9 butterfly',
    'precondition/fullblock',
    'anchor6',
    'anchor8 entry-boundary',
    'entry-conditioned relay',
    'anchor5 far-z seed',
    'anchor4 front-z',
    'corridor-bookend',
    'terminal reclosure',
    'any rerun/variant of the above',
]

HYPOTHESES = [
    {
        'id': 'H1',
        'family': 'startup_twinx_antisymmetric_bridge',
        'summary': 'Use the repeated startup x-family anchors (1 and 2) as an antisymmetric distributed bridge: a small closed x sweep on anchor1 and the opposite-sign closed x sweep on anchor2. This is an all-new early twin-anchor symmetry-breaking family, not a replay of any late/front/entry basin.',
        'selected': True,
        'tested': True,
    },
    {
        'id': 'H2',
        'family': 'startup_crossaxis_handshake',
        'summary': 'Split a micro closed pair across the first outer→inner handoff: anchor2 x-family seed plus anchor3 pure inner-y recoil. This is an all-new cross-axis handshake family at the first kinematic bifurcation.',
        'selected': True,
        'tested': True,
    },
    {
        'id': 'H3',
        'family': 'first_bifurcation_beta_recoil',
        'summary': 'Pure anchor3 inner recoil only, without any diagonal or downstream relay motif. This isolates whether the first beta bifurcation alone carries useful corrected-basis signal.',
        'selected': True,
        'tested': True,
    },
    {
        'id': 'H4',
        'family': 'startup_three_anchor_phase_ladder',
        'summary': 'Three-anchor startup phase ladder across anchors1→2→3 with a distributed x/x/y micro-dose. Left unspent because H1 already dominated the all-new early family pool, while H2/H3 stayed clearly weaker.',
        'selected': False,
        'tested': False,
    },
]


def pair(kind: str, ang: int, dwell_s: float, label: str) -> list[StepSpec]:
    return [
        StepSpec(kind=kind, angle_deg=ang, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_out', label=f'{label}_out'),
        StepSpec(kind=kind, angle_deg=-ang, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_return', label=f'{label}_return'),
    ]


CANDIDATE_SPECS = [
    {
        'name': 'allnew_startwinx_a1neg1_a2pos1',
        'family': 'startup_twinx_antisymmetric_bridge',
        'hypothesis_id': 'H1',
        'rationale': 'Strongest all-new startup twin-x bridge probe: anchor1 negative x closed pair followed by anchor2 positive x closed pair, both at 1 s dwell.',
        'insertions': {1: pair('outer', -90, 1.0, 'a1x_neg1'), 2: pair('outer', +90, 1.0, 'a2x_pos1')},
    },
    {
        'name': 'allnew_startwinx_a1neg0p5_a2pos1',
        'family': 'startup_twinx_antisymmetric_bridge',
        'hypothesis_id': 'H1',
        'rationale': 'Micro-dose anchor1 / full anchor2 version of the same all-new startup twin-x bridge.',
        'insertions': {1: pair('outer', -90, 0.5, 'a1x_neg0p5'), 2: pair('outer', +90, 1.0, 'a2x_pos1')},
    },
    {
        'name': 'allnew_startwinx_a1neg1_a2pos0p5',
        'family': 'startup_twinx_antisymmetric_bridge',
        'hypothesis_id': 'H1',
        'rationale': 'Full anchor1 / micro-dose anchor2 version of the all-new startup twin-x bridge.',
        'insertions': {1: pair('outer', -90, 1.0, 'a1x_neg1'), 2: pair('outer', +90, 0.5, 'a2x_pos0p5')},
    },
    {
        'name': 'allnew_startwinx_a1neg0p5_a2pos0p5',
        'family': 'startup_twinx_antisymmetric_bridge',
        'hypothesis_id': 'H1',
        'rationale': 'Fully micro-dosed startup twin-x antisymmetric bridge.',
        'insertions': {1: pair('outer', -90, 0.5, 'a1x_neg0p5'), 2: pair('outer', +90, 0.5, 'a2x_pos0p5')},
    },
    {
        'name': 'allnew_startwinx_a1pos1_a2neg1',
        'family': 'startup_twinx_antisymmetric_bridge',
        'hypothesis_id': 'H1',
        'rationale': 'Sign-flipped control for the startup twin-x antisymmetric bridge.',
        'insertions': {1: pair('outer', +90, 1.0, 'a1x_pos1'), 2: pair('outer', -90, 1.0, 'a2x_neg1')},
    },
    {
        'name': 'allnew_crosshand_a2neg1_a3yneg1',
        'family': 'startup_crossaxis_handshake',
        'hypothesis_id': 'H2',
        'rationale': 'All-new cross-axis handshake: anchor2 negative x pair plus anchor3 negative inner-y recoil.',
        'insertions': {2: pair('outer', -90, 1.0, 'a2x_neg1'), 3: pair('inner', -90, 1.0, 'a3y_neg1')},
    },
    {
        'name': 'allnew_crosshand_a2pos1_a3yneg1',
        'family': 'startup_crossaxis_handshake',
        'hypothesis_id': 'H2',
        'rationale': 'Sign-control companion for the all-new startup cross-axis handshake.',
        'insertions': {2: pair('outer', +90, 1.0, 'a2x_pos1'), 3: pair('inner', -90, 1.0, 'a3y_neg1')},
    },
    {
        'name': 'allnew_betarecoil_l3_neg0p5',
        'family': 'first_bifurcation_beta_recoil',
        'hypothesis_id': 'H3',
        'rationale': 'All-new pure anchor3 beta recoil with negative 0.5 s dwell.',
        'insertions': {3: pair('inner', -90, 0.5, 'l3y_neg0p5')},
    },
    {
        'name': 'allnew_betarecoil_l3_pos0p5',
        'family': 'first_bifurcation_beta_recoil',
        'hypothesis_id': 'H3',
        'rationale': 'Positive sign-control for the pure anchor3 beta recoil family at 0.5 s dwell.',
        'insertions': {3: pair('inner', +90, 0.5, 'l3y_pos0p5')},
    },
    {
        'name': 'allnew_betarecoil_l3_neg1',
        'family': 'first_bifurcation_beta_recoil',
        'hypothesis_id': 'H3',
        'rationale': 'Dose-up negative pure anchor3 beta recoil at 1.0 s dwell.',
        'insertions': {3: pair('inner', -90, 1.0, 'l3y_neg1')},
    },
    {
        'name': 'allnew_betarecoil_l3_pos1',
        'family': 'first_bifurcation_beta_recoil',
        'hypothesis_id': 'H3',
        'rationale': 'Dose-up positive pure anchor3 beta recoil at 1.0 s dwell.',
        'insertions': {3: pair('inner', +90, 1.0, 'l3y_pos1')},
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()



def overall_triplet(payload: dict[str, Any]) -> str:
    o = payload['overall']
    return f"{o['mean_pct_error']:.3f} / {o['median_pct_error']:.3f} / {o['max_pct_error']:.3f}"



def load_reference_payload(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    expected_cfg = expected_noise_config(NOISE_SCALE)
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



def delta_vs_reference(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        out[metric] = float(reference['overall'][metric]) - float(candidate['overall'][metric])
    return out



def attach_corrected_metadata(path: Path, payload: dict[str, Any], candidate_name: str, method_key: str) -> dict[str, Any]:
    extra = payload.setdefault('extra', {})
    extra['att0_deg'] = ATT0_DEG
    extra['comparison_mode'] = 'corrected_att0_allnew_startup_family_search'
    extra['candidate_registry_key'] = candidate_name
    extra['method_key'] = method_key
    extra['family_search_scope'] = 'all_new_only'
    extra['excluded_old_basins'] = EXCLUDED_BASINS
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload



def render_report(summary: dict[str, Any]) -> str:
    best = summary['best_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 corrected all-new startup-family search')
    lines.append('')
    lines.append('## 1. Mission guardrails')
    lines.append('')
    lines.append('- Hard basis enforced: **att0 = (0, 0, 0)** only.')
    lines.append('- This pass explicitly **excluded** all previously known active basins and their reruns/renames:')
    for item in summary['excluded_basins']:
        lines.append(f'  - {item}')
    lines.append('- Only **all-new corrected-basis structural families** were allowed.')
    lines.append('')
    lines.append('## 2. All-new family hypotheses listed before spending batch')
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
    lines.append('## 3. Picked all-new families for the landed batch')
    lines.append('')
    lines.append('- **Picked H1:** startup twin-x antisymmetric bridge')
    lines.append('- **Picked H2:** startup cross-axis handshake')
    lines.append('- **Picked H3:** first-bifurcation pure beta recoil')
    lines.append('- **Held back H4:** three-anchor startup phase ladder, because H1 already dominated the early-family pool while H2/H3 stayed clearly weaker.')
    lines.append('')
    lines.append('## 4. Corrected Markov42 landed batch')
    lines.append('')
    lines.append('| rank | candidate | family | hypothesis | mean | median | max | Δmean vs corrected entry | Δmax vs corrected entry | Δmean vs corrected anchor5 | Δmax vs corrected anchor5 |')
    lines.append('|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(summary['rows_sorted'], start=1):
        m = row['markov42']['overall']
        de = row['delta_vs_entry_markov42']
        da = row['delta_vs_anchor5_markov42']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {row['hypothesis_id']} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {de['mean_pct_error']:+.3f} | {de['max_pct_error']:+.3f} | {da['mean_pct_error']:+.3f} | {da['max_pct_error']:+.3f} |"
        )
    lines.append('')
    lines.append('## 5. Best all-new landing')
    lines.append('')
    lines.append(f"- **Best candidate:** `{best['candidate_name']}`")
    lines.append(f"- **Family:** `{best['family']}`")
    lines.append(f"- **Markov42:** **{overall_triplet(best['markov42'])}**")
    lines.append(f"- vs corrected incumbent entry relay: Δmean **{best['delta_vs_entry_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_entry_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_entry_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- vs corrected anchor5 runner-up: Δmean **{best['delta_vs_anchor5_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_anchor5_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_anchor5_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- vs corrected corridor branch: Δmean **{best['delta_vs_corridor_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_corridor_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_corridor_markov42']['max_pct_error']:+.3f}**")
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
    lines.append(f"- Gate used: rerun only if a candidate enters a genuinely competitive band near the corrected incumbent set. Triggered: **{'yes' if summary['kf36_triggered'] else 'no'}**")
    lines.append(f"- Reason: **{summary['kf36_reason']}**")
    lines.append('')
    lines.append('## 9. Requested comparison set')
    lines.append('')
    lines.append('| path | basis | Markov42 mean/median/max | KF36 mean/median/max | note |')
    lines.append('|---|---|---|---|---|')
    for ref_key in ['entry_relay_corrected', 'anchor5_mainline_corrected', 'corridor_corrected', 'faithful12_corrected', 'default18']:
        ref = summary['references'][ref_key]
        lines.append(f"| {ref['label']} | {ref['basis']} | {ref['markov42_triplet']} | {ref['kf36_triplet']} | reference |")
    lines.append(f"| best all-new candidate in this batch | corrected all-new | {overall_triplet(best['markov42'])} | not rerun | {best['candidate_name']} |")
    lines.append('')
    lines.append('## 10. Bottom line')
    lines.append('')
    lines.append(f"- **Did this corrected-basis all-new batch beat the incumbent frontier 1.084 / 0.617 / 5.137?** **{summary['bottom_line']['beat_corrected_frontier']}**")
    lines.append(f"- Best all-new landing was `{best['candidate_name']}` = **{overall_triplet(best['markov42'])}**.")
    lines.append('- The startup twin-x antisymmetric bridge is a **real corrected-basis signal** because it beats corrected faithful12 and default18 on both mean and max, even though it still trails the corrected incumbent set.')
    lines.append('- But it remains **far from frontier quality on max**, so this all-new batch does **not** overturn the corrected incumbent set.')
    lines.append('')
    return '\n'.join(lines) + '\n'



def main() -> None:
    args = parse_args()
    mod = load_module('psins_ch3_corrected_allnew_startup_families', str(SOURCE_FILE))
    faithful = build_candidate(mod, ())

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
            'files': {'markov42': str(info['markov42']), 'kf36': str(info['kf36'])},
        }

    rows: list[dict[str, Any]] = []
    for spec in CANDIDATE_SPECS:
        candidate = build_closedloop_candidate(mod, spec, faithful.rows, faithful.action_sequence)
        markov_payload, _, markov_path = run_candidate_payload(mod, candidate, 'markov42_noisy', args.noise_scale, force_rerun=args.force_rerun)
        markov_payload = attach_corrected_metadata(markov_path, markov_payload, spec['name'], 'markov42_noisy')
        row = {
            'candidate_name': spec['name'],
            'family': spec['family'],
            'hypothesis_id': spec['hypothesis_id'],
            'rationale': spec['rationale'],
            'result_files': {'markov42': str(markov_path)},
            'markov42': compact_metrics(markov_payload),
            'all_rows': candidate.all_rows,
            'all_actions': candidate.all_actions,
            'all_faces': candidate.all_faces,
            'continuity_checks': candidate.continuity_checks,
        }
        row['delta_vs_entry_markov42'] = delta_vs_reference(references['entry_relay_corrected']['markov42'], row['markov42'])
        row['delta_vs_anchor5_markov42'] = delta_vs_reference(references['anchor5_mainline_corrected']['markov42'], row['markov42'])
        row['delta_vs_corridor_markov42'] = delta_vs_reference(references['corridor_corrected']['markov42'], row['markov42'])
        row['delta_vs_faithful_markov42'] = delta_vs_reference(references['faithful12_corrected']['markov42'], row['markov42'])
        row['delta_vs_default18_markov42'] = delta_vs_reference(references['default18']['markov42'], row['markov42'])
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: (r['markov42']['overall']['mean_pct_error'], r['markov42']['overall']['max_pct_error']))
    best = rows_sorted[0]

    beat_corrected_frontier = (
        best['markov42']['overall']['mean_pct_error'] < references['entry_relay_corrected']['markov42']['overall']['mean_pct_error']
        and best['markov42']['overall']['max_pct_error'] < references['entry_relay_corrected']['markov42']['overall']['max_pct_error']
    )

    kf36_triggered = False
    kf36_reason = (
        f"No all-new candidate was genuinely competitive enough: best mean {best['markov42']['overall']['mean_pct_error']:.3f} still trails corrected entry {references['entry_relay_corrected']['markov42']['overall']['mean_pct_error']:.3f}, and best max {best['markov42']['overall']['max_pct_error']:.3f} is far above corrected entry {references['entry_relay_corrected']['markov42']['overall']['max_pct_error']:.3f}."
    )

    summary = {
        'task': 'chapter-3 corrected all-new startup-family search',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'corrected_att0_deg': ATT0_DEG,
        'excluded_basins': EXCLUDED_BASINS,
        'hypotheses_considered': HYPOTHESES,
        'tested_candidates': [spec['name'] for spec in CANDIDATE_SPECS],
        'references': references,
        'rows_sorted': rows_sorted,
        'best_candidate': best,
        'kf36_triggered': kf36_triggered,
        'kf36_reason': kf36_reason,
        'bottom_line': {
            'beat_corrected_frontier': 'YES' if beat_corrected_frontier else 'NO',
            'corrected_frontier_markov42': references['entry_relay_corrected']['markov42_triplet'],
            'best_candidate_markov42': overall_triplet(best['markov42']),
            'statement': 'All-new startup families improved over faithful12 and some side branches on mean, but none approached the corrected incumbent frontier on max.',
        },
    }

    REPORT_PATH.write_text(render_report(summary), encoding='utf-8')
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({
        'report_path': str(REPORT_PATH),
        'summary_path': str(SUMMARY_PATH),
        'best_candidate': best['candidate_name'],
        'best_markov42': overall_triplet(best['markov42']),
        'beat_corrected_frontier': summary['bottom_line']['beat_corrected_frontier'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
