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
from search_ch3_12pos_closedloop_local_insertions import build_closedloop_candidate, render_action, run_candidate_payload
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate
from search_ch3_entry_conditioned_relay_family import NOISE_SCALE, xpair_outerhold
from search_ch3_corrected_inbasin_ridge_resume import ATT0_DEG

REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
REPORT_PATH = REPORTS_DIR / f'psins_ch3_corrected_faithful12_vs_anchor11_xpair_only_{REPORT_DATE}.md'
SUMMARY_PATH = RESULTS_DIR / f'ch3_corrected_faithful12_vs_anchor11_xpair_only_{REPORT_DATE}.json'
FAITHFUL_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
FAITHFUL_KF = RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json'


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


def attach_metadata(path: Path, payload: dict[str, Any], candidate_name: str, method_key: str) -> dict[str, Any]:
    extra = payload.setdefault('extra', {})
    extra['att0_deg'] = ATT0_DEG
    extra['comparison_mode'] = 'corrected_faithful12_vs_anchor11_xpair_only'
    extra['candidate_registry_key'] = candidate_name
    extra['method_key'] = method_key
    extra['hard_constraints'] = {
        'keep_original_faithful12_backbone_unchanged': True,
        'insert_only_anchor11_xpair_outerhold_after_anchor11_before_anchor12': True,
        'real_dual_axis_legality_only': True,
        'continuity_safe_execution': True,
        'no_other_motifs_mixed': True,
        'same_shared_low_noise_setup': True,
        'base_row_timing_preserved': True,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
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
            'dKa_xx': float(payload['param_errors']['dKa_xx']['pct_error']),
            'dKg_yy': float(payload['param_errors']['dKg_yy']['pct_error']),
        },
        'max_driver': max(
            ({'name': k, 'pct_error': float(v['pct_error'])} for k, v in payload['param_errors'].items()),
            key=lambda x: x['pct_error'],
        ),
    }


def overall_triplet(payload: dict[str, Any]) -> str:
    o = payload['overall']
    return f"{o['mean_pct_error']:.3f} / {o['median_pct_error']:.3f} / {o['max_pct_error']:.3f}"


def delta_vs_reference(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        out[metric] = float(reference['overall'][metric]) - float(candidate['overall'][metric])
    return out


def build_timing_table(candidate) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    for row, action in zip(candidate.all_rows, candidate.all_actions):
        table.append({
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
    return table


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
    cand = summary['candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 corrected faithful12 vs anchor11 xpair-only comparison')
    lines.append('')
    lines.append('## 1. Exact comparison requested')
    lines.append('')
    lines.append('- Path A: original corrected faithful12.')
    lines.append('- Path B: faithful12 + **only** the 4-step anchor11 xpair outerhold motif.')
    lines.append('- Fixed constraints enforced: **att0=(0,0,0)**, real dual-axis legality only, exact continuity-safe closure before anchor12 resumes, same shared low-noise setup, and **no anchor8 / anchor9 / anchor10 / anchor12 additions**.')
    lines.append('- Timing rule used here: keep the original faithful12 base-row timing exactly, and append only the shared xpair-only motif timing already used in the corrected builder line.')
    lines.append('')
    lines.append('## 2. Headline result')
    lines.append('')
    lines.append(f"- corrected faithful12 Markov42: **{refs['faithful12']['markov42_triplet']}**")
    lines.append(f"- corrected faithful12 KF36: **{refs['faithful12']['kf36_triplet']}**")
    lines.append(f"- xpair-only candidate Markov42: **{overall_triplet(cand['markov42'])}**")
    lines.append(f"- xpair-only candidate KF36: **{overall_triplet(cand['kf36'])}**")
    lines.append('')
    lines.append('## 3. Direct side-by-side')
    lines.append('')
    lines.append('| path | total_time_s | Markov42 | KF36 | max driver (Markov42) |')
    lines.append('|---|---:|---:|---:|---|')
    lines.append(f"| faithful12 | {refs['faithful12']['total_time_s']:.3f} | {refs['faithful12']['markov42_triplet']} | {refs['faithful12']['kf36_triplet']} | {refs['faithful12']['markov42']['max_driver']['name']} {refs['faithful12']['markov42']['max_driver']['pct_error']:.3f} |")
    lines.append(f"| faithful12 + only anchor11 xpair outerhold | {cand['total_time_s']:.3f} | {overall_triplet(cand['markov42'])} | {overall_triplet(cand['kf36'])} | {cand['markov42']['max_driver']['name']} {cand['markov42']['max_driver']['pct_error']:.3f} |")
    lines.append('')
    lines.append('| comparison | Δmean | Δmedian | Δmax | interpretation |')
    lines.append('|---|---:|---:|---:|---|')
    lines.append(
        f"| Markov42: xpair-only minus faithful12 | {cand['delta_vs_faithful12_markov42']['mean_pct_error']:+.3f} | {cand['delta_vs_faithful12_markov42']['median_pct_error']:+.3f} | {cand['delta_vs_faithful12_markov42']['max_pct_error']:+.3f} | {'better' if cand['beats_faithful12_on_markov42_mean_and_max'] else 'mixed / worse'} |"
    )
    lines.append(
        f"| KF36: xpair-only minus faithful12 | {cand['delta_vs_faithful12_kf36']['mean_pct_error']:+.3f} | {cand['delta_vs_faithful12_kf36']['median_pct_error']:+.3f} | {cand['delta_vs_faithful12_kf36']['max_pct_error']:+.3f} | {'better' if cand['beats_faithful12_on_kf36_mean_and_max'] else 'mixed / worse'} |"
    )
    lines.append('')
    lines.append('## 4. Continuity / legality check')
    lines.append('')
    for check in cand['continuity_checks']:
        before = check['state_before_insertion']
        after = check['state_after_insertion']
        preview = check['next_base_action_preview']
        lines.append(f"- anchor {check['anchor_id']} exact closure before anchor12 resume: **{'yes' if check['closure_ok'] else 'no'}**")
        lines.append(f"  - before insertion: beta={before['beta_deg']}°, outer_axis={before['outer_axis_body']}, face={before['face_name']}")
        lines.append(f"  - after insertion : beta={after['beta_deg']}°, outer_axis={after['outer_axis_body']}, face={after['face_name']}")
        if preview is not None:
            lines.append(f"  - anchor12 resume preview: `{preview['kind']}` {preview['motor_angle_deg']:+d}° with effective axis {preview['effective_body_axis']}")
    lines.append('')
    lines.append('## 5. Exact legal motor / timing table for the xpair-only candidate')
    lines.append('')
    lines.extend(render_timing_table_md(summary['xpair_only_timing_table']))
    lines.append('')
    lines.append('## 6. Bottom line')
    lines.append('')
    lines.append(f"- Markov42 verdict: **{summary['bottom_line']['markov42_statement']}**")
    lines.append(f"- KF36 verdict: **{summary['bottom_line']['kf36_statement']}**")
    lines.append(f"- Overall: **{summary['bottom_line']['overall_statement']}**")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module(str(METHOD_DIR / 'method_42state_gm1.py'), str(SOURCE_FILE))
    faithful = build_candidate(mod, ())
    faithful_markov_payload = load_reference_payload(FAITHFUL_MARKOV, args.noise_scale)
    faithful_kf_payload = load_reference_payload(FAITHFUL_KF, args.noise_scale)

    spec = {
        'name': 'anchor11_xpair_outerhold_only_on_faithful12',
        'family': 'xpair_only_comparison',
        'hypothesis_id': 'X1',
        'rationale': 'Keep the original faithful12 backbone unchanged and insert only the 4-step anchor11 xpair outerhold motif after anchor11 and before anchor12.',
        'insertions': {11: xpair_outerhold(10.0, 'l11_xpair_outerhold')},
    }
    candidate = build_closedloop_candidate(mod, spec, faithful.rows, faithful.action_sequence)

    markov_payload, markov_mode, markov_path = run_candidate_payload(mod, candidate, 'markov42_noisy', args.noise_scale, force_rerun=args.force_rerun)
    markov_payload = attach_metadata(markov_path, markov_payload, candidate.name, 'markov42_noisy')
    kf_payload, kf_mode, kf_path = run_candidate_payload(mod, candidate, 'kf36_noisy', args.noise_scale, force_rerun=args.force_rerun)
    kf_payload = attach_metadata(kf_path, kf_payload, candidate.name, 'kf36_noisy')

    faithful_markov = compact_metrics(faithful_markov_payload)
    faithful_kf = compact_metrics(faithful_kf_payload)
    candidate_markov = compact_metrics(markov_payload)
    candidate_kf = compact_metrics(kf_payload)

    summary = {
        'task': 'corrected faithful12 vs faithful12 + only anchor11 xpair outerhold',
        'report_date': REPORT_DATE,
        'noise_scale': args.noise_scale,
        'corrected_att0_deg': ATT0_DEG,
        'hard_constraints': {
            'keep_original_faithful12_backbone_unchanged': True,
            'insert_only_anchor11_xpair_outerhold_after_anchor11_before_anchor12': True,
            'real_dual_axis_legality_only': True,
            'continuity_safe_execution': True,
            'same_shared_low_noise_setup': True,
            'no_other_motifs_mixed': True,
            'same_base_row_timing_for_original_12_steps': True,
        },
        'references': {
            'faithful12': {
                'label': 'corrected faithful12',
                'markov42': faithful_markov,
                'kf36': faithful_kf,
                'markov42_triplet': overall_triplet(faithful_markov),
                'kf36_triplet': overall_triplet(faithful_kf),
                'total_time_s': float(sum(row['node_total_s'] for row in faithful.rows)),
                'files': {
                    'markov42': str(FAITHFUL_MARKOV),
                    'kf36': str(FAITHFUL_KF),
                },
            }
        },
        'candidate': {
            'candidate_name': candidate.name,
            'family': spec['family'],
            'hypothesis_id': spec['hypothesis_id'],
            'rationale': spec['rationale'],
            'total_time_s': float(candidate.total_time_s),
            'markov42': candidate_markov,
            'kf36': candidate_kf,
            'continuity_checks': candidate.continuity_checks,
            'result_files': {
                'markov42': str(markov_path),
                'kf36': str(kf_path),
            },
            'result_modes': {
                'markov42': markov_mode,
                'kf36': kf_mode,
            },
            'delta_vs_faithful12_markov42': delta_vs_reference(faithful_markov, candidate_markov),
            'delta_vs_faithful12_kf36': delta_vs_reference(faithful_kf, candidate_kf),
        },
        'xpair_only_timing_table': build_timing_table(candidate),
    }

    summary['candidate']['beats_faithful12_on_markov42_mean_and_max'] = (
        summary['candidate']['delta_vs_faithful12_markov42']['mean_pct_error'] > 0
        and summary['candidate']['delta_vs_faithful12_markov42']['max_pct_error'] > 0
    )
    summary['candidate']['beats_faithful12_on_kf36_mean_and_max'] = (
        summary['candidate']['delta_vs_faithful12_kf36']['mean_pct_error'] > 0
        and summary['candidate']['delta_vs_faithful12_kf36']['max_pct_error'] > 0
    )

    if summary['candidate']['beats_faithful12_on_markov42_mean_and_max']:
        markov42_statement = 'xpair-only is a genuine improvement over corrected faithful12 on the Markov42 mean+max gate.'
    else:
        markov42_statement = 'xpair-only does not clear corrected faithful12 on the Markov42 mean+max gate.'
    if summary['candidate']['beats_faithful12_on_kf36_mean_and_max']:
        kf36_statement = 'KF36 confirms the same mean+max improvement direction.'
    else:
        kf36_statement = 'KF36 does not confirm a clean mean+max improvement.'

    overall_statement = (
        'Under the original faithful12 timing convention, adding only the anchor11 xpair outerhold motif is still a clean positive move versus corrected faithful12.'
        if summary['candidate']['beats_faithful12_on_markov42_mean_and_max'] and summary['candidate']['beats_faithful12_on_kf36_mean_and_max']
        else 'Under the original faithful12 timing convention, the xpair-only insertion is not a clean standalone win.'
    )
    summary['bottom_line'] = {
        'markov42_statement': markov42_statement,
        'kf36_statement': kf36_statement,
        'overall_statement': overall_statement,
    }
    summary['files'] = {
        'report_md': str(REPORT_PATH),
        'summary_json': str(SUMMARY_PATH),
        'markov42_run_json': str(markov_path),
        'kf36_run_json': str(kf_path),
    }

    REPORT_PATH.write_text(render_report(summary), encoding='utf-8')
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({
        'report_path': str(REPORT_PATH),
        'summary_path': str(SUMMARY_PATH),
        'markov42': overall_triplet(candidate_markov),
        'kf36': overall_triplet(candidate_kf),
        'delta_vs_faithful12_markov42': summary['candidate']['delta_vs_faithful12_markov42'],
        'delta_vs_faithful12_kf36': summary['candidate']['delta_vs_faithful12_kf36'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
