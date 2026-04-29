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
from search_ch3_corrected_inbasin_ridge_resume import compact_metrics, delta_vs_reference, overall_triplet

NOISE_SCALE = 0.08
REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
ATT0_DEG = [0.0, 0.0, 0.0]
ROT_S = 6.0
PRE_S = 6.0
BASE_ROW_TOTAL_S = 60.0
BASE_POST_S = BASE_ROW_TOTAL_S - ROT_S - PRE_S
TARGET_ROWS = 20
TARGET_TOTAL_S = 1200.0
COMPARISON_MODE = 'corrected_symmetric20_followup'

REPORT_PATH = REPORTS_DIR / f'psins_ch3_corrected_symmetric20_followup_{REPORT_DATE}.md'
SUMMARY_PATH = RESULTS_DIR / f'ch3_corrected_symmetric20_followup_{REPORT_DATE}.json'

REFERENCE_FILES = {
    'faithful12': {
        'label': 'corrected faithful12',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json',
    },
    'prior_unified16': {
        'label': 'prior corrected best 16-step leader / anchor11_xpair_outerhold_unified16_75s',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3corrected_anchor11_xpair_outerhold_unified16_75s_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3corrected_anchor11_xpair_outerhold_unified16_75s_shared_noise0p08_param_errors.json',
    },
    'default18': {
        'label': 'default18',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json',
    },
    'disk_symmetric20': {
        'label': 'existing on-disk symmetric20 rerun / anchor2_zpair_anchor11_xpair_symmetric20_60s',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_anchor2_zpair_anchor11_xpair_symmetric20_60s_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_anchor2_zpair_anchor11_xpair_symmetric20_60s_shared_noise0p08_param_errors.json',
    },
}

USER_CURRENT_LEADER = {
    'label': 'requester-specified current symmetric20 leader',
    'markov42': {'overall': {'mean_pct_error': 0.876, 'median_pct_error': 0.425, 'max_pct_error': 4.292}},
    'kf36': {'overall': {'mean_pct_error': 0.876, 'median_pct_error': 0.424, 'max_pct_error': 4.292}},
}

HYPOTHESES = [
    {
        'id': 'A1',
        'family': 'symmetry_preserving_timing_redistribution',
        'summary': 'Keep the exact 20-step mirrored structure, but move dwell budget from the inner open/close rows into the outer sweep/return rows. If the gain mainly comes from extra x/z-family observation time, outer-heavy timing should help.',
        'candidate_name': 'anchor2_zpair_anchor11_xpair_symmetric20_outerheavy_50_70',
    },
    {
        'id': 'A2',
        'family': 'symmetry_preserving_timing_redistribution',
        'summary': 'Mirror test of A1: push more time into the inner open/close rows and lighten the outer sweep/return rows. This checks whether the benefit is really from inner beta-gating rather than late outer dwell.',
        'candidate_name': 'anchor2_zpair_anchor11_xpair_symmetric20_innerheavy_70_50',
    },
    {
        'id': 'B1',
        'family': 'front_anchor2_soft_counterpart',
        'summary': 'Keep the same anchor11 back motif, but flip only the anchor2 z-pair sweep sense under the same legal open/close envelope. This is the lightest check of whether the exact mirrored front sweep direction was over-committed.',
        'candidate_name': 'anchor2_zpair_softrev_anchor11_xpair_symmetric20_60s',
    },
    {
        'id': 'B2',
        'family': 'front_back_sense_flipped_symmetry',
        'summary': 'Flip both the front z-pair and the back x-pair sweep senses while keeping the full front/back symmetry concept and uniform 60 s rows. This checks whether the 20-step idea is sound but the selected sweep orientation is not.',
        'candidate_name': 'anchor2_zpair_softrev_anchor11_xpair_softrev_symmetric20_60s',
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



def make_step(kind: str, angle_deg: int, row_total_s: float, role: str, label: str) -> StepSpec:
    post = float(row_total_s) - ROT_S - PRE_S
    if post < -1e-9:
        raise ValueError(f'row_total_s too small: {row_total_s}')
    return StepSpec(kind=kind, angle_deg=int(angle_deg), rotation_time_s=ROT_S, pre_static_s=PRE_S, post_static_s=post, segment_role=role, label=label)



def front_motif(row_totals: list[float], outer_signs: tuple[int, int], prefix: str) -> list[StepSpec]:
    return [
        make_step('inner', +90, row_totals[0], 'motif_inner_open', f'{prefix}_inner_open'),
        make_step('outer', outer_signs[0], row_totals[1], 'motif_outer_sweep', f'{prefix}_outer_sweep'),
        make_step('outer', outer_signs[1], row_totals[2], 'motif_outer_return', f'{prefix}_outer_return'),
        make_step('inner', -90, row_totals[3], 'motif_inner_close', f'{prefix}_inner_close'),
    ]



def back_motif(row_totals: list[float], outer_signs: tuple[int, int], prefix: str) -> list[StepSpec]:
    return [
        make_step('inner', -90, row_totals[0], 'motif_inner_open', f'{prefix}_inner_open'),
        make_step('outer', outer_signs[0], row_totals[1], 'motif_outer_sweep', f'{prefix}_outer_sweep'),
        make_step('outer', outer_signs[1], row_totals[2], 'motif_outer_return', f'{prefix}_outer_return'),
        make_step('inner', +90, row_totals[3], 'motif_inner_close', f'{prefix}_inner_close'),
    ]



def base_rows_60s(mod):
    faithful = build_candidate(mod, ())
    rows = []
    for row in faithful.rows:
        new_row = dict(row)
        new_row['rotation_time_s'] = float(ROT_S)
        new_row['pre_static_s'] = float(PRE_S)
        new_row['post_static_s'] = float(BASE_POST_S)
        new_row['node_total_s'] = float(BASE_ROW_TOTAL_S)
        rows.append(new_row)
    return faithful, rows



def candidate_specs() -> list[dict[str, Any]]:
    uniform = [60.0, 60.0, 60.0, 60.0]
    outer_heavy = [50.0, 70.0, 70.0, 50.0]
    inner_heavy = [70.0, 50.0, 50.0, 70.0]
    return [
        {
            'name': 'anchor2_zpair_anchor11_xpair_symmetric20_outerheavy_50_70',
            'hypothesis_id': 'A1',
            'family': 'symmetry_preserving_timing_redistribution',
            'rationale': HYPOTHESES[0]['summary'],
            'insertions': {
                2: front_motif(outer_heavy, (-90, +90), 'anchor2_zpair_outerheavy'),
                11: back_motif(outer_heavy, (+90, -90), 'anchor11_xpair_outerheavy'),
            },
        },
        {
            'name': 'anchor2_zpair_anchor11_xpair_symmetric20_innerheavy_70_50',
            'hypothesis_id': 'A2',
            'family': 'symmetry_preserving_timing_redistribution',
            'rationale': HYPOTHESES[1]['summary'],
            'insertions': {
                2: front_motif(inner_heavy, (-90, +90), 'anchor2_zpair_innerheavy'),
                11: back_motif(inner_heavy, (+90, -90), 'anchor11_xpair_innerheavy'),
            },
        },
        {
            'name': 'anchor2_zpair_softrev_anchor11_xpair_symmetric20_60s',
            'hypothesis_id': 'B1',
            'family': 'front_anchor2_soft_counterpart',
            'rationale': HYPOTHESES[2]['summary'],
            'insertions': {
                2: front_motif(uniform, (+90, -90), 'anchor2_zpair_softrev'),
                11: back_motif(uniform, (+90, -90), 'anchor11_xpair_reference'),
            },
        },
        {
            'name': 'anchor2_zpair_softrev_anchor11_xpair_softrev_symmetric20_60s',
            'hypothesis_id': 'B2',
            'family': 'front_back_sense_flipped_symmetry',
            'rationale': HYPOTHESES[3]['summary'],
            'insertions': {
                2: front_motif(uniform, (+90, -90), 'anchor2_zpair_softrev'),
                11: back_motif(uniform, (-90, +90), 'anchor11_xpair_softrev'),
            },
        },
    ]



def build_candidate_from_spec(mod, spec: dict[str, Any]):
    faithful, rows = base_rows_60s(mod)
    cand = build_closedloop_candidate(mod, spec, rows, faithful.action_sequence)
    if len(cand.all_rows) != TARGET_ROWS:
        raise ValueError(f'{cand.name}: expected {TARGET_ROWS} rows, got {len(cand.all_rows)}')
    if abs(cand.total_time_s - TARGET_TOTAL_S) > 1e-9:
        raise ValueError(f'{cand.name}: expected {TARGET_TOTAL_S}s, got {cand.total_time_s}s')
    return cand



def attach_metadata(path: Path, payload: dict[str, Any], *, candidate_name: str, family: str, hypothesis_id: str, method_key: str) -> dict[str, Any]:
    extra = payload.setdefault('extra', {})
    extra['att0_deg'] = ATT0_DEG
    extra['comparison_mode'] = COMPARISON_MODE
    extra['candidate_registry_key'] = candidate_name
    extra['family'] = family
    extra['hypothesis_id'] = hypothesis_id
    extra['method_key'] = method_key
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload



def build_timing_table(candidate) -> list[dict[str, Any]]:
    table = []
    for row, action in zip(candidate.all_rows, candidate.all_actions):
        table.append({
            'pos_id': int(row['pos_id']),
            'anchor_id': int(row['anchor_id']),
            'segment_role': row['segment_role'],
            'label': row['label'],
            'motor_action': render_action(action),
            'effective_body_axis': list(action['effective_body_axis']),
            'angle_deg': float(row['angle_deg']),
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
    lines.append('| seq | anchor | role | label | motor action | axis | angle_deg | rot_s | pre_s | post_s | total_s | face_after | beta_after |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---|---:|')
    for item in table:
        axis = '[' + ', '.join(str(v) for v in item['effective_body_axis']) + ']'
        lines.append(
            f"| {item['pos_id']} | {item['anchor_id']} | {item['segment_role']} | {item['label']} | {item['motor_action']} | {axis} | {item['angle_deg']:+.0f} | {item['rotation_time_s']:.1f} | {item['pre_static_s']:.1f} | {item['post_static_s']:.1f} | {item['node_total_s']:.1f} | {item['face_after']} | {item['inner_beta_after_deg']} |"
        )
    return lines



def render_report(summary: dict[str, Any]) -> str:
    refs = summary['references']
    best = summary['best_new_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 corrected symmetric20 family follow-up')
    lines.append('')
    lines.append('## 1. Scope and incumbent used for the gate')
    lines.append('')
    lines.append(f"- Requester-specified incumbent gate: **{summary['requester_gate_markov42']}** (Markov42), **{summary['requester_gate_kf36']}** (KF36).")
    lines.append(f"- Existing on-disk symmetric20 rerun still reads: **{refs['disk_symmetric20']['markov42_triplet']}** (Markov42), **{refs['disk_symmetric20']['kf36_triplet']}** (KF36).")
    lines.append('- This batch therefore used the requester threshold as the formal pass/fail gate, while also checking whether any follow-up variant could beat the stronger on-disk symmetric20 rerun.')
    lines.append('')
    lines.append('## 2. Focused batch actually run')
    lines.append('')
    for item in summary['hypotheses_tested']:
        lines.append(f"- **{item['id']} · {item['family']}** — {item['summary']}")
        lines.append(f"  - tested: `{item['candidate_name']}`")
    lines.append('')
    lines.append('## 3. Markov42 results')
    lines.append('')
    lines.append('| rank | candidate | mean | median | max | Δmean vs requester gate | Δmedian vs requester gate | Δmax vs requester gate | Δmean vs disk symmetric20 | Δmax vs disk symmetric20 | max driver |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for idx, row in enumerate(summary['rows_sorted'], start=1):
        m = row['markov42']['overall']
        dg = row['delta_vs_requester_gate_markov42']
        dd = row['delta_vs_disk_symmetric20_markov42']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {dg['mean_pct_error']:+.3f} | {dg['median_pct_error']:+.3f} | {dg['max_pct_error']:+.3f} | {dd['mean_pct_error']:+.3f} | {dd['max_pct_error']:+.3f} | {row['markov42']['max_driver']['name']} {row['markov42']['max_driver']['pct_error']:.3f} |"
        )
    lines.append('')
    lines.append('## 4. KF36 recheck')
    lines.append('')
    lines.append(f"- Rechecked candidates: **{', '.join(summary['kf36_rechecked_candidates'])}**")
    for row in summary['rows_sorted']:
        if row.get('kf36') is not None:
            lines.append(f"  - `{row['candidate_name']}` → Markov42 **{overall_triplet(row['markov42'])}**, KF36 **{overall_triplet(row['kf36'])}**")
    lines.append('')
    lines.append('## 5. Best new candidate exact legal motor / timing table')
    lines.append('')
    lines.append(f"- Best new candidate in this microbatch: **`{best['candidate_name']}`**")
    lines.append(f"- Markov42: **{overall_triplet(best['markov42'])}**")
    if best.get('kf36') is not None:
        lines.append(f"- KF36: **{overall_triplet(best['kf36'])}**")
    lines.append('')
    lines.extend(render_timing_table_md(best['timing_table']))
    lines.append('')
    lines.append('## 6. Comparison vs required references')
    lines.append('')
    lines.append('| reference | Markov42 triplet | best-new Δmean | best-new Δmedian | best-new Δmax |')
    lines.append('|---|---|---:|---:|---:|')
    for key in ['current_symmetric20_requester_gate', 'faithful12', 'prior_unified16', 'default18', 'disk_symmetric20']:
        ref = refs[key]
        delta = best['markov42_deltas'][key]
        lines.append(
            f"| {ref['label']} | {ref['markov42_triplet']} | {delta['mean_pct_error']:+.3f} | {delta['median_pct_error']:+.3f} | {delta['max_pct_error']:+.3f} |"
        )
    lines.append('')
    lines.append('## 7. Bottom line')
    lines.append('')
    lines.append(f"- {summary['bottom_line']}")
    lines.append('')
    return '\n'.join(lines) + '\n'



def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module('search_ch3_corrected_symmetric20_followup_src', str(SOURCE_FILE))

    refs: dict[str, Any] = {
        'current_symmetric20_requester_gate': {
            'label': USER_CURRENT_LEADER['label'],
            'markov42': USER_CURRENT_LEADER['markov42'],
            'kf36': USER_CURRENT_LEADER['kf36'],
            'markov42_triplet': overall_triplet(USER_CURRENT_LEADER['markov42']),
            'kf36_triplet': overall_triplet(USER_CURRENT_LEADER['kf36']),
            'files': None,
        }
    }
    for key, cfg in REFERENCE_FILES.items():
        markov = load_reference_payload(cfg['markov42'], args.noise_scale)
        kf = load_reference_payload(cfg['kf36'], args.noise_scale)
        refs[key] = {
            'label': cfg['label'],
            'markov42': compact_metrics(markov),
            'kf36': compact_metrics(kf),
            'markov42_triplet': overall_triplet(compact_metrics(markov)),
            'kf36_triplet': overall_triplet(compact_metrics(kf)),
            'files': {'markov42': str(cfg['markov42']), 'kf36': str(cfg['kf36'])},
        }

    rows = []
    built_candidates = {}
    for spec in candidate_specs():
        candidate = build_candidate_from_spec(mod, spec)
        built_candidates[spec['name']] = candidate
        markov_payload, markov_status, markov_path = run_candidate_payload(mod, candidate, 'markov42_noisy', args.noise_scale, args.force_rerun)
        markov_payload = attach_metadata(markov_path, markov_payload, candidate_name=spec['name'], family=spec['family'], hypothesis_id=spec['hypothesis_id'], method_key='markov42_noisy')
        markov = compact_metrics(markov_payload)
        row = {
            'candidate_name': spec['name'],
            'hypothesis_id': spec['hypothesis_id'],
            'family': spec['family'],
            'rationale': spec['rationale'],
            'markov42': markov,
            'markov42_status': markov_status,
            'markov42_file': str(markov_path),
            'continuity_checks': candidate.continuity_checks,
            'timing_table': build_timing_table(candidate),
            'row_count': len(candidate.all_rows),
            'total_time_s': candidate.total_time_s,
            'method_tag': candidate.method_tag,
        }
        row['delta_vs_requester_gate_markov42'] = delta_vs_reference(refs['current_symmetric20_requester_gate']['markov42'], markov)
        row['delta_vs_disk_symmetric20_markov42'] = delta_vs_reference(refs['disk_symmetric20']['markov42'], markov)
        rows.append(row)

    rows.sort(key=lambda r: (r['markov42']['overall']['mean_pct_error'], r['markov42']['overall']['max_pct_error'], r['markov42']['overall']['median_pct_error']))

    kf36_rechecked = [rows[0]['candidate_name']]
    if len(rows) > 1:
        kf36_rechecked.append(rows[1]['candidate_name'])
    kf36_rechecked = list(dict.fromkeys(kf36_rechecked))

    for row in rows:
        if row['candidate_name'] in kf36_rechecked:
            spec_name = row['candidate_name']
            spec = next(item for item in candidate_specs() if item['name'] == spec_name)
            candidate = built_candidates[spec_name]
            kf_payload, kf_status, kf_path = run_candidate_payload(mod, candidate, 'kf36_noisy', args.noise_scale, args.force_rerun)
            kf_payload = attach_metadata(kf_path, kf_payload, candidate_name=spec['name'], family=spec['family'], hypothesis_id=spec['hypothesis_id'], method_key='kf36_noisy')
            row['kf36'] = compact_metrics(kf_payload)
            row['kf36_status'] = kf_status
            row['kf36_file'] = str(kf_path)

    best = rows[0]
    best['markov42_deltas'] = {}
    for key, ref in refs.items():
        best['markov42_deltas'][key] = delta_vs_reference(ref['markov42'], best['markov42'])

    beats_requester_gate = (
        best['markov42']['overall']['mean_pct_error'] < refs['current_symmetric20_requester_gate']['markov42']['overall']['mean_pct_error']
        and best['markov42']['overall']['median_pct_error'] < refs['current_symmetric20_requester_gate']['markov42']['overall']['median_pct_error']
        and best['markov42']['overall']['max_pct_error'] < refs['current_symmetric20_requester_gate']['markov42']['overall']['max_pct_error']
    )
    beats_disk_baseline = (
        best['markov42']['overall']['mean_pct_error'] < refs['disk_symmetric20']['markov42']['overall']['mean_pct_error']
        and best['markov42']['overall']['max_pct_error'] < refs['disk_symmetric20']['markov42']['overall']['max_pct_error']
    )

    if beats_requester_gate and beats_disk_baseline:
        bottom_line = (
            f"The symmetric20 family can still be pushed further: `{best['candidate_name']}` beats the requester gate 0.876 / 0.425 / 4.292 and also clears the stronger on-disk symmetric20 rerun on the Markov42 mean/max gate."
        )
    elif beats_requester_gate:
        bottom_line = (
            f"Relative to the requester gate 0.876 / 0.425 / 4.292, `{best['candidate_name']}` is better, but none of the follow-up variants beats the stronger on-disk symmetric20 rerun on the Markov42 mean/max gate."
        )
    elif beats_disk_baseline:
        bottom_line = (
            f"None of the new variants improves the requester gate simultaneously on mean/median/max, but `{best['candidate_name']}` does beat the existing on-disk symmetric20 rerun on the Markov42 mean/max gate."
        )
    else:
        bottom_line = (
            f"No follow-up variant in this focused symmetric20 microbatch improved beyond the requester gate 0.876 / 0.425 / 4.292, and none beat the stronger on-disk symmetric20 rerun either. The best new point was `{best['candidate_name']}` at {overall_triplet(best['markov42'])} (Markov42)."
        )

    summary = {
        'task': 'chapter-3 corrected symmetric20 family follow-up',
        'report_date': REPORT_DATE,
        'noise_scale': args.noise_scale,
        'att0_deg': ATT0_DEG,
        'hard_constraints': {
            'att0_deg': ATT0_DEG,
            'real_dual_axis_legality_only': True,
            'continuity_safe_execution': True,
            'target_total_time_s': TARGET_TOTAL_S,
            'target_rows': TARGET_ROWS,
            'theory_guided_only': True,
        },
        'requester_gate_markov42': refs['current_symmetric20_requester_gate']['markov42_triplet'],
        'requester_gate_kf36': refs['current_symmetric20_requester_gate']['kf36_triplet'],
        'references': refs,
        'hypotheses_tested': HYPOTHESES,
        'rows_sorted': rows,
        'kf36_rechecked_candidates': kf36_rechecked,
        'best_new_candidate': best,
        'bottom_line': bottom_line,
        'files': {
            'report_md': str(REPORT_PATH),
            'summary_json': str(SUMMARY_PATH),
        },
    }

    REPORT_PATH.write_text(render_report(summary), encoding='utf-8')
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps(summary['files'], ensure_ascii=False), flush=True)
    print('BEST_NEW_CANDIDATE', best['candidate_name'], overall_triplet(best['markov42']), flush=True)
    print('BOTTOM_LINE', bottom_line, flush=True)


if __name__ == '__main__':
    main()
