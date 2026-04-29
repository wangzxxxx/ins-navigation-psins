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

from benchmark_ch3_12pos_goalA_repairs import compact_result, rows_to_paras
from common_markov import load_module
from compare_ch3_12pos_path_baselines import build_dataset_with_path
from compare_four_methods_shared_noise import _load_json, _noise_matches, compute_payload, expected_noise_config
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs
from search_ch3_12pos_closedloop_local_insertions import StepSpec, build_closedloop_candidate
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate, make_suffix, render_action

NOISE_SCALE = 0.08
REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
ATT0_DEG = [0.0, 0.0, 0.0]
UNIFIED_ROT_S = 6.0
UNIFIED_PRE_S = 6.0
UNIFIED_POST_S = 48.0
UNIFIED_ROW_TOTAL_S = UNIFIED_ROT_S + UNIFIED_PRE_S + UNIFIED_POST_S
TARGET_TOTAL_S = 1200.0
TARGET_ROWS = 20
CANDIDATE_NAME = 'anchor2_zpair_anchor11_xpair_symmetric20_60s'
COMPARISON_MODE = 'corrected_att0_symmetric20_front2_back11_probe'

FAITHFUL_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
FAITHFUL_KF = RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json'
CURRENT_LOCAL_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l11back0p5_l12y0p125_on_entry_shared_noise0p08_param_errors.json'
CURRENT_LOCAL_KF = RESULTS_DIR / 'KF36_ch3closedloop_relay_l11back0p5_l12y0p125_on_entry_shared_noise0p08_param_errors.json'
PRIOR_UNIFIED16_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3corrected_anchor11_xpair_outerhold_unified16_75s_shared_noise0p08_param_errors.json'
PRIOR_UNIFIED16_KF = RESULTS_DIR / 'KF36_ch3corrected_anchor11_xpair_outerhold_unified16_75s_shared_noise0p08_param_errors.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def triplet_text(compact: dict[str, Any]) -> str:
    overall = compact['overall']
    return f"{overall['mean_pct_error']:.3f} / {overall['median_pct_error']:.3f} / {overall['max_pct_error']:.3f}"


def delta_vs_reference(ref_payload: dict[str, Any], cand_payload: dict[str, Any]) -> dict[str, Any]:
    out = {}
    ref_compact = compact_result(ref_payload)
    cand_compact = compact_result(cand_payload)
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        rv = float(ref_compact['overall'][metric])
        cv = float(cand_compact['overall'][metric])
        out[metric] = {
            'reference': rv,
            'candidate': cv,
            'improvement_pct_points': rv - cv,
            'relative_improvement_pct': ((rv - cv) / rv * 100.0) if abs(rv) > 1e-12 else None,
        }
    return out


def load_reference_payload(path: Path, noise_scale: float) -> dict[str, Any]:
    payload = _load_json(path)
    if not _noise_matches(payload, expected_noise_config(noise_scale)):
        raise ValueError(f'noise configuration mismatch: {path}')
    return payload


def load_reference_payloads(noise_scale: float) -> dict[str, Any]:
    return {
        'faithful12': {
            'candidate_name': 'ch3faithful12',
            'label': 'corrected faithful12',
            'markov42': load_reference_payload(FAITHFUL_MARKOV, noise_scale),
            'kf36': load_reference_payload(FAITHFUL_KF, noise_scale),
            'files': {
                'markov42': str(FAITHFUL_MARKOV),
                'kf36': str(FAITHFUL_KF),
            },
        },
        'current_corrected_best_local': {
            'candidate_name': 'relay_l11back0p5_l12y0p125_on_entry',
            'label': 'current corrected best local leader',
            'markov42': load_reference_payload(CURRENT_LOCAL_MARKOV, noise_scale),
            'kf36': load_reference_payload(CURRENT_LOCAL_KF, noise_scale),
            'files': {
                'markov42': str(CURRENT_LOCAL_MARKOV),
                'kf36': str(CURRENT_LOCAL_KF),
            },
        },
        'prior_unified16': {
            'candidate_name': 'anchor11_xpair_outerhold_unified16_75s',
            'label': 'prior unified16 probe',
            'markov42': load_reference_payload(PRIOR_UNIFIED16_MARKOV, noise_scale),
            'kf36': load_reference_payload(PRIOR_UNIFIED16_KF, noise_scale),
            'files': {
                'markov42': str(PRIOR_UNIFIED16_MARKOV),
                'kf36': str(PRIOR_UNIFIED16_KF),
            },
        },
    }


def front_mirror_motif() -> list[StepSpec]:
    return [
        StepSpec('inner', +90, UNIFIED_ROT_S, UNIFIED_PRE_S, UNIFIED_POST_S, 'motif_inner_open', 'anchor2_zpair_outerhold_inner_open'),
        StepSpec('outer', -90, UNIFIED_ROT_S, UNIFIED_PRE_S, UNIFIED_POST_S, 'motif_outer_sweep', 'anchor2_zpair_outerhold_outer_sweep'),
        StepSpec('outer', +90, UNIFIED_ROT_S, UNIFIED_PRE_S, UNIFIED_POST_S, 'motif_outer_return', 'anchor2_zpair_outerhold_outer_return'),
        StepSpec('inner', -90, UNIFIED_ROT_S, UNIFIED_PRE_S, UNIFIED_POST_S, 'motif_inner_close', 'anchor2_zpair_outerhold_inner_close'),
    ]


def back_reference_motif() -> list[StepSpec]:
    return [
        StepSpec('inner', -90, UNIFIED_ROT_S, UNIFIED_PRE_S, UNIFIED_POST_S, 'motif_inner_open', 'anchor11_xpair_outerhold_inner_open'),
        StepSpec('outer', +90, UNIFIED_ROT_S, UNIFIED_PRE_S, UNIFIED_POST_S, 'motif_outer_sweep', 'anchor11_xpair_outerhold_outer_sweep'),
        StepSpec('outer', -90, UNIFIED_ROT_S, UNIFIED_PRE_S, UNIFIED_POST_S, 'motif_outer_return', 'anchor11_xpair_outerhold_outer_return'),
        StepSpec('inner', +90, UNIFIED_ROT_S, UNIFIED_PRE_S, UNIFIED_POST_S, 'motif_inner_close', 'anchor11_xpair_outerhold_inner_close'),
    ]


def build_symmetric20_candidate(mod):
    faithful = build_candidate(mod, ())
    base_rows = []
    for row in faithful.rows:
        new_row = dict(row)
        new_row['rotation_time_s'] = float(UNIFIED_ROT_S)
        new_row['pre_static_s'] = float(UNIFIED_PRE_S)
        new_row['post_static_s'] = float(UNIFIED_POST_S)
        new_row['node_total_s'] = float(UNIFIED_ROW_TOTAL_S)
        base_rows.append(new_row)

    spec = {
        'name': CANDIDATE_NAME,
        'rationale': (
            'Restore stronger front/back symmetry under the real two-axis mechanism by pairing the existing late '
            'anchor11 x-family outerhold with its front-half counterpart after anchor2. Under legality, the mirrored '
            'front motif must sweep the z-family (not x-family) because anchor2 inner-open +90 shifts the legal outer '
            'axis from +X to +Z before the outer pair executes.'
        ),
        'insertions': {
            2: front_mirror_motif(),
            11: back_reference_motif(),
        },
    }
    candidate = build_closedloop_candidate(mod, spec, base_rows, faithful.action_sequence)
    if len(candidate.all_rows) != TARGET_ROWS:
        raise ValueError(f'expected {TARGET_ROWS} rows, got {len(candidate.all_rows)}')
    if abs(candidate.total_time_s - TARGET_TOTAL_S) > 1e-9:
        raise ValueError(f'total time mismatch: got {candidate.total_time_s}, want {TARGET_TOTAL_S}')
    return candidate


def candidate_output_path(candidate, method_key: str, noise_scale: float) -> Path:
    prefix = 'M_markov_42state_gm1' if method_key == 'markov42_noisy' else 'KF36'
    return RESULTS_DIR / f'{prefix}_{candidate.method_tag}_shared_{make_suffix(noise_scale)}_param_errors.json'


def run_candidate_payload(mod, candidate, method_key: str, noise_scale: float, force_rerun: bool = False):
    expected_cfg = expected_noise_config(noise_scale)
    out_path = candidate_output_path(candidate, method_key, noise_scale)
    if out_path.exists() and (not force_rerun):
        payload = _load_json(out_path)
        extra = payload.get('extra', {})
        if (
            _noise_matches(payload, expected_cfg)
            and extra.get('candidate_name') == candidate.name
            and int(extra.get('n_rows', -1)) == TARGET_ROWS
            and abs(float(extra.get('time_total_s', -1.0)) - TARGET_TOTAL_S) < 1e-9
            and extra.get('comparison_mode') == COMPARISON_MODE
        ):
            return payload, 'reused_verified', out_path

    paras = rows_to_paras(mod, candidate.all_rows)
    dataset = build_dataset_with_path(mod, noise_scale, paras)
    params = _param_specs(mod)

    if method_key == 'markov42_noisy':
        clbt, _, _, _, _ = mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=42,
            bi_g=dataset['bi_g'],
            tau_g=dataset['tau_g'],
            bi_a=dataset['bi_a'],
            tau_a=dataset['tau_a'],
            label=f'{candidate.name}_{method_key}_{make_suffix(noise_scale)}',
        )
    elif method_key == 'kf36_noisy':
        clbt, _, _, _, _ = mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=36,
            label=f'{candidate.name}_{method_key}_{make_suffix(noise_scale)}',
        )
    else:
        raise KeyError(method_key)

    payload = compute_payload(
        mod,
        clbt,
        params,
        variant=f'{candidate.name}_{method_key}_{make_suffix(noise_scale)}',
        method_file='probe_ch3_corrected_symmetric20_front2_back11.py',
        extra={
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'comparison_mode': COMPARISON_MODE,
            'candidate_name': candidate.name,
            'method_key': method_key,
            'att0_deg': ATT0_DEG,
            'legality': 'true_dual_axis_mechanism_with_exact_anchor2_and_anchor11_closed_loop_insertions',
            'time_total_s': candidate.total_time_s,
            'n_rows': len(candidate.all_rows),
            'row_timing_s': {
                'rotation_time_s': UNIFIED_ROT_S,
                'pre_static_s': UNIFIED_PRE_S,
                'post_static_s': UNIFIED_POST_S,
                'row_total_s': UNIFIED_ROW_TOTAL_S,
            },
            'insertion_anchor_ids': [2, 11],
            'front_counterpart_rule': 'anchor2 mirrored counterpart is z-family outerhold because legal outer axis after inner +90 is +Z',
            'front_motif_signature': 'inner+90 -> outer-90 -> outer+90 -> inner-90',
            'back_motif_signature': 'inner-90 -> outer+90 -> outer-90 -> inner+90',
        },
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', out_path


def render_report(summary: dict[str, Any]) -> str:
    cand = summary['candidate']
    refs = summary['references']
    mapping = summary['symmetry_mapping']
    lines: list[str] = []
    lines.append('# Chapter-3 corrected 20-step symmetric-schedule probe')
    lines.append('')
    lines.append('## 1. What was constructed')
    lines.append('')
    lines.append('- Base scaffold: original corrected chapter-3 faithful 12-step legal backbone.')
    lines.append('- Kept the already-proven late 4-step motif after **anchor11**: **inner open → outer sweep → outer return → inner close**.')
    lines.append('- Added a front-half 4-step counterpart after **anchor2** so the full path becomes **20 rows**.')
    lines.append('- Uniform timing enforced on **all 20 rows** exactly as requested: **20 × 60.0 s = 1200 s**, with per-row **rot/pre/post = 6.0 / 6.0 / 48.0 s**.')
    lines.append('- Hard basis held fixed: **att0 = (0,0,0)**, real dual-axis legality only, continuity-safe exact resume, same shared low-noise setup (`noise_scale=0.08`).')
    lines.append('')
    lines.append('## 2. Why the front motif is the symmetry-matching counterpart')
    lines.append('')
    lines.append(f"- Faithful12 front/back pairing puts **anchor2 ↔ anchor11** as the mirrored outer-anchor pair inside the original 12-step scaffold.")
    lines.append('- The late motif at anchor11 is an **x-family outerhold** only because the legal outer axis after its inner-open step becomes **-X**.')
    lines.append('- At anchor2, the exact mirrored front-half construction requires an inner-open **+90°** first; under the real mechanism this changes the legal outer axis from **+X** to **+Z**.')
    lines.append('- Therefore the legality-preserving front counterpart is not a literal x-pair copy; it is the exact mirrored **z-family outerhold**:')
    lines.append(f"  - front motif: `{mapping['front_motif_signature']}`")
    lines.append(f"  - back motif : `{mapping['back_motif_signature']}`")
    lines.append(f"- This is why the chosen front motif counts as the symmetry match **under the real mechanism**, not under an abstract free-axis mirror.")
    lines.append('')
    lines.append('## 3. Headline result')
    lines.append('')
    lines.append(f"- Probe candidate `{cand['candidate_name']}` Markov42: **{triplet_text(cand['markov42'])}**")
    lines.append(f"- Probe candidate `{cand['candidate_name']}` KF36: **{triplet_text(cand['kf36'])}**")
    lines.append(f"- Corrected faithful12 reference: **{triplet_text(refs['faithful12']['markov42'])}**")
    lines.append(f"- Current corrected best local leader (`{refs['current_corrected_best_local']['candidate_name']}`): **{triplet_text(refs['current_corrected_best_local']['markov42'])}**")
    lines.append(f"- Prior unified16 probe (`{refs['prior_unified16']['candidate_name']}`): **{triplet_text(refs['prior_unified16']['markov42'])}**")
    lines.append('')
    lines.append('## 4. Direct comparison')
    lines.append('')
    lines.append('| comparison | method | reference triplet | candidate triplet | Δmean | Δmedian | Δmax |')
    lines.append('|---|---|---|---|---:|---:|---:|')
    for ref_key, label in [
        ('faithful12', 'vs corrected faithful12'),
        ('current_corrected_best_local', 'vs current corrected best local leader'),
        ('prior_unified16', 'vs prior unified16 probe'),
    ]:
        for method_key, method_label in [('markov42', 'Markov42'), ('kf36', 'KF36')]:
            delta = cand[f'delta_vs_{ref_key}'][method_key]
            lines.append(
                f"| {label} | {method_label} | {triplet_text(refs[ref_key][method_key])} | {triplet_text(cand[method_key])} | "
                f"{delta['mean_pct_error']['improvement_pct_points']:+.3f} | {delta['median_pct_error']['improvement_pct_points']:+.3f} | {delta['max_pct_error']['improvement_pct_points']:+.3f} |"
            )
    lines.append('')
    lines.append('## 5. Continuity / legality check')
    lines.append('')
    for check in cand['continuity_checks']:
        before = check['state_before_insertion']
        after = check['state_after_insertion']
        preview = check['next_base_action_preview']
        anchor_desc = 'front counterpart' if check['anchor_id'] == 2 else 'late reference motif'
        lines.append(f"- anchor {check['anchor_id']} ({anchor_desc}) returns to the exact same mechanism state before resume: **{'yes' if check['closure_ok'] else 'no'}**")
        lines.append(f"  - before insertion: beta={before['beta_deg']}°, outer_axis={before['outer_axis_body']}, face={before['face_name']}")
        lines.append(f"  - after insertion : beta={after['beta_deg']}°, outer_axis={after['outer_axis_body']}, face={after['face_name']}")
        if preview is not None:
            lines.append(
                f"  - next base action preview: `{preview['kind']}` {preview['motor_angle_deg']:+d}° on axis {preview['effective_body_axis']}"
            )
    lines.append('')
    lines.append('## 6. Exact 20-row legal motor / timing table')
    lines.append('')
    lines.append('| seq | source_anchor | role | label | legal motor action | body axis | angle_deg | rot_s | pre_s | post_s | face after row |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|')
    for row, action, face in zip(cand['rows'], cand['actions'], cand['faces']):
        lines.append(
            f"| {row['pos_id']} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {render_action(action)} | {row['axis']} | {row['angle_deg']:+.0f} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {face['face_name']} |"
        )
    lines.append('')
    lines.append('## 7. Bottom line')
    lines.append('')
    lines.append(f"- Conclusion: {summary['bottom_line']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module('probe_ch3_corrected_symmetric20_front2_back11_src', str(SOURCE_FILE))
    refs_raw = load_reference_payloads(args.noise_scale)
    candidate = build_symmetric20_candidate(mod)

    markov_payload, markov_status, markov_path = run_candidate_payload(mod, candidate, 'markov42_noisy', args.noise_scale, args.force_rerun)
    kf_payload, kf_status, kf_path = run_candidate_payload(mod, candidate, 'kf36_noisy', args.noise_scale, args.force_rerun)

    candidate_summary = {
        'candidate_name': candidate.name,
        'method_tag': candidate.method_tag,
        'n_rows': len(candidate.all_rows),
        'total_time_s': candidate.total_time_s,
        'row_timing_s': {
            'rotation_time_s': UNIFIED_ROT_S,
            'pre_static_s': UNIFIED_PRE_S,
            'post_static_s': UNIFIED_POST_S,
            'row_total_s': UNIFIED_ROW_TOTAL_S,
        },
        'rationale': candidate.rationale,
        'rows': candidate.all_rows,
        'actions': candidate.all_actions,
        'faces': candidate.all_faces,
        'continuity_checks': candidate.continuity_checks,
        'markov42': compact_result(markov_payload),
        'markov42_run_json': str(markov_path),
        'markov42_status': markov_status,
        'kf36': compact_result(kf_payload),
        'kf36_run_json': str(kf_path),
        'kf36_status': kf_status,
    }

    references = {}
    for key, entry in refs_raw.items():
        references[key] = {
            'candidate_name': entry['candidate_name'],
            'label': entry['label'],
            'markov42': compact_result(entry['markov42']),
            'kf36': compact_result(entry['kf36']),
            'files': entry['files'],
        }

    for key, entry in refs_raw.items():
        candidate_summary[f'delta_vs_{key}'] = {
            'markov42': delta_vs_reference(entry['markov42'], markov_payload),
            'kf36': delta_vs_reference(entry['kf36'], kf_payload),
        }

    anchor_checks = {item['anchor_id']: item for item in candidate.continuity_checks}
    symmetry_mapping = {
        'base_pairing': 'anchor2 <-> anchor11 under faithful12 front/back reversal',
        'front_anchor': 2,
        'back_anchor': 11,
        'front_motif_signature': 'inner+90 -> outer-90 -> outer+90 -> inner-90',
        'back_motif_signature': 'inner-90 -> outer+90 -> outer-90 -> inner+90',
        'front_before_state': anchor_checks[2]['state_before_insertion'],
        'back_before_state': anchor_checks[11]['state_before_insertion'],
        'why_zpair_not_xpair': (
            'At anchor2, the legal outer axis after the mirrored inner-open +90 step is +Z, so the front counterpart must '
            'be a z-family outerhold; a literal x-family sweep there would not follow the real dual-axis mechanism.'
        ),
    }

    beats_faithful = (
        candidate_summary['markov42']['overall']['mean_pct_error'] < references['faithful12']['markov42']['overall']['mean_pct_error']
        and candidate_summary['markov42']['overall']['max_pct_error'] < references['faithful12']['markov42']['overall']['max_pct_error']
    )
    beats_local = (
        candidate_summary['markov42']['overall']['mean_pct_error'] < references['current_corrected_best_local']['markov42']['overall']['mean_pct_error']
        and candidate_summary['markov42']['overall']['max_pct_error'] < references['current_corrected_best_local']['markov42']['overall']['max_pct_error']
    )
    beats_unified16 = (
        candidate_summary['markov42']['overall']['mean_pct_error'] < references['prior_unified16']['markov42']['overall']['mean_pct_error']
        and candidate_summary['markov42']['overall']['max_pct_error'] < references['prior_unified16']['markov42']['overall']['max_pct_error']
    )

    if beats_unified16:
        bottom_line = (
            'The corrected symmetric 20-step probe is a real frontier improvement: the front anchor2 z-family counterpart plus '
            'the retained anchor11 x-family motif beats even the prior unified16 probe on both Markov42 mean and max.'
        )
    elif beats_local:
        bottom_line = (
            'The corrected symmetric 20-step probe remains scientifically real and beats the old corrected local leader, but the '
            'extra front-half symmetry does not surpass the prior unified16 probe on the combined Markov42 mean/max gate.'
        )
    elif beats_faithful:
        bottom_line = (
            'The corrected symmetric 20-step probe improves on corrected faithful12, but the added front-half symmetry does not '
            'recover enough benefit to beat the old corrected local leader or the prior unified16 probe.'
        )
    else:
        bottom_line = (
            'The corrected symmetric 20-step probe does not clear corrected faithful12 on the combined Markov42 mean/max gate; '
            'under the real mechanism, forcing a front-half symmetry mate to the late anchor11 motif over-regularizes the path.'
        )

    summary = {
        'task': 'chapter-3 corrected 20-step symmetric front2/back11 probe',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'corrected_att0_deg': ATT0_DEG,
        'hard_constraints': {
            'att0_deg': ATT0_DEG,
            'real_dual_axis_legality_only': True,
            'continuity_safe_execution': True,
            'total_time_s': TARGET_TOTAL_S,
            'same_low_noise_shared_setup': True,
            'uniform_row_timing': {
                'rotation_time_s': UNIFIED_ROT_S,
                'pre_static_s': UNIFIED_PRE_S,
                'post_static_s': UNIFIED_POST_S,
                'row_total_s': UNIFIED_ROW_TOTAL_S,
            },
            'total_rows': TARGET_ROWS,
            'front_half_counterpart_rule': 'anchor2 legal z-family outerhold as the mirrored counterpart of anchor11 x-family outerhold',
        },
        'references': references,
        'candidate': candidate_summary,
        'symmetry_mapping': symmetry_mapping,
        'bottom_line': bottom_line,
        'comparative_flags': {
            'beats_corrected_faithful12_on_markov42_mean_and_max': beats_faithful,
            'beats_current_corrected_best_local_on_markov42_mean_and_max': beats_local,
            'beats_prior_unified16_on_markov42_mean_and_max': beats_unified16,
        },
    }

    summary_path = RESULTS_DIR / f'ch3_corrected_symmetric20_front2_back11_probe_{args.report_date}_summary.json'
    report_path = REPORTS_DIR / f'psins_ch3_corrected_symmetric20_front2_back11_probe_{args.report_date}.md'
    summary['files'] = {
        'summary_json': str(summary_path),
        'report_md': str(report_path),
        'markov42_run_json': str(markov_path),
        'kf36_run_json': str(kf_path),
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_path.write_text(render_report(summary), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps(summary['files'], ensure_ascii=False), flush=True)
    print('CANDIDATE_MARKOV42', triplet_text(candidate_summary['markov42']), flush=True)
    print('CANDIDATE_KF36', triplet_text(candidate_summary['kf36']), flush=True)
    print('BOTTOM_LINE', bottom_line, flush=True)


if __name__ == '__main__':
    main()
