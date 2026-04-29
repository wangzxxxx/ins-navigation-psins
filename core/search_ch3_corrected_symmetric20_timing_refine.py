from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import types
from dataclasses import dataclass
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
from probe_ch3_corrected_symmetric20_front2_back11 import build_symmetric20_candidate
from search_ch3_12pos_legal_dualaxis_repairs import make_suffix, render_action

NOISE_SCALE = 0.08
REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
ATT0_DEG = [0.0, 0.0, 0.0]
TARGET_TOTAL_S = 1200.0
COMPARISON_MODE = 'corrected_att0_symmetric20_timing_refine'

LEADER_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_anchor2_zpair_anchor11_xpair_symmetric20_60s_shared_noise0p08_param_errors.json'
LEADER_KF = RESULTS_DIR / 'KF36_ch3closedloop_anchor2_zpair_anchor11_xpair_symmetric20_60s_shared_noise0p08_param_errors.json'
UNIFIED16_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3corrected_anchor11_xpair_outerhold_unified16_75s_shared_noise0p08_param_errors.json'
UNIFIED16_KF = RESULTS_DIR / 'KF36_ch3corrected_anchor11_xpair_outerhold_unified16_75s_shared_noise0p08_param_errors.json'
FAITHFUL_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
FAITHFUL_KF = RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json'


@dataclass
class TimingVariant:
    name: str
    family: str
    hypothesis_id: str
    rationale: str
    method_tag: str
    total_time_s: float
    all_rows: list[dict[str, Any]]
    all_actions: list[dict[str, Any]]
    all_faces: list[dict[str, Any]]
    continuity_checks: list[dict[str, Any]]


def sanitize_tag(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    return text.strip('_')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def triplet_text(compact: dict[str, Any]) -> str:
    overall = compact['overall']
    return f"{overall['mean_pct_error']:.3f} / {overall['median_pct_error']:.3f} / {overall['max_pct_error']:.3f}"


def delta_vs_reference(ref_payload: dict[str, Any], cand_payload: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    ref_compact = compact_result(ref_payload)
    cand_compact = compact_result(cand_payload)
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        out[metric] = float(ref_compact['overall'][metric]) - float(cand_compact['overall'][metric])
    return out


def load_reference_payload(path: Path, noise_scale: float) -> dict[str, Any]:
    payload = _load_json(path)
    if not _noise_matches(payload, expected_noise_config(noise_scale)):
        raise ValueError(f'noise configuration mismatch: {path}')
    return payload


def load_reference_payloads(noise_scale: float) -> dict[str, Any]:
    return {
        'symmetric20_leader': {
            'label': 'symmetric20 first leader / anchor2_zpair_anchor11_xpair_symmetric20_60s',
            'markov42': load_reference_payload(LEADER_MARKOV, noise_scale),
            'kf36': load_reference_payload(LEADER_KF, noise_scale),
            'files': {'markov42': str(LEADER_MARKOV), 'kf36': str(LEADER_KF)},
        },
        'prior_unified16': {
            'label': 'prior unified16 / anchor11_xpair_outerhold_unified16_75s',
            'markov42': load_reference_payload(UNIFIED16_MARKOV, noise_scale),
            'kf36': load_reference_payload(UNIFIED16_KF, noise_scale),
            'files': {'markov42': str(UNIFIED16_MARKOV), 'kf36': str(UNIFIED16_KF)},
        },
        'faithful12': {
            'label': 'corrected faithful12',
            'markov42': load_reference_payload(FAITHFUL_MARKOV, noise_scale),
            'kf36': load_reference_payload(FAITHFUL_KF, noise_scale),
            'files': {'markov42': str(FAITHFUL_MARKOV), 'kf36': str(FAITHFUL_KF)},
        },
    }


def make_row(rot_s: float, pre_s: float, post_s: float) -> tuple[float, float, float, float]:
    total = float(rot_s + pre_s + post_s)
    return float(rot_s), float(pre_s), float(post_s), total


def apply_row_timing(base_rows: list[dict[str, Any]], timing_by_pos: dict[int, tuple[float, float, float, float]]) -> list[dict[str, Any]]:
    rows = copy.deepcopy(base_rows)
    for row in rows:
        pos_id = int(row['pos_id'])
        rot_s, pre_s, post_s, total = timing_by_pos[pos_id]
        row['rotation_time_s'] = rot_s
        row['pre_static_s'] = pre_s
        row['post_static_s'] = post_s
        row['node_total_s'] = total
    total_time = sum(float(r['node_total_s']) for r in rows)
    if abs(total_time - TARGET_TOTAL_S) > 1e-9:
        raise ValueError(f'total time mismatch: {total_time} vs {TARGET_TOTAL_S}')
    return rows


def uniform_map(total_s: float = 60.0) -> dict[int, tuple[float, float, float, float]]:
    rot = total_s * 0.1
    pre = total_s * 0.1
    post = total_s * 0.8
    return {i: make_row(rot, pre, post) for i in range(1, 21)}


def timing_map_back_loaded() -> dict[int, tuple[float, float, float, float]]:
    m = uniform_map()
    for i in [3, 4, 5, 6]:
        m[i] = make_row(5.6, 5.6, 44.8)
    for i in [16, 17, 18, 19]:
        m[i] = make_row(6.4, 6.4, 51.2)
    return m


def timing_map_back_loaded_strong() -> dict[int, tuple[float, float, float, float]]:
    m = uniform_map()
    for i in [3, 4, 5, 6]:
        m[i] = make_row(5.2, 5.2, 41.6)
    for i in [16, 17, 18, 19]:
        m[i] = make_row(6.8, 6.8, 54.4)
    return m


def timing_map_dual_motif_emphasis() -> dict[int, tuple[float, float, float, float]]:
    m = uniform_map()
    for i in [3, 4, 5, 6, 16, 17, 18, 19]:
        m[i] = make_row(6.4, 6.4, 51.2)
    for i in [7, 8, 9, 10, 11, 12, 13, 14]:
        m[i] = make_row(5.6, 5.6, 44.8)
    return m


def timing_map_late_half_emphasis() -> dict[int, tuple[float, float, float, float]]:
    m = uniform_map()
    for i in [1, 2, 3, 4, 5, 6]:
        m[i] = make_row(5.0, 5.0, 40.0)
    for i in [15, 16, 17, 18, 19, 20]:
        m[i] = make_row(7.0, 7.0, 56.0)
    return m


def build_variant(base_candidate, spec: dict[str, Any]) -> TimingVariant:
    rows = apply_row_timing(base_candidate.all_rows, spec['timing_map'])
    return TimingVariant(
        name=spec['name'],
        family=spec['family'],
        hypothesis_id=spec['hypothesis_id'],
        rationale=spec['rationale'],
        method_tag='ch3closedloop_' + sanitize_tag(spec['name']),
        total_time_s=sum(float(r['node_total_s']) for r in rows),
        all_rows=rows,
        all_actions=copy.deepcopy(base_candidate.all_actions),
        all_faces=copy.deepcopy(base_candidate.all_faces),
        continuity_checks=copy.deepcopy(base_candidate.continuity_checks),
    )


def candidate_output_path(candidate: TimingVariant, method_key: str, noise_scale: float) -> Path:
    prefix = 'M_markov_42state_gm1' if method_key == 'markov42_noisy' else 'KF36'
    return RESULTS_DIR / f'{prefix}_{candidate.method_tag}_shared_{make_suffix(noise_scale)}_param_errors.json'


def run_candidate_payload(mod, candidate: TimingVariant, method_key: str, noise_scale: float, force_rerun: bool = False):
    expected_cfg = expected_noise_config(noise_scale)
    out_path = candidate_output_path(candidate, method_key, noise_scale)
    if out_path.exists() and (not force_rerun):
        payload = _load_json(out_path)
        extra = payload.get('extra', {})
        if (
            _noise_matches(payload, expected_cfg)
            and extra.get('candidate_name') == candidate.name
            and abs(float(extra.get('time_total_s', -1.0)) - TARGET_TOTAL_S) < 1e-9
            and extra.get('comparison_mode') == COMPARISON_MODE
        ):
            return payload, 'reused_verified', out_path

    paras = rows_to_paras(mod, candidate.all_rows)
    dataset = build_dataset_with_path(mod, noise_scale, paras)
    params = _param_specs(mod)

    if method_key == 'markov42_noisy':
        clbt, _, _, _, _ = mod.run_calibration(
            dataset['imu_noisy'], dataset['pos0'], dataset['ts'], n_states=42,
            bi_g=dataset['bi_g'], tau_g=dataset['tau_g'], bi_a=dataset['bi_a'], tau_a=dataset['tau_a'],
            label=f'{candidate.name}_{method_key}_{make_suffix(noise_scale)}',
        )
    elif method_key == 'kf36_noisy':
        clbt, _, _, _, _ = mod.run_calibration(
            dataset['imu_noisy'], dataset['pos0'], dataset['ts'], n_states=36,
            label=f'{candidate.name}_{method_key}_{make_suffix(noise_scale)}',
        )
    else:
        raise KeyError(method_key)

    payload = compute_payload(
        mod,
        clbt,
        params,
        variant=f'{candidate.name}_{method_key}_{make_suffix(noise_scale)}',
        method_file='search_ch3_corrected_symmetric20_timing_refine.py',
        extra={
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'comparison_mode': COMPARISON_MODE,
            'candidate_name': candidate.name,
            'method_key': method_key,
            'att0_deg': ATT0_DEG,
            'time_total_s': candidate.total_time_s,
            'n_rows': len(candidate.all_rows),
            'family': candidate.family,
            'hypothesis_id': candidate.hypothesis_id,
            'rationale': candidate.rationale,
            'row_timing_table': [
                {
                    'pos_id': int(r['pos_id']),
                    'anchor_id': int(r['anchor_id']),
                    'segment_role': r['segment_role'],
                    'rotation_time_s': float(r['rotation_time_s']),
                    'pre_static_s': float(r['pre_static_s']),
                    'post_static_s': float(r['post_static_s']),
                    'node_total_s': float(r['node_total_s']),
                }
                for r in candidate.all_rows
            ],
        },
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', out_path


def render_report(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append('# Chapter-3 corrected symmetric20 timing refinement')
    lines.append('')
    lines.append('## 1. Relaunch goal')
    lines.append('')
    lines.append('- Stay inside the new 20-step symmetric family rather than falling back to older 16-step lines.')
    lines.append('- Keep the exact same legal action skeleton and shared `noise0p08` basis.')
    lines.append('- Only test timing redistribution hypotheses inside the symmetric20 family.')
    lines.append('')
    lines.append('## 2. Batch design')
    lines.append('')
    for h in summary['hypotheses_tested']:
        lines.append(f"- **{h['id']} · {h['family']}** — {h['summary']}")
        lines.append(f"  - tested: {', '.join(h['candidate_names'])}")
    lines.append('')
    lines.append('## 3. Markov42 results')
    lines.append('')
    lines.append('| rank | candidate | family | mean | median | max | Δmean vs sym20 leader | Δmedian | Δmax | Δmean vs unified16 | Δmax vs unified16 |')
    lines.append('|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
    for i, row in enumerate(summary['rows_sorted'], start=1):
        m = row['markov42']['overall']
        ds = row['delta_vs_sym20_leader_markov42']
        du = row['delta_vs_unified16_markov42']
        lines.append(
            f"| {i} | {row['candidate_name']} | {row['family']} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {ds['mean_pct_error']:+.3f} | {ds['median_pct_error']:+.3f} | {ds['max_pct_error']:+.3f} | {du['mean_pct_error']:+.3f} | {du['max_pct_error']:+.3f} |"
        )
    lines.append('')
    lines.append('## 4. KF36 recheck')
    lines.append('')
    if summary['kf36_rechecked_candidates']:
        for row in summary['rows_sorted']:
            if row.get('kf36') is not None:
                lines.append(f"- `{row['candidate_name']}` → Markov42 **{triplet_text(row['markov42'])}**, KF36 **{triplet_text(row['kf36'])}**")
    else:
        lines.append(f"- No KF36 reruns triggered. Gate reason: {summary['kf36_gate_reason']}")
    lines.append('')
    lines.append('## 5. Decision')
    lines.append('')
    best = summary['best_candidate']
    lines.append(f"- Batch best: `{best['candidate_name']}` = **{triplet_text(best['markov42'])}**")
    if best.get('kf36') is not None:
        lines.append(f"- KF36 for batch best: **{triplet_text(best['kf36'])}**")
    lines.append(f"- Did any timing variant beat the existing symmetric20 leader on Markov42 mean+max? **{summary['bottom_line']['new_leader_found']}**")
    lines.append(f"- Bottom line: {summary['bottom_line']['statement']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def score_key(row: dict[str, Any]) -> tuple[float, float, float]:
    o = row['markov42']['overall']
    return (float(o['mean_pct_error']), float(o['max_pct_error']), float(o['median_pct_error']))


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module('search_ch3_corrected_symmetric20_timing_refine_src', str(SOURCE_FILE))
    base_candidate = build_symmetric20_candidate(mod)
    refs_raw = load_reference_payloads(args.noise_scale)

    specs = [
        {
            'name': 'anchor2_zpair_anchor11_xpair_symmetric20_backloaded64_56',
            'family': 'late_motif_time_redistribution',
            'hypothesis_id': 'H1',
            'rationale': 'Keep the exact 20-step skeleton, but shift 4 s from each front motif row to the late anchor11 motif row counterpart: test whether the family wants a mild late-back emphasis.',
            'timing_map': timing_map_back_loaded(),
        },
        {
            'name': 'anchor2_zpair_anchor11_xpair_symmetric20_backloaded68_52',
            'family': 'late_motif_time_redistribution',
            'hypothesis_id': 'H1',
            'rationale': 'Same as H1 but stronger: a harder transfer from front motif dwell into the late anchor11 motif.',
            'timing_map': timing_map_back_loaded_strong(),
        },
        {
            'name': 'anchor2_zpair_anchor11_xpair_symmetric20_dualmotif64_mid56',
            'family': 'motif_vs_backbone_time_redistribution',
            'hypothesis_id': 'H2',
            'rationale': 'Emphasize both inserted motifs equally and pay for it by shrinking the middle faithful backbone rows 7-14, testing whether the benefit is carried by the two added motifs rather than the middle scaffold.',
            'timing_map': timing_map_dual_motif_emphasis(),
        },
        {
            'name': 'anchor2_zpair_anchor11_xpair_symmetric20_latehalf70_early50',
            'family': 'global_early_late_redistribution',
            'hypothesis_id': 'H3',
            'rationale': 'Give the entire late half more dwell and tax the early half equally, testing whether the symmetric20 family is really succeeding because of stronger late-half observability rather than strict front/back symmetry itself.',
            'timing_map': timing_map_late_half_emphasis(),
        },
    ]

    variants = [build_variant(base_candidate, s) for s in specs]

    rows = []
    for variant in variants:
        markov_payload, markov_mode, markov_path = run_candidate_payload(mod, variant, 'markov42_noisy', args.noise_scale, args.force_rerun)
        row = {
            'candidate_name': variant.name,
            'family': variant.family,
            'hypothesis_id': variant.hypothesis_id,
            'rationale': variant.rationale,
            'total_time_s': variant.total_time_s,
            'markov42': compact_result(markov_payload),
            'markov42_file': str(markov_path),
            'markov42_mode': markov_mode,
            'delta_vs_sym20_leader_markov42': delta_vs_reference(refs_raw['symmetric20_leader']['markov42'], markov_payload),
            'delta_vs_unified16_markov42': delta_vs_reference(refs_raw['prior_unified16']['markov42'], markov_payload),
            'delta_vs_faithful12_markov42': delta_vs_reference(refs_raw['faithful12']['markov42'], markov_payload),
            'timing_table': [
                {
                    'pos_id': int(r['pos_id']),
                    'anchor_id': int(r['anchor_id']),
                    'segment_role': r['segment_role'],
                    'label': r['label'],
                    'motor_action': render_action(a),
                    'rotation_time_s': float(r['rotation_time_s']),
                    'pre_static_s': float(r['pre_static_s']),
                    'post_static_s': float(r['post_static_s']),
                    'node_total_s': float(r['node_total_s']),
                    'face_after': f['face_name'],
                }
                for r, a, f in zip(variant.all_rows, variant.all_actions, variant.all_faces)
            ],
        }
        rows.append(row)

    rows_sorted = sorted(rows, key=score_key)

    kf36_rechecked_candidates: list[str] = []
    kf36_gate_reason = 'No candidate stayed within a plausible improvement envelope.'
    for row in rows_sorted[:2]:
        ds = row['delta_vs_sym20_leader_markov42']
        if ds['mean_pct_error'] > -0.03 and ds['max_pct_error'] > -0.20:
            variant = next(v for v in variants if v.name == row['candidate_name'])
            kf_payload, kf_mode, kf_path = run_candidate_payload(mod, variant, 'kf36_noisy', args.noise_scale, args.force_rerun)
            row['kf36'] = compact_result(kf_payload)
            row['kf36_file'] = str(kf_path)
            row['kf36_mode'] = kf_mode
            row['delta_vs_sym20_leader_kf36'] = delta_vs_reference(refs_raw['symmetric20_leader']['kf36'], kf_payload)
            row['delta_vs_unified16_kf36'] = delta_vs_reference(refs_raw['prior_unified16']['kf36'], kf_payload)
            kf36_rechecked_candidates.append(row['candidate_name'])
    if kf36_rechecked_candidates:
        kf36_gate_reason = 'Rechecked the top-2 Markov42 variants that remained close enough to the sym20 leader to be decision-relevant.'

    best = rows_sorted[0]
    new_leader_found = best['delta_vs_sym20_leader_markov42']['mean_pct_error'] > 0 and best['delta_vs_sym20_leader_markov42']['max_pct_error'] > 0

    hypotheses_tested = []
    seen = {}
    for s in specs:
        key = (s['hypothesis_id'], s['family'])
        if key not in seen:
            seen[key] = {'id': s['hypothesis_id'], 'family': s['family'], 'summary': s['rationale'], 'candidate_names': []}
            hypotheses_tested.append(seen[key])
        seen[key]['candidate_names'].append(s['name'])

    statement = (
        f"No timing-only refinement beat the existing symmetric20 leader; the best retry was `{best['candidate_name']}` at {triplet_text(best['markov42'])}, "
        f"which remained behind the current leader by Δmean {best['delta_vs_sym20_leader_markov42']['mean_pct_error']:+.3f} and Δmax {best['delta_vs_sym20_leader_markov42']['max_pct_error']:+.3f}."
    )
    if new_leader_found:
        statement = (
            f"A new symmetric20 timing leader emerged: `{best['candidate_name']}` = {triplet_text(best['markov42'])}, "
            f"improving on the previous symmetric20 leader by Δmean {best['delta_vs_sym20_leader_markov42']['mean_pct_error']:+.3f} and Δmax {best['delta_vs_sym20_leader_markov42']['max_pct_error']:+.3f}."
        )

    summary = {
        'task': 'chapter-3 corrected symmetric20 timing refinement',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'corrected_att0_deg': ATT0_DEG,
        'references': {
            key: {
                'label': val['label'],
                'markov42': compact_result(val['markov42']),
                'kf36': compact_result(val['kf36']),
                'files': val['files'],
                'markov42_triplet': triplet_text(compact_result(val['markov42'])),
                'kf36_triplet': triplet_text(compact_result(val['kf36'])),
            }
            for key, val in refs_raw.items()
        },
        'hypotheses_tested': hypotheses_tested,
        'rows_sorted': rows_sorted,
        'best_candidate': best,
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
        'kf36_gate_reason': kf36_gate_reason,
        'bottom_line': {
            'new_leader_found': new_leader_found,
            'statement': statement,
        },
    }

    summary_path = RESULTS_DIR / f'ch3_corrected_symmetric20_timing_refine_{args.report_date}.json'
    report_path = REPORTS_DIR / f'psins_ch3_corrected_symmetric20_timing_refine_{args.report_date}.md'
    summary['files'] = {'summary_json': str(summary_path), 'report_md': str(report_path)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_path.write_text(render_report(summary), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps(summary['files'], ensure_ascii=False), flush=True)
    print('BEST_MARKOV42', best['candidate_name'], triplet_text(best['markov42']), flush=True)
    print('NEW_LEADER_FOUND', str(new_leader_found).lower(), flush=True)
    print('BOTTOM_LINE', statement, flush=True)


if __name__ == '__main__':
    main()
