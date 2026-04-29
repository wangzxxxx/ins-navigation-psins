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
from search_ch3_corrected_inbasin_ridge_resume import ATT0_DEG, compact_metrics, delta_vs_reference, overall_triplet
from search_ch3_entry_conditioned_relay_family import (
    NOISE_SCALE,
    closed_pair,
    l8_xpair,
    l9_ypair_neg,
    merge_insertions,
    xpair_outerhold,
    zquad,
)

REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
REPORT_PATH = REPORTS_DIR / f'psins_ch3_corrected_sparse_ablation_{REPORT_DATE}.md'
SUMMARY_PATH = RESULTS_DIR / f'ch3_corrected_sparse_ablation_{REPORT_DATE}.json'
EPS = 1e-9

REFERENCE_FILES = {
    'full_best': {
        'label': 'current full corrected best / relay_l11back0p5_l12y0p125_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l11back0p5_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_relay_l11back0p5_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
    },
    'q4_relaxed_reference': {
        'label': 'q4 relaxed control / relay_r3_l9y0p8125_l12y0p125_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_r3_l9y0p8125_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_relay_r3_l9y0p8125_l12y0p125_on_entry_shared_noise0p08_param_errors.json',
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

MOTIF_META = {
    'anchor8_x_bookend': {
        'label': 'anchor8 x bookend',
        'description': 'Entry outer-axis x closed pair at anchor8.',
        'turns': 2,
    },
    'anchor9_y_pair': {
        'label': 'anchor9 negative y pair',
        'description': 'Inner-axis negative y closed pair at anchor9.',
        'turns': 2,
    },
    'anchor10_y_pair': {
        'label': 'anchor10 y pair',
        'description': 'Anchor10 inner-axis y closed pair.',
        'turns': 2,
    },
    'anchor10_z_pair': {
        'label': 'anchor10 z pair',
        'description': 'Anchor10 outer-axis z closed pair.',
        'turns': 2,
    },
    'anchor11_xpair_outerhold': {
        'label': 'anchor11 xpair outerhold',
        'description': 'Four-step anchor11 y-open / x sweep / x return / y-close motif.',
        'turns': 4,
    },
    'anchor11_q4_backdwell': {
        'label': 'anchor11 q4 back-dwell reduction',
        'description': 'Retiming of anchor11 zquad q4 back dwell from 2.0 s down to 0.5 s.',
        'turns': 0,
    },
    'anchor12_terminal_y': {
        'label': 'anchor12 tiny terminal y closure',
        'description': 'Final terminal inner-axis y micro-closure at anchor12.',
        'turns': 2,
    },
}

PRIMARY_PLAN = [
    {
        'name': 'ablate_anchor8_x_bookend',
        'family': 'single_motif_ablation',
        'hypothesis_id': 'A1',
        'motifs_removed': ['anchor8_x_bookend'],
        'summary': 'Remove only the anchor8 x bookend while keeping the later corrected core intact.',
    },
    {
        'name': 'ablate_anchor9_y_pair',
        'family': 'single_motif_ablation',
        'hypothesis_id': 'A2',
        'motifs_removed': ['anchor9_y_pair'],
        'summary': 'Remove only the anchor9 negative y pair.',
    },
    {
        'name': 'ablate_anchor10_y_pair',
        'family': 'single_motif_ablation',
        'hypothesis_id': 'A3',
        'motifs_removed': ['anchor10_y_pair'],
        'summary': 'Keep anchor10 z pair but remove anchor10 y pair only.',
    },
    {
        'name': 'ablate_anchor10_z_pair',
        'family': 'single_motif_ablation',
        'hypothesis_id': 'A4',
        'motifs_removed': ['anchor10_z_pair'],
        'summary': 'Keep anchor10 y pair but remove anchor10 z pair only.',
    },
    {
        'name': 'ablate_anchor11_xpair_outerhold',
        'family': 'single_motif_ablation',
        'hypothesis_id': 'A5',
        'motifs_removed': ['anchor11_xpair_outerhold'],
        'summary': 'Remove only the anchor11 xpair outerhold and keep the zquad + q4 back0.5 intact.',
    },
    {
        'name': 'relax_anchor11_q4_backdwell',
        'family': 'timing_control_ablation',
        'hypothesis_id': 'A6',
        'motifs_removed': ['anchor11_q4_backdwell'],
        'summary': 'Relax the anchor11 q4 back-dwell reduction from 0.5 s back to the old 2.0 s control.',
        'reuse_reference_key': 'q4_relaxed_reference',
    },
    {
        'name': 'ablate_anchor12_terminal_y',
        'family': 'single_motif_ablation',
        'hypothesis_id': 'A7',
        'motifs_removed': ['anchor12_terminal_y'],
        'summary': 'Remove only the tiny terminal y closure at anchor12.',
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



def attach_att0(path: Path, payload: dict[str, Any], candidate_name: str, method_key: str, family: str, hypothesis_id: str) -> dict[str, Any]:
    extra = payload.setdefault('extra', {})
    extra['att0_deg'] = ATT0_DEG
    extra['comparison_mode'] = 'corrected_sparse_ablation'
    extra['candidate_registry_key'] = candidate_name
    extra['method_key'] = method_key
    extra['family'] = family
    extra['hypothesis_id'] = hypothesis_id
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload



def y_pair(dwell_s: float, label: str) -> list[StepSpec]:
    return [
        StepSpec(kind='inner', angle_deg=-90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_out', label=f'{label}_out'),
        StepSpec(kind='inner', angle_deg=+90, rotation_time_s=5.0, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_return', label=f'{label}_return'),
    ]



def q4_label(back_dwell_s: float) -> str:
    return str(back_dwell_s).replace('-', 'neg').replace('.', 'p')



def build_insertions(
    *,
    keep_anchor8: bool = True,
    keep_anchor9: bool = True,
    keep_anchor10_z: bool = True,
    keep_anchor10_y: bool = True,
    keep_anchor11_x: bool = True,
    q4_back_dwell_s: float = 0.5,
    keep_anchor12: bool = True,
) -> dict[int, list[StepSpec]]:
    pieces: list[dict[int, list[StepSpec]]] = []
    if keep_anchor8:
        pieces.append(l8_xpair(1.0, 'l8_x1'))
    if keep_anchor9:
        pieces.append(l9_ypair_neg(0.8125, 'l9_ypair_neg0p8125'))

    anchor10_steps: list[StepSpec] = []
    if keep_anchor10_z:
        anchor10_steps.extend(closed_pair('outer', -90, 5.0, 'l10_zpair_neg5'))
    if keep_anchor10_y:
        anchor10_steps.extend(closed_pair('inner', -90, 1.0, 'l10_ypair_neg1'))
    if anchor10_steps:
        pieces.append({10: anchor10_steps})

    anchor11_steps: list[StepSpec] = []
    if keep_anchor11_x:
        anchor11_steps.extend(xpair_outerhold(10.0, 'l11_xpair_outerhold'))
    anchor11_steps.extend(zquad(10.0, 0.0, float(q4_back_dwell_s), f'l11_zquad_y10x0back{q4_label(q4_back_dwell_s)}'))
    pieces.append({11: anchor11_steps})

    if keep_anchor12:
        pieces.append({12: y_pair(0.125, 'l12_yneg0p125')})

    return merge_insertions(*pieces)



def primary_spec(plan: dict[str, Any]) -> dict[str, Any] | None:
    removed = set(plan['motifs_removed'])
    if plan.get('reuse_reference_key') is not None:
        return None
    return {
        'name': plan['name'],
        'family': plan['family'],
        'hypothesis_id': plan['hypothesis_id'],
        'rationale': plan['summary'],
        'insertions': build_insertions(
            keep_anchor8='anchor8_x_bookend' not in removed,
            keep_anchor9='anchor9_y_pair' not in removed,
            keep_anchor10_z='anchor10_z_pair' not in removed,
            keep_anchor10_y='anchor10_y_pair' not in removed,
            keep_anchor11_x='anchor11_xpair_outerhold' not in removed,
            q4_back_dwell_s=2.0 if 'anchor11_q4_backdwell' in removed else 0.5,
            keep_anchor12='anchor12_terminal_y' not in removed,
        ),
    }



def combo_spec(combo_index: int, motif_ids: list[str]) -> dict[str, Any]:
    removed = set(motif_ids)
    motif_suffix = '_'.join(m.replace('anchor', 'a').replace('_bookend', '').replace('_pair', '').replace('_terminal_y', 'a12y').replace('_outerhold', '').replace('_backdwell', 'q4') for m in motif_ids)
    return {
        'name': f'combo_sparse_{combo_index}_{motif_suffix}',
        'family': 'compacted_recombination',
        'hypothesis_id': f'C{combo_index}',
        'motifs_removed': list(motif_ids),
        'summary': 'Compacted recombination after primary ablation screen.',
        'rationale': 'Drop the best damage-per-turn removable motifs together and test whether the corrected leader can be compressed further without reopening any new family.',
        'insertions': build_insertions(
            keep_anchor8='anchor8_x_bookend' not in removed,
            keep_anchor9='anchor9_y_pair' not in removed,
            keep_anchor10_z='anchor10_z_pair' not in removed,
            keep_anchor10_y='anchor10_y_pair' not in removed,
            keep_anchor11_x='anchor11_xpair_outerhold' not in removed,
            q4_back_dwell_s=0.5,
            keep_anchor12='anchor12_terminal_y' not in removed,
        ),
    }



def extra_turn_count(candidate) -> int:
    return sum(1 for row in candidate.all_rows if row['segment_role'] != 'anchor')



def build_timing_table(candidate) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row, action in zip(candidate.all_rows, candidate.all_actions):
        rows.append({
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
    return rows



def render_timing_table_md(table: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    lines.append('| pos | anchor | role | label | motor action | axis | rot_s | pre_s | post_s | total_s | face_after | beta_after |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|')
    for item in table:
        axis = '[' + ','.join(str(v) for v in item['effective_body_axis']) + ']'
        lines.append(
            f"| {item['pos_id']} | {item['anchor_id']} | {item['segment_role']} | {item['label']} | {item['motor_action']} | {axis} | {item['rotation_time_s']:.3f} | {item['pre_static_s']:.3f} | {item['post_static_s']:.3f} | {item['node_total_s']:.3f} | {item['face_after']} | {item['inner_beta_after_deg']} |"
        )
    return lines



def loss_triplet(delta_vs_leader: dict[str, float]) -> dict[str, float]:
    return {metric: max(0.0, -float(delta_vs_leader[metric])) for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']}



def loss_sum(delta_vs_leader: dict[str, float]) -> float:
    losses = loss_triplet(delta_vs_leader)
    return losses['mean_pct_error'] + losses['median_pct_error'] + losses['max_pct_error']



def classify_primary(losses: dict[str, float], motif_id: str) -> str:
    if motif_id == 'anchor11_q4_backdwell':
        if losses['mean_pct_error'] <= 0.010 and losses['max_pct_error'] <= 0.150:
            return 'important timing lever'
        return 'critical timing lever'
    if losses['mean_pct_error'] <= 0.010 and losses['max_pct_error'] <= 0.150 and losses['median_pct_error'] <= 0.040:
        return 'removable candidate'
    if losses['mean_pct_error'] <= 0.030 and losses['max_pct_error'] <= 0.400:
        return 'supportive'
    return 'core'



def gain_retention(leader: dict[str, Any], faithful: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for metric in ['mean_pct_error', 'max_pct_error']:
        leader_gain = float(faithful['overall'][metric]) - float(leader['overall'][metric])
        cand_gain = float(faithful['overall'][metric]) - float(candidate['overall'][metric])
        out[metric] = cand_gain / leader_gain if abs(leader_gain) > 1e-12 else None
    return out



def candidate_row_from_payload(
    *,
    candidate_name: str,
    family: str,
    hypothesis_id: str,
    rationale: str,
    motifs_removed: list[str],
    candidate_obj,
    markov_payload: dict[str, Any],
    markov_path: Path,
    markov_mode: str,
    references: dict[str, Any],
    leader_extra_turns: int,
    leader_total_time_s: float,
) -> dict[str, Any]:
    row = {
        'candidate_name': candidate_name,
        'family': family,
        'hypothesis_id': hypothesis_id,
        'rationale': rationale,
        'motifs_removed': motifs_removed,
        'motif_labels_removed': [MOTIF_META[m]['label'] for m in motifs_removed],
        'result_files': {'markov42': str(markov_path)},
        'result_modes': {'markov42': markov_mode},
        'markov42': compact_metrics(markov_payload),
    }
    row['total_time_s'] = float(candidate_obj.total_time_s)
    row['extra_turns'] = extra_turn_count(candidate_obj)
    row['turns_removed'] = leader_extra_turns - row['extra_turns']
    row['time_removed_s'] = leader_total_time_s - row['total_time_s']
    row['delta_vs_leader_markov42'] = delta_vs_reference(references['full_best']['markov42'], row['markov42'])
    row['delta_vs_faithful12_markov42'] = delta_vs_reference(references['faithful12']['markov42'], row['markov42'])
    row['delta_vs_default18_markov42'] = delta_vs_reference(references['default18']['markov42'], row['markov42'])
    row['gain_retention_vs_faithful12'] = gain_retention(references['full_best']['markov42'], references['faithful12']['markov42'], row['markov42'])
    row['losses_vs_leader'] = loss_triplet(row['delta_vs_leader_markov42'])
    row['loss_sum_vs_leader'] = loss_sum(row['delta_vs_leader_markov42'])
    return row



def run_spec(mod, faithful_rows, faithful_actions, spec: dict[str, Any], references: dict[str, Any], leader_extra_turns: int, leader_total_time_s: float, force_rerun: bool):
    candidate = build_closedloop_candidate(mod, spec, faithful_rows, faithful_actions)
    payload, mode, path = run_candidate_payload(mod, candidate, 'markov42_noisy', NOISE_SCALE, force_rerun=force_rerun)
    payload = attach_att0(path, payload, spec['name'], 'markov42_noisy', spec['family'], spec['hypothesis_id'])
    row = candidate_row_from_payload(
        candidate_name=spec['name'],
        family=spec['family'],
        hypothesis_id=spec['hypothesis_id'],
        rationale=spec['rationale'],
        motifs_removed=spec['motifs_removed'],
        candidate_obj=candidate,
        markov_payload=payload,
        markov_path=path,
        markov_mode=mode,
        references=references,
        leader_extra_turns=leader_extra_turns,
        leader_total_time_s=leader_total_time_s,
    )
    return candidate, row



def render_report(summary: dict[str, Any]) -> str:
    refs = summary['references']
    best_sparse = summary['recommended_sparse_candidate']
    ranking = summary['motif_importance_ranking']
    lines: list[str] = []
    lines.append('# Chapter-3 corrected-basis sparse ablation / pruning batch')
    lines.append('')
    lines.append('## 1. Objective and fixed context')
    lines.append('')
    lines.append('- Current full corrected best held fixed as the pruning parent: `relay_l11back0p5_l12y0p125_on_entry` = **1.057 / 0.588 / 4.560** (Markov42), **1.056 / 0.588 / 4.558** (KF36).')
    lines.append('- Hard constraints remained fixed: real dual-axis legality only, exact continuity-safe reconnection, faithful original 12-position backbone, total time inside 20–30 min, theory-guided only, `att0=(0,0,0)` exactly.')
    lines.append(f"- Full parent path uses **{summary['full_best_structure']['extra_turns']} extra turns** over faithful12 and totals **{summary['full_best_structure']['total_time_s']:.3f} s**.")
    lines.append('- This batch explicitly switched from expansion to **ablation / sparsification**: remove one motif at a time, identify what is essential, and then test one or two compacted recombinations only if the single-motif screen says they are plausible.')
    lines.append('')
    lines.append('## 2. Full-parent motif decomposition')
    lines.append('')
    lines.append('| motif | label | extra turns in parent | role |')
    lines.append('|---|---|---:|---|')
    for motif_id in summary['full_best_structure']['motif_order']:
        meta = MOTIF_META[motif_id]
        lines.append(f"| `{motif_id}` | {meta['label']} | {meta['turns']} | {meta['description']} |")
    lines.append('')
    lines.append('## 3. Primary leave-one-motif-out ablations (Markov42)')
    lines.append('')
    lines.append('| rank | candidate | removed motif | turns removed | mean | median | max | loss vs parent (mean/median/max) | gain retention vs faithful12 (mean/max) | verdict |')
    lines.append('|---:|---|---|---:|---:|---:|---:|---|---|---|')
    for idx, row in enumerate(summary['primary_rows_sorted'], start=1):
        losses = row['losses_vs_leader']
        retention = row['gain_retention_vs_faithful12']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {', '.join(row['motif_labels_removed'])} | {row['turns_removed']} | {row['markov42']['overall']['mean_pct_error']:.3f} | {row['markov42']['overall']['median_pct_error']:.3f} | {row['markov42']['overall']['max_pct_error']:.3f} | {losses['mean_pct_error']:.3f} / {losses['median_pct_error']:.3f} / {losses['max_pct_error']:.3f} | {retention['mean_pct_error']*100:.1f}% / {retention['max_pct_error']*100:.1f}% | {row['classification']} |"
        )
    lines.append('')
    lines.append('## 4. Explicit motif-importance ranking')
    lines.append('')
    lines.append('| importance rank | motif | damage score | loss if removed (mean/median/max) | turns saved | interpretation |')
    lines.append('|---:|---|---:|---|---:|---|')
    for idx, item in enumerate(ranking, start=1):
        losses = item['losses_vs_leader']
        lines.append(
            f"| {idx} | {item['motif_label']} | {item['damage_score']:.3f} | {losses['mean_pct_error']:.3f} / {losses['median_pct_error']:.3f} / {losses['max_pct_error']:.3f} | {item['turns_removed']} | {item['classification']} |"
        )
    lines.append('')
    lines.append('## 5. Optional compacted sparse recombinations')
    lines.append('')
    if summary['combo_rows_sorted']:
        lines.append('| rank | candidate | removed motifs | turns removed | mean | median | max | loss vs parent (mean/max) | note |')
        lines.append('|---:|---|---|---:|---:|---:|---:|---|---|')
        for idx, row in enumerate(summary['combo_rows_sorted'], start=1):
            losses = row['losses_vs_leader']
            lines.append(
                f"| {idx} | {row['candidate_name']} | {', '.join(row['motif_labels_removed'])} | {row['turns_removed']} | {row['markov42']['overall']['mean_pct_error']:.3f} | {row['markov42']['overall']['median_pct_error']:.3f} | {row['markov42']['overall']['max_pct_error']:.3f} | {losses['mean_pct_error']:.3f} / {losses['max_pct_error']:.3f} | {row['rationale']} |"
            )
    else:
        lines.append(f"- No combo reruns were launched. Gate reason: {summary['combo_gate_reason']}")
    lines.append('')
    lines.append('## 6. Recommended sparse candidate')
    lines.append('')
    lines.append(f"- **Recommended sparse path:** `{best_sparse['candidate_name']}`")
    lines.append(f"- Removed motifs: **{', '.join(best_sparse['motif_labels_removed'])}**")
    lines.append(f"- Turn reduction vs full parent: **{best_sparse['turns_removed']} fewer extra turns** (retain {best_sparse['extra_turns']} extra turns total)")
    lines.append(f"- Markov42: **{overall_triplet(best_sparse['markov42'])}**")
    if best_sparse.get('kf36') is not None:
        lines.append(f"- KF36: **{overall_triplet(best_sparse['kf36'])}**")
    else:
        lines.append(f"- KF36: **not rerun** ({summary['kf36_gate_reason']})")
    lines.append(
        f"- Loss vs full parent: Δmean **-{best_sparse['losses_vs_leader']['mean_pct_error']:.6f}**, Δmedian **-{best_sparse['losses_vs_leader']['median_pct_error']:.6f}**, Δmax **-{best_sparse['losses_vs_leader']['max_pct_error']:.6f}**"
    )
    lines.append(
        f"- Gain retained vs faithful12: mean **{best_sparse['gain_retention_vs_faithful12']['mean_pct_error']*100:.1f}%**, max **{best_sparse['gain_retention_vs_faithful12']['max_pct_error']*100:.1f}%** of the full parent improvement"
    )
    lines.append(f"- Selection rule: **{summary['selection_rule']}**")
    lines.append('')
    lines.append('## 7. Required comparison set')
    lines.append('')
    lines.append('| path | extra turns | Markov42 | KF36 | Δmean vs sparse rec | Δmedian vs sparse rec | Δmax vs sparse rec | note |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---|')
    for row in summary['required_comparison_rows']:
        d = row['delta_vs_recommended_sparse_markov42']
        lines.append(
            f"| {row['label']} | {row['extra_turns']} | {row['markov42_triplet']} | {row['kf36_triplet']} | {d['mean_pct_error']:+.3f} | {d['median_pct_error']:+.3f} | {d['max_pct_error']:+.3f} | {row['note']} |"
        )
    lines.append('')
    lines.append('## 8. Exact legal motor / timing table for the recommended sparse candidate')
    lines.append('')
    lines.extend(render_timing_table_md(summary['recommended_sparse_timing_table']))
    lines.append('')
    lines.append('## 9. Bottom line')
    lines.append('')
    lines.append(f"- Core/essential motifs from this pruning pass: **{', '.join(summary['bottom_line']['core_motifs'])}**")
    lines.append(f"- Removable / low-value motifs from this pruning pass: **{', '.join(summary['bottom_line']['removable_motifs'])}**")
    lines.append(f"- Best sparse conclusion: **{summary['bottom_line']['statement']}**")
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

    full_best_spec = {
        'name': 'relay_l11back0p5_l12y0p125_on_entry_local_builder',
        'family': 'reference_rebuild',
        'hypothesis_id': 'ref',
        'rationale': 'Current full corrected best rebuilt locally for timing/turn accounting.',
        'insertions': build_insertions(),
    }
    full_best_candidate = build_closedloop_candidate(mod, full_best_spec, faithful.rows, faithful.action_sequence)
    leader_extra_turns = extra_turn_count(full_best_candidate)
    leader_total_time_s = float(full_best_candidate.total_time_s)

    candidates_by_name: dict[str, Any] = {'full_best_reference': full_best_candidate}
    all_rows: list[dict[str, Any]] = []
    primary_rows: list[dict[str, Any]] = []

    for plan in PRIMARY_PLAN:
        spec = primary_spec(plan)
        if spec is None:
            ref = references[plan['reuse_reference_key']]
            control_candidate = build_closedloop_candidate(
                mod,
                {
                    'name': plan['name'],
                    'family': plan['family'],
                    'hypothesis_id': plan['hypothesis_id'],
                    'rationale': plan['summary'],
                    'insertions': build_insertions(q4_back_dwell_s=2.0),
                },
                faithful.rows,
                faithful.action_sequence,
            )
            candidates_by_name[plan['name']] = control_candidate
            row = {
                'candidate_name': plan['name'],
                'family': plan['family'],
                'hypothesis_id': plan['hypothesis_id'],
                'rationale': plan['summary'],
                'motifs_removed': plan['motifs_removed'],
                'motif_labels_removed': [MOTIF_META[m]['label'] for m in plan['motifs_removed']],
                'result_files': dict(ref['files']),
                'result_modes': {'markov42': 'reused_reference', 'kf36': 'reused_reference'},
                'markov42': ref['markov42'],
                'kf36': ref['kf36'],
                'total_time_s': float(control_candidate.total_time_s),
                'extra_turns': extra_turn_count(control_candidate),
            }
            row['turns_removed'] = leader_extra_turns - row['extra_turns']
            row['time_removed_s'] = leader_total_time_s - row['total_time_s']
            row['delta_vs_leader_markov42'] = delta_vs_reference(references['full_best']['markov42'], row['markov42'])
            row['delta_vs_leader_kf36'] = delta_vs_reference(references['full_best']['kf36'], row['kf36'])
            row['delta_vs_faithful12_markov42'] = delta_vs_reference(references['faithful12']['markov42'], row['markov42'])
            row['delta_vs_default18_markov42'] = delta_vs_reference(references['default18']['markov42'], row['markov42'])
            row['gain_retention_vs_faithful12'] = gain_retention(references['full_best']['markov42'], references['faithful12']['markov42'], row['markov42'])
            row['losses_vs_leader'] = loss_triplet(row['delta_vs_leader_markov42'])
            row['loss_sum_vs_leader'] = loss_sum(row['delta_vs_leader_markov42'])
        else:
            spec['motifs_removed'] = plan['motifs_removed']
            candidate, row = run_spec(
                mod,
                faithful.rows,
                faithful.action_sequence,
                spec,
                references,
                leader_extra_turns,
                leader_total_time_s,
                args.force_rerun,
            )
            candidates_by_name[spec['name']] = candidate
        motif_id = row['motifs_removed'][0]
        row['motif_id'] = motif_id
        row['classification'] = classify_primary(row['losses_vs_leader'], motif_id)
        row['damage_score'] = row['loss_sum_vs_leader']
        primary_rows.append(row)
        all_rows.append(row)

    primary_rows_sorted = sorted(primary_rows, key=lambda r: (r['loss_sum_vs_leader'], r['turns_removed'] == 0, -r['turns_removed']))

    ranking = []
    for row in sorted(primary_rows, key=lambda r: (-r['damage_score'], -r['losses_vs_leader']['max_pct_error'], -r['losses_vs_leader']['mean_pct_error'])):
        ranking.append({
            'motif_id': row['motif_id'],
            'motif_label': MOTIF_META[row['motif_id']]['label'],
            'candidate_name': row['candidate_name'],
            'damage_score': row['damage_score'],
            'losses_vs_leader': row['losses_vs_leader'],
            'turns_removed': row['turns_removed'],
            'classification': row['classification'],
            'markov42': row['markov42'],
        })

    removable_candidates = [
        row for row in primary_rows
        if row['motif_id'] != 'anchor11_q4_backdwell' and row['classification'] == 'removable candidate'
    ]
    removable_candidates = sorted(
        removable_candidates,
        key=lambda r: (r['damage_score'] / max(r['turns_removed'], 1), r['damage_score'], -r['turns_removed'])
    )

    combo_specs = []
    if len(removable_candidates) >= 2:
        combo_specs.append(combo_spec(1, [removable_candidates[0]['motif_id'], removable_candidates[1]['motif_id']]))
    if len(removable_candidates) >= 3:
        combo_specs.append(combo_spec(2, [x['motif_id'] for x in removable_candidates[:3]]))

    combo_rows: list[dict[str, Any]] = []
    combo_gate_reason = 'No two primary motif ablations qualified as low-damage removable candidates, so no compacted recombination was justified.'
    if combo_specs:
        combo_gate_reason = 'Compacted recombinations were launched from the best low-damage primary removables ranked by damage-per-turn.'
        for spec in combo_specs:
            candidate, row = run_spec(
                mod,
                faithful.rows,
                faithful.action_sequence,
                spec,
                references,
                leader_extra_turns,
                leader_total_time_s,
                args.force_rerun,
            )
            candidates_by_name[spec['name']] = candidate
            combo_rows.append(row)
            all_rows.append(row)

    combo_rows_sorted = sorted(combo_rows, key=lambda r: (r['loss_sum_vs_leader'], -r['turns_removed']))

    sparse_pool = [row for row in all_rows if row['turns_removed'] > 0]
    near_competitive_pool = [
        row for row in sparse_pool
        if row['losses_vs_leader']['mean_pct_error'] <= 0.015 + EPS and row['losses_vs_leader']['max_pct_error'] <= 0.250 + EPS
    ]
    if near_competitive_pool:
        recommended_sparse = max(
            near_competitive_pool,
            key=lambda r: (r['turns_removed'], -r['losses_vs_leader']['mean_pct_error'], -r['losses_vs_leader']['max_pct_error'], -r['losses_vs_leader']['median_pct_error'])
        )
        selection_rule = 'Choose the maximum-turn-reduction candidate inside the near-competitive envelope (mean loss ≤ 0.015, max loss ≤ 0.250), then break ties by smaller mean/max loss.'
    else:
        recommended_sparse = min(
            sparse_pool,
            key=lambda r: (r['loss_sum_vs_leader'] / max(r['turns_removed'], 1), r['loss_sum_vs_leader'], -r['turns_removed'])
        )
        selection_rule = 'No sparse candidate cleared the near-competitive envelope, so choose the best damage-per-turn trade-off instead.'

    kf36_rechecked_candidates: list[str] = []
    kf36_gate_reason = 'KF36 not rerun because the recommended sparse candidate did not stay inside the near-competitive envelope.'
    if recommended_sparse['losses_vs_leader']['mean_pct_error'] <= 0.015 + EPS and recommended_sparse['losses_vs_leader']['max_pct_error'] <= 0.250 + EPS:
        name = recommended_sparse['candidate_name']
        if name in {'relax_anchor11_q4_backdwell'}:
            kf36_rechecked_candidates.append(name)
            kf36_gate_reason = 'KF36 already existed on disk for the reused q4-relaxed control.'
        else:
            candidate = candidates_by_name[name]
            family = recommended_sparse['family']
            hypothesis_id = recommended_sparse['hypothesis_id']
            kf_payload, kf_mode, kf_path = run_candidate_payload(mod, candidate, 'kf36_noisy', args.noise_scale, force_rerun=args.force_rerun)
            kf_payload = attach_att0(kf_path, kf_payload, name, 'kf36_noisy', family, hypothesis_id)
            recommended_sparse['result_files']['kf36'] = str(kf_path)
            recommended_sparse['result_modes']['kf36'] = kf_mode
            recommended_sparse['kf36'] = compact_metrics(kf_payload)
            recommended_sparse['delta_vs_leader_kf36'] = delta_vs_reference(references['full_best']['kf36'], recommended_sparse['kf36'])
            kf36_rechecked_candidates.append(name)
            kf36_gate_reason = 'KF36 rerun was triggered because the recommended sparse candidate stayed genuinely competitive on Markov42.'

    recommended_sparse_timing_table = build_timing_table(candidates_by_name[recommended_sparse['candidate_name']])

    required_comparison_rows = [
        {
            'label': refs_label,
            'extra_turns': extra_turns,
            'markov42_triplet': markov_triplet,
            'kf36_triplet': kf_triplet,
            'note': note,
            'delta_vs_recommended_sparse_markov42': delta_vs_reference(markov_payload, recommended_sparse['markov42']),
        }
        for refs_label, extra_turns, markov_payload, markov_triplet, kf_triplet, note in [
            (
                references['full_best']['label'],
                leader_extra_turns,
                references['full_best']['markov42'],
                references['full_best']['markov42_triplet'],
                references['full_best']['kf36_triplet'],
                'full corrected best parent',
            ),
            (
                references['faithful12']['label'],
                0,
                references['faithful12']['markov42'],
                references['faithful12']['markov42_triplet'],
                references['faithful12']['kf36_triplet'],
                'faithful 12-position backbone',
            ),
            (
                references['default18']['label'],
                6,
                references['default18']['markov42'],
                references['default18']['markov42_triplet'],
                references['default18']['kf36_triplet'],
                'default 18-position reference',
            ),
            (
                f"recommended sparse / {recommended_sparse['candidate_name']}",
                recommended_sparse['extra_turns'],
                recommended_sparse['markov42'],
                overall_triplet(recommended_sparse['markov42']),
                overall_triplet(recommended_sparse['kf36']) if recommended_sparse.get('kf36') is not None else 'not rerun',
                'best sparse trade-off found in this pruning batch',
            ),
        ]
    ]

    core_motifs = [item['motif_label'] for item in ranking if item['classification'] in {'core', 'critical timing lever'}]
    removable_motifs = [item['motif_label'] for item in ranking if item['classification'] in {'removable candidate', 'important timing lever'}]

    if recommended_sparse['losses_vs_leader']['mean_pct_error'] <= 0.015 + EPS and recommended_sparse['losses_vs_leader']['max_pct_error'] <= 0.250 + EPS:
        statement = (
            f"A genuinely useful sparse path exists: `{recommended_sparse['candidate_name']}` removes {recommended_sparse['turns_removed']} extra turns while keeping the full-parent losses to "
            f"{recommended_sparse['losses_vs_leader']['mean_pct_error']:.3f} mean-points and {recommended_sparse['losses_vs_leader']['max_pct_error']:.3f} max-points."
        )
    else:
        statement = (
            f"No sparse candidate preserved the full corrected best cleanly enough to replace it outright, but `{recommended_sparse['candidate_name']}` is still the best interpretability / compression trade-off by damage-per-turn."
        )

    summary = {
        'task': 'chapter-3 corrected sparse ablation / pruning batch',
        'report_date': REPORT_DATE,
        'noise_scale': args.noise_scale,
        'corrected_att0_deg': ATT0_DEG,
        'references': references,
        'full_best_structure': {
            'candidate_name': 'relay_l11back0p5_l12y0p125_on_entry',
            'extra_turns': leader_extra_turns,
            'total_time_s': leader_total_time_s,
            'motif_order': list(MOTIF_META.keys()),
        },
        'primary_plan': PRIMARY_PLAN,
        'primary_rows_sorted': primary_rows_sorted,
        'motif_importance_ranking': ranking,
        'combo_gate_reason': combo_gate_reason,
        'combo_rows_sorted': combo_rows_sorted,
        'selection_rule': selection_rule,
        'recommended_sparse_candidate': recommended_sparse,
        'recommended_sparse_timing_table': recommended_sparse_timing_table,
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
        'kf36_gate_reason': kf36_gate_reason,
        'required_comparison_rows': required_comparison_rows,
        'bottom_line': {
            'core_motifs': core_motifs,
            'removable_motifs': removable_motifs,
            'statement': statement,
        },
    }

    REPORT_PATH.write_text(render_report(summary), encoding='utf-8')
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({
        'report_path': str(REPORT_PATH),
        'summary_path': str(SUMMARY_PATH),
        'recommended_sparse_candidate': recommended_sparse['candidate_name'],
        'recommended_sparse_markov42': overall_triplet(recommended_sparse['markov42']),
        'recommended_sparse_kf36': overall_triplet(recommended_sparse['kf36']) if recommended_sparse.get('kf36') is not None else None,
        'turns_removed': recommended_sparse['turns_removed'],
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
