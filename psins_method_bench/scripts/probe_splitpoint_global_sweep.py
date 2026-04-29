from __future__ import annotations

import copy
import gc
import json
from pathlib import Path

from probe_round65b_dualgate_repair import (
    CANDIDATE_JSON as ROUND65B_CANDIDATE_JSON,
    METHOD_DIR,
    OUTPUT_JSON as ROUND65B_OUTPUT_JSON,
    REPORTS_DIR,
    RESULTS_DIR,
    ROUND61_BASE_NAME,
    ROUND61_REF_JSON,
    ROUND65B_CANDIDATES,
    ROUND65_SUMMARY_JSON,
    R53_METHOD_FILE,
    SOURCE_FILE,
    SUMMARY_DIR,
    _build_patched_method,
    _build_shared_dataset,
    _compute_payload,
    _delta_block,
    _is_clean_winner,
    _merge_round65b_candidate,
    _relative_improvement_block,
    _run_internalized_hybrid_scd_dualgate,
    _score_candidate,
    _selection_note,
    load_module,
)

OUTPUT_JSON = RESULTS_DIR / 'splitpoint_global_sweep_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'splitpoint_global_sweep_candidates.json'
REPORT_MD = REPORTS_DIR / 'psins_splitpoint_global_sweep_2026-03-29.md'
ROUND_RECORD_MD = SUMMARY_DIR / 'splitpoint_global_sweep_record_2026-03-29.md'

BASE_CANDIDATE_NAME = 'r65b_split_xxzz_fb96_frozen'
SPLIT_POINT_VALUES = [round(x, 2) for x in [0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]]


def _get_base_candidate() -> dict:
    for candidate in ROUND65B_CANDIDATES:
        if candidate['name'] == BASE_CANDIDATE_NAME:
            return copy.deepcopy(candidate)
    raise KeyError(BASE_CANDIDATE_NAME)


def _build_split_candidate(split_point: float) -> dict:
    candidate = _get_base_candidate()
    tag = f"sp{int(round(split_point * 100)):03d}"
    candidate['name'] = f'splitpoint_{tag}'
    candidate['description'] = (
        'Single-knob global sweep on the dual-gate split point: '
        'only the feedback freeze point is changed, while the rest of the Round65-B best body stays fixed.'
    )
    candidate['rationale'] = (
        'Operationalize “分离点” as feedback_gate floor / freeze point, and search the full allowed axis '
        'under the same fixed dataset to see where the split-gate body scores best.'
    )
    candidate['feedback_channel']['gate_floor'] = float(split_point)
    candidate['feedback_channel']['apply_floor'] = float(split_point)
    return candidate


def _render_report(out: dict) -> str:
    lines: list[str] = []
    lines.append('# Split-Point Global Sweep Record')
    lines.append('')
    lines.append('## A. Experiment definition')
    lines.append('- axis: `dual-channel split-gate feedback freeze point`')
    lines.append('- operational definition of split point: `feedback_channel.gate_floor == feedback_channel.apply_floor`')
    lines.append(f"- sweep values: `{', '.join(f'{v:.2f}' for v in out['split_point_values'])}`")
    lines.append(f"- base body: `{BASE_CANDIDATE_NAME}` (all non-split knobs fixed)")
    lines.append('- dataset constraint: same fixed `D_ref_mainline` dataset as Round65 / Round65-B / Round66 / Round68 / Round69')
    lines.append('')
    lines.append('## B. Best point')
    best = out['best_point']
    lines.append(f"- best split point: `{best['split_point']:.2f}`")
    lines.append(f"- best candidate: `{best['name']}`")
    lines.append(f"- score: `{best['score']:.6f}`")
    lines.append(f"- result classification: `{out['result_classification']}`")
    lines.append(f"- clean winner: `{out['winner']}`")
    lines.append(f"- conclusion: {out['conclusion_line']}")
    lines.append('')
    lines.append('## C. Scoreboard')
    lines.append('')
    lines.append('| split_point | score | dKg_xx | dKg_xy | dKg_yy | dKa_xx | rx_y | mean | max | note |')
    lines.append('|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for row in out['scoreboard']:
        note = row['selection_note'].replace('|', '/').replace('\n', ' ')
        lines.append(
            f"| {row['split_point']:.2f} | {row['score']:.3f} | {row['delta_vs_round61']['dKg_xx']:.3f} | "
            f"{row['delta_vs_round61']['dKg_xy']:.3f} | {row['delta_vs_round61']['dKg_yy']:.3f} | "
            f"{row['delta_vs_round61']['dKa_xx']:.3f} | {row['delta_vs_round61']['rx_y']:.3f} | "
            f"{row['delta_vs_round61']['mean_pct_error']:.3f} | {row['delta_vs_round61']['max_pct_error']:.3f} | {note} |"
        )
    lines.append('')
    lines.append('## D. Reading')
    lines.append(f"- trend summary: {out['trend_summary']}")
    lines.append(f"- next move: {out['next_best_move']}")
    lines.append('')
    lines.append('## E. Artifacts')
    lines.append(f"- candidates: `{CANDIDATE_JSON}`")
    lines.append(f"- summary: `{OUTPUT_JSON}`")
    lines.append(f"- report: `{REPORT_MD}`")
    lines.append(f"- round record: `{ROUND_RECORD_MD}`")
    lines.append('')
    return '\n'.join(lines) + '\n'


def _render_round_record(out: dict) -> str:
    lines: list[str] = []
    lines.append('# Split-Point Global Sweep Round Record')
    lines.append('')
    lines.append('## 1. Goal')
    lines.append('- 用户要求：尽可能找出 dual-channel split-gate 的“全局最优分离点”。')
    lines.append('- 本轮把“分离点”严格 operationalize 为 **feedback freeze point / feedback gate floor**。')
    lines.append('- 其余旋钮全部固定在 `r65b_split_xxzz_fb96_frozen` 体型上，不混入额外大改。')
    lines.append('')
    lines.append('## 2. Sweep axis')
    lines.append(f"- split_point values: `{', '.join(f'{v:.2f}' for v in out['split_point_values'])}`")
    lines.append('- fixed knobs: Round65-B best body except feedback floor / apply_floor')
    lines.append('- fixed dataset: same `D_ref_mainline`, seed=42')
    lines.append('')
    lines.append('## 3. Best result')
    best = out['best_point']
    lines.append(f"- best split point: `{best['split_point']:.2f}`")
    lines.append(f"- best candidate name: `{best['name']}`")
    lines.append(f"- score: `{best['score']:.6f}`")
    lines.append(f"- delta vs Round61: `{json.dumps(best['delta_vs_round61'], ensure_ascii=False)}`")
    lines.append(f"- penalties: `{json.dumps(best['penalties'], ensure_ascii=False)}`")
    lines.append('')
    lines.append('## 4. Judgment')
    lines.append(f"- conclusion: {out['conclusion_line']}")
    lines.append(f"- trend summary: {out['trend_summary']}")
    lines.append(f"- next best move: {out['next_best_move']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    round65_summary = json.loads(ROUND65_SUMMARY_JSON.read_text(encoding='utf-8'))
    round61_payload = json.loads(ROUND61_REF_JSON.read_text(encoding='utf-8'))
    round65b_candidate_dump = json.loads(ROUND65B_CANDIDATE_JSON.read_text(encoding='utf-8')) if ROUND65B_CANDIDATE_JSON.exists() else None
    round65b_summary = json.loads(ROUND65B_OUTPUT_JSON.read_text(encoding='utf-8')) if ROUND65B_OUTPUT_JSON.exists() else None

    round65_ref_name = round65_summary['strongest_signal']['name']
    round65_ref_json = Path(round65_summary['candidates'][round65_ref_name]['param_errors_json'])
    round65_ref_payload = json.loads(round65_ref_json.read_text(encoding='utf-8'))

    source_mod = load_module('markov_pruned_source_splitpoint_sweep', str(SOURCE_FILE))
    dataset = _build_shared_dataset(source_mod)

    candidate_dump = {
        'round_name': 'SplitPoint_Global_Sweep',
        'round_type': 'global one-knob sweep',
        'base_round61_candidate': ROUND61_BASE_NAME,
        'same_dataset_round61_json': str(ROUND61_REF_JSON),
        'same_dataset_round65_reference': {
            'name': round65_ref_name,
            'json': str(round65_ref_json),
        },
        'base_body_candidate': BASE_CANDIDATE_NAME,
        'operational_definition': 'split_point := feedback_channel.gate_floor := feedback_channel.apply_floor',
        'split_point_values': SPLIT_POINT_VALUES,
        'dataset': {
            'source_file': str(SOURCE_FILE),
            'noise_config': dataset['noise_config'],
            'constraint_note': 'Use exactly the same fixed noisy dataset/noise strength/seed as Round65 mainline and Round65-B.',
        },
        'reference_round65b_candidate_dump': round65b_candidate_dump,
        'reference_round65b_summary': round65b_summary,
        'candidates': [],
    }

    out = {
        'round_name': 'SplitPoint_Global_Sweep',
        'round_type': 'global one-knob sweep',
        'operational_definition': 'split_point := feedback_channel.gate_floor := feedback_channel.apply_floor',
        'base_body_candidate': BASE_CANDIDATE_NAME,
        'split_point_values': SPLIT_POINT_VALUES,
        'dataset': {
            'source_file': str(SOURCE_FILE),
            'noise_config': dataset['noise_config'],
            'seed': dataset['noise_config']['seed'],
        },
        'base_round61_candidate': ROUND61_BASE_NAME,
        'base_round61_json': str(ROUND61_REF_JSON),
        'base_round65_reference_name': round65_ref_name,
        'base_round65_reference_json': str(round65_ref_json),
        'candidate_json': str(CANDIDATE_JSON),
        'summary_json': str(OUTPUT_JSON),
        'scoreboard': [],
        'candidates': {},
        'best_point': None,
        'winner': None,
        'result_classification': None,
        'conclusion_line': None,
        'trend_summary': None,
        'next_best_move': None,
    }

    ts = dataset['ts']
    pos0 = dataset['pos0']
    imu_noisy = dataset['imu_noisy']
    bi_g = dataset['bi_g']
    bi_a = dataset['bi_a']
    tau_g = dataset['tau_g']
    tau_a = dataset['tau_a']

    for idx, split_point in enumerate(SPLIT_POINT_VALUES, start=1):
        candidate = _build_split_candidate(split_point)
        merged_candidate = _merge_round65b_candidate(candidate)

        method_mod = load_module(f'markov_method_splitpoint_candidate_{idx}', str(R53_METHOD_FILE))
        method_mod = _build_patched_method(method_mod, merged_candidate)

        result = list(_run_internalized_hybrid_scd_dualgate(
            method_mod,
            source_mod,
            imu_noisy,
            pos0,
            ts,
            bi_g=bi_g,
            bi_a=bi_a,
            tau_g=tau_g,
            tau_a=tau_a,
            label=f'SPLIT-SWEEP-{idx}',
            scd_cfg=merged_candidate['scd'],
            feedback_cfg=merged_candidate['feedback_channel'],
            scd_gate_cfg=merged_candidate['scd_channel'],
        ))
        clbt_candidate = result[0]
        runtime_log = {
            'schedule_log': result[4].get('schedule_log') if len(result) >= 5 else None,
            'feedback_log': result[4].get('feedback_log') if len(result) >= 5 else None,
            'scd_log': result[4].get('scd_log') if len(result) >= 5 else None,
            'dual_gate_log': result[4].get('dual_gate_log') if len(result) >= 5 else None,
        }
        del result
        gc.collect()

        payload_candidate = _compute_payload(
            source_mod,
            clbt_candidate,
            variant=f"splitpoint_global_sweep_{candidate['name']}",
            method_file='probe_splitpoint_global_sweep::feedback_floor_single_knob',
            extra={
                'dataset_noise_config': dataset['noise_config'],
                'base_round61_candidate': ROUND61_BASE_NAME,
                'base_body_candidate': BASE_CANDIDATE_NAME,
                'split_point': float(split_point),
                'feedback_channel': copy.deepcopy(candidate['feedback_channel']),
                'scd_channel': copy.deepcopy(candidate['scd_channel']),
                'scd_cfg': copy.deepcopy(candidate['scd']),
                'runtime_log': runtime_log,
            },
        )
        del clbt_candidate
        gc.collect()

        candidate_json_path = RESULTS_DIR / f"splitpoint_global_sweep_{candidate['name']}_param_errors.json"
        candidate_json_path.write_text(json.dumps(payload_candidate, ensure_ascii=False, indent=2), encoding='utf-8')

        delta_vs_r61 = {
            **_delta_block(payload_candidate['focus_scale_pct'], round61_payload['focus_scale_pct']),
            **_delta_block(payload_candidate['lever_guard_pct'], round61_payload['lever_guard_pct']),
            **_delta_block(payload_candidate['overall'], round61_payload['overall']),
        }
        delta_vs_r65_ref = {
            **_delta_block(payload_candidate['focus_scale_pct'], round65_ref_payload['focus_scale_pct']),
            **_delta_block(payload_candidate['lever_guard_pct'], round65_ref_payload['lever_guard_pct']),
            **_delta_block(payload_candidate['overall'], round65_ref_payload['overall']),
        }
        score, penalties = _score_candidate(delta_vs_r61, delta_vs_r65_ref)
        note = _selection_note(delta_vs_r61, delta_vs_r65_ref, penalties)

        row = {
            'name': candidate['name'],
            'split_point': float(split_point),
            'score': float(score),
            'delta_vs_round61': delta_vs_r61,
            'delta_vs_round65_ref': delta_vs_r65_ref,
            'selection_note': note,
            'penalties': penalties,
        }
        out['scoreboard'].append(row)

        out['candidates'][candidate['name']] = {
            'split_point': float(split_point),
            'description': candidate['description'],
            'rationale': candidate['rationale'],
            'feedback_channel': copy.deepcopy(candidate['feedback_channel']),
            'scd_channel': copy.deepcopy(candidate['scd_channel']),
            'scd_cfg': copy.deepcopy(candidate['scd']),
            'param_errors_json': str(candidate_json_path),
            'focus_scale_pct': payload_candidate['focus_scale_pct'],
            'lever_guard_pct': payload_candidate['lever_guard_pct'],
            'overall': payload_candidate['overall'],
            'delta_vs_round61': delta_vs_r61,
            'delta_vs_round65_ref': delta_vs_r65_ref,
            'selection': {
                'score': float(score),
                'penalties': penalties,
                'note': note,
            },
            'runtime_log': payload_candidate['extra']['runtime_log'],
            'vs_round61_relative_improvement': _relative_improvement_block(
                round61_payload,
                payload_candidate,
                ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'mean_pct_error', 'median_pct_error', 'max_pct_error'],
            ),
            'vs_round65_ref_relative_improvement': _relative_improvement_block(
                round65_ref_payload,
                payload_candidate,
                ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'mean_pct_error', 'median_pct_error', 'max_pct_error'],
            ),
        }

        candidate_dump['candidates'].append({
            'name': candidate['name'],
            'split_point': float(split_point),
            'feedback_channel': copy.deepcopy(candidate['feedback_channel']),
            'scd_channel': copy.deepcopy(candidate['scd_channel']),
            'scd_cfg': copy.deepcopy(candidate['scd']),
            'param_errors_json': str(candidate_json_path),
        })

        print(candidate['name'], json.dumps({
            'split_point': split_point,
            'delta_vs_round61': delta_vs_r61,
            'score': score,
            'penalties': penalties,
            'note': note,
        }, ensure_ascii=False))

    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    ordered = sorted(out['scoreboard'], key=lambda x: x['score'], reverse=True)
    best = ordered[0]
    best_name = best['name']
    best_delta = best['delta_vs_round61']
    best_penalties = best['penalties']
    clean = _is_clean_winner(best_delta, best_penalties)

    out['best_point'] = {
        'name': best_name,
        'split_point': best['split_point'],
        'score': best['score'],
        'delta_vs_round61': best_delta,
        'penalties': best_penalties,
        'selection_note': best['selection_note'],
    }

    if clean:
        out['winner'] = {
            'name': best_name,
            'split_point': best['split_point'],
            'score': best['score'],
            'reason': 'Clean same-dataset winner over Round61 under split-point global sweep.',
        }
        out['result_classification'] = 'clean win'
        out['conclusion_line'] = (
            f"Split-point global sweep found a clean winner at split_point={best['split_point']:.2f}."
        )
    else:
        out['winner'] = None
        out['result_classification'] = 'partial signal' if any(float(best_delta[k]) < 0.0 for k in ['dKg_yy', 'dKa_xx', 'rx_y']) else 'no useful signal'
        out['conclusion_line'] = (
            f"Split-point global sweep did not produce a clean winner over Round61; best score occurs at split_point={best['split_point']:.2f}."
        )

    min_floor = ordered[-1]['split_point']
    max_floor = ordered[0]['split_point']
    boundary_note = ''
    if abs(best['split_point'] - SPLIT_POINT_VALUES[-1]) < 1e-9:
        boundary_note = ' Best point lands on the upper boundary, so the sweep suggests the mechanism prefers almost fully frozen feedback.'
    elif abs(best['split_point'] - SPLIT_POINT_VALUES[0]) < 1e-9:
        boundary_note = ' Best point lands on the lower boundary, so the sweep suggests less freezing / more shared behavior.'
    out['trend_summary'] = (
        f"Across split_point={min_floor:.2f}..{SPLIT_POINT_VALUES[-1]:.2f}, the top score is {best['score']:.3f} at {best['split_point']:.2f}."
        + boundary_note
    )
    out['next_best_move'] = (
        'If the user wants one more refinement, keep the winning split point fixed and only run a tiny xx/zz SCD alpha ladder '
        '(e.g. ±0.0001 around the current alpha) to see whether a local no-regression pocket exists.'
    )

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    REPORT_MD.write_text(_render_report(out), encoding='utf-8')
    ROUND_RECORD_MD.write_text(_render_round_record(out), encoding='utf-8')

    print(f'Wrote {CANDIDATE_JSON}')
    print(f'Wrote {OUTPUT_JSON}')
    print(f'Wrote {REPORT_MD}')
    print(f'Wrote {ROUND_RECORD_MD}')
    print('__RESULT_JSON__=' + json.dumps({
        'candidate_json': str(CANDIDATE_JSON),
        'summary_json': str(OUTPUT_JSON),
        'report_md': str(REPORT_MD),
        'round_record_md': str(ROUND_RECORD_MD),
        'best_point': out['best_point'],
        'winner': out['winner'],
        'result_classification': out['result_classification'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
