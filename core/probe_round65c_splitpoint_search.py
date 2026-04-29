from __future__ import annotations

import copy
import gc
import json
from pathlib import Path

import probe_round65b_dualgate_repair as base

RESULTS_DIR = base.RESULTS_DIR
REPORTS_DIR = base.REPORTS_DIR
SUMMARY_DIR = base.SUMMARY_DIR
SOURCE_FILE = base.SOURCE_FILE
R53_METHOD_FILE = base.R53_METHOD_FILE
ROUND65_SUMMARY_JSON = base.ROUND65_SUMMARY_JSON
ROUND61_REF_JSON = base.ROUND61_REF_JSON
ROUND61_BASE_NAME = base.ROUND61_BASE_NAME

OUTPUT_JSON = RESULTS_DIR / 'round65c_splitpoint_probe_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'round65c_splitpoint_candidates.json'
REPORT_MD = REPORTS_DIR / 'psins_round65c_splitpoint_probe_2026-03-29.md'
ROUND65C_RECORD_MD = SUMMARY_DIR / 'round65c_splitpoint_record_2026-03-29.md'

AVG_FLOOR = 0.755
GAP_VALUES = [-0.24, -0.18, -0.12, -0.06, 0.00, 0.06, 0.12, 0.18, 0.24, 0.32, 0.40]
BASE_FEEDBACK = {
    'target_nis': 1.0,
    'ema_beta': 0.03,
    'slope': 1.00,
    'warmup_static_meas': 8,
    'power': 1.0,
}
BASE_SCD_GATE = {
    'target_nis': 1.0,
    'ema_beta': 0.12,
    'slope': 2.10,
    'warmup_static_meas': 8,
    'power': 1.35,
}
BASE_SCD_CFG = {
    'mode': 'once_per_phase',
    'alpha': 0.9987,
    'transition_duration': 2.0,
    'target': 'xxzz_pair',
    'bias_to_target': True,
    'apply_policy_names': ['iter2_commit'],
}
BASE_ITER_PATCHES = {
    1: {
        'state_alpha_mult': {16: 1.014, 21: 1.012},
    },
}


def _clip_floor(x: float) -> float:
    return max(0.55, min(0.96, float(x)))


def _gap_tag(gap: float) -> str:
    mag = int(round(abs(gap) * 100))
    return f"m{mag:02d}" if gap < 0 else f"p{mag:02d}"


def _make_candidate(gap: float) -> dict:
    fb = _clip_floor(AVG_FLOOR + gap / 2.0)
    scd = _clip_floor(AVG_FLOOR - gap / 2.0)
    tag = _gap_tag(gap)
    return {
        'name': f'r65c_split_gap_{tag}',
        'description': (
            f'1D split-point sweep on the dual-channel axis with fixed average floor {AVG_FLOOR:.3f}; '
            f'gap={gap:+.2f} -> feedback_floor={fb:.3f}, scd_floor={scd:.3f}.'
        ),
        'rationale': (
            'Hold the dual-channel skeleton fixed and search only the feedback-vs-SCD floor separation, '
            'so the optimum really answers where the split point should sit on this mechanism axis.'
        ),
        'gap': float(gap),
        'avg_floor': float(AVG_FLOOR),
        'feedback_channel': {
            **copy.deepcopy(BASE_FEEDBACK),
            'gate_floor': float(fb),
            'apply_floor': float(fb),
        },
        'scd_channel': {
            **copy.deepcopy(BASE_SCD_GATE),
            'gate_floor': float(scd),
            'apply_floor': float(scd),
        },
        'scd': copy.deepcopy(BASE_SCD_CFG),
        'iter_patches': copy.deepcopy(BASE_ITER_PATCHES),
    }


ROUND65C_CANDIDATES = [_make_candidate(gap) for gap in GAP_VALUES]


def _render_report(summary: dict) -> str:
    lines: list[str] = []
    lines.append('<callout emoji="🎯" background-color="light-green">')
    lines.append('Round65-C 把“分离点”明确收敛成一条 **1D split-point axis**：只扫 `feedback_floor - scd_floor`，平均 floor 固定，其余 dual-channel 骨架不动。')
    lines.append('</callout>')
    lines.append('')
    lines.append('## 1. Fixed setup')
    lines.append('')
    lines.append('- base mechanism: `Round65-B dual-channel split gate` skeleton')
    lines.append('- fixed average floor: `0.755`')
    lines.append('- searched axis: `gap = feedback_floor - scd_floor`')
    lines.append('- same dataset / same noise / same seed as Round65 mainline (seed=42)')
    lines.append(f"- Round61 reference: `{summary['base_round61_json']}`")
    lines.append(f"- Round65 reference: `{summary['base_round65_reference_name']}` / `{summary['base_round65_reference_json']}`")
    lines.append('')
    lines.append('## 2. Split-point sweep')
    lines.append('')
    lines.append('| candidate | gap | fb_floor | scd_floor | dKg_yy ΔR61 | dKa_xx ΔR61 | rx_y ΔR61 | dKg_xy ΔR61 | dKg_xx ΔR61 | dKg_zz ΔR61 | mean ΔR61 | max ΔR61 | score | note |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for name in summary['candidate_order']:
        cand = summary['candidates'][name]
        d61 = cand['delta_vs_round61']
        lines.append(
            f"| `{name}` | {cand['gap']:+.2f} | {cand['feedback_floor']:.3f} | {cand['scd_floor']:.3f} | "
            f"{d61['dKg_yy']:.6f} | {d61['dKa_xx']:.6f} | {d61['rx_y']:.6f} | {d61['dKg_xy']:.6f} | "
            f"{d61['dKg_xx']:.6f} | {d61['dKg_zz']:.6f} | {d61['mean_pct_error']:.6f} | {d61['max_pct_error']:.6f} | "
            f"{cand['selection']['score']:.6f} | {cand['selection']['note']} |"
        )
    lines.append('')
    lines.append('## 3. Best split point')
    lines.append('')
    best = summary['strongest_signal']
    lines.append(f"- best candidate in this search: `{best['name']}`")
    lines.append(f"- best gap: `{best['gap']:+.2f}`")
    lines.append(f"- floors: feedback=`{best['feedback_floor']:.3f}`, scd=`{best['scd_floor']:.3f}`")
    lines.append(f"- summary: {best['signal']}")
    if summary['winner']:
        lines.append('- decision: clean winner found, promotable')
    else:
        lines.append('- decision: no clean winner; split-point optimum still stays probe-only')
        lines.append(f"- reason: {summary['no_winner_reason']}")
    lines.append('')
    lines.append('## 4. Interpretation')
    lines.append('')
    lines.append(f"- interpretation: {summary['interpretation']}")
    lines.append(f"- next move: {summary['next_best_move']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def _render_round_record(summary: dict) -> str:
    lines: list[str] = []
    lines.append('# Round65-C Record (split-point search)')
    lines.append('')
    lines.append('## A. Round 基本信息')
    lines.append('- Round name: Round65C_SplitPoint_Search')
    lines.append('- Round type: `repair-axis search`')
    lines.append(f'- Base candidate: `{ROUND61_BASE_NAME}`')
    lines.append('- Dataset / regime: `D_ref_mainline` (same fixed noisy dataset as Round65 / Round65-B)')
    lines.append('- D_ref_mainline definition:')
    lines.append('  - source trajectory: `method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset`')
    lines.append(f"  - arw = `{summary['dataset']['noise_config']['arw_dpsh']} * dpsh`")
    lines.append(f"  - vrw = `{summary['dataset']['noise_config']['vrw_ugpsHz']} * ugpsHz`")
    lines.append(f"  - bi_g = `{summary['dataset']['noise_config']['bi_g_dph']} * dph`")
    lines.append(f"  - bi_a = `{summary['dataset']['noise_config']['bi_a_ug']} * ug`")
    lines.append(f"  - tau_g = tau_a = `{summary['dataset']['noise_config']['tau_g']}`")
    lines.append(f"  - seed = `{summary['dataset']['noise_config']['seed']}`")
    lines.append('')
    lines.append('## B. 本轮目标')
    lines.append('- 不再散着试 dual-channel patch；把“分离点”压缩成一条可解释的 1D 轴。')
    lines.append('- 搜索轴定义：`gap = feedback_floor - scd_floor`。')
    lines.append('- 约束：平均 floor 固定为 `0.755`，其余 dual-channel skeleton 固定，避免混入额外自由度。')
    lines.append('')
    lines.append('## C. Fixed skeleton')
    lines.append(f'- feedback base cfg: `{json.dumps(BASE_FEEDBACK, ensure_ascii=False)}`')
    lines.append(f'- scd gate base cfg: `{json.dumps(BASE_SCD_GATE, ensure_ascii=False)}`')
    lines.append(f'- scd cfg: `{json.dumps(BASE_SCD_CFG, ensure_ascii=False)}`')
    lines.append(f'- iter patch: `{json.dumps(BASE_ITER_PATCHES, ensure_ascii=False)}`')
    lines.append('')
    lines.append('## D. Candidate design')
    for idx, candidate in enumerate(ROUND65C_CANDIDATES, start=1):
        lines.append(f'### candidate {idx}')
        lines.append(f"- name: `{candidate['name']}`")
        lines.append(f"- gap: `{candidate['gap']:+.2f}`")
        lines.append(f"- feedback floor: `{candidate['feedback_channel']['gate_floor']:.3f}`")
        lines.append(f"- scd floor: `{candidate['scd_channel']['gate_floor']:.3f}`")
        lines.append(f"- rationale: {candidate['rationale']}")
        lines.append('')
    lines.append('## E. Clean-win gate')
    lines.append('- same-dataset vs Round61: mean<0, max<=0, dKg_xx<0')
    lines.append('- hard-protected metrics must not regress: dKg_xy / dKg_yy / dKa_xx / rx_y / ry_z')
    lines.append('- repair-first protected set: dKg_yy / dKa_xx / rx_y')
    lines.append('')
    lines.append('## F. Result summary')
    lines.append(f"- result class: `{summary['result_classification']}`")
    if summary['winner']:
        lines.append(f"- winner: `{summary['winner']['name']}`")
    else:
        lines.append('- winner: none')
    lines.append(f"- conclusion: {summary['conclusion_line']}")
    lines.append(f"- strongest signal: {summary['strongest_signal']['signal']}")
    lines.append('')
    lines.append('## G. Interpretation')
    lines.append(f"- {summary['interpretation']}")
    lines.append(f"- next move: {summary['next_best_move']}")
    lines.append('')
    lines.append('## H. Artifacts')
    lines.append(f'- candidate_json: `{CANDIDATE_JSON}`')
    lines.append(f'- summary_json: `{OUTPUT_JSON}`')
    lines.append(f'- report_md: `{REPORT_MD}`')
    lines.append(f'- round_record_md: `{ROUND65C_RECORD_MD}`')
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    round65_summary = json.loads(ROUND65_SUMMARY_JSON.read_text(encoding='utf-8'))
    round61_payload = json.loads(ROUND61_REF_JSON.read_text(encoding='utf-8'))

    round65_ref_name = round65_summary['strongest_signal']['name']
    round65_ref_json = Path(round65_summary['candidates'][round65_ref_name]['param_errors_json'])
    round65_ref_payload = json.loads(round65_ref_json.read_text(encoding='utf-8'))

    source_mod = base.load_module('markov_pruned_source_round65c', str(SOURCE_FILE))
    dataset = base._build_shared_dataset(source_mod)

    candidate_dump = {
        'round_name': 'Round65C_SplitPoint_Search',
        'round_type': 'repair-axis search',
        'innovation_direction': 'dual-channel split-point search on feedback-vs-SCD axis',
        'base_round61_candidate': ROUND61_BASE_NAME,
        'same_dataset_round61_json': str(ROUND61_REF_JSON),
        'same_dataset_round65_reference': {
            'name': round65_ref_name,
            'json': str(round65_ref_json),
        },
        'dataset': {
            'source_file': str(SOURCE_FILE),
            'source_trajectory_reference': 'method_42state_gm1_round53_internalized_trustcov_release.py::_build_dataset',
            'noise_config': dataset['noise_config'],
            'constraint_note': 'Use exactly the same fixed noisy dataset/noise strength/seed as Round65 mainline',
        },
        'search_axis': {
            'definition': 'gap = feedback_floor - scd_floor',
            'avg_floor_fixed': AVG_FLOOR,
            'gap_values': GAP_VALUES,
        },
        'protected_metrics': base.HARD_PROTECTED_KEYS,
        'repair_metrics': base.PRIMARY_REPAIR_KEYS,
        'round65c_candidates': ROUND65C_CANDIDATES,
    }
    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    out = {
        'round_name': 'Round65C_SplitPoint_Search',
        'round_type': 'repair-axis search',
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
        'candidate_order': [c['name'] for c in ROUND65C_CANDIDATES],
        'candidates': {},
        'winner': None,
        'no_winner_reason': None,
        'result_classification': None,
        'strongest_signal': None,
        'interpretation': None,
        'next_best_move': None,
        'conclusion_line': None,
    }

    ts = dataset['ts']
    pos0 = dataset['pos0']
    imu_noisy = dataset['imu_noisy']
    bi_g = dataset['bi_g']
    bi_a = dataset['bi_a']
    tau_g = dataset['tau_g']
    tau_a = dataset['tau_a']

    for idx, candidate in enumerate(ROUND65C_CANDIDATES, start=1):
        merged_candidate = base._merge_round65b_candidate(candidate)

        method_mod = base.load_module(f'markov_method_round65c_candidate_{idx}', str(R53_METHOD_FILE))
        method_mod = base._build_patched_method(method_mod, merged_candidate)

        result = list(base._run_internalized_hybrid_scd_dualgate(
            method_mod,
            source_mod,
            imu_noisy,
            pos0,
            ts,
            bi_g=bi_g,
            bi_a=bi_a,
            tau_g=tau_g,
            tau_a=tau_a,
            label=f'R65C-SPLIT-{idx}',
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

        payload_candidate = base._compute_payload(
            source_mod,
            clbt_candidate,
            variant=f"r65c_splitpoint_{candidate['name']}",
            method_file='probe_round65c_splitpoint_search::dual_channel_splitpoint_axis',
            extra={
                'dataset_noise_config': dataset['noise_config'],
                'base_round61_candidate': ROUND61_BASE_NAME,
                'gap': float(candidate['gap']),
                'avg_floor': float(candidate['avg_floor']),
                'feedback_channel': copy.deepcopy(candidate['feedback_channel']),
                'scd_channel': copy.deepcopy(candidate['scd_channel']),
                'scd_cfg': copy.deepcopy(candidate['scd']),
                'runtime_log': runtime_log,
            },
        )
        del clbt_candidate
        gc.collect()

        candidate_json_path = RESULTS_DIR / f"R65C_splitpoint_{candidate['name']}_param_errors.json"
        candidate_json_path.write_text(json.dumps(payload_candidate, ensure_ascii=False, indent=2), encoding='utf-8')

        delta_vs_r61 = {
            **base._delta_block(payload_candidate['focus_scale_pct'], round61_payload['focus_scale_pct']),
            **base._delta_block(payload_candidate['lever_guard_pct'], round61_payload['lever_guard_pct']),
            **base._delta_block(payload_candidate['overall'], round61_payload['overall']),
        }
        delta_vs_r65_ref = {
            **base._delta_block(payload_candidate['focus_scale_pct'], round65_ref_payload['focus_scale_pct']),
            **base._delta_block(payload_candidate['lever_guard_pct'], round65_ref_payload['lever_guard_pct']),
            **base._delta_block(payload_candidate['overall'], round65_ref_payload['overall']),
        }

        score, penalties = base._score_candidate(delta_vs_r61, delta_vs_r65_ref)
        note = base._selection_note(delta_vs_r61, delta_vs_r65_ref, penalties)

        out['candidates'][candidate['name']] = {
            'description': candidate['description'],
            'rationale': candidate['rationale'],
            'gap': float(candidate['gap']),
            'avg_floor': float(candidate['avg_floor']),
            'feedback_floor': float(candidate['feedback_channel']['gate_floor']),
            'scd_floor': float(candidate['scd_channel']['gate_floor']),
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
            'vs_round61_relative_improvement': base._relative_improvement_block(
                round61_payload,
                payload_candidate,
                ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'mean_pct_error', 'median_pct_error', 'max_pct_error'],
            ),
            'vs_round65_ref_relative_improvement': base._relative_improvement_block(
                round65_ref_payload,
                payload_candidate,
                ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'mean_pct_error', 'median_pct_error', 'max_pct_error'],
            ),
        }

        print(candidate['name'], json.dumps({
            'gap': candidate['gap'],
            'feedback_floor': candidate['feedback_channel']['gate_floor'],
            'scd_floor': candidate['scd_channel']['gate_floor'],
            'delta_vs_round61': delta_vs_r61,
            'score': score,
            'note': note,
        }, ensure_ascii=False))

    ordered = sorted(
        [(name, out['candidates'][name]['selection']['score']) for name in out['candidate_order']],
        key=lambda x: x[1],
        reverse=True,
    )
    best_name, best_score = ordered[0]
    best = out['candidates'][best_name]
    best_delta_r61 = best['delta_vs_round61']
    best_penalties = best['selection']['penalties']

    if base._is_clean_winner(best_delta_r61, best_penalties):
        out['winner'] = {
            'name': best_name,
            'score': float(best_score),
            'reason': 'Clean same-dataset winner over Round61 on the 1D split-point axis.',
        }
        out['result_classification'] = 'clean win'
        out['conclusion_line'] = 'Round65-C found a clean promotable winner on the split-point axis.'
    else:
        out['winner'] = None
        out['no_winner_reason'] = 'No split point passed the Round61 clean-win gate under the same fixed dataset.'
        out['result_classification'] = 'partial signal' if any(float(best_delta_r61[k]) < 0.0 for k in base.PRIMARY_REPAIR_KEYS) else 'no useful signal'
        out['conclusion_line'] = 'Round65-C found the best split point in this family, but it still does not cleanly beat Round61.'

    out['strongest_signal'] = {
        'name': best_name,
        'gap': float(best['gap']),
        'feedback_floor': float(best['feedback_floor']),
        'scd_floor': float(best['scd_floor']),
        'signal': (
            f"best split candidate {best_name}: gap={best['gap']:+.2f}, "
            f"fb={best['feedback_floor']:.3f}, scd={best['scd_floor']:.3f}; "
            f"dKg_xy Δ={best_delta_r61['dKg_xy']:.6f}, dKg_yy Δ={best_delta_r61['dKg_yy']:.6f}, "
            f"dKa_xx Δ={best_delta_r61['dKa_xx']:.6f}, rx_y Δ={best_delta_r61['rx_y']:.6f}, "
            f"dKg_xx Δ={best_delta_r61['dKg_xx']:.6f}, dKg_zz Δ={best_delta_r61['dKg_zz']:.6f}, "
            f"mean Δ={best_delta_r61['mean_pct_error']:.6f}, max Δ={best_delta_r61['max_pct_error']:.6f}"
        ),
        'regressions': str(best_penalties),
    }

    if best['gap'] < -1e-9:
        side = '偏 SCD 一侧（feedback 更收、SCD 更放）'
    elif best['gap'] > 1e-9:
        side = '偏 feedback 一侧（feedback 更放、SCD 更收）'
    else:
        side = '接近对称分离点'

    out['interpretation'] = (
        f"在固定 dual-channel skeleton 与固定平均 floor 的前提下，当前最优 split point 落在 **{side}**；"
        f"也就是说，这条轴上最好的结果对应 gap={best['gap']:+.2f}，而不是盲目把 feedback 和 SCD 拉得越开越好。"
    )
    out['next_best_move'] = (
        f"Lock the best gap `{best['gap']:+.2f}` as center, then run one ultra-narrow local refinement on ±0.03 / ±0.06 only, "
        "or conclude that the split-point axis itself has saturated if even the best point remains clearly behind Round61."
    )

    OUTPUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
    REPORT_MD.write_text(_render_report(out), encoding='utf-8')
    ROUND65C_RECORD_MD.write_text(_render_round_record(out), encoding='utf-8')

    print(f'Wrote {CANDIDATE_JSON}')
    print(f'Wrote {OUTPUT_JSON}')
    print(f'Wrote {REPORT_MD}')
    print(f'Wrote {ROUND65C_RECORD_MD}')
    print('__RESULT_JSON__=' + json.dumps({
        'candidate_json': str(CANDIDATE_JSON),
        'summary_json': str(OUTPUT_JSON),
        'report_md': str(REPORT_MD),
        'round_record_md': str(ROUND65C_RECORD_MD),
        'winner': out['winner'],
        'result_classification': out['result_classification'],
        'best_candidate': out['strongest_signal'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
