from __future__ import annotations

import copy
import gc
import json
from pathlib import Path

import probe_round65b_dualgate_repair as base
import probe_round65c_splitpoint_search as splitbase
from compare_four_methods_shared_noise import build_shared_dataset

RESULTS_DIR = base.RESULTS_DIR
REPORTS_DIR = base.REPORTS_DIR
SUMMARY_DIR = base.SUMMARY_DIR
SOURCE_FILE = base.SOURCE_FILE
R53_METHOD_FILE = base.R53_METHOD_FILE
ROUND61_BASE_NAME = base.ROUND61_BASE_NAME

NOISE_SCALE = 0.08
NOISE_TAG = 'shared_noise0p08'
ROUND61_REF_JSON = RESULTS_DIR / 'R61_42state_gm1_round61_h_scd_state20_microtight_commit_shared_noise0p08_param_errors.json'
MARKOV_REF_JSON = RESULTS_DIR / 'M_markov_42state_gm1_shared_noise0p08_param_errors.json'
PURE_SCD_REF_JSON = RESULTS_DIR / 'SCD42_markov_neutral_shared_noise0p08_param_errors.json'
KF36_REF_JSON = RESULTS_DIR / 'KF36_shared_noise0p08_param_errors.json'
FOUR_METHODS_JSON = RESULTS_DIR / 'compare_four_methods_shared_noise0p08_compact.json'

OUTPUT_JSON = RESULTS_DIR / 'round65c_splitpoint_shared_noise0p08_summary.json'
CANDIDATE_JSON = RESULTS_DIR / 'round65c_splitpoint_shared_noise0p08_candidates.json'
REPORT_MD = REPORTS_DIR / 'psins_round65c_splitpoint_shared_noise0p08_2026-03-29.md'
ROUND_RECORD_MD = SUMMARY_DIR / 'round65c_splitpoint_shared_noise0p08_record_2026-03-29.md'

ROUND65C_CANDIDATES = splitbase.ROUND65C_CANDIDATES
AVG_FLOOR = splitbase.AVG_FLOOR
GAP_VALUES = splitbase.GAP_VALUES


def _score_candidate_vs_round61(delta_vs_r61: dict):
    penalties = []
    for key in base.HARD_PROTECTED_KEYS:
        value = float(delta_vs_r61[key])
        if value > 1e-9:
            penalties.append({'metric': key, 'delta': value})

    score = 0.0
    score += -1.35 * float(delta_vs_r61['dKg_yy'])
    score += -1.20 * float(delta_vs_r61['dKa_xx'])
    score += -1.20 * float(delta_vs_r61['rx_y'])
    score += -0.75 * float(delta_vs_r61['dKg_xx'])
    score += -0.55 * float(delta_vs_r61['dKg_zz'])
    score += -0.65 * float(delta_vs_r61['mean_pct_error'])
    score += -0.40 * float(delta_vs_r61['max_pct_error'])
    score += -0.25 * float(delta_vs_r61['dKg_xy'])
    score += -0.20 * float(delta_vs_r61['ry_z'])

    for p in penalties:
        score -= 1000.0 * float(p['delta'])
    return float(score), penalties


def _selection_note(delta_vs_r61: dict, penalties: list[dict]):
    repaired_keys = [k for k in base.PRIMARY_REPAIR_KEYS if float(delta_vs_r61[k]) < 0.0]
    if base._is_clean_winner(delta_vs_r61, penalties):
        return 'Clean same-dataset win over Round61 on shared noise0p08 with all protected metrics improved.'
    if repaired_keys:
        if penalties:
            return f'Partial repair signal on {repaired_keys}, but protected regression vs Round61 remains: {penalties}'
        return f'Partial repair signal: improved {repaired_keys} vs Round61, but clean-win mainline gate not met.'
    if penalties:
        return f'No useful repair signal; protected regression remains vs Round61: {penalties}'
    return 'Near-neutral variant; does not repair the key protected trio.'


def _render_report(summary: dict) -> str:
    lines: list[str] = []
    lines.append('<callout emoji="🌫️" background-color="light-blue">')
    lines.append('这轮不是 Round65 mainline 1x 噪声，而是 **文档 Q4G6d0ZU... 对应的 shared noise0p08**；split-point 采用 Round65-C 的严格 1D 定义。')
    lines.append('</callout>')
    lines.append('')
    lines.append('## 1. Fixed setup')
    lines.append('')
    lines.append('- document-aligned noise: `shared noise0p08`')
    lines.append('- fixed noise config: `arw=0.0004 dpsh, vrw=0.4 ugpsHz, bi_g=0.00016 dph, bi_a=0.4 ug, tau=300, seed=42`')
    lines.append('- base family: `round53_round61_shared`')
    lines.append('- split-point definition: `gap = feedback_floor - scd_floor` with average floor fixed')
    lines.append(f'- fixed average floor: `{AVG_FLOOR:.3f}`')
    lines.append(f"- searched gaps: `{', '.join(f'{g:+.2f}' for g in GAP_VALUES)}`")
    lines.append(f"- Round61 reference: `{summary['base_round61_json']}`")
    lines.append(f"- ladder references: `KF36 / Markov42 / Pure SCD / Round61` from `{FOUR_METHODS_JSON}`")
    lines.append('')
    lines.append('## 2. Split-point sweep scoreboard')
    lines.append('')
    lines.append('| candidate | gap | fb_floor | scd_floor | dKg_yy ΔR61 | dKa_xx ΔR61 | rx_y ΔR61 | dKg_xy ΔR61 | dKg_xx ΔR61 | dKg_zz ΔR61 | mean ΔR61 | max ΔR61 | score | note |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|')
    for name in summary['candidate_order']:
        cand = summary['candidates'][name]
        d61 = cand['delta_vs_round61']
        note = cand['selection']['note'].replace('|', '/').replace('\n', ' ')
        lines.append(
            f"| `{name}` | {cand['gap']:+.2f} | {cand['feedback_floor']:.3f} | {cand['scd_floor']:.3f} | "
            f"{d61['dKg_yy']:.6f} | {d61['dKa_xx']:.6f} | {d61['rx_y']:.6f} | {d61['dKg_xy']:.6f} | "
            f"{d61['dKg_xx']:.6f} | {d61['dKg_zz']:.6f} | {d61['mean_pct_error']:.6f} | {d61['max_pct_error']:.6f} | "
            f"{cand['selection']['score']:.6f} | {note} |"
        )
    lines.append('')
    lines.append('## 3. Best split point')
    lines.append('')
    best = summary['strongest_signal']
    lines.append(f"- best candidate: `{best['name']}`")
    lines.append(f"- best gap: `{best['gap']:+.2f}`")
    lines.append(f"- floors: feedback=`{best['feedback_floor']:.3f}`, scd=`{best['scd_floor']:.3f}`")
    lines.append(f"- summary: {best['signal']}")
    lines.append(f"- conclusion: {summary['conclusion_line']}")
    lines.append('')
    lines.append('## 4. Interpretation')
    lines.append('')
    lines.append(f"- interpretation: {summary['interpretation']}")
    lines.append(f"- next move: {summary['next_best_move']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def _render_round_record(summary: dict) -> str:
    lines: list[str] = []
    lines.append('# Round65-C Shared Noise0p08 Split-Point Record')
    lines.append('')
    lines.append('## A. Round 基本信息')
    lines.append('- Round name: Round65C_SplitPoint_Search_Shared_Noise0p08')
    lines.append('- Round type: `doc-aligned repair-axis search`')
    lines.append(f'- Base candidate: `{ROUND61_BASE_NAME}`')
    lines.append('- Noise regime: `shared noise0p08` from Feishu doc Q4G6d0ZU...')
    lines.append('- Noise config:')
    cfg = summary['dataset']['noise_config']
    lines.append(f"  - arw = `{cfg['arw_dpsh']} * dpsh`")
    lines.append(f"  - vrw = `{cfg['vrw_ugpsHz']} * ugpsHz`")
    lines.append(f"  - bi_g = `{cfg['bi_g_dph']} * dph`")
    lines.append(f"  - bi_a = `{cfg['bi_a_ug']} * ug`")
    lines.append(f"  - tau_g = tau_a = `{cfg['tau_g']}`")
    lines.append(f"  - seed = `{cfg['seed']}`")
    lines.append(f"  - base_family = `{cfg.get('base_family', 'round53_round61_shared')}`")
    lines.append('')
    lines.append('## B. 目标')
    lines.append('- 用户明确纠正：本轮不是 1x mainline 噪声，而是文档里的 `shared noise0p08`。')
    lines.append('- 因此重新在 doc 噪声下搜索 split-point 全局最优。')
    lines.append('- split-point 采用 Round65-C 严格定义：`gap = feedback_floor - scd_floor`，平均 floor 固定。')
    lines.append('')
    lines.append('## C. Best result')
    best = summary['strongest_signal']
    lines.append(f"- best candidate: `{best['name']}`")
    lines.append(f"- best gap: `{best['gap']:+.2f}`")
    lines.append(f"- floors: feedback=`{best['feedback_floor']:.3f}`, scd=`{best['scd_floor']:.3f}`")
    lines.append(f"- signal: {best['signal']}")
    lines.append(f"- conclusion: {summary['conclusion_line']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    round61_payload = json.loads(ROUND61_REF_JSON.read_text(encoding='utf-8'))
    markov_payload = json.loads(MARKOV_REF_JSON.read_text(encoding='utf-8'))
    pure_scd_payload = json.loads(PURE_SCD_REF_JSON.read_text(encoding='utf-8'))
    kf36_payload = json.loads(KF36_REF_JSON.read_text(encoding='utf-8'))
    four_methods_payload = json.loads(FOUR_METHODS_JSON.read_text(encoding='utf-8')) if FOUR_METHODS_JSON.exists() else None

    source_mod = base.load_module('markov_pruned_source_round65c_shared_noise0p08', str(SOURCE_FILE))
    dataset = build_shared_dataset(source_mod, NOISE_SCALE)

    candidate_dump = {
        'round_name': 'Round65C_SplitPoint_Search_Shared_Noise0p08',
        'round_type': 'doc-aligned repair-axis search',
        'innovation_direction': 'dual-channel split-point search on feedback-vs-SCD axis under shared noise0p08',
        'base_round61_candidate': ROUND61_BASE_NAME,
        'same_dataset_round61_json': str(ROUND61_REF_JSON),
        'noise_reference_doc': 'https://ecn6ivbzhayi.feishu.cn/docx/Q4G6d0ZUWo7w3MxIBzTcDAv8n8e',
        'dataset': {
            'source_file': str(SOURCE_FILE),
            'source_trajectory_reference': 'round53_round61_shared',
            'noise_config': dataset['noise_config'],
            'constraint_note': 'Use exactly the same shared noise0p08 dataset/noise/seed as the four-method Feishu doc.',
        },
        'ladder_references': {
            'kf36_json': str(KF36_REF_JSON),
            'markov_json': str(MARKOV_REF_JSON),
            'pure_scd_json': str(PURE_SCD_REF_JSON),
            'round61_json': str(ROUND61_REF_JSON),
            'compact_compare_json': str(FOUR_METHODS_JSON),
        },
        'search_axis': {
            'definition': 'gap = feedback_floor - scd_floor',
            'avg_floor_fixed': AVG_FLOOR,
            'gap_values': GAP_VALUES,
        },
        'protected_metrics': base.HARD_PROTECTED_KEYS,
        'repair_metrics': base.PRIMARY_REPAIR_KEYS,
        'round65c_candidates': ROUND65C_CANDIDATES,
        'four_methods_payload': four_methods_payload,
    }
    CANDIDATE_JSON.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    out = {
        'round_name': 'Round65C_SplitPoint_Search_Shared_Noise0p08',
        'round_type': 'doc-aligned repair-axis search',
        'dataset': {
            'source_file': str(SOURCE_FILE),
            'noise_config': dataset['noise_config'],
            'seed': dataset['noise_config']['seed'],
        },
        'base_round61_candidate': ROUND61_BASE_NAME,
        'base_round61_json': str(ROUND61_REF_JSON),
        'doc_noise_reference': 'https://ecn6ivbzhayi.feishu.cn/docx/Q4G6d0ZUWo7w3MxIBzTcDAv8n8e',
        'ladder_references': {
            'kf36_json': str(KF36_REF_JSON),
            'markov_json': str(MARKOV_REF_JSON),
            'pure_scd_json': str(PURE_SCD_REF_JSON),
            'round61_json': str(ROUND61_REF_JSON),
        },
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

        method_mod = base.load_module(f'markov_method_round65c_shared_noise0p08_candidate_{idx}', str(R53_METHOD_FILE))
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
            label=f'R65C-SH08-SPLIT-{idx}',
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
            variant=f'r65c_splitpoint_{NOISE_TAG}_{candidate["name"]}',
            method_file='probe_round65c_splitpoint_search_shared_noise0p08::dual_channel_splitpoint_axis',
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

        candidate_json_path = RESULTS_DIR / f'R65C_splitpoint_{NOISE_TAG}_{candidate["name"]}_param_errors.json'
        candidate_json_path.write_text(json.dumps(payload_candidate, ensure_ascii=False, indent=2), encoding='utf-8')

        delta_vs_r61 = {
            **base._delta_block(payload_candidate['focus_scale_pct'], round61_payload['focus_scale_pct']),
            **base._delta_block(payload_candidate['lever_guard_pct'], round61_payload['lever_guard_pct']),
            **base._delta_block(payload_candidate['overall'], round61_payload['overall']),
        }
        score, penalties = _score_candidate_vs_round61(delta_vs_r61)
        note = _selection_note(delta_vs_r61, penalties)

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
            'vs_markov_relative_improvement': base._relative_improvement_block(
                markov_payload,
                payload_candidate,
                ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'mean_pct_error', 'median_pct_error', 'max_pct_error'],
            ),
            'vs_pure_scd_relative_improvement': base._relative_improvement_block(
                pure_scd_payload,
                payload_candidate,
                ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z', 'mean_pct_error', 'median_pct_error', 'max_pct_error'],
            ),
            'vs_kf36_relative_improvement': base._relative_improvement_block(
                kf36_payload,
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
            'reason': 'Clean shared-noise0p08 winner over Round61 on the 1D split-point axis.',
        }
        out['result_classification'] = 'clean win'
        out['conclusion_line'] = 'Under shared noise0p08, the split-point axis produced a clean promotable winner over Round61.'
    else:
        out['winner'] = None
        out['no_winner_reason'] = 'No split point passed the Round61 clean-win gate under shared noise0p08.'
        out['result_classification'] = 'partial signal' if any(float(best_delta_r61[k]) < 0.0 for k in base.PRIMARY_REPAIR_KEYS) else 'no useful signal'
        out['conclusion_line'] = 'Under shared noise0p08, the sweep found the best split point on this axis, but it still does not cleanly beat Round61.'

    if best['gap'] < -1e-9:
        side = '偏 SCD 一侧（feedback 更收、SCD 更放）'
    elif best['gap'] > 1e-9:
        side = '偏 feedback 一侧（feedback 更放、SCD 更收）'
    else:
        side = '接近对称分离点'

    out['strongest_signal'] = {
        'name': best_name,
        'gap': float(best['gap']),
        'feedback_floor': float(best['feedback_floor']),
        'scd_floor': float(best['scd_floor']),
        'signal': (
            f"best split candidate {best_name}: gap={best['gap']:+.2f}, fb={best['feedback_floor']:.3f}, scd={best['scd_floor']:.3f}; "
            f"dKg_xy Δ={best_delta_r61['dKg_xy']:.6f}, dKg_yy Δ={best_delta_r61['dKg_yy']:.6f}, "
            f"dKa_xx Δ={best_delta_r61['dKa_xx']:.6f}, rx_y Δ={best_delta_r61['rx_y']:.6f}, "
            f"dKg_xx Δ={best_delta_r61['dKg_xx']:.6f}, dKg_zz Δ={best_delta_r61['dKg_zz']:.6f}, "
            f"mean Δ={best_delta_r61['mean_pct_error']:.6f}, max Δ={best_delta_r61['max_pct_error']:.6f}"
        ),
        'regressions': str(best_penalties),
    }
    out['interpretation'] = (
        f"在 fixed shared noise0p08 + Round65-C 1D split-point 定义下，当前最优 split point 落在 **{side}**；"
        f"也就是说，这组噪声下最好的 gap 是 {best['gap']:+.2f}，而不是默认沿用 1x 噪声的结论。"
    )
    out['next_best_move'] = (
        f"Lock the best gap `{best['gap']:+.2f}` as center, then run one ultra-narrow local refinement on ±0.03 / ±0.06 only under shared noise0p08, "
        'or conclude that this axis has saturated if the best point still clearly trails Round61.'
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
        'winner': out['winner'],
        'result_classification': out['result_classification'],
        'best_candidate': out['strongest_signal'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
