from __future__ import annotations

import argparse
import json
import sys
import types
from datetime import datetime
from pathlib import Path

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
TMP_PSINS_DIR = ROOT / 'tmp_psins_py'
SOURCE_FILE = TMP_PSINS_DIR / 'psins_py' / 'test_calibration_markov_pruned.py'
COMPARE_19_20_FILE = SCRIPTS_DIR / 'compare_ch3_corrected_symmetric20_vs_legacy19pos_1200s.py'
COMPUTE_R61_FILE = SCRIPTS_DIR / 'compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors.py'
COMPARE_SHARED_FILE = SCRIPTS_DIR / 'compare_four_methods_shared_noise.py'
PROBE_R55_FILE = SCRIPTS_DIR / 'probe_round55_newline.py'
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'
R61_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round61_h_scd_state20_microtight_commit.py'

for p in [ROOT, TMP_PSINS_DIR, METHOD_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module

GROUP_ORDER = ['g1_kf19', 'g2_markov19', 'g3_markov20', 'g4_round61_20']
GROUP_DISPLAY = {
    'g1_kf19': 'G1 普通模型 @19位置',
    'g2_markov19': 'G2 Markov @19位置',
    'g3_markov20': 'G3 Markov @20位置',
    'g4_round61_20': 'G4 Markov+LLM+SCD @20位置',
}
FOCUS_PARAMS = ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z']
ROUND61_SYM20_PREFIX = 'R61_42state_gm1_round61_h_scd_state20_microtight_commit'
COMPARISON_MODE = 'four_group_progression_19pos_markov_20pos_round61'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=0.12)
    parser.add_argument('--report-date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def _fmt_pct(x: float) -> str:
    return f'{x:.6f}'


def _fmt_e(x: float) -> str:
    return f'{x:.6e}'


def round61_sym20_output_path(case_tag: str, suffix: str) -> Path:
    return RESULTS_DIR / f'{ROUND61_SYM20_PREFIX}_{case_tag}_shared_{suffix}_param_errors.json'


def load_modules(noise_scale: float):
    suffix = str(noise_scale).replace('.', 'p')
    source_mod = load_module(f'progression_source_{suffix}', str(SOURCE_FILE))
    compare_mod = load_module(f'progression_compare_19_20_{suffix}', str(COMPARE_19_20_FILE))
    shared_mod = load_module(f'progression_shared_{suffix}', str(COMPARE_SHARED_FILE))
    compute_r61_mod = load_module(f'progression_compute_r61_{suffix}', str(COMPUTE_R61_FILE))
    probe_r55_mod = load_module(f'progression_probe_r55_{suffix}', str(PROBE_R55_FILE))
    r53_mod = load_module(f'progression_r53_{suffix}', str(R53_METHOD_FILE))
    r61_mod = load_module(f'progression_r61_{suffix}', str(R61_METHOD_FILE))
    return source_mod, compare_mod, shared_mod, compute_r61_mod, probe_r55_mod, r53_mod, r61_mod


def compact_payload(payload: dict) -> dict:
    overall = payload['overall']
    return {
        'overall': {
            'mean_pct_error': float(overall['mean_pct_error']),
            'median_pct_error': float(overall['median_pct_error']),
            'max_pct_error': float(overall['max_pct_error']),
        },
        'focus_param_pct': {
            name: float(payload['param_errors'][name]['pct_error'])
            for name in FOCUS_PARAMS
        },
    }


def build_group_rows(payloads: dict[str, dict]) -> list[dict]:
    rows = []
    for group_key in GROUP_ORDER:
        payload = payloads[group_key]
        overall = payload['overall']
        rows.append({
            'group_key': group_key,
            'display': GROUP_DISPLAY[group_key],
            'mean_pct_error': float(overall['mean_pct_error']),
            'median_pct_error': float(overall['median_pct_error']),
            'max_pct_error': float(overall['max_pct_error']),
        })
    return rows


def build_progression_metrics(group_rows: list[dict]) -> dict:
    metrics = {}
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        values = [float(row[metric]) for row in group_rows]
        steps = []
        for i in range(1, len(group_rows)):
            prev_row = group_rows[i - 1]
            curr_row = group_rows[i]
            delta = float(prev_row[metric] - curr_row[metric])
            steps.append({
                'from': prev_row['group_key'],
                'to': curr_row['group_key'],
                'from_value': float(prev_row[metric]),
                'to_value': float(curr_row[metric]),
                'improvement_pct_points': delta,
                'improved': bool(delta > 0),
            })
        metrics[metric] = {
            'strict_progression': all(step['improved'] for step in steps),
            'steps': steps,
            'best_group': min(group_rows, key=lambda x: float(x[metric]))['group_key'],
        }
    return metrics


def build_all_param_table(payloads: dict[str, dict]) -> dict:
    param_order = payloads['g1_kf19']['param_order']
    out = {}
    for name in param_order:
        out[name] = {
            'true': float(payloads['g1_kf19']['param_errors'][name]['true']),
            'groups': {},
        }
        ranking = []
        for group_key in GROUP_ORDER:
            pct = float(payloads[group_key]['param_errors'][name]['pct_error'])
            est = float(payloads[group_key]['param_errors'][name]['est'])
            out[name]['groups'][group_key] = {
                'est': est,
                'pct_error': pct,
            }
            ranking.append({'group_key': group_key, 'pct_error': pct})
        ranking.sort(key=lambda x: x['pct_error'])
        out[name]['ranking'] = ranking
    return {'param_order': param_order, 'table': out}


def render_report(summary: dict) -> str:
    lines: list[str] = []
    lines.append('# 19位置 / 20位置 四组标定精度递进对照')
    lines.append('')
    lines.append('## 1. 对照口径')
    lines.append('')
    lines.append(f"- noise_scale = **{summary['noise_scale']}**")
    cfg = summary['noise_config']
    lines.append(f"- shared noise family = `{cfg['base_family']}`")
    lines.append(f"- arw = {cfg['arw_dpsh']} dpsh, vrw = {cfg['vrw_ugpsHz']} ugpsHz")
    lines.append(f"- bi_g = {cfg['bi_g_dph']} dph, bi_a = {cfg['bi_a_ug']} ug, tau = {cfg['tau_g']}")
    lines.append('- 前 3 组使用已验证的 19pos / corrected-symmetric20 正式链路；第 4 组是在同一 **20-position corrected symmetric path** 上运行现有 **Round61 (Markov + LLM-guided constrained patch + SCD)** 方法。')
    lines.append('- **注意**：第 4 组是把现有 Round61 方法迁移到 20-position 路径上验证，不是重新为 20-position 单独再做一轮新搜索。')
    lines.append('')

    lines.append('## 2. 四组定义')
    lines.append('')
    lines.append('| 组别 | 定义 | 运行文件 |')
    lines.append('|---|---|---|')
    for group_key in GROUP_ORDER:
        info = summary['groups'][group_key]
        lines.append(f"| {GROUP_DISPLAY[group_key]} | {info['definition']} | `{info['json_path']}` |")
    lines.append('')

    lines.append('## 3. Overall 指标递进（越低越好）')
    lines.append('')
    lines.append('| 组别 | mean% | median% | max% | Δmean vs prev | Δmedian vs prev | Δmax vs prev |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    prev = None
    for row in summary['group_rows']:
        if prev is None:
            d_mean = d_median = d_max = '—'
        else:
            d_mean = f"{prev['mean_pct_error'] - row['mean_pct_error']:+.6f}"
            d_median = f"{prev['median_pct_error'] - row['median_pct_error']:+.6f}"
            d_max = f"{prev['max_pct_error'] - row['max_pct_error']:+.6f}"
        lines.append(
            f"| {row['display']} | {_fmt_pct(row['mean_pct_error'])} | {_fmt_pct(row['median_pct_error'])} | {_fmt_pct(row['max_pct_error'])} | {d_mean} | {d_median} | {d_max} |"
        )
        prev = row
    lines.append('')

    lines.append('## 4. 递进判断')
    lines.append('')
    for metric, label in [
        ('mean_pct_error', 'mean'),
        ('median_pct_error', 'median'),
        ('max_pct_error', 'max'),
    ]:
        pm = summary['progression_metrics'][metric]
        verdict = '严格递进' if pm['strict_progression'] else '不是严格递进'
        best_group = GROUP_DISPLAY[pm['best_group']]
        lines.append(f"- **{label}**：{verdict}；最佳组别是 **{best_group}**")
        for step in pm['steps']:
            step_text = '改善' if step['improved'] else '退化'
            lines.append(
                f"  - {GROUP_DISPLAY[step['from']]} → {GROUP_DISPLAY[step['to']]}：{step_text} {step['improvement_pct_points']:+.6f}"
            )
    lines.append('')

    lines.append('## 5. 关键参数对照（越低越好）')
    lines.append('')
    lines.append('| 参数 | G1 KF19 | G2 Markov19 | G3 Markov20 | G4 Round61@20 | best |')
    lines.append('|---|---:|---:|---:|---:|---|')
    for name in FOCUS_PARAMS:
        item = summary['all_params']['table'][name]
        best_group = GROUP_DISPLAY[item['ranking'][0]['group_key']]
        lines.append(
            f"| {name} | {_fmt_pct(item['groups']['g1_kf19']['pct_error'])} | {_fmt_pct(item['groups']['g2_markov19']['pct_error'])} | {_fmt_pct(item['groups']['g3_markov20']['pct_error'])} | {_fmt_pct(item['groups']['g4_round61_20']['pct_error'])} | {best_group} |"
        )
    lines.append('')

    lines.append('## 6. 结论')
    lines.append('')
    lines.append(f"- 主结论：{summary['headline']}" )
    lines.append(f"- 解释：{summary['interpretation']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def build_headline(summary: dict) -> str:
    mean_prog = summary['progression_metrics']['mean_pct_error']['strict_progression']
    median_prog = summary['progression_metrics']['median_pct_error']['strict_progression']
    max_prog = summary['progression_metrics']['max_pct_error']['strict_progression']
    if mean_prog and median_prog and max_prog:
        return '四组在 mean / median / max 三个 overall 指标上都呈现严格递进改善。'
    if mean_prog:
        return '四组在 **mean 指标** 上形成了清晰递进；但 median / max 不一定同时严格单调。'
    return '这四组**不构成完全严格单调递进**；需要按 mean / median / max 分开看，并结合关键参数判断。'


def build_interpretation(summary: dict) -> str:
    best_mean = GROUP_DISPLAY[summary['progression_metrics']['mean_pct_error']['best_group']]
    best_median = GROUP_DISPLAY[summary['progression_metrics']['median_pct_error']['best_group']]
    best_max = GROUP_DISPLAY[summary['progression_metrics']['max_pct_error']['best_group']]
    return (
        f"当前 best-mean 是 {best_mean}，best-median 是 {best_median}，best-max 是 {best_max}。"
        ' 如果 G4 没有全面压过 G3，这更像是“现有 Round61 方法迁移到 20-position 路径后的 transfer test”，'
        ' 说明是否还需要针对 20-position 再做一轮专门调参/重搜索。'
    )


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    source_mod, compare_mod, shared_mod, compute_r61_mod, probe_r55_mod, r53_mod, r61_mod = load_modules(args.noise_scale)
    expected_cfg = shared_mod.expected_noise_config(args.noise_scale)
    suffix = shared_mod.make_suffix(args.noise_scale)

    case19 = compare_mod.build_legacy19pos_case(source_mod)
    case20 = compare_mod.build_symmetric20_case(source_mod)

    payloads: dict[str, dict] = {}
    execution: dict[str, str] = {}
    json_paths: dict[str, str] = {}

    # G1/G2/G3 reuse or rerun through the formal compare script helpers.
    for group_key, case, method_key in [
        ('g1_kf19', case19, 'kf36_noisy'),
        ('g2_markov19', case19, 'markov42_noisy'),
        ('g3_markov20', case20, 'markov42_noisy'),
    ]:
        payload, status, out_path = compare_mod.run_case_method(
            source_mod,
            case,
            method_key,
            args.noise_scale,
            force_rerun=args.force_rerun,
        )
        payloads[group_key] = payload
        execution[group_key] = status
        json_paths[group_key] = str(out_path)

    # G4: Round61 hybrid transferred onto the same corrected symmetric20 path.
    g4_path = round61_sym20_output_path(compare_mod.CASE_TAGS['symmetric20'], suffix)
    if (not args.force_rerun) and g4_path.exists():
        p = shared_mod._load_json(g4_path)
        extra = p.get('extra', {}) if isinstance(p, dict) else {}
        if (
            shared_mod._noise_matches(p, expected_cfg)
            and extra.get('comparison_mode') == COMPARISON_MODE
            and extra.get('case_key') == 'symmetric20'
            and extra.get('group_key') == 'g4_round61_20'
        ):
            payloads['g4_round61_20'] = p
            execution['g4_round61_20'] = 'reused_verified'
            json_paths['g4_round61_20'] = str(g4_path)

    if 'g4_round61_20' not in payloads:
        dataset = compare_mod.build_dataset(source_mod, case20['paras'], case20['att0_deg'], args.noise_scale)
        params = compute_r61_mod._param_specs(source_mod)

        candidate = r61_mod._pick_candidate()
        merged_candidate = r61_mod._merge_round61_candidate(candidate)
        patched_method = probe_r55_mod._build_patched_method(r53_mod, merged_candidate)
        result = list(r61_mod._run_internalized_hybrid_scd(
            patched_method,
            source_mod,
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            bi_g=dataset['bi_g'],
            bi_a=dataset['bi_a'],
            tau_g=dataset['tau_g'],
            tau_a=dataset['tau_a'],
            label=f'R61-SYM20-{suffix.upper()}',
            scd_cfg=merged_candidate['scd'],
        ))
        extra = result[4] if len(result) >= 5 and isinstance(result[4], dict) else {}
        extra = dict(extra)
        extra.update({
            'comparison_mode': COMPARISON_MODE,
            'group_key': 'g4_round61_20',
            'case_key': 'symmetric20',
            'case_tag': compare_mod.CASE_TAGS['symmetric20'],
            'case_display_name': compare_mod.CASE_LABELS['symmetric20'],
            'att0_deg': case20['att0_deg'],
            'n_motion_rows': case20['n_motion_rows'],
            'claimed_position_count': case20['claimed_position_count'],
            'total_time_s': case20['total_time_s'],
            'timing_note': case20['timing_note'],
            'source_builder': case20['source_builder'],
            'source_reference': case20['source_reference'],
            'noise_scale': args.noise_scale,
            'noise_config': dataset['noise_config'],
            'round61_selected_candidate': candidate['name'],
            'round61_candidate_description': candidate['description'],
            'round61_candidate_rationale': candidate['rationale'],
            'transfer_note': 'Round61 hybrid method transferred onto corrected symmetric20 path without a new 20pos-specific search.',
        })
        payload_g4 = shared_mod.compute_payload(
            source_mod,
            result[0],
            params,
            variant=f'42state_gm1_round61_h_scd_state20_microtight_commit_{compare_mod.CASE_TAGS["symmetric20"]}_{suffix}',
            method_file=f'{R61_METHOD_FILE} on corrected symmetric20 path',
            extra=extra,
        )
        g4_path.write_text(json.dumps(payload_g4, ensure_ascii=False, indent=2), encoding='utf-8')
        payloads['g4_round61_20'] = payload_g4
        execution['g4_round61_20'] = 'rerun'
        json_paths['g4_round61_20'] = str(g4_path)

    group_rows = build_group_rows(payloads)
    progression_metrics = build_progression_metrics(group_rows)
    all_params = build_all_param_table(payloads)

    summary = {
        'experiment': 'four_group_progression_19_20',
        'comparison_mode': COMPARISON_MODE,
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'noise_config': expected_cfg,
        'execution': execution,
        'group_rows': group_rows,
        'progression_metrics': progression_metrics,
        'all_params': all_params,
        'groups': {
            'g1_kf19': {
                'definition': 'KF36 baseline on legacy 19-position path (own historical att0, normalized to 1200 s)',
                'json_path': json_paths['g1_kf19'],
                'compact': compact_payload(payloads['g1_kf19']),
            },
            'g2_markov19': {
                'definition': 'Markov42 baseline on legacy 19-position path (own historical att0, normalized to 1200 s)',
                'json_path': json_paths['g2_markov19'],
                'compact': compact_payload(payloads['g2_markov19']),
            },
            'g3_markov20': {
                'definition': 'Markov42 baseline on corrected symmetric 20-position path (att0=(0,0,0), 1200 s)',
                'json_path': json_paths['g3_markov20'],
                'compact': compact_payload(payloads['g3_markov20']),
            },
            'g4_round61_20': {
                'definition': 'Round61 hybrid (Markov + LLM-guided constrained patch + SCD) transferred onto corrected symmetric 20-position path',
                'json_path': json_paths['g4_round61_20'],
                'compact': compact_payload(payloads['g4_round61_20']),
            },
        },
    }
    summary['headline'] = build_headline(summary)
    summary['interpretation'] = build_interpretation(summary)

    compare_json = RESULTS_DIR / f'compare_four_group_progression_19_20_{suffix}.json'
    report_md = REPORTS_DIR / f'psins_four_group_progression_19_20_{args.report_date}_{suffix}.md'
    summary['files'] = {
        'compare_json': str(compare_json),
        'report_md': str(report_md),
        'run_jsons': json_paths,
    }

    compare_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_md.write_text(render_report(summary), encoding='utf-8')

    result = {
        'compare_json': str(compare_json),
        'report_md': str(report_md),
        'execution': execution,
        'headline': summary['headline'],
        'group_rows': group_rows,
        'progression_metrics': progression_metrics,
        'run_jsons': json_paths,
    }
    print('__RESULT_JSON__=' + json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
