from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path('/root/.openclaw/workspace')
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'
REPORTS_DIR = ROOT / 'reports'
COMPARE_SCRIPT = SCRIPTS_DIR / 'compare_four_methods_shared_noise.py'

METHOD_ORDER = ['kf36_noisy', 'markov42_noisy', 'scd42_neutral', 'round61']
METHOD_DISPLAY = {
    'kf36_noisy': 'KF36',
    'markov42_noisy': 'Markov42',
    'scd42_neutral': 'Pure SCD',
    'round61': 'Round61',
}
OVERALL_KEYS = ['mean_pct_error', 'median_pct_error', 'max_pct_error']
OVERALL_DISPLAY = {
    'mean_pct_error': 'overall mean',
    'median_pct_error': 'overall median',
    'max_pct_error': 'overall max',
}

SCALE_SPECS = [
    {'value': 0.03, 'label': '0.03x', 'display': '0.03'},
    {'value': 0.05, 'label': '0.05x', 'display': '0.05'},
    {'value': 0.08, 'label': '0.08x', 'display': '0.08'},
    {'value': 0.10, 'label': '0.10x', 'display': '0.10'},
    {'value': 0.15, 'label': '0.15x', 'display': '0.15'},
    {'value': 0.25, 'label': '0.25x', 'display': '0.25'},
    {'value': 1.0 / 3.0, 'label': '0.3333333333x', 'display': '0.3333333333'},
    {'value': 0.50, 'label': '0.50x', 'display': '0.50'},
    {'value': 0.75, 'label': '0.75x', 'display': '0.75'},
    {'value': 1.00, 'label': '1.00x', 'display': '1.00'},
    {'value': 1.50, 'label': '1.50x', 'display': '1.50'},
    {'value': 2.00, 'label': '2.00x', 'display': '2.00'},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-date', default=datetime.now().strftime('%Y-%m-%d'))
    return parser.parse_args()


def make_suffix(noise_scale: float) -> str:
    mapping = {
        1.0: 'noise1x',
        1.0 / 3.0: 'noise1over3',
        0.2: 'noise1over5',
        0.5: 'noise0p5',
        2.0: 'noise2p0',
    }
    for key, value in mapping.items():
        if abs(noise_scale - key) < 1e-12:
            return value
    return f"noise{str(noise_scale).replace('.', 'p')}"


def compare_json_path(scale: float) -> Path:
    return RESULTS_DIR / f'compare_four_methods_shared_{make_suffix(scale)}.json'


def compact_json_path(scale: float) -> Path:
    return RESULTS_DIR / f'compare_four_methods_shared_{make_suffix(scale)}_compact.json'


def summary_json_path(report_date: str) -> Path:
    return RESULTS_DIR / f'compare_four_methods_shared_noise_dense_curve_{report_date}.json'


def report_md_path(report_date: str) -> Path:
    return REPORTS_DIR / f'psins_four_methods_shared_noise_dense_curve_{report_date}.md'


def fmt_scale_arg(value: float) -> str:
    return format(value, '.17g')


def fmt_pct(value: float | None) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return 'NA'
    return f'{value:.3f}'


def fmt_sec(value: float) -> str:
    return f'{value:.1f}s'


def sparkline(values: list[float]) -> str:
    bars = '▁▂▃▄▅▆▇█'
    if not values:
        return ''
    lo = min(values)
    hi = max(values)
    if math.isclose(lo, hi):
        return bars[len(bars) // 2] * len(values)
    out = []
    for v in values:
        idx = round((v - lo) / (hi - lo) * (len(bars) - 1))
        out.append(bars[idx])
    return ''.join(out)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def run_scale(scale_spec: dict, report_date: str) -> dict:
    cmd = [
        sys.executable,
        str(COMPARE_SCRIPT),
        '--noise-scale',
        fmt_scale_arg(scale_spec['value']),
        '--report-date',
        report_date,
    ]
    start = time.time()
    result_json = None
    tail_lines: list[str] = []

    print(f"=== RUN {scale_spec['label']} ({make_suffix(scale_spec['value'])}) ===", flush=True)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        tail_lines.append(line.rstrip('\n'))
        if len(tail_lines) > 200:
            tail_lines.pop(0)
        if line.startswith('__RESULT_JSON__='):
            result_json = json.loads(line.split('=', 1)[1])
    return_code = proc.wait()
    runtime_seconds = time.time() - start
    print(f"=== DONE {scale_spec['label']} in {fmt_sec(runtime_seconds)} ===", flush=True)

    if return_code != 0:
        tail = '\n'.join(tail_lines[-40:])
        raise RuntimeError(
            f'compare script failed for {scale_spec["label"]} with code {return_code}\n'
            f'Command: {cmd}\n'
            f'--- tail ---\n{tail}'
        )

    if result_json is None:
        expected_path = compare_json_path(scale_spec['value'])
        if expected_path.exists():
            result_json = {
                'compare_json': str(expected_path),
                'compact_json': str(compact_json_path(scale_spec['value'])),
            }
        else:
            raise RuntimeError(f'No __RESULT_JSON__ captured and compare file missing for {scale_spec["label"]}')

    compare_payload = read_json(Path(result_json['compare_json']))
    compact_payload = read_json(Path(result_json['compact_json'])) if Path(result_json['compact_json']).exists() else {}
    return {
        'scale_spec': scale_spec,
        'runtime_seconds': runtime_seconds,
        'result_json': result_json,
        'compare': compare_payload,
        'compact': compact_payload,
    }


def compute_round61_vs_kf36(compare: dict) -> dict:
    overall = {}
    for metric in OVERALL_KEYS:
        kf = float(compare['overall']['by_method']['kf36_noisy'][metric])
        r61 = float(compare['overall']['by_method']['round61'][metric])
        delta = kf - r61
        overall[metric] = {
            'kf36_pct_error': kf,
            'round61_pct_error': r61,
            'delta_pct_points': delta,
            'relative_improvement_pct': (delta / kf * 100.0) if abs(kf) > 1e-15 else None,
            'round61_trails_kf36': bool(delta < 0.0),
        }

    param_deltas = {}
    regressions = []
    improvements = []
    for name in compare['param_order']:
        kf = float(compare['all_params'][name]['methods']['kf36_noisy']['pct_error'])
        r61 = float(compare['all_params'][name]['methods']['round61']['pct_error'])
        delta = kf - r61
        item = {
            'kf36_pct_error': kf,
            'round61_pct_error': r61,
            'delta_pct_points': delta,
            'relative_improvement_pct': (delta / kf * 100.0) if abs(kf) > 1e-15 else None,
            'round61_trails_kf36': bool(delta < 0.0),
        }
        param_deltas[name] = item
        summary_item = {'param': name, **item}
        if delta < 0.0:
            regressions.append(summary_item)
        else:
            improvements.append(summary_item)

    regressions.sort(key=lambda x: x['delta_pct_points'])
    improvements.sort(key=lambda x: x['delta_pct_points'], reverse=True)
    return {
        'overall': overall,
        'param_deltas': param_deltas,
        'regressions': regressions,
        'improvements': improvements,
        'regression_count': len(regressions),
        'improvement_count': len(improvements),
    }


def build_summary(run_records: list[dict], report_date: str) -> dict:
    per_scale = []
    best_counts = {metric: Counter() for metric in OVERALL_KEYS}
    execution_counts_by_method: dict[str, Counter] = {method: Counter() for method in METHOD_ORDER}
    execution_log = []
    param_regression_scales: dict[str, list[str]] = defaultdict(list)
    per_scale_regression_counts = []

    curve_overall_by_method = {
        method: {metric: [] for metric in OVERALL_KEYS}
        for method in METHOD_ORDER
    }
    curve_round61_vs_kf36 = {
        metric: {
            'relative_improvement_pct': [],
            'delta_pct_points': [],
            'trailing_scales': [],
        }
        for metric in OVERALL_KEYS
    }

    for record in run_records:
        scale_spec = record['scale_spec']
        compare = record['compare']
        round61_vs_kf36 = compute_round61_vs_kf36(compare)
        suffix = make_suffix(scale_spec['value'])
        compact = record['compact']
        execution = compact.get('execution', compare.get('execution', {}))

        for method in METHOD_ORDER:
            for metric in OVERALL_KEYS:
                curve_overall_by_method[method][metric].append(float(compare['overall']['by_method'][method][metric]))
            execution_counts_by_method[method][execution.get(method, 'unknown')] += 1

        for metric in OVERALL_KEYS:
            best_method = compare['overall']['best_by_metric'][metric]['method']
            best_counts[metric][best_method] += 1
            curve_round61_vs_kf36[metric]['relative_improvement_pct'].append(
                round61_vs_kf36['overall'][metric]['relative_improvement_pct']
            )
            curve_round61_vs_kf36[metric]['delta_pct_points'].append(
                round61_vs_kf36['overall'][metric]['delta_pct_points']
            )
            if round61_vs_kf36['overall'][metric]['round61_trails_kf36']:
                curve_round61_vs_kf36[metric]['trailing_scales'].append(scale_spec['display'])

        for item in round61_vs_kf36['regressions']:
            param_regression_scales[item['param']].append(scale_spec['display'])

        per_scale_regression_counts.append({
            'scale': scale_spec['display'],
            'count': round61_vs_kf36['regression_count'],
        })

        execution_log.append({
            'scale': scale_spec['display'],
            'suffix': suffix,
            'runtime_seconds': record['runtime_seconds'],
            'execution': execution,
            'compare_json': compare['methods']['round61']['json_path'] if False else str(compare_json_path(scale_spec['value'])),
        })

        per_scale.append({
            'scale': scale_spec['display'],
            'scale_label': scale_spec['label'],
            'scale_value': scale_spec['value'],
            'suffix': suffix,
            'noise_config': compare['noise_config'],
            'runtime_seconds': record['runtime_seconds'],
            'files': compact.get('files', {
                'compare_json': str(compare_json_path(scale_spec['value'])),
                'compact_json': str(compact_json_path(scale_spec['value'])),
            }),
            'execution': execution,
            'best_by_metric': compare['overall']['best_by_metric'],
            'round61_rank': compare['overall']['round61_rank'],
            'overall_by_method': compare['overall']['by_method'],
            'best_param_count': compare['overall'].get('best_param_count', {}),
            'round61_vs_kf36': round61_vs_kf36,
        })

    param_regression_frequency = []
    for param, scales in param_regression_scales.items():
        param_regression_frequency.append({
            'param': param,
            'count': len(scales),
            'scales': scales,
        })
    param_regression_frequency.sort(key=lambda x: (-x['count'], x['param']))

    worst_overall_by_metric = {}
    best_overall_by_metric = {}
    for metric in OVERALL_KEYS:
        points = [
            {
                'scale': scale['scale'],
                'relative_improvement_pct': scale['round61_vs_kf36']['overall'][metric]['relative_improvement_pct'],
                'delta_pct_points': scale['round61_vs_kf36']['overall'][metric]['delta_pct_points'],
                'kf36_pct_error': scale['round61_vs_kf36']['overall'][metric]['kf36_pct_error'],
                'round61_pct_error': scale['round61_vs_kf36']['overall'][metric]['round61_pct_error'],
            }
            for scale in per_scale
        ]
        best_overall_by_metric[metric] = max(points, key=lambda x: x['relative_improvement_pct'])
        worst_overall_by_metric[metric] = min(points, key=lambda x: x['relative_improvement_pct'])

    all_regressions = []
    for scale in per_scale:
        for item in scale['round61_vs_kf36']['regressions']:
            all_regressions.append({
                'scale': scale['scale'],
                'param': item['param'],
                'delta_pct_points': item['delta_pct_points'],
                'relative_improvement_pct': item['relative_improvement_pct'],
                'kf36_pct_error': item['kf36_pct_error'],
                'round61_pct_error': item['round61_pct_error'],
            })
    all_regressions.sort(key=lambda x: x['delta_pct_points'])
    worst_param_regression = all_regressions[0] if all_regressions else None

    summary = {
        'report_date': report_date,
        'scales_requested': [spec['display'] for spec in SCALE_SPECS],
        'scale_labels': [spec['label'] for spec in SCALE_SPECS],
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'report_md': str(report_md_path(report_date)),
        'per_scale': per_scale,
        'curve_overall_by_method': curve_overall_by_method,
        'curve_round61_vs_kf36': curve_round61_vs_kf36,
        'best_method_counts': {
            metric: dict(best_counts[metric])
            for metric in OVERALL_KEYS
        },
        'execution_counts_by_method': {
            method: dict(counter)
            for method, counter in execution_counts_by_method.items()
        },
        'execution_log': execution_log,
        'diagnostics': {
            'overall_metrics_where_round61_trails_kf36': {
                metric: curve_round61_vs_kf36[metric]['trailing_scales']
                for metric in OVERALL_KEYS
            },
            'best_overall_by_metric': best_overall_by_metric,
            'worst_overall_by_metric': worst_overall_by_metric,
            'per_scale_regression_counts': per_scale_regression_counts,
            'param_regression_frequency': param_regression_frequency,
            'worst_param_regression': worst_param_regression,
            'scales_with_any_param_regression': [
                scale['scale'] for scale in per_scale if scale['round61_vs_kf36']['regression_count'] > 0
            ],
        },
        'notes': [
            'compare_four_methods_shared_noise.py can reuse verified component JSONs; this sweep relies on that behavior and only reruns missing or mismatched component artifacts.',
            'Existing per-scale compare_four_methods report template still contains hard-coded noise0p08 wording in section headers/body text even when running other scales.',
        ],
    }
    return summary


def build_report(summary: dict) -> str:
    per_scale = summary['per_scale']
    scale_order = [item['scale'] for item in per_scale]
    lines: list[str] = []

    lines.append('<callout emoji="🧪" background-color="light-blue">')
    lines.append('这份汇总给出 **四方法 shared-noise cross-scale curve**：`KF36 / Markov42 / Pure SCD baseline / Round61`，并单独抽出 **Round61 相对标准 KF36 仍然落后的尺度/指标/参数**。')
    lines.append('</callout>')
    lines.append('')

    lines.append('## 1. 扫描设置')
    lines.append('')
    lines.append('- 统一数据口径：same dataset family / same seed / same shared-noise construction')
    lines.append('- 扫描点：`' + ', '.join(summary['scales_requested']) + '`')
    lines.append('- 方法：`KF36`, `Markov42`, `Pure SCD`, `Round61`')
    lines.append('')

    lines.append('## 2. 复用 / 重跑概览')
    lines.append('')
    lines.append('| 方法 | execution status 统计 |')
    lines.append('|---|---|')
    for method in METHOD_ORDER:
        counts = summary['execution_counts_by_method'].get(method, {})
        cell = ', '.join(f'{k}×{v}' for k, v in sorted(counts.items())) if counts else 'NA'
        lines.append(f"| {METHOD_DISPLAY[method]} | {cell} |")
    lines.append('')
    total_runtime = sum(item['runtime_seconds'] for item in per_scale)
    lines.append(f"- 总 wall runtime（本 sweep 命令累计）≈ **{fmt_sec(total_runtime)}**")
    lines.append('')

    lines.append('## 3. Overall 曲线（每项越低越好）')
    lines.append('')
    lines.append(f"scale: {' | '.join(scale_order)}")
    lines.append('')
    for metric in OVERALL_KEYS:
        lines.append(f"### {OVERALL_DISPLAY[metric]}")
        lines.append('')
        for method in METHOD_ORDER:
            values = summary['curve_overall_by_method'][method][metric]
            lines.append(
                f"- {METHOD_DISPLAY[method]}: `{sparkline(values)}`  (" + ' | '.join(f'{v:.3f}' for v in values) + ')'
            )
        lines.append('')

    lines.append('## 4. 每个 noise scale 的 best-by-metric')
    lines.append('')
    lines.append('| scale | best mean | best median | best max | Round61 mean rank | Round61 median rank | Round61 max rank |')
    lines.append('|---|---|---|---|---:|---:|---:|')
    for item in per_scale:
        lines.append(
            f"| {item['scale']} | {METHOD_DISPLAY[item['best_by_metric']['mean_pct_error']['method']]} | "
            f"{METHOD_DISPLAY[item['best_by_metric']['median_pct_error']['method']]} | "
            f"{METHOD_DISPLAY[item['best_by_metric']['max_pct_error']['method']]} | "
            f"{item['round61_rank']['mean_pct_error']} | {item['round61_rank']['median_pct_error']} | {item['round61_rank']['max_pct_error']} |"
        )
    lines.append('')

    lines.append('## 5. Round61 vs KF36：总体指标差值（正值=Round61 更好，负值=Round61 落后）')
    lines.append('')
    lines.append('| scale | mean rel% | median rel% | max rel% | param regressions |')
    lines.append('|---|---:|---:|---:|---:|')
    for item in per_scale:
        ov = item['round61_vs_kf36']['overall']
        lines.append(
            f"| {item['scale']} | {fmt_pct(ov['mean_pct_error']['relative_improvement_pct'])} | "
            f"{fmt_pct(ov['median_pct_error']['relative_improvement_pct'])} | "
            f"{fmt_pct(ov['max_pct_error']['relative_improvement_pct'])} | "
            f"{item['round61_vs_kf36']['regression_count']} |"
        )
    lines.append('')

    lines.append('## 6. 明确指出：Round61 仍落后 KF36 的总体指标')
    lines.append('')
    for metric in OVERALL_KEYS:
        trailing = summary['diagnostics']['overall_metrics_where_round61_trails_kf36'][metric]
        worst = summary['diagnostics']['worst_overall_by_metric'][metric]
        if trailing:
            lines.append(
                f"- **{OVERALL_DISPLAY[metric]}** 落后尺度：`{', '.join(trailing)}`；"
                f"最差点是 `{worst['scale']}`（relative improvement {fmt_pct(worst['relative_improvement_pct'])}%）。"
            )
        else:
            best = summary['diagnostics']['best_overall_by_metric'][metric]
            lines.append(
                f"- **{OVERALL_DISPLAY[metric]}**：Round61 在全部扫描点都不落后 KF36；最好点 `{best['scale']}`（{fmt_pct(best['relative_improvement_pct'])}%）。"
            )
    lines.append('')

    lines.append('## 7. 明确指出：参数级回退（Round61 vs KF36）')
    lines.append('')
    lines.append('| scale | regression count | worst regression | regressed params |')
    lines.append('|---|---:|---|---|')
    for item in per_scale:
        regressions = item['round61_vs_kf36']['regressions']
        if regressions:
            worst = regressions[0]
            worst_txt = f"{worst['param']} ({fmt_pct(worst['delta_pct_points'])} pct-pts)"
            params_txt = ', '.join(reg['param'] for reg in regressions)
        else:
            worst_txt = 'none'
            params_txt = 'none'
        lines.append(f"| {item['scale']} | {item['round61_vs_kf36']['regression_count']} | {worst_txt} | {params_txt} |")
    lines.append('')

    lines.append('### 回退参数出现频次（按 scale 统计）')
    lines.append('')
    for item in summary['diagnostics']['param_regression_frequency']:
        lines.append(f"- `{item['param']}`: {item['count']}/{len(per_scale)} scales -> {', '.join(item['scales'])}")
    if not summary['diagnostics']['param_regression_frequency']:
        lines.append('- 无参数级回退。')
    lines.append('')

    worst_param = summary['diagnostics']['worst_param_regression']
    if worst_param:
        lines.append('### 最严重的单点参数回退')
        lines.append('')
        lines.append(
            f"- scale `{worst_param['scale']}` / param `{worst_param['param']}` / "
            f"delta `{fmt_pct(worst_param['delta_pct_points'])}` pct-pts / "
            f"relative improvement `{fmt_pct(worst_param['relative_improvement_pct'])}%`"
        )
        lines.append('')

    lines.append('## 8. 简洁诊断')
    lines.append('')
    best_mean = summary['best_method_counts']['mean_pct_error']
    best_median = summary['best_method_counts']['median_pct_error']
    best_max = summary['best_method_counts']['max_pct_error']
    lines.append(f"- **overall mean**：Round61 在 {best_mean.get('round61', 0)}/{len(per_scale)} 个尺度上是四方法最优。")
    lines.append(f"- **overall median**：Round61 在 {best_median.get('round61', 0)}/{len(per_scale)} 个尺度上是四方法最优。")
    lines.append(f"- **overall max**：Round61 在 {best_max.get('round61', 0)}/{len(per_scale)} 个尺度上是四方法最优；KF36 在 {best_max.get('kf36_noisy', 0)}/{len(per_scale)} 个尺度上最优。")
    lines.append('- **就“Round61 何时仍输给标准 KF36”而言，首先看 overall max，其次看参数级局部回退，而不是看 overall mean。**')
    lines.append('')

    lines.append('## 9. 输出文件')
    lines.append('')
    lines.append(f"- summary json: `{summary_json_path(summary['report_date'])}`")
    lines.append(f"- report md: `{report_md_path(summary['report_date'])}`")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    run_records = []
    started = time.time()
    for scale_spec in SCALE_SPECS:
        run_records.append(run_scale(scale_spec, args.report_date))
    total_runtime = time.time() - started

    summary = build_summary(run_records, args.report_date)
    summary['total_runtime_seconds'] = total_runtime
    summary_path = summary_json_path(args.report_date)
    report_path = report_md_path(args.report_date)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_path.write_text(build_report(summary), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps({
        'summary_json': str(summary_path),
        'report_md': str(report_path),
        'total_runtime_seconds': total_runtime,
        'scales': summary['scales_requested'],
        'overall_max_trailing_scales': summary['diagnostics']['overall_metrics_where_round61_trails_kf36']['max_pct_error'],
        'param_regression_frequency': summary['diagnostics']['param_regression_frequency'][:10],
        'notes': summary['notes'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
