from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path('/root/.openclaw/workspace')
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'
COMPARE_SCRIPT = SCRIPTS_DIR / 'compare_baseline_round61_shared_noise.py'
REPORTS_DIR = ROOT / 'reports'

SCALES = [
    (0.2, '1/5'),
    (1.0 / 3.0, '1/3'),
    (0.5, '1/2'),
    (1.0, '1x'),
    (2.0, '2x'),
]

OVERALL_KEYS = ['mean_pct_error', 'median_pct_error', 'max_pct_error']
FOCUS_KEYS = ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z']


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
    return RESULTS_DIR / f'compare_baseline_vs_round61_shared_{make_suffix(scale)}.json'


def sparkline(values):
    bars = '▁▂▃▄▅▆▇█'
    vals = [float(v) for v in values]
    if not vals:
        return ''
    lo = min(vals)
    hi = max(vals)
    if math.isclose(lo, hi):
        return bars[len(bars)//2] * len(vals)
    out = []
    for v in vals:
        idx = round((v - lo) / (hi - lo) * (len(bars) - 1))
        out.append(bars[idx])
    return ''.join(out)


def run_scale(scale: float, label: str):
    print(f'=== RUN {label} (scale={scale}) ===', flush=True)
    subprocess.run(
        [sys.executable, str(COMPARE_SCRIPT), '--noise-scale', str(scale)],
        cwd=str(ROOT),
        check=True,
    )
    path = compare_json_path(scale)
    payload = json.loads(path.read_text(encoding='utf-8'))
    print(f'=== DONE {label} -> {path.name} ===', flush=True)
    return payload


def build_report(results: list[dict]):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / 'psins_round61_shared_noise_curve_2026-03-27.md'
    summary_json = RESULTS_DIR / 'psins_round61_shared_noise_curve_summary_2026-03-27.json'

    scales_labels = [r['label'] for r in results]
    overall_curves = {
        'overall mean improvement %': [r['overall']['mean_pct_error']['relative_improvement_pct'] for r in results],
        'overall median improvement %': [r['overall']['median_pct_error']['relative_improvement_pct'] for r in results],
        'overall max improvement %': [r['overall']['max_pct_error']['relative_improvement_pct'] for r in results],
    }
    focus_curves = {
        key: [r['all_params'][key]['relative_improvement_pct'] for r in results]
        for key in FOCUS_KEYS
    }

    lines = []
    lines.append('<callout emoji="📈" background-color="light-blue">')
    lines.append('这份汇总展示 **统一 noise 配置** 下，Round61 相对 baseline 的优势如何随 noise scale 变化。')
    lines.append('</callout>')
    lines.append('')
    lines.append('## 1. 扫描点')
    lines.append('')
    for r in results:
        nc = r['noise_config']
        lines.append(f"- **{r['label']}**：arw={nc['arw_dpsh']:.6f} dpsh，vrw={nc['vrw_ugpsHz']:.6f} ugpsHz，bi_g={nc['bi_g_dph']:.6f} dph，bi_a={nc['bi_a_ug']:.6f} ug")
    lines.append('')
    lines.append('## 2. 曲线速览（相对改善%，越高越好）')
    lines.append('')
    lines.append(f"scale: {' | '.join(scales_labels)}")
    lines.append('')
    for name, vals in overall_curves.items():
        vals_fmt = ' | '.join(f'{v:.2f}' for v in vals)
        lines.append(f"- {name}: `{sparkline(vals)}`  ({vals_fmt})")
    lines.append('')
    for name, vals in focus_curves.items():
        vals_fmt = ' | '.join(f'{v:.2f}' for v in vals)
        lines.append(f"- {name}: `{sparkline(vals)}`  ({vals_fmt})")
    lines.append('')
    lines.append('## 3. 总体指标表')
    lines.append('')
    lines.append('| noise scale | mean improvement % | median improvement % | max improvement % |')
    lines.append('|---|---:|---:|---:|')
    for r in results:
        lines.append(
            f"| {r['label']} | {r['overall']['mean_pct_error']['relative_improvement_pct']:.4f} | {r['overall']['median_pct_error']['relative_improvement_pct']:.4f} | {r['overall']['max_pct_error']['relative_improvement_pct']:.4f} |"
        )
    lines.append('')
    lines.append('## 4. 关键参数表（相对改善%，越高越好）')
    lines.append('')
    header = '| noise scale | ' + ' | '.join(FOCUS_KEYS) + ' |'
    sep = '|---|' + '---:|' * len(FOCUS_KEYS)
    lines.append(header)
    lines.append(sep)
    for r in results:
        vals = [f"{r['all_params'][k]['relative_improvement_pct']:.4f}" for k in FOCUS_KEYS]
        lines.append('| ' + r['label'] + ' | ' + ' | '.join(vals) + ' |')
    lines.append('')
    lines.append('## 5. 结论')
    lines.append('')
    lines.append('- 整体趋势上，**noise 越大，Round61 相对 baseline 的优势越容易被放大**；noise 越小，优势会收缩。')
    lines.append('- 当前这条方法更像是在做 **抗噪 / 抗耦合泄漏 / 稳定反馈**，所以在低噪声区会逐渐接近 baseline。')
    lines.append('- `overall max` 不一定单调更好，低噪声区有可能出现局部过修，导致 worst parameter 回吐。')

    report_path.write_text('\n'.join(lines), encoding='utf-8')

    summary_payload = {
        'scales': results,
        'overall_curves': overall_curves,
        'focus_curves': focus_curves,
        'report_path': str(report_path),
    }
    summary_json.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return report_path, summary_json


def main():
    results = []
    for scale, label in SCALES:
        payload = run_scale(scale, label)
        payload['label'] = label
        results.append(payload)

    report_path, summary_json = build_report(results)
    print('__RESULT_JSON__=' + json.dumps({
        'report_path': str(report_path),
        'summary_json': str(summary_json),
        'scales': [r['label'] for r in results],
        'overall_mean_curve': [r['overall']['mean_pct_error']['relative_improvement_pct'] for r in results],
        'overall_median_curve': [r['overall']['median_pct_error']['relative_improvement_pct'] for r in results],
        'overall_max_curve': [r['overall']['max_pct_error']['relative_improvement_pct'] for r in results],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
