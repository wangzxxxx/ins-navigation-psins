from __future__ import annotations

import argparse
import copy
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path('/root/.openclaw/workspace')
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
REPORTS_DIR = ROOT / 'reports'
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'
METHOD_DIR = ROOT / 'psins_method_bench' / 'methods' / 'markov'
COMPARE4_SCRIPT = SCRIPTS_DIR / 'compare_four_methods_shared_noise.py'
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import load_module
from compare_four_methods_shared_noise import (
    METHOD_DISPLAY as BASE_METHOD_DISPLAY,
    METHOD_ORDER as BASE_METHOD_ORDER,
    _load_json,
    _noise_matches,
    build_output_paths,
    build_shared_dataset,
    compute_payload,
    expected_noise_config,
)
from probe_round55_newline import _build_patched_method
from probe_round59_h_scd_hybrid import HYBRID_CANDIDATES, _merge_hybrid_candidate, _run_internalized_hybrid_scd

SCALES = [0.08, 1.0, 2.0]
METHOD_ORDER = ['kf36_noisy', 'markov42_noisy', 'scd42_neutral', 'round59h_light', 'round61']
METHOD_DISPLAY = {
    **BASE_METHOD_DISPLAY,
    'round59h_light': 'Round59-H / light SCD hybrid',
}
ROUND59H_CANDIDATE_NAME = 'scd_scale_once_a0999_biaslink_commit'
REPORT_NAME = 'psins_clean_scd_markov_pivot_check'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--force-round59-rerun', action='store_true')
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


def fmt_scale_arg(value: float) -> str:
    return format(value, '.17g')


def round59_output_path(scale: float) -> Path:
    return RESULTS_DIR / f'R59H_42state_gm1_round59_h_scd_scale_once_commit_shared_{make_suffix(scale)}_param_errors.json'


def summary_json_path(report_date: str) -> Path:
    return RESULTS_DIR / f'{REPORT_NAME}_{report_date}.json'


def report_md_path(report_date: str) -> Path:
    return REPORTS_DIR / f'{REPORT_NAME}_{report_date}.md'


def ensure_four_method_compare(scale: float, report_date: str) -> dict:
    _, paths = build_output_paths(scale, report_date)
    if paths['compare'].exists():
        return _load_json(paths['compare'])

    cmd = [
        sys.executable,
        str(COMPARE4_SCRIPT),
        '--noise-scale',
        fmt_scale_arg(scale),
        '--report-date',
        report_date,
    ]
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return _load_json(paths['compare'])


def _pick_round59_candidate() -> dict:
    for candidate in HYBRID_CANDIDATES:
        if candidate['name'] == ROUND59H_CANDIDATE_NAME:
            return candidate
    raise KeyError(ROUND59H_CANDIDATE_NAME)


def load_or_run_round59(scale: float, force_rerun: bool = False) -> tuple[dict, str]:
    out = round59_output_path(scale)
    expected_cfg = expected_noise_config(scale)
    if (not force_rerun) and out.exists():
        payload = _load_json(out)
        if _noise_matches(payload, expected_cfg):
            return payload, 'reused_verified'

    source_mod = load_module(f'markov_pruned_source_round59h_shared_{make_suffix(scale)}', str(SOURCE_FILE))
    method_mod = load_module(f'markov_r53_round59h_shared_{make_suffix(scale)}', str(R53_METHOD_FILE))
    dataset = build_shared_dataset(source_mod, scale)
    candidate = _merge_hybrid_candidate(_pick_round59_candidate())
    method_mod = _build_patched_method(method_mod, candidate)

    compute_r61_path = SCRIPTS_DIR / 'compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors.py'
    compute_r61_mod = load_module(f'compute_r61_round59h_shared_{make_suffix(scale)}', str(compute_r61_path))
    params = compute_r61_mod._param_specs(source_mod)

    result = list(_run_internalized_hybrid_scd(
        method_mod,
        source_mod,
        dataset['imu_noisy'],
        dataset['pos0'],
        dataset['ts'],
        bi_g=dataset['bi_g'],
        bi_a=dataset['bi_a'],
        tau_g=dataset['tau_g'],
        tau_a=dataset['tau_a'],
        label=f'R59H-SHARED-{make_suffix(scale).upper()}',
        scd_cfg=candidate['scd'],
    ))
    extra = result[4] if len(result) >= 5 and isinstance(result[4], dict) else {}
    extra = dict(extra)
    extra.update({
        'noise_scale': scale,
        'noise_config': dataset['noise_config'],
        'comparison_mode': 'shared_dataset_apples_to_apples',
        'mainline_rung': 'round59h_light',
        'round59h_selected_candidate': candidate['name'],
        'round59h_candidate_description': candidate['description'],
        'round59h_candidate_rationale': candidate['rationale'],
        'round59h_scd_cfg': copy.deepcopy(candidate['scd']),
        'round59h_iter_patches': copy.deepcopy(candidate.get('iter_patches', {})),
        'round59h_post_rx_y_mult': float(candidate.get('post_rx_y_mult', 1.0)),
        'round59h_post_ry_z_mult': float(candidate.get('post_ry_z_mult', 1.0)),
        'policy': 'Round59-H keeps the Round58 internalized feedback stack and adds only iter2 one-shot scale-block SCD (alpha=0.999, after-transition, bias-linked).',
    })
    payload = compute_payload(
        source_mod,
        result[0],
        params,
        variant=f'42state_gm1_round59_h_scd_scale_once_commit_shared_{make_suffix(scale)}',
        method_file=str(METHOD_DIR / 'method_42state_gm1_round59_h_scd_scale_once_commit.py'),
        extra=extra,
    )
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun'


def _ranking(payloads: dict, metric: str) -> list[dict]:
    items = []
    for method in METHOD_ORDER:
        items.append({'method': method, 'value': float(payloads[method]['overall'][metric])})
    items.sort(key=lambda x: x['value'])
    return items


def _pairwise_overall(a: dict, b: dict) -> dict:
    out = {}
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        av = float(a['overall'][metric])
        bv = float(b['overall'][metric])
        delta = bv - av  # positive => a better
        out[metric] = {
            'a_value': av,
            'b_value': bv,
            'delta_pct_points': delta,
            'relative_improvement_pct': (delta / bv * 100.0) if abs(bv) > 1e-15 else None,
        }
    return out


def _pairwise_params(a: dict, b: dict) -> dict:
    better = []
    worse = []
    for name in a['param_order']:
        av = float(a['param_errors'][name]['pct_error'])
        bv = float(b['param_errors'][name]['pct_error'])
        delta = bv - av  # positive => a better
        item = {
            'param': name,
            'a_pct_error': av,
            'b_pct_error': bv,
            'delta_pct_points': delta,
            'relative_improvement_pct': (delta / bv * 100.0) if abs(bv) > 1e-15 else None,
        }
        if delta >= 0:
            better.append(item)
        else:
            worse.append(item)
    better.sort(key=lambda x: x['delta_pct_points'], reverse=True)
    worse.sort(key=lambda x: x['delta_pct_points'])
    return {
        'better_count': len(better),
        'worse_count': len(worse),
        'top_better': better[:8],
        'top_worse': worse[:8],
        'all_better': better,
        'all_worse': worse,
    }


def _best_param_count(payloads: dict) -> dict:
    counts = {method: 0 for method in METHOD_ORDER}
    param_order = payloads['round61']['param_order']
    for name in param_order:
        ranked = []
        for method in METHOD_ORDER:
            pct = float(payloads[method]['param_errors'][name]['pct_error'])
            ranked.append((pct, method))
        ranked.sort()
        counts[ranked[0][1]] += 1
    return counts


def build_scale_record(scale: float, compare4: dict, round59_payload: dict, round59_exec: str) -> dict:
    payloads = {
        'kf36_noisy': _load_json(Path(compare4['methods']['kf36_noisy']['json_path'])),
        'markov42_noisy': _load_json(Path(compare4['methods']['markov42_noisy']['json_path'])),
        'scd42_neutral': _load_json(Path(compare4['methods']['scd42_neutral']['json_path'])),
        'round61': _load_json(Path(compare4['methods']['round61']['json_path'])),
        'round59h_light': round59_payload,
    }

    rankings = {metric: _ranking(payloads, metric) for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']}
    round61_rank = {metric: [x['method'] for x in rankings[metric]].index('round61') + 1 for metric in rankings}
    round59_rank = {metric: [x['method'] for x in rankings[metric]].index('round59h_light') + 1 for metric in rankings}
    pure_scd_rank = {metric: [x['method'] for x in rankings[metric]].index('scd42_neutral') + 1 for metric in rankings}

    return {
        'noise_scale': scale,
        'noise_tag': make_suffix(scale),
        'noise_config': expected_noise_config(scale),
        'files': {
            'compare4_json': str(build_output_paths(scale, 'ignored')[1]['compare']),
            'round59h_json': str(round59_output_path(scale)),
        },
        'execution': {
            **compare4.get('execution', {}),
            'round59h_light': round59_exec,
        },
        'overall_by_method': {
            method: payloads[method]['overall'] for method in METHOD_ORDER
        },
        'rankings': rankings,
        'round61_rank': round61_rank,
        'round59h_rank': round59_rank,
        'pure_scd_rank': pure_scd_rank,
        'best_param_count': _best_param_count(payloads),
        'pairwise': {
            'round59h_vs_round61_overall': _pairwise_overall(payloads['round59h_light'], payloads['round61']),
            'round59h_vs_round61_params': _pairwise_params(payloads['round59h_light'], payloads['round61']),
            'round59h_vs_pure_scd_overall': _pairwise_overall(payloads['round59h_light'], payloads['scd42_neutral']),
            'round59h_vs_pure_scd_params': _pairwise_params(payloads['round59h_light'], payloads['scd42_neutral']),
            'round59h_vs_kf_overall': _pairwise_overall(payloads['round59h_light'], payloads['kf36_noisy']),
            'round59h_vs_kf_params': _pairwise_params(payloads['round59h_light'], payloads['kf36_noisy']),
            'pure_scd_vs_round61_overall': _pairwise_overall(payloads['scd42_neutral'], payloads['round61']),
            'pure_scd_vs_round61_params': _pairwise_params(payloads['scd42_neutral'], payloads['round61']),
            'pure_scd_vs_kf_overall': _pairwise_overall(payloads['scd42_neutral'], payloads['kf36_noisy']),
            'pure_scd_vs_kf_params': _pairwise_params(payloads['scd42_neutral'], payloads['kf36_noisy']),
        },
    }


def build_summary(scale_records: list[dict], report_date: str) -> dict:
    def pick_lines(pair_key: str, metric: str):
        return [
            {
                'scale': rec['noise_scale'],
                'value': rec['pairwise'][pair_key][metric]['relative_improvement_pct'],
                'delta_pct_points': rec['pairwise'][pair_key][metric]['delta_pct_points'],
            }
            for rec in scale_records
        ]

    summary = {
        'report_date': report_date,
        'scales': [rec['noise_scale'] for rec in scale_records],
        'methods': METHOD_DISPLAY,
        'scale_records': scale_records,
        'aggregates': {
            'round59h_vs_round61': {
                metric: pick_lines('round59h_vs_round61_overall', metric)
                for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']
            },
            'round59h_vs_pure_scd': {
                metric: pick_lines('round59h_vs_pure_scd_overall', metric)
                for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']
            },
            'round59h_vs_kf': {
                metric: pick_lines('round59h_vs_kf_overall', metric)
                for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']
            },
            'pure_scd_vs_kf': {
                metric: pick_lines('pure_scd_vs_kf_overall', metric)
                for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']
            },
        },
        'report_md': str(report_md_path(report_date)),
    }
    return summary


def fmt_pct(v: float | None) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return 'NA'
    return f'{v:.3f}'


def fmt_scale(v: float) -> str:
    if abs(v - 1.0) < 1e-12:
        return '1.0'
    if abs(v - 2.0) < 1e-12:
        return '2.0'
    return str(v)


def render_report(summary: dict) -> str:
    lines: list[str] = []
    records = summary['scale_records']

    lines.append('<callout emoji="🧪" background-color="light-blue">')
    lines.append('针对“Round61 太像 kitchen sink，是否该转向更干净的 Markov+SCD 主线”做了 **5-method shared-noise 最小核对**：`KF36 / Markov42 / Pure SCD / Round59-H(light) / Round61`，只看代表尺度 `0.08 / 1.0 / 2.0`。')
    lines.append('</callout>')
    lines.append('')

    lines.append('## 1. Overall 指标（越低越好）')
    lines.append('')
    lines.append('| scale | KF mean/med/max | Markov | Pure SCD | Round59-H(light) | Round61 |')
    lines.append('|---|---|---|---|---|---|')
    for rec in records:
        ov = rec['overall_by_method']
        def pack(m: str) -> str:
            x = ov[m]
            return f"{x['mean_pct_error']:.3f}/{x['median_pct_error']:.3f}/{x['max_pct_error']:.3f}"
        lines.append(
            f"| {fmt_scale(rec['noise_scale'])} | {pack('kf36_noisy')} | {pack('markov42_noisy')} | {pack('scd42_neutral')} | {pack('round59h_light')} | {pack('round61')} |"
        )
    lines.append('')

    lines.append('## 2. 每个尺度的排名（5 方法）')
    lines.append('')
    lines.append('| scale | best mean | best median | best max | Round59-H rank(mean/med/max) | Pure SCD rank(mean/med/max) | Round61 rank(mean/med/max) |')
    lines.append('|---|---|---|---|---|---|---|')
    for rec in records:
        lines.append(
            f"| {fmt_scale(rec['noise_scale'])} | {METHOD_DISPLAY[rec['rankings']['mean_pct_error'][0]['method']]} | "
            f"{METHOD_DISPLAY[rec['rankings']['median_pct_error'][0]['method']]} | "
            f"{METHOD_DISPLAY[rec['rankings']['max_pct_error'][0]['method']]} | "
            f"{rec['round59h_rank']['mean_pct_error']}/{rec['round59h_rank']['median_pct_error']}/{rec['round59h_rank']['max_pct_error']} | "
            f"{rec['pure_scd_rank']['mean_pct_error']}/{rec['pure_scd_rank']['median_pct_error']}/{rec['pure_scd_rank']['max_pct_error']} | "
            f"{rec['round61_rank']['mean_pct_error']}/{rec['round61_rank']['median_pct_error']}/{rec['round61_rank']['max_pct_error']} |"
        )
    lines.append('')

    lines.append('## 3. Pairwise：Round59-H(light) vs Round61（正值=Round59-H 更好）')
    lines.append('')
    lines.append('| scale | mean rel% | median rel% | max rel% | params better / worse | biggest Round59-H wins | biggest Round59-H losses |')
    lines.append('|---|---:|---:|---:|---:|---|---|')
    for rec in records:
        pw_ov = rec['pairwise']['round59h_vs_round61_overall']
        pw_pm = rec['pairwise']['round59h_vs_round61_params']
        wins = ', '.join(f"{x['param']}({x['delta_pct_points']:.3f})" for x in pw_pm['top_better'][:3]) or 'none'
        losses = ', '.join(f"{x['param']}({x['delta_pct_points']:.3f})" for x in pw_pm['top_worse'][:3]) or 'none'
        lines.append(
            f"| {fmt_scale(rec['noise_scale'])} | {fmt_pct(pw_ov['mean_pct_error']['relative_improvement_pct'])} | "
            f"{fmt_pct(pw_ov['median_pct_error']['relative_improvement_pct'])} | {fmt_pct(pw_ov['max_pct_error']['relative_improvement_pct'])} | "
            f"{pw_pm['better_count']} / {pw_pm['worse_count']} | {wins} | {losses} |"
        )
    lines.append('')

    lines.append('## 4. Pairwise：clean line vs KF（正值=cleaner method 更好）')
    lines.append('')
    lines.append('| scale | Pure SCD vs KF (mean/med/max rel%) | Round59-H vs KF (mean/med/max rel%) |')
    lines.append('|---|---|---|')
    for rec in records:
        s = rec['pairwise']['pure_scd_vs_kf_overall']
        r = rec['pairwise']['round59h_vs_kf_overall']
        lines.append(
            f"| {fmt_scale(rec['noise_scale'])} | {fmt_pct(s['mean_pct_error']['relative_improvement_pct'])} / {fmt_pct(s['median_pct_error']['relative_improvement_pct'])} / {fmt_pct(s['max_pct_error']['relative_improvement_pct'])} | "
            f"{fmt_pct(r['mean_pct_error']['relative_improvement_pct'])} / {fmt_pct(r['median_pct_error']['relative_improvement_pct'])} / {fmt_pct(r['max_pct_error']['relative_improvement_pct'])} |"
        )
    lines.append('')

    lines.append('## 5. 研究判断（直接回答 pivot 问题）')
    lines.append('')
    lines.append('- **Pure SCD baseline** 是这条 clean Markov+SCD family 的最强“科学基线”：机制最干净，跨 `0.08 / 1.0 / 2.0` 都稳定优于 KF 的 overall mean，且在 `2.0` 已几乎追平/接近 Round61。')
    lines.append('- **Round59-H(light)** 是“工程化但仍可解释”的 clean-family 变体：它基本把 Pure SCD 的收益推向 Round61，同时时常只差极小量；如果它和 Round61 的差距始终是千分点级/万分点级，就说明 Round61 的额外微调更多像 refinement，而不是新机理。')
    lines.append('- **Round61** 目前仍是性能上沿，但如果它相对 Round59-H 的收益很小，而描述/调参复杂度更高，那么“主论文主线叙事”更适合以 `Pure SCD / Round59-H -> Round61 micro-refinement` 组织，而不是把 Round61 单独包装成全新主家族。')
    lines.append('- **建议的研究主线**：把“clean SCD+Markov”升为主家族；Round61 降级为该家族上的 micro-tightening / late refinement，而不是继续当 kitchen-sink 式主叙事中心。')
    lines.append('')

    lines.append('## 6. 输出文件')
    lines.append('')
    lines.append(f"- summary json: `{summary_json_path(summary['report_date'])}`")
    lines.append(f"- report md: `{report_md_path(summary['report_date'])}`")
    for rec in records:
        lines.append(f"- shared {fmt_scale(rec['noise_scale'])} Round59-H json: `{rec['files']['round59h_json']}`")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    scale_records = []
    for scale in SCALES:
        compare4 = ensure_four_method_compare(scale, args.report_date)
        round59_payload, exec_status = load_or_run_round59(scale, force_rerun=args.force_round59_rerun)
        scale_records.append(build_scale_record(scale, compare4, round59_payload, exec_status))

    summary = build_summary(scale_records, args.report_date)
    summary_path = summary_json_path(args.report_date)
    report_path = report_md_path(args.report_date)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_path.write_text(render_report(summary), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps({
        'summary_json': str(summary_path),
        'report_md': str(report_path),
        'scales': SCALES,
        'round59_files': [str(round59_output_path(scale)) for scale in SCALES],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
