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
COMPUTE_R61_FILE = SCRIPTS_DIR / 'compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors.py'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import load_module
from compare_four_methods_shared_noise import (
    _build_neutral_scd_candidate,
    _load_json,
    _noise_matches,
    build_output_paths,
    build_shared_dataset,
    compute_payload,
    expected_noise_config,
)
from probe_round55_newline import _build_patched_method
from probe_round59_h_scd_hybrid import _run_internalized_hybrid_scd

SCALES = [0.08, 1.0, 2.0]
FORBIDDEN_ITER_KEYS = {
    'state_alpha_mult',
    'state_prior_diag_mult',
    'state_q_static_mult',
    'state_q_dynamic_mult',
    'state_q_late_mult',
}
FORBIDDEN_ROOT_KEYS = {'post_rx_y_mult', 'post_ry_z_mult'}
REPORT_BASENAME = 'psins_llm_scd_only_clean_probe'
SUMMARY_BASENAME = 'llm_scd_only_clean_probe_summary'
CANDIDATE_BASENAME = 'llm_scd_only_clean_candidates'
METHOD_ORDER = ['kf36_noisy', 'markov42_noisy', 'scd42_neutral']
CANDIDATE_ORDER = ['candidate_A', 'candidate_B', 'candidate_C', 'candidate_D']
METHOD_DISPLAY = {
    'kf36_noisy': 'KF36 baseline',
    'markov42_noisy': 'Markov42 baseline',
    'scd42_neutral': 'Pure SCD baseline (neutral Markov+SCD)',
    'candidate_A': 'A · selected / once_per_phase / alpha=0.999 / td=2.0 / bias=True',
    'candidate_B': 'B · scale_block / once_per_phase / alpha=0.9995 / td=2.0 / bias=True',
    'candidate_C': 'C · scale_block / once_per_phase / alpha=0.999 / td=4.0 / bias=True',
    'candidate_D': 'D · selected / repeat_after_transition / alpha=0.9998 / td=2.0 / bias=True',
}

CANDIDATES = [
    {
        'name': 'candidate_A',
        'label': 'A',
        'description': 'selected scope, once_per_phase, alpha=0.999, transition_duration=2.0, bias_to_target=True',
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.999,
            'transition_duration': 2.0,
            'target': 'selected',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    },
    {
        'name': 'candidate_B',
        'label': 'B',
        'description': 'scale_block scope, once_per_phase, alpha=0.9995, transition_duration=2.0, bias_to_target=True',
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.9995,
            'transition_duration': 2.0,
            'target': 'scale_block',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    },
    {
        'name': 'candidate_C',
        'label': 'C',
        'description': 'scale_block scope, once_per_phase, alpha=0.999, transition_duration=4.0, bias_to_target=True',
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.999,
            'transition_duration': 4.0,
            'target': 'scale_block',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    },
    {
        'name': 'candidate_D',
        'label': 'D',
        'description': 'selected scope, repeat_after_transition, alpha=0.9998, transition_duration=2.0, bias_to_target=True',
        'scd': {
            'mode': 'repeat_after_transition',
            'alpha': 0.9998,
            'transition_duration': 2.0,
            'target': 'selected',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--force-rerun', action='store_true')
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


def fmt_scale_text(value: float) -> str:
    if abs(value - 1.0) < 1e-12:
        return '1.0'
    if abs(value - 2.0) < 1e-12:
        return '2.0'
    return str(value)


def candidate_result_path(scale: float, candidate_name: str) -> Path:
    return RESULTS_DIR / f"SCD42_llm_clean_{candidate_name}_shared_{make_suffix(scale)}_param_errors.json"


def summary_json_path(report_date: str) -> Path:
    return RESULTS_DIR / f'{SUMMARY_BASENAME}_{report_date}.json'


def candidate_json_path(report_date: str) -> Path:
    return RESULTS_DIR / f'{CANDIDATE_BASENAME}_{report_date}.json'


def report_md_path(report_date: str) -> Path:
    return REPORTS_DIR / f'{REPORT_BASENAME}_{report_date}.md'


def _candidate_signature(candidate: dict) -> str:
    sig = {
        'name': candidate['name'],
        'label': candidate['label'],
        'description': candidate['description'],
        'scd': candidate['scd'],
    }
    return json.dumps(sig, ensure_ascii=False, sort_keys=True)


def ensure_candidate_constraints(candidate: dict) -> None:
    for key in FORBIDDEN_ROOT_KEYS:
        if key in candidate:
            raise ValueError(f'Candidate {candidate["name"]} illegally sets forbidden root key: {key}')
    for iter_idx, patch in candidate.get('iter_patches', {}).items():
        for key in patch:
            if key in FORBIDDEN_ITER_KEYS:
                raise ValueError(f'Candidate {candidate["name"]} illegally sets forbidden iter patch key {key} at iter {iter_idx}')


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


def build_clean_candidate(candidate: dict) -> dict:
    ensure_candidate_constraints(candidate)
    base = _build_neutral_scd_candidate()
    base['name'] = candidate['name']
    base['description'] = candidate['description']
    base['scd'] = copy.deepcopy(candidate['scd'])
    return base


def load_or_run_candidate(scale: float, candidate: dict, force_rerun: bool = False) -> tuple[dict, str]:
    out = candidate_result_path(scale, candidate['name'])
    expected_cfg = expected_noise_config(scale)
    candidate_sig = _candidate_signature(candidate)

    if (not force_rerun) and out.exists():
        payload = _load_json(out)
        if _noise_matches(payload, expected_cfg):
            extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
            if extra.get('candidate_signature') == candidate_sig:
                return payload, 'reused_verified'

    source_mod = load_module(f'markov_pruned_source_llm_scd_clean_{candidate["name"]}_{make_suffix(scale)}', str(SOURCE_FILE))
    method_mod = load_module(f'markov_r53_llm_scd_clean_{candidate["name"]}_{make_suffix(scale)}', str(R53_METHOD_FILE))
    compute_mod = load_module(f'compute_r61_llm_scd_clean_{candidate["name"]}_{make_suffix(scale)}', str(COMPUTE_R61_FILE))

    dataset = build_shared_dataset(source_mod, scale)
    params = compute_mod._param_specs(source_mod)
    merged_candidate = build_clean_candidate(candidate)
    method_mod = _build_patched_method(method_mod, merged_candidate)

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
        label=f'LLM-SCD-CLEAN-{candidate["label"]}-{make_suffix(scale).upper()}',
        scd_cfg=merged_candidate['scd'],
    ))

    runtime = result[4] if len(result) >= 5 and isinstance(result[4], dict) else {}
    payload = compute_payload(
        source_mod,
        result[0],
        params,
        variant=f'42state_gm1_llm_scd_only_clean_{candidate["name"]}_{make_suffix(scale)}',
        method_file='probe_llm_scd_only_clean::neutral_markov42_plus_scd_with_llm_scd_only_cfg',
        extra={
            'noise_scale': scale,
            'noise_config': dataset['noise_config'],
            'comparison_mode': 'shared_dataset_apples_to_apples',
            'mainline_rung': candidate['name'],
            'candidate_name': candidate['name'],
            'candidate_label': candidate['label'],
            'candidate_description': candidate['description'],
            'candidate_signature': candidate_sig,
            'candidate_scd_cfg': copy.deepcopy(candidate['scd']),
            'constraint_note': 'SCD-only candidate on top of neutral Markov+SCD baseline; no state alpha/prior/q multipliers; no post_rx_y_mult; no post_ry_z_mult.',
            'runtime_log': {
                'schedule_log': runtime.get('schedule_log'),
                'feedback_log': runtime.get('feedback_log'),
                'scd_log': runtime.get('scd_log'),
            },
        },
    )
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun'


def pairwise_overall(a: dict, b: dict) -> dict:
    out = {}
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        av = float(a['overall'][metric])
        bv = float(b['overall'][metric])
        delta = bv - av  # positive => a better than b
        out[metric] = {
            'a_value': av,
            'b_value': bv,
            'delta_pct_points': delta,
            'relative_improvement_pct': (delta / bv * 100.0) if abs(bv) > 1e-15 else None,
            'a_better': delta > 0,
        }
    return out


def pairwise_params(a: dict, b: dict) -> dict:
    better = []
    worse = []
    same = []
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
        if delta > 1e-12:
            better.append(item)
        elif delta < -1e-12:
            worse.append(item)
        else:
            same.append(item)
    better.sort(key=lambda x: x['delta_pct_points'], reverse=True)
    worse.sort(key=lambda x: x['delta_pct_points'])
    return {
        'better_count': len(better),
        'worse_count': len(worse),
        'same_count': len(same),
        'top_better': better[:8],
        'top_worse': worse[:8],
    }


def build_scale_record(scale: float, compare4: dict, candidate_payloads: dict, candidate_exec: dict) -> dict:
    payloads = {
        'kf36_noisy': _load_json(Path(compare4['methods']['kf36_noisy']['json_path'])),
        'markov42_noisy': _load_json(Path(compare4['methods']['markov42_noisy']['json_path'])),
        'scd42_neutral': _load_json(Path(compare4['methods']['scd42_neutral']['json_path'])),
        **candidate_payloads,
    }

    best_candidate_by_metric = {}
    beats_pure_scd = {'mean_pct_error': [], 'median_pct_error': [], 'max_pct_error': []}

    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        ranked = sorted(
            [{'candidate': name, 'value': float(payloads[name]['overall'][metric])} for name in CANDIDATE_ORDER],
            key=lambda x: x['value']
        )
        best_candidate_by_metric[metric] = ranked[0]
        scd_value = float(payloads['scd42_neutral']['overall'][metric])
        for item in ranked:
            if item['value'] < scd_value:
                beats_pure_scd[metric].append(item['candidate'])

    candidate_vs_pure = {}
    candidate_vs_round61 = {}
    for name in CANDIDATE_ORDER:
        candidate_vs_pure[name] = {
            'overall': pairwise_overall(payloads[name], payloads['scd42_neutral']),
            'params': pairwise_params(payloads[name], payloads['scd42_neutral']),
        }
        if compare4['methods'].get('round61'):
            round61_payload = _load_json(Path(compare4['methods']['round61']['json_path']))
            candidate_vs_round61[name] = {
                'overall': pairwise_overall(payloads[name], round61_payload),
                'params': pairwise_params(payloads[name], round61_payload),
            }

    return {
        'noise_scale': scale,
        'noise_tag': make_suffix(scale),
        'noise_config': expected_noise_config(scale),
        'files': {
            'compare4_json': str(build_output_paths(scale, 'ignored')[1]['compare']),
            'baseline_kf36_json': compare4['methods']['kf36_noisy']['json_path'],
            'baseline_markov42_json': compare4['methods']['markov42_noisy']['json_path'],
            'baseline_pure_scd_json': compare4['methods']['scd42_neutral']['json_path'],
            'candidate_jsons': {name: str(candidate_result_path(scale, name)) for name in CANDIDATE_ORDER},
        },
        'execution': {
            **compare4.get('execution', {}),
            **candidate_exec,
        },
        'overall_by_method': {
            name: payloads[name]['overall']
            for name in ['kf36_noisy', 'markov42_noisy', 'scd42_neutral', *CANDIDATE_ORDER]
        },
        'best_candidate_by_metric': best_candidate_by_metric,
        'beats_pure_scd': beats_pure_scd,
        'candidate_vs_pure_scd': candidate_vs_pure,
        'candidate_vs_round61': candidate_vs_round61,
    }


def build_headline(scale_records: list[dict]) -> dict:
    any_beats = {'mean_pct_error': False, 'median_pct_error': False, 'max_pct_error': False}
    beat_instances = {'mean_pct_error': [], 'median_pct_error': [], 'max_pct_error': []}
    full_win_scales = []

    for rec in scale_records:
        scale = rec['noise_scale']
        for metric in any_beats:
            winners = rec['beats_pure_scd'][metric]
            if winners:
                any_beats[metric] = True
                beat_instances[metric].append({'scale': scale, 'candidates': winners})
        best_mean_name = rec['best_candidate_by_metric']['mean_pct_error']['candidate']
        pair = rec['candidate_vs_pure_scd'][best_mean_name]['overall']
        if all(pair[m]['a_better'] for m in ['mean_pct_error', 'median_pct_error', 'max_pct_error']):
            full_win_scales.append(scale)

    local_signal = bool(full_win_scales)
    cross_scale_robust = len(full_win_scales) >= 2
    continue_narrowly = local_signal

    if not any(any_beats.values()):
        judgment = 'No SCD-only LLM candidate beats the pure SCD baseline on mean/median/max at any tested scale.'
    elif cross_scale_robust:
        judgment = 'SCD-only LLM guidance shows cross-scale robust wins over pure SCD.'
    elif local_signal:
        judgment = 'SCD-only LLM guidance shows a real but local clean win over pure SCD (only at part of the tested noise range, not cross-scale robust yet).'
    else:
        judgment = 'Some isolated metric wins exist, but they are fragmented and not scientifically meaningful as a stable improvement over pure SCD.'

    return {
        'any_candidate_beats_pure_scd': any_beats,
        'beat_instances': beat_instances,
        'full_win_scales': full_win_scales,
        'local_signal': local_signal,
        'cross_scale_robust': cross_scale_robust,
        'continue_narrowly': continue_narrowly,
        'judgment': judgment,
    }


def fmt_pct(v: float | None) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return 'NA'
    return f'{v:.3f}'


def render_report(summary: dict) -> str:
    lines: list[str] = []
    lines.append('<callout emoji="🧪" background-color="light-blue">')
    lines.append('本轮是 **clean SCD-only LLM guidance** 试验：固定 shared dataset / noise / seed，只允许在纯 SCD baseline 上改 SCD 配置，不允许改 state alpha/prior/q 微调，也不允许 lever 后处理。')
    lines.append('</callout>')
    lines.append('')

    lines.append('## 1. Candidate batch（严格 SCD-only）')
    lines.append('')
    lines.append('| candidate | definition |')
    lines.append('|---|---|')
    for cand in CANDIDATES:
        lines.append(f"| {cand['label']} | {cand['description']} |")
    lines.append('')

    lines.append('## 2. Overall metrics by scale（越低越好）')
    lines.append('')
    lines.append('| scale | KF36 mean/med/max | Markov42 | Pure SCD | A | B | C | D |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for rec in summary['scale_records']:
        ov = rec['overall_by_method']
        def pack(name: str) -> str:
            x = ov[name]
            return f"{x['mean_pct_error']:.3f}/{x['median_pct_error']:.3f}/{x['max_pct_error']:.3f}"
        lines.append(
            f"| {fmt_scale_text(rec['noise_scale'])} | {pack('kf36_noisy')} | {pack('markov42_noisy')} | {pack('scd42_neutral')} | {pack('candidate_A')} | {pack('candidate_B')} | {pack('candidate_C')} | {pack('candidate_D')} |"
        )
    lines.append('')

    lines.append('## 3. Best candidate per scale（按 mean 选）')
    lines.append('')
    lines.append('| scale | best candidate by mean | vs Pure SCD mean rel% | vs Pure SCD median rel% | vs Pure SCD max rel% | better params / worse params |')
    lines.append('|---|---|---:|---:|---:|---:|')
    for rec in summary['scale_records']:
        best_name = rec['best_candidate_by_metric']['mean_pct_error']['candidate']
        pair = rec['candidate_vs_pure_scd'][best_name]
        lines.append(
            f"| {fmt_scale_text(rec['noise_scale'])} | {METHOD_DISPLAY[best_name]} | "
            f"{fmt_pct(pair['overall']['mean_pct_error']['relative_improvement_pct'])} | "
            f"{fmt_pct(pair['overall']['median_pct_error']['relative_improvement_pct'])} | "
            f"{fmt_pct(pair['overall']['max_pct_error']['relative_improvement_pct'])} | "
            f"{pair['params']['better_count']} / {pair['params']['worse_count']} |"
        )
    lines.append('')

    lines.append('## 4. Where does any SCD-only candidate beat pure SCD?')
    lines.append('')
    lines.append('| scale | beat pure SCD on mean | beat pure SCD on median | beat pure SCD on max |')
    lines.append('|---|---|---|---|')
    for rec in summary['scale_records']:
        def show(metric: str) -> str:
            vals = rec['beats_pure_scd'][metric]
            return ', '.join(vals) if vals else 'none'
        lines.append(
            f"| {fmt_scale_text(rec['noise_scale'])} | {show('mean_pct_error')} | {show('median_pct_error')} | {show('max_pct_error')} |"
        )
    lines.append('')

    lines.append('## 5. Direct answers')
    lines.append('')
    headline = summary['headline']
    lines.append(f"- **Does SCD-only LLM guidance beat pure SCD anywhere?** {headline['judgment']}")
    lines.append(
        '- **Metric-level answer**: '
        f"mean={headline['any_candidate_beats_pure_scd']['mean_pct_error']}, "
        f"median={headline['any_candidate_beats_pure_scd']['median_pct_error']}, "
        f"max={headline['any_candidate_beats_pure_scd']['max_pct_error']}."
    )
    lines.append(
        '- **Scientifically meaningful?** '
        + (
            'Yes across scales.' if headline['cross_scale_robust']
            else 'Only as a local low-noise signal; not yet a cross-scale robust scientific improvement.' if headline['local_signal']
            else 'No; current wins are too fragmented or absent.'
        )
    )
    lines.append(
        '- **Should the clean SCD+Markov+LLM line continue?** '
        + (
            'Yes, as a main clean line.' if headline['cross_scale_robust']
            else 'Yes, but only as a narrow follow-up around the winning low-noise pattern (not as a promoted general method yet).' if headline['continue_narrowly']
            else 'Not as-is; this exact SCD-only batch is more a negative/weak probe than a promotable improvement.'
        )
    )
    lines.append('')

    lines.append('## 6. Output files')
    lines.append('')
    lines.append(f"- candidate json: `{summary['candidate_json']}`")
    lines.append(f"- summary json: `{summary['summary_json']}`")
    lines.append(f"- report md: `{summary['report_md']}`")
    lines.append('')
    for rec in summary['scale_records']:
        lines.append(f"- scale {fmt_scale_text(rec['noise_scale'])} candidate jsons: `{json.dumps(rec['files']['candidate_jsons'], ensure_ascii=False)}`")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    candidate_dump = {
        'experiment': 'clean_scd_only_llm_guidance',
        'constraint_note': 'LLM is allowed to modify only SCD on top of neutral Markov+SCD baseline. No state_alpha_mult/state_prior_diag_mult/state_q_* changes. No post_rx_y_mult/post_ry_z_mult.',
        'baseline': 'neutral_markov42_plus_scd_baseline',
        'scales': SCALES,
        'candidates': CANDIDATES,
    }
    candidate_path = candidate_json_path(args.report_date)
    candidate_path.write_text(json.dumps(candidate_dump, ensure_ascii=False, indent=2), encoding='utf-8')

    scale_records = []
    for scale in SCALES:
        compare4 = ensure_four_method_compare(scale, args.report_date)
        candidate_payloads = {}
        candidate_exec = {}
        for cand in CANDIDATES:
            payload, exec_status = load_or_run_candidate(scale, cand, force_rerun=args.force_rerun)
            candidate_payloads[cand['name']] = payload
            candidate_exec[cand['name']] = exec_status
        scale_records.append(build_scale_record(scale, compare4, candidate_payloads, candidate_exec))

    headline = build_headline(scale_records)
    summary = {
        'report_date': args.report_date,
        'experiment': 'clean_scd_only_llm_guidance',
        'scales': SCALES,
        'candidate_json': str(candidate_path),
        'summary_json': str(summary_json_path(args.report_date)),
        'report_md': str(report_md_path(args.report_date)),
        'methods': METHOD_DISPLAY,
        'scale_records': scale_records,
        'headline': headline,
        'best_candidate_per_scale_by_mean': [
            {
                'scale': rec['noise_scale'],
                'candidate': rec['best_candidate_by_metric']['mean_pct_error']['candidate'],
                'overall': rec['overall_by_method'][rec['best_candidate_by_metric']['mean_pct_error']['candidate']],
                'vs_pure_scd_overall': rec['candidate_vs_pure_scd'][rec['best_candidate_by_metric']['mean_pct_error']['candidate']]['overall'],
            }
            for rec in scale_records
        ],
    }

    summary_path = summary_json_path(args.report_date)
    report_path = report_md_path(args.report_date)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_path.write_text(render_report(summary), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps({
        'candidate_json': str(candidate_path),
        'summary_json': str(summary_path),
        'report_md': str(report_path),
        'best_candidate_per_scale_by_mean': summary['best_candidate_per_scale_by_mean'],
        'headline': headline,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
