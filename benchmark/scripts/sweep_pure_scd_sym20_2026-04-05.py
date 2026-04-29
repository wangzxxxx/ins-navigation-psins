from __future__ import annotations

import argparse
import copy
import json
import re
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
TMP_PSINS_DIR = ROOT / 'tmp_psins_py'
SOURCE_FILE = TMP_PSINS_DIR / 'psins_py' / 'test_calibration_markov_pruned.py'
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'

COMPARE_CH3_FILE = SCRIPTS_DIR / 'compare_ch3_corrected_symmetric20_vs_legacy19pos_1200s.py'
COMPARE_SHARED_FILE = SCRIPTS_DIR / 'compare_four_methods_shared_noise.py'
COMPUTE_R61_FILE = SCRIPTS_DIR / 'compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors.py'
PROBE_R55_FILE = SCRIPTS_DIR / 'probe_round55_newline.py'
PROBE_R59_FILE = SCRIPTS_DIR / 'probe_round59_h_scd_hybrid.py'

G3_BASELINE_JSON = RESULTS_DIR / 'M_markov_42state_gm1_ch3corrected_symmetric20_att0zero_1200s_shared_noise0p12_param_errors.json'
PURE_NEUTRAL_JSON = RESULTS_DIR / 'G4_pure_scd_neutral_ch3corrected_symmetric20_att0zero_1200s_shared_noise0p12_param_errors.json'
PRIOR_BEST_G4_JSON = RESULTS_DIR / 'G4_sym20_retune_scd_scale_once_a0999_biaslink_commit_shared_noise0p12_param_errors.json'

COMPARISON_MODE = 'g4_pure_scd_sym20_sweep_2026_04_05'

ALLOWED_TARGETS = ['selected', 'scale_block']
ALLOWED_ALPHAS = [0.9998, 0.9995, 0.999, 0.998]
ALLOWED_TDS = [1.0, 2.0, 4.0]
ALLOWED_MODES = ['once_per_phase', 'repeat_after_transition']
ALLOWED_BIAS = [True, False]

for p in [ROOT, TMP_PSINS_DIR, METHOD_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=0.12)
    parser.add_argument('--report-date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def _overall_triplet(payload: dict[str, Any]) -> dict[str, float]:
    ov = payload['overall']
    return {
        'mean_pct_error': float(ov['mean_pct_error']),
        'median_pct_error': float(ov['median_pct_error']),
        'max_pct_error': float(ov['max_pct_error']),
    }


def _sanitize_name(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_]+', '_', text)


def _alpha_tag(alpha: float) -> str:
    return f"a{int(round(alpha * 10000)):05d}"


def _td_tag(td: float) -> str:
    if abs(td - round(td)) < 1e-12:
        return f"td{int(round(td))}"
    return f"td{str(td).replace('.', 'p')}"


def _make_config_name(*, phase: str, mode: str, target: str, alpha: float, td: float, bias_to_target: bool, parent_name: str | None = None) -> str:
    target_tag = 'sel' if target == 'selected' else 'scb'
    mode_tag = 'once' if mode == 'once_per_phase' else 'repeat'
    bias_tag = 'b1' if bias_to_target else 'b0'
    core = f"{phase}_{target_tag}_{mode_tag}_{_alpha_tag(alpha)}_{_td_tag(td)}_{bias_tag}"
    if parent_name:
        return f"{core}_from_{_sanitize_name(parent_name)}"
    return core


def _config_signature(config: dict[str, Any]) -> str:
    keep = {
        'name': config['name'],
        'phase': config['phase'],
        'parent': config.get('parent'),
        'scd': config['scd'],
    }
    return json.dumps(keep, ensure_ascii=False, sort_keys=True)


def _build_config(*, phase: str, mode: str, target: str, alpha: float, td: float, bias_to_target: bool, parent: str | None = None) -> dict[str, Any]:
    if target not in ALLOWED_TARGETS:
        raise ValueError(f'Illegal target: {target}')
    if alpha not in ALLOWED_ALPHAS:
        raise ValueError(f'Illegal alpha: {alpha}')
    if td not in ALLOWED_TDS:
        raise ValueError(f'Illegal transition_duration: {td}')
    if mode not in ALLOWED_MODES:
        raise ValueError(f'Illegal mode: {mode}')
    if bias_to_target not in ALLOWED_BIAS:
        raise ValueError(f'Illegal bias_to_target: {bias_to_target}')

    name = _make_config_name(
        phase=phase,
        mode=mode,
        target=target,
        alpha=alpha,
        td=td,
        bias_to_target=bias_to_target,
        parent_name=parent,
    )

    scd = {
        'mode': mode,
        'alpha': float(alpha),
        'transition_duration': float(td),
        'target': target,
        'bias_to_target': bool(bias_to_target),
        'apply_policy_names': ['iter2_commit'],
    }

    return {
        'name': name,
        'phase': phase,
        'parent': parent,
        'description': (
            f"{phase}: target={target}, mode={mode}, alpha={alpha}, transition_duration={td}, "
            f"bias_to_target={bias_to_target}, apply_policy_names=['iter2_commit']"
        ),
        'scd': scd,
    }


def _build_phase1_configs() -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    for target in ALLOWED_TARGETS:
        for alpha in ALLOWED_ALPHAS:
            for td in ALLOWED_TDS:
                configs.append(_build_config(
                    phase='phase1',
                    mode='once_per_phase',
                    target=target,
                    alpha=alpha,
                    td=td,
                    bias_to_target=True,
                ))
    return configs


def _build_phase2_configs(top4: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in top4:
        scd = row['config']['scd']
        parent = row['config']['name']

        out.append(_build_config(
            phase='phase2_bias_false',
            mode=str(scd['mode']),
            target=str(scd['target']),
            alpha=float(scd['alpha']),
            td=float(scd['transition_duration']),
            bias_to_target=False,
            parent=parent,
        ))

        if scd['mode'] == 'once_per_phase':
            out.append(_build_config(
                phase='phase2_repeat_confirm',
                mode='repeat_after_transition',
                target=str(scd['target']),
                alpha=float(scd['alpha']),
                td=float(scd['transition_duration']),
                bias_to_target=bool(scd['bias_to_target']),
                parent=parent,
            ))

    return out[:8]


def _result_json_path(compare_shared_mod, config_name: str, noise_scale: float) -> Path:
    suffix = compare_shared_mod.make_suffix(noise_scale)
    return RESULTS_DIR / f'G4_pure_scd_sym20_sweep_{config_name}_shared_{suffix}_param_errors.json'


def _ranking_key(row: dict[str, Any]) -> tuple[float, float, float]:
    ov = row['overall']
    return (
        float(ov['mean_pct_error']),
        float(ov['max_pct_error']),
        float(ov['median_pct_error']),
    )


def _rank_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=_ranking_key)
    for idx, row in enumerate(ranked, start=1):
        row['rank'] = idx
    return ranked


def _compare_overall(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, Any]:
    metrics = {}
    wins = 0
    for key in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        delta = float(reference[key]) - float(candidate[key])
        better = delta > 0.0
        if better:
            wins += 1
        metrics[key] = {
            'candidate_value': float(candidate[key]),
            'reference_value': float(reference[key]),
            'improvement_pct_points': float(delta),
            'candidate_better': bool(better),
            'remaining_gap_pct_points': float(max(0.0, float(candidate[key]) - float(reference[key]))),
        }

    return {
        'wins_count': wins,
        'beats_all_three': wins == 3,
        'beats_mean': metrics['mean_pct_error']['candidate_better'],
        'beats_median': metrics['median_pct_error']['candidate_better'],
        'beats_max': metrics['max_pct_error']['candidate_better'],
        'metrics': metrics,
    }


def _run_one_config(
    *,
    idx: int,
    config: dict[str, Any],
    args: argparse.Namespace,
    compare_shared_mod,
    compare_ch3_mod,
    probe_r55_mod,
    probe_r59_mod,
    source_mod,
    case: dict[str, Any],
    dataset: dict[str, Any],
    params,
) -> dict[str, Any]:
    result_path = _result_json_path(compare_shared_mod, config['name'], args.noise_scale)
    expected_cfg = compare_shared_mod.expected_noise_config(args.noise_scale)
    signature = _config_signature(config)

    if (not args.force_rerun) and result_path.exists():
        old = _load_json(result_path)
        extra = old.get('extra', {}) if isinstance(old, dict) else {}
        if (
            compare_shared_mod._noise_matches(old, expected_cfg)
            and extra.get('comparison_mode') == COMPARISON_MODE
            and extra.get('path_case_tag') == case['case_tag']
            and extra.get('config_signature') == signature
        ):
            return {
                'config': copy.deepcopy(config),
                'overall': _overall_triplet(old),
                'result_json': str(result_path),
                'status': 'reused_verified',
            }

    method_mod = load_module(
        f"pure_scd_sym20_sweep_r53_{idx}_{_sanitize_name(config['name'])}",
        str(R53_METHOD_FILE),
    )
    neutral_candidate = compare_shared_mod._build_neutral_scd_candidate()
    neutral_candidate['name'] = config['name']
    neutral_candidate['description'] = config['description']
    neutral_candidate['scd'] = copy.deepcopy(config['scd'])

    method_mod = probe_r55_mod._build_patched_method(method_mod, neutral_candidate)
    scd_result = list(probe_r59_mod._run_internalized_hybrid_scd(
        method_mod,
        source_mod,
        dataset['imu_noisy'],
        dataset['pos0'],
        dataset['ts'],
        bi_g=dataset['bi_g'],
        bi_a=dataset['bi_a'],
        tau_g=dataset['tau_g'],
        tau_a=dataset['tau_a'],
        label=f"PURE-SCD-SWEEP-{idx:02d}-{_sanitize_name(config['name']).upper()}",
        scd_cfg=config['scd'],
    ))
    runtime = scd_result[4] if len(scd_result) >= 5 and isinstance(scd_result[4], dict) else {}

    payload = compare_shared_mod.compute_payload(
        source_mod,
        scd_result[0],
        params,
        variant=f"g4_pure_scd_sym20_sweep_{config['name']}_{compare_shared_mod.make_suffix(args.noise_scale)}",
        method_file='round53_base + _build_patched_method(neutral) + _run_internalized_hybrid_scd',
        extra={
            'comparison_mode': COMPARISON_MODE,
            'path_case_key': case['case_key'],
            'path_case_tag': case['case_tag'],
            'path_case_display_name': case['display_name'],
            'att0_deg': case['att0_deg'],
            'n_motion_rows': case['n_motion_rows'],
            'claimed_position_count': case['claimed_position_count'],
            'total_time_s': case['total_time_s'],
            'timing_note': case['timing_note'],
            'source_builder': case['source_builder'],
            'source_reference': case['source_reference'],
            'builder_method_tag': case.get('builder_method_tag'),
            'rationale': case['rationale'],
            'noise_scale': args.noise_scale,
            'noise_config': dataset['noise_config'],
            'config_name': config['name'],
            'config_phase': config['phase'],
            'config_parent': config.get('parent'),
            'config_signature': signature,
            'scd_cfg': copy.deepcopy(config['scd']),
            'iter_patches': copy.deepcopy(neutral_candidate['iter_patches']),
            'runtime_log': {
                'schedule_log': runtime.get('schedule_log'),
                'feedback_log': runtime.get('feedback_log'),
                'scd_log': runtime.get('scd_log'),
            },
            'baseline_g3_json': str(G3_BASELINE_JSON),
            'baseline_pure_neutral_json': str(PURE_NEUTRAL_JSON),
            'baseline_prior_best_g4_json': str(PRIOR_BEST_G4_JSON),
        },
    )
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')

    return {
        'config': copy.deepcopy(config),
        'overall': _overall_triplet(payload),
        'result_json': str(result_path),
        'status': 'rerun',
    }


def _assessment_label(best_vs_g3: dict[str, Any], best_vs_neutral: dict[str, Any]) -> dict[str, str]:
    if best_vs_g3['beats_all_three']:
        return {
            'label': 'meaningful_shift',
            'reason': 'Best pure-SCD config beats G3 on mean/median/max simultaneously.',
        }

    mean_gain = float(best_vs_neutral['metrics']['mean_pct_error']['improvement_pct_points'])
    if mean_gain >= 0.05 and (best_vs_g3['beats_mean'] or best_vs_g3['beats_median'] or best_vs_g3['beats_max']):
        return {
            'label': 'meaningful_shift',
            'reason': 'Best pure-SCD config shows a sizable gain over neutral and also beats G3 on at least one core metric.',
        }

    return {
        'label': 'small_nudge',
        'reason': 'No all-three G3 win and gain over neutral is limited; improvement is a local nudge rather than a robust shift.',
    }


def _render_report(summary: dict[str, Any]) -> str:
    g3 = summary['baselines']['g3_markov20']['overall']
    neutral = summary['baselines']['pure_scd_neutral']['overall']
    prior = summary['baselines']['prior_best_g4_retune']['overall']
    best = summary['best_config']

    def t(d: dict[str, float]) -> str:
        return f"{d['mean_pct_error']:.6f} / {d['median_pct_error']:.6f} / {d['max_pct_error']:.6f}"

    lines: list[str] = []
    lines.append('# G4 pure-SCD-only sweep on corrected symmetric20 (noise0p12, 42-state)')
    lines.append('')
    lines.append('## Setup')
    lines.append('')
    lines.append('- Path: corrected symmetric20 (20-position), att0=(0,0,0), total time fixed to 1200 s')
    lines.append(f"- noise_scale: {summary['noise_scale']}")
    lines.append('- Fixed family: neutral Markov42 + pure SCD-only knobs; no LLM micro-guard / no lever confirmation knobs')
    lines.append('- Ranking rule: mean → max → median (ascending)')
    lines.append('')

    lines.append('## Baselines (mean / median / max, % error)')
    lines.append('')
    lines.append(f"- G3 Markov@20: **{t(g3)}**")
    lines.append(f"- Pure SCD neutral: **{t(neutral)}**")
    lines.append(f"- Prior best G4 retune: **{t(prior)}**")
    lines.append('')

    lines.append('## Sweep coverage')
    lines.append('')
    lines.append(f"- Phase 1 coarse sweep: {summary['phase1']['count']} configs")
    lines.append(f"- Phase 2 refinement: {summary['phase2']['count']} configs (cap=8)")
    lines.append(f"- Total tried: {summary['total_tried_configs']} configs")
    lines.append('')

    lines.append('## Final ranking (all tried configs)')
    lines.append('')
    lines.append('| rank | config | phase | target | mode | alpha | td | bias | mean | median | max | Δmean vs G3 | Δmedian vs G3 | Δmax vs G3 |')
    lines.append('|---:|---|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|')
    for row in summary['ranking']:
        scd = row['config']['scd']
        d = row['delta_vs_g3']['metrics']
        lines.append(
            f"| {row['rank']} | {row['config']['name']} | {row['config']['phase']} | {scd['target']} | {scd['mode']} | "
            f"{scd['alpha']:.4f} | {scd['transition_duration']:.1f} | {scd['bias_to_target']} | "
            f"{row['overall']['mean_pct_error']:.6f} | {row['overall']['median_pct_error']:.6f} | {row['overall']['max_pct_error']:.6f} | "
            f"{d['mean_pct_error']['improvement_pct_points']:+.6f} | {d['median_pct_error']['improvement_pct_points']:+.6f} | {d['max_pct_error']['improvement_pct_points']:+.6f} |"
        )
    lines.append('')

    lines.append('## Conclusion')
    lines.append('')
    lines.append(f"- Best config: **{best['config']['name']}**")
    lines.append(
        f"  - SCD = target={best['config']['scd']['target']}, mode={best['config']['scd']['mode']}, "
        f"alpha={best['config']['scd']['alpha']}, td={best['config']['scd']['transition_duration']}, "
        f"bias_to_target={best['config']['scd']['bias_to_target']}"
    )
    lines.append(f"  - Overall: **{t(best['overall'])}**")
    lines.append(
        f"  - vs G3: Δmean={best['delta_vs_g3']['metrics']['mean_pct_error']['improvement_pct_points']:+.6f}, "
        f"Δmedian={best['delta_vs_g3']['metrics']['median_pct_error']['improvement_pct_points']:+.6f}, "
        f"Δmax={best['delta_vs_g3']['metrics']['max_pct_error']['improvement_pct_points']:+.6f}"
    )
    lines.append(
        f"  - vs pure neutral: Δmean={best['delta_vs_pure_neutral']['metrics']['mean_pct_error']['improvement_pct_points']:+.6f}, "
        f"Δmedian={best['delta_vs_pure_neutral']['metrics']['median_pct_error']['improvement_pct_points']:+.6f}, "
        f"Δmax={best['delta_vs_pure_neutral']['metrics']['max_pct_error']['improvement_pct_points']:+.6f}"
    )
    lines.append(
        f"- Any pure-SCD config beats G3? mean={summary['any_beats_g3']['mean']}, "
        f"median={summary['any_beats_g3']['median']}, max={summary['any_beats_g3']['max']}, "
        f"all_three={summary['any_beats_g3']['all_three']}"
    )
    if not summary['any_beats_g3']['all_three']:
        gap = summary['nearest_remaining_gap_to_g3']
        lines.append(
            f"- Nearest remaining gap to full G3 win: `{gap['candidate_name']}` still behind by "
            f"mean={gap['remaining_gap_pct_points']['mean_pct_error']:.6f}, "
            f"median={gap['remaining_gap_pct_points']['median_pct_error']:.6f}, "
            f"max={gap['remaining_gap_pct_points']['max_pct_error']:.6f}."
        )
    lines.append(f"- Shift assessment: **{summary['shift_assessment']['label']}** — {summary['shift_assessment']['reason']}")
    lines.append('')

    lines.append('## Artifact files')
    lines.append('')
    lines.append(f"- sweep script: `{summary['files']['script']}`")
    lines.append(f"- summary json: `{summary['files']['summary_json']}`")
    lines.append(f"- report md: `{summary['files']['report_md']}`")
    lines.append(f"- best result json: `{summary['files']['best_result_json']}`")
    lines.append('')

    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    compare_ch3_mod = load_module('pure_scd_sweep_compare_ch3_mod', str(COMPARE_CH3_FILE))
    compare_shared_mod = load_module('pure_scd_sweep_compare_shared_mod', str(COMPARE_SHARED_FILE))
    compute_r61_mod = load_module('pure_scd_sweep_compute_r61_mod', str(COMPUTE_R61_FILE))
    probe_r55_mod = load_module('pure_scd_sweep_probe_r55_mod', str(PROBE_R55_FILE))
    probe_r59_mod = load_module('pure_scd_sweep_probe_r59_mod', str(PROBE_R59_FILE))

    source_mod = load_module(
        f"pure_scd_sweep_source_{compare_shared_mod.make_suffix(args.noise_scale)}",
        str(SOURCE_FILE),
    )

    case = compare_ch3_mod.build_symmetric20_case(source_mod)
    dataset = compare_ch3_mod.build_dataset(source_mod, case['paras'], case['att0_deg'], args.noise_scale)
    params = compute_r61_mod._param_specs(source_mod)

    baseline_g3 = _load_json(G3_BASELINE_JSON)
    baseline_neutral = _load_json(PURE_NEUTRAL_JSON)
    baseline_prior = _load_json(PRIOR_BEST_G4_JSON)

    phase1_configs = _build_phase1_configs()
    phase1_rows: list[dict[str, Any]] = []
    for idx, cfg in enumerate(phase1_configs, start=1):
        phase1_rows.append(_run_one_config(
            idx=idx,
            config=cfg,
            args=args,
            compare_shared_mod=compare_shared_mod,
            compare_ch3_mod=compare_ch3_mod,
            probe_r55_mod=probe_r55_mod,
            probe_r59_mod=probe_r59_mod,
            source_mod=source_mod,
            case=case,
            dataset=dataset,
            params=params,
        ))

    phase1_ranked = _rank_rows(phase1_rows)
    top4 = phase1_ranked[:4]
    phase2_configs = _build_phase2_configs(top4)

    phase2_rows: list[dict[str, Any]] = []
    for local_idx, cfg in enumerate(phase2_configs, start=1):
        phase2_rows.append(_run_one_config(
            idx=100 + local_idx,
            config=cfg,
            args=args,
            compare_shared_mod=compare_shared_mod,
            compare_ch3_mod=compare_ch3_mod,
            probe_r55_mod=probe_r55_mod,
            probe_r59_mod=probe_r59_mod,
            source_mod=source_mod,
            case=case,
            dataset=dataset,
            params=params,
        ))

    all_rows = phase1_rows + phase2_rows
    ranked = _rank_rows(all_rows)

    g3_overall = _overall_triplet(baseline_g3)
    neutral_overall = _overall_triplet(baseline_neutral)
    prior_overall = _overall_triplet(baseline_prior)

    for row in ranked:
        row['delta_vs_g3'] = _compare_overall(row['overall'], g3_overall)
        row['delta_vs_pure_neutral'] = _compare_overall(row['overall'], neutral_overall)
        row['delta_vs_prior_best_g4'] = _compare_overall(row['overall'], prior_overall)

    best = copy.deepcopy(ranked[0])

    any_beats_g3 = {
        'mean': any(row['delta_vs_g3']['beats_mean'] for row in ranked),
        'median': any(row['delta_vs_g3']['beats_median'] for row in ranked),
        'max': any(row['delta_vs_g3']['beats_max'] for row in ranked),
        'all_three': any(row['delta_vs_g3']['beats_all_three'] for row in ranked),
    }

    def _gap_score(row: dict[str, Any]) -> tuple[float, float, float, float]:
        rem = {
            k: max(0.0, float(row['overall'][k]) - float(g3_overall[k]))
            for k in ['mean_pct_error', 'median_pct_error', 'max_pct_error']
        }
        return (
            rem['mean_pct_error'] + rem['median_pct_error'] + rem['max_pct_error'],
            rem['mean_pct_error'],
            rem['max_pct_error'],
            rem['median_pct_error'],
        )

    nearest_gap_row = sorted(ranked, key=_gap_score)[0]
    nearest_remaining_gap_to_g3 = {
        'candidate_name': nearest_gap_row['config']['name'],
        'rank': nearest_gap_row['rank'],
        'remaining_gap_pct_points': {
            k: max(0.0, float(nearest_gap_row['overall'][k]) - float(g3_overall[k]))
            for k in ['mean_pct_error', 'median_pct_error', 'max_pct_error']
        },
        'delta_vs_g3': nearest_gap_row['delta_vs_g3']['metrics'],
    }

    shift_assessment = _assessment_label(best['delta_vs_g3'], best['delta_vs_pure_neutral'])

    suffix = compare_shared_mod.make_suffix(args.noise_scale)
    summary_json = RESULTS_DIR / f'g4_pure_scd_sym20_sweep_{args.report_date}_{suffix}_summary.json'
    report_md = REPORTS_DIR / f'psins_g4_pure_scd_sym20_sweep_{args.report_date}_{suffix}.md'

    summary = {
        'experiment': 'g4_pure_scd_only_sym20_sweep_2026_04_05',
        'comparison_mode': COMPARISON_MODE,
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'noise_config': dataset['noise_config'],
        'ranking_rule': 'sort by mean_pct_error, then max_pct_error, then median_pct_error (ascending)',
        'setup': {
            'path_case_key': case['case_key'],
            'path_case_tag': case['case_tag'],
            'path_case_display_name': case['display_name'],
            'att0_deg': case['att0_deg'],
            'n_motion_rows': case['n_motion_rows'],
            'claimed_position_count': case['claimed_position_count'],
            'total_time_s': case['total_time_s'],
            'timing_note': case['timing_note'],
            'source_builder': case['source_builder'],
            'source_reference': case['source_reference'],
            'builder_method_tag': case.get('builder_method_tag'),
            'state_setting': 42,
            'fixed_constraints': {
                'llm_micro_guard_knobs': 'disabled',
                'lever_confirmation_knobs': 'disabled',
                'apply_policy_names_fixed': ['iter2_commit'],
            },
        },
        'baselines': {
            'g3_markov20': {
                'json_path': str(G3_BASELINE_JSON),
                'overall': g3_overall,
            },
            'pure_scd_neutral': {
                'json_path': str(PURE_NEUTRAL_JSON),
                'overall': neutral_overall,
            },
            'prior_best_g4_retune': {
                'json_path': str(PRIOR_BEST_G4_JSON),
                'overall': prior_overall,
            },
        },
        'phase1': {
            'count': len(phase1_rows),
            'grid': {
                'mode': ['once_per_phase'],
                'target': ALLOWED_TARGETS,
                'alpha': ALLOWED_ALPHAS,
                'transition_duration': ALLOWED_TDS,
                'bias_to_target': [True],
            },
            'top4_after_phase1': [
                {
                    'rank': row['rank'],
                    'config_name': row['config']['name'],
                    'scd': row['config']['scd'],
                    'overall': row['overall'],
                    'result_json': row['result_json'],
                    'status': row['status'],
                }
                for row in top4
            ],
        },
        'phase2': {
            'count': len(phase2_rows),
            'cap': 8,
            'configs': [
                {
                    'config_name': row['config']['name'],
                    'phase': row['config']['phase'],
                    'parent': row['config'].get('parent'),
                    'scd': row['config']['scd'],
                    'overall': row['overall'],
                    'result_json': row['result_json'],
                    'status': row['status'],
                }
                for row in phase2_rows
            ],
        },
        'total_tried_configs': len(ranked),
        'ranking': ranked,
        'best_config': best,
        'best_config_deltas': {
            'vs_g3': best['delta_vs_g3'],
            'vs_pure_scd_neutral': best['delta_vs_pure_neutral'],
            'vs_prior_best_g4': best['delta_vs_prior_best_g4'],
        },
        'any_beats_g3': any_beats_g3,
        'nearest_remaining_gap_to_g3': nearest_remaining_gap_to_g3,
        'shift_assessment': shift_assessment,
        'code_fixes_needed': 'none',
        'files': {
            'script': str(Path(__file__)),
            'summary_json': str(summary_json),
            'report_md': str(report_md),
            'best_result_json': best['result_json'],
        },
    }

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_md.write_text(_render_report(summary), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps({
        'summary_json': str(summary_json),
        'report_md': str(report_md),
        'best_result_json': best['result_json'],
        'best_config': {
            'name': best['config']['name'],
            'phase': best['config']['phase'],
            'scd': best['config']['scd'],
            'overall': best['overall'],
            'status': best['status'],
        },
        'any_beats_g3': any_beats_g3,
        'nearest_remaining_gap_to_g3': nearest_remaining_gap_to_g3,
        'shift_assessment': shift_assessment,
        'code_fixes_needed': summary['code_fixes_needed'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
