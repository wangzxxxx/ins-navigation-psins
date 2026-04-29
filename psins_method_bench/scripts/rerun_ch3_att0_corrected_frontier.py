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

from benchmark_ch3_12pos_goalA_repairs import rows_to_paras
from common_markov import load_module
from compare_ch3_12pos_path_baselines import build_ch3_initial_attitude, build_dataset_with_path
from compare_four_methods_shared_noise import _load_json, compute_payload, expected_noise_config
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs
from search_ch3_12pos_closedloop_local_insertions import build_closedloop_candidate, run_candidate_payload
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate, make_suffix
from search_ch3_anchor5_farz_followup import candidate_specs as anchor5_candidate_specs
from search_ch3_entry_conditioned_relay_family import candidate_specs as entryrelay_candidate_specs

NOISE_SCALE = 0.08
REPORT_DATE = datetime.now().strftime('%Y-%m-%d')
ATT0_DEG = [0.0, 0.0, 0.0]
SEED_INFO = expected_noise_config(NOISE_SCALE)

PATCH_TARGETS = {
    'direct_builder_fixes': [
        'psins_method_bench/scripts/compare_ch3_12pos_path_baselines.py',
        'psins_method_bench/scripts/compare_ch3_12pos_path_bridge.py',
        'psins_method_bench/scripts/compare_ch3_current_vs_faithful_whiteonly.py',
        'psins_method_bench/scripts/diagnose_ch3_12pos_repair.py',
    ],
    'transitive_search_dependents_via_compare_ch3_12pos_path_baselines': [
        'psins_method_bench/scripts/search_ch3_12pos_append_tail.py',
        'psins_method_bench/scripts/search_ch3_12pos_closedloop_local_insertions.py',
        'psins_method_bench/scripts/search_ch3_12pos_legal_dualaxis_excitation_overlay.py',
        'psins_method_bench/scripts/search_ch3_12pos_legal_dualaxis_repairs.py',
        'psins_method_bench/scripts/search_ch3_12pos_legal_dualaxis_second_layer.py',
        'psins_method_bench/scripts/benchmark_ch3_12pos_coupled_repairs.py',
        'psins_method_bench/scripts/benchmark_ch3_12pos_goalA_repairs.py',
        'psins_method_bench/scripts/probe_ch3_12pos_coupled_repair_search.py',
    ],
}

CANDIDATE_REGISTRY = {
    'faithful12': {
        'display_name': 'faithful12 baseline',
        'type': 'base',
        'canonical_name': 'faithful12',
        'markov_path': RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json',
        'kf_path': RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json',
    },
    'anchor5_mainline': {
        'display_name': 'anchor5 far-z unified mainline',
        'type': 'closedloop',
        'spec_name': 'zseed_l5_neg6_plus_relaymax_unified_l9y2p5',
        'canonical_name': 'zseed_l5_neg6_plus_relaymax_unified_l9y2p5',
        'markov_path': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_zseed_l5_neg6_plus_relaymax_unified_l9y2p5_shared_noise0p08_param_errors.json',
        'kf_path': RESULTS_DIR / 'KF36_ch3closedloop_zseed_l5_neg6_plus_relaymax_unified_l9y2p5_shared_noise0p08_param_errors.json',
    },
    'entry_frontier': {
        'display_name': 'entry-conditioned relay frontier',
        'type': 'closedloop',
        'spec_name': 'entryrelay_l8x1_l9y1_unifiedcore',
        'canonical_name': 'entryrelay_l8x1_l9y1_unifiedcore',
        'markov_path': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json',
        'kf_path': RESULTS_DIR / 'KF36_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json',
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--report-date', default=REPORT_DATE)
    return parser.parse_args()


def load_existing(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def overall_triplet(payload: dict[str, Any]) -> str:
    o = payload['overall']
    return f"{o['mean_pct_error']:.3f} / {o['median_pct_error']:.3f} / {o['max_pct_error']:.3f}"


def compact_metrics(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if payload is None:
        return None
    return {
        'overall': {
            'mean_pct_error': float(payload['overall']['mean_pct_error']),
            'median_pct_error': float(payload['overall']['median_pct_error']),
            'max_pct_error': float(payload['overall']['max_pct_error']),
        },
        'key_param_errors': {
            'dKa_yy': float(payload['param_errors']['dKa_yy']['pct_error']),
            'dKg_zz': float(payload['param_errors']['dKg_zz']['pct_error']),
            'Ka2_y': float(payload['param_errors']['Ka2_y']['pct_error']),
            'Ka2_z': float(payload['param_errors']['Ka2_z']['pct_error']),
        },
    }


def delta_overall(old_payload: dict[str, Any] | None, new_payload: dict[str, Any]) -> dict[str, float] | None:
    if old_payload is None:
        return None
    out = {}
    for key in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        out[key] = float(new_payload['overall'][key]) - float(old_payload['overall'][key])
    return out


def direct_gap(a_payload: dict[str, Any], b_payload: dict[str, Any]) -> dict[str, float]:
    out = {}
    for key in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        out[key] = float(a_payload['overall'][key]) - float(b_payload['overall'][key])
    return out


def ranking_order(results: dict[str, dict[str, Any]], metric: str) -> list[str]:
    return sorted(results.keys(), key=lambda k: float(results[k]['overall'][metric]))


def rerun_faithful_payload(mod, faithful, method_key: str, out_path: Path) -> dict[str, Any]:
    paras = rows_to_paras(mod, faithful.rows)
    dataset = build_dataset_with_path(mod, NOISE_SCALE, paras)
    params = _param_specs(mod)
    suffix = make_suffix(NOISE_SCALE)

    if method_key == 'markov42_noisy':
        clbt, _, _, _, _ = mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=42,
            bi_g=dataset['bi_g'],
            tau_g=dataset['tau_g'],
            bi_a=dataset['bi_a'],
            tau_a=dataset['tau_a'],
            label=f'faithful12_markov42_att0zero_{suffix}',
        )
        payload = compute_payload(
            mod,
            clbt,
            params,
            variant='42state_gm1_ch3faithful12_shared_noise0p08',
            method_file='source_mod.run_calibration(n_states=42) on chapter3/custom path',
            extra={
                'candidate_name': 'faithful12',
                'noise_scale': NOISE_SCALE,
                'noise_config': dataset['noise_config'],
                'att0_deg': ATT0_DEG,
                'comparison_mode': 'att0_corrected_rerun',
            },
        )
    elif method_key == 'kf36_noisy':
        clbt, _, _, _, _ = mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=36,
            label=f'faithful12_kf36_att0zero_{suffix}',
        )
        payload = compute_payload(
            mod,
            clbt,
            params,
            variant='kf36_ch3faithful12_shared_noise0p08',
            method_file='source_mod.run_calibration(n_states=36) on chapter3/custom path',
            extra={
                'candidate_name': 'faithful12',
                'noise_scale': NOISE_SCALE,
                'noise_config': dataset['noise_config'],
                'att0_deg': ATT0_DEG,
                'comparison_mode': 'att0_corrected_rerun',
            },
        )
    else:
        raise KeyError(method_key)

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload


def attach_att0(path: Path, payload: dict[str, Any], candidate_key: str, method_key: str) -> dict[str, Any]:
    extra = payload.setdefault('extra', {})
    extra['att0_deg'] = ATT0_DEG
    extra['comparison_mode'] = 'att0_corrected_rerun'
    extra['candidate_registry_key'] = candidate_key
    extra['method_key'] = method_key
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload


def build_closedloop_from_spec(mod, faithful, spec_name: str, source: str):
    if source == 'anchor5':
        spec = next(item for item in anchor5_candidate_specs() if item['name'] == spec_name)
    elif source == 'entry':
        spec = next(item for item in entryrelay_candidate_specs() if item['name'] == spec_name)
    else:
        raise KeyError(source)
    return build_closedloop_candidate(mod, spec, faithful.rows, faithful.action_sequence)


def main() -> None:
    args = parse_args()
    mod = load_module(str(METHOD_DIR / 'method_42state_gm1.py'), str(SOURCE_FILE))
    faithful = build_candidate(mod, ())
    anchor5_candidate = build_closedloop_from_spec(mod, faithful, CANDIDATE_REGISTRY['anchor5_mainline']['spec_name'], 'anchor5')
    entry_candidate = build_closedloop_from_spec(mod, faithful, CANDIDATE_REGISTRY['entry_frontier']['spec_name'], 'entry')

    old_results = {}
    for key, info in CANDIDATE_REGISTRY.items():
        old_results[key] = {
            'markov42': load_existing(info['markov_path']),
            'kf36': load_existing(info['kf_path']),
        }

    new_results: dict[str, dict[str, Any]] = {
        'faithful12': {},
        'anchor5_mainline': {},
        'entry_frontier': {},
    }

    new_results['faithful12']['markov42'] = rerun_faithful_payload(mod, faithful, 'markov42_noisy', CANDIDATE_REGISTRY['faithful12']['markov_path'])
    new_results['faithful12']['kf36'] = rerun_faithful_payload(mod, faithful, 'kf36_noisy', CANDIDATE_REGISTRY['faithful12']['kf_path'])

    payload, _, path = run_candidate_payload(mod, anchor5_candidate, 'markov42_noisy', NOISE_SCALE, force_rerun=True)
    new_results['anchor5_mainline']['markov42'] = attach_att0(path, payload, 'anchor5_mainline', 'markov42_noisy')
    payload, _, path = run_candidate_payload(mod, anchor5_candidate, 'kf36_noisy', NOISE_SCALE, force_rerun=True)
    new_results['anchor5_mainline']['kf36'] = attach_att0(path, payload, 'anchor5_mainline', 'kf36_noisy')

    payload, _, path = run_candidate_payload(mod, entry_candidate, 'markov42_noisy', NOISE_SCALE, force_rerun=True)
    new_results['entry_frontier']['markov42'] = attach_att0(path, payload, 'entry_frontier', 'markov42_noisy')
    payload, _, path = run_candidate_payload(mod, entry_candidate, 'kf36_noisy', NOISE_SCALE, force_rerun=True)
    new_results['entry_frontier']['kf36'] = attach_att0(path, payload, 'entry_frontier', 'kf36_noisy')

    old_markov = {k: old_results[k]['markov42'] for k in new_results if old_results[k]['markov42'] is not None}
    new_markov = {k: new_results[k]['markov42'] for k in new_results}
    old_kf = {k: old_results[k]['kf36'] for k in new_results if old_results[k]['kf36'] is not None}
    new_kf = {k: new_results[k]['kf36'] for k in new_results}

    old_markov_mean_order = ranking_order(old_markov, 'mean_pct_error') if len(old_markov) == len(new_markov) else None
    old_markov_max_order = ranking_order(old_markov, 'max_pct_error') if len(old_markov) == len(new_markov) else None
    new_markov_mean_order = ranking_order(new_markov, 'mean_pct_error')
    new_markov_max_order = ranking_order(new_markov, 'max_pct_error')
    old_kf_mean_order = ranking_order(old_kf, 'mean_pct_error') if len(old_kf) == len(new_kf) else None
    old_kf_max_order = ranking_order(old_kf, 'max_pct_error') if len(old_kf) == len(new_kf) else None
    new_kf_mean_order = ranking_order(new_kf, 'mean_pct_error')
    new_kf_max_order = ranking_order(new_kf, 'max_pct_error')

    stable_markov_mainline = new_markov_mean_order[0] == 'anchor5_mainline'
    stable_markov_frontier = new_markov_max_order[0] == 'entry_frontier'
    stable_kf_mainline = new_kf_mean_order[0] == 'anchor5_mainline'
    stable_kf_frontier = new_kf_max_order[0] == 'entry_frontier'

    material_change = (
        old_markov_mean_order != new_markov_mean_order
        or old_markov_max_order != new_markov_max_order
        or old_kf_mean_order != new_kf_mean_order
        or old_kf_max_order != new_kf_max_order
    )

    markov_gap_old = None
    kf_gap_old = None
    if old_results['anchor5_mainline']['markov42'] and old_results['entry_frontier']['markov42']:
        markov_gap_old = direct_gap(old_results['anchor5_mainline']['markov42'], old_results['entry_frontier']['markov42'])
    if old_results['anchor5_mainline']['kf36'] and old_results['entry_frontier']['kf36']:
        kf_gap_old = direct_gap(old_results['anchor5_mainline']['kf36'], old_results['entry_frontier']['kf36'])
    markov_gap_new = direct_gap(new_results['anchor5_mainline']['markov42'], new_results['entry_frontier']['markov42'])
    kf_gap_new = direct_gap(new_results['anchor5_mainline']['kf36'], new_results['entry_frontier']['kf36'])

    summary = {
        'task': 'chapter-3 att0 correction baseline check',
        'report_date': args.report_date,
        'noise_scale': NOISE_SCALE,
        'shared_noise_config': SEED_INFO,
        'corrected_att0_deg': ATT0_DEG,
        'patch_targets': PATCH_TARGETS,
        'candidate_set': {
            key: {
                'display_name': info['display_name'],
                'canonical_name': info['canonical_name'],
                'updated_result_files': {
                    'markov42': str(info['markov_path']),
                    'kf36': str(info['kf_path']),
                },
            }
            for key, info in CANDIDATE_REGISTRY.items()
        },
        'previous_results': {
            key: {
                'markov42': compact_metrics(old_results[key]['markov42']),
                'kf36': compact_metrics(old_results[key]['kf36']),
            }
            for key in CANDIDATE_REGISTRY
        },
        'corrected_results': {
            key: {
                'markov42': compact_metrics(new_results[key]['markov42']),
                'kf36': compact_metrics(new_results[key]['kf36']),
                'delta_vs_previous_markov42': delta_overall(old_results[key]['markov42'], new_results[key]['markov42']),
                'delta_vs_previous_kf36': delta_overall(old_results[key]['kf36'], new_results[key]['kf36']),
            }
            for key in CANDIDATE_REGISTRY
        },
        'ranking': {
            'markov42': {
                'previous_mean_order': old_markov_mean_order,
                'previous_max_order': old_markov_max_order,
                'corrected_mean_order': new_markov_mean_order,
                'corrected_max_order': new_markov_max_order,
                'anchor5_minus_entry_previous': markov_gap_old,
                'anchor5_minus_entry_corrected': markov_gap_new,
            },
            'kf36': {
                'previous_mean_order': old_kf_mean_order,
                'previous_max_order': old_kf_max_order,
                'corrected_mean_order': new_kf_mean_order,
                'corrected_max_order': new_kf_max_order,
                'anchor5_minus_entry_previous': kf_gap_old,
                'anchor5_minus_entry_corrected': kf_gap_new,
            },
        },
        'stability_read': {
            'material_ranking_change': material_change,
            'markov42_anchor5_still_best_mean': stable_markov_mainline,
            'markov42_entry_still_best_max': stable_markov_frontier,
            'kf36_anchor5_still_best_mean': stable_kf_mainline,
            'kf36_entry_still_best_max': stable_kf_frontier,
        },
    }

    if material_change:
        summary['bottom_line'] = 'Corrected att0=(0,0,0) materially changes at least one saved ordering; prior frontier conclusions must be revised against the new ranking fields above.'
    else:
        summary['bottom_line'] = 'Corrected att0=(0,0,0) does not materially change the saved winner identities: anchor5 remains the mean/mainline winner and entryrelay remains the max-frontier winner in both Markov42 and KF36.'

    report_lines = []
    report_lines.append('# Chapter-3 att0 correction baseline check')
    report_lines.append('')
    report_lines.append('## 1. Correction applied')
    report_lines.append('')
    report_lines.append('- Hard fact enforced: **chapter-3 twelve-position calibration starts from `att0 = (0, 0, 0)`**.')
    report_lines.append(f'- Fixed shared path-search builder to use `att0 = {ATT0_DEG}` instead of the old `[1, -91, -91] deg`.')
    report_lines.append(f'- Rechecked the minimal incumbent frontier under the same low-noise setup: `noise_scale = {NOISE_SCALE}`, seed = `{SEED_INFO["seed"]}`, truth-noise family unchanged apart from the corrected initial attitude.')
    report_lines.append('')
    report_lines.append('## 2. Code targets corrected')
    report_lines.append('')
    report_lines.append('- **Direct builder / duplicate fixes**')
    for item in PATCH_TARGETS['direct_builder_fixes']:
        report_lines.append(f'  - `{item}`')
    report_lines.append('- **Search / benchmark scripts now corrected transitively through the shared builder**')
    for item in PATCH_TARGETS['transitive_search_dependents_via_compare_ch3_12pos_path_baselines']:
        report_lines.append(f'  - `{item}`')
    report_lines.append('')
    report_lines.append('## 3. Corrected frontier comparison')
    report_lines.append('')
    report_lines.append('| candidate | Markov42 corrected mean/median/max | KF36 corrected mean/median/max | Markov42 Δvs old | KF36 Δvs old |')
    report_lines.append('|---|---:|---:|---:|---:|')
    for key, info in CANDIDATE_REGISTRY.items():
        new_m = new_results[key]['markov42']
        new_k = new_results[key]['kf36']
        dm = delta_overall(old_results[key]['markov42'], new_m)
        dk = delta_overall(old_results[key]['kf36'], new_k)
        dm_text = 'n/a' if dm is None else f"{dm['mean_pct_error']:+.3f} / {dm['median_pct_error']:+.3f} / {dm['max_pct_error']:+.3f}"
        dk_text = 'n/a' if dk is None else f"{dk['mean_pct_error']:+.3f} / {dk['median_pct_error']:+.3f} / {dk['max_pct_error']:+.3f}"
        report_lines.append(
            f"| {info['display_name']} | {overall_triplet(new_m)} | {overall_triplet(new_k)} | {dm_text} | {dk_text} |"
        )
    report_lines.append('')
    report_lines.append('## 4. Ranking before vs after correction')
    report_lines.append('')
    report_lines.append(f"- **Markov42 mean order**: `{old_markov_mean_order}` → `{new_markov_mean_order}`")
    report_lines.append(f"- **Markov42 max order**: `{old_markov_max_order}` → `{new_markov_max_order}`")
    report_lines.append(f"- **KF36 mean order**: `{old_kf_mean_order}` → `{new_kf_mean_order}`")
    report_lines.append(f"- **KF36 max order**: `{old_kf_max_order}` → `{new_kf_max_order}`")
    report_lines.append('')
    report_lines.append('- **Anchor5 minus entry (negative is better for anchor5, because error is lower)**')
    if markov_gap_old is not None:
        report_lines.append(
            f"  - Markov42 old: mean {markov_gap_old['mean_pct_error']:+.3f}, median {markov_gap_old['median_pct_error']:+.3f}, max {markov_gap_old['max_pct_error']:+.3f}"
        )
    report_lines.append(
        f"  - Markov42 corrected: mean {markov_gap_new['mean_pct_error']:+.3f}, median {markov_gap_new['median_pct_error']:+.3f}, max {markov_gap_new['max_pct_error']:+.3f}"
    )
    if kf_gap_old is not None:
        report_lines.append(
            f"  - KF36 old: mean {kf_gap_old['mean_pct_error']:+.3f}, median {kf_gap_old['median_pct_error']:+.3f}, max {kf_gap_old['max_pct_error']:+.3f}"
        )
    report_lines.append(
        f"  - KF36 corrected: mean {kf_gap_new['mean_pct_error']:+.3f}, median {kf_gap_new['median_pct_error']:+.3f}, max {kf_gap_new['max_pct_error']:+.3f}"
    )
    report_lines.append('')
    report_lines.append('## 5. Stable conclusions vs revisions')
    report_lines.append('')
    report_lines.append(f"- **Stable**: anchor5 far-z family remains the corrected unified mainline winner on mean? **{'YES' if stable_markov_mainline and stable_kf_mainline else 'NO'}**")
    report_lines.append(f"- **Stable**: entry-conditioned relay remains the corrected absolute max-frontier point? **{'YES' if stable_markov_frontier and stable_kf_frontier else 'NO'}**")
    report_lines.append(f"- **Material ranking change overall?** **{'YES' if material_change else 'NO'}**")
    if material_change:
        report_lines.append('- **Revision required**: at least one previously saved ordering flips under the corrected att0. Use the corrected ranking section above as the new baseline before any hidden-family continuation.')
    else:
        report_lines.append('- **Revision required**: the incumbent winner identities do **not** change, but all future comparisons must now treat these corrected result files as the baseline; the old att0-based absolute numbers are obsolete.')
    report_lines.append('')
    report_lines.append('## 6. Updated result files')
    report_lines.append('')
    for key, info in CANDIDATE_REGISTRY.items():
        report_lines.append(f"- **{info['display_name']}**")
        report_lines.append(f"  - Markov42: `{info['markov_path']}`")
        report_lines.append(f"  - KF36: `{info['kf_path']}`")
    report_lines.append('')
    report_lines.append(f"- JSON summary: `{RESULTS_DIR / f'ch3_att0_corrected_frontier_{args.report_date}.json'}`")
    report_lines.append(f"- Markdown report: `{REPORTS_DIR / f'psins_ch3_att0_corrected_frontier_{args.report_date}.md'}`")

    report_path = REPORTS_DIR / f'psins_ch3_att0_corrected_frontier_{args.report_date}.md'
    summary_path = RESULTS_DIR / f'ch3_att0_corrected_frontier_{args.report_date}.json'
    report_path.write_text('\n'.join(report_lines) + '\n', encoding='utf-8')
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({
        'summary_path': str(summary_path),
        'report_path': str(report_path),
        'bottom_line': summary['bottom_line'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
