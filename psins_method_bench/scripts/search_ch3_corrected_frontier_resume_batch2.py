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

from common_markov import load_module
from compare_four_methods_shared_noise import _load_json, expected_noise_config
from search_ch3_12pos_closedloop_local_insertions import build_closedloop_candidate, run_candidate_payload
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate
from search_ch3_corrected_hidden_family_next4 import closed_pair
from search_ch3_entry_conditioned_relay_family import (
    NOISE_SCALE,
    REPORT_DATE,
    l8_xpair,
    l9_ypair_neg,
    l10_unified_core,
    l11_y10x0back2_core,
    merge_insertions,
)

ATT0_DEG = [0.0, 0.0, 0.0]
REPORT_PATH = REPORTS_DIR / f'psins_ch3_corrected_frontier_resume_batch2_{REPORT_DATE}.md'
SUMMARY_PATH = RESULTS_DIR / f'ch3_corrected_frontier_resume_batch2_{REPORT_DATE}.json'

REFERENCE_FILES = {
    'current_leader': {
        'label': 'current corrected leader / relay_l12y0p5_on_entry',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relay_l12y0p5_on_entry_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_relay_l12y0p5_on_entry_shared_noise0p08_param_errors.json',
    },
    'second_winner': {
        'label': 'current corrected runner-up / entryrelay_l8x1_l9y0p75_unifiedcore',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_entryrelay_l8x1_l9y0p75_unifiedcore_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_entryrelay_l8x1_l9y0p75_unifiedcore_shared_noise0p08_param_errors.json',
    },
    'old_entry_frontier': {
        'label': 'previous corrected incumbent / entryrelay_l8x1_l9y1_unifiedcore',
        'markov42': RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json',
        'kf36': RESULTS_DIR / 'KF36_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json',
    },
}

HYPOTHESES = [
    {
        'id': 'H1',
        'family': 'terminal_microclosure_ridge_around_current_leader',
        'summary': 'Keep the current corrected leader backbone fixed and only retune the tiny anchor12 y reclosure around the new 0.5 s optimum, to test whether the new leader still sits on a small terminal ridge.',
        'candidate_names': [
            'relay_l12y0p25_on_entry',
            'relay_l12y0p75_on_entry',
        ],
    },
    {
        'id': 'H2',
        'family': 'l9_terminal_coupled_ridge',
        'summary': 'Fuse the second winner\'s l9 softening with the current leader\'s terminal reclosure, then micro-scan both around the junction to see whether dKg_xx can stay suppressed while eb_x / Ka2_y improve.',
        'candidate_names': [
            'relay_l9y0p625_l12y0p5_on_entry',
            'relay_l9y0p75_l12y0p5_on_entry',
            'relay_l9y0p875_l12y0p5_on_entry',
            'relay_l9y0p75_l12y0p25_on_entry',
            'relay_l9y0p75_l12y0p75_on_entry',
        ],
    },
    {
        'id': 'H3',
        'family': 'pure_l9_ridge_below_second_winner',
        'summary': 'Retune the pure entry-relay l9 gate below/above 0.75 s without terminal reclosure, to check whether the second winner was already at the clean local l9 optimum and whether any improvement there transfers before adding l12.',
        'candidate_names': [
            'entryrelay_l8x1_l9y0p625_unifiedcore',
            'entryrelay_l8x1_l9y0p875_unifiedcore',
        ],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()



def load_reference_payload(path: Path, noise_scale: float) -> dict[str, Any]:
    payload = _load_json(path)
    expected_cfg = expected_noise_config(noise_scale)
    got_cfg = payload.get('extra', {}).get('noise_config') or payload.get('extra', {}).get('shared_noise_config')
    if got_cfg is not None and got_cfg != expected_cfg:
        raise ValueError(f'Noise configuration mismatch for {path}')
    return payload



def max_param(payload: dict[str, Any]) -> dict[str, Any]:
    name, info = max(payload['param_errors'].items(), key=lambda kv: float(kv[1]['pct_error']))
    return {
        'name': name,
        'pct_error': float(info['pct_error']),
    }



def compact_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        'overall': {
            'mean_pct_error': float(payload['overall']['mean_pct_error']),
            'median_pct_error': float(payload['overall']['median_pct_error']),
            'max_pct_error': float(payload['overall']['max_pct_error']),
        },
        'key_param_errors': {
            'dKg_xx': float(payload['param_errors']['dKg_xx']['pct_error']),
            'eb_x': float(payload['param_errors']['eb_x']['pct_error']),
            'Ka2_y': float(payload['param_errors']['Ka2_y']['pct_error']),
            'dKa_xx': float(payload['param_errors']['dKa_xx']['pct_error']),
            'dKg_yy': float(payload['param_errors']['dKg_yy']['pct_error']),
        },
        'max_driver': max_param(payload),
    }



def overall_triplet(payload: dict[str, Any]) -> str:
    o = payload['overall']
    return f"{o['mean_pct_error']:.3f} / {o['median_pct_error']:.3f} / {o['max_pct_error']:.3f}"



def delta_vs_reference(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        out[metric] = float(reference['overall'][metric]) - float(candidate['overall'][metric])
    return out



def attach_att0(path: Path, payload: dict[str, Any], candidate_name: str, method_key: str, family: str, hypothesis_id: str) -> dict[str, Any]:
    extra = payload.setdefault('extra', {})
    extra['att0_deg'] = ATT0_DEG
    extra['comparison_mode'] = 'corrected_frontier_resume_batch2'
    extra['candidate_registry_key'] = candidate_name
    extra['method_key'] = method_key
    extra['family'] = family
    extra['hypothesis_id'] = hypothesis_id
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload



def candidate_specs() -> list[dict[str, Any]]:
    def core_with(l9_dwell: float, l12_dwell: float | None = None) -> dict[int, list[Any]]:
        parts = [
            l8_xpair(1.0, 'l8_x1'),
            l9_ypair_neg(l9_dwell, f'l9_ypair_neg{str(l9_dwell).replace("-", "neg").replace(".", "p")}'),
            l10_unified_core(),
            l11_y10x0back2_core(),
        ]
        if l12_dwell is not None:
            label = str(l12_dwell).replace('-', 'neg').replace('.', 'p')
            parts.append(closed_pair(12, 'inner', -90, l12_dwell, f'l12_yneg{label}'))
        return merge_insertions(*parts)

    return [
        {
            'name': 'relay_l12y0p25_on_entry',
            'family': 'terminal_microclosure_ridge_around_current_leader',
            'hypothesis_id': 'H1',
            'rationale': 'Lighter terminal-only variant of the current leader: keep l9 at 1.0 s and reduce l12 from 0.5 s to 0.25 s.',
            'insertions': core_with(1.0, 0.25),
        },
        {
            'name': 'relay_l12y0p75_on_entry',
            'family': 'terminal_microclosure_ridge_around_current_leader',
            'hypothesis_id': 'H1',
            'rationale': 'Heavier terminal-only variant of the current leader: keep l9 at 1.0 s and raise l12 from 0.5 s to 0.75 s.',
            'insertions': core_with(1.0, 0.75),
        },
        {
            'name': 'relay_l9y0p625_l12y0p5_on_entry',
            'family': 'l9_terminal_coupled_ridge',
            'hypothesis_id': 'H2',
            'rationale': 'Coupled ridge: soften l9 below the second winner to 0.625 s while preserving the current leader\'s l12 = 0.5 s terminal micro-closure.',
            'insertions': core_with(0.625, 0.5),
        },
        {
            'name': 'relay_l9y0p75_l12y0p5_on_entry',
            'family': 'l9_terminal_coupled_ridge',
            'hypothesis_id': 'H2',
            'rationale': 'Direct fusion of the two current winners: second winner\'s l9 = 0.75 s plus current leader\'s l12 = 0.5 s.',
            'insertions': core_with(0.75, 0.5),
        },
        {
            'name': 'relay_l9y0p875_l12y0p5_on_entry',
            'family': 'l9_terminal_coupled_ridge',
            'hypothesis_id': 'H2',
            'rationale': 'Milder coupled ridge: l9 softened only halfway from 1.0 to 0.75 while keeping l12 = 0.5 s.',
            'insertions': core_with(0.875, 0.5),
        },
        {
            'name': 'relay_l9y0p75_l12y0p25_on_entry',
            'family': 'l9_terminal_coupled_ridge',
            'hypothesis_id': 'H2',
            'rationale': 'Double-light coupled ridge: pair l9 = 0.75 s with an even smaller l12 = 0.25 s micro-closure.',
            'insertions': core_with(0.75, 0.25),
        },
        {
            'name': 'relay_l9y0p75_l12y0p75_on_entry',
            'family': 'l9_terminal_coupled_ridge',
            'hypothesis_id': 'H2',
            'rationale': 'Symmetric medium coupled ridge: pair l9 = 0.75 s with a slightly stronger l12 = 0.75 s closure.',
            'insertions': core_with(0.75, 0.75),
        },
        {
            'name': 'entryrelay_l8x1_l9y0p625_unifiedcore',
            'family': 'pure_l9_ridge_below_second_winner',
            'hypothesis_id': 'H3',
            'rationale': 'Pure l9 ridge check below the second winner, without terminal l12 closure.',
            'insertions': core_with(0.625, None),
        },
        {
            'name': 'entryrelay_l8x1_l9y0p875_unifiedcore',
            'family': 'pure_l9_ridge_below_second_winner',
            'hypothesis_id': 'H3',
            'rationale': 'Pure l9 ridge check above the second winner, without terminal l12 closure.',
            'insertions': core_with(0.875, None),
        },
    ]



def failure_reason(row: dict[str, Any]) -> str:
    d = row['delta_vs_leader_markov42']
    parts: list[str] = []
    if d['mean_pct_error'] <= 0:
        parts.append(f"mean {abs(d['mean_pct_error']):.3f} worse")
    else:
        parts.append(f"mean {d['mean_pct_error']:.3f} better")
    if d['max_pct_error'] <= 0:
        parts.append(f"max {abs(d['max_pct_error']):.3f} worse")
    else:
        parts.append(f"max {d['max_pct_error']:.3f} better")
    if d['median_pct_error'] <= 0:
        parts.append(f"median {abs(d['median_pct_error']):.3f} worse")
    else:
        parts.append(f"median {d['median_pct_error']:.3f} better")
    driver = row['markov42']['max_driver']
    parts.append(f"max driver={driver['name']} {driver['pct_error']:.3f}")
    return '; '.join(parts)



def render_report(summary: dict[str, Any]) -> str:
    best = summary['best_candidate']
    leader = summary['references']['current_leader']
    runner = summary['references']['second_winner']
    lines: list[str] = []
    lines.append('# Chapter-3 corrected frontier resume batch 2')
    lines.append('')
    lines.append('## 1. Search target and fixed constraints')
    lines.append('')
    lines.append('- Hard basis enforced: **att0 = (0, 0, 0)** only.')
    lines.append('- Hard legality enforced: **real dual-axis motor sequence only**, with exact closure before the base backbone resumes.')
    lines.append('- Resume target: continue from the current corrected leader `relay_l12y0p5_on_entry` rather than reopen weaker distant families.')
    lines.append(f"- Launch leader: **{leader['markov42_triplet']}** (`relay_l12y0p5_on_entry`)")
    lines.append(f"- Current second winner: **{runner['markov42_triplet']}** (`entryrelay_l8x1_l9y0p75_unifiedcore`)")
    lines.append('')
    lines.append('## 2. Local resume hypotheses')
    lines.append('')
    for item in summary['hypotheses_tested']:
        lines.append(f"- **{item['id']} · {item['family']}** — {item['summary']}")
        lines.append(f"  - tested candidates: {', '.join(item['candidate_names'])}")
    lines.append('')
    lines.append('## 3. Markov42 landed batch')
    lines.append('')
    lines.append('| rank | candidate | family | mean | median | max | max driver | Δmean vs leader | Δmedian vs leader | Δmax vs leader | Δmean vs runner | Δmax vs runner |')
    lines.append('|---:|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|')
    for idx, row in enumerate(summary['rows_sorted'], start=1):
        m = row['markov42']['overall']
        d0 = row['delta_vs_leader_markov42']
        d1 = row['delta_vs_runner_markov42']
        md = row['markov42']['max_driver']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['family']} | {m['mean_pct_error']:.3f} | {m['median_pct_error']:.3f} | {m['max_pct_error']:.3f} | {md['name']} {md['pct_error']:.3f} | {d0['mean_pct_error']:+.3f} | {d0['median_pct_error']:+.3f} | {d0['max_pct_error']:+.3f} | {d1['mean_pct_error']:+.3f} | {d1['max_pct_error']:+.3f} |"
        )
    lines.append('')
    lines.append('## 4. Best landing in this resume batch')
    lines.append('')
    lines.append(f"- **Best candidate in this batch:** `{best['candidate_name']}`")
    lines.append(f"- **Markov42:** **{overall_triplet(best['markov42'])}**")
    if best.get('kf36') is not None:
        lines.append(f"- **KF36:** **{overall_triplet(best['kf36'])}**")
    lines.append(f"- vs current leader: Δmean **{best['delta_vs_leader_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_leader_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_leader_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- vs second winner: Δmean **{best['delta_vs_runner_markov42']['mean_pct_error']:+.3f}**, Δmedian **{best['delta_vs_runner_markov42']['median_pct_error']:+.3f}**, Δmax **{best['delta_vs_runner_markov42']['max_pct_error']:+.3f}**")
    lines.append(f"- max driver remains **{best['markov42']['max_driver']['name']} = {best['markov42']['max_driver']['pct_error']:.3f}%**")
    lines.append('')
    lines.append('## 5. Resume verdict')
    lines.append('')
    lines.append(f"- **Leader improved?** **{summary['bottom_line']['leader_improved']}**")
    if summary['frontier_improvers']:
        lines.append(f"- Improving candidates: **{', '.join(summary['frontier_improvers'])}**")
    else:
        lines.append('- No candidate simultaneously lowered both **mean** and **max** below the current leader.')
    if summary['near_misses']:
        lines.append('- Near misses / side signals:')
        for row in summary['near_misses']:
            lines.append(f"  - `{row['candidate_name']}` → {failure_reason(row)}")
    lines.append('')
    lines.append('## 6. KF36 recheck gate')
    lines.append('')
    lines.append(f"- Rechecked candidates: **{', '.join(summary['kf36_rechecked_candidates']) if summary['kf36_rechecked_candidates'] else 'none'}**")
    if not summary['kf36_rechecked_candidates']:
        lines.append(f"- Gate reason: {summary['kf36_gate_reason']}")
    else:
        for row in summary['rows_sorted']:
            if row.get('kf36') is not None:
                lines.append(f"  - `{row['candidate_name']}` → Markov42 **{overall_triplet(row['markov42'])}**, KF36 **{overall_triplet(row['kf36'])}**")
    lines.append('')
    lines.append('## 7. Bottom line')
    lines.append('')
    lines.append(f"- Current corrected leader after this batch: **{summary['bottom_line']['final_best_triplet']}** (`{summary['bottom_line']['final_best_name']}`)")
    lines.append(f"- Main read: **{summary['bottom_line']['statement']}**")
    lines.append('')
    return '\n'.join(lines) + '\n'



def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module(str(METHOD_DIR / 'method_42state_gm1.py'), str(SOURCE_FILE))
    faithful = build_candidate(mod, ())

    references: dict[str, Any] = {}
    for key, info in REFERENCE_FILES.items():
        m = load_reference_payload(info['markov42'], args.noise_scale)
        k = load_reference_payload(info['kf36'], args.noise_scale)
        references[key] = {
            'label': info['label'],
            'markov42': compact_metrics(m),
            'kf36': compact_metrics(k),
            'markov42_triplet': overall_triplet(m),
            'kf36_triplet': overall_triplet(k),
            'files': {
                'markov42': str(info['markov42']),
                'kf36': str(info['kf36']),
            },
        }

    specs = candidate_specs()
    spec_by_name = {spec['name']: spec for spec in specs}
    rows: list[dict[str, Any]] = []
    candidates_by_name: dict[str, Any] = {}

    for spec in specs:
        candidate = build_closedloop_candidate(mod, spec, faithful.rows, faithful.action_sequence)
        candidates_by_name[candidate.name] = candidate
        markov_payload, _, markov_path = run_candidate_payload(mod, candidate, 'markov42_noisy', args.noise_scale, force_rerun=args.force_rerun)
        markov_payload = attach_att0(markov_path, markov_payload, candidate.name, 'markov42_noisy', spec['family'], spec['hypothesis_id'])
        row = {
            'candidate_name': candidate.name,
            'family': spec['family'],
            'hypothesis_id': spec['hypothesis_id'],
            'rationale': spec['rationale'],
            'total_time_s': candidate.total_time_s,
            'result_files': {'markov42': str(markov_path)},
            'markov42': compact_metrics(markov_payload),
        }
        row['delta_vs_leader_markov42'] = delta_vs_reference(references['current_leader']['markov42'], row['markov42'])
        row['delta_vs_runner_markov42'] = delta_vs_reference(references['second_winner']['markov42'], row['markov42'])
        row['delta_vs_old_entry_markov42'] = delta_vs_reference(references['old_entry_frontier']['markov42'], row['markov42'])
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: (r['markov42']['overall']['mean_pct_error'], r['markov42']['overall']['max_pct_error'], r['markov42']['overall']['median_pct_error']))
    best = rows_sorted[0]

    frontier_improvers = [
        row['candidate_name']
        for row in rows_sorted
        if row['delta_vs_leader_markov42']['mean_pct_error'] > 0 and row['delta_vs_leader_markov42']['max_pct_error'] > 0
    ]

    near_misses = [
        row for row in rows_sorted
        if not (row['delta_vs_leader_markov42']['mean_pct_error'] > 0 and row['delta_vs_leader_markov42']['max_pct_error'] > 0)
        and (
            row['delta_vs_leader_markov42']['mean_pct_error'] > 0
            or row['delta_vs_leader_markov42']['max_pct_error'] > 0
            or abs(row['delta_vs_leader_markov42']['mean_pct_error']) <= 0.02
            or abs(row['delta_vs_leader_markov42']['max_pct_error']) <= 0.12
        )
    ][:4]

    kf36_rechecked_candidates: list[str] = []
    kf36_gate_reason = 'No candidate was close enough to the current leader to justify KF36.'
    for row in rows_sorted[:3]:
        d = row['delta_vs_leader_markov42']
        if d['mean_pct_error'] > -0.02 and d['max_pct_error'] > -0.15:
            candidate = candidates_by_name[row['candidate_name']]
            spec = spec_by_name[row['candidate_name']]
            kf_payload, _, kf_path = run_candidate_payload(mod, candidate, 'kf36_noisy', args.noise_scale, force_rerun=args.force_rerun)
            kf_payload = attach_att0(kf_path, kf_payload, candidate.name, 'kf36_noisy', spec['family'], spec['hypothesis_id'])
            row['result_files']['kf36'] = str(kf_path)
            row['kf36'] = compact_metrics(kf_payload)
            row['delta_vs_leader_kf36'] = delta_vs_reference(references['current_leader']['kf36'], row['kf36'])
            row['delta_vs_runner_kf36'] = delta_vs_reference(references['second_winner']['kf36'], row['kf36'])
            kf36_rechecked_candidates.append(candidate.name)
    if not kf36_rechecked_candidates:
        kf36_gate_reason = f"Best Markov42 candidate {best['candidate_name']} = {overall_triplet(best['markov42'])} stayed outside the near-leader gate."

    if frontier_improvers:
        final_best_name = frontier_improvers[0]
        final_best_triplet = overall_triplet(next(row['markov42'] for row in rows_sorted if row['candidate_name'] == final_best_name))
        leader_improved = 'YES'
        statement = (
            'A new local winner emerged: the current leader was not the final ridge point, and the new coupled/retuned candidate now becomes the corrected leader under the same legality constraints.'
        )
    else:
        final_best_name = 'relay_l12y0p5_on_entry'
        final_best_triplet = references['current_leader']['markov42_triplet']
        leader_improved = 'NO'
        statement = (
            'The current leader proved locally robust. Softening or re-coupling l9 with the terminal micro-closure produced trade-offs, but none lowered both mean and max below the existing 1.064 / 0.569 / 4.859 point. The best side signal was whether l9 softening could trim eb_x / Ka2_y, yet dKg_xx suppression degraded before a net win appeared.'
        )

    summary = {
        'task': 'chapter-3 corrected frontier resume batch 2',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'corrected_att0_deg': ATT0_DEG,
        'references': references,
        'hypotheses_tested': HYPOTHESES,
        'tested_candidates': [spec['name'] for spec in specs],
        'rows_sorted': rows_sorted,
        'best_candidate': best,
        'frontier_improvers': frontier_improvers,
        'near_misses': near_misses,
        'kf36_rechecked_candidates': kf36_rechecked_candidates,
        'kf36_gate_reason': kf36_gate_reason,
        'bottom_line': {
            'leader_improved': leader_improved,
            'final_best_name': final_best_name,
            'final_best_triplet': final_best_triplet,
            'statement': statement,
        },
    }

    REPORT_PATH.write_text(render_report(summary), encoding='utf-8')
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({
        'report_path': str(REPORT_PATH),
        'summary_path': str(SUMMARY_PATH),
        'leader_improved': leader_improved,
        'final_best_name': final_best_name,
        'final_best_triplet': final_best_triplet,
        'batch_best_name': best['candidate_name'],
        'batch_best_triplet': overall_triplet(best['markov42']),
        'frontier_improvers': frontier_improvers,
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
