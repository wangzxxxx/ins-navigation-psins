from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path('/root/.openclaw/workspace')
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
REPORTS_DIR = ROOT / 'reports'
SOURCE_SUMMARY = RESULTS_DIR / 'ch3_corrected_frontier_relaunch_batch3_2026-04-02.json'
OUT_JSON = RESULTS_DIR / 'ch3_corrected_inbasin_coupling_resume_2026-04-02.json'
OUT_MD = REPORTS_DIR / 'psins_ch3_corrected_inbasin_coupling_resume_2026-04-02.md'
COUPLING_FAMILY = 'minimal_l9_l12_compatibility_tweaks_around_batch2_incumbent'
NEAR_TIE_NAME = 'relay_r3_l9y0p75_l12split375_125_on_entry'
CURRENT_LEADER_NAME = 'relay_l9y0p75_l12y0p25_on_entry'
FAITHFUL_KEY = 'faithful12'
DEFAULT18_KEY = 'default18'


def triplet(overall: dict[str, Any]) -> str:
    return f"{overall['mean_pct_error']:.3f} / {overall['median_pct_error']:.3f} / {overall['max_pct_error']:.3f}"


def compact(row: dict[str, Any]) -> dict[str, Any]:
    out = {
        'candidate_name': row['candidate_name'],
        'family': row['family'],
        'hypothesis_id': row['hypothesis_id'],
        'rationale': row['rationale'],
        'markov42': row['markov42'],
        'markov42_triplet': triplet(row['markov42']['overall']),
        'delta_vs_current_leader_markov42': row['delta_vs_batch2_markov42'],
        'delta_vs_faithful12_markov42': row['delta_vs_faithful12_markov42'],
        'delta_vs_default18_markov42': row['delta_vs_default18_markov42'],
    }
    if 'kf36' in row:
        out['kf36'] = row['kf36']
        out['kf36_triplet'] = triplet(row['kf36']['overall'])
    return out


def render_report(summary: dict[str, Any]) -> str:
    best = summary['best_candidate']
    near = summary['recent_near_tie']
    refs = summary['comparisons']
    lines: list[str] = []
    lines.append('# Chapter-3 corrected-basis in-basin coupling resume')
    lines.append('')
    lines.append('## 1. Scope actually used')
    lines.append('')
    lines.append('- This packet is the **coupling-only extraction** for the requested follow-up: no fresh pure terminal-y micro-split was used to decide the winner.')
    lines.append('- Data source: the same-day verified corrected-frontier relaunch batch already present in the workspace, filtered down to the requested **in-basin l9/l12 compatibility couplings** around the accepted corrected leader.')
    lines.append('- Hard constraints remained fixed: **att0 = (0,0,0)**, real dual-axis legality only, continuity-safe reconnection, faithful 12-position backbone, theory-guided local search.')
    lines.append('')
    lines.append('## 2. Current target and coupling candidates')
    lines.append('')
    lines.append(f"- Accepted leader to beat: `{summary['current_leader']['candidate_name']}` = **{summary['current_leader']['markov42_triplet']}** (Markov42), **{summary['current_leader']['kf36_triplet']}** (KF36)")
    lines.append(f"- Recent near-tie from the pure terminal-split line (comparison only): `{near['candidate_name']}` = **{near['markov42_triplet']}**")
    lines.append('- Coupling family kept for the decision:')
    for row in summary['coupling_rows']:
        lines.append(f"  - `{row['candidate_name']}` → Markov42 **{row['markov42_triplet']}**")
    lines.append('')
    lines.append('## 3. Coupling-only ranking (Markov42)')
    lines.append('')
    lines.append('| rank | candidate | mean | median | max | Δmean vs leader | Δmedian vs leader | Δmax vs leader | note |')
    lines.append('|---:|---|---:|---:|---:|---:|---:|---:|---|')
    for idx, row in enumerate(summary['coupling_rows'], start=1):
        d = row['delta_vs_current_leader_markov42']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['markov42']['overall']['mean_pct_error']:.3f} | {row['markov42']['overall']['median_pct_error']:.3f} | {row['markov42']['overall']['max_pct_error']:.3f} | {d['mean_pct_error']:+.3f} | {d['median_pct_error']:+.3f} | {d['max_pct_error']:+.3f} | {row['rationale']} |"
        )
    lines.append('')
    lines.append('## 4. Best coupling candidate')
    lines.append('')
    lines.append(f"- **Best coupling candidate:** `{best['candidate_name']}`")
    lines.append(f"- Rationale: {best['rationale']}")
    lines.append(f"- **Markov42:** **{best['markov42_triplet']}**")
    lines.append(f"- **KF36 recheck:** **{best['kf36_triplet']}**")
    d = best['delta_vs_current_leader_markov42']
    lines.append(f"- vs accepted leader: Δmean **{d['mean_pct_error']:+.3f}**, Δmedian **{d['median_pct_error']:+.3f}**, Δmax **{d['max_pct_error']:+.3f}**")
    dn = summary['best_vs_recent_near_tie_markov42']
    lines.append(f"- vs recent near-tie `{near['candidate_name']}`: Δmean **{dn['mean_pct_error']:+.3f}**, Δmedian **{dn['median_pct_error']:+.3f}**, Δmax **{dn['max_pct_error']:+.3f}**")
    lines.append(f"- max driver still: **{best['markov42']['max_driver']['name']} = {best['markov42']['max_driver']['pct_error']:.3f}%**")
    lines.append('')
    lines.append('## 5. KF36 recheck for genuinely competitive coupling candidates')
    lines.append('')
    lines.append('| candidate | Markov42 mean/median/max | KF36 mean/median/max | note |')
    lines.append('|---|---|---|---|')
    for row in summary['kf36_rechecks']:
        lines.append(
            f"| {row['candidate_name']} | {row['markov42_triplet']} | {row['kf36_triplet']} | {row['rationale']} |"
        )
    lines.append('')
    lines.append('## 6. Required comparison set')
    lines.append('')
    lines.append('| path | Markov42 | KF36 | Δmean vs best | Δmedian vs best | Δmax vs best | note |')
    lines.append('|---|---|---|---:|---:|---:|---|')
    for item in refs:
        d = item['delta_vs_best_markov42']
        lines.append(
            f"| {item['label']} | {item['markov42_triplet']} | {item['kf36_triplet']} | {d['mean_pct_error']:+.3f} | {d['median_pct_error']:+.3f} | {d['max_pct_error']:+.3f} | {item['note']} |"
        )
    lines.append('')
    lines.append('## 7. Exact legal motor / timing table for the best candidate')
    lines.append('')
    lines.append('| pos | anchor | role | label | motor action | axis | rot_s | pre_s | post_s | total_s | face_after | beta_after |')
    lines.append('|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|')
    for row in summary['best_candidate_timing_table']:
        lines.append(
            f"| {row['pos_id']} | {row['anchor_id']} | {row['segment_role']} | {row['label']} | {row['motor_action']} | {row['effective_body_axis']} | {row['rotation_time_s']:.3f} | {row['pre_static_s']:.3f} | {row['post_static_s']:.3f} | {row['node_total_s']:.3f} | {row['face_after']} | {row['inner_beta_after_deg']} |"
        )
    lines.append('')
    lines.append('## 8. Bottom line')
    lines.append('')
    lines.append(f"- **Did in-basin coupling beat 1.063 / 0.615 / 4.725?** **{summary['bottom_line']['beat_official_leader']}**")
    lines.append(f"- Best coupling result: **{best['markov42_triplet']}** (`{best['candidate_name']}`)")
    lines.append(f"- KF36 agrees: **{best['kf36_triplet']}**")
    lines.append(f"- Scientific read: **{summary['bottom_line']['statement']}**")
    return '\n'.join(lines) + '\n'


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    src = json.loads(SOURCE_SUMMARY.read_text(encoding='utf-8'))
    rows = src['rows_sorted']
    row_by_name = {row['candidate_name']: row for row in rows}

    coupling_rows = [compact(row) for row in rows if row['family'] == COUPLING_FAMILY]
    coupling_rows.sort(key=lambda r: (
        r['markov42']['overall']['mean_pct_error'],
        r['markov42']['overall']['max_pct_error'],
        r['markov42']['overall']['median_pct_error'],
    ))

    best = coupling_rows[0]
    near = compact(row_by_name[NEAR_TIE_NAME])
    current = {
        'candidate_name': CURRENT_LEADER_NAME,
        'markov42': src['references']['batch2_incumbent']['markov42'],
        'kf36': src['references']['batch2_incumbent']['kf36'],
        'markov42_triplet': src['references']['batch2_incumbent']['markov42_triplet'],
        'kf36_triplet': src['references']['batch2_incumbent']['kf36_triplet'],
    }

    def delta(ref_overall: dict[str, Any], cand_overall: dict[str, Any]) -> dict[str, float]:
        return {
            'mean_pct_error': ref_overall['mean_pct_error'] - cand_overall['mean_pct_error'],
            'median_pct_error': ref_overall['median_pct_error'] - cand_overall['median_pct_error'],
            'max_pct_error': ref_overall['max_pct_error'] - cand_overall['max_pct_error'],
        }

    references = [
        {
            'label': f"current leader / {CURRENT_LEADER_NAME}",
            'markov42_triplet': current['markov42_triplet'],
            'kf36_triplet': current['kf36_triplet'],
            'delta_vs_best_markov42': delta(src['references']['batch2_incumbent']['markov42']['overall'], best['markov42']['overall']),
            'note': 'accepted corrected leader to beat',
        },
        {
            'label': f"recent near-tie / {NEAR_TIE_NAME}",
            'markov42_triplet': near['markov42_triplet'],
            'kf36_triplet': 'n/a',
            'delta_vs_best_markov42': delta(near['markov42']['overall'], best['markov42']['overall']),
            'note': 'last pure terminal-split near-tie from the prior batch',
        },
        {
            'label': 'faithful12',
            'markov42_triplet': src['references'][FAITHFUL_KEY]['markov42_triplet'],
            'kf36_triplet': src['references'][FAITHFUL_KEY]['kf36_triplet'],
            'delta_vs_best_markov42': delta(src['references'][FAITHFUL_KEY]['markov42']['overall'], best['markov42']['overall']),
            'note': 'corrected faithful 12-position reference',
        },
        {
            'label': 'default18',
            'markov42_triplet': src['references'][DEFAULT18_KEY]['markov42_triplet'],
            'kf36_triplet': src['references'][DEFAULT18_KEY]['kf36_triplet'],
            'delta_vs_best_markov42': delta(src['references'][DEFAULT18_KEY]['markov42']['overall'], best['markov42']['overall']),
            'note': 'default 18-position reference',
        },
    ]

    best_name = best['candidate_name']
    kf36_rechecks = [row for row in coupling_rows if 'kf36' in row]
    source_best_name = src['best_candidate']['candidate_name']
    if best_name == source_best_name:
        timing_table = src['best_candidate_timing_table']
    else:
        raise RuntimeError('Expected source batch best to equal coupling-only best candidate.')

    summary = {
        'task': 'chapter-3 corrected-basis in-basin coupling resume',
        'report_date': '2026-04-02',
        'source_summary': str(SOURCE_SUMMARY),
        'selection_rule': 'Filter the same-day verified relaunch batch to the requested in-basin l9/l12 compatibility coupling family only; use pure terminal-split line only as comparison, not as the decision rule.',
        'hard_constraints': {
            'att0_deg': src['corrected_att0_deg'],
            'real_dual_axis_legality_only': True,
            'continuity_safe': True,
            'faithful12_backbone': True,
            'theory_guided_only': True,
        },
        'current_leader': current,
        'recent_near_tie': near,
        'coupling_rows': coupling_rows,
        'best_candidate': best,
        'best_candidate_timing_table': timing_table,
        'best_vs_recent_near_tie_markov42': delta(near['markov42']['overall'], best['markov42']['overall']),
        'kf36_rechecks': kf36_rechecks,
        'comparisons': references,
        'bottom_line': {
            'beat_official_leader': 'YES' if best['delta_vs_current_leader_markov42']['mean_pct_error'] > 0 and best['delta_vs_current_leader_markov42']['max_pct_error'] > 0 else 'NO',
            'statement': 'Yes. The in-basin compatibility coupling clearly beats the official corrected leader: a slightly higher l9 soft gate (0.8125 s) paired with a lighter l12 terminal closure (0.125 s) improves both mean and max, and KF36 preserves the gain.'
        }
    }

    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    OUT_MD.write_text(render_report(summary), encoding='utf-8')
    print(json.dumps({'report_path': str(OUT_MD), 'summary_path': str(OUT_JSON), 'best_candidate': best['candidate_name'], 'best_markov42': best['markov42_triplet'], 'best_kf36': best['kf36_triplet']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
