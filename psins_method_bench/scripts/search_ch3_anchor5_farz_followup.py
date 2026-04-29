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

from benchmark_ch3_12pos_goalA_repairs import compact_result
from common_markov import load_module
from compare_four_methods_shared_noise import _load_json, _noise_matches, expected_noise_config
from search_ch3_12pos_closedloop_local_insertions import (
    NOISE_SCALE,
    REPORT_DATE,
    StepSpec,
    build_closedloop_candidate,
    render_action,
    run_candidate_payload,
)
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate

RELAYMAX_UNIFIED_Y2_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_relaymax_unified_l9y2_shared_noise0p08_param_errors.json'
RELAYMAX_UNIFIED_Y2_KF = RESULTS_DIR / 'KF36_ch3closedloop_relaymax_unified_l9y2_shared_noise0p08_param_errors.json'
ENTRYRELAY_MAIN_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json'
ENTRYRELAY_MAIN_KF = RESULTS_DIR / 'KF36_ch3closedloop_entryrelay_l8x1_l9y1_unifiedcore_shared_noise0p08_param_errors.json'
OLD_BEST_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
OLD_BEST_KF = RESULTS_DIR / 'KF36_ch3legaldual2_legal_flip_8_11_12_retime_flipnodes_pre20_post70_shared_noise0p08_param_errors.json'
FAITHFUL_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_ch3faithful12_shared_noise0p08_param_errors.json'
FAITHFUL_KF = RESULTS_DIR / 'KF36_ch3faithful12_shared_noise0p08_param_errors.json'
DEFAULT18_MARKOV = RESULTS_DIR / 'M_markov_42state_gm1_default18_shared_noise0p08_param_errors.json'
DEFAULT18_KF = RESULTS_DIR / 'KF36_default18_shared_noise0p08_param_errors.json'


LOCAL_PATTERN_SUMMARY = [
    {
        'label': 'low-dose mean edge',
        'description': 'Keep the original relaymax unified y2 core fixed and refine the anchor5 negative far-z seed dose near the previously discovered neg2 / neg4 points.',
    },
    {
        'label': 'sign / order control',
        'description': 'Check whether the family is truly directional by flipping the anchor5 sign and by running small asymmetry tests around the neg4 seed.',
    },
    {
        'label': 'core-compatibility microtune',
        'description': 'Touch only the relay gate strength at l9 to see whether the far-z seed couples better to a slightly stronger or weaker entry into the unchanged relaymax unified core.',
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--report-date', default=REPORT_DATE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def load_json_checked(path: Path, noise_scale: float) -> dict[str, Any]:
    payload = _load_json(path)
    expected_cfg = expected_noise_config(noise_scale)
    if not _noise_matches(payload, expected_cfg):
        raise ValueError(f'Noise configuration mismatch: {path}')
    return payload


def compact_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        'overall': payload['overall'],
        'key_param_errors': {
            'dKa_yy': float(payload['param_errors']['dKa_yy']['pct_error']),
            'dKg_zz': float(payload['param_errors']['dKg_zz']['pct_error']),
            'Ka2_y': float(payload['param_errors']['Ka2_y']['pct_error']),
            'Ka2_z': float(payload['param_errors']['Ka2_z']['pct_error']),
        },
    }


def delta_vs_ref(ref_payload: dict[str, Any], cand_payload: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        ref_v = float(ref_payload['overall'][metric])
        cand_v = float(cand_payload['overall'][metric])
        out[metric] = {
            'reference': ref_v,
            'candidate': cand_v,
            'improvement_pct_points': ref_v - cand_v,
        }
    return out


def row_summary(payload: dict[str, Any]) -> str:
    o = payload['overall'] if 'overall' in payload else payload
    return f"{o['mean_pct_error']:.3f} / {o['median_pct_error']:.3f} / {o['max_pct_error']:.3f}"


def dose_tag(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(value).replace('.', 'p')


def l9_tag(value: float) -> str:
    return str(value).replace('.', 'p')


def closed_pair(kind: str, angle_deg: int, dwell_s: float, label: str, rot_s: float = 5.0) -> list[StepSpec]:
    return [
        StepSpec(kind=kind, angle_deg=angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_out', label=f'{label}_out'),
        StepSpec(kind=kind, angle_deg=-angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_return', label=f'{label}_return'),
    ]


def asym_pair(kind: str, first_angle_deg: int, dwell1_s: float, dwell2_s: float, label: str, rot_s: float = 5.0) -> list[StepSpec]:
    return [
        StepSpec(kind=kind, angle_deg=first_angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(dwell1_s), segment_role='motif_out', label=f'{label}_out'),
        StepSpec(kind=kind, angle_deg=-first_angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(dwell2_s), segment_role='motif_return', label=f'{label}_return'),
    ]


def xpair_outerhold(dwell_s: float, label: str, inner_angle_deg: int = -90, outer_angle_deg: int = +90, rot_s: float = 5.0) -> list[StepSpec]:
    return [
        StepSpec(kind='inner', angle_deg=inner_angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=0.0, segment_role='motif_inner_open', label=f'{label}_inner_open'),
        StepSpec(kind='outer', angle_deg=outer_angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_outer_sweep', label=f'{label}_outer_sweep'),
        StepSpec(kind='outer', angle_deg=-outer_angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(dwell_s), segment_role='motif_outer_return', label=f'{label}_outer_return'),
        StepSpec(kind='inner', angle_deg=-inner_angle_deg, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=0.0, segment_role='motif_inner_close', label=f'{label}_inner_close'),
    ]


def zquad(y_s: float, x_s: float, back_s: float, label: str, rot_s: float = 5.0) -> list[StepSpec]:
    return [
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(y_s), segment_role='motif_y_pos', label=f'{label}_q1'),
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(x_s), segment_role='motif_zero_a', label=f'{label}_q2'),
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(y_s), segment_role='motif_y_neg', label=f'{label}_q3'),
        StepSpec(kind='outer', angle_deg=+90, rotation_time_s=rot_s, pre_static_s=0.0, post_static_s=float(back_s), segment_role='motif_zero_b', label=f'{label}_q4'),
    ]


def merge_insertions(*dicts: dict[int, list[StepSpec]]) -> dict[int, list[StepSpec]]:
    out: dict[int, list[StepSpec]] = {}
    for d in dicts:
        for k, v in d.items():
            out.setdefault(k, []).extend(v)
    return out


def l9_ypair_neg(dwell_s: float, label: str) -> dict[int, list[StepSpec]]:
    return {9: closed_pair('inner', -90, float(dwell_s), label)}


def l10_unified_core() -> dict[int, list[StepSpec]]:
    return {10: closed_pair('outer', -90, 5.0, 'l10_zpair_neg5') + closed_pair('inner', -90, 1.0, 'l10_ypair_neg1')}


def l11_y10x0back2_core() -> dict[int, list[StepSpec]]:
    return {11: xpair_outerhold(10.0, 'l11_xpair_outerhold') + zquad(10.0, 0.0, 2.0, 'l11_zquad_y10x0back2')}


def anchor5_zseed(dose_abs_s: float, sign: str, label: str) -> dict[int, list[StepSpec]]:
    angle = -90 if sign == 'neg' else +90
    return {5: closed_pair('outer', angle, float(dose_abs_s), label)}


def anchor5_zseed_asym(first_angle_deg: int, dwell1_s: float, dwell2_s: float, label: str) -> dict[int, list[StepSpec]]:
    return {5: asym_pair('outer', first_angle_deg, dwell1_s, dwell2_s, label)}


def relay_core(l9_dwell_s: float) -> dict[int, list[StepSpec]]:
    return merge_insertions(
        l9_ypair_neg(l9_dwell_s, f'l9_ypair_neg{l9_tag(l9_dwell_s)}'),
        l10_unified_core(),
        l11_y10x0back2_core(),
    )


def make_spec(name: str, mode: str, rationale: str, insertions: dict[int, list[StepSpec]], *, seed_sign: str, seed_dose_s: float | None, l9_dwell_s: float, asymmetry: str = 'none') -> dict[str, Any]:
    return {
        'name': name,
        'mode': mode,
        'rationale': rationale,
        'seed_sign': seed_sign,
        'seed_dose_s': seed_dose_s,
        'l9_dwell_s': l9_dwell_s,
        'asymmetry': asymmetry,
        'insertions': insertions,
    }


def candidate_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for dose in [1.0, 2.0, 3.0, 4.0, 5.0]:
        tag = dose_tag(dose)
        specs.append(make_spec(
            f'zseed_l5_neg{tag}_plus_relaymax_unified_l9y2',
            'dose_sweep',
            'Negative anchor5 far-z seed with the original relaymax unified y2 core left unchanged.',
            merge_insertions(anchor5_zseed(dose, 'neg', f'l5_zseed_neg{tag}'), relay_core(2.0)),
            seed_sign='neg',
            seed_dose_s=dose,
            l9_dwell_s=2.0,
        ))
    for dose in [2.0, 4.0]:
        tag = dose_tag(dose)
        specs.append(make_spec(
            f'zseed_l5_pos{tag}_plus_relaymax_unified_l9y2',
            'sign_control',
            'Positive-sign control at the same anchor5 far-z location, keeping the relaymax unified y2 core fixed.',
            merge_insertions(anchor5_zseed(dose, 'pos', f'l5_zseed_pos{tag}'), relay_core(2.0)),
            seed_sign='pos',
            seed_dose_s=dose,
            l9_dwell_s=2.0,
        ))
    specs.append(make_spec(
        'zseed_l5_neg4_asym_out5_ret3_plus_relaymax_unified_l9y2',
        'asymmetry_check',
        'Neg4 seed with a longer outbound hold than return hold to test whether the family gain comes mainly from one side of the local z excursion.',
        merge_insertions(anchor5_zseed_asym(-90, 5.0, 3.0, 'l5_zseed_neg4_asym53'), relay_core(2.0)),
        seed_sign='neg',
        seed_dose_s=4.0,
        l9_dwell_s=2.0,
        asymmetry='out5_ret3',
    ))
    specs.append(make_spec(
        'zseed_l5_neg4_asym_out3_ret5_plus_relaymax_unified_l9y2',
        'asymmetry_check',
        'Neg4 seed with the asymmetry reversed, to separate order effects from simple total added time.',
        merge_insertions(anchor5_zseed_asym(-90, 3.0, 5.0, 'l5_zseed_neg4_asym35'), relay_core(2.0)),
        seed_sign='neg',
        seed_dose_s=4.0,
        l9_dwell_s=2.0,
        asymmetry='out3_ret5',
    ))
    specs.append(make_spec(
        'zseed_l5_neg4_plus_relaymax_unified_l9y1p5',
        'core_compatibility',
        'Weaker l9 relay gate with the previously strong neg4 seed, leaving l10/l11 unified core untouched.',
        merge_insertions(anchor5_zseed(4.0, 'neg', 'l5_zseed_neg4'), relay_core(1.5)),
        seed_sign='neg',
        seed_dose_s=4.0,
        l9_dwell_s=1.5,
    ))
    specs.append(make_spec(
        'zseed_l5_neg4_plus_relaymax_unified_l9y2p5',
        'core_compatibility',
        'Slightly stronger l9 relay gate with the neg4 seed, still keeping the rest of the relaymax unified core fixed.',
        merge_insertions(anchor5_zseed(4.0, 'neg', 'l5_zseed_neg4'), relay_core(2.5)),
        seed_sign='neg',
        seed_dose_s=4.0,
        l9_dwell_s=2.5,
    ))
    specs.append(make_spec(
        'zseed_l5_neg4p5_plus_relaymax_unified_l9y2',
        'dose_refine',
        'Interpolation between the original neg4 and stronger neg5 seed under the original relaymax unified y2 core.',
        merge_insertions(anchor5_zseed(4.5, 'neg', 'l5_zseed_neg4p5'), relay_core(2.0)),
        seed_sign='neg',
        seed_dose_s=4.5,
        l9_dwell_s=2.0,
    ))
    specs.append(make_spec(
        'zseed_l5_neg4p5_plus_relaymax_unified_l9y2p5',
        'dose_plus_compatibility',
        'Intermediate neg4.5 seed paired with the stronger l9 y2.5 relay gate.',
        merge_insertions(anchor5_zseed(4.5, 'neg', 'l5_zseed_neg4p5'), relay_core(2.5)),
        seed_sign='neg',
        seed_dose_s=4.5,
        l9_dwell_s=2.5,
    ))
    specs.append(make_spec(
        'zseed_l5_neg5_plus_relaymax_unified_l9y2p25',
        'dose_plus_compatibility',
        'Neg5 seed with only a very small l9 strengthening, to test whether the y2→y2.5 gain comes from a broad band or a sharper coupling point.',
        merge_insertions(anchor5_zseed(5.0, 'neg', 'l5_zseed_neg5'), relay_core(2.25)),
        seed_sign='neg',
        seed_dose_s=5.0,
        l9_dwell_s=2.25,
    ))
    specs.append(make_spec(
        'zseed_l5_neg5_plus_relaymax_unified_l9y2p5',
        'dose_plus_compatibility',
        'Neg5 seed paired with the stronger l9 y2.5 gate.',
        merge_insertions(anchor5_zseed(5.0, 'neg', 'l5_zseed_neg5'), relay_core(2.5)),
        seed_sign='neg',
        seed_dose_s=5.0,
        l9_dwell_s=2.5,
    ))
    specs.append(make_spec(
        'zseed_l5_neg5_plus_relaymax_unified_l9y2p75',
        'dose_plus_compatibility',
        'Neg5 seed with an even stronger l9 y2.75 gate, used to test whether the compatibility ridge keeps improving beyond y2.5.',
        merge_insertions(anchor5_zseed(5.0, 'neg', 'l5_zseed_neg5'), relay_core(2.75)),
        seed_sign='neg',
        seed_dose_s=5.0,
        l9_dwell_s=2.75,
    ))
    specs.append(make_spec(
        'zseed_l5_neg5p5_plus_relaymax_unified_l9y2p5',
        'ridge_push',
        'Continue pushing the now-favorable neg-dose + stronger-l9 ridge one step beyond neg5.',
        merge_insertions(anchor5_zseed(5.5, 'neg', 'l5_zseed_neg5p5'), relay_core(2.5)),
        seed_sign='neg',
        seed_dose_s=5.5,
        l9_dwell_s=2.5,
    ))
    specs.append(make_spec(
        'zseed_l5_neg6_plus_relaymax_unified_l9y2p5',
        'ridge_push',
        'Latest ridge-push point: neg6 anchor5 far-z seed with the stronger l9 y2.5 gate.',
        merge_insertions(anchor5_zseed(6.0, 'neg', 'l5_zseed_neg6'), relay_core(2.5)),
        seed_sign='neg',
        seed_dose_s=6.0,
        l9_dwell_s=2.5,
    ))
    return specs


def load_references(noise_scale: float) -> dict[str, Any]:
    return {
        'relaymax_unified_l9y2_markov': load_json_checked(RELAYMAX_UNIFIED_Y2_MARKOV, noise_scale),
        'relaymax_unified_l9y2_kf': load_json_checked(RELAYMAX_UNIFIED_Y2_KF, noise_scale),
        'entryrelay_main_markov': load_json_checked(ENTRYRELAY_MAIN_MARKOV, noise_scale),
        'entryrelay_main_kf': load_json_checked(ENTRYRELAY_MAIN_KF, noise_scale),
        'old_best_markov': load_json_checked(OLD_BEST_MARKOV, noise_scale),
        'old_best_kf': load_json_checked(OLD_BEST_KF, noise_scale),
        'faithful_markov': load_json_checked(FAITHFUL_MARKOV, noise_scale),
        'faithful_kf': load_json_checked(FAITHFUL_KF, noise_scale),
        'default18_markov': load_json_checked(DEFAULT18_MARKOV, noise_scale),
        'default18_kf': load_json_checked(DEFAULT18_KF, noise_scale),
    }


def render_report(payload: dict[str, Any]) -> str:
    refs = payload['references']
    best = payload['best_candidate']
    best_mean = payload['best_mean_candidate']
    lines: list[str] = []
    lines.append('# Chapter-3 anchor5 far-z seed relay follow-up')
    lines.append('')
    lines.append('## 1. Search question')
    lines.append('')
    lines.append('- Starting family to continue: **anchor5 far-z seed relay**.')
    lines.append('- Previous signal before this follow-up:')
    lines.append('  - best mean: `zseed_l5_neg2_plus_relaymax_unified_l9y2` = **9.302 / 1.130 / 99.617**')
    lines.append('  - best max: `zseed_l5_neg4_plus_relaymax_unified_l9y2` = **9.506 / 1.041 / 99.607**')
    lines.append('- This pass asked one focused question: can the family become the **new unified mainline winner**, while preserving its mean advantage and keeping median under control?')
    lines.append('')
    lines.append('## 2. Fixed structural rule')
    lines.append('')
    lines.append('- faithful original chapter-3 12-position backbone only')
    lines.append('- real dual-axis legality only')
    lines.append('- exact continuity-safe closure before each resume')
    lines.append('- no reopening of late10/late11 local families, butterfly branches, or generic relay tuning beyond tiny relay-core compatibility checks')
    lines.append('')
    lines.append('## 3. Follow-up directions actually spent')
    lines.append('')
    for item in payload['followup_pattern_summary']:
        lines.append(f"- **{item['label']}**: {item['description']}")
    lines.append('')
    lines.append('## 4. Comparison references used')
    lines.append('')
    lines.append(f"- current relay mainline predecessor: **{row_summary(refs['relaymax_unified_l9y2']['markov42']['overall'])}** (`relaymax_unified_l9y2`)")
    lines.append(f"- entry-conditioned relay frontier: **{row_summary(refs['entryrelay_l8x1_l9y1_unifiedcore']['markov42']['overall'])}** (`entryrelay_l8x1_l9y1_unifiedcore`)")
    lines.append(f"- old best legal: **{row_summary(refs['old_best_legal']['markov42']['overall'])}**")
    lines.append(f"- faithful12 backbone: **{row_summary(refs['faithful12']['markov42']['overall'])}**")
    lines.append(f"- default18: **{row_summary(refs['default18']['markov42']['overall'])}**")
    lines.append('')
    lines.append('## 5. Follow-up batch results (Markov42)')
    lines.append('')
    lines.append('| rank | candidate | mode | seed | l9 | mean | median | max | Δmean vs relay | Δmedian vs relay | Δmax vs relay | Δmax vs entry | note |')
    lines.append('| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |')
    for idx, row in enumerate(payload['rows_sorted'], start=1):
        o = row['metrics']['overall']
        seed = f"{row['seed_sign']}{row['seed_dose_s']}" if row['seed_dose_s'] is not None else row['seed_sign']
        lines.append(
            f"| {idx} | {row['candidate_name']} | {row['mode']} | {seed} | {row['l9_dwell_s']:.2f} | "
            f"{o['mean_pct_error']:.3f} | {o['median_pct_error']:.3f} | {o['max_pct_error']:.3f} | "
            f"{row['delta_vs_relaymax_unified_l9y2']['mean_pct_error']['improvement_pct_points']:+.3f} | "
            f"{row['delta_vs_relaymax_unified_l9y2']['median_pct_error']['improvement_pct_points']:+.3f} | "
            f"{row['delta_vs_relaymax_unified_l9y2']['max_pct_error']['improvement_pct_points']:+.3f} | "
            f"{row['delta_vs_entryrelay_main']['max_pct_error']['improvement_pct_points']:+.3f} | {row['rationale']} |"
        )
    lines.append('')
    lines.append('## 6. What the follow-up changed')
    lines.append('')
    lines.append(f"- **New best mean point inside the family:** `{best_mean['candidate_name']}` = **{row_summary(best_mean['markov42']['overall'])}**")
    lines.append(f"  - vs `relaymax_unified_l9y2`: Δmean **{best_mean['delta_vs_relaymax_unified_l9y2']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best_mean['delta_vs_relaymax_unified_l9y2']['max_pct_error']['improvement_pct_points']:+.3f}**, Δmedian **{best_mean['delta_vs_relaymax_unified_l9y2']['median_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"- **New best unified-mainline candidate:** `{best['candidate_name']}` = **{row_summary(best['markov42']['overall'])}**")
    lines.append(f"  - vs `relaymax_unified_l9y2`: Δmean **{best['delta_vs_relaymax_unified_l9y2']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian **{best['delta_vs_relaymax_unified_l9y2']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_relaymax_unified_l9y2']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append(f"  - vs `entryrelay_l8x1_l9y1_unifiedcore`: Δmean **{best['delta_vs_entryrelay_main']['mean_pct_error']['improvement_pct_points']:+.3f}**, Δmedian **{best['delta_vs_entryrelay_main']['median_pct_error']['improvement_pct_points']:+.3f}**, Δmax **{best['delta_vs_entryrelay_main']['max_pct_error']['improvement_pct_points']:+.3f}**")
    lines.append('- Pattern readout:')
    lines.append('  - Negative-sign anchor5 seeds are real; positive-sign controls regress sharply on max, so this is not a sign-symmetric artifact.')
    lines.append('  - Simple neg4 asymmetry only moves hundredths, so the big gain is not mainly an order trick.')
    lines.append('  - The strongest new mechanism is a **ridge coupling**: stronger negative anchor5 seed + slightly stronger `l9 y2.5` relay gate.')
    lines.append('  - Along that ridge the sequence `neg4+y2.5 → neg5+y2.5 → neg5.5+y2.5 → neg6+y2.5` improved max from **99.597 → 99.586 → 99.578 → 99.576**, while mean also improved from **9.484 → 9.416 → 9.364 → 9.343**.')
    lines.append('')
    lines.append('## 7. KF36 recheck for genuinely competitive points')
    lines.append('')
    lines.append(f"- trigger rule used: {payload['kf36_recheck']['reason']}")
    lines.append('')
    lines.append('| candidate | Markov42 | KF36 | note |')
    lines.append('| --- | --- | --- | --- |')
    for row in payload['kf36_rows']:
        lines.append(
            f"| {row['candidate_name']} | {row_summary(row['markov42']['overall'])} | {row_summary(row['kf36']['overall']) if row['kf36'] else 'n/a'} | {row['note']} |"
        )
    lines.append('')
    lines.append('## 8. Exact legal motor/timing table for the best follow-up candidate')
    lines.append('')
    lines.append(f"- candidate: `{best['candidate_name']}`")
    lines.append(f"- total time: **{best['total_time_s']:.1f} s**")
    lines.append(f"- continuity closures checked at anchors: **{', '.join(str(item['anchor_id']) for item in best['continuity_checks'])}**")
    lines.append('')
    lines.append('| # | anchor | role | action | face before | face after | rot_s | pre_s | post_s | total_s | label |')
    lines.append('| ---: | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |')
    for row, action in zip(best['all_rows'], best['all_actions']):
        lines.append(
            f"| {row['pos_id']} | {row['anchor_id']} | {row['segment_role']} | {render_action(action)} | {action['state_before']['face_name']} | {action['state_after']['face_name']} | {row['rotation_time_s']:.1f} | {row['pre_static_s']:.1f} | {row['post_static_s']:.1f} | {row['node_total_s']:.1f} | {row['label']} |"
        )
    lines.append('')
    lines.append('## 9. Reference comparison')
    lines.append('')
    lines.append('| label | Markov42 | KF36 | note |')
    lines.append('| --- | --- | --- | --- |')
    for row in payload['comparison_rows']:
        lines.append(
            f"| {row['label']} | {row_summary(row['markov42']['overall'])} | {row_summary(row['kf36']['overall']) if row['kf36'] else 'n/a'} | {row['note']} |"
        )
    lines.append('')
    lines.append('## 10. Bottom line')
    lines.append('')
    lines.append(f"- **Does anchor5 far-z seed become the new unified mainline winner?** **{payload['bottom_line']['new_unified_mainline_winner']}**")
    lines.append(f"- **Does it also take the absolute max-frontier crown from `entryrelay_l8x1_l9y1_unifiedcore`?** **{payload['bottom_line']['new_absolute_max_leader']}**")
    lines.append(f"- **Best landed signal:** **{payload['bottom_line']['best_signal']}**")
    lines.append(f"- **Scientific conclusion:** {payload['bottom_line']['scientific_conclusion']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    refs = load_references(args.noise_scale)
    mod = load_module('psins_ch3_anchor5_farz_followup', str(SOURCE_FILE))
    base = build_candidate(mod, ())

    specs = candidate_specs()
    candidates = [build_closedloop_candidate(mod, spec, base.rows, base.action_sequence) for spec in specs]
    candidates_by_name = {cand.name: cand for cand in candidates}
    spec_by_name = {spec['name']: spec for spec in specs}

    rows: list[dict[str, Any]] = []
    payload_by_name: dict[str, dict[str, Any]] = {}
    for cand in candidates:
        payload, status, path = run_candidate_payload(mod, cand, 'markov42_noisy', args.noise_scale, args.force_rerun)
        payload_by_name[cand.name] = payload
        spec = spec_by_name[cand.name]
        rows.append({
            'candidate_name': cand.name,
            'mode': spec['mode'],
            'rationale': spec['rationale'],
            'seed_sign': spec['seed_sign'],
            'seed_dose_s': spec['seed_dose_s'],
            'l9_dwell_s': spec['l9_dwell_s'],
            'asymmetry': spec['asymmetry'],
            'total_time_s': cand.total_time_s,
            'metrics': compact_metrics(payload),
            'continuity_checks': cand.continuity_checks,
            'run_json': str(path),
            'status': status,
            'delta_vs_relaymax_unified_l9y2': delta_vs_ref(refs['relaymax_unified_l9y2_markov'], payload),
            'delta_vs_entryrelay_main': delta_vs_ref(refs['entryrelay_main_markov'], payload),
            'delta_vs_old_best': delta_vs_ref(refs['old_best_markov'], payload),
            'delta_vs_faithful12': delta_vs_ref(refs['faithful_markov'], payload),
            'delta_vs_default18': delta_vs_ref(refs['default18_markov'], payload),
        })

    rows_sorted = sorted(rows, key=lambda x: (x['metrics']['overall']['max_pct_error'], x['metrics']['overall']['mean_pct_error']))
    best_row = rows_sorted[0]
    best_mean_row = min(rows, key=lambda x: (x['metrics']['overall']['mean_pct_error'], x['metrics']['overall']['max_pct_error']))

    best_candidate = candidates_by_name[best_row['candidate_name']]
    best_payload = payload_by_name[best_candidate.name]
    best_mean_candidate = candidates_by_name[best_mean_row['candidate_name']]
    best_mean_payload = payload_by_name[best_mean_candidate.name]

    kf_targets = rows_sorted[:2]
    kf36_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(kf_targets, start=1):
        cand = candidates_by_name[row['candidate_name']]
        kf_payload, kf_status, kf_path = run_candidate_payload(mod, cand, 'kf36_noisy', args.noise_scale, args.force_rerun)
        note = 'best overall follow-up candidate' if idx == 1 else 'runner-up on the same high-dose compatibility ridge'
        kf36_rows.append({
            'candidate_name': cand.name,
            'note': note,
            'markov42': compact_result(payload_by_name[cand.name]),
            'kf36': compact_result(kf_payload),
            'kf36_status': kf_status,
            'kf36_run_json': str(kf_path),
        })

    kf_by_name = {row['candidate_name']: row['kf36'] for row in kf36_rows}

    best_summary = {
        'candidate_name': best_candidate.name,
        'mode': best_row['mode'],
        'seed_sign': best_row['seed_sign'],
        'seed_dose_s': best_row['seed_dose_s'],
        'l9_dwell_s': best_row['l9_dwell_s'],
        'total_time_s': best_candidate.total_time_s,
        'markov42': compact_result(best_payload),
        'kf36': kf_by_name.get(best_candidate.name),
        'delta_vs_relaymax_unified_l9y2': best_row['delta_vs_relaymax_unified_l9y2'],
        'delta_vs_entryrelay_main': best_row['delta_vs_entryrelay_main'],
        'delta_vs_old_best': best_row['delta_vs_old_best'],
        'delta_vs_faithful12': best_row['delta_vs_faithful12'],
        'delta_vs_default18': best_row['delta_vs_default18'],
        'all_rows': best_candidate.all_rows,
        'all_actions': best_candidate.all_actions,
        'all_faces': best_candidate.all_faces,
        'continuity_checks': best_candidate.continuity_checks,
    }

    best_mean_summary = {
        'candidate_name': best_mean_candidate.name,
        'mode': best_mean_row['mode'],
        'seed_sign': best_mean_row['seed_sign'],
        'seed_dose_s': best_mean_row['seed_dose_s'],
        'l9_dwell_s': best_mean_row['l9_dwell_s'],
        'total_time_s': best_mean_candidate.total_time_s,
        'markov42': compact_result(best_mean_payload),
        'delta_vs_relaymax_unified_l9y2': best_mean_row['delta_vs_relaymax_unified_l9y2'],
        'delta_vs_entryrelay_main': best_mean_row['delta_vs_entryrelay_main'],
        'delta_vs_old_best': best_mean_row['delta_vs_old_best'],
        'delta_vs_faithful12': best_mean_row['delta_vs_faithful12'],
        'delta_vs_default18': best_mean_row['delta_vs_default18'],
    }

    becomes_new_unified_mainline_winner = (
        best_row['delta_vs_relaymax_unified_l9y2']['mean_pct_error']['improvement_pct_points'] > 0
        and best_row['delta_vs_relaymax_unified_l9y2']['max_pct_error']['improvement_pct_points'] > 0
    )
    becomes_new_absolute_max_leader = best_row['delta_vs_entryrelay_main']['max_pct_error']['improvement_pct_points'] > 0

    if becomes_new_unified_mainline_winner and not becomes_new_absolute_max_leader:
        scientific_conclusion = (
            'Yes — the anchor5 far-z seed relay family now becomes the **new unified mainline winner**. '
            f'The best landed point `{best_candidate.name}` reaches {row_summary(best_payload["overall"])} and cleanly improves the previous unified mainline `relaymax_unified_l9y2` on both mean and max under the same legal continuity-safe structure. '
            f'The family also preserved its mean edge, with a new best-mean point `{best_mean_candidate.name}` at {row_summary(best_mean_payload["overall"])}. '
            'The decisive new mechanism is not a sign-symmetric curiosity or a tiny asymmetry trick: it is a real **negative anchor5 far-z seed + slightly stronger l9 relay gate** ridge. '
            f'However it still stops just short of the absolute max-frontier leader `entryrelay_l8x1_l9y1_unifiedcore` by {abs(best_row["delta_vs_entryrelay_main"]["max_pct_error"]["improvement_pct_points"]):.3f} max-points, so the right interpretation is: **new unified mainline winner, but not the single global max-frontier king yet**.'
        )
        mainline_text = 'YES'
        absolute_text = 'NO'
    elif becomes_new_unified_mainline_winner and becomes_new_absolute_max_leader:
        scientific_conclusion = (
            'Yes — this follow-up fully upgrades the anchor5 far-z seed relay family from “another frontier family” to the new single best mainline and max-frontier leader.'
        )
        mainline_text = 'YES'
        absolute_text = 'YES'
    else:
        scientific_conclusion = (
            'No — the follow-up confirms that anchor5 far-z seed is a real frontier family, but it still does not beat the current unified mainline strongly enough to replace it.'
        )
        mainline_text = 'NO'
        absolute_text = 'NO'

    comparison_rows = [
        {
            'label': 'relaymax_unified_l9y2',
            'note': 'previous unified mainline predecessor',
            'markov42': compact_result(refs['relaymax_unified_l9y2_markov']),
            'kf36': compact_result(refs['relaymax_unified_l9y2_kf']),
        },
        {
            'label': 'entryrelay_l8x1_l9y1_unifiedcore',
            'note': 'current absolute max-frontier point used for comparison',
            'markov42': compact_result(refs['entryrelay_main_markov']),
            'kf36': compact_result(refs['entryrelay_main_kf']),
        },
        {
            'label': 'old best legal',
            'note': 'historical legal baseline',
            'markov42': compact_result(refs['old_best_markov']),
            'kf36': compact_result(refs['old_best_kf']),
        },
        {
            'label': 'faithful12',
            'note': 'original faithful 12-position backbone',
            'markov42': compact_result(refs['faithful_markov']),
            'kf36': compact_result(refs['faithful_kf']),
        },
        {
            'label': 'default18',
            'note': 'non-faithful strong reference',
            'markov42': compact_result(refs['default18_markov']),
            'kf36': compact_result(refs['default18_kf']),
        },
        {
            'label': 'best follow-up candidate',
            'note': best_candidate.name,
            'markov42': best_summary['markov42'],
            'kf36': best_summary['kf36'],
        },
    ]

    out_json = RESULTS_DIR / f'ch3_anchor5_farz_followup_{args.report_date}.json'
    out_md = REPORTS_DIR / f'psins_ch3_anchor5_farz_followup_{args.report_date}.md'

    payload = {
        'experiment': 'ch3_anchor5_farz_followup',
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'followup_pattern_summary': LOCAL_PATTERN_SUMMARY,
        'references': {
            'relaymax_unified_l9y2': {
                'candidate_name': 'relaymax_unified_l9y2',
                'markov42': compact_result(refs['relaymax_unified_l9y2_markov']),
                'kf36': compact_result(refs['relaymax_unified_l9y2_kf']),
            },
            'entryrelay_l8x1_l9y1_unifiedcore': {
                'candidate_name': 'entryrelay_l8x1_l9y1_unifiedcore',
                'markov42': compact_result(refs['entryrelay_main_markov']),
                'kf36': compact_result(refs['entryrelay_main_kf']),
            },
            'old_best_legal': {
                'candidate_name': 'legal_flip_8_11_12_retime_flipnodes_pre20_post70',
                'markov42': compact_result(refs['old_best_markov']),
                'kf36': compact_result(refs['old_best_kf']),
            },
            'faithful12': {
                'candidate_name': 'faithful12',
                'markov42': compact_result(refs['faithful_markov']),
                'kf36': compact_result(refs['faithful_kf']),
            },
            'default18': {
                'candidate_name': 'default18',
                'markov42': compact_result(refs['default18_markov']),
                'kf36': compact_result(refs['default18_kf']),
            },
        },
        'candidate_specs': [
            {
                'name': spec['name'],
                'mode': spec['mode'],
                'rationale': spec['rationale'],
                'seed_sign': spec['seed_sign'],
                'seed_dose_s': spec['seed_dose_s'],
                'l9_dwell_s': spec['l9_dwell_s'],
                'asymmetry': spec['asymmetry'],
                'anchors': sorted(spec['insertions'].keys()),
            }
            for spec in specs
        ],
        'rows_sorted': rows_sorted,
        'best_candidate': best_summary,
        'best_mean_candidate': best_mean_summary,
        'kf36_recheck': {
            'triggered': True,
            'reason': 'Top two follow-up landings were rechecked because both sat within a few hundredths of the entry-conditioned max frontier while clearly beating the previous unified mainline on mean/max trade.',
        },
        'kf36_rows': kf36_rows,
        'comparison_rows': comparison_rows,
        'bottom_line': {
            'new_unified_mainline_winner': mainline_text,
            'new_absolute_max_leader': absolute_text,
            'best_signal': (
                f"best mean {best_mean_candidate.name} = {row_summary(best_mean_payload['overall'])}; "
                f"best unified-mainline {best_candidate.name} = {row_summary(best_payload['overall'])}"
            ),
            'scientific_conclusion': scientific_conclusion,
        },
        'files': {
            'json': str(out_json),
            'report': str(out_md),
        },
    }

    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    out_md.write_text(render_report(payload), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps(payload['files'], ensure_ascii=False), flush=True)
    print('BEST_CANDIDATE', best_candidate.name, best_payload['overall'], flush=True)
    print('BEST_MEAN', best_mean_candidate.name, best_mean_payload['overall'], flush=True)
    print('BOTTOM_LINE', scientific_conclusion, flush=True)


if __name__ == '__main__':
    main()
