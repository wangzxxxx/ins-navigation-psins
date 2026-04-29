from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

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
from benchmark_ch3_12pos_goalA_repairs import rows_to_paras
from compare_ch3_12pos_path_baselines import build_ch3_initial_attitude
from compare_four_methods_shared_noise import compute_payload, expected_noise_config
from compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors import _param_specs
from search_ch3_12pos_closedloop_local_insertions import build_closedloop_candidate, closed_pair
from search_ch3_12pos_closedloop_zquad_followup import xpair_outerhold, zquad_split
from search_ch3_12pos_legal_dualaxis_repairs import build_candidate

NOISE_SCALE = 0.08
REPORT_DATE = datetime.now().strftime('%Y-%m-%d')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=NOISE_SCALE)
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def make_suffix(noise_scale: float) -> str:
    if abs(noise_scale - 0.08) < 1e-12:
        return 'noise0p08'
    return f"noise{str(noise_scale).replace('.', 'p')}"


def build_dataset_with_path_whiteonly(mod, noise_scale: float, paras):
    ts = 0.01
    att0 = build_ch3_initial_attitude(mod)
    pos0 = mod.posset(34.0, 0.0, 0.0)
    att = mod.attrottt(att0, paras, ts)
    imu, _ = mod.avp2imu(att, pos0)
    clbt_truth = mod.get_default_clbt()
    imu_clean = mod.imuclbt(imu, clbt_truth)

    cfg = expected_noise_config(noise_scale)
    arw = cfg['arw_dpsh'] * mod.glv.dpsh
    vrw = cfg['vrw_ugpsHz'] * mod.glv.ugpsHz
    bi_g = 0.0
    bi_a = 0.0

    imu_noisy = mod.imuadderr_full(
        imu_clean,
        ts,
        arw=arw,
        vrw=vrw,
        bi_g=bi_g,
        tau_g=cfg['tau_g'],
        bi_a=bi_a,
        tau_a=cfg['tau_a'],
        seed=cfg['seed'],
    )

    white_cfg = dict(cfg)
    white_cfg['bi_g_dph'] = 0.0
    white_cfg['bi_a_ug'] = 0.0
    white_cfg['mode'] = 'white_noise_only'

    return {
        'ts': ts,
        'pos0': pos0,
        'imu_noisy': imu_noisy,
        'bi_g': bi_g,
        'bi_a': bi_a,
        'tau_g': cfg['tau_g'],
        'tau_a': cfg['tau_a'],
        'noise_scale': noise_scale,
        'noise_config': white_cfg,
    }


def candidate_result_path(tag: str, method_key: str, noise_scale: float) -> Path:
    suffix = make_suffix(noise_scale)
    prefix = 'M_markov_42state_gm1' if method_key == 'markov42_noisy' else 'KF36'
    return RESULTS_DIR / f'{prefix}_{tag}_whiteonly_{suffix}_param_errors.json'


def run_payload(mod, tag: str, candidate_name: str, rows, noise_scale: float, method_key: str, force_rerun: bool = False):
    out_path = candidate_result_path(tag, method_key, noise_scale)
    if out_path.exists() and not force_rerun:
        with open(out_path, 'r', encoding='utf-8') as f:
            return json.load(f), out_path

    paras = rows_to_paras(mod, rows)
    dataset = build_dataset_with_path_whiteonly(mod, noise_scale, paras)
    params = _param_specs(mod)

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
            label=f'{candidate_name}_{method_key}_whiteonly_{make_suffix(noise_scale)}',
        )
    elif method_key == 'kf36_noisy':
        clbt, _, _, _, _ = mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=36,
            label=f'{candidate_name}_{method_key}_whiteonly_{make_suffix(noise_scale)}',
        )
    else:
        raise KeyError(method_key)

    payload = compute_payload(
        mod,
        clbt,
        params,
        variant=f'{candidate_name}_{method_key}_whiteonly_{make_suffix(noise_scale)}',
        method_file='compare_ch3_current_vs_faithful_whiteonly.py',
        extra={
            'candidate_name': candidate_name,
            'method_key': method_key,
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'comparison_mode': 'white_noise_only',
        },
    )
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return payload, out_path


def pack(payload: dict) -> str:
    o = payload['overall']
    return f"{o['mean_pct_error']:.3f} / {o['median_pct_error']:.3f} / {o['max_pct_error']:.3f}"


def main() -> None:
    args = parse_args()
    mod = load_module(str(METHOD_DIR / 'method_42state_gm1.py'), str(SOURCE_FILE))

    faithful = build_candidate(mod, ())
    current_spec = {
        'name': 'twoanchor_l10_zpair_neg6_plus_l11_bestmean',
        'rationale': 'Current latest best-mean two-anchor branch discussed in-session: anchor10 adds a mild negative z-pair; anchor11 uses the current best mean late11 zquad motif.',
        'insertions': {
            10: closed_pair('outer', -90, 5.0, 6.0, 'l10_zpair_neg6'),
            11: xpair_outerhold(10.0, 'yneg_xpair_outerhold') + zquad_split(10.0, 2.0, 10.0, 2.0, 'zquad_y10x2'),
        },
    }
    current = build_closedloop_candidate(mod, current_spec, faithful.rows, faithful.action_sequence)

    candidates = [
        ('faithful12_original', 'faithful12_original', faithful.rows),
        ('current_twoanchor_bestmean', current.name, current.all_rows),
    ]

    summary_rows = []
    for tag, name, rows in candidates:
        markov_payload, markov_path = run_payload(mod, tag, name, rows, args.noise_scale, 'markov42_noisy', args.force_rerun)
        kf_payload, kf_path = run_payload(mod, tag, name, rows, args.noise_scale, 'kf36_noisy', args.force_rerun)
        summary_rows.append({
            'tag': tag,
            'candidate_name': name,
            'markov42': markov_payload,
            'kf36': kf_payload,
            'files': {
                'markov42': str(markov_path),
                'kf36': str(kf_path),
            },
        })

    faithful_row = next(r for r in summary_rows if r['tag'] == 'faithful12_original')
    current_row = next(r for r in summary_rows if r['tag'] == 'current_twoanchor_bestmean')

    def delta(cur: float, base: float) -> float:
        return base - cur

    report_lines = []
    report_lines.append('# Chapter-3 current strategy vs faithful12 under white-noise-only')
    report_lines.append('')
    report_lines.append('## 1. Benchmark condition')
    report_lines.append('')
    report_lines.append('- Compared two strategies under **white-noise-only** setting.')
    report_lines.append('- Same path truth family and same seed as current shared benchmark.')
    report_lines.append('- Kept **ARW / VRW** white noise.')
    report_lines.append('- Removed random-drift terms by setting **`bi_g = bi_a = 0`**.')
    report_lines.append(f"- noise_scale = `{args.noise_scale}`")
    report_lines.append('- Compared strategies:')
    report_lines.append('  1. original faithful12')
    report_lines.append('  2. current latest best-mean two-anchor strategy (`twoanchor_l10_zpair_neg6_plus_l11_bestmean`)')
    report_lines.append('')
    report_lines.append('## 2. Overall comparison')
    report_lines.append('')
    report_lines.append('| strategy | Markov42 mean/median/max | KF36 mean/median/max | dKa_yy | dKg_zz | Ka2_y | Ka2_z |')
    report_lines.append('|---|---:|---:|---:|---:|---:|---:|')
    for row in summary_rows:
        mk = row['markov42']['param_errors']
        report_lines.append(
            f"| {row['candidate_name']} | {pack(row['markov42'])} | {pack(row['kf36'])} | {mk['dKa_yy']['pct_error']:.3f} | {mk['dKg_zz']['pct_error']:.3f} | {mk['Ka2_y']['pct_error']:.3f} | {mk['Ka2_z']['pct_error']:.3f} |"
        )
    report_lines.append('')
    report_lines.append('## 3. Delta: current strategy vs faithful12')
    report_lines.append('')
    f_ov = faithful_row['markov42']['overall']
    c_ov = current_row['markov42']['overall']
    f_pk = faithful_row['markov42']['param_errors']
    c_pk = current_row['markov42']['param_errors']
    report_lines.append(f"- Markov42 overall Δ = **{delta(c_ov['mean_pct_error'], f_ov['mean_pct_error']):+.3f} / {delta(c_ov['median_pct_error'], f_ov['median_pct_error']):+.3f} / {delta(c_ov['max_pct_error'], f_ov['max_pct_error']):+.3f}**")
    report_lines.append(f"- dKa_yy Δ = **{delta(c_pk['dKa_yy']['pct_error'], f_pk['dKa_yy']['pct_error']):+.3f}**")
    report_lines.append(f"- dKg_zz Δ = **{delta(c_pk['dKg_zz']['pct_error'], f_pk['dKg_zz']['pct_error']):+.3f}**")
    report_lines.append(f"- Ka2_y Δ = **{delta(c_pk['Ka2_y']['pct_error'], f_pk['Ka2_y']['pct_error']):+.3f}**")
    report_lines.append(f"- Ka2_z Δ = **{delta(c_pk['Ka2_z']['pct_error'], f_pk['Ka2_z']['pct_error']):+.3f}**")
    report_lines.append('')
    report_lines.append('## 4. Bottom line')
    report_lines.append('')
    report_lines.append('- This comparison isolates the role of **white noise only**, without GM/random-drift terms.')
    report_lines.append('- If the current strategy still beats faithful12 here, that means its advantage is not only from handling colored/random drift, but also from giving a cleaner deterministic/white-noise observability structure.')
    report_lines.append('')
    report_lines.append('## 5. Files')
    report_lines.append('')
    for row in summary_rows:
        report_lines.append(f"- {row['candidate_name']} Markov42: `{row['files']['markov42']}`")
        report_lines.append(f"- {row['candidate_name']} KF36: `{row['files']['kf36']}`")

    out_json = RESULTS_DIR / f'ch3_current_vs_faithful_whiteonly_{make_suffix(args.noise_scale)}.json'
    out_report = REPORTS_DIR / f'psins_ch3_current_vs_faithful_whiteonly_{REPORT_DATE}.md'

    payload = {
        'experiment': 'ch3_current_vs_faithful_whiteonly',
        'report_date': REPORT_DATE,
        'noise_scale': args.noise_scale,
        'noise_mode': 'white_noise_only',
        'noise_config': {
            **expected_noise_config(args.noise_scale),
            'bi_g_dph': 0.0,
            'bi_a_ug': 0.0,
        },
        'strategies': summary_rows,
        'files': {
            'report': str(out_report),
            'json': str(out_json),
        },
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    out_report.write_text('\n'.join(report_lines) + '\n', encoding='utf-8')
    print(str(out_json))
    print(str(out_report))


if __name__ == '__main__':
    main()
