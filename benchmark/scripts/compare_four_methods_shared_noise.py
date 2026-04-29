from __future__ import annotations

import argparse
import copy
import json
import sys
import types
from pathlib import Path

if 'matplotlib' not in sys.modules:
    matplotlib_stub = types.ModuleType('matplotlib')
    pyplot_stub = types.ModuleType('matplotlib.pyplot')
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules['matplotlib'] = matplotlib_stub
    sys.modules['matplotlib.pyplot'] = pyplot_stub
if 'seaborn' not in sys.modules:
    sys.modules['seaborn'] = types.ModuleType('seaborn')

ROOT = Path('/root/.openclaw/workspace')
METHOD_DIR = ROOT / 'psins_method_bench' / 'methods' / 'markov'
SCRIPTS_DIR = ROOT / 'psins_method_bench' / 'scripts'
RESULTS_DIR = ROOT / 'psins_method_bench' / 'results'
REPORTS_DIR = ROOT / 'reports'
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'
R61_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round61_h_scd_state20_microtight_commit.py'
COMPUTE_R61_FILE = SCRIPTS_DIR / 'compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors.py'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import load_module
from probe_round55_newline import _build_patched_method
from probe_round59_h_scd_hybrid import _run_internalized_hybrid_scd
from probe_round61_hybrid_micro import ROUND61_CANDIDATES, _merge_round61_candidate

BASE_ARW = 0.005
BASE_VRW = 5.0
BASE_BI_G = 0.002
BASE_BI_A = 5.0
TAU_G = 300.0
TAU_A = 300.0
ROUND61_CANDIDATE_NAME = 'r61_s20_08988_ryz00116'
METHOD_ORDER = ['kf36_noisy', 'markov42_noisy', 'scd42_neutral', 'round61']
METHOD_DISPLAY = {
    'kf36_noisy': 'Standard KF baseline (36-state noisy baseline)',
    'markov42_noisy': 'Pure Markov (42-state GM1 baseline)',
    'scd42_neutral': 'Pure SCD baseline (42-state GM1 neutral Markov+SCD)',
    'round61': 'Round61 (42-state GM1 round61_h_scd_state20_microtight_commit)',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=0.08)
    parser.add_argument('--report-date', type=str, default='2026-03-28')
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


def expected_noise_config(noise_scale: float):
    return {
        'arw_dpsh': BASE_ARW * noise_scale,
        'vrw_ugpsHz': BASE_VRW * noise_scale,
        'bi_g_dph': BASE_BI_G * noise_scale,
        'bi_a_ug': BASE_BI_A * noise_scale,
        'tau_g': TAU_G,
        'tau_a': TAU_A,
        'seed': 42,
        'base_family': 'round53_round61_shared',
    }


def build_output_paths(noise_scale: float, report_date: str):
    suffix = make_suffix(noise_scale)
    paths = {
        'kf36_noisy': RESULTS_DIR / f'KF36_shared_{suffix}_param_errors.json',
        'markov42_noisy': RESULTS_DIR / f'M_markov_42state_gm1_shared_{suffix}_param_errors.json',
        'scd42_neutral': RESULTS_DIR / f'SCD42_markov_neutral_shared_{suffix}_param_errors.json',
        'round61': RESULTS_DIR / f'R61_42state_gm1_round61_h_scd_state20_microtight_commit_shared_{suffix}_param_errors.json',
        'compare': RESULTS_DIR / f'compare_four_methods_shared_{suffix}.json',
        'compact': RESULTS_DIR / f'compare_four_methods_shared_{suffix}_compact.json',
        'report': REPORTS_DIR / f'psins_four_methods_shared_{suffix}_full_params_{report_date}.md',
    }
    return suffix, paths


def build_shared_dataset(mod, noise_scale: float):
    ts = 0.01
    att0 = mod.np.array([1.0, -91.0, -91.0]) * mod.glv.deg
    pos0 = mod.posset(34.0, 0.0, 0.0)
    paras = mod.np.array([
        [1, 0, 1, 0, 90, 9, 70, 70], [2, 0, 1, 0, 90, 9, 20, 20], [3, 0, 1, 0, 90, 9, 20, 20],
        [4, 0, 1, 0, -90, 9, 20, 20], [5, 0, 1, 0, -90, 9, 20, 20], [6, 0, 1, 0, -90, 9, 20, 20],
        [7, 0, 0, 1, 90, 9, 20, 20], [8, 1, 0, 0, 90, 9, 20, 20], [9, 1, 0, 0, 90, 9, 20, 20],
        [10, 1, 0, 0, 90, 9, 20, 20], [11, -1, 0, 0, 90, 9, 20, 20], [12, -1, 0, 0, 90, 9, 20, 20],
        [13, -1, 0, 0, 90, 9, 20, 20], [14, 0, 0, 1, 90, 9, 20, 20], [15, 0, 0, 1, 90, 9, 20, 20],
        [16, 0, 0, -1, 90, 9, 20, 20], [17, 0, 0, -1, 90, 9, 20, 20], [18, 0, 0, -1, 90, 9, 20, 20],
    ], dtype=float)
    paras[:, 4] = paras[:, 4] * mod.glv.deg
    att = mod.attrottt(att0, paras, ts)
    imu, _ = mod.avp2imu(att, pos0)
    clbt_truth = mod.get_default_clbt()
    imu_clean = mod.imuclbt(imu, clbt_truth)

    arw = BASE_ARW * noise_scale * mod.glv.dpsh
    vrw = BASE_VRW * noise_scale * mod.glv.ugpsHz
    bi_g = BASE_BI_G * noise_scale * mod.glv.dph
    bi_a = BASE_BI_A * noise_scale * mod.glv.ug
    imu_noisy = mod.imuadderr_full(
        imu_clean,
        ts,
        arw=arw,
        vrw=vrw,
        bi_g=bi_g,
        tau_g=TAU_G,
        bi_a=bi_a,
        tau_a=TAU_A,
        seed=42,
    )

    return {
        'ts': ts,
        'pos0': pos0,
        'imu_noisy': imu_noisy,
        'bi_g': bi_g,
        'bi_a': bi_a,
        'tau_g': TAU_G,
        'tau_a': TAU_A,
        'noise_scale': noise_scale,
        'noise_config': expected_noise_config(noise_scale),
    }


def compute_payload(source_mod, clbt, params, variant: str, method_file: str, extra=None):
    param_errors = {}
    pct_values = []
    for name, true_v, get_est in params:
        true_f = float(true_v)
        est_f = float(get_est(clbt))
        abs_err = abs(true_f - est_f)
        pct_err = abs_err / abs(true_f) * 100.0 if abs(true_f) > 1e-15 else 0.0
        param_errors[name] = {
            'true': true_f,
            'est': est_f,
            'abs_error': abs_err,
            'pct_error': pct_err,
        }
        pct_values.append(pct_err)

    pct_arr = source_mod.np.asarray(pct_values, dtype=float)
    focus_scale_pct = {
        'dKg_xx': param_errors['dKg_xx']['pct_error'],
        'dKg_xy': param_errors['dKg_xy']['pct_error'],
        'dKg_yy': param_errors['dKg_yy']['pct_error'],
        'dKg_zz': param_errors['dKg_zz']['pct_error'],
        'dKa_xx': param_errors['dKa_xx']['pct_error'],
    }
    lever_guard_pct = {
        'rx_y': param_errors['rx_y']['pct_error'],
        'ry_z': param_errors['ry_z']['pct_error'],
    }
    overall = {
        'mean_pct_error': float(source_mod.np.mean(pct_arr)),
        'median_pct_error': float(source_mod.np.median(pct_arr)),
        'max_pct_error': float(source_mod.np.max(pct_arr)),
    }
    return {
        'variant': variant,
        'method_file': method_file,
        'source_file': str(SOURCE_FILE),
        'param_order': [name for name, _, _ in params],
        'param_errors': param_errors,
        'focus_scale_pct': focus_scale_pct,
        'lever_guard_pct': lever_guard_pct,
        'overall': overall,
        'extra': extra or {},
    }


def _pick_round61_candidate():
    for candidate in ROUND61_CANDIDATES:
        if candidate['name'] == ROUND61_CANDIDATE_NAME:
            return candidate
    raise KeyError(f'Round61 candidate not found: {ROUND61_CANDIDATE_NAME}')


def _build_neutral_scd_candidate():
    neutral_policy_patch = {
        'selected_prior_scale': 1.0,
        'other_scale_prior_scale': 1.0,
        'ka2_prior_scale': 1.0,
        'lever_prior_scale': 1.0,
        'selected_q_static_scale': 1.0,
        'selected_q_dynamic_scale': 1.0,
        'selected_q_late_mult': 1.0,
        'other_scale_q_scale': 1.0,
        'other_scale_q_late_mult': 1.0,
        'ka2_q_scale': 1.0,
        'lever_q_scale': 1.0,
        'static_r_scale': 1.0,
        'dynamic_r_scale': 1.0,
        'late_r_mult': 1.0,
        'late_release_frac': 0.58,
        'selected_alpha_floor': 1.0,
        'selected_alpha_span': 0.0,
        'other_scale_alpha': 1.0,
        'ka2_alpha': 1.0,
        'lever_alpha': 1.0,
        'markov_alpha': 1.0,
        'trust_score_soft': 2.1,
        'trust_cov_soft': 0.44,
        'trust_mix': 0.58,
        'state_alpha_mult': {},
        'state_alpha_add': {},
        'state_prior_diag_mult': {},
        'state_q_static_mult': {},
        'state_q_dynamic_mult': {},
        'state_q_late_mult': {},
    }
    return {
        'name': 'neutral_markov42_plus_scd_baseline',
        'description': 'Controlled neutral Markov+SCD baseline (iter2 once-per-phase SCD only).',
        'iter_patches': {
            0: copy.deepcopy(neutral_policy_patch),
            1: copy.deepcopy(neutral_policy_patch),
        },
        'post_rx_y_mult': 1.0,
        'post_ry_z_mult': 1.0,
        'scd': {
            'mode': 'once_per_phase',
            'alpha': 0.999,
            'transition_duration': 2.0,
            'target': 'scale_block',
            'bias_to_target': True,
            'apply_policy_names': ['iter2_commit'],
        },
    }


def _load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _noise_matches(payload: dict, expected_cfg: dict):
    extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
    cfg = extra.get('noise_config') or extra.get('dataset_noise_config')
    if not isinstance(cfg, dict):
        return False

    numeric_keys = ['arw_dpsh', 'vrw_ugpsHz', 'bi_g_dph', 'bi_a_ug', 'tau_g', 'tau_a']
    for key in numeric_keys:
        if key not in cfg:
            return False
        if abs(float(cfg[key]) - float(expected_cfg[key])) > 1e-12:
            return False

    if int(cfg.get('seed', -1)) != int(expected_cfg['seed']):
        return False
    if cfg.get('base_family') != expected_cfg['base_family']:
        return False

    return True


def _fmt_e(x: float) -> str:
    return f'{x:.6e}'


def _fmt_pct(x: float) -> str:
    return f'{x:.6f}'


def render_report(compare: dict) -> str:
    lines = []
    lines.append('<callout emoji="🧪" background-color="light-blue">')
    lines.append('同一固定 shared dataset（noise0p08）下的四方法全参数对照：`KF36 / Markov42 / Pure SCD baseline / Round61`。')
    lines.append('</callout>')
    lines.append('')

    cfg = compare['noise_config']
    lines.append('## 1. 固定 noise 设置（shared noise0p08）')
    lines.append('')
    lines.append(f"- `arw = {cfg['arw_dpsh']} dpsh`")
    lines.append(f"- `vrw = {cfg['vrw_ugpsHz']} ugpsHz`")
    lines.append(f"- `bi_g = {cfg['bi_g_dph']} dph`")
    lines.append(f"- `bi_a = {cfg['bi_a_ug']} ug`")
    lines.append(f"- `tau_g = tau_a = {cfg['tau_g']}`")
    lines.append(f"- `seed = {cfg['seed']}`")
    lines.append(f"- `base_family = {cfg['base_family']}`")
    lines.append('')

    lines.append('## 2. 方法定义（本次对照）')
    lines.append('')
    for method_key in METHOD_ORDER:
        m = compare['methods'][method_key]
        lines.append(f"- **{METHOD_DISPLAY[method_key]}**：{m['definition']}")
    lines.append('')

    lines.append('## 3. Overall 指标对比（30 参数）')
    lines.append('')
    lines.append('| 方法 | mean% | median% | max% |')
    lines.append('|---|---:|---:|---:|')
    for method_key in METHOD_ORDER:
        ov = compare['overall']['by_method'][method_key]
        lines.append(f"| {METHOD_DISPLAY[method_key]} | {_fmt_pct(ov['mean_pct_error'])} | {_fmt_pct(ov['median_pct_error'])} | {_fmt_pct(ov['max_pct_error'])} |")
    lines.append('')

    best = compare['overall']['best_by_metric']
    lines.append('- **best by mean**: ' + METHOD_DISPLAY[best['mean_pct_error']['method']])
    lines.append('- **best by median**: ' + METHOD_DISPLAY[best['median_pct_error']['method']])
    lines.append('- **best by max**: ' + METHOD_DISPLAY[best['max_pct_error']['method']])
    lines.append('')

    lines.append('## 4. 全 30 参数逐项对照（estimate / error%）')
    lines.append('')
    lines.append('| 参数 | 真值 | KF36 est | KF36 err% | Markov est | Markov err% | Pure SCD est | Pure SCD err% | Round61 est | Round61 err% |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|')

    for name in compare['param_order']:
        item = compare['all_params'][name]
        row = [
            name,
            _fmt_e(item['true']),
            _fmt_e(item['methods']['kf36_noisy']['est']),
            _fmt_pct(item['methods']['kf36_noisy']['pct_error']),
            _fmt_e(item['methods']['markov42_noisy']['est']),
            _fmt_pct(item['methods']['markov42_noisy']['pct_error']),
            _fmt_e(item['methods']['scd42_neutral']['est']),
            _fmt_pct(item['methods']['scd42_neutral']['pct_error']),
            _fmt_e(item['methods']['round61']['est']),
            _fmt_pct(item['methods']['round61']['pct_error']),
        ]
        lines.append('| ' + ' | '.join(row) + ' |')

    lines.append('')
    lines.append('## 5. 简短结论')
    lines.append('')
    mean_rank = compare['overall']['rankings']['mean_pct_error']
    lines.append(f"- 全局 mean 排名：1) {METHOD_DISPLAY[mean_rank[0]['method']]}  2) {METHOD_DISPLAY[mean_rank[1]['method']]}  3) {METHOD_DISPLAY[mean_rank[2]['method']]}  4) {METHOD_DISPLAY[mean_rank[3]['method']]}")
    lines.append(f"- Round61 的位置：mean 第 **{compare['overall']['round61_rank']['mean_pct_error']}**，median 第 **{compare['overall']['round61_rank']['median_pct_error']}**，max 第 **{compare['overall']['round61_rank']['max_pct_error']}**。")

    interp = compare['overall']['interpretation']
    lines.append(f"- 解释：{interp}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    args = parse_args()
    noise_scale = args.noise_scale
    suffix, out = build_output_paths(noise_scale, args.report_date)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    expected_cfg = expected_noise_config(noise_scale)

    source_mod = load_module(f'markov_pruned_source_four_{suffix}', str(SOURCE_FILE))
    compute_r61_mod = load_module(f'compute_r61_four_{suffix}', str(COMPUTE_R61_FILE))
    r53_mod = load_module(f'markov_r53_four_{suffix}', str(R53_METHOD_FILE))
    r61_mod = load_module(f'markov_r61_four_{suffix}', str(R61_METHOD_FILE))

    dataset = build_shared_dataset(source_mod, noise_scale)
    params = compute_r61_mod._param_specs(source_mod)

    payloads = {}
    execution = {}

    # 1) KF36
    if (not args.force_rerun) and out['kf36_noisy'].exists():
        p = _load_json(out['kf36_noisy'])
        if _noise_matches(p, expected_cfg):
            payloads['kf36_noisy'] = p
            execution['kf36_noisy'] = 'reused_verified'
        else:
            execution['kf36_noisy'] = 'rerun_noise_mismatch'
    if 'kf36_noisy' not in payloads:
        kf36_res = source_mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=36,
            label=f'KF36-SHARED-{suffix.upper()}',
        )
        payload_kf = compute_payload(
            source_mod,
            kf36_res[0],
            params,
            variant=f'kf36_shared_{suffix}',
            method_file='source_mod.run_calibration(n_states=36)',
            extra={
                'noise_scale': noise_scale,
                'noise_config': dataset['noise_config'],
                'comparison_mode': 'shared_dataset_apples_to_apples',
                'mainline_rung': 'kf36_noisy',
            },
        )
        out['kf36_noisy'].write_text(json.dumps(payload_kf, ensure_ascii=False, indent=2), encoding='utf-8')
        payloads['kf36_noisy'] = payload_kf
        execution['kf36_noisy'] = execution.get('kf36_noisy', 'rerun')

    # 2) Markov42 baseline
    if (not args.force_rerun) and out['markov42_noisy'].exists():
        p = _load_json(out['markov42_noisy'])
        if _noise_matches(p, expected_cfg):
            payloads['markov42_noisy'] = p
            execution['markov42_noisy'] = 'reused_verified'
        else:
            execution['markov42_noisy'] = 'rerun_noise_mismatch'
    if 'markov42_noisy' not in payloads:
        markov_res = source_mod.run_calibration(
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            n_states=42,
            bi_g=dataset['bi_g'],
            tau_g=dataset['tau_g'],
            bi_a=dataset['bi_a'],
            tau_a=dataset['tau_a'],
            label=f'MARKOV42-SHARED-{suffix.upper()}',
        )
        payload_markov = compute_payload(
            source_mod,
            markov_res[0],
            params,
            variant=f'42state_gm1_shared_{suffix}',
            method_file='source_mod.run_calibration(n_states=42)',
            extra={
                'noise_scale': noise_scale,
                'noise_config': dataset['noise_config'],
                'comparison_mode': 'shared_dataset_apples_to_apples',
                'mainline_rung': 'markov42_noisy',
            },
        )
        out['markov42_noisy'].write_text(json.dumps(payload_markov, ensure_ascii=False, indent=2), encoding='utf-8')
        payloads['markov42_noisy'] = payload_markov
        execution['markov42_noisy'] = execution.get('markov42_noisy', 'rerun')

    # 3) Pure SCD baseline (neutral Markov+SCD)
    if (not args.force_rerun) and out['scd42_neutral'].exists():
        p = _load_json(out['scd42_neutral'])
        if _noise_matches(p, expected_cfg):
            payloads['scd42_neutral'] = p
            execution['scd42_neutral'] = 'reused_verified'
        else:
            execution['scd42_neutral'] = 'rerun_noise_mismatch'
    if 'scd42_neutral' not in payloads:
        scd_base = _build_neutral_scd_candidate()
        method_mod_s = load_module(f'markov_method_scd_neutral_{suffix}', str(R53_METHOD_FILE))
        method_mod_s = _build_patched_method(method_mod_s, scd_base)
        scd_result = list(_run_internalized_hybrid_scd(
            method_mod_s,
            source_mod,
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            bi_g=dataset['bi_g'],
            bi_a=dataset['bi_a'],
            tau_g=dataset['tau_g'],
            tau_a=dataset['tau_a'],
            label=f'SCD42-NEUTRAL-SHARED-{suffix.upper()}',
            scd_cfg=scd_base['scd'],
        ))
        scd_extra_runtime = scd_result[4] if len(scd_result) >= 5 and isinstance(scd_result[4], dict) else {}
        payload_scd = compute_payload(
            source_mod,
            scd_result[0],
            params,
            variant=f'42state_gm1_scdneutral_shared_{suffix}',
            method_file='neutral_markov42_plus_once_scd_on_shared_dataset',
            extra={
                'noise_scale': noise_scale,
                'noise_config': dataset['noise_config'],
                'comparison_mode': 'shared_dataset_apples_to_apples',
                'mainline_rung': 'scd42_neutral',
                'selected_candidate': scd_base['name'],
                'candidate_description': scd_base['description'],
                'scd_cfg': copy.deepcopy(scd_base['scd']),
                'iter_patches': copy.deepcopy(scd_base['iter_patches']),
                'runtime_log': {
                    'schedule_log': scd_extra_runtime.get('schedule_log'),
                    'feedback_log': scd_extra_runtime.get('feedback_log'),
                    'scd_log': scd_extra_runtime.get('scd_log'),
                },
            },
        )
        out['scd42_neutral'].write_text(json.dumps(payload_scd, ensure_ascii=False, indent=2), encoding='utf-8')
        payloads['scd42_neutral'] = payload_scd
        execution['scd42_neutral'] = execution.get('scd42_neutral', 'rerun')

    # 4) Round61
    if (not args.force_rerun) and out['round61'].exists():
        p = _load_json(out['round61'])
        if _noise_matches(p, expected_cfg):
            payloads['round61'] = p
            execution['round61'] = 'reused_verified'
        else:
            execution['round61'] = 'rerun_noise_mismatch'
    if 'round61' not in payloads:
        candidate = _pick_round61_candidate()
        merged_candidate = _merge_round61_candidate(candidate)
        patched_method = _build_patched_method(r53_mod, merged_candidate)
        r61_result = list(r61_mod._run_internalized_hybrid_scd(
            patched_method,
            source_mod,
            dataset['imu_noisy'],
            dataset['pos0'],
            dataset['ts'],
            bi_g=dataset['bi_g'],
            bi_a=dataset['bi_a'],
            tau_g=dataset['tau_g'],
            tau_a=dataset['tau_a'],
            label=f'R61-SHARED-{suffix.upper()}',
            scd_cfg=merged_candidate['scd'],
        ))
        r61_extra = r61_result[4] if len(r61_result) >= 5 and isinstance(r61_result[4], dict) else {}
        r61_extra = dict(r61_extra)
        r61_extra.update({
            'noise_scale': noise_scale,
            'noise_config': dataset['noise_config'],
            'comparison_mode': 'shared_dataset_apples_to_apples',
            'round61_selected_candidate': candidate['name'],
        })
        payload_r61 = compute_payload(
            source_mod,
            r61_result[0],
            params,
            variant=f'42state_gm1_round61_h_scd_state20_microtight_commit_shared_{suffix}',
            method_file=str(R61_METHOD_FILE),
            extra=r61_extra,
        )
        out['round61'].write_text(json.dumps(payload_r61, ensure_ascii=False, indent=2), encoding='utf-8')
        payloads['round61'] = payload_r61
        execution['round61'] = execution.get('round61', 'rerun')

    compare = {
        'mode': 'shared_dataset_apples_to_apples',
        'noise_scale': noise_scale,
        'noise_scale_tag': suffix,
        'noise_config': dataset['noise_config'],
        'source_file': str(SOURCE_FILE),
        'method_order': METHOD_ORDER,
        'methods': {
            'kf36_noisy': {
                'name': METHOD_DISPLAY['kf36_noisy'],
                'definition': 'source_mod.run_calibration(n_states=36) on the same shared noisy dataset',
                'json_path': str(out['kf36_noisy']),
            },
            'markov42_noisy': {
                'name': METHOD_DISPLAY['markov42_noisy'],
                'definition': 'source_mod.run_calibration(n_states=42, bi/tau set from shared noise) on the same shared dataset',
                'json_path': str(out['markov42_noisy']),
            },
            'scd42_neutral': {
                'name': METHOD_DISPLAY['scd42_neutral'],
                'definition': 'Round53 backbone + neutral trust/cov schedule + iter2 once-per-phase SCD on the same shared dataset',
                'json_path': str(out['scd42_neutral']),
            },
            'round61': {
                'name': METHOD_DISPLAY['round61'],
                'definition': 'method_42state_gm1_round61_h_scd_state20_microtight_commit on the same shared dataset',
                'json_path': str(out['round61']),
            },
        },
        'param_order': payloads['round61']['param_order'],
        'all_params': {},
        'overall': {
            'by_method': {},
            'best_by_metric': {},
            'rankings': {},
            'round61_rank': {},
            'best_param_count': {},
            'interpretation': '',
        },
        'execution': execution,
    }

    for method in METHOD_ORDER:
        compare['overall']['by_method'][method] = payloads[method]['overall']

    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        ranking = sorted(
            [{'method': m, 'value': float(payloads[m]['overall'][metric])} for m in METHOD_ORDER],
            key=lambda x: x['value']
        )
        compare['overall']['rankings'][metric] = ranking
        compare['overall']['best_by_metric'][metric] = ranking[0]
        compare['overall']['round61_rank'][metric] = [r['method'] for r in ranking].index('round61') + 1

    best_param_count = {m: 0 for m in METHOD_ORDER}
    for name in compare['param_order']:
        true_v = float(payloads['round61']['param_errors'][name]['true'])
        methods_block = {}
        ranking = []
        for method in METHOD_ORDER:
            est = float(payloads[method]['param_errors'][name]['est'])
            pct = float(payloads[method]['param_errors'][name]['pct_error'])
            methods_block[method] = {
                'est': est,
                'pct_error': pct,
            }
            ranking.append({'method': method, 'pct_error': pct})
        ranking.sort(key=lambda x: x['pct_error'])
        best_param_count[ranking[0]['method']] += 1
        compare['all_params'][name] = {
            'true': true_v,
            'methods': methods_block,
            'ranking_by_pct_error': ranking,
        }

    compare['overall']['best_param_count'] = best_param_count

    best_mean_method = compare['overall']['best_by_metric']['mean_pct_error']['method']
    best_median_method = compare['overall']['best_by_metric']['median_pct_error']['method']
    best_max_method = compare['overall']['best_by_metric']['max_pct_error']['method']
    compare['overall']['interpretation'] = (
        f"Global metrics show best mean={METHOD_DISPLAY[best_mean_method]}, "
        f"best median={METHOD_DISPLAY[best_median_method]}, best max={METHOD_DISPLAY[best_max_method]}. "
        f"Round61 ranks mean#{compare['overall']['round61_rank']['mean_pct_error']}, "
        f"median#{compare['overall']['round61_rank']['median_pct_error']}, "
        f"max#{compare['overall']['round61_rank']['max_pct_error']} among the four methods."
    )

    out['compare'].write_text(json.dumps(compare, ensure_ascii=False, indent=2), encoding='utf-8')

    compact = {
        'mode': compare['mode'],
        'noise_scale': compare['noise_scale'],
        'noise_scale_tag': compare['noise_scale_tag'],
        'noise_config': compare['noise_config'],
        'files': {
            'kf36_json': str(out['kf36_noisy']),
            'markov_json': str(out['markov42_noisy']),
            'scd_json': str(out['scd42_neutral']),
            'round61_json': str(out['round61']),
            'compare_json': str(out['compare']),
            'report_md': str(out['report']),
        },
        'overall_by_method': compare['overall']['by_method'],
        'best_by_metric': compare['overall']['best_by_metric'],
        'round61_rank': compare['overall']['round61_rank'],
        'best_param_count': compare['overall']['best_param_count'],
        'execution': compare['execution'],
    }
    out['compact'].write_text(json.dumps(compact, ensure_ascii=False, indent=2), encoding='utf-8')

    report_text = render_report(compare)
    out['report'].write_text(report_text, encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps({
        'noise_scale': noise_scale,
        'noise_config': compare['noise_config'],
        'execution': execution,
        'compare_json': str(out['compare']),
        'compact_json': str(out['compact']),
        'report_md': str(out['report']),
        'overall_by_method': compare['overall']['by_method'],
        'best_by_metric': compare['overall']['best_by_metric'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
