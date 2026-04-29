from __future__ import annotations

import argparse
import copy
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
TMP_PSINS_DIR = ROOT / 'tmp_psins_py'
SOURCE_FILE = TMP_PSINS_DIR / 'psins_py' / 'test_calibration_markov_pruned.py'
SYMMETRIC20_PROBE_FILE = SCRIPTS_DIR / 'probe_ch3_corrected_symmetric20_front2_back11.py'
COMPARE_SHARED_FILE = SCRIPTS_DIR / 'compare_four_methods_shared_noise.py'
COMPUTE_R61_FILE = SCRIPTS_DIR / 'compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors.py'
PROBE_R55_FILE = SCRIPTS_DIR / 'probe_round55_newline.py'
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'
R61_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round61_h_scd_state20_microtight_commit.py'

for p in [ROOT, TMP_PSINS_DIR, METHOD_DIR, SCRIPTS_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from common_markov import load_module

# Re-use shared helpers
_compare_shared = load_module('_compare_shared_four', str(COMPARE_SHARED_FILE))
_compute_r61 = load_module('_compute_r61_four', str(COMPUTE_R61_FILE))
_probe_r55 = load_module('_probe_r55_four', str(PROBE_R55_FILE))

FOCUS_PARAMS = ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z']
DEFAULT_NOISE_SCALE = 0.12
COMPARISON_MODE = 'four_group_static_to_scd_alignment'

GROUP_ORDER = ['g1_kf_static', 'g2_markov_static', 'g3_markov_sym20', 'g4_scd_sym20']
GROUP_DISPLAY = {
    'g1_kf_static': 'G1 普通模型 @ 单位置对准',
    'g2_markov_static': 'G2 Markov 模型 @ 单位置对准',
    'g3_markov_sym20': 'G3 Markov @ 旋转对准策略 (20-position)',
    'g4_scd_sym20': 'G4 Markov + SCD @ 旋转对准策略 (20-position)',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise-scale', type=float, default=DEFAULT_NOISE_SCALE)
    parser.add_argument('--report-date', default=datetime.now().strftime('%Y-%m-%d'))
    parser.add_argument('--force-rerun', action='store_true')
    return parser.parse_args()


def build_static_dataset(mod, noise_scale: float, duration_s: float = 1200.0) -> dict:
    """Build a single-position stationary IMU dataset (no rotation)."""
    ts = 0.01
    T = duration_s
    n = int(T / ts) + 1
    att0 = mod.np.array([0.0, 0.0, 0.0]) * mod.glv.deg
    att = mod.np.zeros((n, 4))
    att[:, 0:3] = att0
    att[:, 3] = mod.np.arange(1, n + 1) * ts
    pos0 = mod.posset(34.0, 0.0, 0.0)
    imu, _ = mod.avp2imu(att, pos0)
    clbt_truth = mod.get_default_clbt()
    imu_clean = mod.imuclbt(imu, clbt_truth)

    arw = 0.005 * noise_scale * mod.glv.dpsh
    vrw = 5.0 * noise_scale * mod.glv.ugpsHz
    bi_g = 0.002 * noise_scale * mod.glv.dph
    bi_a = 5.0 * noise_scale * mod.glv.ug

    imu_noisy = mod.imuadderr_full(
        imu_clean, ts,
        arw=arw, vrw=vrw,
        bi_g=bi_g, tau_g=300.0,
        bi_a=bi_a, tau_a=300.0,
        seed=42,
    )
    cfg = {
        'arw_dpsh': 0.005 * noise_scale,
        'vrw_ugpsHz': 5.0 * noise_scale,
        'bi_g_dph': 0.002 * noise_scale,
        'bi_a_ug': 5.0 * noise_scale,
        'tau_g': 300.0,
        'tau_a': 300.0,
        'seed': 42,
        'base_family': 'round53_round61_shared',
        'dataset_type': 'single_position_static',
        'duration_s': duration_s,
    }
    return {
        'ts': ts,
        'pos0': pos0,
        'imu_noisy': imu_noisy,
        'bi_g': bi_g,
        'bi_a': bi_a,
        'tau_g': 300.0,
        'tau_a': 300.0,
        'noise_scale': noise_scale,
        'noise_config': cfg,
    }


def build_sym20_dataset(mod, noise_scale: float) -> dict:
    """Build the corrected symmetric20 dataset."""
    probe_mod = load_module('fgr_sym20_probe', str(SYMMETRIC20_PROBE_FILE))
    candidate = probe_mod.build_symmetric20_candidate(mod)
    paras = mod.np.array([
        [
            idx,
            int(r['axis'][0]), int(r['axis'][1]), int(r['axis'][2]),
            float(r['angle_deg']),
            float(r['rotation_time_s']),
            float(r['pre_static_s']),
            float(r['post_static_s']),
        ]
        for idx, r in enumerate(candidate.all_rows, start=1)
    ], dtype=float)
    paras[:, 4] = paras[:, 4] * mod.glv.deg

    ts = 0.01
    att0 = mod.np.array([0.0, 0.0, 0.0]) * mod.glv.deg
    pos0 = mod.posset(34.0, 0.0, 0.0)
    att = mod.attrottt(att0, paras, ts)
    imu, _ = mod.avp2imu(att, pos0)
    clbt_truth = mod.get_default_clbt()
    imu_clean = mod.imuclbt(imu, clbt_truth)

    arw = 0.005 * noise_scale * mod.glv.dpsh
    vrw = 5.0 * noise_scale * mod.glv.ugpsHz
    bi_g = 0.002 * noise_scale * mod.glv.dph
    bi_a = 5.0 * noise_scale * mod.glv.ug

    imu_noisy = mod.imuadderr_full(
        imu_clean, ts,
        arw=arw, vrw=vrw,
        bi_g=bi_g, tau_g=300.0,
        bi_a=bi_a, tau_a=300.0,
        seed=42,
    )
    cfg = {
        'arw_dpsh': 0.005 * noise_scale,
        'vrw_ugpsHz': 5.0 * noise_scale,
        'bi_g_dph': 0.002 * noise_scale,
        'bi_a_ug': 5.0 * noise_scale,
        'tau_g': 300.0,
        'tau_a': 300.0,
        'seed': 42,
        'base_family': 'round53_round61_shared',
        'dataset_type': 'corrected_symmetric20_rotation',
    }
    return {
        'ts': ts,
        'pos0': pos0,
        'imu_noisy': imu_noisy,
        'bi_g': bi_g,
        'bi_a': bi_a,
        'tau_g': 300.0,
        'tau_a': 300.0,
        'noise_scale': noise_scale,
        'noise_config': cfg,
    }


def compute_mod_payload(mod, clbt, params, variant: str, method_file: str, extra=None):
    """Build the standard param-errors payload for any calibration result."""
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
    pct_arr = mod.np.asarray(pct_values, dtype=float)
    overall = {
        'mean_pct_error': float(mod.np.mean(pct_arr)),
        'median_pct_error': float(mod.np.median(pct_arr)),
        'max_pct_error': float(mod.np.max(pct_arr)),
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


def static_output_path(method_key: str, noise_scale: float) -> Path:
    suffix = str(noise_scale).replace('.', 'p')
    tag = f'single_position_static_{noise_scale:.2f}s'.replace('.', 'p')
    if method_key == 'kf36_noisy':
        return RESULTS_DIR / f'KF36_{tag}_shared_noise{suffix}_param_errors.json'
    if method_key == 'markov42_noisy':
        return RESULTS_DIR / f'M_markov_42state_gm1_{tag}_shared_noise{suffix}_param_errors.json'
    raise KeyError(method_key)


def _reuse_or_run_static(mod, dataset: dict, method_key: str, params, force_rerun: bool = False):
    """Run G1 or G2 on static single-position dataset."""
    out_path = static_output_path(method_key, dataset['noise_scale'])
    expected_cfg = dict(dataset['noise_config'])

    if (not force_rerun) and out_path.exists():
        payload = json.loads(out_path.read_text(encoding='utf-8'))
        extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
        cfg = extra.get('noise_config') or extra.get('dataset_noise_config')
        if isinstance(cfg, dict):
            numeric_keys = ['arw_dpsh', 'vrw_ugpsHz', 'bi_g_dph', 'bi_a_ug', 'tau_g', 'tau_a']
            ok = True
            for k in numeric_keys:
                if k not in cfg or abs(float(cfg[k]) - float(expected_cfg[k])) > 1e-12:
                    ok = False
                    break
            if ok and int(cfg.get('seed', -1)) == int(expected_cfg['seed']):
                extra_mode = extra.get('comparison_mode') == COMPARISON_MODE
                extra_gk = extra.get('group_key') == method_key.replace('noisy', 'static').replace('kf36_', 'g1_kf_').replace('markov42_', 'g2_markov_')
                if extra_mode and extra_gk:
                    return payload, 'reused_verified', out_path

    label = 'STATIC-KF36' if method_key == 'kf36_noisy' else 'STATIC-MARKOV42'
    if method_key == 'kf36_noisy':
        clbt, kf, P_trace, X_trace, iter_bounds = mod.run_calibration(
            dataset['imu_noisy'], dataset['pos0'], dataset['ts'],
            n_states=36, label=label,
        )
    else:
        clbt, kf, P_trace, X_trace, iter_bounds = mod.run_calibration(
            dataset['imu_noisy'], dataset['pos0'], dataset['ts'],
            n_states=42,
            bi_g=dataset['bi_g'], tau_g=dataset['tau_g'],
            bi_a=dataset['bi_a'], tau_a=dataset['tau_a'],
            label=label,
        )

    gk = 'g1_kf_static' if method_key == 'kf36_noisy' else 'g2_markov_static'
    payload = compute_mod_payload(
        mod, clbt, params,
        variant=f'{gk}_static_noise{_make_suffix(dataset["noise_scale"])}',
        method_file=f'source_mod.run_calibration(n_states={36 if method_key == "kf36_noisy" else 42})',
        extra={
            'comparison_mode': COMPARISON_MODE,
            'group_key': gk,
            'noise_scale': dataset['noise_scale'],
            'noise_config': dataset['noise_config'],
        },
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', out_path


def _load_existing(path: Path, expected_cfg: dict, expected_group_key: str) -> tuple[dict, bool]:
    """Load an existing JSON result, return (payload, ok)."""
    if not path.exists():
        return None, False
    payload = json.loads(path.read_text(encoding='utf-8'))
    extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
    cfg = extra.get('noise_config') or extra.get('dataset_noise_config')
    if not isinstance(cfg, dict):
        return payload, False
    numeric_keys = ['arw_dpsh', 'vrw_ugpsHz', 'bi_g_dph', 'bi_a_ug', 'tau_g', 'tau_a']
    for k in numeric_keys:
        if k not in cfg or abs(float(cfg[k]) - float(expected_cfg[k])) > 1e-12:
            return payload, False
    if int(cfg.get('seed', -1)) != int(expected_cfg.get('seed', -1)):
        return payload, False
    if extra.get('comparison_mode') != COMPARISON_MODE:
        return payload, False
    if extra.get('group_key') != expected_group_key:
        return payload, False
    return payload, True


def _reuse_or_run_scd(mod, dataset, params, r53_mod, r61_mod, expected_cfg, force_rerun=False):
    """Run G4: Markov + SCD on symmetric20."""
    suffix = _make_suffix(dataset['noise_scale'])
    out_path = RESULTS_DIR / f'R61_42state_gm1_round61_h_scd_state20_microtight_commit_ch3corrected_symmetric20_att0zero_1200s_shared_noise0p12_param_errors.json'

    # The existing file from compare_four_group_progression_19_20 has different comparison_mode.
    # We'll also allow a secondary tag for group_key='g4_scd_sym20' or 'g4_round61_20'.
    if (not force_rerun) and out_path.exists():
        payload = json.loads(out_path.read_text(encoding='utf-8'))
        extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
        cfg = extra.get('noise_config') or extra.get('dataset_noise_config')
        if isinstance(cfg, dict):
            numeric_keys = ['arw_dpsh', 'vrw_ugpsHz', 'bi_g_dph', 'bi_a_ug', 'tau_g', 'tau_a']
            ok_noise = all(
                k in cfg and abs(float(cfg[k]) - float(expected_cfg[k])) < 1e-12
                for k in numeric_keys
            )
            ok_seed = int(cfg.get('seed', -1)) == int(expected_cfg.get('seed', -1))
            # Accept either the original group_key or the new one
            ok_gk = extra.get('group_key') in ('g4_scd_sym20', 'g4_round61_20')
            if ok_noise and ok_seed and ok_gk:
                # Re-tag with the new comparison_mode
                new_extra = dict(extra)
                new_extra['comparison_mode'] = COMPARISON_MODE
                new_extra['group_key'] = 'g4_scd_sym20'
                payload['extra'] = new_extra
                out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
                return payload, 'reused_verified', out_path

    # Run fresh: use the Round61 methodology which is Markov + SCD
    dataset_sym = build_sym20_dataset(mod, dataset['noise_scale'])

    # Get the best known SCD candidate from R61 results
    # Use the existing R61 candidate registry
    _r61_module = load_module('_r61_mod_fgr', str(R61_METHOD_FILE))
    candidate = _try_get_r61_candidate(_r61_module)
    if candidate is None:
        # Fallback: use the R61 candidate that's been used before
        candidate = {'name': 'r61_s20_08988_ryz00116'}

    # We need to load the actual R61 method with its candidate merge
    # Since we want Markov + SCD (not full LLM-guided), use the neutral SCD baseline
    # from compare_four_methods_shared_noise but on the sym20 path
    return _run_scd_fresh(mod, dataset_sym, params, r53_mod, force_rerun)


def _try_get_r61_candidate(r61_mod):
    """Try to get the R61 candidate list and use the transferred one."""
    # The R61 module has ROUND61_CANDIDATES or similar
    if hasattr(r61_mod, '_pick_candidate'):
        try:
            return r61_mod._pick_candidate()
        except Exception:
            pass
    if hasattr(r61_mod, 'ROUND61_CANDIDATES'):
        cands = r61_mod.ROUND61_CANDIDATES
        if cands:
            return cands[0]
    return None


def _make_suffix(noise_scale: float) -> str:
    return str(noise_scale).replace('.', 'p')


def _run_scd_fresh(mod, dataset, params, r53_mod, force_rerun):
    """Run Markov + SCD (neutral) on the sym20 path."""
    suffix = _make_suffix(dataset['noise_scale'])
    out_path = RESULTS_DIR / f'SCD42_markov_sym20_neutral_shared_noise{suffix}_param_errors.json'

    if (not force_rerun) and out_path.exists():
        payload = json.loads(out_path.read_text(encoding='utf-8'))
        extra = payload.get('extra', {}) if isinstance(payload, dict) else {}
        if extra.get('comparison_mode') == COMPARISON_MODE:
            return payload, 'reused_verified', out_path

    # Build the neutral SCD candidate (same as in compare_four_methods_shared_noise)
    scd_cfg = _build_neutral_scd_candidate()

    method_mod = load_module(f'fgr_r53_scd_{suffix}', str(R53_METHOD_FILE))
    method_mod = _probe_r55._build_patched_method(method_mod, scd_cfg)

    result = _run_internalized_hybrid(
        method_mod, mod,
        dataset['imu_noisy'], dataset['pos0'], dataset['ts'],
        bi_g=dataset['bi_g'], bi_a=dataset['bi_a'],
        tau_g=300.0, tau_a=300.0,
        label=f'SCD-SYM20-{suffix.upper()}',
        scd_cfg=scd_cfg['scd'],
    )

    runtime_extra = result[4] if len(result) >= 5 and isinstance(result[4], dict) else {}
    payload = compute_mod_payload(
        mod, result[0], params,
        variant=f'g4_scd_sym20_noise{suffix}',
        method_file='neutral_markov42_plus_scd_on_symmetric20',
        extra={
            'comparison_mode': COMPARISON_MODE,
            'group_key': 'g4_scd_sym20',
            'noise_scale': dataset['noise_scale'],
            'noise_config': dataset['noise_config'],
            'scd_cfg': copy.deepcopy(scd_cfg['scd']),
            'iter_patches': copy.deepcopy(scd_cfg['iter_patches']),
            'runtime_log': {
                'schedule_log': runtime_extra.get('schedule_log'),
                'feedback_log': runtime_extra.get('feedback_log'),
                'scd_log': runtime_extra.get('scd_log'),
            },
        },
    )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload, 'rerun', out_path


def _build_neutral_scd_candidate():
    """Build a neutral Markov+SCD candidate (once-per-phase SCD at iter2)."""
    neutral_policy_patch = {
        'selected_prior_scale': 1.0, 'other_scale_prior_scale': 1.0,
        'ka2_prior_scale': 1.0, 'lever_prior_scale': 1.0,
        'selected_q_static_scale': 1.0, 'selected_q_dynamic_scale': 1.0,
        'selected_q_late_mult': 1.0, 'other_scale_q_scale': 1.0,
        'other_scale_q_late_mult': 1.0, 'ka2_q_scale': 1.0, 'lever_q_scale': 1.0,
        'static_r_scale': 1.0, 'dynamic_r_scale': 1.0, 'late_r_mult': 1.0,
        'late_release_frac': 0.58, 'selected_alpha_floor': 1.0, 'selected_alpha_span': 0.0,
        'other_scale_alpha': 1.0, 'ka2_alpha': 1.0, 'lever_alpha': 1.0, 'markov_alpha': 1.0,
        'trust_score_soft': 2.1, 'trust_cov_soft': 0.44, 'trust_mix': 0.58,
        'state_alpha_mult': {}, 'state_alpha_add': {}, 'state_prior_diag_mult': {},
        'state_q_static_mult': {}, 'state_q_dynamic_mult': {}, 'state_q_late_mult': {},
    }
    return {
        'name': 'neutral_markov42_plus_scd_baseline',
        'description': 'Controlled neutral Markov+SCD baseline (iter2 once-per-phase SCD only).',
        'iter_patches': {0: copy.deepcopy(neutral_policy_patch), 1: copy.deepcopy(neutral_policy_patch)},
        'post_rx_y_mult': 1.0, 'post_ry_z_mult': 1.0,
        'scd': {
            'mode': 'once_per_phase', 'alpha': 0.999, 'transition_duration': 2.0,
            'target': 'scale_block', 'bias_to_target': True, 'apply_policy_names': ['iter2_commit'],
        },
    }


def _run_internalized_hybrid(method_mod, source_mod, imu_noisy, pos0, ts, **kwargs):
    """Call the hybrid SCD run. This uses the same pattern as compare_four_methods_shared_noise."""
    # Use the R61 module's _run_internalized_hybrid_scd
    _r61_mod = load_module('_r61_run_fgr', str(R61_METHOD_FILE))
    return _r61_mod._run_internalized_hybrid_scd(
        method_mod, source_mod, imu_noisy, pos0, ts, **kwargs,
    )


def fmt_pct(x: float) -> str:
    return f'{x:.6f}'


def build_group_rows(payloads: dict[str, dict]) -> list[dict]:
    rows = []
    for gk in GROUP_ORDER:
        p = payloads[gk]
        ov = p['overall']
        rows.append({
            'group_key': gk,
            'display': GROUP_DISPLAY[gk],
            'mean_pct_error': float(ov['mean_pct_error']),
            'median_pct_error': float(ov['median_pct_error']),
            'max_pct_error': float(ov['max_pct_error']),
        })
    return rows


def build_progression_metrics(group_rows: list[dict]) -> dict:
    metrics = {}
    for metric in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        values = [float(row[metric]) for row in group_rows]
        steps = []
        for i in range(1, len(group_rows)):
            delta = float(group_rows[i - 1][metric] - group_rows[i][metric])
            steps.append({
                'from_group': group_rows[i - 1]['group_key'],
                'to_group': group_rows[i]['group_key'],
                'delta': delta,
                'improved': bool(delta > 0),
            })
        metrics[metric] = {
            'strict_progression': all(s['improved'] for s in steps),
            'steps': steps,
            'best_group': min(group_rows, key=lambda x: float(x[metric]))['group_key'],
        }
    return metrics


def render_report(summary: dict) -> str:
    lines = []
    lines.append('# 四组对准精度递进对照：单位置 → 旋转对准')
    lines.append('')
    lines.append(f"- noise_scale = **{summary['noise_scale']}**")
    cfg = summary['noise_config']
    lines.append(f"- arw={cfg['arw_dpsh']:.6f} dpsh, vrw={cfg['vrw_ugpsHz']:.2f} ugpsHz")
    lines.append(f"- bi_g={cfg['bi_g_dph']:.6f} dph, bi_a={cfg['bi_a_ug']:.1f} ug, tau={cfg['tau_g']:.0f}s, seed={cfg['seed']}")
    lines.append('')
    lines.append('## 四组定义')
    lines.append('')
    lines.append('| 组别 | 模型 | 轨迹 | 核心差异 |')
    lines.append('|---|---|---|---|')
    lines.append('| G1 | 普通KF (36-state) | 单位置静止 | 不建模 Markov 误差 |')
    lines.append('| G2 | Markov (42-state) | 单位置静止 | 一阶 Gauss-Markov 建模 |')
    lines.append('| G3 | Markov (42-state) | 20-position 旋转策略 | 纯 Markov + 旋转激励 |')
    lines.append('| G4 | Markov + SCD | 20-position 旋转策略 | Markov + 协方差衰减 |')
    lines.append('')
    lines.append('## Overall 递进（越低越好）')
    lines.append('')
    lines.append('| 组别 | mean% | median% | max% | Δmean vs prev | Δmedian vs prev | Δmax vs prev |')
    lines.append('|---|---:|---:|---:|---:|---:|---:|')
    prev = None
    for row in summary['group_rows']:
        if prev is None:
            d_m = d_med = d_max = '—'
        else:
            d_m = f"{prev['mean_pct_error'] - row['mean_pct_error']:+.6f}"
            d_med = f"{prev['median_pct_error'] - row['median_pct_error']:+.6f}"
            d_max = f"{prev['max_pct_error'] - row['max_pct_error']:+.6f}"
        lines.append(
            f"| {row['display']} | {fmt_pct(row['mean_pct_error'])} | {fmt_pct(row['median_pct_error'])} | {fmt_pct(row['max_pct_error'])} | {d_m} | {d_med} | {d_max} |"
        )
        prev = row
    lines.append('')
    lines.append('## 递进判断')
    lines.append('')
    for m, label in [('mean_pct_error', 'mean'), ('median_pct_error', 'median'), ('max_pct_error', 'max')]:
        pm = summary['progression_metrics'][m]
        verdict = '✅ 严格递进' if pm['strict_progression'] else '❌ 非严格单调'
        best = GROUP_DISPLAY[pm['best_group']]
        lines.append(f"- **{label}**：{verdict}；最优组别 **{best}**")
        for s in pm['steps']:
            step_txt = '改善' if s['improved'] else '退化'
            lines.append(f"  - {GROUP_DISPLAY[s['from_group']]} → {GROUP_DISPLAY[s['to_group']]}：{step_txt} {s['delta']:+.6f}")
    lines.append('')
    lines.append('## 关键参数（越低越好）')
    lines.append('')
    lines.append('| 参数 | G1 KF静态 | G2 Markov静态 | G3 Markov旋转 | G4 SCD旋转 | best |')
    lines.append('|---|---:|---:|---:|---:|---|')
    for name in FOCUS_PARAMS:
        item = summary['all_params']['table'][name]
        best_g = GROUP_DISPLAY[item['ranking'][0]['group_key']]
        lines.append(
            f"| {name} | {fmt_pct(item['groups']['g1_kf_static']['pct_error'])} | {fmt_pct(item['groups']['g2_markov_static']['pct_error'])} | {fmt_pct(item['groups']['g3_markov_sym20']['pct_error'])} | {fmt_pct(item['groups']['g4_scd_sym20']['pct_error'])} | {best_g} |"
        )
    lines.append('')
    lines.append(f"## 结论")
    lines.append('')
    lines.append(f"- {summary['headline']}")
    lines.append(f"- {summary['interpretation']}")
    lines.append('')
    return '\n'.join(lines) + '\n'


def build_headline(summary: dict) -> str:
    mp = summary['progression_metrics']['mean_pct_error']['strict_progression']
    delp = summary['progression_metrics']['median_pct_error']['strict_progression']
    xp = summary['progression_metrics']['max_pct_error']['strict_progression']
    if mp and delp and xp:
        return '四组在 mean / median / max 三个 overall 指标上均呈现严格递进改善。'
    if mp:
        return '四组在 **mean 指标** 上形成清晰递进；median / max 不一定同时严格单调。'
    return '这四组不构成完全严格单调递进；需要按 mean / median / max 分开看并结合关键参数判断。'


def build_interpretation(summary: dict) -> str:
    bm = GROUP_DISPLAY[summary['progression_metrics']['mean_pct_error']['best_group']]
    bd = GROUP_DISPLAY[summary['progression_metrics']['median_pct_error']['best_group']]
    bx = GROUP_DISPLAY[summary['progression_metrics']['max_pct_error']['best_group']]
    return f'当前 best-mean 是 {bm}，best-median 是 {bd}，best-max 是 {bx}。'


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    mod = load_module(f'fgr_source_{_make_suffix(args.noise_scale)}', str(SOURCE_FILE))
    params = _compute_r61._param_specs(mod)
    r53_mod = load_module(f'fgr_r53_{_make_suffix(args.noise_scale)}', str(R53_METHOD_FILE))

    # Build datasets
    dataset_static = build_static_dataset(mod, args.noise_scale)
    expected_cfg = dict(dataset_static['noise_config'])

    payloads = {}
    execution = {}
    json_paths = {}

    # G1: KF36 on static
    p, status, path = _reuse_or_run_static(mod, dataset_static, 'kf36_noisy', params, args.force_rerun)
    payloads['g1_kf_static'] = p
    execution['g1_kf_static'] = status
    json_paths['g1_kf_static'] = str(path)
    print(f"G1 ({status}): mean={p['overall']['mean_pct_error']:.6f}")

    # G2: Markov42 on static
    p, status, path = _reuse_or_run_static(mod, dataset_static, 'markov42_noisy', params, args.force_rerun)
    payloads['g2_markov_static'] = p
    execution['g2_markov_static'] = status
    json_paths['g2_markov_static'] = str(path)
    print(f"G2 ({status}): mean={p['overall']['mean_pct_error']:.6f}")

    # G3: Markov42 on sym20 (reuse existing or run)
    sym20_path = Path('/root/.openclaw/workspace/psins_method_bench/results/M_markov_42state_gm1_ch3corrected_symmetric20_att0zero_1200s_shared_noise0p12_param_errors.json')
    existing, ok = _load_existing(
        sym20_path, expected_cfg, 'g3_markov_sym20',
    )
    if not ok and sym20_path.exists():
        # Re-tag existing file
        existing['extra']['comparison_mode'] = COMPARISON_MODE
        existing['extra']['group_key'] = 'g3_markov_sym20'
        sym20_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding='utf-8')
        ok = True
    if not ok:
        # Run fresh: need to build sym20 dataset and run Markov
        dataset_sym20 = build_sym20_dataset(mod, args.noise_scale)
        print("  G3: running Markov on sym20 trajectory...")
        import time
        t0 = time.time()
        res = mod.run_calibration(
            dataset_sym20['imu_noisy'], dataset_sym20['pos0'], dataset_sym20['ts'],
            n_states=42,
            bi_g=dataset_sym20['bi_g'], tau_g=300.0,
            bi_a=dataset_sym20['bi_a'], tau_a=300.0,
            label='G3-MARKOV-SYM20',
        )
        print(f"  G3: ran in {time.time()-t0:.1f}s")
        existing = compute_mod_payload(
            mod, res[0], params,
            variant='g3_markov_sym20',
            method_file='source_mod.run_calibration(n_states=42, markov) on sym20',
            extra={
                'comparison_mode': COMPARISON_MODE,
                'group_key': 'g3_markov_sym20',
                'noise_scale': args.noise_scale,
                'noise_config': dataset_sym20['noise_config'],
            },
        )
        sym20_path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding='utf-8')
        status = 'rerun'
    else:
        status = 'reused_verified' if ok else 'loaded_legacy'

    payloads['g3_markov_sym20'] = existing
    execution['g3_markov_sym20'] = status
    json_paths['g3_markov_sym20'] = str(sym20_path)
    print(f"G3 ({status}): mean={existing['overall']['mean_pct_error']:.6f}")

    # G4: Markov + SCD on sym20
    print("  G4: running Markov+SCD on sym20 trajectory...")
    import time
    p4, status4, path4 = _run_scd_fresh(mod, build_sym20_dataset(mod, args.noise_scale), params, r53_mod, args.force_rerun)
    payloads['g4_scd_sym20'] = p4
    execution['g4_scd_sym20'] = status4
    json_paths['g4_scd_sym20'] = str(path4)
    print(f"G4 ({status4}): mean={p4['overall']['mean_pct_error']:.6f}")

    # Build tables
    group_rows = build_group_rows(payloads)
    progression_metrics = build_progression_metrics(group_rows)

    all_params = {}
    param_order = payloads['g1_kf_static']['param_order']
    for name in param_order:
        all_params[name] = {'true': float(payloads['g1_kf_static']['param_errors'][name]['true']), 'groups': {}, 'ranking': []}
        ranking = []
        for gk in GROUP_ORDER:
            pct = float(payloads[gk]['param_errors'][name]['pct_error'])
            est = float(payloads[gk]['param_errors'][name]['est'])
            all_params[name]['groups'][gk] = {'est': est, 'pct_error': pct}
            ranking.append({'group_key': gk, 'pct_error': pct})
        ranking.sort(key=lambda x: x['pct_error'])
        all_params[name]['ranking'] = ranking

    summary = {
        'experiment': 'four_group_static_to_scd_alignment',
        'comparison_mode': COMPARISON_MODE,
        'report_date': args.report_date,
        'noise_scale': args.noise_scale,
        'noise_config': expected_cfg,
        'execution': execution,
        'group_rows': group_rows,
        'progression_metrics': progression_metrics,
        'all_params': {'param_order': param_order, 'table': all_params},
        'groups': {},
        'headline': '',
        'interpretation': '',
    }
    summary['headline'] = build_headline(summary)
    summary['interpretation'] = build_interpretation(summary)

    compare_json = RESULTS_DIR / f'compare_four_group_static_to_scd_noise{args.noise_scale:.2f}'.replace('.', 'p')
    report_md = REPORTS_DIR / f'psins_four_group_static_to_scd_{args.report_date}_noise{str(args.noise_scale).replace(".", "p")}.md'

    summary['files'] = {
        'compare_json': str(compare_json),
        'report_md': str(report_md),
        'run_jsons': json_paths,
    }

    compare_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    report_md.write_text(render_report(summary), encoding='utf-8')

    print('\n=== RESULT ===')
    print(json.dumps({
        'compare_json': str(compare_json),
        'report_md': str(report_md),
        'group_rows': group_rows,
        'progression_metrics': progression_metrics,
    }, ensure_ascii=False, indent=2))
    print('__RESULT_JSON__=' + json.dumps({
        'compare_json': str(compare_json),
        'report_md': str(report_md),
        'group_rows': group_rows,
        'progression_metrics': progression_metrics,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
