from __future__ import annotations

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
SOURCE_FILE = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_calibration_markov_pruned.py'
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'
R61_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round61_h_scd_state20_microtight_commit.py'
COMPUTE_R61_FILE = SCRIPTS_DIR / 'compute_R61_42state_gm1_round61_h_scd_state20_microtight_commit_param_errors.py'

OUT_BASELINE = RESULTS_DIR / 'M_markov_42state_gm1_shared_noise1over3_param_errors.json'
OUT_R61 = RESULTS_DIR / 'R61_42state_gm1_round61_h_scd_state20_microtight_commit_shared_noise1over3_param_errors.json'
OUT_COMPARE = RESULTS_DIR / 'compare_baseline_vs_round61_shared_noise1over3.json'

if str(METHOD_DIR) not in sys.path:
    sys.path.insert(0, str(METHOD_DIR))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import load_module

NOISE_SCALE = 1.0 / 3.0
BASE_ARW = 0.005
BASE_VRW = 5.0
BASE_BI_G = 0.002
BASE_BI_A = 5.0
TAU_G = 300.0
TAU_A = 300.0


def build_shared_dataset(mod, noise_scale: float = NOISE_SCALE):
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
        imu_clean, ts,
        arw=arw, vrw=vrw,
        bi_g=bi_g, tau_g=TAU_G,
        bi_a=bi_a, tau_a=TAU_A, seed=42,
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
        'noise_config': {
            'arw_dpsh': BASE_ARW * noise_scale,
            'vrw_ugpsHz': BASE_VRW * noise_scale,
            'bi_g_dph': BASE_BI_G * noise_scale,
            'bi_a_ug': BASE_BI_A * noise_scale,
            'tau_g': TAU_G,
            'tau_a': TAU_A,
            'seed': 42,
            'base_family': 'round53_round61_shared',
        }
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


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    source_mod = load_module('markov_pruned_source_shared_noise1over3', str(SOURCE_FILE))
    r53_mod = load_module('markov_r53_shared_noise1over3', str(R53_METHOD_FILE))
    r61_mod = load_module('markov_r61_shared_noise1over3', str(R61_METHOD_FILE))
    compute_r61_mod = load_module('compute_r61_shared_noise1over3', str(COMPUTE_R61_FILE))

    dataset = build_shared_dataset(source_mod, NOISE_SCALE)
    ts = dataset['ts']
    pos0 = dataset['pos0']
    imu_noisy = dataset['imu_noisy']
    bi_g = dataset['bi_g']
    bi_a = dataset['bi_a']
    tau_g = dataset['tau_g']
    tau_a = dataset['tau_a']

    params = compute_r61_mod._param_specs(source_mod)

    # Baseline under the same shared 1/3-noise dataset.
    baseline_res = source_mod.run_calibration(
        imu_noisy, pos0, ts, n_states=42,
        bi_g=bi_g, tau_g=tau_g,
        bi_a=bi_a, tau_a=tau_a,
        label='42-GM1-BASELINE-SHARED-NOISE1/3'
    )
    baseline_clbt = baseline_res[0]
    baseline_extra = {
        'noise_scale': NOISE_SCALE,
        'noise_config': dataset['noise_config'],
        'comparison_mode': 'shared_dataset_apples_to_apples',
        'label': '42-GM1-BASELINE-SHARED-NOISE1/3',
    }
    baseline_payload = compute_payload(
        source_mod,
        baseline_clbt,
        params,
        variant='42state_gm1_shared_noise1over3',
        method_file='baseline_direct_run_calibration_shared_noise1over3',
        extra=baseline_extra,
    )
    OUT_BASELINE.write_text(json.dumps(baseline_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    # Round61 under the exact same shared 1/3-noise dataset.
    candidate = r61_mod._pick_candidate()
    merged_candidate = r61_mod._merge_round61_candidate(candidate)
    patched_method = r61_mod._build_patched_method(r53_mod, merged_candidate)
    r61_result = list(r61_mod._run_internalized_hybrid_scd(
        patched_method,
        source_mod,
        imu_noisy,
        pos0,
        ts,
        bi_g=bi_g,
        bi_a=bi_a,
        tau_g=tau_g,
        tau_a=tau_a,
        label='42-GM1-R61-SHARED-NOISE1/3',
        scd_cfg=merged_candidate['scd'],
    ))
    r61_clbt = r61_result[0]
    r61_extra = r61_result[4] if len(r61_result) >= 5 and isinstance(r61_result[4], dict) else {}
    r61_extra = dict(r61_extra)
    r61_extra.update({
        'noise_scale': NOISE_SCALE,
        'noise_config': dataset['noise_config'],
        'comparison_mode': 'shared_dataset_apples_to_apples',
        'label': '42-GM1-R61-SHARED-NOISE1/3',
        'round61_selected_candidate': candidate['name'],
    })
    r61_payload = compute_payload(
        source_mod,
        r61_clbt,
        params,
        variant='42state_gm1_round61_h_scd_state20_microtight_commit_shared_noise1over3',
        method_file=str(R61_METHOD_FILE),
        extra=r61_extra,
    )
    OUT_R61.write_text(json.dumps(r61_payload, ensure_ascii=False, indent=2), encoding='utf-8')

    compare_keys = ['dKg_xx', 'dKg_xy', 'dKg_yy', 'dKg_zz', 'dKa_xx', 'rx_y', 'ry_z']
    comparison = {
        'mode': 'shared_dataset_apples_to_apples',
        'noise_scale': NOISE_SCALE,
        'noise_config': dataset['noise_config'],
        'baseline_json': str(OUT_BASELINE),
        'round61_json': str(OUT_R61),
        'key_params': {},
        'overall': {},
    }

    for key in compare_keys:
        b = baseline_payload['param_errors'][key]['pct_error']
        r = r61_payload['param_errors'][key]['pct_error']
        comparison['key_params'][key] = {
            'baseline_pct_error': b,
            'round61_pct_error': r,
            'delta_pct_points': b - r,
            'relative_improvement_pct': ((b - r) / b * 100.0) if abs(b) > 1e-15 else None,
        }

    for key in ['mean_pct_error', 'median_pct_error', 'max_pct_error']:
        b = baseline_payload['overall'][key]
        r = r61_payload['overall'][key]
        comparison['overall'][key] = {
            'baseline': b,
            'round61': r,
            'delta_pct_points': b - r,
            'relative_improvement_pct': ((b - r) / b * 100.0) if abs(b) > 1e-15 else None,
        }

    OUT_COMPARE.write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding='utf-8')

    print('__RESULT_JSON__=' + json.dumps({
        'mode': comparison['mode'],
        'noise_scale': NOISE_SCALE,
        'noise_config': dataset['noise_config'],
        'baseline_json': str(OUT_BASELINE),
        'round61_json': str(OUT_R61),
        'compare_json': str(OUT_COMPARE),
        'key_params': comparison['key_params'],
        'overall': comparison['overall'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
