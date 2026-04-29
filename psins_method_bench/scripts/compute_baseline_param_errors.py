import json
from pathlib import Path
import numpy as np
import importlib.util

ROOT = Path('/root/.openclaw/workspace')
OUT = ROOT / 'psins_method_bench' / 'results' / 'baseline_param_error_table.json'

METHOD_FILES = {
    'A_noisy_baseline': ROOT / 'psins_method_bench' / 'methods' / 'correlation_decay' / 'extracted_methods' / 'method_noisy_baseline.py',
    'S_scd': ROOT / 'psins_method_bench' / 'methods' / 'correlation_decay' / 'extracted_methods' / 'method_scd.py',
    'M_markov_42state_gm1': ROOT / 'psins_method_bench' / 'methods' / 'markov' / 'method_42state_gm1.py',
    'B_markov_42state_gm1_scdnoise': ROOT / 'psins_method_bench' / 'methods' / 'markov' / 'method_42state_gm1_scdnoise.py',
}
CORR_COMMON = ROOT / 'psins_method_bench' / 'methods' / 'correlation_decay' / 'extracted_methods' / 'common_setup.py'

TRUTH = {
    'eb': np.array([0.1, 0.2, 0.3]),   # dph
    'db': np.array([100.0, 200.0, 300.0]),  # ug
    'Ka2': np.array([10.0, 20.0, 30.0]),    # ugpg2
    'rx': np.array([1.0, 2.0, 3.0]) / 100.0,
    'ry': np.array([4.0, 5.0, 6.0]) / 100.0,
}

# unit scaling from internal SI-ish units back to human-readable truth units
# based on source definitions using glv constants.
CONVERTERS = {
    'eb': lambda arr, mod: np.array(arr, dtype=float) / mod.glv.dph,
    'db': lambda arr, mod: np.array(arr, dtype=float) / mod.glv.ug,
    'Ka2': lambda arr, mod: np.array(arr, dtype=float) / mod.glv.ugpg2,
    'rx': lambda arr, mod: np.array(arr, dtype=float),
    'ry': lambda arr, mod: np.array(arr, dtype=float),
}


def load_module(name, path):
    import sys
    path = Path(path)
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_and_extract(label, path):
    if label == 'A_noisy_baseline' or label == 'S_scd':
        common = load_module('corr_common_setup', CORR_COMMON)
        imu_clean, imu_noisy, pos0, ts = common.build_dataset()
        scd_mode = (label == 'S_scd')
        result = common.run_calibration(imu_noisy, pos0, ts, scd_mode=scd_mode, label=('SCD' if scd_mode else 'Noisy KF'))
        mod = load_module('corr_decay_truth_mod', ROOT / 'tmp_psins_py' / 'psins_py' / 'correlation_decay_llm' / 'test_calibration_correlation_decay.py')
    else:
        mod = load_module(f'mod_{label}', path)
        result = mod.run_method()
    final_state = result[0] if isinstance(result, tuple) else result
    rows = []
    summary = {}
    pct_values = []
    for group in ['eb', 'db', 'Ka2', 'rx', 'ry']:
        est = CONVERTERS[group](final_state[group], mod)
        truth = TRUTH[group]
        group_pct = []
        for i, (e, t) in enumerate(zip(est, truth)):
            abs_err = abs(float(e) - float(t))
            pct_err = abs_err / abs(float(t)) * 100.0 if abs(float(t)) > 1e-18 else None
            rows.append({
                'method': label,
                'param': f'{group}_{i}',
                'truth': float(t),
                'estimate': float(e),
                'abs_error': abs_err,
                'pct_error': pct_err,
            })
            if pct_err is not None:
                group_pct.append(pct_err)
                pct_values.append(pct_err)
        summary[group + '_mean_pct_error'] = float(np.mean(group_pct)) if group_pct else None
    summary['overall_mean_pct_error'] = float(np.mean(pct_values)) if pct_values else None
    summary['overall_median_pct_error'] = float(np.median(pct_values)) if pct_values else None
    summary['worst_param_pct_error'] = float(np.max(pct_values)) if pct_values else None
    return {'method': label, 'summary': summary, 'rows': rows}


def main():
    data = {}
    for label, path in METHOD_FILES.items():
        print('RUN', label, flush=True)
        data[label] = run_and_extract(label, path)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    print(f'WROTE {OUT}', flush=True)


if __name__ == '__main__':
    main()
