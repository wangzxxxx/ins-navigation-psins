"""Run selected psins benchmark methods, extract metrics, and plot results.

当前先串三种已拆方法：
- clean_baseline
- noisy_baseline
- scd
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, '/root/.openclaw/workspace/psins_method_bench')

ROOT = Path('/root/.openclaw/workspace/psins_method_bench')
VENV_PY = Path('/root/.openclaw/workspace/.venv_psins/bin/python')
EXTRACTED = ROOT / 'methods' / 'correlation_decay' / 'extracted_methods'
RESULTS = ROOT / 'results'
PLOTS = ROOT / 'summary' / 'plots'

METHODS = {
    'clean_baseline': EXTRACTED / 'method_clean_baseline.py',
    'noisy_baseline': EXTRACTED / 'method_noisy_baseline.py',
    'scd': EXTRACTED / 'method_scd.py',
}


def run_method(path: Path):
    code = (
        "import json; from pathlib import Path; "
        "g={}; exec(Path(r'%s').read_text(), g); "
        "res = g['run_calibration'](*g['build_dataset']()[:1], None, None)"
    )
    raise RuntimeError('unused')


def run_method_via_import(method_name: str, path: Path):
    wrapper = f'''
import json
import sys
from pathlib import Path
sys.path.insert(0, r"{ROOT}")
sys.path.insert(0, r"{EXTRACTED}")
from common_setup import build_dataset, run_calibration
from summary.extract_metrics import extract_result_metrics
imu_clean, imu_noisy, pos0, ts = build_dataset()
if "{method_name}" == "clean_baseline":
    res = run_calibration(imu_clean, pos0, ts, scd_mode=False, label='Clean')
elif "{method_name}" == "noisy_baseline":
    res = run_calibration(imu_noisy, pos0, ts, scd_mode=False, label='Noisy KF')
else:
    res = run_calibration(imu_noisy, pos0, ts, scd_mode=True, label='SCD')
metrics = extract_result_metrics("{method_name}", res)
print(json.dumps(metrics, ensure_ascii=False))
'''
    proc = subprocess.run([str(VENV_PY), '-c', wrapper], capture_output=True, text=True, cwd='/root/.openclaw/workspace')
    return proc


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for name, path in METHODS.items():
        print(f'== running {name} ==')
        proc = run_method_via_import(name, path)
        (RESULTS / f'{name}.stdout.txt').write_text(proc.stdout)
        (RESULTS / f'{name}.stderr.txt').write_text(proc.stderr)
        if proc.returncode != 0:
            print(f'{name} failed: {proc.returncode}')
            continue
        lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
        metrics = json.loads(lines[-1])
        all_metrics.append(metrics)
        print(f'{name} ok')
    out = RESULTS / 'metrics.json'
    out.write_text(json.dumps(all_metrics, ensure_ascii=False, indent=2))
    print(f'metrics saved to {out}')
    if all_metrics:
        from summary.plot_metrics import plot_metric_bars
        plot_metric_bars(all_metrics, PLOTS)
        print(f'plots saved to {PLOTS}')


if __name__ == '__main__':
    main()
