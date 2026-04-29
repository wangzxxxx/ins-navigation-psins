from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/root/.openclaw/workspace/psins_method_bench')
METHOD_DIR = ROOT / 'methods' / 'adaptive_robust'
RESULTS = ROOT / 'results'
VENV_PY = Path('/root/.openclaw/workspace/.venv_psins/bin/python')

sys.path.insert(0, str(ROOT))
from summary.extract_metrics import extract_result_metrics

METHODS = {
    'adaptive_rq': METHOD_DIR / 'method_adaptive_rq.py',
    'huber_robust': METHOD_DIR / 'method_huber_robust.py',
    'innovation_gating': METHOD_DIR / 'method_innovation_gating.py',
    'attention_inflation': METHOD_DIR / 'method_attention_inflation.py',
}


def run_metric_method(name: str, path: Path):
    wrapper = f'''
import json
import sys
from pathlib import Path
sys.path.insert(0, r"{ROOT}")
sys.path.insert(0, r"{METHOD_DIR}")
g = {{}}
exec(Path(r"{path}").read_text(), g)
res = g['run_method']()
from summary.extract_metrics import extract_result_metrics
print(json.dumps(extract_result_metrics("{name}", res), ensure_ascii=False))
'''
    return subprocess.run([str(VENV_PY), '-c', wrapper], capture_output=True, text=True, cwd='/root/.openclaw/workspace')


def run_script_method(path: Path):
    return subprocess.run([str(VENV_PY), str(path)], capture_output=True, text=True, cwd='/root/.openclaw/workspace')


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = []
    metric_methods = {'adaptive_rq', 'huber_robust', 'innovation_gating'}
    for name, path in METHODS.items():
        if name in metric_methods:
            proc = run_metric_method(name, path)
        else:
            proc = run_script_method(path)
        (RESULTS / f'{name}.stdout.txt').write_text(proc.stdout)
        (RESULTS / f'{name}.stderr.txt').write_text(proc.stderr)
        record = {
            'method': name,
            'path': str(path),
            'returncode': proc.returncode,
            'status': 'ok' if proc.returncode == 0 else 'failed',
            'stdout_file': str(RESULTS / f'{name}.stdout.txt'),
            'stderr_file': str(RESULTS / f'{name}.stderr.txt'),
        }
        if proc.returncode == 0 and name in metric_methods:
            lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
            try:
                record['metrics'] = json.loads(lines[-1])
            except Exception as e:
                record['status'] = 'parse_failed'
                record['parse_error'] = str(e)
        elif proc.returncode != 0:
            record['error_tail'] = (proc.stderr or proc.stdout)[-1200:]
        out.append(record)
    out_path = RESULTS / 'adaptive_robust_results.json'
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(out_path)


if __name__ == '__main__':
    main()
