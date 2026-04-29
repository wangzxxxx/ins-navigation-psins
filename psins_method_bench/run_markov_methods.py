from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path('/root/.openclaw/workspace/psins_method_bench')
METHODS_DIR = ROOT / 'methods' / 'markov'
RESULTS_DIR = ROOT / 'results'
PYTHON = Path('/root/.openclaw/workspace/.venv_psins/bin/python')

METHODS = {
    '42state_gm1': METHODS_DIR / 'method_42state_gm1.py',
    '48state_gm2': METHODS_DIR / 'method_48state_gm2.py',
    '46state_gm1_correct': METHODS_DIR / 'method_46state_gm1_correct.py',
    '46state_gm1_wrong': METHODS_DIR / 'method_46state_gm1_wrong.py',
    '49state_gm1_correct': METHODS_DIR / 'method_49state_gm1_correct.py',
    '49state_gm1_wrong': METHODS_DIR / 'method_49state_gm1_wrong.py',
    '55state_gm2': METHODS_DIR / 'method_55state_gm2.py',
}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for name, path in METHODS.items():
        proc = subprocess.run(
            [str(PYTHON), str(path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=3600,
        )
        (RESULTS_DIR / f'markov_{name}.stdout.txt').write_text(proc.stdout)
        (RESULTS_DIR / f'markov_{name}.stderr.txt').write_text(proc.stderr)

        record = {
            'method_key': name,
            'path': str(path),
            'returncode': proc.returncode,
            'stdout_file': str(RESULTS_DIR / f'markov_{name}.stdout.txt'),
            'stderr_file': str(RESULTS_DIR / f'markov_{name}.stderr.txt'),
            'ok': proc.returncode == 0,
        }

        marker_line = None
        for line in reversed(proc.stdout.splitlines()):
            if line.startswith('__RESULT_JSON__='):
                marker_line = line.split('=', 1)[1]
                break
        if marker_line:
            try:
                record['summary'] = json.loads(marker_line)
            except Exception as e:
                record['ok'] = False
                record['parse_error'] = str(e)
        elif proc.returncode == 0:
            record['ok'] = False
            record['parse_error'] = 'missing result marker'

        if proc.returncode != 0:
            record['error_excerpt'] = '\n'.join(proc.stderr.splitlines()[-20:])

        all_results.append(record)
        print(f"{name}: {'OK' if record['ok'] else 'FAIL'}")

    out_path = RESULTS_DIR / 'markov_results.json'
    out_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(f'saved {out_path}')

    failed = [r['method_key'] for r in all_results if not r['ok']]
    if failed:
        print('failed_methods=' + ','.join(failed))
        sys.exit(1)


if __name__ == '__main__':
    main()
