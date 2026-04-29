from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

ROOT = Path('/root/.openclaw/workspace')
RESULTS = ROOT / 'psins_method_bench' / 'results'
VENV_PY = ROOT / '.venv_psins' / 'bin' / 'python'

METHODS = {
    'llm_masking_scheme_a': ROOT / 'psins_method_bench' / 'methods' / 'schmidt_shadow' / 'method_llm_masking_scheme_a.py',
    'shadow_observer': ROOT / 'psins_method_bench' / 'methods' / 'schmidt_shadow' / 'method_shadow_observer.py',
    'shadow_hybrid_update': ROOT / 'psins_method_bench' / 'methods' / 'schmidt_shadow' / 'method_shadow_hybrid_update.py',
    'llm_staged_graduation': ROOT / 'psins_method_bench' / 'methods' / 'hybrid_staged' / 'method_llm_staged_graduation.py',
    'hybrid_shadow_kf': ROOT / 'psins_method_bench' / 'methods' / 'hybrid_staged' / 'method_hybrid_shadow_kf.py',
}


def run_one(name: str, path: Path):
    env = os.environ.copy()
    env['MPLBACKEND'] = env.get('MPLBACKEND', 'Agg')
    proc = subprocess.run(
        [str(VENV_PY), str(path)],
        capture_output=True,
        text=True,
        cwd=str(ROOT / 'psins_method_bench' / 'results'),
        env=env,
    )
    (RESULTS / f'{name}.stdout.txt').write_text(proc.stdout)
    (RESULTS / f'{name}.stderr.txt').write_text(proc.stderr)

    result = {
        'method': name,
        'path': str(path),
        'returncode': proc.returncode,
        'stdout_file': str(RESULTS / f'{name}.stdout.txt'),
        'stderr_file': str(RESULTS / f'{name}.stderr.txt'),
    }

    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    if lines:
        try:
            result['summary'] = json.loads(lines[-1])
        except Exception:
            result['summary_parse_error'] = True
            result['stdout_tail'] = lines[-10:]
    else:
        result['stdout_tail'] = []
    if proc.stderr.strip():
        result['stderr_tail'] = proc.stderr.splitlines()[-10:]
    return result


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    all_results = []
    for name, path in METHODS.items():
        print(f'== running {name} ==')
        all_results.append(run_one(name, path))
    out = RESULTS / 'shadow_hybrid_results.json'
    out.write_text(json.dumps(all_results, ensure_ascii=False, indent=2))
    print(json.dumps({'saved': str(out), 'count': len(all_results)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
