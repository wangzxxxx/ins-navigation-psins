from pathlib import Path
from dotenv import load_dotenv
PROJECT_ENV = Path('/root/.openclaw/workspace/tmp_psins_py/psins_py/.env')
if PROJECT_ENV.exists():
    load_dotenv(PROJECT_ENV, override=True)

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/root/.openclaw/workspace')
SCRIPT = ROOT / 'tmp_psins_py' / 'psins_py' / 'test_system_calibration_19pos_hybrid.py'
VENV_PY = ROOT / '.venv_psins' / 'bin' / 'python'
SOURCE = 'test_system_calibration_19pos_hybrid.py'
METHOD = 'shadow KF hybrid update with pseudo measurement + inflation'


def run_method():
    env = os.environ.copy()
    env['MPLBACKEND'] = env.get('MPLBACKEND', 'Agg')
    proc = subprocess.run(
        [str(VENV_PY), str(SCRIPT)],
        capture_output=True,
        text=True,
        cwd=str(ROOT / 'psins_method_bench'),
        env=env,
    )
    return {
        'method': 'shadow_hybrid_update',
        'source': SOURCE,
        'method_label': METHOD,
        'returncode': proc.returncode,
        'stdout_tail': proc.stdout.splitlines()[-20:],
        'stderr_tail': proc.stderr.splitlines()[-20:],
    }


if __name__ == '__main__':
    print(json.dumps(run_method(), ensure_ascii=False))
