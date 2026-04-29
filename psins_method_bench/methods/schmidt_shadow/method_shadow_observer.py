from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path('/root/.openclaw/workspace')
BENCH = ROOT / 'psins_method_bench'
TMP = ROOT / 'tmp_psins_py'
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))
if str(TMP) not in sys.path:
    sys.path.insert(0, str(TMP))

SOURCE = 'test_shadow_observer.py'
METHOD = 'shadow observer standalone test'
PLOT = ROOT / 'psins_method_bench' / 'results' / 'shadow_test_convergence.png'


def run_method():
    from psins_py.test_shadow_observer import main
    main()
    return {
        'method': 'shadow_observer',
        'source': SOURCE,
        'method_label': METHOD,
        'plot_exists': PLOT.exists(),
        'plot_path': str(PLOT),
    }


if __name__ == '__main__':
    print(json.dumps(run_method(), ensure_ascii=False))
