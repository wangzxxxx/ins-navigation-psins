from __future__ import annotations

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
R53_METHOD_FILE = METHOD_DIR / 'method_42state_gm1_round53_internalized_trustcov_release.py'

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from common_markov import TMP_PSINS, emit_result, load_module, summarize_result
from probe_round55_newline import CANDIDATES, _build_patched_method

SOURCE = 'test_calibration_markov_pruned.py'
METHOD = '42-state GM1 round55 internalized targeted xy/zz repair'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round55_internalized_xyzz_targeted_repair'
CANDIDATE_NAME = 'alpha_split_xy_up_zz_damp'


def _pick_candidate():
    for candidate in CANDIDATES:
        if candidate['name'] == CANDIDATE_NAME:
            return candidate
    raise KeyError(f'Candidate not found: {CANDIDATE_NAME}')


def run_method():
    candidate = _pick_candidate()
    method_mod = load_module('markov_method_r55_from_r53_targeted_repair', str(R53_METHOD_FILE))
    method_mod = _build_patched_method(method_mod, candidate)
    result = list(method_mod.run_method())
    extra = result[4] if len(result) >= 5 else {}
    if isinstance(extra, dict):
        extra = dict(extra)
        extra.update({
            'round55_selected_candidate': candidate['name'],
            'round55_candidate_description': candidate['description'],
            'round55_policy_patch': candidate.get('iter_patches', {}),
            'round55_post_rx_y_mult': float(candidate.get('post_rx_y_mult', 1.0)),
            'round55_post_ry_z_mult': float(candidate.get('post_ry_z_mult', 1.0)),
            'round55_selection_rationale': 'Selected from round55 newline probes because it delivered a large dKg_xy drop, a meaningful dKg_zz drop, and a clearly better overall mean while keeping dKg_xx essentially intact. Tiny yy / Ka_xx / lever regressions remained within acceptable micro-variation for this stage.',
            'policy': 'Round55 keeps the Round53 trust/covariance/release internalization backbone, but adds a targeted asymmetric repair focused on state15/state20 instead of global alpha-only micro-tweaks.',
        })
        result[4] = extra
    return tuple(result)


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
