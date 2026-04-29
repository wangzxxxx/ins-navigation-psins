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

from common_markov import emit_result, load_module, summarize_result
from probe_round55_newline import _build_patched_method
from probe_round57_narrow import ROUND57_CANDIDATES, _merge_round57_candidate

SOURCE = 'test_calibration_markov_pruned.py'
METHOD = '42-state GM1 round57 internalized xy/zz/yy/Ka preserve + state12 micro-guard'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round57_internalized_xyzz_yyka_state12_guard'
CANDIDATE_NAME = 'state12_guard_keep_yyka'


def _pick_candidate():
    for candidate in ROUND57_CANDIDATES:
        if candidate['name'] == CANDIDATE_NAME:
            return candidate
    raise KeyError(f'Candidate not found: {CANDIDATE_NAME}')


def run_method():
    candidate = _pick_candidate()
    merged_candidate = _merge_round57_candidate(candidate)
    method_mod = load_module('markov_method_r57_from_r53_state12_guard', str(R53_METHOD_FILE))
    method_mod = _build_patched_method(method_mod, merged_candidate)
    result = list(method_mod.run_method())
    extra = result[4] if len(result) >= 5 else {}
    if isinstance(extra, dict):
        extra = dict(extra)
        extra.update({
            'round57_selected_candidate': candidate['name'],
            'round57_candidate_description': candidate['description'],
            'round57_base_round56_candidate': 'commit_yyka_micro_up_pair_stronger',
            'round57_base_round55_candidate': 'alpha_split_xy_up_zz_damp',
            'round57_policy_patch': merged_candidate.get('iter_patches', {}),
            'round57_extra_patch': candidate.get('iter_patches', {}),
            'round57_post_rx_y_mult': float(merged_candidate.get('post_rx_y_mult', 1.0)),
            'round57_post_ry_z_mult': float(merged_candidate.get('post_ry_z_mult', 1.0)),
            'round57_selection_rationale': 'Selected from round57 narrow probes because it cleanly improved dKg_xx and overall max while preserving the full Round56 repair stack on dKg_xy, dKg_yy, dKg_zz, dKa_xx, rx_y, ry_z, and also shaving overall mean. The only movement outside xx/max/mean was a negligible numerical improvement on dKg_xy.',
            'policy': 'Round57 keeps the Round56 targeted state15/state20 profile, retains the tiny iter2 state16/state21 upward repair and paired lever guard, and adds only a very small iter2 state12 upward micro-guard to suppress the remaining dKg_xx/max drift.',
        })
        result[4] = extra
    return tuple(result)


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
