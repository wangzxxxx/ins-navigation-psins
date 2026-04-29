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
from probe_round58_llm_guided import ROUND58_LLM_CANDIDATES, _merge_round58_candidate

SOURCE = 'test_calibration_markov_pruned.py'
METHOD = '42-state GM1 round58 llm-guided alpha12 + lever micro-plus'
FAMILY = 'pruned_markov'
VARIANT = '42state_gm1_round58_llm_guided_alpha12_lever_plus'
CANDIDATE_NAME = 'llm_alpha12_plus_lever_plus'


def _pick_candidate():
    for candidate in ROUND58_LLM_CANDIDATES:
        if candidate['name'] == CANDIDATE_NAME:
            return candidate
    raise KeyError(f'Candidate not found: {CANDIDATE_NAME}')


def run_method():
    candidate = _pick_candidate()
    merged_candidate = _merge_round58_candidate(candidate)
    method_mod = load_module('markov_method_r58_llm_guided_alpha12_lever_plus', str(R53_METHOD_FILE))
    method_mod = _build_patched_method(method_mod, merged_candidate)
    result = list(method_mod.run_method())
    extra = result[4] if len(result) >= 5 else {}
    if isinstance(extra, dict):
        extra = dict(extra)
        extra.update({
            'round58_selected_candidate': candidate['name'],
            'round58_candidate_description': candidate['description'],
            'round58_candidate_rationale': candidate['rationale'],
            'round58_llm_guided': True,
            'round58_base_round57_candidate': 'state12_guard_keep_yyka',
            'round58_base_round56_candidate': 'commit_yyka_micro_up_pair_stronger',
            'round58_base_round55_candidate': 'alpha_split_xy_up_zz_damp',
            'round58_policy_patch': merged_candidate.get('iter_patches', {}),
            'round58_llm_extra_patch': candidate.get('iter_patches', {}),
            'round58_post_rx_y_mult': float(merged_candidate.get('post_rx_y_mult', 1.0)),
            'round58_post_ry_z_mult': float(merged_candidate.get('post_ry_z_mult', 1.0)),
            'round58_selection_rationale': 'Selected from round58 llm-guided probes because it preserved the full R57 protected stack, further improved dKg_xx/max, and also reduced mean by pairing the proven state12 micro-guard with an equally tiny positive lever confirmation.',
            'policy': 'Round58 keeps the R57 state12 micro-guard and yy/Ka + lever preservation stack, adds only a slightly stronger iter2 state12 alpha lift, and nudges the post rx_y/ry_z confirmation by an equally small amount to co-improve max and mean without opening any new large search direction.',
        })
        result[4] = extra
    return tuple(result)


if __name__ == '__main__':
    emit_result(summarize_result(METHOD, SOURCE, FAMILY, VARIANT, run_method()))
