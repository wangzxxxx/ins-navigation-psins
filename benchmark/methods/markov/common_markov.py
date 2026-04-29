from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path('/root/.openclaw/workspace')
TMP_PSINS = ROOT / 'tmp_psins_py' / 'psins_py'


def load_module(module_name: str, file_path: str):
    path = Path(file_path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'Cannot load module from {file_path}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _norm3(value: Any):
    if isinstance(value, np.ndarray):
        return float(np.linalg.norm(value))
    if isinstance(value, (list, tuple)):
        return float(np.linalg.norm(np.array(value, dtype=float)))
    return None


def summarize_result(method_name: str, source: str, family: str, variant: str, result: Any):
    summary = {
        'method': method_name,
        'source': source,
        'family': family,
        'variant': variant,
        'result_type': type(result).__name__,
    }

    final_state = None
    kf_state = None
    p_trace = None
    x_trace = None
    iter_info = None

    if isinstance(result, dict):
        final_state = result
    elif isinstance(result, tuple):
        summary['tuple_len'] = len(result)
        if len(result) >= 1 and isinstance(result[0], dict):
            final_state = result[0]
        if len(result) >= 2 and isinstance(result[1], dict):
            kf_state = result[1]
        if len(result) >= 3:
            p_trace = result[2]
        if len(result) >= 4:
            x_trace = result[3]
        if len(result) >= 5:
            iter_info = result[4]

    if isinstance(final_state, dict):
        summary['final_keys'] = sorted(final_state.keys())
        for key in ['eb', 'db', 'Ka2', 'rx', 'ry', 'rz']:
            if key in final_state:
                summary[f'{key}_norm'] = _norm3(final_state[key])
        if 'tGA' in final_state:
            try:
                summary['tGA'] = float(final_state['tGA'])
            except Exception:
                pass

    if isinstance(kf_state, dict):
        summary['kf_keys'] = sorted(kf_state.keys())
        if 'Pxk' in kf_state and hasattr(kf_state['Pxk'], 'shape'):
            summary['Pxk_shape'] = list(kf_state['Pxk'].shape)
        if 'xk' in kf_state and hasattr(kf_state['xk'], 'shape'):
            summary['xk_shape'] = list(kf_state['xk'].shape)

    if hasattr(p_trace, 'shape'):
        summary['p_trace_shape'] = list(p_trace.shape)
    if hasattr(x_trace, 'shape'):
        summary['x_trace_shape'] = list(x_trace.shape)
    if iter_info is not None:
        summary['iter_info_type'] = type(iter_info).__name__

    return summary


def emit_result(summary: dict):
    print('__RESULT_JSON__=' + json.dumps(summary, ensure_ascii=False))
