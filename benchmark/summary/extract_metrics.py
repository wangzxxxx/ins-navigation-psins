"""Extract benchmark metrics from calibration result tuples.

输入：各方法返回的 result tuple
输出：适合汇总成表的核心指标字典
"""
from __future__ import annotations

import numpy as np


def extract_result_metrics(name: str, result):
    if not isinstance(result, tuple) or len(result) < 5:
        raise ValueError(f"Unexpected result format for {name}: {type(result)}")

    final_state, kf_state, p_trace, x_trace, iter_info = result[:5]

    metrics = {
        "method": name,
        "tuple_len": len(result),
        "final_keys": sorted(final_state.keys()) if isinstance(final_state, dict) else [],
        "eb_norm": float(np.linalg.norm(final_state.get("eb", np.zeros(3)))) if isinstance(final_state, dict) else None,
        "db_norm": float(np.linalg.norm(final_state.get("db", np.zeros(3)))) if isinstance(final_state, dict) else None,
        "Ka2_norm": float(np.linalg.norm(final_state.get("Ka2", np.zeros(3)))) if isinstance(final_state, dict) else None,
        "rx_norm": float(np.linalg.norm(final_state.get("rx", np.zeros(3)))) if isinstance(final_state, dict) else None,
        "ry_norm": float(np.linalg.norm(final_state.get("ry", np.zeros(3)))) if isinstance(final_state, dict) else None,
        "p_trace_shape": getattr(p_trace, "shape", None),
        "x_trace_shape": getattr(x_trace, "shape", None),
        "iter_info_type": type(iter_info).__name__,
    }
    return metrics


def metrics_table(metrics_list):
    header = ["method", "eb_norm", "db_norm", "Ka2_norm", "rx_norm", "ry_norm", "p_trace_shape", "x_trace_shape"]
    rows = [header]
    for m in metrics_list:
        rows.append([
            m.get("method"),
            m.get("eb_norm"),
            m.get("db_norm"),
            m.get("Ka2_norm"),
            m.get("rx_norm"),
            m.get("ry_norm"),
            m.get("p_trace_shape"),
            m.get("x_trace_shape"),
        ])
    return rows
