"""Plot benchmark metrics for psins calibration methods."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

METRIC_KEYS = ["eb_norm", "db_norm", "Ka2_norm", "rx_norm", "ry_norm"]


def load_metrics(path: str | Path):
    return json.loads(Path(path).read_text())


def plot_metric_bars(metrics, outdir: str | Path):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    methods = [m["method"] for m in metrics]
    for key in METRIC_KEYS:
        values = [m.get(key) for m in metrics]
        plt.figure(figsize=(8, 4))
        plt.bar(methods, values)
        plt.title(key)
        plt.ylabel(key)
        plt.tight_layout()
        plt.savefig(outdir / f"{key}.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python plot_metrics.py metrics.json outdir")
        raise SystemExit(1)
    metrics = load_metrics(sys.argv[1])
    plot_metric_bars(metrics, sys.argv[2])
    print(f"plots saved to {sys.argv[2]}")
