"""Pull a run's metrics from MLflow and save a learning curve PNG.

Usage:
    python scripts/plot_learning_curve.py \\
        --tracking-uri http://10.1.4.230:5000 \\
        --run-id <run_id> \\
        --out assets/learning_curve_m2.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracking-uri", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", default="assets/learning_curve.png")
    args = ap.parse_args()

    mlflow.set_tracking_uri(args.tracking_uri)
    client = mlflow.tracking.MlflowClient()

    history = {}
    for key in ("rollout_ep_rew_mean", "rollout_ep_rew_min", "rollout_ep_rew_max"):
        history[key] = client.get_metric_history(args.run_id, key)

    run = client.get_run(args.run_id)
    name = run.info.run_name
    total_ts = int(run.data.params.get("total_timesteps", "?"))

    fig, ax = plt.subplots(figsize=(10, 5))
    if history["rollout_ep_rew_min"] and history["rollout_ep_rew_max"]:
        ax.fill_between(
            [h.step for h in history["rollout_ep_rew_mean"]],
            [h.value for h in history["rollout_ep_rew_min"]],
            [h.value for h in history["rollout_ep_rew_max"]],
            alpha=0.2, label="rollout min/max",
        )
    ax.plot(
        [h.step for h in history["rollout_ep_rew_mean"]],
        [h.value for h in history["rollout_ep_rew_mean"]],
        "-", linewidth=2, label="rollout mean (last 50 ep.)",
    )
    ax.set_xlabel("timesteps")
    ax.set_ylabel("episode return")
    ax.set_title(f"{name}  (total {total_ts:,} steps)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
