"""Centralized MLflow client configuration.

By default, runs log to a local mlruns/ folder. Point at a remote tracking
server by exporting MLFLOW_TRACKING_URI before launching training:

    export MLFLOW_TRACKING_URI=http://your-host:5000
    python -m training.train_m2_upright
"""

from __future__ import annotations

import os

import mlflow

DEFAULT_TRACKING_URI = "file:./mlruns"
DEFAULT_EXPERIMENT = "triple-pendulum-sim2real"


def init_mlflow(experiment: str = DEFAULT_EXPERIMENT) -> str:
    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)
    return uri


if __name__ == "__main__":
    uri = init_mlflow()
    print(f"MLflow tracking URI: {uri}")
    print(f"Experiment         : {DEFAULT_EXPERIMENT}")
    with mlflow.start_run(run_name="env_smoke_test"):
        mlflow.log_param("status", "alive")
        mlflow.log_metric("ping", 1.0)
        print("Smoke run logged.")
