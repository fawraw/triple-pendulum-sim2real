"""Resilient wrappers around mlflow.set_tag / log_param / log_metric.

A late-run network blip on a tag/param call must NOT abort an 8h training.
Use these helpers everywhere training scripts touch MLflow."""
from __future__ import annotations

import mlflow


def safe_tag(key: str, value) -> None:
    try:
        mlflow.set_tag(key, value if not isinstance(value, str) else value[:500])
    except Exception as exc:
        print(f"[mlflow_safe] set_tag({key!r}) failed: {exc}", flush=True)


def safe_param(key: str, value) -> None:
    try:
        mlflow.log_param(key, value)
    except Exception as exc:
        print(f"[mlflow_safe] log_param({key!r}) failed: {exc}", flush=True)


def safe_metric(key: str, value: float, step: int | None = None) -> None:
    try:
        if step is None:
            mlflow.log_metric(key, float(value))
        else:
            mlflow.log_metric(key, float(value), step=step)
    except Exception as exc:
        print(f"[mlflow_safe] log_metric({key!r}) failed: {exc}", flush=True)


def safe_artifact(local_path: str, artifact_path: str | None = None) -> bool:
    """Returns True on success, False on failure. Caller may want to set a
    tag with the local path so the artifact can be recovered manually."""
    try:
        if artifact_path:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(local_path)
        return True
    except Exception as exc:
        print(f"[mlflow_safe] log_artifact({local_path!r}) failed: {exc}", flush=True)
        safe_tag("artifact_path_local", local_path)
        safe_tag("artifact_log_error", repr(exc))
        return False
