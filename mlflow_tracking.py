from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import mlflow


DEFAULT_TRACKING_DIR = Path("./mlruns")


def configure_local_tracking(tracking_dir: Path | str = DEFAULT_TRACKING_DIR) -> Path:
    tracking_path = Path(tracking_dir).resolve()
    tracking_path.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_path.as_uri())
    return tracking_path


def start_experiment_run(*, experiment_name: str, run_name: str | None = None):
    configure_local_tracking()
    mlflow.set_experiment(experiment_name)
    return mlflow.start_run(run_name=run_name)


def log_run_parameters(params: dict[str, Any]) -> None:
    mlflow.log_params(params)


def log_run_metrics(metrics: dict[str, float]) -> None:
    mlflow.log_metrics(metrics)


def log_confusion_matrix(matrix: Any, artifact_name: str = "confusion_matrix.json") -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        artifact_file = Path(temp_dir) / artifact_name
        with artifact_file.open("w") as tmp_file:
            json.dump({"confusion_matrix": matrix}, tmp_file, indent=2)

        mlflow.log_artifact(str(artifact_file), artifact_path="evaluation")


def log_artifact(path: Path | str, artifact_path: str | None = None) -> None:
    artifact_file = Path(path)
    if artifact_file.exists():
        mlflow.log_artifact(str(artifact_file), artifact_path=artifact_path)


def end_run() -> None:
    if mlflow.active_run() is not None:
        mlflow.end_run()
