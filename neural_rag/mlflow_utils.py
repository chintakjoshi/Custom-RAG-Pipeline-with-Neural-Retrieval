from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from typing import Any, Iterator, Mapping


def flatten_mapping(
    mapping: Mapping[str, Any],
    *,
    prefix: str = "",
) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in mapping.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(flatten_mapping(value, prefix=name))
            continue

        if value is None:
            continue

        if isinstance(value, Path):
            flat[name] = str(value)
            continue

        if isinstance(value, (list, tuple, set)):
            flat[name] = json.dumps(list(value))
            continue

        flat[name] = value

    return flat


def sanitize_metric_name(name: str) -> str:
    return (
        name.replace("@", "_at_")
        .replace("-", "_")
        .replace("/", "_")
        .replace(" ", "_")
    )


def _resolve_tracking_uri(
    tracking_uri: str | None,
) -> tuple[str | None, Path | None]:
    if not tracking_uri:
        return None, None

    if "://" in tracking_uri or tracking_uri.startswith("file:"):
        return tracking_uri, None

    tracking_root = Path(tracking_uri).resolve()
    tracking_root.mkdir(parents=True, exist_ok=True)
    backend_uri = "sqlite:///" + (tracking_root / "mlflow.db").as_posix()
    return backend_uri, tracking_root


def _sanitize_artifact_dir_name(name: str) -> str:
    return "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in name)


class MLflowTracker:
    def __init__(self, mlflow_module: Any | None = None) -> None:
        self._mlflow = mlflow_module
        self.enabled = mlflow_module is not None

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if not self.enabled:
            return

        payload = {
            str(key): str(value)
            for key, value in tags.items()
            if value is not None
        }
        if payload:
            self._mlflow.set_tags(payload)

    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self.enabled:
            return

        payload = {
            str(key): str(value)
            for key, value in params.items()
            if value is not None
        }
        if payload:
            self._mlflow.log_params(payload)

    def log_metrics(self, metrics: Mapping[str, Any]) -> None:
        if not self.enabled:
            return

        payload: dict[str, float] = {}
        for key, value in metrics.items():
            if value is None:
                continue
            try:
                payload[str(key)] = float(value)
            except (TypeError, ValueError):
                continue

        if payload:
            self._mlflow.log_metrics(payload)

    def log_dict(self, payload: Any, artifact_file: str) -> None:
        if self.enabled:
            self._mlflow.log_dict(payload, artifact_file)

    def log_text(self, text: str, artifact_file: str) -> None:
        if self.enabled:
            self._mlflow.log_text(text, artifact_file)

    def log_artifact(
        self,
        path: str | Path,
        *,
        artifact_path: str | None = None,
    ) -> None:
        if not self.enabled:
            return

        artifact = Path(path)
        if artifact.exists():
            self._mlflow.log_artifact(str(artifact.resolve()), artifact_path=artifact_path)


@contextmanager
def start_mlflow_run(
    mlflow_config: Mapping[str, Any] | None,
    *,
    config_path: str | Path | None = None,
    default_run_name: str,
    default_experiment_name: str = "custom-rag-pipeline",
    default_tags: Mapping[str, Any] | None = None,
) -> Iterator[MLflowTracker]:
    config = dict(mlflow_config or {})
    if not bool(config.get("enabled", False)):
        yield MLflowTracker()
        return

    try:
        import mlflow
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "MLflow logging was enabled in the config, but the mlflow package is not installed."
        ) from exc

    tracking_uri, tracking_root = _resolve_tracking_uri(
        str(config.get("tracking_uri", "artifacts/mlruns"))
    )
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_registry_uri(tracking_uri)

    experiment_name = str(config.get("experiment_name", default_experiment_name))
    experiment_id: str | None = None

    if tracking_root is not None:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is None:
            artifact_root = tracking_root / "artifacts" / _sanitize_artifact_dir_name(experiment_name)
            artifact_root.mkdir(parents=True, exist_ok=True)
            experiment_id = client.create_experiment(
                experiment_name,
                artifact_location=artifact_root.resolve().as_uri(),
            )
        else:
            experiment_id = experiment.experiment_id
    else:
        mlflow.set_experiment(experiment_name)

    start_run_kwargs = {"run_name": str(config.get("run_name", default_run_name))}
    if experiment_id is not None:
        start_run_kwargs["experiment_id"] = experiment_id

    with mlflow.start_run(**start_run_kwargs):
        tracker = MLflowTracker(mlflow)
        merged_tags = {
            str(key): value
            for key, value in (default_tags or {}).items()
            if value is not None
        }
        merged_tags.update(
            {
                str(key): value
                for key, value in dict(config.get("tags", {})).items()
                if value is not None
            }
        )
        tracker.set_tags(merged_tags)

        if config_path is not None:
            tracker.log_artifact(config_path, artifact_path="configs")

        yield tracker
