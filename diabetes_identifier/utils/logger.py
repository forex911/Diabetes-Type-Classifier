"""Structured JSON logger and MLflow ExperimentTracker wrapper."""
from __future__ import annotations

import logging
import sys
from typing import Any, Dict, Optional

try:
    from pythonjsonlogger import jsonlogger  # type: ignore
    _JSON_LOGGER_AVAILABLE = True
except ImportError:
    _JSON_LOGGER_AVAILABLE = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a logger that emits structured JSON records.

    Falls back to a plain text formatter when ``python-json-logger`` is not
    installed so the module remains importable in minimal environments.

    Args:
        name:  Logger name (typically ``__name__``).
        level: Logging level (default ``logging.INFO``).

    Returns:
        A configured :class:`logging.Logger` instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured — avoid duplicate handlers.
        logger.setLevel(level)
        return logger

    handler = logging.StreamHandler(sys.stdout)

    if _JSON_LOGGER_AVAILABLE:
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


_tracker_logger = get_logger(__name__)


class ExperimentTracker:
    """Thin MLflow wrapper for experiment tracking.

    All MLflow imports are deferred so the class can be instantiated in
    environments where MLflow is not yet installed (e.g. during unit tests
    that don't exercise tracking).
    """

    def __init__(self, tracking_uri: str = "mlruns", experiment_name: str = "diabetes-classifier") -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self._run = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, Any]] = None) -> str:
        """Create a new MLflow run and return its run ID.

        Args:
            run_name: Optional human-readable name for the run.
            tags:     Optional dict of string tags to attach to the run.

        Returns:
            The MLflow run ID string.
        """
        import mlflow  # type: ignore

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self._run = mlflow.start_run(run_name=run_name, tags=tags)
        run_id = self._run.info.run_id
        _tracker_logger.info("MLflow run started", extra={"run_id": run_id, "experiment": self.experiment_name})
        print(f"MLflow run ID: {run_id}")
        return run_id

    def log_params(self, params: Dict[str, Any]) -> None:
        """Log a dictionary of parameters to the active MLflow run.

        Args:
            params: Key-value pairs where values are coerced to strings by MLflow.
        """
        import mlflow  # type: ignore

        mlflow.log_params(params)
        _tracker_logger.debug("Logged params", extra={"params": params})

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """Log a dictionary of numeric metrics to the active MLflow run.

        Args:
            metrics: Key-value pairs of metric name → float value.
            step:    Optional step index (e.g. fold number).
        """
        import mlflow  # type: ignore

        mlflow.log_metrics(metrics, step=step)
        _tracker_logger.debug("Logged metrics", extra={"metrics": metrics, "step": step})

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Upload a local file or directory as an MLflow artifact.

        Args:
            local_path:    Path to the local file or directory.
            artifact_path: Optional subdirectory within the run's artifact store.
        """
        import mlflow  # type: ignore

        mlflow.log_artifact(local_path, artifact_path=artifact_path)
        _tracker_logger.info("Logged artifact", extra={"local_path": local_path})

    def register_model(self, model_uri: str, name: str = "diabetes-classifier-production") -> None:
        """Register a model version in the MLflow Model Registry.

        Args:
            model_uri: MLflow model URI, e.g. ``"runs:/<run_id>/model"``.
            name:      Registered model name (default ``"diabetes-classifier-production"``).
        """
        import mlflow  # type: ignore

        mlflow.register_model(model_uri=model_uri, name=name)
        _tracker_logger.info("Registered model", extra={"model_uri": model_uri, "name": name})

    def end_run(self) -> None:
        """End the active MLflow run."""
        import mlflow  # type: ignore

        mlflow.end_run()
        self._run = None
        _tracker_logger.info("MLflow run ended")
