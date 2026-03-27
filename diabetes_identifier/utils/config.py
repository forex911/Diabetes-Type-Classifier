"""Configuration dataclass hierarchy for the diabetes-classifier system."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IngestionConfig:
    csv_path: str = "data/raw/ehr_synthetic.csv"
    missing_strategy: str = "mean"  # "mean" | "median" | "drop"


@dataclass
class PreprocConfig:
    clip_sigma: float = 3.0
    smote_threshold: float = 0.15


@dataclass
class NLPConfig:
    default_notes_text: str = "no clinical notes provided"
    phi_tokens: List[str] = field(default_factory=lambda: ["[NAME]", "[DOB]", "[ID]", "[PHONE]", "[ADDRESS]", "[EMAIL]"])


@dataclass
class EmbeddingConfig:
    mode: str = "tfidf"  # "bert" | "tfidf"
    bert_model_name: str = "emilyalsentzer/Bio_ClinicalBERT"


@dataclass
class TrainingConfig:
    random_seed: int = 42
    ensemble_weights: List[float] = field(default_factory=lambda: [0.5, 0.5])
    class_weights: Optional[str] = "balanced"  # None | "balanced"


@dataclass
class EvaluationConfig:
    min_macro_f1: float = 0.70
    n_folds: int = 5


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"


@dataclass
class MLflowConfig:
    tracking_uri: str = "mlruns"
    experiment_name: str = "diabetes-classifier"


@dataclass
class Config:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    preprocessing: PreprocConfig = field(default_factory=PreprocConfig)
    nlp: NLPConfig = field(default_factory=NLPConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    api: APIConfig = field(default_factory=APIConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)


def _dict_to_config(data: Dict[str, Any]) -> Config:
    """Recursively build a Config from a plain dict."""
    def _build(cls, src: dict):
        import dataclasses
        if not dataclasses.is_dataclass(cls):
            return src
        kwargs: Dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            val = src.get(f.name)
            if val is None:
                continue
            if dataclasses.is_dataclass(f.type) or (
                isinstance(f.type, type) and dataclasses.is_dataclass(f.type)
            ):
                kwargs[f.name] = _build(f.type, val)
            else:
                # Resolve string type annotations
                actual_type = f.type
                if isinstance(actual_type, str):
                    actual_type = {
                        "IngestionConfig": IngestionConfig,
                        "PreprocConfig": PreprocConfig,
                        "NLPConfig": NLPConfig,
                        "EmbeddingConfig": EmbeddingConfig,
                        "TrainingConfig": TrainingConfig,
                        "EvaluationConfig": EvaluationConfig,
                        "APIConfig": APIConfig,
                        "MLflowConfig": MLflowConfig,
                    }.get(actual_type)
                if actual_type and isinstance(actual_type, type) and dataclasses.is_dataclass(actual_type):
                    kwargs[f.name] = _build(actual_type, val)
                else:
                    kwargs[f.name] = val
        return cls(**kwargs)

    return _build(Config, data)


def load_config(path: str) -> Config:
    """Load a Config from a YAML or JSON file.

    Args:
        path: Filesystem path to a ``.yaml``, ``.yml``, or ``.json`` config file.

    Returns:
        A fully populated :class:`Config` instance.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError("PyYAML is required to load YAML configs: pip install pyyaml") from exc
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    elif ext == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    else:
        raise ValueError(f"Unsupported config format '{ext}'. Use .yaml, .yml, or .json.")

    return _dict_to_config(data)
