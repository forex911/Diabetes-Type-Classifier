"""Model training for the diabetes-classifier pipeline.

This module contains:
- ModelTrainer: trains baseline sklearn models, fine-tunes Bio_ClinicalBERT,
  and builds an ensemble combining both.
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np

from diabetes_identifier.utils.config import TrainingConfig
from diabetes_identifier.utils.logger import ExperimentTracker, get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODELS_DIR = "models"
NUM_LABELS = 4  # Type 1, Type 2, Gestational, Other


# ---------------------------------------------------------------------------
# EnsembleClassifier
# ---------------------------------------------------------------------------


class EnsembleClassifier:
    """Combines probability outputs from a structured and a BERT classifier.

    Args:
        structured_model: Fitted sklearn classifier with ``predict_proba``.
        bert_model: Fine-tuned HuggingFace model (wrapped in BertClassifier).
        weights: List of two floats ``[structured_weight, bert_weight]`` that
            sum to 1.0.  Defaults to ``[0.5, 0.5]``.
    """

    def __init__(
        self,
        structured_model: Any,
        bert_model: Any,
        weights: Optional[List[float]] = None,
    ) -> None:
        self.structured_model = structured_model
        self.bert_model = bert_model
        self.weights = weights or [0.5, 0.5]

    def predict_proba(self, X: np.ndarray, notes: List[str]) -> np.ndarray:
        """Return weighted-average class probabilities.

        Args:
            X: Structured feature matrix ``(n_samples, n_features)``.
            notes: Clinical note strings, length ``n_samples``.

        Returns:
            Float64 ndarray of shape ``(n_samples, 4)``.
        """
        w_struct, w_bert = self.weights
        struct_proba = self.structured_model.predict_proba(X)
        bert_proba = self.bert_model.predict_proba(notes)
        return w_struct * struct_proba + w_bert * bert_proba

    def predict(self, X: np.ndarray, notes: List[str]) -> np.ndarray:
        """Return predicted class indices.

        Args:
            X: Structured feature matrix.
            notes: Clinical note strings.

        Returns:
            Integer ndarray of shape ``(n_samples,)``.
        """
        proba = self.predict_proba(X, notes)
        return np.argmax(proba, axis=1)


# ---------------------------------------------------------------------------
# BertClassifier wrapper
# ---------------------------------------------------------------------------


class BertClassifier:
    """Thin wrapper around a fine-tuned HuggingFace sequence-classification model.

    Provides a ``predict_proba`` interface compatible with the ensemble layer.

    Args:
        model: HuggingFace ``AutoModelForSequenceClassification`` instance.
        tokenizer: Corresponding tokenizer.
        batch_size: Inference batch size.
    """

    def __init__(self, model: Any, tokenizer: Any, batch_size: int = 16) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def predict_proba(self, notes: List[str]) -> np.ndarray:
        """Return softmax class probabilities for each note.

        Args:
            notes: Clinical note strings.

        Returns:
            Float64 ndarray of shape ``(n_samples, 4)``.
        """
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        self.model.eval()
        all_probs: List[np.ndarray] = []

        for start in range(0, len(notes), self.batch_size):
            batch = notes[start: start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            with torch.no_grad():
                outputs = self.model(**encoded)
            probs = F.softmax(outputs.logits, dim=-1).cpu().numpy().astype(np.float64)
            all_probs.append(probs)

        return np.vstack(all_probs)

    def predict(self, notes: List[str]) -> np.ndarray:
        """Return predicted class indices.

        Args:
            notes: Clinical note strings.

        Returns:
            Integer ndarray of shape ``(n_samples,)``.
        """
        return np.argmax(self.predict_proba(notes), axis=1)


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------


class ModelTrainer:
    """Train baseline, BERT, and ensemble models for diabetes classification.

    Responsibilities (Requirements 5.1 – 5.3, 5.6):
    - Train LogisticRegression and DecisionTreeClassifier baselines on combined
      feature vectors (Req 5.1).
    - Fine-tune Bio_ClinicalBERT for 4-class classification via HuggingFace
      Trainer (Req 5.2).
    - Train an Ensemble model combining probability outputs from structured and
      BERT classifiers with configurable weights (Req 5.3).
    - Use a fixed random seed for reproducibility (Req 5.6).

    Serialized artifacts (written to ``models_dir``):
    - ``structured_classifier.pkl`` — fitted LogisticRegression
    - ``bert_classifier/`` — HuggingFace saved model directory
    - ``ensemble_weights.json`` — ensemble weight configuration

    Args:
        training_config: Training hyper-parameters.  Defaults to
            ``TrainingConfig()``.
        models_dir: Directory where serialized artifacts are saved.
            Defaults to ``"models"``.
    """

    def __init__(
        self,
        training_config: Optional[TrainingConfig] = None,
        models_dir: str = MODELS_DIR,
        tracker: Optional[ExperimentTracker] = None,
    ) -> None:
        self.training_config = training_config or TrainingConfig()
        self.models_dir = models_dir
        self.tracker = tracker

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_baseline(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train LogisticRegression and DecisionTreeClassifier baselines.

        Both models are trained on the combined feature matrix ``X`` with
        the fixed ``random_seed`` from ``training_config`` (Req 5.1, 5.6).
        The LogisticRegression model is serialized to
        ``{models_dir}/structured_classifier.pkl``.

        Args:
            X: Combined feature matrix ``(n_samples, n_features)``.
            y: Integer label array ``(n_samples,)`` with values in ``[0, 3]``.

        Returns:
            Dict with keys ``"logistic_regression"`` and ``"decision_tree"``,
            each mapping to the corresponding fitted sklearn estimator.
        """
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.tree import DecisionTreeClassifier  # type: ignore

        seed = self.training_config.random_seed

        logger.info(
            "Training baseline models",
            extra={"n_samples": len(y), "random_seed": seed},
        )

        # Start experiment tracking (Req 5.4, 12.1, 12.2)
        if self.tracker is not None:
            self.tracker.start_run(run_name="baseline")
            self.tracker.log_params({
                "model_type": "logistic_regression",
                "random_seed": seed,
                "n_train_samples": len(y),
            })

        # Logistic Regression (Req 5.1)
        lr = LogisticRegression(
            random_state=seed,
            max_iter=1000,
            solver="lbfgs",
        )
        lr.fit(X, y)
        logger.info("LogisticRegression trained")

        # Decision Tree (Req 5.1)
        dt = DecisionTreeClassifier(random_state=seed)
        dt.fit(X, y)
        logger.info("DecisionTreeClassifier trained")

        # Serialize LogisticRegression (Req 5.1 artifact)
        os.makedirs(self.models_dir, exist_ok=True)
        lr_path = os.path.join(self.models_dir, "structured_classifier.pkl")
        with open(lr_path, "wb") as fh:
            pickle.dump(lr, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("LogisticRegression serialized", extra={"path": lr_path})

        return {
            "logistic_regression": lr,
            "decision_tree": dt,
        }

    def train_bert(self, notes: List[str], y: np.ndarray) -> BertClassifier:
        """Fine-tune Bio_ClinicalBERT for 4-class diabetes classification.

        Uses HuggingFace ``Trainer`` with ``TrainingArguments``.  The trained
        model is saved to ``{models_dir}/bert_classifier/`` (Req 5.2).

        BERT fine-tuning is computationally expensive.  The default config uses
        a minimal number of epochs (1) so the implementation is correct but
        fast for development/testing.

        Args:
            notes: Clinical note strings, length ``n_samples``.
            y: Integer label array ``(n_samples,)`` with values in ``[0, 3]``.

        Returns:
            A :class:`BertClassifier` wrapping the fine-tuned model.
        """
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
        import torch  # type: ignore
        from torch.utils.data import Dataset  # type: ignore

        seed = self.training_config.random_seed
        model_name = "emilyalsentzer/Bio_ClinicalBERT"

        logger.info(
            "Loading Bio_ClinicalBERT for fine-tuning",
            extra={"model": model_name, "n_samples": len(notes), "random_seed": seed},
        )

        # Start experiment tracking (Req 5.4, 12.1, 12.2)
        if self.tracker is not None:
            self.tracker.start_run(run_name="bert")
            self.tracker.log_params({
                "model_type": "bert",
                "random_seed": seed,
                "n_train_samples": len(notes),
                "bert_model_name": model_name,
            })

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=NUM_LABELS,
            ignore_mismatched_sizes=True,
        )

        # Build a simple PyTorch Dataset
        class _NoteDataset(Dataset):
            def __init__(self, texts: List[str], labels: np.ndarray, tok: Any) -> None:
                self.encodings = tok(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                self.labels = torch.tensor(labels, dtype=torch.long)

            def __len__(self) -> int:
                return len(self.labels)

            def __getitem__(self, idx: int) -> Dict[str, Any]:
                item = {k: v[idx] for k, v in self.encodings.items()}
                item["labels"] = self.labels[idx]
                return item

        dataset = _NoteDataset(notes, y, tokenizer)

        bert_output_dir = os.path.join(self.models_dir, "bert_classifier")
        os.makedirs(self.models_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=bert_output_dir,
            num_train_epochs=1,
            per_device_train_batch_size=8,
            seed=seed,
            data_seed=seed,
            save_strategy="no",
            logging_steps=50,
            report_to="none",
            no_cuda=not torch.cuda.is_available(),
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
        )

        logger.info("Starting BERT fine-tuning")
        trainer.train()
        logger.info("BERT fine-tuning complete")

        # Save the fine-tuned model and tokenizer (Req 5.2 artifact)
        model.save_pretrained(bert_output_dir)
        tokenizer.save_pretrained(bert_output_dir)
        logger.info("BERT classifier saved", extra={"path": bert_output_dir})

        return BertClassifier(model=model, tokenizer=tokenizer)

    def train_ensemble(
        self,
        structured_model: Any,
        bert_model: Any,
        X: np.ndarray,
        notes: List[str],
        y: np.ndarray,
    ) -> EnsembleClassifier:
        """Build an ensemble combining structured and BERT classifiers.

        Combines probability outputs using configurable ``ensemble_weights``
        from ``training_config``.  Serializes ``ensemble_weights.json`` to
        ``{models_dir}/`` (Req 5.3).

        Args:
            structured_model: Fitted sklearn classifier with ``predict_proba``.
            bert_model: :class:`BertClassifier` wrapping the fine-tuned model.
            X: Structured feature matrix ``(n_samples, n_features)``.
            notes: Clinical note strings, length ``n_samples``.
            y: Integer label array ``(n_samples,)`` — used for logging only.

        Returns:
            A fitted :class:`EnsembleClassifier`.
        """
        weights = list(self.training_config.ensemble_weights)

        logger.info(
            "Building ensemble model",
            extra={
                "ensemble_weights": weights,
                "n_samples": len(y),
            },
        )

        # Log ensemble params to active tracking run (Req 5.4, 12.2)
        if self.tracker is not None:
            self.tracker.log_params({
                "model_type": "ensemble",
                "ensemble_weights": str(weights),
            })

        ensemble = EnsembleClassifier(
            structured_model=structured_model,
            bert_model=bert_model,
            weights=weights,
        )

        # Serialize ensemble weights (Req 5.3 artifact)
        os.makedirs(self.models_dir, exist_ok=True)
        weights_path = os.path.join(self.models_dir, "ensemble_weights.json")
        with open(weights_path, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "structured_weight": weights[0],
                    "bert_weight": weights[1],
                    "random_seed": self.training_config.random_seed,
                },
                fh,
                indent=2,
            )
        logger.info("Ensemble weights serialized", extra={"path": weights_path})

        return ensemble


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Generate data (if needed) and train the baseline structured classifier."""
    import os
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[2]
    data_path = project_root / "diabetes_identifier" / "data" / "raw" / "ehr_synthetic.csv"
    models_dir = str(project_root / "diabetes_identifier" / "models")

    # Generate synthetic data if not present
    if not data_path.exists():
        print("Generating synthetic data...")
        from diabetes_identifier.utils.generate_mock_data import generate
        os.makedirs(data_path.parent, exist_ok=True)
        generate(n_per_class=130, output_path=str(data_path))

    # Load data
    import pandas as pd
    df = pd.read_csv(data_path)
    X = df[["age", "bmi", "glucose", "insulin"]].values
    y = df["label"].values

    # Train baseline
    trainer = ModelTrainer(models_dir=models_dir)
    result = trainer.train_baseline(X, y)
    print(f"Training complete. Model saved to {models_dir}/structured_classifier.pkl")


if __name__ == "__main__":
    main()
