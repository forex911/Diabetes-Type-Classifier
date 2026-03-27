"""Feature extraction and embedding for the diabetes-classifier pipeline.

This module contains:
- FeatureExtractor: combines structured EHR features with TF-IDF or ClinicalBERT
  text embeddings into a single combined feature vector.
"""
from __future__ import annotations

import json
import os
import pickle
from typing import List, Optional

import numpy as np

from diabetes_identifier.utils.config import EmbeddingConfig
from diabetes_identifier.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TFIDF_MAX_FEATURES = 10_000
BERT_EMBEDDING_DIM = 768
BERT_BATCH_SIZE = 16


# ---------------------------------------------------------------------------
# FeatureExtractor
# ---------------------------------------------------------------------------


class FeatureExtractor:
    """Combine structured EHR features with text embeddings.

    Responsibilities (Requirements 4.1 – 4.5):
    - Produce TF-IDF baseline embeddings (≤ 10,000 dims) from clinical notes
      when ``mode="tfidf"`` (Req 4.1).
    - Produce ClinicalBERT embeddings by mean-pooling the last hidden layer of
      ``emilyalsentzer/Bio_ClinicalBERT`` → 768-dim vector when ``mode="bert"``
      (Req 4.2).
    - Concatenate normalized structured vector + text embedding → combined
      feature vector (Req 4.3, 4.4).
    - Serialize fitted TF-IDF vectorizer and BERT tokenizer config to disk so
      inference uses identical transformations to training (Req 4.5).

    BERT model loading is lazy: the tokenizer and model are only instantiated
    the first time ``transform`` is called in ``"bert"`` mode, so importing
    this module in TF-IDF mode does not trigger slow HuggingFace downloads.

    Args:
        config: Embedding configuration.  Defaults to ``EmbeddingConfig()``.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        self.config = config or EmbeddingConfig()

        # TF-IDF fitted state
        self._tfidf_vectorizer = None
        self._is_fitted: bool = False

        # BERT lazy-loaded state (only populated in bert mode)
        self._bert_tokenizer = None
        self._bert_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, notes: List[str]) -> None:
        """Fit the TF-IDF vectorizer on clinical notes.

        Only meaningful when ``mode="tfidf"``.  In ``"bert"`` mode this is a
        no-op (BERT is a pre-trained model and does not require fitting).

        Args:
            notes: List of preprocessed clinical note strings.
        """
        if self.config.mode == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore

            self._tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
            self._tfidf_vectorizer.fit(notes)
            logger.info(
                "TF-IDF vectorizer fitted",
                extra={"n_notes": len(notes), "max_features": TFIDF_MAX_FEATURES},
            )
        else:
            # BERT mode — no fitting required
            logger.info("fit() called in bert mode — no-op (BERT is pre-trained)")

        self._is_fitted = True

    def transform(self, structured: np.ndarray, notes: List[str]) -> np.ndarray:
        """Produce combined feature vectors from structured data and clinical notes.

        Args:
            structured: Float ndarray of shape ``(n_samples, n_structured_features)``
                containing normalized structured EHR features.
            notes: List of preprocessed clinical note strings, length ``n_samples``.

        Returns:
            Float64 ndarray of shape
            ``(n_samples, n_structured_features + embedding_dim)`` where
            ``embedding_dim`` is ≤ 10,000 for TF-IDF or 768 for BERT.
        """
        self._check_fitted()

        if len(notes) != structured.shape[0]:
            raise ValueError(
                f"Length mismatch: structured has {structured.shape[0]} rows "
                f"but notes has {len(notes)} entries."
            )

        if self.config.mode == "tfidf":
            text_embeddings = self._embed_tfidf(notes)
        elif self.config.mode == "bert":
            text_embeddings = self._embed_bert(notes)
        else:
            raise ValueError(
                f"Unknown embedding mode '{self.config.mode}'. "
                "Expected 'tfidf' or 'bert'."
            )

        # Concatenate structured + text embeddings (Req 4.3)
        combined = np.concatenate([structured, text_embeddings], axis=1)
        logger.info(
            "Feature vectors produced",
            extra={
                "mode": self.config.mode,
                "n_samples": combined.shape[0],
                "combined_dim": combined.shape[1],
            },
        )
        return combined

    def serialize(self, path: str) -> None:
        """Save fitted state to disk.

        Artifacts written (Req 4.5):
        - ``{path}/tfidf_vectorizer.pkl`` — fitted TF-IDF vectorizer (tfidf mode)
        - ``{path}/bert_tokenizer_config.json`` — BERT tokenizer config (bert mode)

        Args:
            path: Directory path under which artifacts are saved (e.g. ``"models"``).
        """
        self._check_fitted()
        os.makedirs(path, exist_ok=True)

        if self.config.mode == "tfidf":
            tfidf_path = os.path.join(path, "tfidf_vectorizer.pkl")
            with open(tfidf_path, "wb") as fh:
                pickle.dump(self._tfidf_vectorizer, fh, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info("TF-IDF vectorizer serialized", extra={"path": tfidf_path})

        elif self.config.mode == "bert":
            self._ensure_bert_loaded()
            config_path = os.path.join(path, "bert_tokenizer_config.json")
            tokenizer_config = self._bert_tokenizer.init_kwargs if hasattr(self._bert_tokenizer, "init_kwargs") else {}
            # Also include the model name so we can reload it
            tokenizer_config["_model_name"] = self.config.bert_model_name
            with open(config_path, "w", encoding="utf-8") as fh:
                json.dump(tokenizer_config, fh, indent=2, default=str)
            logger.info("BERT tokenizer config serialized", extra={"path": config_path})

    def deserialize(self, path: str) -> None:
        """Load fitted state from disk.

        Args:
            path: Directory path containing previously serialized artifacts
                (e.g. ``"models"``).
        """
        if self.config.mode == "tfidf":
            tfidf_path = os.path.join(path, "tfidf_vectorizer.pkl")
            if not os.path.exists(tfidf_path):
                raise FileNotFoundError(
                    f"TF-IDF vectorizer artifact not found: {tfidf_path}"
                )
            with open(tfidf_path, "rb") as fh:
                self._tfidf_vectorizer = pickle.load(fh)
            logger.info("TF-IDF vectorizer deserialized", extra={"path": tfidf_path})

        elif self.config.mode == "bert":
            config_path = os.path.join(path, "bert_tokenizer_config.json")
            if not os.path.exists(config_path):
                raise FileNotFoundError(
                    f"BERT tokenizer config artifact not found: {config_path}"
                )
            with open(config_path, "r", encoding="utf-8") as fh:
                saved_config = json.load(fh)
            model_name = saved_config.get("_model_name", self.config.bert_model_name)
            self.config.bert_model_name = model_name
            logger.info("BERT tokenizer config deserialized", extra={"path": config_path})

        self._is_fitted = True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "FeatureExtractor is not fitted. Call fit() first."
            )

    def _embed_tfidf(self, notes: List[str]) -> np.ndarray:
        """Transform notes to TF-IDF sparse matrix and return as dense float64 array."""
        if self._tfidf_vectorizer is None:
            raise RuntimeError(
                "TF-IDF vectorizer is not fitted. Call fit() first."
            )
        sparse = self._tfidf_vectorizer.transform(notes)
        return sparse.toarray().astype(np.float64)

    def _embed_bert(self, notes: List[str]) -> np.ndarray:
        """Mean-pool last hidden layer of ClinicalBERT for each note.

        Processes notes in batches to handle large inputs efficiently.

        Returns:
            Float64 ndarray of shape ``(n_samples, 768)``.
        """
        self._ensure_bert_loaded()

        import torch  # type: ignore

        all_embeddings: List[np.ndarray] = []

        for batch_start in range(0, len(notes), BERT_BATCH_SIZE):
            batch = notes[batch_start: batch_start + BERT_BATCH_SIZE]

            encoded = self._bert_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            with torch.no_grad():
                outputs = self._bert_model(**encoded)

            # Mean-pool last hidden state over token dimension (Req 4.2)
            last_hidden = outputs.last_hidden_state  # (batch, seq_len, 768)
            attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
            summed = (last_hidden * attention_mask).sum(dim=1)
            counts = attention_mask.sum(dim=1).clamp(min=1e-9)
            mean_pooled = (summed / counts).cpu().numpy().astype(np.float64)

            all_embeddings.append(mean_pooled)

        return np.vstack(all_embeddings)

    def _ensure_bert_loaded(self) -> None:
        """Lazily load the BERT tokenizer and model on first use."""
        if self._bert_tokenizer is not None and self._bert_model is not None:
            return

        from transformers import AutoModel, AutoTokenizer  # type: ignore

        model_name = self.config.bert_model_name
        logger.info("Loading ClinicalBERT tokenizer and model", extra={"model": model_name})

        self._bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._bert_model = AutoModel.from_pretrained(model_name)
        self._bert_model.eval()

        logger.info("ClinicalBERT loaded successfully", extra={"model": model_name})
