"""Preprocessing classes for the diabetes-classifier pipeline.

This module contains:
- StructuredPreprocessor: normalizes, clips, encodes, and optionally resamples
  structured EHR features (age, bmi, glucose, insulin).
- NLPPreprocessor: will be added in Task 4.1.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from diabetes_identifier.utils.config import PreprocConfig, TrainingConfig
from diabetes_identifier.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_MAP: Dict[int, str] = {
    0: "Type 1",
    1: "Type 2",
    2: "Gestational",
    3: "Other",
}

NUMERIC_FEATURES = ["age", "bmi", "glucose", "insulin"]

# Inverse mapping used during fit to build the LabelEncoder consistently.
_LABEL_MAP_INV: Dict[str, int] = {v: k for k, v in LABEL_MAP.items()}


# ---------------------------------------------------------------------------
# StructuredPreprocessor
# ---------------------------------------------------------------------------


class StructuredPreprocessor:
    """Preprocess structured EHR features for model training and inference.

    Responsibilities (Requirements 2.1 – 2.5):
    - Normalize numeric fields to zero mean / unit variance using *training*
      statistics only (Req 2.1).
    - Clip values outside ±clip_sigma standard deviations to the boundary
      (Req 2.2).
    - Encode the target label to int ∈ [0, 3] via a fixed LABEL_MAP (Req 2.3).
    - Apply SMOTE when any class represents < smote_threshold of training
      samples (Req 2.4).
    - Return inverse-frequency class weights when class_weights is configured
      (Req 2.5).

    Args:
        preproc_config: Preprocessing hyper-parameters (clip_sigma,
            smote_threshold).  Defaults to ``PreprocConfig()``.
        training_config: Training hyper-parameters (class_weights, random_seed).
            Defaults to ``TrainingConfig()``.
    """

    def __init__(
        self,
        preproc_config: Optional[PreprocConfig] = None,
        training_config: Optional[TrainingConfig] = None,
    ) -> None:
        self.preproc_config = preproc_config or PreprocConfig()
        self.training_config = training_config or TrainingConfig()

        # Fitted state — populated by fit()
        self._means: Optional[pd.Series] = None
        self._stds: Optional[pd.Series] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> None:
        """Compute training-split statistics and fit the label encoder.

        Args:
            df: Training DataFrame.  Must contain the columns in
                ``NUMERIC_FEATURES`` and a ``label`` column whose values are
                strings from ``LABEL_MAP.values()``.
        """
        self._validate_columns(df)

        # Compute mean / std on numeric features (Req 2.1)
        self._means = df[NUMERIC_FEATURES].mean()
        self._stds = df[NUMERIC_FEATURES].std(ddof=1)

        # Replace zero std with 1 to avoid division-by-zero
        self._stds = self._stds.replace(0.0, 1.0)

        # Fit label encoder with fixed ordering from LABEL_MAP (Req 2.3)
        self._label_encoder = LabelEncoder()
        self._label_encoder.fit(list(LABEL_MAP.values()))

        self._is_fitted = True
        logger.info(
            "StructuredPreprocessor fitted",
            extra={"n_samples": len(df), "features": NUMERIC_FEATURES},
        )

    def transform(
        self, df: pd.DataFrame, apply_smote: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply normalization, clipping, and label encoding.

        Args:
            df: DataFrame with ``NUMERIC_FEATURES`` and ``label`` columns.
            apply_smote: When ``True`` and the class distribution triggers the
                SMOTE threshold, SMOTE is applied.  Should only be ``True``
                during training.

        Returns:
            A tuple ``(X, y)`` where:
            - ``X`` is a float64 ndarray of shape ``(n_samples, n_features)``
              with normalized and clipped numeric features.
            - ``y`` is an int ndarray of shape ``(n_samples,)`` with encoded
              labels.
        """
        self._check_fitted()
        self._validate_columns(df)

        sigma = self.preproc_config.clip_sigma

        # Normalize (Req 2.1)
        X = (df[NUMERIC_FEATURES].values.astype(np.float64) - self._means.values) / self._stds.values

        # Clip to ±clip_sigma (Req 2.2)
        X = np.clip(X, -sigma, sigma)

        # Encode labels (Req 2.3)
        y = self._label_encoder.transform(df["label"].values).astype(int)

        # Optionally apply SMOTE (Req 2.4)
        if apply_smote and self._should_apply_smote(y):
            X, y = self._apply_smote(X, y)

        return X, y

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Fit on ``df`` then transform it (with SMOTE if triggered).

        Returns:
            A tuple ``(X, y)`` — see :meth:`transform`.
        """
        self.fit(df)
        return self.transform(df, apply_smote=True)

    def get_class_weights(self, y: np.ndarray) -> Optional[Dict[int, float]]:
        """Return inverse-frequency class weights when configured (Req 2.5).

        Args:
            y: Integer label array (post-encoding).

        Returns:
            A dict mapping class index → weight, or ``None`` when
            ``training_config.class_weights`` is not set.
        """
        if not self.training_config.class_weights:
            return None

        classes, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(classes)
        weights = {int(cls): float(n_samples / (n_classes * cnt)) for cls, cnt in zip(classes, counts)}
        return weights

    def serialize(self, path: str) -> None:
        """Persist the fitted state to disk using pickle.

        Args:
            path: Filesystem path for the output file (e.g. ``models/scaler.pkl``).
        """
        self._check_fitted()
        state = {
            "means": self._means,
            "stds": self._stds,
            "label_encoder": self._label_encoder,
            "preproc_config": self.preproc_config,
            "training_config": self.training_config,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("StructuredPreprocessor serialized", extra={"path": path})

    def deserialize(self, path: str) -> None:
        """Load fitted state from disk.

        Args:
            path: Filesystem path to a previously serialized preprocessor.
        """
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        self._means = state["means"]
        self._stds = state["stds"]
        self._label_encoder = state["label_encoder"]
        self.preproc_config = state["preproc_config"]
        self.training_config = state["training_config"]
        self._is_fitted = True
        logger.info("StructuredPreprocessor deserialized", extra={"path": path})

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _validate_columns(self, df: pd.DataFrame) -> None:
        required = NUMERIC_FEATURES + ["label"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {missing}")

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "StructuredPreprocessor is not fitted. Call fit() or fit_transform() first."
            )

    def _should_apply_smote(self, y: np.ndarray) -> bool:
        """Return True when any class is under-represented (Req 2.4)."""
        _, counts = np.unique(y, return_counts=True)
        fractions = counts / len(y)
        return bool(np.any(fractions < self.preproc_config.smote_threshold))

    def _apply_smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE to balance the training set."""
        try:
            from imblearn.over_sampling import SMOTE  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "imbalanced-learn is required for SMOTE: pip install imbalanced-learn"
            ) from exc

        seed = self.training_config.random_seed
        smote = SMOTE(random_state=seed)
        X_res, y_res = smote.fit_resample(X, y)
        logger.info(
            "SMOTE applied",
            extra={
                "original_size": len(y),
                "resampled_size": len(y_res),
            },
        )
        return X_res, y_res


# ---------------------------------------------------------------------------
# NLP Constants
# ---------------------------------------------------------------------------

PHI_TOKENS = ["[NAME]", "[DOB]", "[ID]", "[PHONE]", "[ADDRESS]", "[EMAIL]"]

# ---------------------------------------------------------------------------
# NLPPreprocessor
# ---------------------------------------------------------------------------

from typing import List, TypedDict

from diabetes_identifier.utils.config import NLPConfig


class EntityMetadata(TypedDict):
    symptoms: List[str]
    medications: List[str]
    lab_values: List[str]


class NLPPreprocessor:
    """Preprocess clinical notes for NLP model input.

    Responsibilities (Requirements 3.1 – 3.5):
    - Lowercase all text (Req 3.1).
    - Replace PHI placeholder tokens with ``<PHI>`` (Req 3.2).
    - Segment into sentences via spaCy sentencizer (Req 3.3).
    - Substitute empty/null notes with a configurable default string and log a
      warning (Req 3.4).
    - Extract NER entities (symptoms, medications, lab values) via spaCy model
      (Req 3.5).

    Args:
        config: NLP configuration.  Defaults to ``NLPConfig()``.
    """

    def __init__(self, config: Optional[NLPConfig] = None) -> None:
        self.config = config or NLPConfig()
        self._nlp = self._build_spacy_model()
        self._nlp_ner = self._build_ner_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, notes: List[str]) -> List[str]:
        """Preprocess a list of clinical notes.

        Steps applied per note:
        1. Substitute empty/null notes with the default text (Req 3.4).
        2. Lowercase (Req 3.1).
        3. Replace PHI tokens with ``<PHI>`` (Req 3.2).
        4. Segment into sentences and rejoin with a single space (Req 3.3).

        Args:
            notes: Raw clinical note strings.

        Returns:
            Processed note strings, one per input note.
        """
        results: List[str] = []
        for note in notes:
            # Req 3.4 — substitute empty/null
            if not note or not note.strip():
                logger.warning(
                    "Empty or null clinical note encountered; substituting default text.",
                    extra={"default": self.config.default_notes_text},
                )
                note = self.config.default_notes_text

            # Req 3.1 — lowercase
            note = note.lower()

            # Req 3.2 — replace PHI tokens (already lowercased, so match lower)
            for token in self.config.phi_tokens:
                note = note.replace(token.lower(), "<phi>")

            # Req 3.3 — sentence segmentation then rejoin
            doc = self._nlp(note)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            note = " ".join(sentences) if sentences else note

            results.append(note)
        return results

    def extract_entities(self, notes: List[str]) -> List[EntityMetadata]:
        """Extract NER entities from clinical notes (Req 3.5).

        Uses the spaCy NER model.  When the model has no NER component (e.g.
        a blank model), all entity lists are returned empty.

        Args:
            notes: Clinical note strings (ideally already preprocessed).

        Returns:
            A list of :class:`EntityMetadata` dicts, one per input note.
        """
        results: List[EntityMetadata] = []
        for note in notes:
            if not note or not note.strip():
                note = self.config.default_notes_text

            doc = self._nlp_ner(note)

            symptoms: List[str] = []
            medications: List[str] = []
            lab_values: List[str] = []

            for ent in doc.ents:
                label = ent.label_.upper()
                text = ent.text.strip()
                if label in ("SYMPTOM", "DISEASE", "CONDITION"):
                    symptoms.append(text)
                elif label in ("DRUG", "MEDICATION", "CHEMICAL"):
                    medications.append(text)
                elif label in ("LAB", "LAB_VALUE", "QUANTITY", "PERCENT"):
                    lab_values.append(text)

            results.append(
                EntityMetadata(
                    symptoms=symptoms,
                    medications=medications,
                    lab_values=lab_values,
                )
            )
        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_spacy_model(self):
        """Build a blank spaCy model with a sentencizer component."""
        import spacy  # type: ignore

        nlp = spacy.blank("en")
        if "sentencizer" not in nlp.pipe_names:
            nlp.add_pipe("sentencizer")
        return nlp

    def _build_ner_model(self):
        """Return a spaCy model with NER.  Falls back to blank model gracefully."""
        try:
            import spacy  # type: ignore
            return spacy.load("en_core_web_sm")
        except OSError:
            # en_core_web_sm not installed — return blank model (no NER)
            import spacy  # type: ignore
            return spacy.blank("en")
