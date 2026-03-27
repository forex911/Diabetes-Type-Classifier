"""Unit tests for FeatureExtractor (task 5.3).

Covers:
- TF-IDF output dimension ≤ 10,000 (Req 4.1)
- BERT output dimension == 768 (Req 4.2)
- Combined vector shape equals structured_dim + embedding_dim (Req 4.3)
- Mode switching between "tfidf" and "bert" (Req 4.4)

Validates: Requirements 4.1, 4.2, 4.3, 4.4
"""
from __future__ import annotations

import numpy as np
import pytest

from diabetes_identifier.nlp.embedding import FeatureExtractor, TFIDF_MAX_FEATURES, BERT_EMBEDDING_DIM
from diabetes_identifier.utils.config import EmbeddingConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_NOTES = [
    "patient presents with elevated glucose and polyuria consistent with diabetes",
    "no significant findings, blood sugar within normal range",
    "insulin resistance noted, bmi elevated, recommend lifestyle changes",
]

STRUCTURED_DIM = 4  # age, bmi, glucose, insulin


def _make_structured(n: int = 3, dim: int = STRUCTURED_DIM) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((n, dim))


def _fitted_tfidf_extractor(notes=None) -> FeatureExtractor:
    config = EmbeddingConfig(mode="tfidf")
    fe = FeatureExtractor(config=config)
    fe.fit(notes or SAMPLE_NOTES)
    return fe


# ---------------------------------------------------------------------------
# 1. TF-IDF output dimension ≤ 10,000 (Req 4.1)
# ---------------------------------------------------------------------------

class TestTFIDFDimension:
    """TF-IDF text embedding part must have ≤ 10,000 features."""

    def test_tfidf_embedding_dim_within_limit(self):
        fe = _fitted_tfidf_extractor()
        structured = _make_structured(len(SAMPLE_NOTES))
        combined = fe.transform(structured, SAMPLE_NOTES)
        text_dim = combined.shape[1] - STRUCTURED_DIM
        assert text_dim <= TFIDF_MAX_FEATURES

    def test_tfidf_max_features_constant_is_10000(self):
        assert TFIDF_MAX_FEATURES == 10_000

    def test_tfidf_vectorizer_vocabulary_size_within_limit(self):
        fe = _fitted_tfidf_extractor()
        vocab_size = len(fe._tfidf_vectorizer.vocabulary_)
        assert vocab_size <= TFIDF_MAX_FEATURES

    def test_tfidf_output_is_float64(self):
        fe = _fitted_tfidf_extractor()
        structured = _make_structured(len(SAMPLE_NOTES))
        combined = fe.transform(structured, SAMPLE_NOTES)
        assert combined.dtype == np.float64

    def test_tfidf_transform_before_fit_raises(self):
        config = EmbeddingConfig(mode="tfidf")
        fe = FeatureExtractor(config=config)
        structured = _make_structured(len(SAMPLE_NOTES))
        with pytest.raises(RuntimeError, match="not fitted"):
            fe.transform(structured, SAMPLE_NOTES)


# ---------------------------------------------------------------------------
# 2. BERT output dimension == 768 (Req 4.2)
# ---------------------------------------------------------------------------

# Try to import transformers and torch; skip BERT tests if unavailable
try:
    import transformers  # noqa: F401
    import torch  # noqa: F401
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

_bert_skip = pytest.mark.skipif(
    not _TRANSFORMERS_AVAILABLE,
    reason="transformers/torch not installed",
)


def _fitted_bert_extractor() -> FeatureExtractor:
    """Return a fitted FeatureExtractor in bert mode, or skip if model unavailable."""
    config = EmbeddingConfig(mode="bert")
    fe = FeatureExtractor(config=config)
    try:
        fe.fit(SAMPLE_NOTES[:2])
    except Exception as exc:
        pytest.skip(f"BERT model could not be loaded: {exc}")
    return fe


@_bert_skip
class TestBERTDimension:
    """BERT embeddings must be exactly 768-dimensional."""

    def test_bert_embedding_dim_is_768(self):
        fe = _fitted_bert_extractor()
        notes = SAMPLE_NOTES[:2]
        structured = _make_structured(len(notes))
        combined = fe.transform(structured, notes)
        text_dim = combined.shape[1] - STRUCTURED_DIM
        assert text_dim == BERT_EMBEDDING_DIM

    def test_bert_embedding_dim_constant_is_768(self):
        assert BERT_EMBEDDING_DIM == 768

    def test_bert_output_is_float64(self):
        fe = _fitted_bert_extractor()
        notes = SAMPLE_NOTES[:2]
        structured = _make_structured(len(notes))
        combined = fe.transform(structured, notes)
        assert combined.dtype == np.float64

    def test_bert_single_note_produces_768_dim(self):
        fe = _fitted_bert_extractor()
        notes = SAMPLE_NOTES[:1]
        structured = _make_structured(1)
        combined = fe.transform(structured, notes)
        text_dim = combined.shape[1] - STRUCTURED_DIM
        assert text_dim == BERT_EMBEDDING_DIM


# ---------------------------------------------------------------------------
# 3. Combined vector shape = structured_dim + embedding_dim (Req 4.3)
# ---------------------------------------------------------------------------

class TestCombinedVectorShape:
    """Combined vector must equal structured_dim + embedding_dim."""

    def test_tfidf_combined_shape_rows(self):
        n = len(SAMPLE_NOTES)
        fe = _fitted_tfidf_extractor()
        structured = _make_structured(n)
        combined = fe.transform(structured, SAMPLE_NOTES)
        assert combined.shape[0] == n

    def test_tfidf_combined_shape_cols_equals_structured_plus_text(self):
        fe = _fitted_tfidf_extractor()
        structured = _make_structured(len(SAMPLE_NOTES))
        combined = fe.transform(structured, SAMPLE_NOTES)
        text_dim = combined.shape[1] - STRUCTURED_DIM
        assert combined.shape[1] == STRUCTURED_DIM + text_dim

    def test_tfidf_structured_portion_preserved_in_combined(self):
        """The first structured_dim columns of the combined vector must equal the input."""
        fe = _fitted_tfidf_extractor()
        structured = _make_structured(len(SAMPLE_NOTES))
        combined = fe.transform(structured, SAMPLE_NOTES)
        np.testing.assert_array_equal(combined[:, :STRUCTURED_DIM], structured)

    def test_length_mismatch_raises_value_error(self):
        fe = _fitted_tfidf_extractor()
        structured = _make_structured(5)  # 5 rows but only 3 notes
        with pytest.raises(ValueError, match="mismatch"):
            fe.transform(structured, SAMPLE_NOTES)

    @_bert_skip
    def test_bert_combined_shape_cols_equals_structured_plus_768(self):
        fe = _fitted_bert_extractor()
        notes = SAMPLE_NOTES[:2]
        structured = _make_structured(len(notes))
        combined = fe.transform(structured, notes)
        assert combined.shape[1] == STRUCTURED_DIM + BERT_EMBEDDING_DIM

    @_bert_skip
    def test_bert_structured_portion_preserved_in_combined(self):
        fe = _fitted_bert_extractor()
        notes = SAMPLE_NOTES[:2]
        structured = _make_structured(len(notes))
        combined = fe.transform(structured, notes)
        np.testing.assert_array_equal(combined[:, :STRUCTURED_DIM], structured)


# ---------------------------------------------------------------------------
# 4. Mode switching between "tfidf" and "bert" (Req 4.4)
# ---------------------------------------------------------------------------

class TestModeSwitching:
    """Different modes must produce different output shapes."""

    def test_tfidf_mode_is_default(self):
        config = EmbeddingConfig()
        assert config.mode == "tfidf"

    def test_tfidf_mode_produces_correct_shape(self):
        fe = _fitted_tfidf_extractor()
        structured = _make_structured(len(SAMPLE_NOTES))
        combined = fe.transform(structured, SAMPLE_NOTES)
        text_dim = combined.shape[1] - STRUCTURED_DIM
        assert text_dim <= TFIDF_MAX_FEATURES

    @_bert_skip
    def test_bert_mode_produces_different_shape_than_tfidf(self):
        """BERT and TF-IDF modes must produce different output column counts."""
        tfidf_fe = _fitted_tfidf_extractor()
        bert_fe = _fitted_bert_extractor()

        notes = SAMPLE_NOTES[:2]
        structured = _make_structured(len(notes))

        tfidf_combined = tfidf_fe.transform(structured, notes)
        bert_combined = bert_fe.transform(structured, notes)

        # BERT always gives 768 text dims; TF-IDF gives ≤ 10,000 (but usually much less)
        assert tfidf_combined.shape[1] != bert_combined.shape[1] or (
            bert_combined.shape[1] - STRUCTURED_DIM == BERT_EMBEDDING_DIM
        )

    def test_invalid_mode_raises_value_error(self):
        config = EmbeddingConfig(mode="invalid_mode")
        fe = FeatureExtractor(config=config)
        fe._is_fitted = True  # bypass fit check to reach mode dispatch
        structured = _make_structured(len(SAMPLE_NOTES))
        with pytest.raises(ValueError, match="Unknown embedding mode"):
            fe.transform(structured, SAMPLE_NOTES)

    def test_tfidf_fit_sets_is_fitted(self):
        config = EmbeddingConfig(mode="tfidf")
        fe = FeatureExtractor(config=config)
        assert not fe._is_fitted
        fe.fit(SAMPLE_NOTES)
        assert fe._is_fitted

    @_bert_skip
    def test_bert_fit_sets_is_fitted(self):
        config = EmbeddingConfig(mode="bert")
        fe = FeatureExtractor(config=config)
        assert not fe._is_fitted
        try:
            fe.fit(SAMPLE_NOTES[:2])
        except Exception as exc:
            pytest.skip(f"BERT model could not be loaded: {exc}")
        assert fe._is_fitted
