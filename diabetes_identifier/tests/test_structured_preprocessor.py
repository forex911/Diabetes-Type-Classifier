"""Unit tests for StructuredPreprocessor (task 3.3).

Covers:
- NaN-free DataFrame handling (Req 2.1)
- ±3σ clipping boundary values (Req 2.2)
- Label encoding consistency to [0, 3] (Req 2.3)
- SMOTE activation when a class is < 15% (Req 2.4)
- Validates: Requirements 2.1, 2.2, 2.3, 2.4, 11.1
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from diabetes_identifier.nlp.preprocessing import (
    LABEL_MAP,
    NUMERIC_FEATURES,
    StructuredPreprocessor,
)
from diabetes_identifier.utils.config import PreprocConfig, TrainingConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_LABELS = list(LABEL_MAP.values())  # ["Type 1", "Type 2", "Gestational", "Other"]


def _make_df(
    n_per_class: int = 50,
    age_mean: float = 40.0,
    bmi_mean: float = 25.0,
    glucose_mean: float = 100.0,
    insulin_mean: float = 15.0,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build a balanced, NaN-free DataFrame with all four label classes."""
    if rng is None:
        rng = np.random.default_rng(42)

    n = n_per_class * 4
    rows = {
        "age": rng.normal(age_mean, 10.0, n),
        "bmi": rng.normal(bmi_mean, 5.0, n),
        "glucose": rng.normal(glucose_mean, 20.0, n),
        "insulin": rng.normal(insulin_mean, 5.0, n),
        "label": ALL_LABELS * n_per_class,
    }
    return pd.DataFrame(rows)


def _make_imbalanced_df(
    majority_n: int = 200,
    minority_n: int = 5,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Build a DataFrame where one class is severely under-represented (< 15%)."""
    if rng is None:
        rng = np.random.default_rng(42)

    # Three majority classes + one tiny minority class
    labels = (
        ["Type 1"] * majority_n
        + ["Type 2"] * majority_n
        + ["Gestational"] * majority_n
        + ["Other"] * minority_n
    )
    n = len(labels)
    rows = {
        "age": rng.normal(40.0, 10.0, n),
        "bmi": rng.normal(25.0, 5.0, n),
        "glucose": rng.normal(100.0, 20.0, n),
        "insulin": rng.normal(15.0, 5.0, n),
        "label": labels,
    }
    return pd.DataFrame(rows)


def _fitted_preprocessor(df: pd.DataFrame | None = None) -> StructuredPreprocessor:
    """Return a StructuredPreprocessor already fitted on *df* (or a default balanced df)."""
    if df is None:
        df = _make_df()
    sp = StructuredPreprocessor()
    sp.fit(df)
    return sp


# ---------------------------------------------------------------------------
# 1. NaN-free DataFrame handling (Req 2.1)
# ---------------------------------------------------------------------------

class TestNaNFreeHandling:
    """StructuredPreprocessor should work correctly on DataFrames without NaNs."""

    def test_fit_does_not_raise_on_clean_df(self):
        df = _make_df()
        sp = StructuredPreprocessor()
        sp.fit(df)  # must not raise
        assert sp._is_fitted

    def test_transform_returns_correct_shape(self):
        df = _make_df(n_per_class=25)
        sp = _fitted_preprocessor(df)
        X, y = sp.transform(df)
        assert X.shape == (len(df), len(NUMERIC_FEATURES))
        assert y.shape == (len(df),)

    def test_transform_output_has_no_nans(self):
        df = _make_df()
        sp = _fitted_preprocessor(df)
        X, y = sp.transform(df)
        assert not np.isnan(X).any()
        assert not np.isnan(y.astype(float)).any()

    def test_fit_transform_returns_correct_shape(self):
        df = _make_df(n_per_class=30)
        sp = StructuredPreprocessor()
        X, y = sp.fit_transform(df)
        assert X.shape[1] == len(NUMERIC_FEATURES)
        assert len(y) == X.shape[0]

    def test_transform_before_fit_raises(self):
        df = _make_df()
        sp = StructuredPreprocessor()
        with pytest.raises(RuntimeError, match="not fitted"):
            sp.transform(df)

    def test_missing_column_raises_value_error(self):
        df = _make_df().drop(columns=["glucose"])
        sp = StructuredPreprocessor()
        with pytest.raises(ValueError, match="glucose"):
            sp.fit(df)

    def test_normalized_output_near_zero_mean(self):
        """After normalization the training data should have ~0 mean per feature."""
        df = _make_df(n_per_class=100)
        sp = _fitted_preprocessor(df)
        X, _ = sp.transform(df)
        # Mean of normalized training data should be close to 0
        assert np.abs(X.mean(axis=0)).max() < 0.1

    def test_normalized_output_near_unit_std(self):
        """After normalization the training data should have ~1 std per feature."""
        df = _make_df(n_per_class=100)
        sp = _fitted_preprocessor(df)
        X, _ = sp.transform(df)
        # Std of normalized training data should be close to 1 (before clipping)
        # Use a relaxed tolerance because clipping can reduce std slightly
        assert np.abs(X.std(axis=0) - 1.0).max() < 0.5


# ---------------------------------------------------------------------------
# 2. ±3σ clipping boundary values (Req 2.2)
# ---------------------------------------------------------------------------

class TestClipping:
    """Values at exactly ±3σ must be preserved; values beyond must be clipped."""

    def _make_boundary_df(self, sigma_multiplier: float) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (train_df, test_df) where test_df has values at sigma_multiplier * std."""
        train_df = _make_df(n_per_class=50)
        sp = StructuredPreprocessor()
        sp.fit(train_df)

        # Build a single-row test record with values exactly at sigma_multiplier * std
        test_values = {
            feat: sp._means[feat] + sigma_multiplier * sp._stds[feat]
            for feat in NUMERIC_FEATURES
        }
        test_values["label"] = "Type 1"
        test_df = pd.DataFrame([test_values])
        return train_df, test_df, sp

    def test_values_at_positive_3sigma_are_preserved(self):
        train_df, test_df, sp = self._make_boundary_df(3.0)
        X, _ = sp.transform(test_df)
        # Each feature should be exactly 3.0 after normalization (at the boundary)
        np.testing.assert_allclose(X[0], [3.0] * len(NUMERIC_FEATURES), atol=1e-9)

    def test_values_at_negative_3sigma_are_preserved(self):
        train_df, test_df, sp = self._make_boundary_df(-3.0)
        X, _ = sp.transform(test_df)
        np.testing.assert_allclose(X[0], [-3.0] * len(NUMERIC_FEATURES), atol=1e-9)

    def test_values_beyond_positive_3sigma_are_clipped(self):
        train_df, test_df, sp = self._make_boundary_df(5.0)
        X, _ = sp.transform(test_df)
        # All values should be clipped to 3.0
        np.testing.assert_allclose(X[0], [3.0] * len(NUMERIC_FEATURES), atol=1e-9)

    def test_values_beyond_negative_3sigma_are_clipped(self):
        train_df, test_df, sp = self._make_boundary_df(-5.0)
        X, _ = sp.transform(test_df)
        np.testing.assert_allclose(X[0], [-3.0] * len(NUMERIC_FEATURES), atol=1e-9)

    def test_values_within_3sigma_are_unchanged(self):
        train_df, test_df, sp = self._make_boundary_df(1.5)
        X, _ = sp.transform(test_df)
        # 1.5σ is within bounds — should not be clipped
        np.testing.assert_allclose(X[0], [1.5] * len(NUMERIC_FEATURES), atol=1e-9)

    def test_custom_clip_sigma_is_respected(self):
        """A clip_sigma of 2.0 should clip values at 3σ down to 2.0."""
        train_df = _make_df(n_per_class=50)
        sp = StructuredPreprocessor(preproc_config=PreprocConfig(clip_sigma=2.0))
        sp.fit(train_df)

        # Build test row at exactly 3σ above mean
        test_values = {
            feat: sp._means[feat] + 3.0 * sp._stds[feat]
            for feat in NUMERIC_FEATURES
        }
        test_values["label"] = "Type 2"
        test_df = pd.DataFrame([test_values])

        X, _ = sp.transform(test_df)
        # Should be clipped to 2.0 (the custom sigma)
        np.testing.assert_allclose(X[0], [2.0] * len(NUMERIC_FEATURES), atol=1e-9)

    def test_output_never_exceeds_clip_sigma(self):
        """No value in the output should exceed clip_sigma in absolute value."""
        df = _make_df(n_per_class=100)
        sp = _fitted_preprocessor(df)
        X, _ = sp.transform(df)
        assert np.all(np.abs(X) <= sp.preproc_config.clip_sigma + 1e-9)


# ---------------------------------------------------------------------------
# 3. Label encoding consistency (Req 2.3)
# ---------------------------------------------------------------------------

class TestLabelEncoding:
    """Labels must be consistently encoded to integers in [0, 3]."""

    def test_all_labels_map_to_0_through_3(self):
        df = _make_df()
        sp = _fitted_preprocessor(df)
        _, y = sp.transform(df)
        unique_labels = set(y.tolist())
        assert unique_labels == {0, 1, 2, 3}

    def test_label_encoding_is_deterministic(self):
        """Same input must always produce the same encoded output."""
        df = _make_df(rng=np.random.default_rng(0))
        sp1 = _fitted_preprocessor(df)
        sp2 = _fitted_preprocessor(df)
        _, y1 = sp1.transform(df)
        _, y2 = sp2.transform(df)
        np.testing.assert_array_equal(y1, y2)

    def test_label_encoding_consistent_across_fits(self):
        """Two preprocessors fitted on different data must use the same label mapping."""
        df1 = _make_df(rng=np.random.default_rng(1))
        df2 = _make_df(rng=np.random.default_rng(2))
        sp1 = _fitted_preprocessor(df1)
        sp2 = _fitted_preprocessor(df2)

        # Build a single-row df with a known label
        single_row = pd.DataFrame([{
            "age": 40.0, "bmi": 25.0, "glucose": 100.0, "insulin": 15.0,
            "label": "Type 1",
        }])
        _, y1 = sp1.transform(single_row)
        _, y2 = sp2.transform(single_row)
        assert y1[0] == y2[0], "Label encoding must be consistent regardless of training data"

    def test_each_label_maps_to_consistent_integer(self):
        """Each label string must always map to the same integer across preprocessors."""
        df = _make_df()
        sp1 = _fitted_preprocessor(df)
        sp2 = _fitted_preprocessor(_make_df(rng=np.random.default_rng(99)))

        for label_str in ALL_LABELS:
            row = pd.DataFrame([{
                "age": 40.0, "bmi": 25.0, "glucose": 100.0, "insulin": 15.0,
                "label": label_str,
            }])
            _, y1 = sp1.transform(row)
            _, y2 = sp2.transform(row)
            assert y1[0] == y2[0], (
                f"Label '{label_str}' mapped to different integers: {y1[0]} vs {y2[0]}"
            )

    def test_encoded_labels_are_integers(self):
        df = _make_df()
        sp = _fitted_preprocessor(df)
        _, y = sp.transform(df)
        assert y.dtype in (np.int32, np.int64, int)

    def test_encoded_labels_within_range(self):
        df = _make_df(n_per_class=100)
        sp = _fitted_preprocessor(df)
        _, y = sp.transform(df)
        assert y.min() >= 0
        assert y.max() <= 3


# ---------------------------------------------------------------------------
# 4. SMOTE activation (Req 2.4)
# ---------------------------------------------------------------------------

class TestSMOTEActivation:
    """SMOTE must be triggered when any class is < 15% of training samples."""

    def test_smote_not_triggered_on_balanced_data(self):
        """Balanced data (25% each) should not trigger SMOTE."""
        df = _make_df(n_per_class=50)
        sp = _fitted_preprocessor(df)
        _, y = sp.transform(df, apply_smote=False)
        # Verify the internal check returns False for balanced data
        assert not sp._should_apply_smote(y)

    def test_smote_triggered_on_imbalanced_data(self):
        """A class with < 15% of samples should trigger SMOTE."""
        df = _make_imbalanced_df(majority_n=200, minority_n=5)
        sp = _fitted_preprocessor(df)
        _, y = sp.transform(df, apply_smote=False)
        # Minority class is ~0.8% — well below 15%
        assert sp._should_apply_smote(y)

    def test_smote_balances_class_distribution(self):
        """After SMOTE, no class should be < 15% of the resampled set."""
        # minority_n >= 6 so SMOTE's default k_neighbors=5 has enough neighbours
        df = _make_imbalanced_df(majority_n=200, minority_n=10)
        sp = _fitted_preprocessor(df)
        X, y = sp.transform(df, apply_smote=True)

        _, counts = np.unique(y, return_counts=True)
        fractions = counts / len(y)
        assert np.all(fractions >= 0.15), (
            f"After SMOTE, all classes should be >= 15%. Got fractions: {fractions}"
        )

    def test_smote_increases_sample_count(self):
        """SMOTE should produce more samples than the original imbalanced set."""
        # minority_n >= 6 so SMOTE's default k_neighbors=5 has enough neighbours
        df = _make_imbalanced_df(majority_n=200, minority_n=10)
        sp = _fitted_preprocessor(df)
        X_no_smote, _ = sp.transform(df, apply_smote=False)
        X_smote, _ = sp.transform(df, apply_smote=True)
        assert len(X_smote) > len(X_no_smote)

    def test_smote_not_applied_when_apply_smote_false(self):
        """Even with imbalanced data, SMOTE must not run when apply_smote=False."""
        df = _make_imbalanced_df(majority_n=200, minority_n=5)
        sp = _fitted_preprocessor(df)
        X, y = sp.transform(df, apply_smote=False)
        # Sample count must equal original
        assert len(X) == len(df)

    def test_smote_threshold_boundary_exactly_15_percent(self):
        """A class at exactly 15% should NOT trigger SMOTE (strict < threshold)."""
        # 3 classes × 85 samples + 1 class × 45 samples = 300 total
        # minority fraction = 45/300 = 15.0% — exactly at threshold, should NOT trigger
        labels = ["Type 1"] * 85 + ["Type 2"] * 85 + ["Gestational"] * 85 + ["Other"] * 45
        rng = np.random.default_rng(42)
        n = len(labels)
        df = pd.DataFrame({
            "age": rng.normal(40.0, 10.0, n),
            "bmi": rng.normal(25.0, 5.0, n),
            "glucose": rng.normal(100.0, 20.0, n),
            "insulin": rng.normal(15.0, 5.0, n),
            "label": labels,
        })
        sp = _fitted_preprocessor(df)
        _, y = sp.transform(df, apply_smote=False)
        # 15% exactly — should NOT trigger (condition is strictly <)
        assert not sp._should_apply_smote(y)

    def test_smote_threshold_just_below_15_percent(self):
        """A class just below 15% should trigger SMOTE."""
        # 3 classes × 86 samples + 1 class × 42 samples = 300 total
        # minority fraction = 42/300 = 14.0% — just below threshold
        labels = ["Type 1"] * 86 + ["Type 2"] * 86 + ["Gestational"] * 86 + ["Other"] * 42
        rng = np.random.default_rng(42)
        n = len(labels)
        df = pd.DataFrame({
            "age": rng.normal(40.0, 10.0, n),
            "bmi": rng.normal(25.0, 5.0, n),
            "glucose": rng.normal(100.0, 20.0, n),
            "insulin": rng.normal(15.0, 5.0, n),
            "label": labels,
        })
        sp = _fitted_preprocessor(df)
        _, y = sp.transform(df, apply_smote=False)
        assert sp._should_apply_smote(y)

    def test_smote_is_deterministic_with_fixed_seed(self):
        """Two SMOTE runs with the same seed must produce identical results."""
        # minority_n >= 6 so SMOTE's default k_neighbors=5 has enough neighbours
        df = _make_imbalanced_df(majority_n=200, minority_n=10)
        training_cfg = TrainingConfig(random_seed=42)

        sp1 = StructuredPreprocessor(training_config=training_cfg)
        sp1.fit(df)
        X1, y1 = sp1.transform(df, apply_smote=True)

        sp2 = StructuredPreprocessor(training_config=training_cfg)
        sp2.fit(df)
        X2, y2 = sp2.transform(df, apply_smote=True)

        np.testing.assert_array_equal(y1, y2)
        np.testing.assert_allclose(X1, X2)
