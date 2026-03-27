"""Property-based tests for StructuredPreprocessor serialization round-trip (task 3.2).

**Validates: Requirements 9.1, 9.2**
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from diabetes_identifier.nlp.preprocessing import LABEL_MAP, StructuredPreprocessor

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

_label_values = list(LABEL_MAP.values())  # ["Type 1", "Type 2", "Gestational", "Other"]

_label_strategy = st.sampled_from(_label_values)

_age_strategy = st.floats(min_value=0.0, max_value=120.0, allow_nan=False, allow_infinity=False)
_bmi_strategy = st.floats(min_value=10.0, max_value=80.0, allow_nan=False, allow_infinity=False)
_glucose_strategy = st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False)
_insulin_strategy = st.floats(min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False)


def _row_strategy():
    """Generate a single valid EHR row as a dict with string labels."""
    return st.fixed_dictionaries(
        {
            "age": _age_strategy,
            "bmi": _bmi_strategy,
            "glucose": _glucose_strategy,
            "insulin": _insulin_strategy,
            "label": _label_strategy,
        }
    )


def _dataframe_strategy():
    """Generate a non-empty DataFrame with at least one row per label class
    to ensure the LabelEncoder sees all classes during fit."""
    # Build one guaranteed row per class, then add random rows
    fixed_rows = st.just(
        [
            {"age": 30.0, "bmi": 25.0, "glucose": 100.0, "insulin": 10.0, "label": lbl}
            for lbl in _label_values
        ]
    )
    extra_rows = st.lists(_row_strategy(), min_size=0, max_size=20)
    return st.builds(
        lambda fixed, extra: pd.DataFrame(fixed + extra),
        fixed=fixed_rows,
        extra=extra_rows,
    )


# ---------------------------------------------------------------------------
# Property 2: Serialization round-trip
# ---------------------------------------------------------------------------


@given(df=_dataframe_strategy())
@settings(max_examples=50)
def test_serialization_roundtrip(df: pd.DataFrame):
    """For any valid structured feature record, serialize then deserialize then
    transform produces byte-identical output to transform without serialization.

    **Validates: Requirements 9.1, 9.2**
    """
    # 1. Fit a StructuredPreprocessor on the data
    preprocessor = StructuredPreprocessor()
    preprocessor.fit(df)

    # 2. Transform the data to get original outputs
    X_original, y_original = preprocessor.transform(df)

    # 3. Serialize to a temp file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        preprocessor.serialize(tmp_path)

        # 4. Create a new StructuredPreprocessor and deserialize
        preprocessor_rt = StructuredPreprocessor()
        preprocessor_rt.deserialize(tmp_path)

        # 5. Transform the same data with the deserialized preprocessor
        X_roundtrip, y_roundtrip = preprocessor_rt.transform(df)

        # 6. Assert feature arrays are identical
        assert np.array_equal(X_original, X_roundtrip), (
            f"X mismatch after round-trip serialization.\n"
            f"X_original:\n{X_original}\nX_roundtrip:\n{X_roundtrip}"
        )

        # 7. Assert label arrays are identical
        assert np.array_equal(y_original, y_roundtrip), (
            f"y mismatch after round-trip serialization.\n"
            f"y_original: {y_original}\ny_roundtrip: {y_roundtrip}"
        )
    finally:
        os.unlink(tmp_path)
