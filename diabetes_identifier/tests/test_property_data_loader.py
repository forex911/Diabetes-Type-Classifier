"""Property-based tests for DataLoader.load (task 2.2).

**Validates: Requirements 1.1**
"""
from __future__ import annotations

import os
import tempfile

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from diabetes_identifier.utils.data_loader import DataLoader

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Valid label values matching the four diabetes classes
_label_strategy = st.integers(min_value=0, max_value=3)

# Realistic but arbitrary numeric ranges for each structured field
_age_strategy = st.floats(min_value=0.0, max_value=120.0, allow_nan=False, allow_infinity=False)
_bmi_strategy = st.floats(min_value=10.0, max_value=80.0, allow_nan=False, allow_infinity=False)
_glucose_strategy = st.floats(min_value=50.0, max_value=500.0, allow_nan=False, allow_infinity=False)
_insulin_strategy = st.floats(min_value=0.0, max_value=300.0, allow_nan=False, allow_infinity=False)


def _row_strategy():
    """Generate a single valid row as a dict."""
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
    """Generate a non-empty DataFrame with valid rows (no malformed rows)."""
    return st.lists(_row_strategy(), min_size=1, max_size=50).map(pd.DataFrame)


# ---------------------------------------------------------------------------
# Property 1: No data loss on valid CSV
# ---------------------------------------------------------------------------

@given(df=_dataframe_strategy())
@settings(max_examples=50)
def test_no_data_loss_on_valid_csv(df: pd.DataFrame):
    """For any DataFrame with no malformed rows, load(path) returns a DataFrame
    with the same number of rows.

    **Validates: Requirements 1.1**
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        path = f.name
        df.to_csv(f, index=False)

    try:
        result = DataLoader().load(path)
        assert len(result) == len(df), (
            f"Expected {len(df)} rows but got {len(result)} rows. "
            f"Input df:\n{df}"
        )
    finally:
        os.unlink(path)
