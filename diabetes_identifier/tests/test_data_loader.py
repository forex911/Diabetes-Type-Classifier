"""Unit tests for DataLoader (task 2.1)."""
from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest

from diabetes_identifier.utils.config import IngestionConfig
from diabetes_identifier.utils.data_loader import DataLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(content: str) -> str:
    """Write *content* to a temp CSV file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write(content)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# FileNotFoundError
# ---------------------------------------------------------------------------

def test_load_raises_file_not_found():
    dl = DataLoader()
    with pytest.raises(FileNotFoundError) as exc_info:
        dl.load("/nonexistent/path/data.csv")
    assert "/nonexistent/path/data.csv" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Normal CSV loading
# ---------------------------------------------------------------------------

def test_load_returns_dataframe():
    path = _write_csv("age,bmi,glucose,insulin,label\n30,25.0,90.0,10.0,0\n45,28.5,110.0,15.0,1\n")
    try:
        df = DataLoader().load(path)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
    finally:
        os.unlink(path)


def test_load_preserves_all_columns():
    path = _write_csv("age,bmi,glucose,insulin,label\n30,25.0,90.0,10.0,0\n")
    try:
        df = DataLoader().load(path)
        assert set(["age", "bmi", "glucose", "insulin", "label"]).issubset(df.columns)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Malformed row handling
# ---------------------------------------------------------------------------

def test_malformed_rows_are_skipped():
    path = _write_csv(
        "age,bmi,glucose,insulin,label\n"
        "30,25.0,90.0,10.0,0\n"
        "bad,row\n"          # malformed — wrong column count
        "45,28.5,110.0,15.0,1\n"
    )
    try:
        df = DataLoader().load(path)
        assert len(df) == 2
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# Missing value strategies
# ---------------------------------------------------------------------------

def test_mean_strategy_fills_missing():
    path = _write_csv(
        "age,bmi,glucose,insulin,label\n"
        "30,25.0,,10.0,0\n"
        "50,30.0,120.0,20.0,1\n"
    )
    try:
        df = DataLoader("mean").load(path)
        assert not df["glucose"].isna().any()
        assert df["glucose"].iloc[0] == pytest.approx(120.0)
    finally:
        os.unlink(path)


def test_median_strategy_fills_missing():
    path = _write_csv(
        "age,bmi,glucose,insulin,label\n"
        "30,25.0,,10.0,0\n"
        "50,30.0,100.0,20.0,1\n"
        "60,32.0,200.0,30.0,2\n"
    )
    try:
        df = DataLoader("median").load(path)
        assert not df["glucose"].isna().any()
        # median of [100, 200] = 150
        assert df["glucose"].iloc[0] == pytest.approx(150.0)
    finally:
        os.unlink(path)


def test_drop_strategy_removes_rows_with_missing():
    path = _write_csv(
        "age,bmi,glucose,insulin,label\n"
        "30,25.0,,10.0,0\n"
        "50,30.0,120.0,20.0,1\n"
    )
    try:
        df = DataLoader("drop").load(path)
        assert len(df) == 1
        assert df["glucose"].iloc[0] == pytest.approx(120.0)
    finally:
        os.unlink(path)


# ---------------------------------------------------------------------------
# IngestionConfig accepted
# ---------------------------------------------------------------------------

def test_accepts_ingestion_config():
    cfg = IngestionConfig(missing_strategy="median")
    dl = DataLoader(cfg)
    assert dl._strategy == "median"


def test_default_strategy_is_mean():
    dl = DataLoader()
    assert dl._strategy == "mean"
