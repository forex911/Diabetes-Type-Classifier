"""DataLoader: reads CSV files into Pandas DataFrames with configurable missing-value handling."""
from __future__ import annotations

import io
import os
from typing import Optional, Union

import pandas as pd

from diabetes_identifier.utils.config import IngestionConfig
from diabetes_identifier.utils.logger import get_logger

logger = get_logger(__name__)

STRUCTURED_FIELDS = ["age", "bmi", "glucose", "insulin"]


class DataLoader:
    """Load a CSV file into a :class:`pandas.DataFrame`.

    Args:
        config: An :class:`IngestionConfig` instance **or** a missing-strategy
                string (``"mean"``, ``"median"``, or ``"drop"``).  When omitted
                the default strategy is ``"mean"``.
    """

    def __init__(
        self,
        config: Optional[Union[IngestionConfig, str]] = None,
    ) -> None:
        if config is None:
            self._strategy = "mean"
        elif isinstance(config, str):
            self._strategy = config
        else:
            self._strategy = config.missing_strategy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, path: str) -> pd.DataFrame:
        """Read *path* into a DataFrame, applying missing-value strategy.

        Args:
            path: Filesystem path to a CSV file.

        Returns:
            A :class:`pandas.DataFrame` with imputed/filtered structured fields.

        Raises:
            FileNotFoundError: When *path* does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")

        df = self._read_csv_skip_malformed(path)
        df = self._handle_missing(df)
        return df

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _read_csv_skip_malformed(self, path: str) -> pd.DataFrame:
        """Read CSV, logging and skipping rows that cannot be parsed."""
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            raw = fh.read()

        lines = raw.splitlines()
        if not lines:
            return pd.DataFrame()

        header = lines[0]
        good_lines: list[str] = [header]
        expected_cols = len(header.split(","))

        for idx, line in enumerate(lines[1:], start=1):
            # A row is considered malformed when the number of comma-separated
            # fields does not match the header column count.
            if len(line.split(",")) != expected_cols:
                logger.warning(
                    "Skipping malformed row",
                    extra={"row_index": idx, "content": line[:120]},
                )
                continue
            good_lines.append(line)

        csv_text = "\n".join(good_lines)
        try:
            df = pd.read_csv(io.StringIO(csv_text))
        except Exception as exc:  # pragma: no cover
            logger.error("Failed to parse CSV after filtering malformed rows", extra={"error": str(exc)})
            raise

        return df

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute or drop missing values in structured fields."""
        if df.empty:
            return df

        present_fields = [f for f in STRUCTURED_FIELDS if f in df.columns]
        if not present_fields:
            return df

        strategy = self._strategy

        if strategy == "mean":
            for col in present_fields:
                if df[col].isna().any():
                    fill_val = df[col].mean()
                    logger.info(
                        "Imputing missing values with mean",
                        extra={"column": col, "fill_value": fill_val},
                    )
                    df[col] = df[col].fillna(fill_val)

        elif strategy == "median":
            for col in present_fields:
                if df[col].isna().any():
                    fill_val = df[col].median()
                    logger.info(
                        "Imputing missing values with median",
                        extra={"column": col, "fill_value": fill_val},
                    )
                    df[col] = df[col].fillna(fill_val)

        elif strategy == "drop":
            before = len(df)
            df = df.dropna(subset=present_fields)
            dropped = before - len(df)
            if dropped:
                logger.info(
                    "Dropped rows with missing structured fields",
                    extra={"dropped_rows": dropped, "strategy": "drop"},
                )

        else:
            logger.warning(
                "Unknown missing_strategy; no imputation applied",
                extra={"strategy": strategy},
            )

        return df
