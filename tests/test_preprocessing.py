"""Tests for `src.preprocessing`.

All tests use the small in-memory `small_kpi_df` fixture (14 days x 48
segments, revenue 100.0 everywhere), so every expected number can be worked
out by hand.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.preprocessing import (
    N_SEGMENTS,
    DataValidationError,
    aggregate_kpi,
    clean_data,
    load_data,
    validate_data,
)


def _segment_rows(df: pd.DataFrame, date: str, region: str, product: str, channel: str) -> pd.Series:
    return (
        (df["date"] == date)
        & (df["region"] == region)
        & (df["product"] == product)
        & (df["channel"] == channel)
    )


def test_valid_data_passes_and_is_kept_intact(small_kpi_df: pd.DataFrame) -> None:
    """Clean input: no errors, no warnings, nothing added or removed."""
    report = validate_data(small_kpi_df)
    assert report.is_valid
    assert report.errors == [] and report.warnings == []
    assert report.counts["rows"] == 14 * N_SEGMENTS

    cleaned = clean_data(small_kpi_df)
    assert len(cleaned) == 14 * N_SEGMENTS
    assert pd.api.types.is_datetime64_any_dtype(cleaned["date"])
    assert cleaned["revenue"].sum() == 14 * N_SEGMENTS * 100.0


def test_duplicates_and_invalid_categories_are_reported(small_kpi_df: pd.DataFrame) -> None:
    """Spec test 1: duplicate records and invalid categories are reported."""
    conflicting = small_kpi_df.iloc[[0]].assign(revenue=999.0)  # same key, other value
    exact_copy = small_kpi_df.iloc[[1]]                           # identical row
    unknown = small_kpi_df.iloc[[2]].assign(region="Atlantis")
    df = pd.concat([small_kpi_df, conflicting, exact_copy, unknown], ignore_index=True)

    report = validate_data(df)

    assert not report.is_valid
    assert report.counts["conflicting_duplicates"] == 1
    assert report.counts["exact_duplicates"] == 1
    assert report.counts["invalid_categories"] == 1
    assert any("Atlantis" in message for message in report.errors)
    assert any("more than once" in message for message in report.errors)
    assert any("exact copies" in message for message in report.warnings)

    # Bad data is never silently passed on.
    with pytest.raises(DataValidationError):
        clean_data(df)


def test_missing_combinations_are_filled_so_series_are_continuous(
    small_kpi_df: pd.DataFrame,
) -> None:
    """Spec test 2: absent date x segment rows are filled with 0, no gaps remain."""
    one_row = _segment_rows(small_kpi_df, "2024-01-03", "West", "Product B", "Retail")
    whole_day = small_kpi_df["date"] == "2024-01-05"
    df = small_kpi_df[~one_row & ~whole_day]

    report = validate_data(df)
    assert report.is_valid                      # gaps are warnings, not errors
    assert report.counts["missing_combinations"] == 1 + N_SEGMENTS
    assert report.counts["missing_days"] == 1
    assert any("whole day" in message for message in report.warnings)

    cleaned = clean_data(df)
    assert len(cleaned) == 14 * N_SEGMENTS
    filled = cleaned[_segment_rows(cleaned, "2024-01-03", "West", "Product B", "Retail")]
    assert filled["revenue"].tolist() == [0.0]

    total = aggregate_kpi(cleaned)
    assert total.index.equals(pd.date_range("2024-01-01", periods=14, freq="D"))
    assert total[pd.Timestamp("2024-01-03")] == 4_800.0 - 100.0
    assert total[pd.Timestamp("2024-01-05")] == 0.0


def test_bad_kpi_values_are_errors_and_missing_keys_are_warnings(
    small_kpi_df: pd.DataFrame,
) -> None:
    """Missing / non-numeric / negative revenue stop the pipeline; a row without a
    region is only dropped (with a warning)."""
    df = small_kpi_df.copy()
    df["revenue"] = df["revenue"].astype(object)
    df.loc[0, "revenue"] = None
    df.loc[1, "revenue"] = -5.0
    df.loc[2, "revenue"] = "abc"
    df.loc[3, "region"] = None

    report = validate_data(df)

    assert report.counts["missing_revenue"] == 1
    assert report.counts["negative_revenue"] == 1
    assert report.counts["non_numeric_revenue"] == 1
    assert report.counts["missing_keys"] == 1
    assert len(report.errors) == 3
    assert any("will be dropped" in message for message in report.warnings)


def test_aggregate_kpi_company_and_segment_totals(small_kpi_df: pd.DataFrame) -> None:
    """Totals match the hand calculation; invalid requests fail clearly."""
    cleaned = clean_data(small_kpi_df)

    assert (aggregate_kpi(cleaned) == 4_800.0).all()
    assert (aggregate_kpi(cleaned, dimension="region", category="West") == 1_200.0).all()
    assert (aggregate_kpi(cleaned, dimension="channel", category="Online") == 1_600.0).all()

    with pytest.raises(ValueError, match="Unsupported KPI"):
        aggregate_kpi(cleaned, kpi="profit")
    with pytest.raises(ValueError, match="Unknown category"):
        aggregate_kpi(cleaned, dimension="region", category="Atlantis")
    with pytest.raises(ValueError, match="clean_data"):
        aggregate_kpi(small_kpi_df)             # raw data: dates are still strings


def test_load_data_round_trip_and_missing_file(
    small_kpi_df: pd.DataFrame, tmp_path
) -> None:
    """A written CSV loads back unchanged; a missing file explains how to create it."""
    path = tmp_path / "kpi.csv"
    small_kpi_df.to_csv(path, index=False)
    loaded = load_data(path)
    assert len(loaded) == len(small_kpi_df)
    assert validate_data(loaded).is_valid

    with pytest.raises(FileNotFoundError, match="generate_data.py"):
        load_data(tmp_path / "does_not_exist.csv")
