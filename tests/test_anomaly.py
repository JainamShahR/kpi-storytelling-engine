"""Tests for `src.anomaly_detection`.

Phase 4 covers the seasonal-naive baseline. Every series here is a tiny
hand-made daily series starting on Monday 2024-01-01, so each expected value
can be worked out on paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import compute_deviations, seasonal_baseline
from src.config import Settings


def _daily(values: list[float], start: str = "2024-01-01") -> pd.Series:
    """Daily series starting on `start` (2024-01-01 was a Monday)."""
    index = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=index, dtype=float)


def test_seasonal_baseline_is_median_of_same_weekday_lags() -> None:
    """Spec test 4: median of t-7, t-14, t-21, t-28; NaN with too little history."""
    series = _daily([100.0 + t for t in range(35)])      # value on day t = 100 + t
    expected = seasonal_baseline(series, period=7, weeks=4)

    # Days 0-20: fewer than 3 of the 4 reference values exist -> not evaluated.
    assert expected.iloc[:21].isna().all()
    # Day 21: only t-7, t-14, t-21 exist -> median(114, 107, 100) = 107.
    assert expected.iloc[21] == 107.0
    # Day 30: median(123, 116, 109, 102) = (109 + 116) / 2 = 112.5.
    assert expected.iloc[30] == 112.5


def test_expected_value_follows_the_weekly_pattern(default_settings: Settings) -> None:
    """Saturdays are compared with Saturdays, so a normal weekend is no deviation."""
    dates = pd.date_range("2024-01-01", periods=42, freq="D")
    series = pd.Series(np.where(dates.dayofweek >= 5, 150.0, 100.0), index=dates)

    result = compute_deviations(series, default_settings)
    evaluated = result.dropna(subset=["expected"])

    assert len(evaluated) == 42 - 21
    assert (evaluated["expected"] == evaluated["actual"]).all()
    assert (evaluated["deviation"] == 0.0).all()


def test_one_off_spike_does_not_echo_into_next_week(default_settings: Settings) -> None:
    """The median ignores one odd value among four, so last week's anomaly does
    not make this week look abnormal (the "echo" problem)."""
    values = [100.0] * 36
    values[28] = 1_000.0                    # one-off spike on day 28
    result = compute_deviations(_daily(values), default_settings)

    one_week_later = result.iloc[35]        # same weekday, 7 days after the spike
    assert one_week_later["expected"] == 100.0   # the mean of the 4 lags would be 325
    assert one_week_later["deviation"] == 0.0


def test_deviation_matches_spec_example(default_settings: Settings) -> None:
    """Actual 7.5M vs expected 10.0M -> deviation -2.5M, relative deviation -25%."""
    series = _daily([10_000_000.0] * 35 + [7_500_000.0])
    last_day = compute_deviations(series, default_settings).iloc[-1]

    assert last_day["expected"] == 10_000_000.0
    assert last_day["deviation"] == -2_500_000.0
    assert last_day["relative_deviation"] == pytest.approx(-0.25)


def test_zero_baseline_and_gaps_are_handled(default_settings: Settings) -> None:
    """A zero baseline gives an undefined (NaN) percentage, never infinity; a
    series with a missing day is rejected because shifts would compare the
    wrong days."""
    zeros = _daily([0.0] * 35 + [50.0])
    last_day = compute_deviations(zeros, default_settings).iloc[-1]
    assert last_day["expected"] == 0.0
    assert last_day["deviation"] == 50.0
    assert np.isnan(last_day["relative_deviation"])

    with_gap = _daily([100.0] * 30).drop(pd.Timestamp("2024-01-10"))
    with pytest.raises(ValueError, match="consecutive day"):
        seasonal_baseline(with_gap)
