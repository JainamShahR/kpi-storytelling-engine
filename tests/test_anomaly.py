"""Tests for `src.anomaly_detection`.

Every series here is a small hand-made daily series starting on Monday
2024-01-01. Baseline tests use exact values that can be checked on paper;
detection tests use a level of 100 with seeded 1% noise, so the result is
identical on every run.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import (
    DETECTION_LEVELS,
    compute_deviations,
    detect_all_levels,
    detect_anomalies,
    naive_detector,
    seasonal_baseline,
    severity_from_z,
)
from src.config import Settings
from src.preprocessing import clean_data


def _daily(values: list[float], start: str = "2024-01-01") -> pd.Series:
    """Daily series starting on `start` (2024-01-01 was a Monday)."""
    index = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(values, index=index, dtype=float)


def _noisy_daily(n_days: int, noise: float = 1.0, seed: int = 7) -> pd.Series:
    """Level 100 plus seeded normal noise (1.0 = 1% of the level)."""
    rng = np.random.default_rng(seed)
    return _daily(list(100.0 + rng.normal(0.0, noise, n_days)))


# ---------------------------------------------------------------------------
# Phase 4 - seasonal-naive baseline
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Phase 5 - robust z-score, rule, severity, naive detector
# ---------------------------------------------------------------------------


def test_known_drop_is_detected_with_correct_date_and_direction(
    default_settings: Settings,
) -> None:
    """Spec test 3: an injected -30% drop is the only flagged day, as a decline."""
    series = _noisy_daily(70)
    series.iloc[60] *= 0.7                      # injected -30% drop on day 60

    result = detect_anomalies(series, default_settings)
    flagged = result[result["is_anomaly"]]

    assert flagged["date"].tolist() == [series.index[60]]
    assert flagged["direction"].tolist() == ["decline"]
    # The first z-score appears exactly after the configured warm-up (21 + 28).
    assert result["z_score"].first_valid_index() == default_settings.anomaly.warmup_days


def test_zscore_uses_only_past_data(default_settings: Settings) -> None:
    """Spec test 5: changing a later value never changes an earlier z-score."""
    series = _noisy_daily(80)
    before = detect_anomalies(series, default_settings)["z_score"]

    changed = series.copy()
    changed.iloc[65] = 1_000.0                  # change the "future" (day 65)
    after = detect_anomalies(changed, default_settings)["z_score"]

    pd.testing.assert_series_equal(before.iloc[:65], after.iloc[:65])
    assert after.iloc[65] != before.iloc[65]    # the changed day itself does move


def test_small_change_is_not_flagged_even_if_z_is_large(default_settings: Settings) -> None:
    """Spec test 6: statistically unusual but commercially tiny (+3% < 5%)."""
    series = _noisy_daily(70, noise=0.01)       # an extremely stable series
    series.iloc[60] *= 1.03

    day = detect_anomalies(series, default_settings).iloc[60]

    assert abs(day["z_score"]) >= default_settings.anomaly.zscore_threshold
    assert day["relative_deviation"] == pytest.approx(0.03, abs=0.001)
    assert not day["is_anomaly"]
    assert day["severity"] == "Normal"


def test_severity_follows_configured_thresholds() -> None:
    """Spec test 7: bands are threshold, +1, +2 and move with the threshold."""
    z = pd.Series([2.99, 3.0, -3.99, 4.0, 4.99, -5.0, 9.0])
    assert severity_from_z(z, threshold=3.0).tolist() == [
        "Normal", "Moderate", "Moderate", "High", "High", "Critical", "Critical",
    ]
    assert severity_from_z(pd.Series([2.5, 3.5, 4.5]), threshold=2.5).tolist() == [
        "Moderate", "High", "Critical",
    ]


def test_naive_rule_raises_echo_false_alarm_that_robust_rule_avoids(
    default_settings: Settings,
) -> None:
    """A one-day -50% outage: the naive rule also flags the normal day one week
    later (+100% vs the outage), the robust rule flags only the outage."""
    series = _noisy_daily(63)
    series.iloc[50] *= 0.5

    naive = naive_detector(series)
    robust = detect_anomalies(series, default_settings)["is_anomaly"]

    assert naive.iloc[50] and naive.iloc[57]
    assert robust.iloc[50] and not robust.iloc[57]


def test_zero_mad_uses_epsilon_and_logs_a_warning(
    default_settings: Settings, caplog: pytest.LogCaptureFixture
) -> None:
    """A perfectly constant history has MAD = 0: no division by zero, a warning
    is logged and a real -20% change is still detected."""
    series = _daily([100.0] * 56 + [80.0])
    with caplog.at_level(logging.WARNING, logger="src.anomaly_detection"):
        result = detect_anomalies(series, default_settings)

    assert any("MAD" in record.message for record in caplog.records)
    last_day = result.iloc[-1]
    assert np.isfinite(last_day["z_score"])
    assert last_day["is_anomaly"] and last_day["direction"] == "decline"


def test_detect_all_levels_monitors_company_and_every_segment(
    small_kpi_df: pd.DataFrame, default_settings: Settings
) -> None:
    """12 series (total + 4 regions + 4 products + 3 channels); 14 days of data
    is too little history, so nothing may be evaluated or flagged."""
    results = detect_all_levels(clean_data(small_kpi_df), default_settings)

    assert len(DETECTION_LEVELS) == 1 + 4 + 4 + 3
    assert results["level"].nunique() == 12
    assert len(results) == 12 * 14

    west = results[results["level"] == "region=West"]
    assert set(west["dimension"]) == {"region"} and set(west["category"]) == {"West"}
    assert (west["actual"] == 1_200.0).all()

    assert (results["severity"] == "Not evaluated").all()
    assert not results["is_anomaly"].any()
