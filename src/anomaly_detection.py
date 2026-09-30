"""Expected KPI values (seasonal-naive baseline) and deviations from them.

WHAT
    For every day, an "expected" value of the KPI and how far the actual value
    is from it:

        expected(t)        = median( KPI(t-7), KPI(t-14), KPI(t-21), KPI(t-28) )
        deviation          = actual - expected
        relative_deviation = deviation / expected

WHY
    "Revenue is 11.8M today" means nothing on its own: a Saturday is normally
    ~10% higher than a Monday. Comparing each day with the SAME WEEKDAY in the
    previous weeks removes the weekly pattern, so what remains is the part of
    the movement that is actually unusual.

HOW
    Shift the daily series by 7, 14, 21 and 28 days (same weekday, 1-4 weeks
    earlier) and take the median of those values.

    Why the median of 4 weeks and not simply "same day last week"? Because of
    the ECHO problem: if last Monday had an outage (-50%), a plain "vs last
    week" rule makes this perfectly normal Monday look like a +100% spike. One
    odd value among four barely moves a median, so the echo disappears. (The
    mean would not help: one -50% day drags a 4-week mean down by 12.5%.)

ASSUMPTIONS
    - One value per consecutive day (`clean_data` guarantees this), so shifting
      by 7 positions means "7 days earlier".
    - The weekly pattern is stable and the level changes slowly.

LIMITATIONS
    - Needs history: a day is only evaluated when at least 3 of the 4 reference
      weeks exist, so the first 21 days have no expected value.
    - Holidays that move between weekdays and permanent level shifts (e.g. a
      new store that stays open) are not modelled; after a permanent jump the
      baseline needs 2-3 weeks to catch up.
    - A simple, explainable baseline by design - no forecasting model.
"""

from __future__ import annotations

import logging

import pandas as pd

from src.config import Settings

logger = logging.getLogger(__name__)


def _check_daily_index(series: pd.Series) -> None:
    """Positional shifts equal calendar shifts only with one value per day."""
    if isinstance(series.index, pd.DatetimeIndex) and len(series) > 1:
        steps = series.index.to_series().diff().dropna()
        if not (steps == pd.Timedelta(days=1)).all():
            raise ValueError(
                "The series must have exactly one value per consecutive day - "
                "run clean_data() first."
            )


def seasonal_baseline(series: pd.Series, period: int = 7, weeks: int = 4) -> pd.Series:
    """Expected value of each day: median of the same weekday in previous weeks.

    Example with period=7, weeks=4: the expected value of a Monday is the median
    of the 4 previous Mondays.

    A day is evaluated only if at least `weeks - 1` reference values exist
    (3 of 4 by default); otherwise its expected value is NaN. This keeps the
    median meaningful at the start of the data instead of trusting 1-2 values.

    Args:
        series: KPI values, one per consecutive day, in date order.
        period: Seasonal period in days (7 = weekly).
        weeks: Number of previous periods to take the median over.

    Returns:
        Series aligned with `series`, named "expected".
    """
    _check_daily_index(series)
    # One column per reference week: value 7, 14, 21 and 28 days earlier.
    lagged = pd.concat(
        {week: series.shift(period * week) for week in range(1, weeks + 1)}, axis=1
    )
    min_values = max(1, weeks - 1)
    enough_history = lagged.notna().sum(axis=1) >= min_values
    expected = lagged.median(axis=1).where(enough_history)
    expected.name = "expected"
    return expected


def compute_deviations(series: pd.Series, settings: Settings) -> pd.DataFrame:
    """Actual vs expected for every day.

    Args:
        series: Daily KPI series, e.g. from `aggregate_kpi`.
        settings: Uses `anomaly.seasonal_period_days` and `anomaly.baseline_weeks`.

    Returns:
        DataFrame with columns date, actual, expected, deviation,
        relative_deviation. `relative_deviation` is NaN where there is no
        baseline yet (warm-up) or where expected is 0, because a percentage of
        zero is undefined (it would otherwise be infinite).
    """
    expected = seasonal_baseline(
        series,
        period=settings.anomaly.seasonal_period_days,
        weeks=settings.anomaly.baseline_weeks,
    )
    deviation = series - expected
    relative_deviation = deviation / expected.where(expected != 0)
    return pd.DataFrame({
        "date": series.index,
        "actual": series.to_numpy(dtype=float),
        "expected": expected.to_numpy(),
        "deviation": deviation.to_numpy(),
        "relative_deviation": relative_deviation.to_numpy(),
    })


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from src.config import INJECTED_ANOMALIES_PATH, get_settings
    from src.preprocessing import aggregate_kpi, clean_data, load_data

    settings = get_settings()
    kpi_df = clean_data(load_data())

    company = compute_deviations(aggregate_kpi(kpi_df), settings)
    first_day = company.loc[company["expected"].notna(), "date"].iloc[0]
    print(f"Company total: {int(company['expected'].isna().sum())} warm-up days without "
          f"a baseline; first evaluated day {first_day:%Y-%m-%d}")

    # Sanity check against the answer key. The answer key is only read here and
    # in the evaluation - the pipeline itself never sees it.
    answer_key = pd.read_csv(INJECTED_ANOMALIES_PATH, parse_dates=["start_date", "end_date"])

    demo = answer_key[answer_key["is_demo"]].iloc[0]
    print("\nDemo case at company level (three overlapping declines)")
    demo_days = company[company["date"].between(demo["start_date"], demo["end_date"])]
    for row in demo_days.itertuples(index=False):
        print(f"  {row.date:%a %Y-%m-%d}  actual {row.actual / 1e6:6.2f}M  "
              f"expected {row.expected / 1e6:6.2f}M  "
              f"deviation {row.deviation / 1e6:+6.2f}M  ({row.relative_deviation:+.1%})")

    print("\nSingle-segment anomalies: injected size vs measured relative deviation")
    print("  (measured = average over the anomaly days, on the segment's own series)")
    print(f"  {'id':<6} {'segment':<20} {'injected':>9} {'measured':>9}")
    single = answer_key[answer_key["second_dimension"].isna() & ~answer_key["is_demo"]]
    for anomaly in single.itertuples(index=False):
        series = aggregate_kpi(kpi_df, dimension=anomaly.dimension, category=anomaly.category)
        result = compute_deviations(series, settings)
        in_period = result["date"].between(anomaly.start_date, anomaly.end_date)
        measured = result.loc[in_period, "relative_deviation"].mean()
        segment = f"{anomaly.dimension}={anomaly.category}"
        print(f"  {anomaly.anomaly_id:<6} {segment:<20} {anomaly.pct_change:>+9.0%} "
              f"{measured:>+9.1%}")
    print("\n  Region x product anomalies are diluted in single-dimension series; "
          "they are checked with the cross-check in Phase 6.")
