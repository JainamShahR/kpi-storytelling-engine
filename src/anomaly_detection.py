"""Anomaly detection: seasonal-naive baseline + robust z-score rule.

WHAT
    Step 1 - expected value (seasonal-naive baseline):
        expected(t)        = median( KPI(t-7), KPI(t-14), KPI(t-21), KPI(t-28) )
        deviation          = actual - expected
        relative_deviation = deviation / expected

    Step 2 - how unusual is today's deviation? (robust z-score)
        z(t) = (deviation(t) - median(past deviations)) / (1.4826 x MAD(past deviations))

    Step 3 - the rule. A day is an anomaly when BOTH hold:
        |z| >= ZSCORE_THRESHOLD (3.0)  and  |relative_deviation| >= MIN_RELATIVE_CHANGE (5%)

    Results are "anomalies under the configured statistical rule" - they are
    not claims of statistical significance.

WHY
    Step 1 removes the weekly pattern: a Saturday is compared with Saturdays.
    Step 2 asks whether a deviation is big FOR THIS SERIES: a 4% move is
    extreme for the stable company total but routine for a noisy segment.
    Step 3's second condition drops movements that are statistically unusual
    but too small to matter commercially.
    Detection runs on the company total AND on every region, product and
    channel, because a problem in one segment can be diluted to invisibility
    in the total (e.g. Partner -10% moves the total by only ~-2%).

HOW
    Baseline: the median of 4 same weekdays avoids the ECHO problem. With a
    plain "same day last week" rule, last week's outage makes this normal week
    look like a spike; one odd value among four barely moves a median.
    Z-score: median and MAD of the previous 56 days, EXCLUDING today, so today
    never influences its own yardstick (no data leakage). Median/MAD instead of
    mean/standard deviation because past anomalies sit in that window too, and
    one -50% day inflates a standard deviation enough to hide the next anomaly.

ASSUMPTIONS
    - One value per consecutive day (`clean_data` guarantees this).
    - The weekly pattern is stable and the level changes slowly.
    - Normal day-to-day deviations are roughly symmetric around a typical value.

LIMITATIONS
    - Needs 49 days of history (21 for the baseline + 28 past deviations).
    - Monitoring 12 series multiplies the chances of a false alarm.
    - Small anomalies in small segments can stay below the rule.
    - Holidays that move between weekdays and permanent level shifts are not
      modelled; after a permanent jump the baseline needs 2-3 weeks to adapt.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.config import DIMENSIONS, Settings
from src.preprocessing import aggregate_kpi

logger = logging.getLogger(__name__)

#: For normally distributed data, MAD x 1.4826 equals the standard deviation,
#: so the robust z-score reads on the familiar scale: |z| = 3 means "about three
#: standard deviations from typical".
MAD_TO_STD = 1.4826

#: Replaces a zero MAD (a perfectly constant history) to avoid dividing by zero.
EPSILON = 1e-9

#: Series monitored by `detect_all_levels`: the company total plus every
#: category of every dimension (1 + 4 + 4 + 3 = 12). Edit this list to change
#: what is monitored.
DETECTION_LEVELS: list[tuple[str | None, str | None]] = [(None, None)] + [
    (dimension, category)
    for dimension, categories in DIMENSIONS.items()
    for category in categories
]


def level_name(dimension: str | None, category: str | None) -> str:
    """'Company total' or e.g. 'region=West'."""
    return "Company total" if dimension is None else f"{dimension}={category}"


# ---------------------------------------------------------------------------
# Step 1 - seasonal-naive baseline
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Step 2 - robust z-score
# ---------------------------------------------------------------------------


def robust_zscore(deviation: pd.Series, window: int = 56, min_periods: int = 28) -> pd.Series:
    """How unusual each day's deviation is compared with PAST deviations.

        z(t) = (deviation(t) - median(past)) / (1.4826 x MAD(past))
        MAD  = median of |value - median| (median absolute deviation)

    `past` is the previous `window` days EXCLUDING day t, so today's value
    never influences the yardstick it is measured with (no data leakage).
    With fewer than `min_periods` past values the z-score is NaN.

    If the MAD is 0 (a perfectly constant history) it is replaced by a tiny
    epsilon and a warning is logged: any change then gets a huge |z|, and the
    minimum-relative-change condition decides whether it matters.
    """
    past = deviation.shift(1)  # day t only sees days before t
    rolling = past.rolling(window=window, min_periods=min_periods)
    center = rolling.median()
    mad = rolling.apply(
        lambda values: np.nanmedian(np.abs(values - np.nanmedian(values))), raw=True
    )
    scale = MAD_TO_STD * mad
    zero_scale = scale == 0
    if zero_scale.any():
        logger.warning(
            "MAD of past deviations is 0 on %d day(s) (constant history); using "
            "epsilon=%g, so any change on those days gets a very large z-score.",
            int(zero_scale.sum()), EPSILON,
        )
        scale = scale.mask(zero_scale, EPSILON)
    return ((deviation - center) / scale).rename("z_score")


# ---------------------------------------------------------------------------
# Step 3 - the rule, severity and direction
# ---------------------------------------------------------------------------


def severity_from_z(z_score: pd.Series, threshold: float) -> pd.Series:
    """Severity band of |z|; the bands are derived from the threshold.

    With the default threshold of 3:
        |z| < 3        -> Normal
        3 <= |z| < 4   -> Moderate
        4 <= |z| < 5   -> High
        |z| >= 5       -> Critical
    """
    abs_z = z_score.abs()
    labels = np.select(
        [abs_z >= threshold + 2, abs_z >= threshold + 1, abs_z >= threshold],
        ["Critical", "High", "Moderate"],
        default="Normal",
    )
    return pd.Series(labels, index=z_score.index, name="severity")


def detect_anomalies(series: pd.Series, settings: Settings) -> pd.DataFrame:
    """Apply baseline + robust z-score + rule to one daily series.

    Returns:
        DataFrame with columns date, actual, expected, deviation,
        relative_deviation, z_score, is_anomaly, severity, direction.
        severity is "Not evaluated" during the warm-up (no baseline or not
        enough past deviations), "Normal" for evaluated non-anomalies, and
        Moderate / High / Critical for anomalies.
    """
    rule = settings.anomaly
    result = compute_deviations(series, settings)
    result["z_score"] = robust_zscore(
        result["deviation"], rule.zscore_window_days, rule.zscore_min_periods
    )

    evaluated = result["z_score"].notna() & result["relative_deviation"].notna()
    unusual = result["z_score"].abs() >= rule.zscore_threshold
    big_enough = result["relative_deviation"].abs() >= rule.min_relative_change
    result["is_anomaly"] = evaluated & unusual & big_enough

    result["severity"] = (
        severity_from_z(result["z_score"], rule.zscore_threshold)
        .where(result["is_anomaly"], "Normal")
        .where(evaluated, "Not evaluated")
    )
    result["direction"] = np.select(
        [result["deviation"] < 0, result["deviation"] > 0], ["decline", "increase"], default=""
    )
    return result


def detect_all_levels(
    df: pd.DataFrame,
    settings: Settings,
    levels: Sequence[tuple[str | None, str | None]] = DETECTION_LEVELS,
) -> pd.DataFrame:
    """Run `detect_anomalies` on the company total and on every segment.

    Args:
        df: Output of `clean_data`.
        settings: Full settings (the KPI name comes from `settings.kpi`).
        levels: (dimension, category) pairs; (None, None) is the company total.

    Returns:
        The `detect_anomalies` columns for every level stacked together, with
        `level`, `dimension` and `category` added in front.
    """
    frames = []
    for dimension, category in levels:
        series = aggregate_kpi(df, kpi=settings.kpi, dimension=dimension, category=category)
        result = detect_anomalies(series, settings)
        result.insert(0, "level", level_name(dimension, category))
        result.insert(1, "dimension", dimension)
        result.insert(2, "category", category)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Naive rule - used ONLY as a comparison baseline in the evaluation
# ---------------------------------------------------------------------------


def naive_detector(series: pd.Series, threshold: float = 0.10) -> pd.Series:
    """Flag a day if it differs from the same weekday last week by more than 10%.

        flag(t) = | value(t) / value(t-7) - 1 | > threshold

    Deliberately simple: this is the rule a busy analyst might use, so the
    evaluation can show what the robust method adds. It suffers from the echo
    problem (an anomaly last week makes a normal day this week look unusual)
    and ignores how noisy each series normally is.
    """
    _check_daily_index(series)
    last_week = series.shift(7)
    change = series / last_week.where(last_week != 0) - 1
    return (change.abs() > threshold).rename("naive_flag")


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from src.config import INJECTED_ANOMALIES_PATH, get_settings
    from src.preprocessing import clean_data, load_data

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    rule = settings.anomaly
    kpi_df = clean_data(load_data())

    results = detect_all_levels(kpi_df, settings)
    flagged = results[results["is_anomaly"]]

    print(f"Rule: |z| >= {rule.zscore_threshold} and |relative change| >= "
          f"{rule.min_relative_change:.0%}; first evaluated day = day {rule.warmup_days}")
    print(f"Monitored series: {results['level'].nunique()}  |  flagged series-days: "
          f"{len(flagged)}  |  distinct flagged dates: {flagged['date'].nunique()}")
    print("Severity of flags: " + ", ".join(
        f"{name}: {count}" for name, count in flagged["severity"].value_counts().items()))

    print("\nFlagged days per monitored series")
    per_level = flagged["level"].value_counts().reindex(results["level"].unique(), fill_value=0)
    for level, count in per_level.items():
        print(f"  {level:<22}{count:>4}")

    # Check against the answer key (read only here and in the evaluation).
    answer_key = pd.read_csv(INJECTED_ANOMALIES_PATH, parse_dates=["start_date", "end_date"])
    print("\nInjected anomalies - flagged on the company total or on the injected segment?")
    print(f"  {'id':<6} {'segment':<34} {'change':>6}  {'detected':<9}{'max |z|':>8}  severity")
    detected = 0
    injected_dates: set[pd.Timestamp] = set()
    for anomaly in answer_key.itertuples(index=False):
        segment = level_name(anomaly.dimension, anomaly.category)
        levels = {"Company total", segment}
        if isinstance(anomaly.second_dimension, str):
            second = level_name(anomaly.second_dimension, anomaly.second_category)
            levels.add(second)
            segment = f"{segment} & {second}"
        in_period = results["date"].between(anomaly.start_date, anomaly.end_date)
        injected_dates.update(results.loc[in_period, "date"])
        rows = results[in_period & results["level"].isin(levels)]
        hits = rows[rows["is_anomaly"]]
        detected += int(not hits.empty)
        severity = hits.loc[hits["z_score"].abs().idxmax(), "severity"] if len(hits) else "-"
        print(f"  {anomaly.anomaly_id:<6} {segment:<34} {anomaly.pct_change:>+6.0%}  "
              f"{'yes' if len(hits) else 'no':<9}{rows['z_score'].abs().max():>8.1f}  {severity}")

    false_dates = sorted(set(flagged["date"]) - injected_dates)
    print(f"\nDetected {detected} of {len(answer_key)} injected anomalies "
          "(at least one day of the period flagged).")
    print(f"Flagged dates with no injected anomaly: {len(false_dates)} "
          "(false alarms under this rule)")
    if false_dates:
        print("  e.g. " + ", ".join(f"{day:%Y-%m-%d}" for day in false_dates[:6]))
