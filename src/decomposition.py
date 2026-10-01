"""Root-cause analysis: which regions, products and channels drove a change.

WHAT
    For an anomalous date t, splits the company's change against its baseline
    into the contribution of every region, product and channel:

        baseline(c)         = mean of category c on the reference days
                              (same weekday 1-4 weeks earlier: t-7, t-14, t-21, t-28)
        absolute_change(c)  = actual(c) - baseline(c)
        pct_change(c)       = absolute_change(c) / baseline(c) x 100        (display only)
        contribution_pct(c) = absolute_change(c) / total absolute change x 100

    plus a region x product cross-check listing the top combined segments.

WHY
    "Revenue fell 25%" is not actionable. "West explains 57% of the drop" is.

HOW - the four key ideas
    1. Rank by CONTRIBUTION, not by percentage change. A small category can
       fall a lot and still explain little: Product D (10% of revenue) at -50%
       moves the total by -5%, while Product A (40%) at -20% moves it by -8%.
       The contribution answers "where did the money actually go?".
    2. Contributions add up to 100% WITHIN each dimension, never across.
       Region, product and channel are three views of the SAME change: West's
       Product B sales are counted in "West" and again in "Product B". Adding
       the three views would count the change three times.
    3. Mean baseline here, median baseline in detection. Category means add up
       exactly to the total mean, so contributions sum to exactly 100%. Medians
       do not add up, which is why detection (which needs robustness) and
       decomposition (which needs additivity) use different statistics.
    4. If the total change is below 2% of the baseline, decomposition is
       skipped: dividing by a near-zero total makes the percentages explode.

ASSUMPTIONS
    - The KPI is additive: category values sum to the company total.
    - Data is clean (every date has every segment), as `clean_data` ensures.

LIMITATIONS
    - Single-dimension views overlap; the cross-check shows whether a change
      sits in one combined segment (e.g. West x Product B).
    - The mean is not robust: an anomaly on one of the reference days shifts
      every baseline. In the generated data this happens often enough to
      cause a visible mis-attribution (see docs/phase_log.md, Phase 6).
    - It explains WHERE the change happened, not WHY - that is the job of the
      business-note retrieval.
    - It always decomposes the company total; drilling down inside a segment
      is listed as a future improvement in the spec.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

from src.config import DIMENSIONS, MIN_DECOMPOSITION_CHANGE, Settings

logger = logging.getLogger(__name__)


@dataclass
class Decomposition:
    """Result of `decompose` for one date.

    Totals are for the whole company; `by_dimension` maps "region",
    "product" and "channel" to a table with the columns dimension, category,
    baseline, actual, absolute_change, pct_change, contribution_pct - sorted
    by contribution, so the first row is that dimension's top contributor.
    pct_change and contribution_pct are in percent (-46.0 means -46%).
    """

    date: pd.Timestamp
    reference_dates: list[pd.Timestamp]
    baseline: float
    actual: float
    absolute_change: float
    pct_change: float
    by_dimension: dict[str, pd.DataFrame] = field(default_factory=dict)
    message: str = ""

    @property
    def skipped(self) -> bool:
        """True when the change was too small to decompose (see `message`)."""
        return not self.by_dimension


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reference_dates(df: pd.DataFrame, date: pd.Timestamp, settings: Settings) -> list[pd.Timestamp]:
    """The same weekday 1..BASELINE_WEEKS weeks earlier - the days detection used."""
    period = settings.anomaly.seasonal_period_days
    weeks = settings.anomaly.baseline_weeks
    available = pd.DatetimeIndex(df["date"].unique())
    if date not in available:
        raise ValueError(f"{date:%Y-%m-%d} is not in the data.")
    candidates = [date - pd.Timedelta(days=period * week) for week in range(1, weeks + 1)]
    found = [day for day in candidates if day in available]
    if len(found) < max(1, weeks - 1):
        raise ValueError(
            f"Not enough history before {date:%Y-%m-%d}: {len(found)} of {weeks} "
            "reference days are in the data."
        )
    return found


def _totals(
    df: pd.DataFrame, date: pd.Timestamp, references: list[pd.Timestamp], kpi: str
) -> tuple[float, float]:
    """(baseline, actual) of the company total: mean of the reference days vs the day."""
    actual = float(df.loc[df["date"] == date, kpi].sum())
    baseline = float(df.loc[df["date"].isin(references), kpi].sum()) / len(references)
    return baseline, actual


def _breakdown(
    df: pd.DataFrame,
    date: pd.Timestamp,
    references: list[pd.Timestamp],
    by: list[str],
    total_change: float,
    kpi: str,
) -> pd.DataFrame:
    """Baseline, actual, change and contribution for every group of `by`.

    The baseline of a group is its total over the reference days divided by the
    number of reference days - i.e. the mean of its daily values - so the group
    baselines add up exactly to the company baseline.
    """
    actual = df[df["date"] == date].groupby(by)[kpi].sum()
    baseline = df[df["date"].isin(references)].groupby(by)[kpi].sum() / len(references)

    table = pd.DataFrame({"baseline": baseline, "actual": actual}).fillna(0.0)
    table["absolute_change"] = table["actual"] - table["baseline"]
    table["pct_change"] = (
        table["absolute_change"] / table["baseline"].where(table["baseline"] != 0) * 100
    )
    # Positive = moved in the same direction as the total change.
    table["contribution_pct"] = table["absolute_change"] / total_change * 100

    # Rank by contribution (ties broken by name, so the order is deterministic).
    return (
        table.reset_index()
        .sort_values(["contribution_pct", *by], ascending=[False] + [True] * len(by),
                     kind="stable")
        .reset_index(drop=True)
    )


def _too_small(baseline: float, change: float) -> bool:
    return baseline == 0 or abs(change) < MIN_DECOMPOSITION_CHANGE * baseline


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def decompose(df: pd.DataFrame, date: str | pd.Timestamp, settings: Settings) -> Decomposition:
    """Split the company's change on `date` into region, product and channel contributions.

    Args:
        df: Output of `clean_data`.
        date: The (anomalous) day to explain.
        settings: Uses the baseline settings and `settings.kpi`.

    Returns:
        A `Decomposition`. If the total change is below 2% of the baseline,
        `skipped` is True, `by_dimension` is empty and `message` explains why.

    Raises:
        ValueError: the date is not in the data or has too little history.
    """
    date = pd.Timestamp(date)
    kpi = settings.kpi
    references = _reference_dates(df, date, settings)
    baseline, actual = _totals(df, date, references, kpi)
    change = actual - baseline
    pct_change = change / baseline * 100 if baseline else float("nan")

    result = Decomposition(
        date=date,
        reference_dates=references,
        baseline=baseline,
        actual=actual,
        absolute_change=change,
        pct_change=pct_change,
    )
    if _too_small(baseline, change):
        result.message = (
            f"The total change on {date:%Y-%m-%d} is {pct_change:+.1f}% of the baseline, "
            f"below the {MIN_DECOMPOSITION_CHANGE:.0%} needed for stable contribution "
            "percentages, so the decomposition was skipped."
        )
        logger.info(result.message)
        return result

    for dimension in DIMENSIONS:
        table = _breakdown(df, date, references, [dimension], change, kpi)
        table = table.rename(columns={dimension: "category"})
        table.insert(0, "dimension", dimension)
        result.by_dimension[dimension] = table
    return result


def top_contributors(decomposition: Decomposition, n: int = 3) -> list[dict]:
    """The n largest contributions across all dimensions, as plain dicts.

    The first item is the OVERALL top contributor: the dimension whose top
    category has the highest contribution, i.e. the most concentrated
    explanation. The percentages come from overlapping views of the same
    change, so they must not be added together.
    """
    if decomposition.skipped:
        return []
    combined = pd.concat(decomposition.by_dimension.values(), ignore_index=True)
    combined = combined.sort_values(
        ["contribution_pct", "dimension", "category"],
        ascending=[False, True, True],
        kind="stable",
    )
    return combined.head(n).to_dict("records")


def cross_dimension(
    df: pd.DataFrame,
    date: str | pd.Timestamp,
    settings: Settings,
    dims: tuple[str, str] = ("region", "product"),
    n: int = 3,
) -> pd.DataFrame:
    """Top n combined segments (e.g. region x product) by contribution.

    Shows whether single-dimension findings overlap: if "West" and "Product B"
    are both top contributors, this reveals whether the change sits in the
    combined segment West x Product B.

    Returns:
        Columns: the two dimension names, baseline, actual, absolute_change,
        pct_change, contribution_pct. Empty when the total change is too small.
    """
    date = pd.Timestamp(date)
    references = _reference_dates(df, date, settings)
    baseline, actual = _totals(df, date, references, settings.kpi)
    change = actual - baseline
    if _too_small(baseline, change):
        columns = [*dims, "baseline", "actual", "absolute_change", "pct_change",
                   "contribution_pct"]
        return pd.DataFrame(columns=columns)
    return _breakdown(df, date, references, list(dims), change, settings.kpi).head(n)


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    from src.config import INJECTED_ANOMALIES_PATH, get_settings
    from src.preprocessing import clean_data, load_data

    settings = get_settings()
    kpi_df = clean_data(load_data())
    answer_key = pd.read_csv(INJECTED_ANOMALIES_PATH, parse_dates=["start_date", "end_date"])

    # 1. The demo case in full (three overlapping declines).
    demo_day = answer_key.loc[answer_key["is_demo"], "start_date"].iloc[0]
    demo = decompose(kpi_df, demo_day, settings)
    print(f"Demo case {demo_day:%a %Y-%m-%d}: total {demo.actual / 1e6:.2f}M vs baseline "
          f"{demo.baseline / 1e6:.2f}M ({demo.pct_change:+.1f}%)")
    for dimension, table in demo.by_dimension.items():
        print(f"\n  {dimension} - contributions add up to 100% within this dimension")
        for row in table.itertuples(index=False):
            print(f"    {row.category:<10} baseline {row.baseline / 1e6:5.2f}M  "
                  f"actual {row.actual / 1e6:5.2f}M  change {row.pct_change:+6.1f}%  "
                  f"contribution {row.contribution_pct:6.1f}%")
    print("\n  Top 3 across dimensions (overlapping views - never add these up)")
    for item in top_contributors(demo):
        print(f"    {item['dimension']}={item['category']:<10} {item['contribution_pct']:6.1f}%")
    print("\n  Region x product cross-check (top 3)")
    for row in cross_dimension(kpi_df, demo_day, settings).itertuples(index=False):
        print(f"    {row.region:<6} x {row.product:<10} change {row.pct_change:+6.1f}%  "
              f"contribution {row.contribution_pct:6.1f}%")

    # 2. Every regular injected anomaly, decomposed on its first day.
    print("\nRegular injected anomalies - is the injected segment the top contributor?")
    print(f"  {'id':<6} {'injected segment':<34} {'total':>7}  result")
    top1 = top3 = single_done = pairs_found = pairs_done = skipped = 0
    for anomaly in answer_key[~answer_key["is_demo"]].itertuples(index=False):
        result = decompose(kpi_df, anomaly.start_date, settings)
        is_pair = isinstance(anomaly.second_dimension, str)
        segment = f"{anomaly.dimension}={anomaly.category}"
        if is_pair:
            segment += f" & {anomaly.second_dimension}={anomaly.second_category}"
        prefix = f"  {anomaly.anomaly_id:<6} {segment:<34} {result.pct_change:>+6.1f}%  "
        if result.skipped:
            skipped += 1
            print(prefix + "skipped (total change below 2%)")
            continue
        if is_pair:
            pairs_done += 1
            cross = cross_dimension(kpi_df, anomaly.start_date, settings)
            pairs = list(zip(cross["region"], cross["product"]))
            found = (anomaly.category, anomaly.second_category) in pairs
            pairs_found += int(found)
            rank = pairs.index((anomaly.category, anomaly.second_category)) + 1 if found else None
            print(prefix + (f"pair ranked #{rank} in the cross-check" if found
                            else "pair NOT in the cross-check top 3"))
        else:
            single_done += 1
            table = result.by_dimension[anomaly.dimension]
            is_top1 = table.iloc[0]["category"] == anomaly.category
            leaders = [(item["dimension"], item["category"]) for item in top_contributors(result)]
            is_top3 = (anomaly.dimension, anomaly.category) in leaders
            top1 += int(is_top1)
            top3 += int(is_top3)
            share = table.set_index("category").loc[anomaly.category, "contribution_pct"]
            print(prefix + f"top-1 {'yes' if is_top1 else 'no '}  top-3 "
                  f"{'yes' if is_top3 else 'no '}  (contribution {share:.0f}%)")

    print(f"\nSkipped (total change below 2%): {skipped}")
    print(f"Single-dimension anomalies decomposed: {single_done}  |  "
          f"top-1 correct: {top1}  |  top-3 correct: {top3}")
    print(f"Region x product anomalies decomposed: {pairs_done}  |  "
          f"pair in cross-check top 3: {pairs_found}")
