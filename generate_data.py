"""Generate the synthetic KPI dataset and the injected-anomaly answer key.

WHAT
    Writes two CSV files to `data/`:

    * `kpi_data.csv` - two years of daily revenue and orders for every
      region x product x channel combination (4 x 4 x 3 = 48 series per day).
    * `injected_anomalies.csv` - the answer key: every anomaly deliberately
      planted in the data, with its dates, segment, size and business cause.

WHY
    Real company data never comes with a list of the "true" anomalies, so there
    is no way to measure whether a detector is right. With synthetic data we
    plant the anomalies ourselves, which turns "the dashboard looks plausible"
    into measurable precision, recall and root-cause accuracy (Phase 14).

HOW
    revenue = base level          (region share x product share x channel share)
              x weekly pattern    (weekends differ, depending on the channel)
              x trend             (+10% per year)
              x yearly wave       (+/-4%, peak in the festive season)
              x random noise      (8% per series per day)

    Then every injected anomaly multiplies the revenue and orders of the
    matching rows and dates by (1 + pct_change).

ASSUMPTIONS
    - The three dimensions are independent: every region has the same product
      and channel mix. This keeps the model small and explainable.
    - Noise is independent per series and per day (no shared daily shocks).

LIMITATIONS
    - Real data contains unlabelled holidays, outliers and missing days; here
      everything unusual is planted and known, so measured detection quality
      will be optimistic compared with messy real data.

Usage:
    python generate_data.py            # write the CSV files and print a summary
    python generate_data.py --plot     # also save reports/data_overview.html
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CHANNELS,
    DATA_DIR,
    DIMENSIONS,
    INJECTED_ANOMALIES_PATH,
    KPI_DATA_PATH,
    PRODUCTS,
    PROJECT_ROOT,
    RANDOM_SEED,
    REGIONS,
    REPORTS_DIR,
)

# ---------------------------------------------------------------------------
# Shape of the synthetic business
# ---------------------------------------------------------------------------
# These numbers describe the fake company, not the analysis, and only this
# script uses them - so they live here rather than in src/config.py.

START_DATE = "2024-01-01"
END_DATE = "2025-12-31"

#: Company revenue on an average day at the start of the data.
TOTAL_BASE_REVENUE = 10_000_000.0

#: Share of revenue per category; each dimension sums to 1. The base level of
#: one series is TOTAL_BASE_REVENUE x region share x product share x channel share.
REGION_SHARE = {"North": 0.28, "South": 0.22, "East": 0.24, "West": 0.26}
PRODUCT_SHARE = {"Product A": 0.35, "Product B": 0.30, "Product C": 0.20, "Product D": 0.15}
CHANNEL_SHARE = {"Online": 0.45, "Retail": 0.35, "Partner": 0.20}

#: Weekend level relative to a weekday, per channel: people shop in stores at
#: the weekend (Retail up), business partners do not order (Partner down).
WEEKEND_UPLIFT = {"Online": 1.15, "Retail": 1.40, "Partner": 0.60}

ANNUAL_GROWTH = 0.10          # linear upward trend: +10% per year
YEARLY_AMPLITUDE = 0.04       # yearly wave of +/-4% ...
FESTIVE_PEAK = "2024-11-16"   # ... peaking in mid-November (festive season)
NOISE_SD = 0.08               # day-to-day noise of a single series: 8%

#: Average order value per product - used only to derive realistic order counts.
AVG_ORDER_VALUE = {"Product A": 120.0, "Product B": 80.0, "Product C": 250.0, "Product D": 45.0}

# ---------------------------------------------------------------------------
# Design of the injected anomalies
# ---------------------------------------------------------------------------

WARMUP_DAYS = 60              # no anomaly in the first 60 days: baselines need history
MIN_GAP_DAYS = 21             # start dates at least 3 weeks apart
MAX_DURATION_DAYS = 3         # each anomaly lasts 1-3 days
ANOMALY_SIZES = (0.10, 0.20, 0.30, 0.50)
REPEATS_PER_SIZE_AND_DIRECTION = 3   # 4 sizes x 2 directions x 3 = 24 regular anomalies

#: How the 24 regular anomalies are split: 20 hit one category of one
#: dimension, 4 hit a region x product pair.
SINGLE_DIMENSION_COUNTS = {"region": 7, "product": 7, "channel": 6}
#: A region x product pair is only ~8% of revenue, so a 10% change there would be
#: invisible in every monitored series. The pairs therefore get the large sizes.
TWO_DIMENSION_CHANGES = (-0.50, -0.30, 0.30, 0.50)

NOTE_COVERAGE = 0.70          # share of regular anomalies that get a business note
DEMO_SLOT = 20                # which time slot (of 25) holds the demo case

#: The "combined" demo case: three overlapping problems on the same days. It is
#: excluded from root-cause accuracy (several true causes at once) but counted
#: in the detection metrics.
DEMO_ANOMALIES = (
    {"dimension": "region", "category": "West", "pct_change": -0.50,
     "cause": "distributor disruption"},
    {"dimension": "product", "category": "Product B", "pct_change": -0.30,
     "cause": "inventory shortage"},
    {"dimension": "channel", "category": "Retail", "pct_change": -0.20,
     "cause": "store traffic decline"},
)
DEMO_DURATION_DAYS = 3

#: Realistic causes. Channel events depend on the channel itself (a website
#: outage can only hit Online), so channels are keyed by category name.
CAUSES = {
    ("region", "decline"): ["distributor disruption", "severe weather disruption",
                            "regional logistics delays", "warehouse system outage"],
    ("region", "increase"): ["regional marketing campaign", "local festival promotion",
                             "new store openings"],
    ("product", "decline"): ["inventory shortage", "price increase", "supplier quality recall"],
    ("product", "increase"): ["marketing campaign", "limited-time discount",
                              "competitor stock-out"],
    ("Online", "decline"): ["website outage", "payment gateway failure"],
    ("Online", "increase"): ["online flash sale", "email marketing campaign"],
    ("Retail", "decline"): ["store traffic decline", "store renovation closures"],
    ("Retail", "increase"): ["in-store promotion", "festive promotion"],
    ("Partner", "decline"): ["partner system integration failure", "partner contract dispute"],
    ("Partner", "increase"): ["partner onboarding", "partner bulk order"],
    ("region+product", "decline"): ["regional distributor disruption", "local stock-out"],
    ("region+product", "increase"): ["regional product launch campaign", "local bundle promotion"],
}

ANSWER_KEY_COLUMNS = [
    "anomaly_id", "start_date", "end_date", "dimension", "category",
    "second_dimension", "second_category", "pct_change", "direction",
    "cause", "has_note", "is_demo",
]

# Chart colours (reference data-viz palette: neutral ink for the data, the
# blue/red diverging pair for the direction of an anomaly).
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS_LINE = "#c3c2b7"
DECLINE_COLOR = "#e34948"
INCREASE_COLOR = "#2a78d6"

WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------------------
# 1. Clean KPI data
# ---------------------------------------------------------------------------


def generate_kpi_data(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Create the anomaly-free daily KPI table.

    One row per date x region x product x channel. Every factor is a multiplier
    around 1, so each effect can be switched off or explained on its own.

    Args:
        dates: Every day of the dataset.
        rng: Random generator used for the noise and the order counts.

    Returns:
        DataFrame with columns date, region, product, channel, revenue, orders.
    """
    segments = pd.MultiIndex.from_product(
        [REGIONS, PRODUCTS, CHANNELS], names=["region", "product", "channel"]
    ).to_frame(index=False)
    kpi = pd.DataFrame({"date": dates}).merge(segments, how="cross")

    # Base level: how big this series is on an average day.
    base = (
        TOTAL_BASE_REVENUE
        * kpi["region"].map(REGION_SHARE).to_numpy()
        * kpi["product"].map(PRODUCT_SHARE).to_numpy()
        * kpi["channel"].map(CHANNEL_SHARE).to_numpy()
    )

    # Weekly pattern. Weekday and weekend levels are rescaled so that the weekly
    # average stays exactly 1:  5 x level + 2 x level x uplift = 7.
    uplift = kpi["channel"].map(WEEKEND_UPLIFT).to_numpy()
    weekday_level = 7.0 / (5.0 + 2.0 * uplift)
    is_weekend = kpi["date"].dt.dayofweek.to_numpy() >= 5
    weekly = np.where(is_weekend, weekday_level * uplift, weekday_level)

    # Trend and yearly wave are smooth functions of "days since the first date".
    # The wave is a slow cosine, so it can never create a sudden spike that the
    # detector would (wrongly) report as an anomaly.
    days = (kpi["date"] - dates[0]).dt.days.to_numpy()
    trend = 1.0 + ANNUAL_GROWTH * days / 365.0
    peak_day = (pd.Timestamp(FESTIVE_PEAK) - dates[0]).days
    yearly = 1.0 + YEARLY_AMPLITUDE * np.cos(2 * np.pi * (days - peak_day) / 365.25)

    noise = rng.normal(loc=1.0, scale=NOISE_SD, size=len(kpi))

    revenue = np.clip(base * weekly * trend * yearly * noise, 0.0, None)
    # Orders follow revenue: a Poisson count around revenue / average order value.
    orders = rng.poisson(revenue / kpi["product"].map(AVG_ORDER_VALUE).to_numpy())

    kpi["revenue"] = revenue
    kpi["orders"] = orders
    return kpi


# ---------------------------------------------------------------------------
# 2. Answer key: which anomalies to plant
# ---------------------------------------------------------------------------


def _slot_start_days(n_slots: int, n_days: int, rng: np.random.Generator) -> np.ndarray:
    """Spread `n_slots` start days evenly over the usable range, then wiggle each.

    Even spacing (~28 days here) guarantees the 21-day minimum gap; the +/-2 day
    jitter stops the anomalies from sitting on a perfectly regular grid.
    """
    jitter = 2
    first = WARMUP_DAYS + jitter                    # earliest start after jitter: day 60
    last = n_days - MAX_DURATION_DAYS - jitter      # a 3-day anomaly still ends in range
    evenly_spaced = np.linspace(first, last, n_slots).round().astype(int)
    return evenly_spaced + rng.integers(-jitter, jitter + 1, size=n_slots)


def _cause_key(dimension: str, category: str, second_dimension: str) -> str:
    """Which cause list applies: channels by name, everything else by dimension."""
    if second_dimension:
        return "region+product"
    if dimension == "channel":
        return category
    return dimension


def build_anomaly_schedule(dates: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Design the anomalies to inject (the answer key). Does not touch the data.

    The design is balanced on purpose so that the evaluation can report recall
    per size and per direction: every size x direction combination appears
    exactly three times among the 24 regular anomalies.

    Args:
        dates: Every day of the dataset.
        rng: Random generator used for segments, dates, causes and notes.

    Returns:
        DataFrame with the columns in ANSWER_KEY_COLUMNS, sorted by start date.
    """
    # 1. Balanced sizes: -10%, +10%, -20%, ... each three times (24 values).
    changes = [sign * size for size in ANOMALY_SIZES for sign in (-1, 1)]
    changes = changes * REPEATS_PER_SIZE_AND_DIRECTION
    for change in TWO_DIMENSION_CHANGES:            # reserved for region x product pairs
        changes.remove(change)
    if len(changes) != sum(SINGLE_DIMENSION_COUNTS.values()):
        raise ValueError("SINGLE_DIMENSION_COUNTS must add up to the single-dimension anomalies.")

    # 2. Single-dimension anomalies. Categories are cycled in a shuffled order so
    #    every region, product and channel is hit at least once.
    dimension_list = [dim for dim, count in SINGLE_DIMENSION_COUNTS.items() for _ in range(count)]
    category_cycle = {
        dim: iter(np.resize(rng.permutation(DIMENSIONS[dim]), count))
        for dim, count in SINGLE_DIMENSION_COUNTS.items()
    }
    specs = []
    for dimension, change in zip(rng.permutation(dimension_list), rng.permutation(changes)):
        dimension = str(dimension)
        specs.append({
            "dimension": dimension,
            "category": str(next(category_cycle[dimension])),
            "second_dimension": "",
            "second_category": "",
            "pct_change": float(change),
        })

    # 3. Two-dimension anomalies: one region x one product.
    for change in TWO_DIMENSION_CHANGES:
        specs.append({
            "dimension": "region",
            "category": str(rng.choice(REGIONS)),
            "second_dimension": "product",
            "second_category": str(rng.choice(PRODUCTS)),
            "pct_change": change,
        })

    # 4. Shuffle so the two-dimension anomalies are not all at the end, then pick
    #    durations and which anomalies will get a business note (Phase 7).
    specs = [specs[i] for i in rng.permutation(len(specs))]
    durations = rng.integers(1, MAX_DURATION_DAYS + 1, size=len(specs))
    has_note = np.zeros(len(specs), dtype=bool)
    has_note[rng.choice(len(specs), size=round(NOTE_COVERAGE * len(specs)), replace=False)] = True

    # 5. Put everything on the calendar: one slot per regular anomaly plus one
    #    slot for the whole demo cluster.
    n_slots = len(specs) + 1
    slot_days = _slot_start_days(n_slots, len(dates), rng)
    regular_slots = [slot for slot in range(n_slots) if slot != DEMO_SLOT]

    rows = []
    for spec, slot, duration, note in zip(specs, regular_slots, durations, has_note):
        start = dates[slot_days[slot]]
        direction = "decline" if spec["pct_change"] < 0 else "increase"
        key = _cause_key(spec["dimension"], spec["category"], spec["second_dimension"])
        rows.append({
            **spec,
            "start_date": start,
            "end_date": start + pd.Timedelta(days=int(duration) - 1),
            "direction": direction,
            "cause": str(rng.choice(CAUSES[(key, direction)])),
            "has_note": bool(note),
            "is_demo": False,
        })

    demo_start = dates[slot_days[DEMO_SLOT]]
    for demo in DEMO_ANOMALIES:
        rows.append({
            **demo,
            "second_dimension": "",
            "second_category": "",
            "start_date": demo_start,
            "end_date": demo_start + pd.Timedelta(days=DEMO_DURATION_DAYS - 1),
            "direction": "decline" if demo["pct_change"] < 0 else "increase",
            "has_note": True,
            "is_demo": True,
        })

    anomalies = pd.DataFrame(rows).sort_values("start_date", kind="stable").reset_index(drop=True)
    anomalies.insert(0, "anomaly_id", [f"A-{i:03d}" for i in range(1, len(anomalies) + 1)])
    return anomalies[ANSWER_KEY_COLUMNS]


# ---------------------------------------------------------------------------
# 3. Plant the anomalies
# ---------------------------------------------------------------------------


def apply_anomalies(kpi: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    """Multiply revenue (and orders) of the matching rows by (1 + pct_change).

    Overlapping anomalies compound: a West / Product B / Retail row inside the
    demo period is multiplied by all three factors (0.5 x 0.7 x 0.8).
    """
    kpi = kpi.copy()
    for anomaly in anomalies.itertuples(index=False):
        mask = kpi["date"].between(anomaly.start_date, anomaly.end_date)
        mask &= kpi[anomaly.dimension] == anomaly.category
        if anomaly.second_dimension:
            mask &= kpi[anomaly.second_dimension] == anomaly.second_category

        factor = 1.0 + anomaly.pct_change
        kpi.loc[mask, "revenue"] = kpi.loc[mask, "revenue"] * factor
        kpi.loc[mask, "orders"] = (kpi.loc[mask, "orders"] * factor).round().astype("int64")
    return kpi


# ---------------------------------------------------------------------------
# 4. Check the result against the specification
# ---------------------------------------------------------------------------


def _cluster_start_dates(anomalies: pd.DataFrame) -> pd.Series:
    """Start date of every anomaly, with the overlapping demo trio counted once."""
    regular = anomalies.loc[~anomalies["is_demo"], "start_date"]
    demo = anomalies.loc[anomalies["is_demo"], "start_date"].head(1)
    return pd.concat([regular, demo]).sort_values()


def validate_generated_data(
    kpi: pd.DataFrame, anomalies: pd.DataFrame, dates: pd.DatetimeIndex
) -> None:
    """Raise ValueError if the generated data breaks any rule of the specification."""
    problems: list[str] = []

    expected_rows = len(dates) * len(REGIONS) * len(PRODUCTS) * len(CHANNELS)
    if len(kpi) != expected_rows:
        problems.append(f"expected {expected_rows} KPI rows, got {len(kpi)}")
    if kpi[["revenue", "orders"]].isna().any().any():
        problems.append("KPI data contains missing values")
    if (kpi["revenue"] < 0).any() or (kpi["orders"] < 0).any():
        problems.append("KPI data contains negative values")

    if not 20 <= len(anomalies) <= 30:
        problems.append(f"expected 20-30 anomalies, got {len(anomalies)}")
    if (anomalies["start_date"] < dates[0] + pd.Timedelta(days=WARMUP_DAYS)).any():
        problems.append("an anomaly starts inside the warm-up period")
    if (anomalies["end_date"] > dates[-1]).any():
        problems.append("an anomaly ends after the last date")
    durations = (anomalies["end_date"] - anomalies["start_date"]).dt.days + 1
    if not durations.between(1, MAX_DURATION_DAYS).all():
        problems.append("an anomaly lasts outside 1-3 days")
    if not anomalies["pct_change"].abs().round(2).isin(ANOMALY_SIZES).all():
        problems.append("an anomaly size is not one of 10/20/30/50%")
    if set(anomalies["direction"]) != {"decline", "increase"}:
        problems.append("anomalies must include both declines and increases")

    min_gap = _cluster_start_dates(anomalies).diff().dt.days.min()
    if min_gap < MIN_GAP_DAYS:
        problems.append(f"two anomalies start only {min_gap:.0f} days apart")

    if problems:
        raise ValueError("Generated data breaks the specification:\n  - " + "\n  - ".join(problems))


# ---------------------------------------------------------------------------
# 5. Human-readable output
# ---------------------------------------------------------------------------


def _millions(value: float) -> str:
    return f"{value / 1e6:.2f}M"


def _segment_labels(anomalies: pd.DataFrame) -> pd.Series:
    """'region=West' or 'region=West & product=Product B'."""
    first = anomalies["dimension"] + "=" + anomalies["category"]
    second = anomalies["second_dimension"] + "=" + anomalies["second_category"]
    return first.where(anomalies["second_dimension"] == "", first + " & " + second)


def print_summary(kpi: pd.DataFrame, anomalies: pd.DataFrame) -> None:
    """Print the numbers needed to sanity-check the generated data by eye."""
    daily = kpi.groupby("date")["revenue"].sum()
    n_series = kpi.groupby(["region", "product", "channel"]).ngroups

    print("KPI data")
    print(f"  file                 : {KPI_DATA_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  rows                 : {len(kpi):,} ({daily.size} days x {n_series} series)")
    print(f"  date range           : {daily.index.min():%Y-%m-%d} -> {daily.index.max():%Y-%m-%d}")
    print(f"  daily total revenue  : mean {_millions(daily.mean())}, "
          f"min {_millions(daily.min())}, max {_millions(daily.max())}")

    weekend = daily[daily.index.dayofweek >= 5].mean()
    weekday = daily[daily.index.dayofweek < 5].mean()
    print(f"  weekend vs weekday   : {weekend / weekday - 1:+.1%}")

    years = daily.index.year
    first_year, last_year = years.min(), years.max()
    growth = daily[years == last_year].mean() / daily[years == first_year].mean() - 1
    print(f"  {last_year} vs {first_year}         : {growth:+.1%} average daily revenue "
          f"(designed trend +{ANNUAL_GROWTH:.0%} per year)")

    print("\n  Average daily total revenue by weekday")
    by_weekday = daily.groupby(daily.index.day_name()).mean().reindex(WEEKDAY_NAMES)
    for day, value in by_weekday.items():
        print(f"    {day:<10}{_millions(value):>8}")

    print("\n  Average daily total revenue by quarter")
    by_quarter = daily.groupby([daily.index.year, daily.index.quarter]).mean().unstack()
    print("          " + "".join(f"{f'Q{q}':>8}" for q in by_quarter.columns))
    for year, row in by_quarter.iterrows():
        print(f"    {year}  " + "".join(f"{_millions(v):>8}" for v in row))

    regular = anomalies[~anomalies["is_demo"]]
    sizes = regular["pct_change"].abs().mul(100).round().astype(int).value_counts().sort_index()
    directions = regular["direction"].value_counts().sort_index()
    kinds = regular["dimension"].where(
        regular["second_dimension"] == "",
        regular["dimension"] + " x " + regular["second_dimension"],
    ).value_counts()
    min_gap = _cluster_start_dates(anomalies).diff().dt.days.min()

    print("\nInjected anomalies (answer key)")
    print(f"  file                 : {INJECTED_ANOMALIES_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  total                : {len(anomalies)} "
          f"({len(regular)} regular + {int(anomalies['is_demo'].sum())} in the demo case)")
    print("  regular by size      : " + ", ".join(f"{s}%: {c}" for s, c in sizes.items()))
    print("  regular by direction : " + ", ".join(f"{d}: {c}" for d, c in directions.items()))
    print("  regular by segment   : " + ", ".join(f"{k}: {c}" for k, c in kinds.items()))
    print(f"  with business note   : {int(anomalies['has_note'].sum())} of {len(anomalies)} "
          f"({anomalies['has_note'].mean():.0%})")
    print(f"  closest start dates  : {min_gap:.0f} days apart (minimum allowed {MIN_GAP_DAYS})")

    table = pd.DataFrame({
        "id": anomalies["anomaly_id"],
        "start": anomalies["start_date"].dt.strftime("%Y-%m-%d"),
        "days": (anomalies["end_date"] - anomalies["start_date"]).dt.days + 1,
        "segment": _segment_labels(anomalies),
        "change": anomalies["pct_change"].map(lambda v: f"{v:+.0%}"),
        "cause": anomalies["cause"],
        "note": anomalies["has_note"].map({True: "yes", False: "-"}),
        "demo": anomalies["is_demo"].map({True: "demo", False: ""}),
    })
    print()
    print(table.to_string(index=False))


def plot_daily_revenue(kpi: pd.DataFrame, anomalies: pd.DataFrame, path: Path) -> Path:
    """Save an interactive chart of daily total revenue with the anomalies marked.

    Small anomalies (for example -10% in one region, about -2.5% of the total)
    are barely visible here. That is expected, and it is exactly why Phase 5
    also runs detection on every region, product and channel separately.
    """
    import plotly.graph_objects as go  # only needed with --plot

    daily = kpi.groupby("date")["revenue"].sum()
    rolling_avg = daily.rolling(28, center=True, min_periods=14).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily.to_numpy(), mode="lines", name="Daily total revenue",
        line={"color": INK_MUTED, "width": 1},
        hovertemplate="%{x|%a %Y-%m-%d}<br>Revenue %{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=rolling_avg.index, y=rolling_avg.to_numpy(), mode="lines", name="28-day average",
        line={"color": INK_PRIMARY, "width": 2},
        hovertemplate="%{x|%Y-%m-%d}<br>28-day average %{y:,.0f}<extra></extra>",
    ))

    # Direction is shown by colour AND marker shape, so it never relies on colour alone.
    for direction, color, symbol, label in (
        ("decline", DECLINE_COLOR, "triangle-down", "Injected decline"),
        ("increase", INCREASE_COLOR, "triangle-up", "Injected increase"),
    ):
        subset = anomalies[anomalies["direction"] == direction]
        for anomaly in subset.itertuples(index=False):
            fig.add_vrect(
                x0=str(anomaly.start_date - pd.Timedelta(hours=12)),
                x1=str(anomaly.end_date + pd.Timedelta(hours=12)),
                fillcolor=color, opacity=0.15, line_width=0, layer="below",
            )
        details = np.column_stack([
            subset["anomaly_id"],
            _segment_labels(subset),
            subset["pct_change"].map(lambda v: f"{v:+.0%}"),
            subset["cause"],
            (subset["end_date"] - subset["start_date"]).dt.days + 1,
        ])
        fig.add_trace(go.Scatter(
            x=subset["start_date"], y=daily.reindex(subset["start_date"]).to_numpy(),
            mode="markers", name=label, customdata=details,
            marker={"color": color, "symbol": symbol, "size": 10,
                    "line": {"color": SURFACE, "width": 2}},
            hovertemplate=("<b>%{customdata[0]}</b> %{customdata[1]} %{customdata[2]}"
                           "<br>%{customdata[3]}<br>starts %{x|%Y-%m-%d}, "
                           "%{customdata[4]} day(s)<extra></extra>"),
        ))

    demo = anomalies[anomalies["is_demo"]]
    if not demo.empty:
        fig.add_annotation(
            x=str(demo["start_date"].iloc[0]), y=1, yref="paper", yanchor="bottom",
            text="Demo case", showarrow=False, font={"color": INK_SECONDARY, "size": 11},
        )

    fig.update_layout(
        title={"text": "Daily total revenue with injected anomalies", "x": 0.01,
               "font": {"color": INK_PRIMARY, "size": 16}},
        font={"family": "system-ui, -apple-system, Segoe UI, sans-serif",
              "color": INK_SECONDARY, "size": 12},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        hovermode="closest",
        legend={"orientation": "h", "x": 0, "y": -0.12},
        margin={"l": 60, "r": 20, "t": 70, "b": 70},
        xaxis={"showgrid": False, "linecolor": AXIS_LINE},
        yaxis={"title": "Revenue", "tickformat": "~s", "gridcolor": GRIDLINE,
               "zeroline": False},
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    # Embed plotly.js in the file (~5 MB) instead of loading it from a CDN, so the
    # chart opens offline and cannot break on a blocked or failed download. The
    # file is git-ignored, so its size does not matter.
    fig.write_html(str(path), include_plotlyjs=True)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic KPI dataset and the injected-anomaly answer key."
    )
    parser.add_argument("--plot", action="store_true",
                        help="also save an interactive chart to reports/data_overview.html")
    args = parser.parse_args(argv)

    # One master seed split into independent random streams: changing how the
    # anomalies are designed can never change the KPI noise, and vice versa.
    kpi_stream, anomaly_stream = np.random.SeedSequence(RANDOM_SEED).spawn(2)

    dates = pd.date_range(START_DATE, END_DATE, freq="D")
    clean_kpi = generate_kpi_data(dates, np.random.default_rng(kpi_stream))
    anomalies = build_anomaly_schedule(dates, np.random.default_rng(anomaly_stream))

    kpi = apply_anomalies(clean_kpi, anomalies)
    kpi["revenue"] = kpi["revenue"].round(2)
    validate_generated_data(kpi, anomalies, dates)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    kpi.to_csv(KPI_DATA_PATH, index=False, date_format="%Y-%m-%d")
    anomalies.to_csv(INJECTED_ANOMALIES_PATH, index=False, date_format="%Y-%m-%d")

    print_summary(kpi, anomalies)
    if args.plot:
        saved = plot_daily_revenue(kpi, anomalies, REPORTS_DIR / "data_overview.html")
        print(f"\nChart saved to {saved.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
