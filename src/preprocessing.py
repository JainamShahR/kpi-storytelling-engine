"""Load, validate and clean the daily KPI data.

WHAT
    Turns the raw CSV into a clean, complete table in which every
    region x product x channel series has exactly one value per day, and
    produces daily KPI series for the whole company or for one segment.

WHY
    Every later stage assumes continuous daily series: the baseline compares a
    day with the same weekday 7/14/21/28 days earlier, which silently breaks if
    a day is missing or counted twice. Bad data must therefore be caught here,
    loudly, before it turns into fake anomalies.

HOW
    1. `load_data` reads the CSV (keys as text, nothing else changed).
    2. `validate_data` inspects the data WITHOUT changing it and returns a
       `ValidationReport`: errors (data unusable), warnings (a documented fix
       will be applied) and counts.
    3. `clean_data` refuses data that has errors, then applies the fixes.
    4. `aggregate_kpi` sums one KPI per day for the company or one segment.

STRATEGY FOR BAD DATA - rule: never silently invent or discard numbers
    Problem                                Decision  Why
    missing column / no rows               error     nothing to analyse
    unparsable date                        error     the row cannot be placed in time
    unknown region / product / channel     error     dropping it would understate totals
    missing, non-numeric or negative KPI   error     filling 0 would invent a -100% drop
    same key twice with different values   error     the correct value is unknown
    exact duplicate row                    warning   a double load: drop the extra copy
    empty date / region / product / channel warning  cannot be placed in a series: dropped
    date x segment combination absent      warning   no row = no sales: filled with 0
    whole day absent                       warning   filled with 0 as specified, but it will
                                                     look like a -100% drop, so it is flagged

ASSUMPTIONS
    - One row per date x region x product x channel; dates are YYYY-MM-DD.
    - An absent combination means "no sales that day", not "data was lost".

LIMITATIONS
    - Cannot tell a real zero-sales day from a lost file; whole missing days are
      flagged but still filled.
    - Unusual values are not removed here: finding them is the job of anomaly
      detection, not of cleaning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from src.config import (
    CHANNELS,
    DEFAULT_KPI,
    DIMENSIONS,
    KPI_DATA_PATH,
    PRODUCTS,
    REGIONS,
    SUPPORTED_KPIS,
)

logger = logging.getLogger(__name__)

DATE_FORMAT = "%Y-%m-%d"
KEY_COLUMNS = ["date", "region", "product", "channel"]
REQUIRED_COLUMNS = KEY_COLUMNS + list(SUPPORTED_KPIS)
N_SEGMENTS = len(REGIONS) * len(PRODUCTS) * len(CHANNELS)


class DataValidationError(Exception):
    """Raised when the KPI data cannot be used safely.

    Carries the full `ValidationReport` (when there is one) so the app can show
    every problem at once instead of only the first.
    """

    def __init__(self, message: str, report: ValidationReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class ValidationReport:
    """Result of `validate_data`.

    errors   - problems that make the data unusable; `clean_data` refuses it.
    warnings - problems with a documented, safe fix that `clean_data` applies.
    counts   - how many rows each check found, for display in the app.
    """

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def raise_if_invalid(self) -> None:
        if self.errors:
            raise DataValidationError(
                "KPI data failed validation:\n- " + "\n- ".join(self.errors), report=self
            )

    def summary(self) -> str:
        """Readable multi-line text: status line, errors, warnings, counts."""
        status = "passed" if self.is_valid else "FAILED"
        lines = [f"Validation {status}: {len(self.errors)} error(s), "
                 f"{len(self.warnings)} warning(s)"]
        lines += [f"  ERROR    {message}" for message in self.errors]
        lines += [f"  WARNING  {message}" for message in self.warnings]
        lines += [f"  {name:<24}{value:>10,}" for name, value in self.counts.items()]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_data(path: str | Path = KPI_DATA_PATH) -> pd.DataFrame:
    """Read the KPI CSV exactly as stored; validation and cleaning happen later.

    Key columns are read as text so that, for example, a malformed date stays
    visible to `validate_data` instead of being silently converted.

    Raises:
        FileNotFoundError: the file does not exist (message says how to create it).
        DataValidationError: the file is empty or not a readable CSV.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"KPI data file not found: {path}. Create it with: python generate_data.py"
        )
    try:
        df = pd.read_csv(path, dtype={column: str for column in KEY_COLUMNS})
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"KPI data file is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise DataValidationError(f"KPI data file is not a valid CSV: {path} ({exc})") from exc

    logger.info("Loaded %d rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_data(df: pd.DataFrame) -> ValidationReport:
    """Check the raw data against every rule WITHOUT modifying it.

    Args:
        df: Raw KPI data, e.g. from `load_data`.

    Returns:
        A `ValidationReport`; `report.is_valid` is False if anything is an error.
    """
    report = ValidationReport()
    report.counts["rows"] = len(df)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        report.errors.append(f"Missing required column(s): {', '.join(missing_columns)}.")
        return report  # every other check needs these columns
    if df.empty:
        report.errors.append("The dataset contains no rows.")
        return report

    dates = pd.to_datetime(df["date"], format=DATE_FORMAT, errors="coerce")
    work = df.assign(date=dates)

    # 1. Keys: rows that cannot be placed in any series.
    missing_key = df[KEY_COLUMNS].isna().any(axis=1)
    report.counts["missing_keys"] = int(missing_key.sum())
    if missing_key.any():
        report.warnings.append(
            f"{int(missing_key.sum())} row(s) have an empty date, region, product or "
            "channel; they cannot be placed in any series and will be dropped."
        )

    unparsable = dates.isna() & df["date"].notna()
    report.counts["unparsable_dates"] = int(unparsable.sum())
    if unparsable.any():
        examples = ", ".join(map(str, df.loc[unparsable, "date"].unique()[:3]))
        report.errors.append(
            f"{int(unparsable.sum())} row(s) have a date that is not YYYY-MM-DD "
            f"(e.g. {examples})."
        )

    # 2. Categories must come from the known lists in src/config.py.
    invalid_category = pd.Series(False, index=df.index)
    for dimension, allowed in DIMENSIONS.items():
        column = df[dimension]
        invalid = column.notna() & ~column.isin(allowed)
        if invalid.any():
            unknown = ", ".join(sorted(column[invalid].astype(str).unique()))
            report.errors.append(
                f"{int(invalid.sum())} row(s) have an unknown {dimension}: {unknown} "
                f"(allowed: {', '.join(allowed)})."
            )
        invalid_category |= invalid
    report.counts["invalid_categories"] = int(invalid_category.sum())

    # 3. KPI values must be present, numeric and not negative.
    for kpi in SUPPORTED_KPIS:
        raw = df[kpi]
        numeric = pd.to_numeric(raw, errors="coerce")
        missing = raw.isna()
        non_numeric = numeric.isna() & ~missing
        negative = numeric < 0
        report.counts[f"missing_{kpi}"] = int(missing.sum())
        report.counts[f"non_numeric_{kpi}"] = int(non_numeric.sum())
        report.counts[f"negative_{kpi}"] = int(negative.sum())
        if missing.any():
            report.errors.append(
                f"{int(missing.sum())} row(s) have no {kpi} value. Filling it with 0 "
                "would invent a -100% drop, so the source data must be fixed."
            )
        if non_numeric.any():
            examples = ", ".join(map(str, raw[non_numeric].unique()[:3]))
            report.errors.append(
                f"{int(non_numeric.sum())} row(s) have a non-numeric {kpi} (e.g. {examples})."
            )
        if negative.any():
            report.errors.append(f"{int(negative.sum())} row(s) have a negative {kpi}.")

    # 4. Duplicates. Only rows with a complete, parsable key can be compared.
    keyed = work.loc[~missing_key & work["date"].notna()]
    exact_copy = keyed.duplicated(keep="first")
    report.counts["exact_duplicates"] = int(exact_copy.sum())
    if exact_copy.any():
        report.warnings.append(
            f"{int(exact_copy.sum())} row(s) are exact copies of another row "
            "(probably loaded twice); the extra copies will be dropped."
        )
    distinct = keyed.loc[~exact_copy]
    clash = distinct.duplicated(subset=KEY_COLUMNS, keep=False)
    clashing_keys = distinct.loc[clash, KEY_COLUMNS].drop_duplicates()
    report.counts["conflicting_duplicates"] = len(clashing_keys)
    if len(clashing_keys):
        first = clashing_keys.iloc[0]
        report.errors.append(
            f"{len(clashing_keys)} date/segment combination(s) appear more than once "
            "with different values, so the correct value is unknown (e.g. "
            f"{first['date'].strftime(DATE_FORMAT)} / {first['region']} / "
            f"{first['product']} / {first['channel']})."
        )

    # 5. Gaps: every date in the range should have all 48 segments.
    usable = keyed.loc[~invalid_category.loc[keyed.index]].drop_duplicates(subset=KEY_COLUMNS)
    if usable.empty:
        report.errors.append("No usable rows remain after removing rows with missing keys.")
        return report
    n_days = (usable["date"].max() - usable["date"].min()).days + 1
    missing_combinations = n_days * N_SEGMENTS - len(usable)
    missing_days = n_days - usable["date"].nunique()
    report.counts["days"] = n_days
    report.counts["missing_combinations"] = missing_combinations
    report.counts["missing_days"] = missing_days
    if missing_combinations:
        report.warnings.append(
            f"{missing_combinations} date/segment combination(s) have no row; "
            "they will be filled with 0 (no sales recorded)."
        )
    if missing_days:
        report.warnings.append(
            f"{missing_days} whole day(s) have no data at all. They will be filled "
            "with 0 and will look like -100% drops - check the data source."
        )
    return report


# ---------------------------------------------------------------------------
# Cleaning and aggregation
# ---------------------------------------------------------------------------


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, then return a complete and continuous KPI table.

    Steps (only reached if validation found no errors):
        parse dates -> drop rows with missing keys -> drop exact duplicate rows
        -> make KPI columns float -> add every missing date x segment with 0.

    Returns:
        One row per date x region x product x channel over the full date range,
        sorted by date, region, product, channel; `date` is a datetime column.

    Raises:
        DataValidationError: if `validate_data` reports any error.
    """
    report = validate_data(df)
    report.raise_if_invalid()
    for message in report.warnings:
        logger.warning(message)

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], format=DATE_FORMAT)
    work = work.dropna(subset=KEY_COLUMNS).drop_duplicates()
    for kpi in SUPPORTED_KPIS:
        work[kpi] = pd.to_numeric(work[kpi]).astype(float)

    # The complete grid: every day in the range x all 48 segments. `reindex`
    # keeps existing rows and creates the absent ones with fill_value=0 (it never
    # touches values that were already present).
    full_grid = pd.MultiIndex.from_product(
        [pd.date_range(work["date"].min(), work["date"].max(), freq="D"),
         REGIONS, PRODUCTS, CHANNELS],
        names=KEY_COLUMNS,
    )
    cleaned = work.set_index(KEY_COLUMNS).reindex(full_grid, fill_value=0).reset_index()
    logger.info("Clean data: %d rows (%d days x %d segments)",
                len(cleaned), len(cleaned) // N_SEGMENTS, N_SEGMENTS)
    return cleaned


def aggregate_kpi(
    df: pd.DataFrame,
    kpi: str = DEFAULT_KPI,
    dimension: str | None = None,
    category: str | None = None,
) -> pd.Series:
    """Daily total of one KPI for the whole company or for one segment.

    Examples:
        aggregate_kpi(df)                                        # company total
        aggregate_kpi(df, dimension="region", category="West")   # West only

    Args:
        df: Output of `clean_data`.
        kpi: KPI column to sum (must be in SUPPORTED_KPIS).
        dimension, category: Both given -> one segment; both None -> company.

    Returns:
        Series indexed by every date in the data (no gaps), named after the KPI.

    Raises:
        ValueError: unsupported KPI, unknown dimension/category, only one of
            dimension/category given, or data that was not cleaned first.
        KeyError: the KPI column is missing from `df`.
    """
    if kpi not in SUPPORTED_KPIS:
        raise ValueError(f"Unsupported KPI {kpi!r}. Supported: {', '.join(SUPPORTED_KPIS)}.")
    if kpi not in df.columns:
        raise KeyError(f"Column {kpi!r} not found in the data.")
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        raise ValueError("'date' is not a datetime column - call clean_data() first.")
    if (dimension is None) != (category is None):
        raise ValueError("Pass both dimension and category, or neither.")

    rows = df
    if dimension is not None:
        if dimension not in DIMENSIONS:
            raise ValueError(
                f"Unknown dimension {dimension!r}. Expected one of: {', '.join(DIMENSIONS)}."
            )
        if category not in DIMENSIONS[dimension]:
            raise ValueError(
                f"Unknown category {category!r} for {dimension}. "
                f"Expected one of: {', '.join(DIMENSIONS[dimension])}."
            )
        rows = df[df[dimension] == category]

    series = rows.groupby("date")[kpi].sum()
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    series = series.reindex(full_range, fill_value=0.0)
    series.index.name = "date"
    series.name = kpi
    return series


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    raw_data = load_data()
    print("=== Generated data ===")
    print(validate_data(raw_data).summary())

    kpi_data = clean_data(raw_data)
    company = aggregate_kpi(kpi_data)
    west = aggregate_kpi(kpi_data, dimension="region", category="West")
    print(f"\nClean table   : {len(kpi_data):,} rows, "
          f"{kpi_data['date'].min():%Y-%m-%d} -> {kpi_data['date'].max():%Y-%m-%d}")
    print(f"Company total : {len(company)} daily values, first day {company.iloc[0]:,.2f}")
    print(f"West region   : {len(west)} daily values, first day {west.iloc[0]:,.2f}")

    # The same data with four deliberate problems, to show the report on bad data.
    broken = pd.concat([raw_data, raw_data.iloc[[0]].assign(revenue=1.0)], ignore_index=True)
    broken.loc[1, "region"] = "Nort"       # typo in a category
    broken.loc[2, "revenue"] = -500.0      # negative revenue
    broken = broken.drop(index=3)          # one missing combination
    print("\n=== Same data with 4 deliberate problems ===")
    print(validate_data(broken).summary())
    try:
        clean_data(broken)
    except DataValidationError as error:
        print(f"\nclean_data refused the broken data ({len(error.report.errors)} errors).")
