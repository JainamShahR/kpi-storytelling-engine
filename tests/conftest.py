"""Shared pytest fixtures.

Every test in this project runs on small, fixed, in-memory data defined here -
never on the generated CSV files. That keeps tests fast, deterministic and
independent of `generate_data.py`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import CHANNELS, PRODUCTS, REGIONS, Settings, load_settings


@pytest.fixture()
def default_settings() -> Settings:
    """Settings built from the documented defaults only.

    An empty environment mapping is passed on purpose so the developer's local
    `.env` can never change a test result.
    """
    return load_settings(env={})


@pytest.fixture()
def small_kpi_df() -> pd.DataFrame:
    """Raw-looking KPI data: 14 days x 48 segments = 672 rows, revenue 100.0 each.

    Dates are strings, exactly as they come out of a CSV. Constant values make
    the expected totals easy to check by hand:
        company total per day  = 48 x 100 = 4,800
        one region per day     = 4 products x 3 channels x 100 = 1,200
        one channel per day    = 4 regions x 4 products x 100  = 1,600
    """
    dates = pd.date_range("2024-01-01", periods=14, freq="D").strftime("%Y-%m-%d")
    segments = pd.MultiIndex.from_product(
        [REGIONS, PRODUCTS, CHANNELS], names=["region", "product", "channel"]
    ).to_frame(index=False)
    df = pd.DataFrame({"date": dates}).merge(segments, how="cross")
    df["revenue"] = 100.0
    df["orders"] = 1
    return df
