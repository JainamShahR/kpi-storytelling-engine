"""Tests for `src.decomposition` on the `decomposition_df` fixture.

Every day of the fixture is identical, so each category's baseline is simply
its daily value (total 1,200; each region 300; Product A 480 ... Product D 120)
and every expected contribution can be computed by hand.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.config import Settings
from src.decomposition import cross_dimension, decompose, top_contributors

DAY = pd.Timestamp("2024-01-29")


def _scale(df: pd.DataFrame, factor: float, **segment: str) -> None:
    """Multiply revenue on DAY for the rows of one segment, e.g. region="West"."""
    rows = df["date"] == DAY
    for column, value in segment.items():
        rows &= df[column] == value
    df.loc[rows, "revenue"] *= factor


def test_contributions_sum_to_100_within_each_dimension(
    decomposition_df: pd.DataFrame, default_settings: Settings
) -> None:
    """Spec test 8, with two overlapping changes (West -40%, Product B -30%)."""
    _scale(decomposition_df, 0.6, region="West")
    _scale(decomposition_df, 0.7, product="Product B")

    result = decompose(decomposition_df, DAY, default_settings)

    assert not result.skipped
    assert set(result.by_dimension) == {"region", "product", "channel"}
    for table in result.by_dimension.values():
        assert table["contribution_pct"].sum() == pytest.approx(100.0)
        assert table["baseline"].sum() == pytest.approx(result.baseline)
        assert table["absolute_change"].sum() == pytest.approx(result.absolute_change)


def test_planted_west_drop_makes_west_the_top_region(
    decomposition_df: pd.DataFrame, default_settings: Settings
) -> None:
    """Spec test 9: West -40% is -120 of the 1,200 total (-10%)."""
    _scale(decomposition_df, 0.6, region="West")

    result = decompose(decomposition_df, DAY, default_settings)
    top_region = result.by_dimension["region"].iloc[0]

    assert result.pct_change == pytest.approx(-10.0)
    assert top_region["category"] == "West"
    assert top_region["absolute_change"] == pytest.approx(-120.0)
    assert top_region["pct_change"] == pytest.approx(-40.0)
    assert top_region["contribution_pct"] == pytest.approx(100.0)
    # West explains 100% of its dimension - the most concentrated explanation.
    assert top_contributors(result, n=1)[0]["category"] == "West"


def test_ranking_uses_contribution_not_percentage_change(
    decomposition_df: pd.DataFrame, default_settings: Settings
) -> None:
    """Spec test 10: small Product D at -50% (-60) must rank below large
    Product A at -20% (-96), even though D's percentage change is bigger."""
    _scale(decomposition_df, 0.5, product="Product D")
    _scale(decomposition_df, 0.8, product="Product A")

    products = decompose(decomposition_df, DAY, default_settings).by_dimension["product"]
    first, second = products.iloc[0], products.iloc[1]

    assert (first["category"], second["category"]) == ("Product A", "Product D")
    assert first["pct_change"] == pytest.approx(-20.0)
    assert second["pct_change"] == pytest.approx(-50.0)
    assert first["contribution_pct"] == pytest.approx(96 / 156 * 100)
    assert second["contribution_pct"] == pytest.approx(60 / 156 * 100)


def test_tiny_total_change_skips_decomposition(
    decomposition_df: pd.DataFrame, default_settings: Settings
) -> None:
    """A +1% total change is below 2%: no unstable percentages, a clear message."""
    _scale(decomposition_df, 1.01)

    result = decompose(decomposition_df, DAY, default_settings)

    assert result.skipped
    assert result.pct_change == pytest.approx(1.0)
    assert "2%" in result.message
    assert top_contributors(result) == []
    assert cross_dimension(decomposition_df, DAY, default_settings).empty


def test_cross_check_finds_the_combined_segment(
    decomposition_df: pd.DataFrame, default_settings: Settings
) -> None:
    """West x Product B -50% (-45 of 1,200). The cross-check pins it down, and
    both single-dimension views claim 100% of the same change - which is why
    contributions are never added across dimensions."""
    _scale(decomposition_df, 0.5, region="West", product="Product B")

    top = cross_dimension(decomposition_df, DAY, default_settings, n=3)
    assert len(top) == 3
    assert (top.iloc[0]["region"], top.iloc[0]["product"]) == ("West", "Product B")
    assert top.iloc[0]["contribution_pct"] == pytest.approx(100.0)

    result = decompose(decomposition_df, DAY, default_settings)
    assert result.by_dimension["region"].iloc[0]["contribution_pct"] == pytest.approx(100.0)
    assert result.by_dimension["product"].iloc[0]["contribution_pct"] == pytest.approx(100.0)
