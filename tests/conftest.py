"""Shared pytest fixtures.

Every test in this project runs on small, fixed, in-memory data defined here -
never on the generated CSV files. That keeps tests fast, deterministic and
independent of `generate_data.py`.
"""

from __future__ import annotations

import pytest

from src.config import Settings, load_settings


@pytest.fixture()
def default_settings() -> Settings:
    """Settings built from the documented defaults only.

    An empty environment mapping is passed on purpose so the developer's local
    `.env` can never change a test result.
    """
    return load_settings(env={})
