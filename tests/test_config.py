"""Tests for `src.config`.

These check behaviour, not just that the code imports: defaults must match the
documented values, bad values must be rejected with a clear error, and the
environment must actually override the defaults.
"""

from __future__ import annotations

import pytest

from src.config import (
    ConfigError,
    Settings,
    get_settings,
    load_settings,
)


def test_defaults_match_env_example(default_settings: Settings) -> None:
    """An empty environment yields exactly the defaults from `.env.example`."""
    anomaly = default_settings.anomaly
    assert (anomaly.seasonal_period_days, anomaly.baseline_weeks) == (7, 4)
    assert (anomaly.zscore_window_days, anomaly.zscore_min_periods) == (56, 28)
    assert anomaly.zscore_threshold == 3.0
    assert anomaly.min_relative_change == 0.05

    assert default_settings.retrieval.top_k == 3
    assert default_settings.retrieval.window_days == 3
    assert default_settings.retrieval.min_similarity == 0.10

    assert default_settings.confidence.z_cap == 6.0
    assert default_settings.llm.enabled is True
    assert default_settings.llm.model == "qwen3:8b"
    assert default_settings.kpi == "revenue"


def test_environment_overrides_defaults() -> None:
    """Values in the environment win over the defaults and are typed."""
    settings = load_settings(
        env={
            "ZSCORE_THRESHOLD": "2.5",
            "TOP_K": "5",
            "LLM_ENABLED": "false",
            "OLLAMA_MODEL": "qwen3:4b",
        }
    )
    assert settings.anomaly.zscore_threshold == 2.5
    assert settings.retrieval.top_k == 5
    assert settings.llm.enabled is False
    assert settings.llm.model == "qwen3:4b"


def test_confidence_weights_must_sum_to_one() -> None:
    """Weights that do not sum to 1 would break the [0, 1] guarantee."""
    with pytest.raises(ConfigError, match="sum to 1"):
        load_settings(
            env={
                "CONF_WEIGHT_ANOMALY": "0.5",
                "CONF_WEIGHT_CONTRIBUTION": "0.3",
                "CONF_WEIGHT_EVIDENCE": "0.3",
            }
        )


@pytest.mark.parametrize(
    ("env", "expected_message"),
    [
        ({"ZSCORE_THRESHOLD": "0"}, "ZSCORE_THRESHOLD"),
        ({"MIN_RELATIVE_CHANGE": "1.5"}, "MIN_RELATIVE_CHANGE"),
        ({"ZSCORE_MIN_PERIODS": "100"}, "ZSCORE_MIN_PERIODS"),
        ({"TOP_K": "0"}, "TOP_K"),
        ({"MIN_SIMILARITY": "2"}, "MIN_SIMILARITY"),
        ({"OLLAMA_URL": "localhost:11434"}, "OLLAMA_URL"),
        ({"OLLAMA_TIMEOUT_SECONDS": "0"}, "OLLAMA_TIMEOUT_SECONDS"),
        ({"KPI": "profit"}, "not supported"),
    ],
)
def test_invalid_values_are_rejected(env: dict[str, str], expected_message: str) -> None:
    """Each invalid value fails fast and names the offending key."""
    with pytest.raises(ConfigError, match=expected_message):
        load_settings(env=env)


def test_unparsable_value_reports_the_key() -> None:
    """A typo in a numeric value must not surface as a raw float() error."""
    with pytest.raises(ConfigError, match="ZSCORE_THRESHOLD must be a number"):
        load_settings(env={"ZSCORE_THRESHOLD": "three"})

    with pytest.raises(ConfigError, match="LLM_ENABLED must be a boolean"):
        load_settings(env={"LLM_ENABLED": "maybe"})


def test_settings_are_immutable(default_settings: Settings) -> None:
    """Frozen dataclasses stop a module from mutating shared configuration."""
    with pytest.raises(Exception):
        default_settings.anomaly.zscore_threshold = 1.0  # type: ignore[misc]


def test_warmup_days_matches_the_rule(default_settings: Settings) -> None:
    """Warm-up = baseline history (3 of 4 weeks = 21 days) + 28 past deviations."""
    assert default_settings.anomaly.warmup_days == 7 * (4 - 1) + 28


def test_llm_urls_are_built_from_the_base_url() -> None:
    """Endpoints are derived, so the host is configured in exactly one place."""
    settings = load_settings(env={"OLLAMA_URL": "http://127.0.0.1:1234/"})
    assert settings.llm.tags_url == "http://127.0.0.1:1234/api/tags"
    assert settings.llm.chat_url == "http://127.0.0.1:1234/api/chat"


def test_get_settings_is_cached() -> None:
    """Streamlit re-runs the script constantly; parsing once is enough."""
    get_settings.cache_clear()
    assert get_settings() is get_settings()
    get_settings.cache_clear()
