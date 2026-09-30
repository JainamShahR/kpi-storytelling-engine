"""Central configuration for the KPI Storytelling Engine.

WHAT
    One single place where every tunable number of the project lives. Values
    are read from environment variables (optionally from a local `.env` file)
    and fall back to the documented defaults from `.env.example`.

WHY
    If thresholds were hard-coded inside the analysis modules, changing the
    z-score threshold would mean editing several files and re-reading the code
    to find out what the "real" value is. Reading them once, validating them
    once, and passing an immutable `Settings` object around keeps the analysis
    modules free of magic numbers and makes every run reproducible: the numbers
    that produced a result can be printed from a single object.

HOW
    `load_settings()` reads the environment, converts each value to the right
    type with a clear error message on failure, and builds four small frozen
    dataclasses (anomaly / retrieval / confidence / llm) wrapped in `Settings`.
    Each dataclass validates itself in `__post_init__`, so an invalid
    configuration fails immediately at start-up instead of silently producing
    wrong analytics later.

ASSUMPTIONS
    - The `.env` file (if present) is for local development only and is never
      committed; `.env.example` documents the keys.
    - Environment values are plain scalars (no lists or JSON).

LIMITATIONS
    - No per-KPI overrides: v1 analyses revenue only.
    - Values are validated individually and in a few obvious combinations
      (e.g. weights summing to 1); it cannot know whether a threshold is a
      *good* choice for a given dataset - that is what the evaluation is for.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Paths and fixed business vocabulary
# --------------------------------------------------------------------------
# These are properties of the project layout and of the (synthetic) business,
# not user-tunable knobs, so they are constants rather than environment values.

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

KPI_DATA_PATH: Path = DATA_DIR / "kpi_data.csv"
BUSINESS_NOTES_PATH: Path = DATA_DIR / "business_notes.csv"
INJECTED_ANOMALIES_PATH: Path = DATA_DIR / "injected_anomalies.csv"

#: Seed used by `generate_data.py` so the dataset is byte-for-byte reproducible.
RANDOM_SEED: int = 42

#: The analysed KPI in v1. Kept as a list so a second additive KPI (orders,
#: units) could be added later purely through configuration.
SUPPORTED_KPIS: tuple[str, ...] = ("revenue",)
DEFAULT_KPI: str = "revenue"

#: Business dimensions and their allowed categories. Used by preprocessing to
#: validate the data and by decomposition to know which views to compute.
REGIONS: tuple[str, ...] = ("North", "South", "East", "West")
PRODUCTS: tuple[str, ...] = ("Product A", "Product B", "Product C", "Product D")
CHANNELS: tuple[str, ...] = ("Online", "Retail", "Partner")

DIMENSIONS: dict[str, tuple[str, ...]] = {
    "region": REGIONS,
    "product": PRODUCTS,
    "channel": CHANNELS,
}


class ConfigError(ValueError):
    """Raised when a configuration value is missing, unparsable or invalid.

    Subclasses `ValueError` so callers that only care about "bad input" can
    catch the broader type, while the app can catch `ConfigError` to show a
    configuration-specific message.
    """


# --------------------------------------------------------------------------
# Typed environment readers
# --------------------------------------------------------------------------
# Every reader reports the *key name* in its error, because a stack trace that
# only says "invalid literal for float()" is useless when 20 keys exist.


def _get_str(env: Mapping[str, str], key: str, default: str) -> str:
    value = env.get(key, default)
    value = value.strip()
    if not value:
        raise ConfigError(f"{key} must not be empty.")
    return value


def _get_int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{key} must be an integer, got {raw!r}.") from exc


def _get_float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        raise ConfigError(f"{key} must be a number, got {raw!r}.") from exc


def _get_bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None or raw.strip() == "":
        return default
    normalised = raw.strip().lower()
    if normalised in {"1", "true", "yes", "on"}:
        return True
    if normalised in {"0", "false", "no", "off"}:
        return False
    raise ConfigError(f"{key} must be a boolean (true/false), got {raw!r}.")


# --------------------------------------------------------------------------
# Settings groups
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AnomalySettings:
    """Parameters of the seasonal-naive baseline and the robust z-score rule.

    seasonal_period_days
        Length of the seasonal cycle. 7 means "compare a day with the same
        weekday", which is the dominant pattern in daily retail revenue.
    baseline_weeks
        How many previous same-weekday values form the baseline (median).
    zscore_window_days / zscore_min_periods
        Size of the trailing window of *past* deviations used to judge how
        unusual today's deviation is, and the minimum number of observations
        required before a z-score is computed at all.
    zscore_threshold
        |z| at which a day is called an anomaly.
    min_relative_change
        Second condition: the relative deviation must also be at least this
        big, so statistically unusual but commercially irrelevant wobbles
        (e.g. 0.3%) are not reported.
    """

    seasonal_period_days: int = 7
    baseline_weeks: int = 4
    zscore_window_days: int = 56
    zscore_min_periods: int = 28
    zscore_threshold: float = 3.0
    min_relative_change: float = 0.05

    def __post_init__(self) -> None:
        if self.seasonal_period_days < 1:
            raise ConfigError("SEASONAL_PERIOD_DAYS must be >= 1.")
        if self.baseline_weeks < 2:
            # With fewer than 2 reference weeks a median is meaningless and a
            # single past anomaly would fully define the "expected" value.
            raise ConfigError("BASELINE_WEEKS must be >= 2.")
        if self.zscore_window_days < 1:
            raise ConfigError("ZSCORE_WINDOW_DAYS must be >= 1.")
        if self.zscore_min_periods < 2:
            raise ConfigError("ZSCORE_MIN_PERIODS must be >= 2.")
        if self.zscore_min_periods > self.zscore_window_days:
            raise ConfigError(
                "ZSCORE_MIN_PERIODS must be <= ZSCORE_WINDOW_DAYS "
                f"(got {self.zscore_min_periods} > {self.zscore_window_days})."
            )
        if self.zscore_threshold <= 0:
            raise ConfigError("ZSCORE_THRESHOLD must be > 0.")
        if not 0.0 <= self.min_relative_change < 1.0:
            raise ConfigError("MIN_RELATIVE_CHANGE must be in [0, 1).")

    @property
    def warmup_days(self) -> int:
        """Days of history needed before any day can be evaluated.

        The baseline needs at least `baseline_weeks - 1` earlier cycles (3 of 4
        weeks = 21 days) and the z-score needs `zscore_min_periods` past
        deviations on top of that: 21 + 28 = 49 days by default. Used by the
        app and the evaluation to skip the un-scorable warm-up period instead
        of reporting it as "no anomalies found".
        """
        return (
            self.seasonal_period_days * (self.baseline_weeks - 1)
            + self.zscore_min_periods
        )


@dataclass(frozen=True)
class RetrievalSettings:
    """Parameters of the TF-IDF business-note retrieval.

    top_k
        Maximum number of notes handed to the narrative layer.
    window_days
        Notes are kept only if dated within this many days of the anomaly
        period. TF-IDF has no concept of time, so the date filter is applied
        *before* ranking.
    min_similarity
        Cosine-similarity floor. Below it a note is treated as noise, which is
        what makes an honest "no relevant evidence found" answer possible.
    """

    top_k: int = 3
    window_days: int = 3
    min_similarity: float = 0.10

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ConfigError("TOP_K must be >= 1.")
        if self.window_days < 0:
            raise ConfigError("RETRIEVAL_WINDOW_DAYS must be >= 0.")
        if not 0.0 <= self.min_similarity <= 1.0:
            raise ConfigError("MIN_SIMILARITY must be in [0, 1].")


@dataclass(frozen=True)
class ConfidenceSettings:
    """Weights and caps of the deterministic Evidence Confidence score.

    The three weights must sum to 1 and each component is clipped to [0, 1],
    which is what guarantees the final score is itself in [0, 1].

    z_cap / evidence_cap
        Saturation points: a |z| of 6 or a cosine similarity of 0.5 already
        counts as "as strong as it gets", so extreme values cannot dominate.
    """

    weight_anomaly: float = 0.4
    weight_contribution: float = 0.3
    weight_evidence: float = 0.3
    z_cap: float = 6.0
    evidence_cap: float = 0.5

    #: Label thresholds (see Section 16 of the spec).
    high_threshold: float = 0.70
    medium_threshold: float = 0.40

    def __post_init__(self) -> None:
        weights = {
            "CONF_WEIGHT_ANOMALY": self.weight_anomaly,
            "CONF_WEIGHT_CONTRIBUTION": self.weight_contribution,
            "CONF_WEIGHT_EVIDENCE": self.weight_evidence,
        }
        for key, value in weights.items():
            if not 0.0 <= value <= 1.0:
                raise ConfigError(f"{key} must be in [0, 1], got {value}.")
        total = sum(weights.values())
        # Floating-point addition of 0.4 + 0.3 + 0.3 is not exactly 1.0, so the
        # comparison uses a small tolerance rather than `== 1`.
        if abs(total - 1.0) > 1e-9:
            raise ConfigError(
                "CONF_WEIGHT_ANOMALY + CONF_WEIGHT_CONTRIBUTION + "
                f"CONF_WEIGHT_EVIDENCE must sum to 1, got {total}."
            )
        if self.z_cap <= 0:
            raise ConfigError("CONF_Z_CAP must be > 0.")
        if not 0.0 < self.evidence_cap <= 1.0:
            raise ConfigError("CONF_EVIDENCE_CAP must be in (0, 1].")
        if not 0.0 < self.medium_threshold < self.high_threshold < 1.0:
            raise ConfigError(
                "Confidence labels require 0 < medium_threshold < "
                "high_threshold < 1."
            )


@dataclass(frozen=True)
class LLMSettings:
    """Local Ollama/Qwen client settings and the fact-check tolerance.

    number_tolerance
        Relative tolerance used when checking that every number in the
        generated text also exists in the computed facts. It exists because
        the model may write "25%" where the facts say "25.0%".
    """

    enabled: bool = True
    ollama_url: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    timeout_seconds: float = 60.0
    temperature: float = 0.0
    number_tolerance: float = 0.005

    def __post_init__(self) -> None:
        if not self.ollama_url.startswith(("http://", "https://")):
            raise ConfigError(
                f"OLLAMA_URL must start with http:// or https://, got {self.ollama_url!r}."
            )
        if not self.model:
            raise ConfigError("OLLAMA_MODEL must not be empty.")
        if self.timeout_seconds <= 0:
            raise ConfigError("OLLAMA_TIMEOUT_SECONDS must be > 0.")
        if self.temperature < 0:
            raise ConfigError("LLM_TEMPERATURE must be >= 0.")
        if self.number_tolerance < 0:
            raise ConfigError("NUMBER_TOLERANCE must be >= 0.")

    @property
    def tags_url(self) -> str:
        """Endpoint listing the models installed in the local Ollama."""
        return f"{self.ollama_url.rstrip('/')}/api/tags"

    @property
    def chat_url(self) -> str:
        """Endpoint used for (non-streaming) narrative generation."""
        return f"{self.ollama_url.rstrip('/')}/api/chat"


@dataclass(frozen=True)
class Settings:
    """Everything the pipeline needs to run, grouped by pipeline stage."""

    anomaly: AnomalySettings = field(default_factory=AnomalySettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    confidence: ConfidenceSettings = field(default_factory=ConfidenceSettings)
    llm: LLMSettings = field(default_factory=LLMSettings)
    kpi: str = DEFAULT_KPI

    def __post_init__(self) -> None:
        if self.kpi not in SUPPORTED_KPIS:
            raise ConfigError(
                f"KPI {self.kpi!r} is not supported. Supported: {list(SUPPORTED_KPIS)}."
            )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build a validated `Settings` object from environment variables.

    Args:
        env: Mapping to read from. Defaults to the real process environment.
            Tests pass a plain dict so they never touch the developer's `.env`.

    Returns:
        A frozen, fully validated `Settings` instance.

    Raises:
        ConfigError: if any value is unparsable or fails validation.
    """
    if env is None:
        # `override=False`: a variable already exported in the shell wins over
        # the file, which is the behaviour people expect from 12-factor apps.
        load_dotenv(PROJECT_ROOT / ".env", override=False)
        env = os.environ

    settings = Settings(
        anomaly=AnomalySettings(
            seasonal_period_days=_get_int(env, "SEASONAL_PERIOD_DAYS", 7),
            baseline_weeks=_get_int(env, "BASELINE_WEEKS", 4),
            zscore_window_days=_get_int(env, "ZSCORE_WINDOW_DAYS", 56),
            zscore_min_periods=_get_int(env, "ZSCORE_MIN_PERIODS", 28),
            zscore_threshold=_get_float(env, "ZSCORE_THRESHOLD", 3.0),
            min_relative_change=_get_float(env, "MIN_RELATIVE_CHANGE", 0.05),
        ),
        retrieval=RetrievalSettings(
            top_k=_get_int(env, "TOP_K", 3),
            window_days=_get_int(env, "RETRIEVAL_WINDOW_DAYS", 3),
            min_similarity=_get_float(env, "MIN_SIMILARITY", 0.10),
        ),
        confidence=ConfidenceSettings(
            weight_anomaly=_get_float(env, "CONF_WEIGHT_ANOMALY", 0.4),
            weight_contribution=_get_float(env, "CONF_WEIGHT_CONTRIBUTION", 0.3),
            weight_evidence=_get_float(env, "CONF_WEIGHT_EVIDENCE", 0.3),
            z_cap=_get_float(env, "CONF_Z_CAP", 6.0),
            evidence_cap=_get_float(env, "CONF_EVIDENCE_CAP", 0.5),
        ),
        llm=LLMSettings(
            enabled=_get_bool(env, "LLM_ENABLED", True),
            ollama_url=_get_str(env, "OLLAMA_URL", "http://localhost:11434"),
            model=_get_str(env, "OLLAMA_MODEL", "qwen3:8b"),
            timeout_seconds=_get_float(env, "OLLAMA_TIMEOUT_SECONDS", 60.0),
            temperature=_get_float(env, "LLM_TEMPERATURE", 0.0),
            number_tolerance=_get_float(env, "NUMBER_TOLERANCE", 0.005),
        ),
        kpi=_get_str(env, "KPI", DEFAULT_KPI).lower(),
    )
    logger.debug("Configuration loaded: %s", settings)
    return settings


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings.

    Streamlit re-runs the whole script on every interaction, so parsing the
    environment each time would be wasteful; the cache makes it a one-off.
    Call `get_settings.cache_clear()` after changing the environment.
    """
    return load_settings()


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    current = load_settings()
    print("Configuration loaded and validated successfully.\n")
    print(f"KPI                : {current.kpi}")
    print(f"Anomaly rule       : |z| >= {current.anomaly.zscore_threshold} and "
          f"|rel. change| >= {current.anomaly.min_relative_change:.0%}")
    print(f"Baseline           : median of {current.anomaly.baseline_weeks} x "
          f"{current.anomaly.seasonal_period_days}-day lags "
          f"(warm-up {current.anomaly.warmup_days} days)")
    print(f"Retrieval          : top {current.retrieval.top_k} notes, "
          f"+/-{current.retrieval.window_days} days, "
          f"min similarity {current.retrieval.min_similarity}")
    print(f"Confidence weights : anomaly {current.confidence.weight_anomaly}, "
          f"contribution {current.confidence.weight_contribution}, "
          f"evidence {current.confidence.weight_evidence}")
    print(f"LLM                : enabled={current.llm.enabled}, "
          f"model={current.llm.model}, url={current.llm.ollama_url}")
    print(f"Data directory     : {DATA_DIR}")
