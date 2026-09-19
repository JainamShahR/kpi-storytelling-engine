"""KPI Storytelling Engine - source package.

The package is deliberately split into small modules that map 1:1 onto the
stages of the pipeline:

    preprocessing -> anomaly_detection -> decomposition -> retrieval
    -> confidence -> narrative -> pipeline

Only `narrative` talks to a language model; every other module is
deterministic, which is what makes the results explainable and testable.
"""

__version__ = "0.1.0"
