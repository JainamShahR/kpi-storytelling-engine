# Development Log

One entry per build phase: the goal, what was built, the key decisions and how
the phase was verified. Every number in the "Verified" sections comes from an
actual run - nothing is estimated. The log is updated at the end of each phase.

## Progress

| Phase | Topic | Status |
|---|---|---|
| 1 | Project skeleton, configuration, CI | ✅ Done |
| 2 | Synthetic KPI data + injected anomalies (answer key) | ✅ Done |
| 3 | Preprocessing and validation + tests | ✅ Done |
| 4 | Seasonal-naive baseline + tests | ✅ Done |
| 5 | Robust z-score detection, severity, naive detector + tests | ✅ Done |
| 6 | Dimensional decomposition + cross-check + tests | ✅ Done |
| 7 | Business notes generation | ⬜ Next |
| 8 | TF-IDF retrieval + tests | ⬜ |
| 9 | Evidence confidence score + tests | ⬜ |
| 10 | Facts builder, prompt, Ollama/Qwen client | ⬜ |
| 11 | Fact checks + deterministic fallback + tests | ⬜ |
| 12 | Pipeline + Streamlit dashboard | ⬜ |
| 13 | Complete test suite, CI green | ⬜ |
| 14 | Evaluation report | ⬜ |
| 15 | README + interview notes | ⬜ |
| 16 | Real-data case study (added by decision, see below) | ⬜ |

## Decisions log

- **Synthetic data stays the core; real data comes later as a case study.**
  Decided after Phase 4 (option A). Synthetic data has an answer key, which is
  what makes precision, recall and F1 measurable; real data has none. After
  the evaluation (Phase 14), Phase 16 will run the same pipeline on a public
  dataset. The main candidate is Kaggle "Store Sales - Time Series
  Forecasting" (Corporación Favorita, Ecuador), which comes with a real
  holidays/events file and contains the 16 April 2016 earthquake. The raw data
  will not be committed (licence); a download script will be provided instead.
- **Git history rewritten once (2026-10-01).** The first commit's message
  contained a co-author line that made an AI assistant appear as a repository
  contributor on GitHub.
  - It was removed with `git filter-branch --msg-filter` and force-pushed.
  - Only commit messages changed: `git diff` between the old and new history
    was empty. All commit IDs changed, and the IDs in this log are the new
    ones.
- **This log was first committed after Phase 6.** It had been kept locally
  since Phase 3 but was never added to a commit.

## Environment

- Python 3.11.16, installed with `uv` (Homebrew could not install it because
  the installed Xcode was too old). Project virtual environment: `.venv/`.
- Dependencies are listed in `requirements.txt`.

---

## Phase 1 - Project skeleton, configuration, CI

**Goal:** a foundation that every later phase plugs into. No data and no
analysis yet.

**Built**
- Folder layout: `src/`, `tests/`, `data/`, `reports/`, `docs/`.
- `src/config.py`: every tunable number in one place, read from environment
  variables with defaults from `.env.example`. There are four frozen groups
  (`anomaly`, `retrieval`, `confidence`, `llm`), and each one validates itself.
- `tests/test_config.py` (16 tests), `pytest.ini`, `requirements.txt`,
  `.gitignore`, `.env.example`, `LICENSE`, a stub `README.md`.
- `.github/workflows/tests.yml`: GitHub Actions runs `pytest` on Python 3.11
  on every push.

**Key decisions**
- No magic numbers in analysis code: thresholds are changed in one place.
- Fail fast: an invalid value (e.g. `ZSCORE_THRESHOLD=three`, or weights that
  do not sum to 1) stops the program at start-up with a message naming the key.
- Settings are frozen (read-only), so no module can change them mid-run.
- `load_settings(env=...)` takes the environment as an argument, so tests
  never depend on the developer's own `.env`.

**Verified**
- `python -m src.config` printed the validated configuration.
- `pytest`: 16 passed.
- GitHub Actions run 35503379998: success.
- Commit `5f8f6c9` - "chore: scaffold project, configuration and CI".

**Interview questions**
1. Why keep all configuration in one validated module?
2. Why compare `abs(total - 1.0) > 1e-9` instead of `total == 1.0`?
3. Why does `load_settings` accept the environment as a parameter?

---

## Phase 2 - Synthetic KPI data and injected anomalies

**Goal:** realistic data containing anomalies whose dates, segments and sizes
are known, so detection quality can be measured later instead of guessed.

**Built**
- `generate_data.py` writes `data/kpi_data.csv` (daily revenue and orders for
  4 regions x 4 products x 3 channels = 48 series) and
  `data/injected_anomalies.csv` (the answer key).
- The revenue model is
  `base share x weekly pattern x trend (+10%/yr) x yearly wave (+/-4%, peak mid-Nov) x noise (8%)`.
- 24 regular anomalies: each of 10/20/30/50% appears 3 times as a decline and
  3 times as an increase. 20 hit a single segment and 4 hit a region x product
  pair.
- A demo case with three overlapping declines on the same 3 days: West -50%,
  Product B -30%, Retail -20%.
- `python generate_data.py --plot` saves an interactive chart to
  `reports/data_overview.html` (git-ignored).

**Key decisions**
- Balanced design (6 anomalies per size, 12 up / 12 down), so recall per size
  can be compared fairly.
- Region x product pairs use only 30% or 50%: a pair is about 8% of revenue,
  so a 10% change there would be undetectable by design.
- The demo trio overlaps on purpose and occupies one calendar slot, which is at
  least 21 days from every other anomaly.
- No anomalies in the first 60 days, because baselines need history.
- One fixed seed (42) is split into independent random streams, so changing the
  anomaly design never changes the revenue noise.
- The script checks every rule of the specification before saving and stops
  with an error if one is broken.
- Fix during the phase: the chart first loaded plotly.js from a CDN and opened
  blank. The library is now embedded in the HTML file.

**Verified** (output of `python generate_data.py --plot`)
- 35,088 rows (731 days x 48 series), 2024-01-01 -> 2025-12-31.
- Daily total revenue: mean 10.98M, min 8.41M, max 13.86M.
- Weekend vs weekday: +10.1%. 2025 vs 2024: +9.5%.
- Q4 is the highest quarter in both years, and every 2025 quarter is above
  the same 2024 quarter.
- 27 anomalies (24 regular + 3 demo): 6 per size, 12 declines / 12 increases.
  By segment: 7 region, 7 product, 6 channel, 4 region x product.
  20 of 27 (74%) have a business note.
- Closest start dates are 24 days apart (minimum 21). The demo case is on
  2025-09-07.
- Reproducible: the MD5 of both CSV files was identical before and after
  re-running (`kpi_data.csv` 9e73eff5b72cc1617a9205991edb9e7d,
  `injected_anomalies.csv` 383c880010566a5d53f925afebef98c4).
- The chart showed the weekly zigzag, the upward trend and the large injected
  anomalies. Small ones were barely visible in the total, as expected.
- `pytest`: 16 passed.
- Commit `6ba8204` - "feat: generate synthetic KPI data with injected anomaly answer key".

**Lessons worth remembering**
- Dilution: impact on the total is roughly segment share x size. West -30%
  moves the total by about -7.8%, and Partner -10% by only about -2%. This is
  why detection must also run per segment.
- The 28-day mean in the chart jumps near big anomalies because a mean is
  pulled by outliers. This is why the baseline (Phase 4) uses a median.

**Interview questions**
1. Why synthetic data instead of a public dataset?
2. Why a fixed seed, and why separate random streams?
3. Why are some anomalies invisible in total revenue, and what does that imply?
4. Why no anomalies in the first 60 days, and why at least 21 days apart?

---

## Phase 3 - Preprocessing and validation

**Goal:** turn the raw CSV into a complete, continuous daily table, and catch
bad data loudly before it can turn into fake anomalies.

**Built**
- `src/preprocessing.py`:
  - `load_data(path)` reads the CSV and gives a helpful error if the file is
    missing.
  - `validate_data(df)` returns a `ValidationReport` (errors, warnings,
    counts) and never modifies the data.
  - `clean_data(df)` refuses data with errors, then drops unusable rows and
    fills gaps.
  - `aggregate_kpi(df, kpi, dimension, category)` returns the daily series for
    the company or one segment.
- `DataValidationError` carries the full report.
- `tests/test_preprocessing.py` (6 tests) and a `small_kpi_df` fixture in
  `tests/conftest.py` (14 days x 48 segments, revenue 100 each).

**Key decisions (bad-data strategy: never silently invent or discard numbers)**

| Problem | Decision | Why |
|---|---|---|
| Unknown region/product/channel | error | dropping it would understate totals |
| Missing, non-numeric or negative revenue | error | filling 0 would invent a -100% drop |
| Same key twice with different values | error | the correct value is unknown |
| Exact duplicate row | warning, drop copy | a double load |
| Empty date/region/product/channel | warning, drop row | cannot be placed in a series |
| Absent date x segment row | warning, fill 0 | no row = no sales (as specified) |
| Whole day absent | warning, fill 0 | as specified, but flagged: it will look like a -100% drop |

**Verified**
- `pytest tests/test_preprocessing.py -v`: 6 passed. `pytest`: 22 passed.
- `python -m src.preprocessing` on the generated data: validation passed with
  0 errors and 0 warnings, and every problem count was 0 over 731 days. The
  clean table has 35,088 rows (2024-01-01 -> 2025-12-31). The company total
  and the West series each have 731 daily values; on the first day West was
  2,687,748.44 of 10,106,514.00 (26.6%, matching its designed 26% share).
- On a copy with 4 planted problems, validation failed with 3 errors and
  1 warning:
  - the unknown region `Nort`;
  - one negative revenue;
  - one conflicting duplicate (2024-01-01 / North / Product A / Online);
  - the warning: 2 missing combinations (the dropped row and the row whose
    region became invalid).
  `clean_data` refused the data.
- Commit `336a628` - "feat: add data validation and cleaning with tests".

**Interview questions**
1. What is the difference between an error and a warning in your validation,
   and who decides?
2. Why is a missing row filled with 0 but a blank revenue value an error?
3. Why is an unknown category an error instead of just dropping the row?
4. Why does `validate_data` never modify the data?
5. What does `reindex(..., fill_value=0)` do, and why does it not overwrite
   existing values?

---

## Phase 4 - Seasonal-naive baseline

**Goal:** an "expected" value for every day, so each day can be judged against
what is normal for that weekday.

**Built**
- `src/anomaly_detection.py`:
  - `seasonal_baseline(series, period, weeks)` returns, for each day, the
    median of the same weekday 1-4 weeks earlier.
  - `compute_deviations(series, settings)` returns date, actual, expected,
    deviation and relative_deviation for every day.
- `tests/test_anomaly.py` (5 tests) on hand-made toy series.

**Key decisions**
- Compare with the same weekday: this removes the weekly pattern, so a normal
  Saturday is not "unusual".
- Use the median of 4 weeks, not "same day last week": this avoids the echo
  problem, where last week's anomaly makes a normal day look abnormal. Also not
  the mean: one -50% day would drag a 4-week mean down by 12.5%.
- A day is evaluated only with at least `BASELINE_WEEKS - 1` reference values
  (3 of 4 by default, as in the spec). This generalises the spec's fixed "3"
  so it stays valid if `BASELINE_WEEKS` is changed; the default behaviour is
  identical.
- If expected is 0, the relative deviation is NaN (undefined) instead of
  infinity.
- A series with a missing day is rejected, because a shift of 7 positions must
  mean 7 days.
- `compute_deviations` is a small helper not named in the spec's function list.
  It produces the per-day values required by the spec (actual, expected,
  deviation, relative deviation), and Phase 5's `detect_anomalies` builds on it.

**Verified** (output of `pytest` and `python -m src.anomaly_detection`)
- `pytest tests/test_anomaly.py -v`: 5 passed. `pytest`: 27 passed.
- Company total: 21 warm-up days without a baseline; the first evaluated day
  is 2024-01-22.
- Demo case at company level:

  | Day | Actual | Expected | Relative deviation |
  |---|---|---|---|
  | Sun 2025-09-07 | 8.97M | 12.31M | -27.1% |
  | Mon 2025-09-08 | 8.41M | 11.22M | -25.0% |
  | Tue 2025-09-09 | 8.63M | 11.41M | -24.4% |

  Sunday drops most because Retail (-20%) is a larger share of weekend revenue.
- 20 single-segment anomalies, measured on each segment's own series: the
  measured deviation was close to the injected size in every case. Examples:
  A-006 North -50% -> -49.9%, A-017 Product B -50% -> -48.8%, A-002 Partner
  -10% -> -10.0%.
  - Average gap about 1.7 percentage points (worked out from the printed
    table).
  - Largest gaps: A-003 North +30% -> +23.8% (a single day, noise) and
    A-024 East +50% -> +55.4%.
- Commit `e0dac87` - "feat: add seasonal-naive baseline with tests".

**Interview questions**
1. What is a seasonal-naive baseline, and why is it "naive"?
2. Why the median of the last 4 same weekdays rather than "same day last week"
   or a 4-week mean? (the echo problem)
3. Why are the first 21 days not evaluated?
4. What is the difference between deviation and relative deviation, and when
   is the relative deviation undefined?
5. What can this baseline not handle? (holidays that move between weekdays,
   permanent level shifts)

---

## Phase 5 - Robust z-score detection

**Goal:** decide which deviations are anomalies, using a rule that adapts to
how noisy each series normally is.

**Built** (in `src/anomaly_detection.py`)
- `robust_zscore(deviation, window, min_periods)` computes
  `(deviation - median(past 56 days)) / (1.4826 x MAD(past 56 days))`.
  The window excludes today.
- `severity_from_z(z, threshold)` gives Moderate / High / Critical at
  |z| >= 3 / 4 / 5 (the bands move with the threshold).
- `detect_anomalies(series, settings)` applies the rule: |z| >= 3 AND
  |relative change| >= 5%. It adds `is_anomaly`, `severity` and `direction`.
- `detect_all_levels(df, settings)` runs the rule on 12 series: the company
  total, 4 regions, 4 products and 3 channels. The list is `DETECTION_LEVELS`.
- `naive_detector(series)` flags days more than 10% away from the same weekday
  last week. It exists only as the evaluation baseline.
- 7 new tests in `tests/test_anomaly.py`.

**Key decisions**
- Use the median and MAD rather than the mean and standard deviation: past
  anomalies sit inside the 56-day window, and a standard deviation would be
  inflated by them.
- The factor 1.4826 makes MAD equal the standard deviation for normal data,
  so |z| = 3 keeps its usual meaning.
- The past window excludes today (no data leakage). A test proves that
  changing a later value never changes an earlier z-score.
- Two conditions must hold. |z| >= 3 alone flags commercially tiny moves on
  very stable series, which is why a 5% minimum change is also required.
- A zero MAD (constant history) is replaced by epsilon = 1e-9 and a warning
  is logged.
- Days without enough history get severity "Not evaluated" instead of
  "Normal", so a warm-up day is never presented as a checked normal day.
  (The spec says such days are "not evaluated"; this makes it visible.)
- Fix to Phase 1 code: `AnomalySettings.warmup_days` said 56 (28 + 28). The
  baseline needs 21 days (3 of 4 weeks), so the true warm-up is 21 + 28 = 49.
  The property, its test, and a new test that checks the first z-score
  appears on day 49 now agree.

**Verified** (output of `pytest` and `python -m src.anomaly_detection`)
- `pytest tests/test_anomaly.py -v`: 12 passed. `pytest`: 34 passed.
- First evaluated day: day 49. Monitored series: 12.
- 271 flagged series-days across 85 distinct dates. By severity: Critical
  123, Moderate 103, High 45. Per series: Company total 27, North 26,
  South 20, East 16, West 18, Product A 21, Product B 30, Product C 24,
  Product D 19, Online 26, Retail 22, Partner 22.
- **25 of 27 injected anomalies detected** (at least one day flagged on the
  company total or on the injected segment).
  - All 4 region x product pairs were caught, with max |z| between 3.4
    and 5.8.
  - Missed: A-018 South -10% and A-020 East +10%, both with max
    |z| = 2.7, just under the threshold. This is the expected "small change
    in a segment" limitation.
- 42 flagged dates had no injected anomaly (false alarms under this rule).
  The first examples were 2024-02-26, 02-29, 03-09, 03-16, 03-17 and 03-23.
  Derived from the same output: 85 - 42 = 43 flagged dates fall inside the
  54 injected anomaly days.

**Observation to investigate in Phase 14 (not changed yet)**
- 4 of the 6 listed false-alarm dates are weekends. A possible cause: the
  spec computes z on the absolute deviation (currency). With noise
  proportional to level, weekend days of Retail/Online (and weekdays of
  Partner) have a larger spread than the 56-day MAD assumes, which inflates
  their z-scores.
- A candidate fix is to compute z on the relative deviation instead. That
  would change the spec's formula, so it will only be tested, with measured
  precision/recall for both versions, if approved.

**Interview questions**
1. What is a z-score, and why a robust one (median/MAD) instead of mean/std?
2. Why multiply MAD by 1.4826?
3. Why exclude today from the window, and how does the test prove it?
4. Why require both |z| >= 3 and a 5% change?
5. Why monitor 12 series, and what does that cost in false alarms?
6. Which anomalies were missed, and why? (A-018, A-020: +/-10% in one
   region, max |z| 2.7)

- Commit `9ab9df7` - "feat: add robust z-score anomaly detection with severity
  and naive baseline".

---

## Phase 6 - Root-cause decomposition

**Goal:** for a flagged day, show which regions, products and channels
caused the company's change, and how much of it each one explains.

**Built**
- `src/decomposition.py`:
  - `decompose(df, date, settings)` returns a `Decomposition` with the
    company totals and one ranked table per dimension. Each table has
    baseline, actual, absolute_change, pct_change and contribution_pct.
  - `top_contributors(decomposition, n)` returns the n largest contributions
    across dimensions; the first one is the overall top contributor.
  - `cross_dimension(df, date, settings, dims, n)` returns the top region x
    product combined segments.
- `MIN_DECOMPOSITION_CHANGE = 0.02` in `src/config.py`.
- `tests/test_decomposition.py` (5 tests) and a `decomposition_df` fixture.
  In the fixture every day is identical and products differ in size
  (A 480, B 360, C 240, D 120 per day).

**Key decisions**
- Rank by contribution (share of the total change), not by percentage
  change. A small category with a large percentage change can explain little
  of the total.
- Contributions add up to 100% within each dimension and are never added
  across dimensions, because region, product and channel are overlapping
  views of the same change. A test shows the same change counted as 100% in
  both the region and the product view.
- The decomposition uses a mean baseline (same 4 reference days as
  detection). Means add up exactly, so contributions sum to exactly 100%.
  Detection uses the median for robustness, and medians do not add up.
- If the total change is below 2% of the baseline, the decomposition is
  skipped and a clear message is returned (the percentages would explode).
- Spec deviation: `decompose` returns a small `Decomposition` dataclass
  instead of a bare `dict[str, DataFrame]`. The dict is still there
  (`by_dimension`), but a bare dict cannot carry the totals or the "skipped"
  message the spec asks for.
- Spec deviation: `cross_dimension` takes `settings`, because it needs the
  baseline settings (same reference days as detection).
- The company total is always decomposed. Drilling down inside a segment is
  listed as a future improvement in the spec, so it is not built.

**Verified** (output of `pytest` and `python -m src.decomposition`)
- `pytest tests/test_decomposition.py -v`: 5 passed. `pytest`: 39 passed.
- Demo case, Sun 2025-09-07: total 8.97M vs baseline 12.36M (-27.4%).
  - The top contributor of each dimension is the planted segment:
    - region: West, -57.0%, contribution 54.3%;
    - product: Product B, -42.5%, contribution 45.9%;
    - channel: Retail, -37.0%, contribution 54.9%.
  - Top 3 across dimensions: Retail 54.9%, West 54.3%, Product B 45.9%. The
    overall top is Retail by 0.6 points; on a Sunday Retail is a larger share
    of revenue.
  - Cross-check: West x Product B is first (-67.0%, contribution 19.3%).
- 24 regular anomalies, each decomposed on its first day:
  - 8 skipped because the company total moved by less than 2%. These include
    all region x product pairs except A-013, and the +/-10% anomalies.
  - 15 single-dimension anomalies decomposed: top-1 correct 14 of 15, top-3
    correct 14 of 15.
  - 1 pair decomposed (A-013 West x Product D): ranked #1 in the
    cross-check.

**Finding: the mean baseline is contaminated by earlier anomalies**
- The one wrong result is A-018 (South -10%, Wed 2025-06-18). The company
  total showed +2.7% instead of a decline, and South's contribution was -19%.
- Its reference day 28 days earlier, Wed 2025-05-21, is inside A-017
  (Product B -50%). That day lowers the 4-day mean baseline by roughly a
  quarter of a -15% day, i.e. about 3.75%. This is consistent with the total
  flipping from about -2% to +2.7%.
- The same alignment (a reference day exactly 28 days back falling inside the
  previous anomaly) also occurs for A-008, A-009, A-011 and A-020.
  - These are 5 of the 6 small single-dimension cases that were skipped or
    wrong.
  - The injected anomalies are about 28 days apart, so t-28 often lands on
    the previous one.
- Detection is not affected: the median of 4 ignores one bad reference day.
  The mean used here (needed so contributions add up to 100%) is affected.
- Candidate fix, to measure in Phase 14 if approved: compute the
  decomposition baseline as the mean of the reference days that were not
  flagged as anomalies. The contributions would still add up to 100%.

**Other observations**
- Contributions above 100% (A-004 120%, A-010 134%, A-026 169%) are correct:
  the other categories moved the opposite way and partly offset the change,
  so the planted category explains more than the whole net change.

**Interview questions**
1. Why rank by contribution instead of percentage change?
2. Why do contributions add to 100% within a dimension but not across?
3. Why a mean baseline here when detection uses a median?
4. What does a contribution above 100% mean?
5. Why skip the decomposition when the total change is below 2%?
6. What went wrong for A-018, and how would you fix it?

- Commit `5f00476` - "feat: add root-cause decomposition with region x product
  cross-check".
