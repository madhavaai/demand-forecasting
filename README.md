# Retail Demand Forecasting — POS Transaction Pipeline

End-to-end demand forecasting system for a fuel / convenience retailer,
modeled on a large multi-store POS environment. This repository is built in
stages; **Stage 1 (data generation + production-grade cleaning) is complete.**

The emphasis is on the parts that are usually skipped in portfolio projects:
realistic data dirtiness and an auditable cleaning pipeline with full lineage.

---

## Why synthetic data

Real POS data is proprietary and privacy-sensitive, so the data is generated
synthetically — but *deliberately dirty*. `generate_data.py` injects every
class of real-world data-quality issue at a known, logged rate (a "dirtiness
ledger"), so the cleaning pipeline can be **validated against ground truth**:
we know exactly what was broken, so we can prove what was caught.

The generator models:
- 25 stores across 4 US timezones, 2 years of daily history
- Fuel (sold per gallon) + in-store merchandise (sold per unit)
- Trend, weekly seasonality (highway peaks weekends, urban peaks weekdays),
  summer driving-season effect, and holiday travel spikes
- Poisson transaction counts around the demand signal

---

## Stage 1 — Data cleaning pipeline

`data_cleaning.py` runs a 10-stage funnel. Its governing principle:
**nothing is silently dropped.** Every rejected row is moved to a quarantine
table tagged with its rejection reason, giving full lineage for audit.

| Stage | What it does |
|-------|--------------|
| 0. Ingest | Read every column as a string so `"N/A"` isn't silently coerced to NaN before it's logged |
| 1. Schema | Validate expected columns; convert literal empty strings to true nulls |
| 2. Dedupe | Remove exact full-row duplicates; quarantine same-id rows with conflicting payloads |
| 3. Normalize strings | Trim whitespace, case-fold, map 25 messy product variants → 9 canonical labels |
| 4. Coerce types | String → numeric for quantity/price/amount, counting coercion failures |
| 5. Parse timestamps | Three mixed formats → single UTC instant; localize naive times by store timezone; derive **local business date** for daily aggregation; quarantine unparseable + clock-skewed |
| 6. Referential integrity | Quarantine null and orphan `store_id` not present in the store dimension |
| 7. Business rules | Label refunds (legit negatives) vs quarantine corrupt half-negatives; repair null amounts from qty×price where possible; reconcile `amount == qty × unit_price` within tolerance |
| 8. Outliers | Hard physical bounds (impossible gallons/prices) → quarantine; statistical IQR-per-category → **flag, not drop** |
| 9. Finalize | Select columns, set types, sort |

### Key design decisions (the interview-relevant parts)

- **Quarantine over drop.** A dropped row is a silent hole in tomorrow's
  totals. A quarantined row is traceable: you can answer *why* a number moved.
- **Local business day, not UTC day.** An 11pm Pacific sale belongs to that
  store's business day, not the next UTC day. Aggregating on the wrong day
  boundary smears demand across days and corrupts the forecast target. This is
  why timestamps are converted to UTC *and* back to store-local.
- **Refund ≠ corruption.** `amount < 0 AND quantity < 0` is a legitimate
  refund (kept and labeled). Only one of the two being negative is a sign
  error (quarantined). Knowing the business is what separates these.
- **Two-tier outliers.** Physically impossible values (800 gallons) are
  quarantined. Plausible-but-extreme values (a genuinely large sale) are
  *flagged* via per-category IQR so the forecasting layer can choose to
  winsorize or down-weight them — destroying them would discard real demand.
- **Repair before reject.** A null `amount` is recomputed from
  `quantity × unit_price` when both are present, rather than thrown away.

### Reconciliation

The report cross-checks caught vs injected. Counts intentionally don't match
1:1 — multiple issues can land on the same row, and the earliest failing gate
claims it (a row that's both a duplicate and a price-mismatch is caught once).
The report explains each gap.

---

## Running it

```bash
pip install pandas numpy pytz holidays
cd src
python generate_data.py     # writes data/pos_transactions.csv + dirtiness_ledger.json
python data_cleaning.py     # writes data/clean_transactions.csv + quarantine.csv + report
```

Outputs land in `data/` and `reports/data_quality_report.txt`.

---

## Roadmap (next stages)

- **Stage 2 — Feature engineering**: aggregate to daily store×category demand;
  build temporal features (day-of-week, rolling 7/30-day means, lags), holiday
  flags, and price features. Strict temporal discipline (no look-ahead).
- **Stage 3 — Forecasting models**: seasonal-naive baseline → Prophet →
  gradient-boosted lag model (LightGBM) → optional LSTM/TFT. Backtest with
  expanding-window temporal splits.
- **Stage 4 — Ensemble**: weighted blend of the base forecasters, weights
  tuned on a validation fold.
- **Stage 5 — Evaluation & monitoring**: MAPE by store/SKU segment, drift
  detection, retrain triggers.

---

## Project structure

```
retail_forecasting/
├── src/
│   ├── generate_data.py      # dirty synthetic POS generator
│   └── data_cleaning.py      # 10-stage cleaning pipeline
├── data/                     # generated CSVs (gitignored in practice)
└── reports/
    └── data_quality_report.txt
```




# Retail Demand Forecasting — POS Transaction Pipeline

End-to-end demand forecasting system for a fuel / convenience retailer,
modeled on a large multi-store POS environment. This repository is built in
stages; **Stages 1–2 (data generation, cleaning, and feature engineering) are complete.**

The emphasis is on the parts usually skipped in portfolio projects: realistic
data dirtiness, an auditable cleaning pipeline with full lineage, and
leakage-safe feature engineering.

---

## Why synthetic data

Real POS data is proprietary and privacy-sensitive, so the data is generated
synthetically — but *deliberately dirty*. `generate_data.py` injects every
class of real-world data-quality issue at a known, logged rate (a "dirtiness
ledger"), so the cleaning pipeline can be **validated against ground truth**:
we know exactly what was broken, so we can prove what was caught.

The generator models:
- 25 stores across 4 US timezones, 2 years of daily history
- Fuel (sold per gallon) + in-store merchandise (sold per unit)
- Trend, weekly seasonality (highway peaks weekends, urban peaks weekdays),
  summer driving-season effect, and holiday travel spikes
- Poisson transaction counts around the demand signal

---

## Stage 1 — Data cleaning pipeline

`data_cleaning.py` runs a 10-stage funnel. Its governing principle:
**nothing is silently dropped.** Every rejected row is moved to a quarantine
table tagged with its rejection reason, giving full lineage for audit.

| Stage | What it does |
|-------|--------------|
| 0. Ingest | Read every column as a string so `"N/A"` isn't silently coerced to NaN before it's logged |
| 1. Schema | Validate expected columns; convert literal empty strings to true nulls |
| 2. Dedupe | Remove exact full-row duplicates; quarantine same-id rows with conflicting payloads |
| 3. Normalize strings | Trim whitespace, case-fold, map 25 messy product variants → 9 canonical labels |
| 4. Coerce types | String → numeric for quantity/price/amount, counting coercion failures |
| 5. Parse timestamps | Three mixed formats → single UTC instant; localize naive times by store timezone; derive **local business date** for daily aggregation; quarantine unparseable + clock-skewed |
| 6. Referential integrity | Quarantine null and orphan `store_id` not present in the store dimension |
| 7. Business rules | Label refunds (legit negatives) vs quarantine corrupt half-negatives; repair null amounts from qty×price where possible; reconcile `amount == qty × unit_price` within tolerance |
| 8. Outliers | Hard physical bounds (impossible gallons/prices) → quarantine; statistical IQR-per-category → **flag, not drop** |
| 9. Finalize | Select columns, set types, sort |

**Result:** ~648K clean rows kept from ~670K raw (96.9%), ~12K quarantined with
reasons, reconciled against the ground-truth ledger.

### Key design decisions (the interview-relevant parts)

- **Quarantine over drop.** A dropped row is a silent hole in tomorrow's
  totals. A quarantined row is traceable: you can answer *why* a number moved.
- **Local business day, not UTC day.** An 11pm Pacific sale belongs to that
  store's business day, not the next UTC day. Aggregating on the wrong day
  boundary smears demand across days and corrupts the forecast target.
- **Refund ≠ corruption.** `amount < 0 AND quantity < 0` is a legitimate
  refund (kept and labeled). Only one of the two being negative is a sign
  error (quarantined).
- **Two-tier outliers.** Physically impossible values → quarantined.
  Plausible-but-extreme values → *flagged* (per-category IQR) so the
  forecasting layer can choose to winsorize rather than discarding real demand.
- **Repair before reject.** A null `amount` is recomputed from
  `quantity × unit_price` when both are present, rather than thrown away.

---

## Stage 2 — Feature engineering

`feature_engineering.py` turns the clean transaction table into a model-ready
daily feature matrix. **Grain: one store × one product × one day.** Target:
that day's net demand (units sold, refunds netted). Each (store, product) is
kept as its own series — fuel (gallons) and merchandise (units) aren't pooled.

**Result:** 158,175 rows × 42 columns, 225 series (25 stores × 9 products),
Jan 2023 – Dec 2024. Leakage self-check **passed**, zero remaining feature NaNs.

### Pipeline

| Step | What it does |
|------|--------------|
| 1. Aggregate | Transactions → daily store×product demand; refunds net out via a plain sum |
| 2. Complete calendar | Build the full (store×product×every-day) grid and **zero-fill no-sale days** (+11,145 rows) so lag/rolling features sit on a gap-free index |
| 3. Calendar features | day-of-week, month, week, holiday flags, days-to-nearest-holiday, cyclical sin/cos |
| 4. Lag & rolling | demand lags (1/7/14/28) and trailing mean/std/max (7/14/28), **all shifted** so row *t* sees only *t−1* and earlier |
| 5. Price features | current price (retailer-set, known in advance) + lagged price + price change |
| 6. Leakage self-check | recomputes a feature the leaky way and asserts ours differs — guards against a missing `.shift()` |

### Key design decisions

- **Completing the calendar is the most-skipped step.** A zero-sale day
  produces no transaction row, so it's *absent* from the aggregate — but the
  demand was genuinely zero. Without zero-filling, `lag_1` can silently reach
  back several days across the gaps. Demand fills with 0; **price fills with
  NaN, not 0** (there's no price on a no-sale day).
- **No look-ahead leakage.** Forecasting is uniquely exposed because rows are
  time-ordered and correlated. Every demand-derived feature is shifted:
  `s.shift(1).rolling(7).mean()` — the `shift(1)` *before* the rolling is what
  prevents the target leaking into its own feature.
- **Features grouped by what's knowable when:**
  - *Known-in-advance* (calendar): deterministic future facts → safe unshifted.
  - *Derived-from-past* (lags, rolling): must be shifted.
  - *Controlled* (price): retailer sets it, so it's known in advance.
- **Cyclical encoding** (sin/cos) so December and January read as adjacent.
  Tree models don't need it; Prophet/LSTM/linear models benefit.

---

## Running it

```bash
pip install pandas numpy pytz holidays
cd src
python generate_data.py        # raw dirty data + dirtiness ledger
python data_cleaning.py        # clean_transactions.csv + quarantine.csv + report
python feature_engineering.py  # features.csv + feature report
```

Outputs land in `data/` and `reports/`.

---

## Roadmap

- ~~**Stage 1 — Data generation + cleaning**~~ ✅ done
- ~~**Stage 2 — Feature engineering**~~ ✅ done
- **Stage 3 — Forecasting models**: seasonal-naive baseline → Prophet →
  gradient-boosted lag model (LightGBM) → optional LSTM/TFT. Expanding-window
  backtesting with strict temporal splits.
- **Stage 4 — Ensemble**: weighted blend of the base forecasters, weights
  tuned on a validation fold.
- **Stage 5 — Evaluation & monitoring**: MAPE by store/SKU segment, drift
  detection, retrain triggers.

---

## Project structure
retail_forecasting/

├── src/

│   ├── generate_data.py         # dirty synthetic POS generator

│   ├── data_cleaning.py         # 10-stage cleaning pipeline

│   └── feature_engineering.py   # leakage-safe daily feature matrix

├── data/                        # generated CSVs (gitignored in practice)

└── reports/

├── data_quality_report.txt

└── feature_report.txt

---

## Notes / lessons

- **Mixed-timezone datetime bug:** a single pandas datetime column can't hold
  mixed UTC offsets — it silently coerces to NaT. Deriving the local *date*
  per timezone group fixes it. (This had been nulling 75% of business dates.)
- **Vectorize over row loops:** the first generator looped per-transaction and
  timed out; vectorizing per store-day made it tractable.
- **Reconciliation counts won't tie 1:1:** issues overlap on rows and the
  earliest gate claims each one; some corruptions create side effects later
  gates catch. The reports explain each gap.













# Retail Demand Forecasting — POS Transaction Pipeline

End-to-end demand forecasting system for a fuel / convenience retailer,
modeled on a large multi-store POS environment. Built in stages;
**Stages 1–3 (cleaning, feature engineering, and modeling) are complete.**

The emphasis is on the parts usually skipped in portfolio projects: realistic
data dirtiness, an auditable cleaning pipeline with full lineage, leakage-safe
feature engineering, and an honest temporal evaluation.

---

## Why synthetic data

Real POS data is proprietary, so the data is generated synthetically — but
*deliberately dirty*. `generate_data.py` injects every class of real-world
data-quality issue at a known, logged rate (a "dirtiness ledger"), so the
cleaning pipeline can be **validated against ground truth**: we know exactly
what was broken, so we can prove what was caught.

The generator models 25 stores across 4 US timezones over 2 years of daily
history, selling fuel (per gallon) and merchandise (per unit), with trend,
weekly seasonality, summer driving-season effects, and holiday travel spikes.

---

## Stage 1 — Data cleaning pipeline

`data_cleaning.py` runs a 10-stage funnel. Governing principle: **nothing is
silently dropped.** Every rejected row is moved to a quarantine table tagged
with its rejection reason, giving full lineage for audit.

| Stage | What it does |
|-------|--------------|
| 0. Ingest | Read every column as a string so `"N/A"` isn't silently coerced to NaN |
| 1. Schema | Validate columns; convert empty strings to true nulls |
| 2. Dedupe | Drop exact duplicates; quarantine same-id rows with conflicting payloads |
| 3. Normalize strings | Trim/case-fold; map 25 messy product variants → 9 canonical labels |
| 4. Coerce types | String → numeric, counting coercion failures |
| 5. Parse timestamps | Three mixed formats → single UTC instant; derive **local business date**; quarantine unparseable + clock-skewed |
| 6. Referential integrity | Quarantine null/orphan `store_id` not in the store dimension |
| 7. Business rules | Label refunds vs quarantine sign errors; repair null amounts; reconcile `amount == qty × price` |
| 8. Outliers | Impossible values → quarantine; statistical IQR → **flag, not drop** |
| 9. Finalize | Select, type, sort |

**Result:** ~96.9% of raw rows kept, the rest quarantined with reasons and
reconciled against the ground-truth ledger.

**Key decisions:** quarantine over drop (traceability); aggregate on the
**local** business day not the UTC day (an 11pm Pacific sale belongs to that
store's day); refund (`amount<0 AND qty<0`) ≠ corruption (one-sided negative);
two-tier outliers (impossible → drop, extreme-but-plausible → flag); repair
before reject.

---

## Stage 2 — Feature engineering

`feature_engineering.py` builds a model-ready daily matrix. **Grain: one store
× one product × one day.** Target: net demand (units sold, refunds netted).

**Result:** 158,175 rows × 42 columns, 225 series (25 stores × 9 products).
Leakage self-check **passed**, zero remaining feature NaNs.

| Step | What it does |
|------|--------------|
| 1. Aggregate | Transactions → daily store×product demand (refunds net out) |
| 2. Complete calendar | Build full grid and **zero-fill no-sale days** (+11,145 rows) so lag/rolling features sit on a gap-free index |
| 3. Calendar features | day-of-week, month, week, holiday flags, days-to-holiday, cyclical sin/cos |
| 4. Lag & rolling | demand lags (1/7/14/28) and trailing mean/std/max (7/14/28), **all shifted** so row *t* sees only *t−1* and earlier |
| 5. Price features | current price + lagged price + price change |
| 6. Leakage self-check | recomputes a feature the leaky way and asserts ours differs |

**Key decisions:** completing the calendar is the most-skipped step (a no-sale
day is genuinely zero demand, but absent from the aggregate — without
zero-filling, `lag_1` silently reaches across gaps); demand fills with 0 but
**price fills with NaN** (no price on a no-sale day); every demand-derived
feature is shifted (`s.shift(1).rolling(7).mean()` — the `shift(1)` is what
stops the target leaking into its own feature); features grouped by what's
knowable when (known-in-advance calendar vs derived-from-past lags vs
retailer-set price).

---

## Stage 3 — Forecasting models

`models.py` trains four forecasters, dumb to sophisticated, each justifying its
complexity against the previous one, and evaluates them on a **held-out
temporal block** (train on the past, test on the final 45 days — never the
reverse).

| Model | WAPE | vs baseline |
|-------|------|-------------|
| Seasonal-naive (same weekday last week) | 0.578 | — |
| Prophet (per-series decomposition) | 0.429 | 26% better |
| **LightGBM (global, on engineered features)** | **0.380** | **34% better** |
| Ensemble (validation-tuned blend) | 0.380 | collapsed to LightGBM |

**Key decisions:**

- **Leakage-safe feature selection.** Of 42 feature columns, only 36 are legal
  model inputs. `revenue`, `txn_count`, `refund_units`, `outlier_txns` are
  *realized on the target day* — using them to predict demand is leakage. Only
  their lagged versions are kept.
- **WAPE, not MAPE.** ~7% of store/product/days have zero demand, which makes
  MAPE (which divides by the actual) undefined/explosive. WAPE
  (`sum|error| / sum|actual|`) pools error over volume and is the retail
  standard for intermittent demand.
- **Honest ensemble.** The validation-tuned blend collapsed to 100% LightGBM —
  the data didn't support a blend, so the single model is the honest answer.
- **Sensible segments.** Fuel forecasts better (WAPE 0.37) than merchandise
  (0.47); merchandise is lower-volume and noisier.

**Caveats kept visible:** evaluation is a single temporal holdout (the stronger
upgrade is expanding-window backtesting across several folds); current-day
average price is treated as known-in-advance (a defensible elasticity-modeling
assumption, worth a robustness check excluding it).

---

## Running it

```bash
pip install pandas numpy pytz holidays lightgbm prophet
cd src
python generate_data.py        # raw dirty data + dirtiness ledger
python data_cleaning.py        # clean_transactions.csv + quarantine.csv + report
python feature_engineering.py  # features.csv + feature report
python models.py               # model_results.txt
```

---

## Roadmap

- ~~**Stage 1 — Data generation + cleaning**~~ ✅
- ~~**Stage 2 — Feature engineering**~~ ✅
- ~~**Stage 3 — Forecasting models**~~ ✅
- **Stage 4 — Robustness & backtesting**: expanding-window cross-validation
  across multiple folds; feature-ablation robustness checks.
- **Stage 5 — Monitoring**: WAPE by store/SKU segment over time, drift
  detection, retrain triggers.

---

## Project structure
retail_forecasting/

├── src/

│   ├── generate_data.py         # dirty synthetic POS generator

│   ├── data_cleaning.py         # 10-stage cleaning pipeline

│   ├── feature_engineering.py   # leakage-safe daily feature matrix

│   └── models.py                # forecasters + temporal evaluation

├── data/                        # generated CSVs (gitignored in practice)

└── reports/

├── data_quality_report.txt

├── feature_report.txt

└── model_results.txt

---

## Notes / lessons

- **Mixed-timezone datetime bug:** a single pandas datetime column can't hold
  mixed UTC offsets — it silently coerces to NaT. Deriving the local *date*
  per timezone group fixes it. (This had been nulling 75% of business dates.)
- **Vectorize over row loops:** the first generator looped per-transaction and
  timed out; vectorizing per store-day made it tractable.
- **Leakage is the forecasting failure mode:** every rolling feature is
  shifted, the model feature list excludes same-day realized columns, and a
  self-check guards against a missing `.shift()`.


# Retail Demand Forecasting — POS Transaction Pipeline

End-to-end demand forecasting system for a fuel / convenience retailer,
modeled on a large multi-store POS environment. Built in stages;
**Stages 1–4 (cleaning, feature engineering, modeling, and robustness) are complete.**

The emphasis is on the parts usually skipped in portfolio projects: realistic
data dirtiness, an auditable cleaning pipeline with full lineage, leakage-safe
feature engineering, an honest temporal evaluation, and robustness checks that
surface the model's weak points rather than hiding them.

---

## Why synthetic data

Real POS data is proprietary, so the data is generated synthetically — but
*deliberately dirty*. `generate_data.py` injects every class of real-world
data-quality issue at a known, logged rate (a "dirtiness ledger"), so the
cleaning pipeline can be **validated against ground truth**: we know exactly
what was broken, so we can prove what was caught.

The generator models 25 stores across 4 US timezones over 2 years of daily
history, selling fuel (per gallon) and merchandise (per unit), with trend,
weekly seasonality, summer driving-season effects, and holiday travel spikes.

---

## Stage 1 — Data cleaning pipeline

`data_cleaning.py` runs a 10-stage funnel. Governing principle: **nothing is
silently dropped.** Every rejected row is moved to a quarantine table tagged
with its rejection reason, giving full lineage for audit.

| Stage | What it does |
|-------|--------------|
| 0. Ingest | Read every column as a string so `"N/A"` isn't silently coerced to NaN |
| 1. Schema | Validate columns; convert empty strings to true nulls |
| 2. Dedupe | Drop exact duplicates; quarantine same-id rows with conflicting payloads |
| 3. Normalize strings | Trim/case-fold; map 25 messy product variants → 9 canonical labels |
| 4. Coerce types | String → numeric, counting coercion failures |
| 5. Parse timestamps | Three mixed formats → single UTC instant; derive **local business date**; quarantine unparseable + clock-skewed |
| 6. Referential integrity | Quarantine null/orphan `store_id` not in the store dimension |
| 7. Business rules | Label refunds vs quarantine sign errors; repair null amounts; reconcile `amount == qty × price` |
| 8. Outliers | Impossible values → quarantine; statistical IQR → **flag, not drop** |
| 9. Finalize | Select, type, sort |

**Result:** ~96.9% of raw rows kept, the rest quarantined with reasons and
reconciled against the ground-truth ledger.

**Key decisions:** quarantine over drop (traceability); aggregate on the
**local** business day not the UTC day (an 11pm Pacific sale belongs to that
store's day); refund (`amount<0 AND qty<0`) ≠ corruption (one-sided negative);
two-tier outliers (impossible → drop, extreme-but-plausible → flag); repair
before reject.

---

## Stage 2 — Feature engineering

`feature_engineering.py` builds a model-ready daily matrix. **Grain: one store
× one product × one day.** Target: net demand (units sold, refunds netted).

**Result:** 158,175 rows × 42 columns, 225 series (25 stores × 9 products).
Leakage self-check **passed**, zero remaining feature NaNs.

| Step | What it does |
|------|--------------|
| 1. Aggregate | Transactions → daily store×product demand (refunds net out) |
| 2. Complete calendar | Build full grid and **zero-fill no-sale days** (+11,145 rows) so lag/rolling features sit on a gap-free index |
| 3. Calendar features | day-of-week, month, week, holiday flags, days-to-holiday, cyclical sin/cos |
| 4. Lag & rolling | demand lags (1/7/14/28) and trailing mean/std/max (7/14/28), **all shifted** so row *t* sees only *t−1* and earlier |
| 5. Price features | current price + lagged price + price change |
| 6. Leakage self-check | recomputes a feature the leaky way and asserts ours differs |

**Key decisions:** completing the calendar is the most-skipped step (a no-sale
day is genuinely zero demand, but absent from the aggregate — without
zero-filling, `lag_1` silently reaches across gaps); demand fills with 0 but
**price fills with NaN** (no price on a no-sale day); every demand-derived
feature is shifted (`s.shift(1).rolling(7).mean()` — the `shift(1)` is what
stops the target leaking into its own feature); features grouped by what's
knowable when (known-in-advance calendar vs derived-from-past lags vs
retailer-set price).

---

## Stage 3 — Forecasting models

`models.py` trains four forecasters, dumb to sophisticated, each justifying its
complexity against the previous one, evaluated on a **held-out temporal block**
(train on the past, test on the future — never the reverse).

| Model | WAPE | vs baseline |
|-------|------|-------------|
| Seasonal-naive (same weekday last week) | 0.578 | — |
| Prophet (per-series decomposition) | 0.429 | 26% better |
| **LightGBM (global, on engineered features)** | **0.380** | **34% better** |
| Ensemble (validation-tuned blend) | 0.380 | collapsed to LightGBM |

**Key decisions:**

- **Leakage-safe feature selection.** Of 42 feature columns, only 36 are legal
  model inputs. `revenue`, `txn_count`, `refund_units`, `outlier_txns` are
  *realized on the target day* — using them to predict demand is leakage. Only
  their lagged versions are kept.
- **WAPE, not MAPE.** ~7% of store/product/days have zero demand, which makes
  MAPE (which divides by the actual) undefined/explosive. WAPE
  (`sum|error| / sum|actual|`) pools error over volume and is the retail
  standard for intermittent demand.
- **Honest ensemble.** The validation-tuned blend collapsed to 100% LightGBM —
  the data didn't support a blend, so the single model is the honest answer.

---

## Stage 4 — Robustness & backtesting

`backtest.py` stress-tests the Stage-3 result two ways.

### Expanding-window (rolling-origin) backtest

Retrain at 5 cutoffs marching forward through time; train expands each fold,
test blocks are non-overlapping. Produces a *distribution* of WAPE, not one
possibly-lucky number.

| Fold | Test window | Baseline WAPE | LightGBM WAPE | Improvement |
|------|-------------|---------------|---------------|-------------|
| 1 | Aug 2024 | 0.511 | 0.345 | +32.4% |
| 2 | Sep 2024 | 0.527 | 0.358 | +32.1% |
| 3 | Oct 2024 | 0.567 | 0.371 | +34.5% |
| 4 | Nov 2024 | 0.565 | 0.369 | +34.7% |
| 5 | Dec 2024 | 0.586 | 0.384 | +34.5% |

**LightGBM wins 5/5 folds; mean improvement 33.6% (±1.3%).** The tight spread
is the point — the gain is stable, not a lucky window.

### Feature-group ablation

Drop one feature group at a time and watch WAPE move (larger positive Δ = that
group mattered more).

| Variant | WAPE | Δ vs full |
|---------|------|-----------|
| FULL (all features) | 0.384 | — |
| drop lags | 0.382 | −0.002 |
| drop rolling | 0.382 | −0.002 |
| drop calendar | 0.391 | +0.007 |
| drop current-day price | 0.417 | +0.033 |
| calendar-only (no history) | 0.417 | +0.033 |

**Two honest findings (kept visible, not hidden):**

1. **Demand-history features add little *on this synthetic data*.** The
   generator builds demand as noise around a deterministic seasonal signal with
   no autoregressive persistence, so once calendar features capture the
   seasonal mean, lags are redundant. On real POS data — where demand has
   genuine momentum (busy stays busy, promotions persist, stockouts depress the
   next day) — lags would carry real signal. The pipeline is built to exploit
   them when they do.

2. **The model leans on current-day price**, treated as known-in-advance (the
   retailer sets it). Worst case — drop it entirely and use only lagged price —
   WAPE 0.417 **still beats the baseline (0.586) by 29%.** The model's value
   survives the worst-case assumption.

---

## Running it

```bash
pip install pandas numpy pytz holidays lightgbm prophet
cd src
python generate_data.py        # raw dirty data + dirtiness ledger
python data_cleaning.py        # clean_transactions.csv + quarantine.csv + report
python feature_engineering.py  # features.csv + feature report
python models.py               # model_results.txt
python backtest.py             # backtest_results.txt + backtest_folds.csv
```

---

## Roadmap

- ~~**Stage 1 — Data generation + cleaning**~~ ✅
- ~~**Stage 2 — Feature engineering**~~ ✅
- ~~**Stage 3 — Forecasting models**~~ ✅
- ~~**Stage 4 — Robustness & backtesting**~~ ✅
- **Stage 5 — Monitoring**: WAPE by store/SKU segment over time, drift
  detection, retrain triggers.

---

## Project structure
retail_forecasting/

├── src/

│   ├── generate_data.py         # dirty synthetic POS generator

│   ├── data_cleaning.py         # 10-stage cleaning pipeline

│   ├── feature_engineering.py   # leakage-safe daily feature matrix

│   ├── models.py                # forecasters + temporal evaluation

│   └── backtest.py              # expanding-window backtest + ablation

├── data/                        # generated CSVs (gitignored in practice)

└── reports/

├── data_quality_report.txt

├── feature_report.txt

├── model_results.txt

├── backtest_results.txt

└── backtest_folds.csv

---

## Notes / lessons

- **Mixed-timezone datetime bug:** a single pandas datetime column can't hold
  mixed UTC offsets — it silently coerces to NaT. Deriving the local *date*
  per timezone group fixes it. (This had been nulling 75% of business dates.)
- **Vectorize over row loops:** the first generator looped per-transaction and
  timed out; vectorizing per store-day made it tractable.
- **Leakage is the forecasting failure mode:** every rolling feature is
  shifted, the model feature list excludes same-day realized columns, and a
  self-check guards against a missing `.shift()`.
- **Ablation earns its keep:** robustness checks surfaced that lags are
  redundant on synthetic data and that the model leans on current-day price —
  both found and quantified before any reviewer could.
