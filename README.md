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
