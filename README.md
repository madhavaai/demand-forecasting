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
