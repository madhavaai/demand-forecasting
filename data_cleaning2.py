"""
data_cleaning.py
================
Production-grade cleaning pipeline for the raw POS feed. The guiding
principles, in order of importance:

  1. NEVER silently drop a row. Every rejected record is moved to a
     quarantine table tagged with the exact reason. That gives full data
     lineage — an auditor (or future you) can ask "why is yesterday's
     fuel total lower than expected?" and trace it to specific quarantined
     rows, not a black hole.

  2. PRESERVE the raw mess on ingestion. We read every column as a string
     so pandas can't silently coerce "N/A" into NaN before we've logged it.
     Coercion is an explicit, counted step, not an accident of read_csv.

  3. DISTINGUISH errors from legitimate edge cases. A negative amount is a
     refund (keep, label it), not corruption. A negative *gallon count of
     -800* is corruption. The pipeline must know the business to tell them
     apart.

  4. RECONCILE at the end. Because the generator wrote a ground-truth
     ledger of what it broke, we check that the cleaner caught it.

The pipeline emits:
  - clean_transactions.csv     (rows that passed every gate, fully typed)
  - quarantine.csv             (rejected rows + reason, for lineage/audit)
  - data_quality_report.txt    (stage-by-stage funnel + reconciliation)
"""

import json
import numpy as np
import pandas as pd

DATA_DIR = "data"
REPORT_DIR = "reports"

# Domain bounds — these come from knowing the business, not from the data.
MAX_FUEL_GALLONS = 60          # no consumer vehicle tank exceeds ~50 gal; 60 is generous
MAX_PRICE_PER_GALLON = 8.0     # sanity ceiling on fuel price
MAX_MERCH_UNIT_PRICE = 100.0   # ceiling on a single convenience item
MAX_MERCH_QTY = 50             # nobody buys 300 sodas in one swipe
AMOUNT_RECON_TOLERANCE = 0.02  # |amount - qty*price| must be within 2 cents

# Canonical category labels — every messy variant maps to one of these
PRODUCT_CANON = {
    "regular_unleaded": "regular_unleaded", "regular unleaded": "regular_unleaded",
    "reg unleaded": "regular_unleaded",
    "midgrade": "midgrade", "mid_grade": "midgrade", "mid grade": "midgrade",
    "premium": "premium",
    "diesel": "diesel",
    "beverages": "beverages", "beverage": "beverages",
    "snacks": "snacks", "snack": "snacks",
    "tobacco": "tobacco",
    "lottery": "lottery", "lotto": "lottery",
    "car_wash": "car_wash", "car wash": "car_wash", "carwash": "car_wash",
}
VALID_PAYMENTS = {"credit", "debit", "cash", "mobile"}
DATE_START = pd.Timestamp("2023-01-01", tz="UTC")
DATE_END = pd.Timestamp("2025-01-02", tz="UTC")  # allow 1-day slack past 2024-12-31


class CleaningPipeline:
    def __init__(self):
        self.quarantine = []   # list of (DataFrame, reason)
        self.stages = []       # (stage_name, rows_remaining) funnel
        self.notes = {}        # freeform counts for the report

    # -- helper: move a boolean-selected subset into quarantine -----------
    def _quarantine(self, df, mask, reason):
        mask = np.asarray(mask)
        bad = df[mask].copy()
        if len(bad):
            bad["_reject_reason"] = reason
            self.quarantine.append(bad)
        # reset index so downstream Series/merges stay aligned (avoids a
        # nasty class of silent misalignment bugs after row removal)
        return df[~mask].copy().reset_index(drop=True)

    def _record_stage(self, name, df):
        self.stages.append((name, len(df)))

    # ====================================================================
    # STAGE 0 — Ingestion: read everything as string, preserve raw values
    # ====================================================================
    def ingest(self):
        # keep_default_na=False + dtype=str => "N/A", "" stay as literal
        # strings instead of being silently turned into NaN.
        df = pd.read_csv(
            f"{DATA_DIR}/pos_transactions.csv",
            dtype=str,
            keep_default_na=False,
            na_values=[],          # nothing is auto-NaN; we decide later
        )
        self.stores = pd.read_csv(f"{DATA_DIR}/stores.csv", dtype={"store_id": str})
        self._record_stage("0. ingested raw", df)
        self.notes["raw_rows"] = len(df)
        return df

    # ====================================================================
    # STAGE 1 — Structural / schema validation
    # ====================================================================
    def validate_schema(self, df):
        expected = {
            "transaction_id", "store_id", "product", "category", "quantity",
            "unit_price", "amount", "payment_type", "loyalty_id", "transaction_ts",
        }
        missing = expected - set(df.columns)
        if missing:
            raise ValueError(f"Schema violation: missing columns {missing}")
        self.notes["columns_ok"] = True
        # treat literal empty strings as true missing from here on
        df = df.replace({"": np.nan, "None": np.nan})
        return df

    # ====================================================================
    # STAGE 2 — Exact duplicate removal
    #   Dedupe on the full business key. transaction_id should be unique;
    #   a repeated id with identical payload is a retry/double-post.
    # ====================================================================
    def dedupe(self, df):
        before = len(df)
        # full-row duplicates first
        df = df.drop_duplicates()
        full_dupes = before - len(df)
        # then duplicate transaction_ids that survived (same id, diff payload)
        id_dupes_mask = df.duplicated(subset=["transaction_id"], keep="first")
        df = self._quarantine(df, id_dupes_mask, "duplicate_transaction_id")
        self.notes["exact_duplicates_removed"] = full_dupes
        self.notes["conflicting_id_duplicates_quarantined"] = int(id_dupes_mask.sum())
        self._record_stage("2. de-duplicated", df)
        return df

    # ====================================================================
    # STAGE 3 — String normalization (trim, case-fold, canonicalize)
    # ====================================================================
    def normalize_strings(self, df):
        # product: trim, lowercase, collapse internal spaces, map to canon
        prod = (df["product"].astype(str)
                .str.strip().str.lower().str.replace(r"\s+", " ", regex=True))
        df["product_raw"] = df["product"]          # keep original for lineage
        df["product"] = prod.map(PRODUCT_CANON)
        unmapped = df["product"].isna() & df["product_raw"].notna()
        self.notes["unmapped_product_labels"] = int(unmapped.sum())
        # any product we couldn't canonicalize is quarantined (unknown SKU)
        df = self._quarantine(df, unmapped, "unmappable_product_label")

        # payment_type: trim + lowercase
        df["payment_type"] = df["payment_type"].astype(str).str.strip().str.lower()
        bad_pay = ~df["payment_type"].isin(VALID_PAYMENTS) & df["payment_type"].notna()
        # don't drop on this — just standardize unknowns to 'other'
        df.loc[bad_pay, "payment_type"] = "other"
        self.notes["payment_types_coerced_to_other"] = int(bad_pay.sum())

        # category: trim + lowercase
        df["category"] = df["category"].astype(str).str.strip().str.lower()

        self._record_stage("3. strings normalized", df)
        return df

    # ====================================================================
    # STAGE 4 — Type coercion (string -> numeric), counting failures
    # ====================================================================
    def coerce_types(self, df):
        for col in ["quantity", "unit_price", "amount"]:
            coerced = pd.to_numeric(df[col], errors="coerce")
            failed = coerced.isna() & df[col].notna()   # was non-null, became NaN
            self.notes[f"{col}_coercion_failures"] = int(failed.sum())
            df[col] = coerced
        self._record_stage("4. types coerced", df)
        return df

    # ====================================================================
    # STAGE 5 — Timestamp parsing -> single UTC instant + store-local time
    #   This is the hard one. Three formats, two with no tz info.
    #   Strategy:
    #     - strings ending 'Z' are UTC-aware
    #     - everything else is naive STORE-LOCAL, localized via the store's
    #       timezone (joined from the dimension table)
    #     - convert all to UTC; derive local wall-clock for feature work
    #     - flag out-of-range / clock-skewed timestamps
    # ====================================================================
    def parse_timestamps(self, df):
        df = df.reset_index(drop=True)
        ts = df["transaction_ts"].astype(str)
        is_utc = ts.str.endswith("Z")

        # --- UTC-aware rows ---
        utc_parsed = pd.to_datetime(ts.where(is_utc), format="mixed", utc=True, errors="coerce")

        # --- naive rows: parse without tz, then localize per store ---
        naive_parsed = pd.to_datetime(ts.where(~is_utc), format="mixed", errors="coerce")

        # join store timezone for localization
        df = df.merge(self.stores[["store_id", "timezone"]], on="store_id", how="left")

        # localize naive times by store tz, then convert to UTC.
        # Done per-timezone group for correctness (DST handled by tz database).
        naive_utc = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")
        naive_mask = ~is_utc & naive_parsed.notna() & df["timezone"].notna()
        for tz, grp in df[naive_mask].groupby("timezone"):
            idx = grp.index
            local = naive_parsed.loc[idx]
            # ambiguous/nonexistent (DST) times: shift forward, infer
            localized = (local.dt.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
                              .dt.tz_convert("UTC"))
            naive_utc.loc[idx] = localized

        # combine
        df["ts_utc"] = utc_parsed.where(is_utc, naive_utc)

        # unparseable timestamps -> quarantine
        unparseable = df["ts_utc"].isna()
        self.notes["timestamps_unparseable"] = int(unparseable.sum())
        df = self._quarantine(df, unparseable, "unparseable_timestamp")

        # out-of-range / clock-skew (future or pre-history) -> quarantine
        oob = (df["ts_utc"] < DATE_START) | (df["ts_utc"] > DATE_END)
        self.notes["timestamps_out_of_range"] = int(oob.sum())
        df = self._quarantine(df, oob, "timestamp_out_of_range")

        # derive store-local wall clock for downstream daily aggregation.
        # We aggregate demand by LOCAL business day, not UTC day — a sale at
        # 11pm Pacific belongs to that store's day, not the next UTC day.
        #
        # IMPORTANT pandas gotcha: a single datetime column CANNOT hold mixed
        # UTC offsets. Our stores span 4 timezones, so we must NOT try to build
        # one tz-aware "ts_local" datetime column — pandas would coerce the
        # mixed-offset rows to NaT. Instead we compute the local business DATE
        # (a plain date, offset-free) per timezone group, plus keep ts_local as
        # a formatted STRING for display/lineage only.
        business_date = pd.Series(index=df.index, dtype="object")
        ts_local_str = pd.Series(index=df.index, dtype="object")
        for tz, grp in df.groupby("timezone"):
            idx = grp.index
            loc = df.loc[idx, "ts_utc"].dt.tz_convert(tz)
            business_date.loc[idx] = loc.dt.date
            ts_local_str.loc[idx] = loc.dt.strftime("%Y-%m-%d %H:%M:%S%z")
        df["ts_local"] = ts_local_str          # string, display only
        df["business_date"] = pd.to_datetime(business_date)  # uniform, tz-naive date

        self._record_stage("5. timestamps parsed", df)
        return df

    # ====================================================================
    # STAGE 6 — Referential integrity (store_id must exist in dimension)
    # ====================================================================
    def check_referential_integrity(self, df):
        # null store_id can't be attributed to a store -> quarantine
        null_store = df["store_id"].isna()
        self.notes["null_store_id"] = int(null_store.sum())
        df = self._quarantine(df, null_store, "null_store_id")

        valid_ids = set(self.stores["store_id"])
        orphan = ~df["store_id"].isin(valid_ids)
        self.notes["orphan_store_id"] = int(orphan.sum())
        df = self._quarantine(df, orphan, "orphan_store_id_not_in_dimension")

        self._record_stage("6. referential integrity", df)
        return df

    # ====================================================================
    # STAGE 7 — Business-rule validation
    #   - separate refunds (legit negatives) and TAG them, don't drop
    #   - quarantine impossible negatives (e.g., -800 gallons that aren't
    #     a clean refund pair)
    #   - quarantine zero-quantity scanner glitches
    #   - reconcile amount == qty*unit_price within tolerance
    # ====================================================================
    def apply_business_rules(self, df):
        # refunds: amount<0 AND quantity<0 together => legitimate refund
        refund = (df["amount"] < 0) & (df["quantity"] < 0)
        df["is_refund"] = refund
        self.notes["refunds_labeled"] = int(refund.sum())

        # half-negative (only one of amount/qty negative) => corruption
        half_neg = ((df["amount"] < 0) ^ (df["quantity"] < 0)) & df["amount"].notna() & df["quantity"].notna()
        self.notes["inconsistent_sign_quarantined"] = int(half_neg.sum())
        df = self._quarantine(df, half_neg, "inconsistent_sign_amount_vs_quantity")

        # zero quantity (scanner glitch — no product actually moved)
        zero_qty = (df["quantity"] == 0)
        self.notes["zero_quantity_quarantined"] = int(zero_qty.sum())
        df = self._quarantine(df, zero_qty, "zero_quantity")

        # null amount: try to repair from qty*unit_price; else quarantine
        null_amt = df["amount"].isna() & df["quantity"].notna() & df["unit_price"].notna()
        df.loc[null_amt, "amount"] = (df.loc[null_amt, "quantity"] * df.loc[null_amt, "unit_price"]).round(2)
        self.notes["null_amount_repaired"] = int(null_amt.sum())
        still_null = df["amount"].isna()
        self.notes["null_amount_unrepairable_quarantined"] = int(still_null.sum())
        df = self._quarantine(df, still_null, "null_amount_unrepairable")

        # amount reconciliation: |amount - qty*price| must be within tolerance.
        # Use absolute values so refunds reconcile too.
        expected_amt = (df["quantity"] * df["unit_price"]).abs().round(2)
        recon_gap = (df["amount"].abs() - expected_amt).abs()
        mismatch = recon_gap > AMOUNT_RECON_TOLERANCE
        self.notes["amount_mismatch_quarantined"] = int(mismatch.sum())
        df = self._quarantine(df, mismatch, "amount_quantity_price_mismatch")

        self._record_stage("7. business rules", df)
        return df

    # ====================================================================
    # STAGE 8 — Outlier detection
    #   Two layers:
    #     (a) HARD domain bounds (physically impossible) -> quarantine
    #     (b) STATISTICAL IQR flag within category (suspicious, not impossible)
    #         -> keep but flag, so forecasting can optionally down-weight
    # ====================================================================
    def detect_outliers(self, df):
        work = df[~df["is_refund"]].copy()  # judge outliers on sales, not refunds
        q = work["quantity"].abs()
        p = work["unit_price"].abs()

        is_fuel = work["category"] == "fuel"
        is_merch = work["category"] == "merchandise"

        # (a) hard physical bounds
        bad_fuel_gal = is_fuel & (q > MAX_FUEL_GALLONS)
        bad_fuel_price = is_fuel & (p > MAX_PRICE_PER_GALLON)
        bad_merch_price = is_merch & (p > MAX_MERCH_UNIT_PRICE)
        bad_merch_qty = is_merch & (q > MAX_MERCH_QTY)
        hard_bad_idx = work[bad_fuel_gal | bad_fuel_price | bad_merch_price | bad_merch_qty].index

        hard_mask = df.index.isin(hard_bad_idx)
        self.notes["hard_bound_outliers_quarantined"] = int(hard_mask.sum())
        df = self._quarantine(df, hard_mask, "outlier_exceeds_physical_bound")

        # (b) statistical IQR flag, computed PER category on what remains.
        # Flag (don't drop) — these are plausible but extreme; forecasting
        # can choose to winsorize or down-weight them.
        df["is_stat_outlier"] = False
        for cat, grp in df.groupby("category"):
            vals = grp["amount"].abs()
            q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
            iqr = q3 - q1
            hi = q3 + 1.5 * iqr
            lo = q1 - 1.5 * iqr
            flagged = grp.index[(vals > hi) | (vals < lo)]
            df.loc[flagged, "is_stat_outlier"] = True
        self.notes["statistical_outliers_flagged"] = int(df["is_stat_outlier"].sum())

        self._record_stage("8. outliers handled", df)
        return df

    # ====================================================================
    # STAGE 9 — Finalize: select columns, set types, sort
    # ====================================================================
    def finalize(self, df):
        keep = [
            "transaction_id", "store_id", "ts_utc", "ts_local", "business_date",
            "product", "category", "quantity", "unit_price", "amount",
            "payment_type", "loyalty_id", "is_refund", "is_stat_outlier",
        ]
        df = df[keep].sort_values("ts_utc").reset_index(drop=True)
        self._record_stage("9. final clean", df)
        return df

    # ====================================================================
    # Reporting
    # ====================================================================
    def build_quarantine_table(self):
        if not self.quarantine:
            return pd.DataFrame()
        return pd.concat(self.quarantine, ignore_index=True)

    def write_report(self, clean, quar):
        lines = []
        lines.append("=" * 70)
        lines.append("DATA QUALITY REPORT — POS transaction cleaning")
        lines.append("=" * 70)

        lines.append("\nFUNNEL (rows surviving each stage):")
        for name, n in self.stages:
            lines.append(f"  {name:32s} {n:>10,}")

        raw = self.notes["raw_rows"]
        kept = len(clean)
        lines.append(f"\n  Retention: {kept:,} / {raw:,} = {100*kept/raw:.2f}% kept")
        lines.append(f"  Quarantined: {len(quar):,} rows")

        lines.append("\nQUARANTINE BREAKDOWN (by reason):")
        if len(quar):
            for reason, cnt in quar["_reject_reason"].value_counts().items():
                lines.append(f"  {reason:42s} {cnt:>8,}")

        lines.append("\nIN-PLACE REPAIRS / RELABELS (kept, not dropped):")
        for key in ["exact_duplicates_removed", "payment_types_coerced_to_other",
                    "null_amount_repaired", "refunds_labeled",
                    "statistical_outliers_flagged"]:
            lines.append(f"  {key:42s} {self.notes.get(key, 0):>8,}")

        # Reconciliation vs the generator's ground-truth ledger
        lines.append("\nRECONCILIATION vs ground-truth dirtiness ledger:")
        try:
            with open(f"{DATA_DIR}/dirtiness_ledger.json") as f:
                ledger = json.load(f)
            checks = [
                ("exact_duplicates_injected", "exact_duplicates_removed"),
                ("refunds_injected", "refunds_labeled"),
                ("orphan_store_id_injected", None),  # special: in quarantine
                ("zero_quantity_injected", "zero_quantity_quarantined"),
                ("null_amount_injected", None),
                ("amount_mismatch_injected", "amount_mismatch_quarantined"),
            ]
            qcounts = quar["_reject_reason"].value_counts().to_dict() if len(quar) else {}
            lines.append(f"  {'injected':38s}{'caught':>10s}")
            lines.append(f"  exact_duplicates       {ledger['exact_duplicates_injected']:>10,}"
                         f"  ->  removed {self.notes['exact_duplicates_removed']:,}")
            lines.append(f"  refunds                {ledger['refunds_injected']:>10,}"
                         f"  ->  labeled {self.notes['refunds_labeled']:,}")
            lines.append(f"  orphan_store_id        {ledger['orphan_store_id_injected']:>10,}"
                         f"  ->  quarantined {qcounts.get('orphan_store_id_not_in_dimension',0):,}")
            lines.append(f"  zero_quantity          {ledger['zero_quantity_injected']:>10,}"
                         f"  ->  quarantined {qcounts.get('zero_quantity',0):,}")
            lines.append(f"  amount_mismatch        {ledger['amount_mismatch_injected']:>10,}"
                         f"  ->  quarantined {qcounts.get('amount_quantity_price_mismatch',0):,}")
            lines.append("\n  (Note: some injected issues overlap on the same row, and the")
            lines.append("   first failing gate claims it — so counts won't match 1:1. That's")
            lines.append("   expected: a row that's both a duplicate and a mismatch is caught")
            lines.append("   once, at the earliest gate.)")
        except FileNotFoundError:
            lines.append("  (ledger not found — skipping reconciliation)")

        report = "\n".join(lines)
        import os
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(f"{REPORT_DIR}/data_quality_report.txt", "w") as f:
            f.write(report)
        return report

    # ====================================================================
    # Orchestration
    # ====================================================================
    def run(self):
        df = self.ingest()
        df = self.validate_schema(df)
        df = self.dedupe(df)
        df = self.normalize_strings(df)
        df = self.coerce_types(df)
        df = self.parse_timestamps(df)
        df = self.check_referential_integrity(df)
        df = self.apply_business_rules(df)
        df = self.detect_outliers(df)
        clean = self.finalize(df)

        quar = self.build_quarantine_table()
        clean.to_csv(f"{DATA_DIR}/clean_transactions.csv", index=False)
        if len(quar):
            quar.to_csv(f"{DATA_DIR}/quarantine.csv", index=False)

        report = self.write_report(clean, quar)
        print(report)
        return clean, quar


if __name__ == "__main__":
    CleaningPipeline().run()
