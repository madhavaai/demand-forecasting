"""
generate_data.py
================
Generates synthetic POS (point-of-sale) transaction data for a fuel /
convenience retailer, modeled on the Murphy USA environment: many stores,
fuel sold by the gallon + in-store merchandise, two years of daily history
with realistic trend, weekly seasonality, summer driving-season effects,
and holiday spikes.

The data is generated DIRTY ON PURPOSE. Real POS feeds arrive with timezone
chaos, terminal clock skew, retry-duplicate rows, fat-finger price entries,
refunds (negative amounts), referential breaks, and inconsistent category
labels. We inject each of these at a known, controlled rate so the cleaning
pipeline downstream has genuine, documented work to do — and so we can verify
the cleaner actually catches what we planted.

Two tables are produced:
  1. stores.csv        — the store dimension (clean reference table)
  2. pos_transactions.csv — the raw, dirty transaction fact table

Everything is reproducible via RANDOM_SEED.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Configuration — tune these to trade realism against runtime / memory
# ---------------------------------------------------------------------------
N_STORES = 25
DATE_START = pd.Timestamp("2023-01-01")
DATE_END = pd.Timestamp("2024-12-31")
AVG_TXN_PER_STORE_DAY = 30          # base transaction volume per store per day
OUTPUT_DIR = "data"

# Store types drive different demand profiles
STORE_TYPES = ["highway", "urban", "suburban"]
STORE_TYPE_WEIGHTS = [0.30, 0.35, 0.35]

# US timezones the chain operates across — the source of timezone messiness
TIMEZONES = ["America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles"]

# Product catalog: fuel sold per-gallon, merchandise sold per-unit
FUEL_PRODUCTS = {
    "regular_unleaded": (2.80, 4.20),   # (min, max) price per gallon range
    "midgrade":         (3.10, 4.50),
    "premium":          (3.40, 4.90),
    "diesel":           (3.20, 5.10),
}
MERCH_PRODUCTS = {
    "beverages":  (1.50, 4.00),
    "snacks":     (1.00, 6.00),
    "tobacco":    (6.00, 12.00),
    "lottery":    (1.00, 20.00),
    "car_wash":   (8.00, 15.00),
}
PAYMENT_TYPES = ["credit", "debit", "cash", "mobile"]
PAYMENT_WEIGHTS = [0.42, 0.30, 0.18, 0.10]

# US holidays where fuel demand spikes (travel) — month, day
TRAVEL_HOLIDAYS = [(7, 4), (11, 27), (12, 24), (5, 27), (9, 2)]  # approx Memorial/Labor


# ---------------------------------------------------------------------------
# 1. Build the store dimension table (this one stays CLEAN — it's the
#    reference table cleaning will validate transactions against)
# ---------------------------------------------------------------------------
def build_stores():
    rows = []
    for i in range(1, N_STORES + 1):
        stype = rng.choice(STORE_TYPES, p=STORE_TYPE_WEIGHTS)
        tz = rng.choice(TIMEZONES)
        # base daily volume multiplier by store type
        base_mult = {"highway": 1.6, "urban": 1.0, "suburban": 0.8}[stype]
        rows.append({
            "store_id": f"S{i:04d}",
            "store_type": stype,
            "timezone": tz,
            "volume_multiplier": round(base_mult * rng.uniform(0.85, 1.15), 3),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Demand signal — the "true" expected transaction count for a given
#    store on a given day, before noise. Trend + weekly + yearly + holiday.
# ---------------------------------------------------------------------------
def expected_txn_count(date, store):
    day_index = (date - DATE_START).days
    base = AVG_TXN_PER_STORE_DAY * store["volume_multiplier"]

    # slow upward trend (~6% per year)
    trend = 1.0 + 0.06 * (day_index / 365.0)

    # weekly seasonality: highway peaks weekends, urban peaks weekdays
    dow = date.dayofweek  # 0=Mon
    if store["store_type"] == "highway":
        weekly = 1.25 if dow >= 5 else 0.95
    elif store["store_type"] == "urban":
        weekly = 0.80 if dow >= 5 else 1.10
    else:  # suburban
        weekly = 1.05

    # yearly seasonality: summer driving season (peaks ~July)
    yearly = 1.0 + 0.20 * np.sin(2 * np.pi * (date.dayofyear - 80) / 365.0)

    # holiday travel spike
    holiday = 1.35 if (date.month, date.day) in TRAVEL_HOLIDAYS else 1.0

    return base * trend * weekly * yearly * holiday


# ---------------------------------------------------------------------------
# 3. Generate clean transactions following the demand signal
# ---------------------------------------------------------------------------
def generate_clean_transactions(stores):
    all_dates = pd.date_range(DATE_START, DATE_END, freq="D")
    fuel_names = list(FUEL_PRODUCTS.keys())
    merch_names = list(MERCH_PRODUCTS.keys())

    frames = []
    txn_counter = 0

    for _, store in stores.iterrows():
        # expected counts for every date at once
        lams = np.array([expected_txn_count(d, store) for d in all_dates])
        counts = rng.poisson(lams)              # transactions per day
        total = int(counts.sum())
        if total == 0:
            continue

        # explode dates according to per-day counts
        day_repeat = np.repeat(all_dates.values, counts)

        # random intraday seconds (5am-11pm) for each transaction
        secs = rng.integers(5 * 3600, 23 * 3600, size=total)
        local_dt = pd.to_datetime(day_repeat) + pd.to_timedelta(secs, unit="s")

        # fuel vs merchandise
        is_fuel = rng.random(total) < 0.65

        product = np.empty(total, dtype=object)
        category = np.empty(total, dtype=object)
        unit_price = np.empty(total, dtype=float)
        quantity = np.empty(total, dtype=float)

        # fuel rows
        fidx = np.where(is_fuel)[0]
        if len(fidx):
            fprods = rng.choice(fuel_names, size=len(fidx))
            product[fidx] = fprods
            category[fidx] = "fuel"
            # price by product range
            for p in fuel_names:
                sel = fidx[fprods == p]
                if len(sel):
                    pmin, pmax = FUEL_PRODUCTS[p]
                    unit_price[sel] = np.round(rng.uniform(pmin, pmax, len(sel)), 3)
            quantity[fidx] = np.round(rng.uniform(4, 28, len(fidx)), 3)

        # merch rows
        midx = np.where(~is_fuel)[0]
        if len(midx):
            mprods = rng.choice(merch_names, size=len(midx))
            product[midx] = mprods
            category[midx] = "merchandise"
            for p in merch_names:
                sel = midx[mprods == p]
                if len(sel):
                    pmin, pmax = MERCH_PRODUCTS[p]
                    unit_price[sel] = np.round(rng.uniform(pmin, pmax, len(sel)), 2)
            quantity[midx] = rng.integers(1, 5, len(midx)).astype(float)

        amount = np.round(quantity * unit_price, 2)

        ids = np.array([f"T{txn_counter + j + 1:09d}" for j in range(total)], dtype=object)
        txn_counter += total

        loyalty_mask = rng.random(total) < 0.40
        loyalty = np.where(
            loyalty_mask,
            np.array([f"L{v}" for v in rng.integers(100000, 999999, total)], dtype=object),
            None,
        )

        frames.append(pd.DataFrame({
            "transaction_id": ids,
            "store_id": store["store_id"],
            "store_timezone": store["timezone"],
            "local_datetime": local_dt,
            "product": product,
            "category": category,
            "quantity": quantity,
            "unit_price": unit_price,
            "amount": amount,
            "payment_type": rng.choice(PAYMENT_TYPES, size=total, p=PAYMENT_WEIGHTS),
            "loyalty_id": loyalty,
        }))

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 4. Convert clean local times into MESSY timestamps
#    Real feeds mix: UTC strings, local naive strings, different formats,
#    and terminals with skewed clocks. We simulate all of it.
# ---------------------------------------------------------------------------
def messify_timestamps(df):
    """Vectorized: convert clean local times into a mix of UTC strings, naive
    local strings, US-format strings, and clock-skewed UTC strings."""
    df = df.copy().reset_index(drop=True)
    n = len(df)

    # Map each store timezone to a fixed UTC offset (hours). We approximate
    # DST away here for speed; the AMBIGUITY this creates downstream is part
    # of the realism — naive local strings genuinely can't be resolved exactly.
    tz_offset_hours = {
        "America/New_York": -5,
        "America/Chicago": -6,
        "America/Denver": -7,
        "America/Los_Angeles": -8,
    }
    offsets = df["store_timezone"].map(tz_offset_hours)
    local = df["local_datetime"]
    utc = local - pd.to_timedelta(offsets, unit="h")  # naive local -> UTC

    roll = rng.random(n)
    out = np.empty(n, dtype=object)

    # 55%: proper UTC ISO string
    m = roll < 0.55
    out[m] = utc[m].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    # 25%: local naive string, no tz info
    m = (roll >= 0.55) & (roll < 0.80)
    out[m] = local[m].dt.strftime("%Y-%m-%d %H:%M:%S")
    # 12%: US M/D/Y format, local naive
    m = (roll >= 0.80) & (roll < 0.92)
    out[m] = local[m].dt.strftime("%m/%d/%Y %H:%M")
    # 8%: UTC with terminal clock skew (+/- up to 90 min)
    m = roll >= 0.92
    skew = pd.to_timedelta(rng.integers(-90, 90, size=m.sum()), unit="m")
    out[m] = (utc[m].reset_index(drop=True) + skew).dt.strftime("%Y-%m-%dT%H:%M:%SZ").values

    df["transaction_ts"] = out
    df = df.drop(columns=["local_datetime", "store_timezone"])
    return df


# ---------------------------------------------------------------------------
# 5. Inject the rest of the dirtiness at known rates
# ---------------------------------------------------------------------------
def inject_dirtiness(df):
    df = df.copy().reset_index(drop=True)
    n = len(df)
    log = {}

    # (a) Exact duplicate rows (retry/double-post) — 1.5%
    k = int(0.015 * n)
    dup_idx = rng.choice(n, size=k, replace=False)
    dupes = df.iloc[dup_idx].copy()
    log["exact_duplicates_injected"] = k

    # (b) Inconsistent category label casing/spelling — 4%
    k = int(0.04 * n)
    idx = rng.choice(n, size=k, replace=False)
    variants = {
        "regular_unleaded": ["Regular Unleaded", "REGULAR_UNLEADED", " regular_unleaded ", "Reg Unleaded"],
        "midgrade": ["MidGrade", "MIDGRADE", "mid_grade"],
        "premium": ["Premium", "PREMIUM", " premium"],
        "diesel": ["Diesel", "DIESEL", "diesel "],
        "beverages": ["Beverages", "BEVERAGE", "beverages "],
        "snacks": ["Snacks", "SNACK"],
        "tobacco": ["Tobacco", "TOBACCO "],
        "lottery": ["Lottery", "LOTTO"],
        "car_wash": ["Car Wash", "CARWASH", "car wash"],
    }
    for i in idx:
        p = df.at[i, "product"]
        if p in variants:
            df.at[i, "product"] = rng.choice(variants[p])
    log["category_label_variants_injected"] = k

    # (c) Null store_id (illegitimate missing key) — 0.3%
    k = int(0.003 * n)
    idx = rng.choice(n, size=k, replace=False)
    df.loc[idx, "store_id"] = None
    log["null_store_id_injected"] = k

    # (d) Null amount — 0.5%
    k = int(0.005 * n)
    idx = rng.choice(n, size=k, replace=False)
    df.loc[idx, "amount"] = np.nan
    log["null_amount_injected"] = k

    # (e) Refunds: legitimate negative amounts/quantities — 1.2%
    k = int(0.012 * n)
    idx = rng.choice(n, size=k, replace=False)
    df.loc[idx, "amount"] = -df.loc[idx, "amount"].abs()
    df.loc[idx, "quantity"] = -df.loc[idx, "quantity"].abs()
    log["refunds_injected"] = k

    # (f) Fat-finger price/amount outliers (impossible values) — 0.4%
    k = int(0.004 * n)
    idx = rng.choice(n, size=k, replace=False)
    for i in idx:
        if rng.random() < 0.5:
            df.at[i, "unit_price"] = float(df.at[i, "unit_price"]) * rng.uniform(50, 200)
        else:
            df.at[i, "quantity"] = float(abs(df.at[i, "quantity"])) * rng.uniform(50, 300)
        # amount left inconsistent on purpose (see check h)
    log["fatfinger_outliers_injected"] = k

    # (g) Zero quantity transactions (scanner glitch) — 0.2%
    k = int(0.002 * n)
    idx = rng.choice(n, size=k, replace=False)
    df.loc[idx, "quantity"] = 0
    log["zero_quantity_injected"] = k

    # (h) amount != quantity*unit_price mismatch (entry error) — 0.6%
    k = int(0.006 * n)
    idx = rng.choice(n, size=k, replace=False)
    df.loc[idx, "amount"] = (df.loc[idx, "amount"].astype(float) * rng.uniform(1.2, 2.5)).round(2)
    log["amount_mismatch_injected"] = k

    # (i) Referential break: store_id not in dimension table — 0.2%
    k = int(0.002 * n)
    idx = rng.choice(n, size=k, replace=False)
    df.loc[idx, "store_id"] = "S9999"  # nonexistent store
    log["orphan_store_id_injected"] = k

    # (j) String contamination in numeric amount ("N/A") — 0.2%
    df["amount"] = df["amount"].astype(object)
    k = int(0.002 * n)
    idx = rng.choice(n, size=k, replace=False)
    df.loc[idx, "amount"] = "N/A"
    log["string_in_amount_injected"] = k

    # (k) Whitespace + case noise in payment_type — 3%
    k = int(0.03 * n)
    idx = rng.choice(n, size=k, replace=False)
    for i in idx:
        pt = str(df.at[i, "payment_type"])
        df.at[i, "payment_type"] = rng.choice([pt.upper(), f" {pt} ", pt.capitalize()])
    log["payment_type_noise_injected"] = k

    # finally append the exact duplicates and shuffle
    df = pd.concat([df, dupes], ignore_index=True)
    df = df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    return df, log


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    import os
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Building store dimension...")
    stores = build_stores()
    stores.to_csv(f"{OUTPUT_DIR}/stores.csv", index=False)
    print(f"  {len(stores)} stores written to {OUTPUT_DIR}/stores.csv")

    print("Generating clean transactions (this is the slow part)...")
    clean = generate_clean_transactions(stores)
    print(f"  {len(clean):,} clean transactions generated")

    print("Messifying timestamps (timezones, formats, clock skew)...")
    clean = messify_timestamps(clean)

    print("Injecting controlled dirtiness...")
    dirty, log = inject_dirtiness(clean)

    dirty.to_csv(f"{OUTPUT_DIR}/pos_transactions.csv", index=False)
    print(f"  {len(dirty):,} rows (incl. duplicates) written to {OUTPUT_DIR}/pos_transactions.csv")

    print("\n--- Dirtiness ledger (ground truth for what cleaning should catch) ---")
    for k, v in log.items():
        print(f"  {k:38s}: {v:,}")

    # persist the ledger so cleaning can be validated against it
    pd.Series(log).to_json(f"{OUTPUT_DIR}/dirtiness_ledger.json")
    print(f"\nLedger saved to {OUTPUT_DIR}/dirtiness_ledger.json")


if __name__ == "__main__":
    main()
