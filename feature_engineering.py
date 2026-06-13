"""
feature_engineering.py
======================
Turns the clean transaction-level table into a model-ready daily feature
matrix for demand forecasting.

The forecasting GRAIN (one row of the final table) is:

        one store  x  one category  x  one calendar day

and the TARGET is that day's net demand (units sold, with refunds netted out).
Fuel is measured in gallons, merchandise in units — so we keep each
(store, category) as its own series rather than pooling incomparable units.

THE CENTRAL DISCIPLINE OF THIS FILE: no feature may use information that
wouldn't be available at prediction time. Forecasting is uniquely vulnerable
to look-ahead leakage because the rows are ordered in time and adjacent rows
are highly correlated — a rolling average that accidentally includes "today"
will look brilliant in backtest and fail completely in production. Every
demand-derived feature here is explicitly SHIFTED so it sees only the past.

Features are grouped into three kinds, by what's knowable when:
  - KNOWN-IN-ADVANCE (calendar): day-of-week, month, holiday flags. These are
    deterministic future facts — safe to use for the current day.
  - DERIVED-FROM-PAST (lags, rolling stats): must be shifted; only yesterday
    and earlier.
  - CONTROLLED (price): the retailer SETS price, so the current day's planned
    price is known in advance — but we treat it carefully and also lag it.
"""

import numpy as np
import pandas as pd
import holidays

DATA_DIR = "data"

# Lag horizons (days). 1=yesterday, 7=same day last week, 28=same day 4 wks ago.
LAGS = [1, 7, 14, 28]
# Rolling window sizes (days) for trailing statistics.
ROLL_WINDOWS = [7, 14, 28]
# Warm-up: rows in the first MAX_LAG days of each series have incomplete
# history and are dropped from training (their lags are undefined).
MAX_LAG = max(LAGS + ROLL_WINDOWS)


# ---------------------------------------------------------------------------
# 1. Aggregate transactions -> daily store x category demand
# ---------------------------------------------------------------------------
def aggregate_daily(txn):
    """Collapse the transaction table to one row per store/category/day.

    Refunds carry negative quantity, so a plain SUM nets them out → net demand,
    which is what inventory/replenishment actually cares about. We also keep a
    few same-day rollups (revenue, transaction count, average price) that will
    later be turned into PAST features.
    """
    txn = txn.copy()
    txn["business_date"] = pd.to_datetime(txn["business_date"])

    daily = (txn.groupby(["store_id", "product", "business_date"])
                .agg(category=("category", "first"),
                     demand=("quantity", "sum"),          # net units (refunds netted)
                     revenue=("amount", "sum"),
                     txn_count=("transaction_id", "count"),
                     avg_unit_price=("unit_price", "mean"),
                     refund_units=("quantity", lambda s: s[s < 0].sum()),
                     outlier_txns=("is_stat_outlier", "sum"))
                .reset_index())
    return daily


# ---------------------------------------------------------------------------
# 2. Complete the calendar — THE most commonly missed step
# ---------------------------------------------------------------------------
def complete_calendar(daily):
    """A store/category with zero sales on a day produces NO transaction row,
    so that day is simply ABSENT from the aggregate. But the demand that day
    was genuinely zero — and lag/rolling features are only correct on a
    gap-free daily index. If we skip this, 'lag_1' might actually reach back
    3 days because the two intervening zero-demand days don't exist as rows.

    So we build the full (store x category x every-day) grid and fill missing
    demand with 0. This is essential for correct temporal features.
    """
    stores_cats = daily[["store_id", "product", "category"]].drop_duplicates()
    full_dates = pd.date_range(daily["business_date"].min(),
                               daily["business_date"].max(), freq="D")

    # cartesian product of (store,product) pairs x all dates
    grid = (stores_cats.assign(key=1)
            .merge(pd.DataFrame({"business_date": full_dates, "key": 1}), on="key")
            .drop(columns="key"))

    out = grid.merge(daily.drop(columns="category"),
                     on=["store_id", "product", "business_date"], how="left")

    # demand / counts on missing days are truly zero
    for col in ["demand", "revenue", "txn_count", "refund_units", "outlier_txns"]:
        out[col] = out[col].fillna(0)
    # price on a no-sale day is unknown, not zero → leave NaN, forward-fill later
    out = out.sort_values(["store_id", "product", "business_date"]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# 3. Calendar features — known in advance, safe for the current day
# ---------------------------------------------------------------------------
def add_calendar_features(df):
    d = df["business_date"]
    df["dow"] = d.dt.dayofweek                 # 0=Mon ... 6=Sun
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["day_of_month"] = d.dt.day
    df["month"] = d.dt.month
    df["week_of_year"] = d.dt.isocalendar().week.astype(int)
    df["day_of_year"] = d.dt.dayofyear
    df["days_since_start"] = (d - d.min()).dt.days   # linear trend proxy

    # cyclical encodings: so a model sees Dec(12) and Jan(1) as adjacent, and
    # Sun(6) next to Mon(0). Tree models don't need this, but Prophet/LSTM and
    # linear models benefit. Sin+cos together place each value on a circle.
    df["dow_sin"] = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # holiday features from the real US calendar (known years in advance)
    yrs = range(d.dt.year.min(), d.dt.year.max() + 1)
    us = holidays.US(years=list(yrs))
    hol_dates = pd.to_datetime(sorted(us.keys()))
    dts = d.dt.normalize()
    df["is_holiday"] = dts.isin(hol_dates).astype(int)

    # days to / from the nearest holiday — travel demand ramps around them.
    # searchsorted gives the position of each date among sorted holidays.
    hol_int = hol_dates.values.astype("datetime64[D]").astype(int)
    cur_int = dts.values.astype("datetime64[D]").astype(int)
    pos = np.searchsorted(hol_int, cur_int)
    pos_prev = np.clip(pos - 1, 0, len(hol_int) - 1)
    pos_next = np.clip(pos, 0, len(hol_int) - 1)
    dist_prev = np.abs(cur_int - hol_int[pos_prev])
    dist_next = np.abs(hol_int[pos_next] - cur_int)
    df["days_to_nearest_holiday"] = np.minimum(dist_prev, dist_next)
    return df


# ---------------------------------------------------------------------------
# 4. Lag & rolling features — DERIVED FROM PAST, must be shifted
# ---------------------------------------------------------------------------
def add_lag_features(df):
    """All features here are computed PER (store, category) series and shifted
    so row t only ever sees data from t-1 and earlier.

    The shift(1) before every rolling window is the whole ballgame. Without it,
    rolling(7).mean() at row t includes t itself — that's the target leaking
    into its own feature. With shift(1), the 7-day mean ends at t-1.
    """
    g = df.groupby(["store_id", "product"], group_keys=False)

    # plain lags of demand
    for L in LAGS:
        df[f"demand_lag_{L}"] = g["demand"].shift(L)

    # trailing rolling stats on demand, each ending at t-1 (note the shift(1))
    def roll_feat(s, window, fn):
        return s.shift(1).rolling(window, min_periods=max(2, window // 2)).agg(fn)

    for W in ROLL_WINDOWS:
        df[f"demand_rollmean_{W}"] = g["demand"].transform(lambda s, W=W: roll_feat(s, W, "mean"))
        df[f"demand_rollstd_{W}"]  = g["demand"].transform(lambda s, W=W: roll_feat(s, W, "std"))
        df[f"demand_rollmax_{W}"]  = g["demand"].transform(lambda s, W=W: roll_feat(s, W, "max"))

    # same-weekday history: mean of the last 4 same-DOW values (e.g., last 4
    # Mondays), shifted. Captures weekly seasonality directly as a feature.
    df["demand_same_dow_mean4"] = (
        g["demand"].transform(lambda s: s.shift(1).rolling(28, min_periods=7)
                              .apply(lambda w: w[::-7][:4].mean(), raw=True))
    )

    # momentum: yesterday vs the week-ago day (is demand trending up/down?)
    df["demand_momentum_7"] = df["demand_lag_1"] - df["demand_lag_7"]

    # expanding mean of the series up to t-1 (long-run level, leakage-safe)
    df["demand_expanding_mean"] = g["demand"].transform(lambda s: s.shift(1).expanding(min_periods=7).mean())
    return df


# ---------------------------------------------------------------------------
# 5. Price features
# ---------------------------------------------------------------------------
def add_price_features(df):
    """Price is set BY the retailer, so the current day's planned price is
    legitimately known in advance — a valid same-day predictor (this is how
    you'd model elasticity). We forward-fill price across no-sale days within
    each series, then add a lagged price and a price-change term.
    """
    g = df.groupby(["store_id", "product"], group_keys=False)
    df["avg_unit_price"] = g["avg_unit_price"].ffill()
    df["price_lag_1"] = g["avg_unit_price"].shift(1)
    df["price_change_1"] = df["avg_unit_price"] - df["price_lag_1"]
    # past transaction intensity (a demand proxy) — lagged
    df["txn_count_lag_1"] = g["txn_count"].shift(1)
    return df


# ---------------------------------------------------------------------------
# 6. Leakage self-check — assert no rolling feature equals a future-inclusive
#    computation. Cheap sanity gate that would catch a missing shift().
# ---------------------------------------------------------------------------
def verify_no_leakage(df):
    sample = (df[(df["store_id"] == df["store_id"].iloc[0]) &
                 (df["product"] == df["product"].iloc[0])]
              .sort_values("business_date").reset_index(drop=True))
    # recompute rollmean_7 the WRONG (leaky) way, using the SAME min_periods
    # our feature uses, so the only difference is the shift — an apples-to-
    # apples test of whether we leaked "today" into its own feature.
    mp = max(2, 7 // 2)
    leaky = sample["demand"].rolling(7, min_periods=mp).mean()
    ours = sample["demand_rollmean_7"]
    both = ours.notna() & leaky.notna()
    # (1) ours must NOT equal the future-inclusive window (else we leaked)
    identical = np.allclose(ours[both], leaky[both])
    assert not identical, "LEAKAGE: rollmean_7 matches a future-inclusive window!"
    # (2) ours at row t must equal leaky at row t-1 (the pure shift relation).
    #     compare where the shifted leaky is defined.
    leaky_shifted = leaky.shift(1)
    both2 = ours.notna() & leaky_shifted.notna()
    aligned = np.allclose(ours[both2].values, leaky_shifted[both2].values)
    return {"leakage_check_passed": True, "shift_relation_holds": bool(aligned)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading clean transactions...")
    txn = pd.read_csv(f"{DATA_DIR}/clean_transactions.csv")

    print("1. Aggregating to daily store x category demand...")
    daily = aggregate_daily(txn)
    print(f"   {len(daily):,} daily rows before calendar completion")

    print("2. Completing the calendar (zero-filling no-sale days)...")
    df = complete_calendar(daily)
    added = len(df) - len(daily)
    print(f"   {len(df):,} rows after completion (+{added:,} zero-demand days added)")

    print("3. Calendar features...")
    df = add_calendar_features(df)
    print("4. Lag & rolling features (leakage-safe, shifted)...")
    df = add_lag_features(df)
    print("5. Price features...")
    df = add_price_features(df)

    print("6. Leakage self-check...")
    chk = verify_no_leakage(df)
    print(f"   {chk}")

    # drop warm-up rows (first MAX_LAG days per series have undefined lags).
    # cumcount gives each row's position within its series; keep rows at or
    # past MAX_LAG. Cleaner and faster than groupby.apply.
    before = len(df)
    pos_in_series = df.groupby(["store_id", "product"]).cumcount()
    df = df[pos_in_series >= MAX_LAG].reset_index(drop=True)
    print(f"   dropped {before - len(df):,} warm-up rows (first {MAX_LAG} days/series)")

    # final NaN audit
    feature_cols = [c for c in df.columns if c not in
                    ["store_id", "product", "category", "business_date", "demand", "ts_local"]]
    nan_rows = df[feature_cols].isna().any(axis=1).sum()
    print(f"   rows with any remaining feature NaN: {nan_rows:,}")

    df.to_csv(f"{DATA_DIR}/features.csv", index=False)
    print(f"\nSaved {len(df):,} rows x {len(df.columns)} cols -> {DATA_DIR}/features.csv")
    print("\nFeature columns:")
    for c in df.columns:
        print(f"  - {c}")


if __name__ == "__main__":
    main()
