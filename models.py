"""
models.py
=========
Demand forecasting models + an honest temporal evaluation.

Four forecasters, deliberately ordered from dumb to sophisticated so each has
to justify its added complexity against the one before it:

  1. Seasonal-naive baseline  — demand = same weekday last week (lag 7).
     The number every other model must beat. If a fancy model can't beat
     "what happened last Tuesday," it isn't earning its keep.
  2. LightGBM (global)         — one gradient-boosted model across ALL series,
     using the engineered lag/rolling/calendar features. Learns cross-series
     patterns a per-series model can't.
  3. Prophet (per series)      — classic additive decomposition
     (trend + weekly + yearly + holidays), fit once per (store, product).
  4. Ensemble                  — weighted blend of the above, weights tuned on
     a validation tail.

TWO THINGS THIS FILE TAKES SERIOUSLY (the interview-relevant parts):

  (A) LEAKAGE-SAFE FEATURE SELECTION. The feature file contains some columns
      that are REALIZED on the target day — revenue, txn_count, refund_units,
      outlier_txns are all computed from the same transactions that produce
      `demand`. Using them to predict `demand` is leakage: they aren't known
      until the day is over. Only their LAGGED versions are legal. We curate
      the model feature list explicitly and exclude the realized columns.

  (B) HONEST EVALUATION. We use a strict temporal holdout (train on the past,
      test on the future — never the reverse) and report metrics that survive
      this data's many zero-demand days. MAPE is intentionally NOT the headline
      metric: it divides by actuals, and ~7% of days have zero demand, making
      MAPE undefined/explosive. WAPE (weighted absolute % error) is the retail
      standard for intermittent demand and is what we lead with.
"""

import warnings
import logging
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

DATA_DIR = "data"
REPORT_DIR = "reports"

# Identifiers / target / REALIZED-on-target-day columns — never model inputs.
LEAKY_OR_ID = {
    "store_id", "product", "category", "business_date", "demand",  # id + target
    "revenue", "txn_count", "refund_units", "outlier_txns",        # realized same-day
    "ts_local",
}
# Categorical features LightGBM will handle natively.
CATEGORICALS = ["store_id", "product", "category"]


# ===========================================================================
# Metrics — chosen to survive zero-demand days
# ===========================================================================
def mae(y, yhat):
    return np.mean(np.abs(y - yhat))

def rmse(y, yhat):
    return np.sqrt(np.mean((y - yhat) ** 2))

def wape(y, yhat):
    """Weighted Absolute Percentage Error = sum|err| / sum|actual|.
    Robust to zero actuals (no per-row division), unlike MAPE. This is the
    headline metric for intermittent retail demand."""
    denom = np.sum(np.abs(y))
    return np.sum(np.abs(y - yhat)) / denom if denom > 0 else np.nan

def bias(y, yhat):
    """Mean signed error. Positive => over-forecasting (overstock risk),
    negative => under-forecasting (stockout risk). Direction matters in retail."""
    return np.mean(yhat - y)

def evaluate(y, yhat):
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return {"WAPE": wape(y, yhat), "MAE": mae(y, yhat),
            "RMSE": rmse(y, yhat), "Bias": bias(y, yhat)}


# ===========================================================================
# Feature plumbing
# ===========================================================================
def get_feature_columns(df):
    feats = [c for c in df.columns if c not in LEAKY_OR_ID]
    # put categoricals back in as model inputs (they're ids but informative)
    return CATEGORICALS + feats


def prep_for_lgbm(df, feature_cols):
    X = df[feature_cols].copy()
    for c in CATEGORICALS:
        X[c] = X[c].astype("category")
    return X


# ===========================================================================
# Temporal split — train strictly before the cutoff, test strictly after
# ===========================================================================
def temporal_split(df, test_days=45, val_days=30):
    """Three contiguous time blocks, no shuffling:
        train  : everything up to (last - test_days - val_days)
        val    : the val_days before the test block (for ensemble weight tuning)
        test   : the final test_days (the honest holdout we report on)
    """
    dmax = df["business_date"].max()
    test_start = dmax - pd.Timedelta(days=test_days - 1)
    val_start = test_start - pd.Timedelta(days=val_days)
    train = df[df["business_date"] < val_start]
    val = df[(df["business_date"] >= val_start) & (df["business_date"] < test_start)]
    test = df[df["business_date"] >= test_start]
    return train, val, test


# ===========================================================================
# Model 1 — Seasonal-naive baseline
# ===========================================================================
def predict_baseline(frame):
    """demand = same weekday last week. We already have demand_lag_7 as a
    leakage-safe column, so the baseline is literally that (clipped at 0)."""
    return frame["demand_lag_7"].clip(lower=0).values


# ===========================================================================
# Model 2 — Global LightGBM
# ===========================================================================
def train_lightgbm(train, val, feature_cols):
    Xtr = prep_for_lgbm(train, feature_cols)
    ytr = train["demand"].values
    Xva = prep_for_lgbm(val, feature_cols)
    yva = val["demand"].values

    model = lgb.LGBMRegressor(
        objective="regression_l1",   # L1 => optimize MAE, robust to outliers
        n_estimators=600,
        learning_rate=0.05,
        num_leaves=63,
        min_child_samples=40,
        subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbose=-1,
    )
    model.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="l1",
              callbacks=[lgb.early_stopping(40, verbose=False)])
    return model


# ===========================================================================
# Model 3 — Prophet, fit per (store, product) series
# ===========================================================================
def predict_prophet(train, test, holidays_df):
    from prophet import Prophet
    preds = np.full(len(test), np.nan)
    test = test.reset_index().rename(columns={"index": "_orig_idx"})
    test_pos = {k: g.index for k, g in test.groupby(["store_id", "product"])}

    out_idx, out_val = [], []
    for key, tr in train.groupby(["store_id", "product"]):
        if key not in test_pos:
            continue
        hist = tr[["business_date", "demand"]].rename(
            columns={"business_date": "ds", "demand": "y"})
        if hist["y"].notna().sum() < 30:
            continue
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True,
                    daily_seasonality=False, holidays=holidays_df)
        m.fit(hist)
        te = test.loc[test_pos[key]]
        future = te[["business_date"]].rename(columns={"business_date": "ds"})
        fc = m.predict(future)["yhat"].clip(lower=0).values
        out_idx.extend(te.index.tolist())
        out_val.extend(fc.tolist())

    preds[out_idx] = out_val
    # any series Prophet skipped: fall back to baseline
    fallback = np.isnan(preds)
    preds[fallback] = test.loc[fallback, "demand_lag_7"].clip(lower=0).values
    return preds


def build_holiday_frame(df):
    import holidays as hol
    yrs = range(df["business_date"].dt.year.min(), df["business_date"].dt.year.max() + 1)
    us = hol.US(years=list(yrs))
    return pd.DataFrame({"holiday": "us_holiday",
                         "ds": pd.to_datetime(sorted(us.keys()))})


# ===========================================================================
# Model 4 — Ensemble (weights tuned on validation, applied to test)
# ===========================================================================
def tune_ensemble_weights(val_preds, y_val):
    """Grid-search convex weights over the base models on the validation tail,
    minimizing WAPE. Returns the best weight vector."""
    names = list(val_preds.keys())
    best_w, best_score = None, np.inf
    grid = np.linspace(0, 1, 11)
    # 2- or 3-model convex combos
    for wa in grid:
        for wb in grid:
            wc = 1 - wa - wb
            if wc < -1e-9:
                continue
            w = {names[0]: wa, names[1]: wb}
            if len(names) > 2:
                w[names[2]] = wc
            elif abs(wc) > 1e-9:
                continue
            blend = sum(w[n] * val_preds[n] for n in w)
            s = wape(y_val, blend)
            if s < best_score:
                best_score, best_w = s, w
    return best_w


# ===========================================================================
# Orchestration
# ===========================================================================
def main():
    print("Loading features...")
    df = pd.read_csv(f"{DATA_DIR}/features.csv", parse_dates=["business_date"])
    df = df.sort_values(["store_id", "product", "business_date"]).reset_index(drop=True)

    feature_cols = get_feature_columns(df)
    print(f"  {len(df):,} rows | {len(feature_cols)} model features")
    print(f"  excluded as leakage/id: {sorted(LEAKY_OR_ID)}")

    train, val, test = temporal_split(df, test_days=45, val_days=30)
    print(f"\nTemporal split (no shuffle):")
    print(f"  train: {train['business_date'].min().date()} -> {train['business_date'].max().date()}  ({len(train):,})")
    print(f"  val  : {val['business_date'].min().date()} -> {val['business_date'].max().date()}  ({len(val):,})")
    print(f"  test : {test['business_date'].min().date()} -> {test['business_date'].max().date()}  ({len(test):,})")

    # --- Model 1: baseline ---
    print("\n[1/4] Seasonal-naive baseline...")
    base_val = predict_baseline(val)
    base_test = predict_baseline(test)

    # --- Model 2: LightGBM ---
    print("[2/4] Training global LightGBM...")
    lgbm = train_lightgbm(train, val, feature_cols)
    lgbm_val = lgbm.predict(prep_for_lgbm(val, feature_cols)).clip(min=0)
    lgbm_test = lgbm.predict(prep_for_lgbm(test, feature_cols)).clip(min=0)

    # --- Model 3: Prophet ---
    print("[3/4] Fitting Prophet per series (225 series)...")
    hol_df = build_holiday_frame(df)
    # Prophet trains on train+val history to forecast the test block
    trainval = pd.concat([train, val])
    prophet_test = predict_prophet(trainval, test, hol_df)
    # also need prophet on val for weight tuning -> train on train, predict val
    prophet_val = predict_prophet(train, val, hol_df)

    # --- Model 4: Ensemble ---
    print("[4/4] Tuning ensemble weights on validation...")
    val_preds = {"baseline": base_val, "lightgbm": lgbm_val, "prophet": prophet_val}
    weights = tune_ensemble_weights(val_preds, val["demand"].values)
    ens_test = sum(weights[n] * p for n, p in
                   {"baseline": base_test, "lightgbm": lgbm_test, "prophet": prophet_test}.items())

    # --- Evaluate on the held-out test block ---
    y_test = test["demand"].values
    results = {
        "Seasonal-naive": evaluate(y_test, base_test),
        "LightGBM":       evaluate(y_test, lgbm_test),
        "Prophet":        evaluate(y_test, prophet_test),
        "Ensemble":       evaluate(y_test, ens_test),
    }

    # per-category WAPE for LightGBM (segment view)
    test_eval = test.copy()
    test_eval["pred_lgbm"] = lgbm_test
    seg = (test_eval.groupby("category")
           .apply(lambda g: wape(g["demand"].values, g["pred_lgbm"].values))
           .to_dict())

    # feature importance
    imp = (pd.Series(lgbm.feature_importances_, index=feature_cols)
           .sort_values(ascending=False).head(12))

    # ---- report ----
    lines = []
    lines.append("=" * 68)
    lines.append("FORECASTING MODEL RESULTS — held-out temporal test block")
    lines.append("=" * 68)
    lines.append(f"Test window: {test['business_date'].min().date()} -> "
                 f"{test['business_date'].max().date()}  ({len(test):,} rows)")
    lines.append("\nLower WAPE/MAE/RMSE = better. Bias>0 = over-forecast.")
    lines.append(f"\n{'Model':<16}{'WAPE':>9}{'MAE':>9}{'RMSE':>9}{'Bias':>9}")
    lines.append("-" * 52)
    for name, m in results.items():
        lines.append(f"{name:<16}{m['WAPE']:>9.4f}{m['MAE']:>9.2f}"
                     f"{m['RMSE']:>9.2f}{m['Bias']:>9.2f}")

    base_wape = results["Seasonal-naive"]["WAPE"]
    best_name = min(results, key=lambda k: results[k]["WAPE"])
    best_wape = results[best_name]["WAPE"]
    lift = 100 * (base_wape - best_wape) / base_wape
    lines.append(f"\nBest model: {best_name} (WAPE {best_wape:.4f}), "
                 f"{lift:.1f}% better than the baseline.")

    lines.append(f"\nEnsemble weights (tuned on validation): "
                 f"{ {k: round(v,2) for k,v in weights.items()} }")

    lines.append("\nLightGBM WAPE by category:")
    for c, w in seg.items():
        lines.append(f"  {c:<14}{w:.4f}")

    lines.append("\nTop LightGBM features (by importance):")
    for f, v in imp.items():
        lines.append(f"  {f:<26}{int(v)}")

    lines.append("\nMETRIC NOTE: WAPE (not MAPE) is the headline metric because")
    lines.append("~7% of store/product/days have zero demand, which makes MAPE")
    lines.append("undefined (division by zero). WAPE pools error over total")
    lines.append("volume and is the retail-standard for intermittent demand.")

    report = "\n".join(lines)
    import os
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(f"{REPORT_DIR}/model_results.txt", "w") as f:
        f.write(report)
    print("\n" + report)


if __name__ == "__main__":
    main()
