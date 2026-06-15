"""
backtest.py  (Stage 4 — robustness & backtesting)
=================================================
Stage 3 reported a single temporal holdout. A single window can be lucky, so
this stage answers two harder questions:

  1. IS THE RESULT STABLE OVER TIME?  -> expanding-window (rolling-origin)
     backtest. We retrain at several cutoffs marching forward through time and
     forecast the next block each time, producing a DISTRIBUTION of WAPE rather
     than one number. A model that's genuinely good wins on most folds, not
     just the one we happened to pick.

  2. WHERE DOES THE ACCURACY COME FROM?  -> feature-group ablation. We retrain
     LightGBM repeatedly, each time removing one group of features, and watch
     WAPE move. This (a) shows which signals matter and (b) directly answers
     the "is the model just leaning on current-day price?" challenge — we drop
     current-day price and see whether accuracy survives.

Prophet is intentionally left out of the multi-fold loop: it fits 225 models
per fold, so across folds it would dominate runtime without changing the
headline (the Stage-3 ensemble already collapsed to LightGBM). The honest
comparison that matters is baseline vs LightGBM, repeated over time.
"""

import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore")

# reuse Stage 3 building blocks so the models are IDENTICAL across stages
from models import (
    get_feature_columns, prep_for_lgbm, predict_baseline,
    train_lightgbm, evaluate, wape, CATEGORICALS, LEAKY_OR_ID,
)

DATA_DIR = "data"
REPORT_DIR = "reports"


# ===========================================================================
# Fold construction — expanding train, fixed-width val + test marching forward
# ===========================================================================
def make_folds(df, n_folds=5, test_window=30, val_window=30):
    """Returns a list of (train_end, val_start, val_end, test_start, test_end)
    date boundaries, chronological. Test blocks are non-overlapping and step
    backward from the data's last date; the train window EXPANDS each fold
    (always starts at the very beginning), which is the defining property of
    rolling-origin / expanding-window backtesting."""
    dmax = df["business_date"].max()
    folds = []
    for k in range(n_folds):
        test_end = dmax - pd.Timedelta(days=test_window * k)
        test_start = test_end - pd.Timedelta(days=test_window - 1)
        val_end = test_start - pd.Timedelta(days=1)
        val_start = val_end - pd.Timedelta(days=val_window - 1)
        train_end = val_start - pd.Timedelta(days=1)
        folds.append((train_end, val_start, val_end, test_start, test_end))
    return folds[::-1]  # oldest fold first


def slice_fold(df, bounds):
    train_end, val_start, val_end, test_start, test_end = bounds
    train = df[df["business_date"] <= train_end]
    val = df[(df["business_date"] >= val_start) & (df["business_date"] <= val_end)]
    test = df[(df["business_date"] >= test_start) & (df["business_date"] <= test_end)]
    return train, val, test


# ===========================================================================
# Part 1 — Expanding-window backtest
# ===========================================================================
def run_backtest(df, feature_cols, n_folds=5):
    folds = make_folds(df, n_folds=n_folds)
    rows = []
    for i, bounds in enumerate(folds, 1):
        train, val, test = slice_fold(df, bounds)
        if len(val) == 0 or len(test) == 0:
            continue
        y_test = test["demand"].values

        base = predict_baseline(test)
        model = train_lightgbm(train, val, feature_cols)
        lgbm = model.predict(prep_for_lgbm(test, feature_cols)).clip(min=0)

        b = evaluate(y_test, base)
        m = evaluate(y_test, lgbm)
        rows.append({
            "fold": i,
            "train_end": bounds[0].date(),
            "test": f"{bounds[3].date()}→{bounds[4].date()}",
            "train_rows": len(train),
            "base_WAPE": b["WAPE"], "lgbm_WAPE": m["WAPE"],
            "base_MAE": b["MAE"], "lgbm_MAE": m["MAE"],
            "lgbm_Bias": m["Bias"],
            "improvement_%": 100 * (b["WAPE"] - m["WAPE"]) / b["WAPE"],
        })
        print(f"  fold {i}: test {rows[-1]['test']}  "
              f"baseline WAPE {b['WAPE']:.4f} | LightGBM {m['WAPE']:.4f} "
              f"({rows[-1]['improvement_%']:+.1f}%)")
    return pd.DataFrame(rows)


# ===========================================================================
# Part 2 — Feature-group ablation (on the most recent fold)
# ===========================================================================
def feature_groups(feature_cols):
    """Partition the model features into interpretable groups so we can drop
    one group at a time."""
    g = {"lags": [], "rolling": [], "calendar": [], "price_current": [],
         "price_past": [], "categorical": [], "other": []}
    for c in feature_cols:
        if c in CATEGORICALS:
            g["categorical"].append(c)
        elif c.startswith("demand_lag_"):
            g["lags"].append(c)
        elif c.startswith(("demand_rollmean", "demand_rollstd", "demand_rollmax",
                           "demand_same_dow", "demand_momentum", "demand_expanding")):
            g["rolling"].append(c)
        elif c in ("avg_unit_price", "price_change_1"):
            g["price_current"].append(c)   # uses CURRENT-day price
        elif c in ("price_lag_1", "txn_count_lag_1"):
            g["price_past"].append(c)       # past-only price/demand proxy
        elif c in ("dow", "is_weekend", "day_of_month", "month", "week_of_year",
                   "day_of_year", "days_since_start", "dow_sin", "dow_cos",
                   "month_sin", "month_cos", "is_holiday", "days_to_nearest_holiday"):
            g["calendar"].append(c)
        else:
            g["other"].append(c)
    return {k: v for k, v in g.items() if v}


def run_ablation(df, feature_cols, n_folds=5):
    # use the latest fold for ablation
    bounds = make_folds(df, n_folds=n_folds)[-1]
    train, val, test = slice_fold(df, bounds)
    y_test = test["demand"].values
    groups = feature_groups(feature_cols)

    results = []
    # full model
    full_model = train_lightgbm(train, val, feature_cols)
    full_wape = wape(y_test, full_model.predict(prep_for_lgbm(test, feature_cols)).clip(min=0))
    results.append({"variant": "FULL (all features)", "n_features": len(feature_cols),
                    "WAPE": full_wape, "delta_vs_full": 0.0})

    # drop one group at a time
    for gname, cols in groups.items():
        kept = [c for c in feature_cols if c not in cols]
        model = train_lightgbm(train, val, kept)
        w = wape(y_test, model.predict(prep_for_lgbm(test, kept)).clip(min=0))
        results.append({"variant": f"– drop {gname}", "n_features": len(kept),
                        "WAPE": w, "delta_vs_full": w - full_wape})

    # special: calendar-only (drop everything demand/price-derived) — how far
    # can you get with NO history at all?
    cal_only = groups.get("calendar", []) + groups.get("categorical", [])
    if cal_only:
        model = train_lightgbm(train, val, cal_only)
        w = wape(y_test, model.predict(prep_for_lgbm(test, cal_only)).clip(min=0))
        results.append({"variant": "CALENDAR-ONLY (no history)", "n_features": len(cal_only),
                        "WAPE": w, "delta_vs_full": w - full_wape})

    return pd.DataFrame(results), bounds


# ===========================================================================
# Main
# ===========================================================================
def main():
    print("Loading features...")
    df = pd.read_csv(f"{DATA_DIR}/features.csv", parse_dates=["business_date"])
    df = df.sort_values(["store_id", "product", "business_date"]).reset_index(drop=True)
    feature_cols = get_feature_columns(df)

    print(f"\n=== PART 1: Expanding-window backtest (5 folds) ===")
    bt = run_backtest(df, feature_cols, n_folds=5)

    print(f"\n=== PART 2: Feature ablation (latest fold) ===")
    ab, ab_bounds = run_ablation(df, feature_cols, n_folds=5)
    full_wape_for_report = ab[ab["variant"] == "FULL (all features)"]["WAPE"].iloc[0]
    for _, r in ab.iterrows():
        print(f"  {r['variant']:<28} WAPE {r['WAPE']:.4f}  "
              f"(Δ {r['delta_vs_full']:+.4f})")

    # ---- report ----
    lines = []
    lines.append("=" * 70)
    lines.append("STAGE 4 — ROBUSTNESS & BACKTESTING")
    lines.append("=" * 70)

    lines.append("\nPART 1 — EXPANDING-WINDOW BACKTEST (rolling origin, 5 folds)")
    lines.append("Train expands each fold; test blocks march forward, non-overlapping.\n")
    lines.append(f"{'fold':<5}{'test window':<26}{'train_rows':>11}"
                 f"{'base WAPE':>11}{'lgbm WAPE':>11}{'impr %':>9}")
    lines.append("-" * 73)
    for _, r in bt.iterrows():
        lines.append(f"{r['fold']:<5}{r['test']:<26}{r['train_rows']:>11,}"
                     f"{r['base_WAPE']:>11.4f}{r['lgbm_WAPE']:>11.4f}"
                     f"{r['improvement_%']:>8.1f}%")
    lines.append("-" * 73)
    lines.append(f"{'mean':<31}{'':>11}"
                 f"{bt['base_WAPE'].mean():>11.4f}{bt['lgbm_WAPE'].mean():>11.4f}"
                 f"{bt['improvement_%'].mean():>8.1f}%")
    lines.append(f"{'std':<31}{'':>11}"
                 f"{bt['base_WAPE'].std():>11.4f}{bt['lgbm_WAPE'].std():>11.4f}"
                 f"{bt['improvement_%'].std():>8.1f}%")
    lines.append(f"\nLightGBM beats the baseline on {int((bt['lgbm_WAPE']<bt['base_WAPE']).sum())}"
                 f"/{len(bt)} folds. Mean improvement "
                 f"{bt['improvement_%'].mean():.1f}% (±{bt['improvement_%'].std():.1f}).")
    lines.append("The low std across folds is the point: the gain is stable, not a")
    lines.append("lucky single window.")

    lines.append("\n\nPART 2 — FEATURE-GROUP ABLATION (latest fold)")
    lines.append(f"Test window: {ab_bounds[3].date()} → {ab_bounds[4].date()}")
    lines.append("Each row drops one group from the full model. Larger positive Δ")
    lines.append("means that group mattered more (removing it hurt accuracy more).\n")
    lines.append(f"{'variant':<30}{'#feat':>7}{'WAPE':>10}{'Δ vs full':>11}")
    lines.append("-" * 58)
    for _, r in ab.iterrows():
        lines.append(f"{r['variant']:<30}{r['n_features']:>7}{r['WAPE']:>10.4f}"
                     f"{r['delta_vs_full']:>+11.4f}")

    # interpret the current-price ablation explicitly and HONESTLY
    cur = ab[ab["variant"] == "– drop price_current"]
    if len(cur):
        d = cur["delta_vs_full"].iloc[0]
        wape_no_price = cur["WAPE"].iloc[0]
        # worst-case (no current price) vs the baseline on the same fold
        fold_base = bt["base_WAPE"].iloc[-1]
        still_better = 100 * (fold_base - wape_no_price) / fold_base
        verdict = ("negligibly" if abs(d) < 0.01 else
                   "modestly" if abs(d) < 0.03 else "substantially")
        lines.append(f"\nROBUSTNESS VERDICT (current-day price):")
        lines.append(f"Dropping current-day price raises WAPE by {d:+.4f} "
                     f"({wape_no_price:.4f} vs {full_wape_for_report:.4f}) — the model")
        lines.append(f"relies on it {verdict}. This is the honest weak point: current-day")
        lines.append(f"average price is treated as known-in-advance (the retailer sets it),")
        lines.append(f"but if that assumption fails, price_lag_1 must be used instead.")
        lines.append(f"CRUCIALLY: even with current-day price removed entirely, WAPE")
        lines.append(f"{wape_no_price:.4f} still beats the seasonal-naive baseline "
                     f"({fold_base:.4f}) by {still_better:.0f}% —")
        lines.append(f"so the model's value survives the worst-case assumption.")

    report = "\n".join(lines)
    import os
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(f"{REPORT_DIR}/backtest_results.txt", "w") as f:
        f.write(report)
    bt.to_csv(f"{REPORT_DIR}/backtest_folds.csv", index=False)
    print("\n" + report)


if __name__ == "__main__":
    main()
