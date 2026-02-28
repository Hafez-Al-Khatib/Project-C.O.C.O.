"""
V2 vs V3 Head-to-Head Comparison
=================================
Runs BOTH approaches through identical walk-forward windows
and logs everything to a single MLFlow experiment for direct comparison.

V2 (Duo's approach): Predict log(MonthlySale) directly
V3 (Ratio approach): Predict growth_multiplier, then multiply back

Same models, same data, same windows. Only the TARGET changes.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.base import clone

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BRANCH_FLAGS = [
    "BranchisConut", "BranchisConutTyre",
    "BranchisConutJnah", "BranchisMainStreetCoffee"
]
BRANCH_DISPLAY = {
    "BranchisConut": "Conut Main",
    "BranchisConutTyre": "Conut Tyre",
    "BranchisConutJnah": "Conut Jnah",
    "BranchisMainStreetCoffee": "Main St Coffee",
}
HOLIDAY_INTENSITY = {8: 0.0, 9: 0.0, 10: 0.0, 11: 0.2, 12: 1.0}


# ─────────────────────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING (shared by both)
# ─────────────────────────────────────────────────────────────

def load_data():
    csv_path = os.path.join(BASE_DIR, "Branches_Cleaned_For_XGBoost.csv")
    df = pd.read_csv(csv_path)
    df["PrevsMonthly"] = pd.to_numeric(df["PrevsMonthly"], errors="coerce").fillna(0)

    df["branch_encoded"] = df[BRANCH_FLAGS].idxmax(axis=1).map({
        "BranchisConut": 0, "BranchisConutTyre": 1,
        "BranchisConutJnah": 2, "BranchisMainStreetCoffee": 3,
    })
    df["branch_name"] = df[BRANCH_FLAGS].idxmax(axis=1).map(BRANCH_DISPLAY)

    # V2 contextual injection features
    df["Holiday_Intensity"] = df["Month_Num"].map(HOLIDAY_INTENSITY).fillna(0.0)
    df["Regional_Demand_Proxy"] = df["Is_Coastal"] * df["Holiday_Intensity"]

    df["avg_daily_rate"] = df["PrevsMonthly"] / 30
    df["estimated_ops_days"] = np.where(
        df["avg_daily_rate"] > 0,
        np.minimum(df["DaysinMonth"], df["MonthlySale"] / df["avg_daily_rate"]),
        df["DaysinMonth"],
    )
    df["Operational_Days_Ratio"] = (df["estimated_ops_days"] / df["DaysinMonth"]).clip(0, 1)

    df["Branch_Status"] = 0
    conut_dec = df[(df["BranchisConut"] == 1) & (df["Month_Num"] == 12)]
    if len(conut_dec) > 0:
        conut_nov = df[(df["BranchisConut"] == 1) & (df["Month_Num"] == 11)]["MonthlySale"]
        if len(conut_nov) > 0 and conut_dec["MonthlySale"].values[0] < conut_nov.values[0] * 0.2:
            df.loc[(df["BranchisConut"] == 1) & (df["Month_Num"] == 12), "Branch_Status"] = 1

    df["WeekendRatio"] = df["WeekendDays"] / df["DaysinMonth"]
    df["is_weekend_heavy"] = (df["WeekendDays"] >= 10).astype(int)
    df["SalePerDay"] = df["PrevsMonthly"] / df["DaysinMonth"]
    df["log_prev"] = np.log1p(df["PrevsMonthly"])
    df["log_sale"] = np.log1p(df["MonthlySale"])

    # Rolling 3m average
    df["rolling_3m_avg"] = (
        df.groupby("branch_encoded")["MonthlySale"]
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df["rolling_3m_avg"] = df.groupby("branch_encoded")["rolling_3m_avg"].transform(
        lambda x: x.fillna(x.mean())
    )

    # Relative branch size
    month_avg = df.groupby("Month_Num")["PrevsMonthly"].transform("mean")
    df["relative_size"] = df["PrevsMonthly"] / month_avg.replace(0, 1)

    # Growth multiplier (V3 target)
    df["growth_multiplier"] = np.where(
        df["PrevsMonthly"] > 0,
        df["MonthlySale"] / df["PrevsMonthly"],
        np.nan
    )

    return df


# Features shared by both approaches
SHARED_FEATURES = [
    "branch_encoded", "Month_Num",
    "Holiday_Intensity", "Operational_Days_Ratio",
    "Regional_Demand_Proxy", "Branch_Status",
    "DaysinMonth", "WeekendDays", "HolidayMonth",
    "WeekendRatio", "is_weekend_heavy",
    "Months Active", "Coldstart", "Is_Coastal",
    "log_prev", "relative_size",
]

# V2 adds lag features that use raw scale
V2_EXTRA_FEATURES = ["PrevsMonthly", "SalePerDay", "rolling_3m_avg"]
V2_FEATURES = SHARED_FEATURES + V2_EXTRA_FEATURES
V3_FEATURES = SHARED_FEATURES


# ─────────────────────────────────────────────────────────────
# MODELS (same for both approaches)
# ─────────────────────────────────────────────────────────────

def get_models():
    models = {}

    kernel_m = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    models["GPR"] = GaussianProcessRegressor(
        kernel=kernel_m, n_restarts_optimizer=10,
        normalize_y=True, random_state=42
    )

    models["BayesianRidge"] = BayesianRidge(
        alpha_1=1e-6, alpha_2=1e-6,
        lambda_1=1e-6, lambda_2=1e-6,
        compute_score=True, max_iter=500
    )

    models["QR_Q90"] = GradientBoostingRegressor(
        loss="quantile", alpha=0.9,
        n_estimators=200, max_depth=2,
        learning_rate=0.05, subsample=0.8, random_state=42
    )

    return models


WINDOWS = [
    {"train": [8, 9],      "test": 10, "label": "W1_Oct"},
    {"train": [8, 9, 10],  "test": 11, "label": "W2_Nov"},
    {"train": [8, 9, 10, 11], "test": 12, "label": "W3_Dec"},
]


# ─────────────────────────────────────────────────────────────
# RUN A SINGLE APPROACH (V2 or V3)
# ─────────────────────────────────────────────────────────────

def run_approach(df, approach_name, features, target_col, back_convert_fn, results_out):
    """
    Run walk-forward CV for one approach and log to MLFlow.

    back_convert_fn(y_pred_raw, test_df) -> predicted_sales in original scale
    """
    models = get_models()
    scaler = StandardScaler()

    for window in WINDOWS:
        train_mask = df["Month_Num"].isin(window["train"])
        test_mask = df["Month_Num"] == window["test"]

        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()

        if len(train_df) == 0 or len(test_df) == 0:
            continue

        # Filter out rows with NaN target for V3
        valid_train = train_df[train_df[target_col].notna() & np.isfinite(train_df[target_col])]
        valid_test = test_df[test_df[target_col].notna() & np.isfinite(test_df[target_col])]

        if len(valid_train) == 0 or len(valid_test) == 0:
            continue

        X_train = valid_train[features].values
        X_test = valid_test[features].values
        y_train = valid_train[target_col].values
        y_test_actual = valid_test["MonthlySale"].values

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)
        branches = valid_test["branch_name"].values

        for model_name, model_template in models.items():
            model = clone(model_template)
            run_name = f"{approach_name}_{model_name}_{window['label']}"

            with mlflow.start_run(run_name=run_name):
                model.fit(X_train_s, y_train)
                y_pred_raw = model.predict(X_test_s)

                # Back-convert to sales
                pred_sales = back_convert_fn(y_pred_raw, valid_test)

                # Metrics
                valid_m = (y_test_actual > 0) & (pred_sales > 0) & np.isfinite(pred_sales)
                if valid_m.sum() > 0:
                    rmse = float(np.sqrt(mean_squared_error(y_test_actual[valid_m], pred_sales[valid_m])))
                    mape = float(mean_absolute_percentage_error(y_test_actual[valid_m], pred_sales[valid_m]) * 100)
                    r2 = float(r2_score(y_test_actual[valid_m], pred_sales[valid_m]))
                else:
                    rmse, mape, r2 = 1e12, 999, -999

                mlflow.log_param("approach", approach_name)
                mlflow.log_param("model", model_name)
                mlflow.log_param("window", window["label"])
                mlflow.log_param("test_month", window["test"])
                mlflow.log_param("target_type", target_col)
                mlflow.log_param("n_features", len(features))
                mlflow.log_metric("rmse", round(rmse, 2))
                mlflow.log_metric("mape", round(mape, 2))
                mlflow.log_metric("r2", round(r2, 4))

                # Per-branch errors
                for i, br in enumerate(branches):
                    if y_test_actual[i] > 0:
                        br_err = abs(pred_sales[i] - y_test_actual[i]) / y_test_actual[i] * 100
                        tag_name = br.replace(" ", "_")
                        mlflow.log_metric(f"err_{tag_name}", round(br_err, 1))

                result = {
                    "approach": approach_name,
                    "model": model_name,
                    "window": window["label"],
                    "test_month": window["test"],
                    "rmse": rmse,
                    "mape": mape,
                    "r2": r2,
                    "branches": branches,
                    "actual": y_test_actual,
                    "preds": pred_sales,
                }
                results_out.append(result)

                status = "PASS" if mape < 60 else "WARN" if mape < 120 else "FAIL"
                print(f"  [{status}] {run_name:40s} | MAPE:{mape:7.1f}% | RMSE:{rmse/1e9:.3f}B | R2:{r2:+.3f}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  V2 vs V3 HEAD-TO-HEAD COMPARISON")
    print("  Same models, same windows, only the TARGET changes")
    print("=" * 70)

    df = load_data()

    # MLFlow setup
    mlruns_dir = os.path.join(BASE_DIR, "mlruns")
    os.makedirs(mlruns_dir, exist_ok=True)
    mlflow.set_tracking_uri("file:///" + mlruns_dir.replace(os.sep, "/"))
    mlflow.set_experiment("V2_vs_V3_HeadToHead")

    all_results = []

    # --- V2: Predict log(sales) directly ---
    print("\n" + "-" * 70)
    print("  APPROACH V2: Predict log(MonthlySale) -> expm1 back")
    print("-" * 70)

    def v2_back_convert(y_pred_log, test_df):
        return np.maximum(np.expm1(y_pred_log), 0)

    run_approach(df, "V2_LogSale", V2_FEATURES, "log_sale", v2_back_convert, all_results)

    # --- V3: Predict growth multiplier ---
    print("\n" + "-" * 70)
    print("  APPROACH V3: Predict growth_multiplier -> multiply by PrevMonth")
    print("-" * 70)

    def v3_back_convert(y_pred_ratio, test_df):
        prev = test_df["PrevsMonthly"].values
        return np.maximum(y_pred_ratio * prev, 0)

    run_approach(df, "V3_Ratio", V3_FEATURES, "growth_multiplier", v3_back_convert, all_results)

    # ─────────────────────────────────────────────────────────
    # COMPARISON TABLE
    # ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  COMPARISON TABLE")
    print("=" * 70)

    # Build pivot: approach x model x window -> mape
    header = f"  {'Approach':12s} {'Model':16s} | {'W1(Oct)':>10s} {'W2(Nov)':>10s} {'W3(Dec)':>10s} | {'AVG':>8s}"
    print(header)
    print("  " + "-" * 75)

    comparison_lines = []
    for approach in ["V2_LogSale", "V3_Ratio"]:
        for model_name in ["GPR", "BayesianRidge", "QR_Q90"]:
            mapes = {}
            for r in all_results:
                if r["approach"] == approach and r["model"] == model_name:
                    mapes[r["window"]] = r["mape"]

            w1 = mapes.get("W1_Oct", float("nan"))
            w2 = mapes.get("W2_Nov", float("nan"))
            w3 = mapes.get("W3_Dec", float("nan"))
            avg = np.nanmean([w1, w2, w3])

            line = f"  {approach:12s} {model_name:16s} | {w1:8.1f}% {w2:9.1f}% {w3:9.1f}% | {avg:6.1f}%"
            print(line)
            comparison_lines.append(line)

    # December detailed comparison
    print("\n" + "=" * 70)
    print("  DECEMBER PER-BRANCH COMPARISON (Window 3)")
    print("=" * 70)

    dec_results_v2 = [r for r in all_results if r["approach"] == "V2_LogSale" and r["test_month"] == 12]
    dec_results_v3 = [r for r in all_results if r["approach"] == "V3_Ratio" and r["test_month"] == 12]

    # Pick best model from each approach for Dec
    if dec_results_v2:
        best_v2 = min(dec_results_v2, key=lambda r: r["mape"])
        print(f"\n  V2 Best (Dec): {best_v2['model']} | MAPE: {best_v2['mape']:.1f}%")
        for i, br in enumerate(best_v2["branches"]):
            actual = best_v2["actual"][i]
            pred = best_v2["preds"][i]
            err = abs(pred - actual) / actual * 100
            print(f"    {br:18s} | Actual: {actual/1e9:.3f}B | Pred: {pred/1e9:.3f}B | Err: {err:.1f}%")

    if dec_results_v3:
        best_v3 = min(dec_results_v3, key=lambda r: r["mape"])
        print(f"\n  V3 Best (Dec): {best_v3['model']} | MAPE: {best_v3['mape']:.1f}%")
        for i, br in enumerate(best_v3["branches"]):
            actual = best_v3["actual"][i]
            pred = best_v3["preds"][i]
            err = abs(pred - actual) / actual * 100
            print(f"    {br:18s} | Actual: {actual/1e9:.3f}B | Pred: {pred/1e9:.3f}B | Err: {err:.1f}%")

    if dec_results_v2 and dec_results_v3:
        improvement = best_v2["mape"] - best_v3["mape"]
        pct_improvement = improvement / best_v2["mape"] * 100
        print(f"\n  IMPROVEMENT: V3 reduced Dec MAPE by {improvement:.1f}pp ({pct_improvement:.0f}% relative improvement)")

    # Save comparison to file
    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
    with open(os.path.join(BASE_DIR, "outputs", "v2_vs_v3_comparison.csv"), "w") as f:
        f.write("approach,model,window,test_month,mape,rmse,r2\n")
        for r in all_results:
            f.write(f"{r['approach']},{r['model']},{r['window']},{r['test_month']},{r['mape']:.2f},{r['rmse']:.2f},{r['r2']:.4f}\n")

    print(f"\n{'='*70}")
    print(f"  Results saved to outputs/v2_vs_v3_comparison.csv")
    print(f"  MLFlow experiment: 'V2_vs_V3_HeadToHead'")
    print(f"  Total runs logged: {len(all_results)}")
    print(f"{'='*70}")
