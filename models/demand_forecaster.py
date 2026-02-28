"""
V3 Demand Forecaster — Deep Analysis & Redesign
Key innovation: RATIO-BASED TARGET ENGINEERING

Instead of predicting raw sales (67M to 3074M = 46x range),
predict the GROWTH MULTIPLIER (Sale / PreviousSale) which ranges 0.05 to 6.3.
Then multiply back by the known previous month to get the forecast.

This:
1. Normalizes across branches (a 2x growth is 2x regardless of base)
2. Makes December tractable (Conut Main closure = mult ~0.05, Jnah spike = mult ~3.0)
3. Dramatically compresses the target space for the model
4. Incorporates the Duo's walk-forward CV approach
5. Full MLFlow tracking
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.base import clone

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

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


# ---------------------------------------------------------------------------
# 1. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def load_and_engineer(csv_path=None):
    """Load data and engineer features for ratio-based forecasting."""
    if csv_path is None:
        csv_path = os.path.join(BASE_DIR, "Branches_Cleaned_For_XGBoost.csv")

    df = pd.read_csv(csv_path)
    df["PrevsMonthly"] = pd.to_numeric(df["PrevsMonthly"], errors="coerce")

    # Branch metadata
    df["branch_encoded"] = df[BRANCH_FLAGS].idxmax(axis=1).map({
        "BranchisConut": 0, "BranchisConutTyre": 1,
        "BranchisConutJnah": 2, "BranchisMainStreetCoffee": 3,
    })
    df["branch_name"] = df[BRANCH_FLAGS].idxmax(axis=1).map(BRANCH_DISPLAY)

    # --- V2 Contextual Injection ---
    df["Holiday_Intensity"] = df["Month_Num"].map(HOLIDAY_INTENSITY).fillna(0.0)
    df["Regional_Demand_Proxy"] = df["Is_Coastal"] * df["Holiday_Intensity"]

    # Operational_Days_Ratio: compute from revenue drop
    df["avg_daily_rate"] = df["PrevsMonthly"] / 30
    df["estimated_ops_days"] = np.where(
        df["avg_daily_rate"] > 0,
        np.minimum(df["DaysinMonth"], df["MonthlySale"] / df["avg_daily_rate"]),
        df["DaysinMonth"],
    )
    df["Operational_Days_Ratio"] = (df["estimated_ops_days"] / df["DaysinMonth"]).clip(0, 1)

    # Branch_Status
    df["Branch_Status"] = 0
    conut_dec = df[(df["BranchisConut"] == 1) & (df["Month_Num"] == 12)]
    if len(conut_dec) > 0:
        conut_nov = df[(df["BranchisConut"] == 1) & (df["Month_Num"] == 11)]["MonthlySale"]
        if len(conut_nov) > 0 and conut_dec["MonthlySale"].values[0] < conut_nov.values[0] * 0.2:
            df.loc[(df["BranchisConut"] == 1) & (df["Month_Num"] == 12), "Branch_Status"] = 1

    # Calendar
    df["WeekendRatio"] = df["WeekendDays"] / df["DaysinMonth"]
    df["is_weekend_heavy"] = (df["WeekendDays"] >= 10).astype(int)

    # --- THE KEY INNOVATION: Growth multiplier target ---
    df["growth_multiplier"] = df["MonthlySale"] / df["PrevsMonthly"]

    # Momentum feature: how fast is the branch accelerating?
    # (requires 2 prior months, so will be NaN for early rows)
    df["prev_growth"] = df.groupby("branch_encoded")["growth_multiplier"].shift(1)

    # Log of previous monthly (scale-invariant lag feature)
    df["log_prev"] = np.log1p(df["PrevsMonthly"].fillna(0))

    # Relative size: branch's prev month vs avg of all branches that month
    month_avg = df.groupby("Month_Num")["PrevsMonthly"].transform("mean")
    df["relative_size"] = df["PrevsMonthly"] / month_avg.replace(0, 1)

    return df


FEATURES_RATIO = [
    "branch_encoded", "Month_Num",
    # V2 contextual injection
    "Holiday_Intensity", "Operational_Days_Ratio",
    "Regional_Demand_Proxy", "Branch_Status",
    # Calendar
    "DaysinMonth", "WeekendDays", "HolidayMonth",
    "WeekendRatio", "is_weekend_heavy",
    # Branch profile
    "Months Active", "Coldstart", "Is_Coastal",
    # Scale-invariant lags
    "log_prev", "relative_size",
]


# ---------------------------------------------------------------------------
# 2. MODELS
# ---------------------------------------------------------------------------

def build_models():
    """Build model registry for ratio-based forecasting."""
    models = {}

    # GPR with Matern kernel (best from V1 experiments)
    kernel_m = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    models["GPR_Matern"] = GaussianProcessRegressor(
        kernel=kernel_m, n_restarts_optimizer=10,
        normalize_y=True, random_state=42
    )

    # GPR with RBF
    kernel_r = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
        length_scale=2.0, length_scale_bounds=(1e-2, 1e2)
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    models["GPR_RBF"] = GaussianProcessRegressor(
        kernel=kernel_r, n_restarts_optimizer=10,
        normalize_y=True, random_state=42
    )

    # Bayesian Ridge
    models["BayesianRidge"] = BayesianRidge(
        alpha_1=1e-6, alpha_2=1e-6,
        lambda_1=1e-6, lambda_2=1e-6,
        compute_score=True, max_iter=500
    )

    # Quantile Regression models removed - they don't serialize well with joblib
    # Use XGBoost or LightGBM for quantile regression if needed

    # Random Forest (ensemble baseline)
    models["RandomForest"] = RandomForestRegressor(
        n_estimators=100, max_depth=3,
        random_state=42, n_jobs=-1
    )

    # Ridge (regularized baseline)
    models["Ridge"] = Ridge(alpha=1.0)

    return models


# ---------------------------------------------------------------------------
# 3. WALK-FORWARD CROSS-VALIDATION (Duo's approach)
# ---------------------------------------------------------------------------

def walk_forward_cv(df, models_dict, features, experiment_name):
    """
    Walk-forward validation:
      Window 1: Train Aug-Sep,  Test Oct
      Window 2: Train Aug-Oct,  Test Nov
      Window 3: Train Aug-Nov,  Test Dec
    """
    mlruns_dir = os.path.join(BASE_DIR, "mlruns")
    os.makedirs(mlruns_dir, exist_ok=True)
    mlflow.set_tracking_uri("file:///" + mlruns_dir.replace(os.sep, "/"))
    mlflow.set_experiment(experiment_name)

    windows = [
        {"train_months": [8, 9],      "test_month": 10, "label": "W1_TestOct"},
        {"train_months": [8, 9, 10],  "test_month": 11, "label": "W2_TestNov"},
        {"train_months": [8, 9, 10, 11], "test_month": 12, "label": "W3_TestDec"},
    ]

    # Only use rows that have a valid growth multiplier (need PrevMonth)
    valid_df = df[df["growth_multiplier"].notna() & np.isfinite(df["growth_multiplier"])].copy()

    all_results = []
    scaler = StandardScaler()

    for window in windows:
        train_mask = valid_df["Month_Num"].isin(window["train_months"])
        test_mask = valid_df["Month_Num"] == window["test_month"]

        train_data = valid_df[train_mask]
        test_data = valid_df[test_mask]

        if len(train_data) == 0 or len(test_data) == 0:
            print(f"  [SKIP] {window['label']}: insufficient data")
            continue

        X_train = train_data[features].values
        X_test = test_data[features].values

        # TARGET: growth multiplier (not raw sales!)
        y_train_ratio = train_data["growth_multiplier"].values
        y_test_ratio = test_data["growth_multiplier"].values

        # For back-conversion: actual sales and previous month
        y_test_actual = test_data["MonthlySale"].values
        prev_monthly_test = test_data["PrevsMonthly"].values

        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        test_branches = test_data["branch_name"].values

        for model_name, model_template in models_dict.items():
            model = clone(model_template)

            with mlflow.start_run(run_name=f"{model_name}_{window['label']}"):
                model.fit(X_train_s, y_train_ratio)

                # Predict growth multiplier
                has_std = hasattr(model, "predict") and (
                    isinstance(model, (GaussianProcessRegressor, BayesianRidge))
                )
                if has_std:
                    try:
                        pred_ratio, pred_std = model.predict(X_test_s, return_std=True)
                    except TypeError:
                        pred_ratio = model.predict(X_test_s)
                        pred_std = np.zeros(len(pred_ratio))
                else:
                    pred_ratio = model.predict(X_test_s)
                    pred_std = np.zeros(len(pred_ratio))

                # Back-convert to actual sales: predicted_sale = predicted_ratio * prev_month
                pred_sales = np.maximum(pred_ratio * prev_monthly_test, 0)

                # Compute metrics on BOTH the ratio and the back-converted sales
                ratio_mape = float(np.mean(np.abs(
                    (y_test_ratio - pred_ratio) / (np.abs(y_test_ratio) + 1e-6)
                )) * 100)

                # Sales-level metrics
                valid_m = (y_test_actual > 0) & (pred_sales > 0) & np.isfinite(pred_sales)
                if valid_m.sum() > 0:
                    sales_rmse = float(np.sqrt(mean_squared_error(y_test_actual[valid_m], pred_sales[valid_m])))
                    sales_mape = float(mean_absolute_percentage_error(y_test_actual[valid_m], pred_sales[valid_m]) * 100)
                    sales_r2 = float(r2_score(y_test_actual[valid_m], pred_sales[valid_m]))
                else:
                    sales_rmse, sales_mape, sales_r2 = 1e12, 999, -999

                avg_ci = float(np.mean(pred_std * 2)) if pred_std.sum() > 0 else 0

                # Log to MLFlow
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("window", window["label"])
                mlflow.log_param("target_type", "growth_multiplier")
                mlflow.log_param("train_months", str(window["train_months"]))
                mlflow.log_param("test_month", window["test_month"])
                mlflow.log_metric("ratio_mape", round(ratio_mape, 2))
                mlflow.log_metric("sales_rmse", round(sales_rmse, 2))
                mlflow.log_metric("sales_mape", round(sales_mape, 2))
                mlflow.log_metric("sales_r2", round(sales_r2, 4))
                mlflow.log_metric("avg_ci_width", round(avg_ci, 4))

                result = {
                    "model": model_name, "window": window["label"],
                    "test_month": window["test_month"],
                    "ratio_mape": ratio_mape,
                    "sales_rmse": sales_rmse,
                    "sales_mape": sales_mape,
                    "sales_r2": sales_r2,
                    "avg_ci": avg_ci,
                    "pred_ratio": pred_ratio.copy(),
                    "pred_sales": pred_sales.copy(),
                    "actual_sales": y_test_actual.copy(),
                    "branches": test_branches.copy(),
                    "model_obj": model,
                    "scaler": clone(scaler).fit(X_train),  # save fitted scaler
                }
                all_results.append(result)

                status = "PASS" if sales_mape < 80 else "WARN" if sales_mape < 150 else "FAIL"
                print(f"  [{status}] {model_name:18s} {window['label']:12s} | "
                      f"Ratio MAPE: {ratio_mape:6.1f}% | "
                      f"Sales MAPE: {sales_mape:6.1f}% | "
                      f"RMSE: {sales_rmse/1e9:.3f}B | "
                      f"R2: {sales_r2:+.3f}")

    return all_results


# ---------------------------------------------------------------------------
# 4. FINAL MODEL SELECTION & FORECASTER CLASS
# ---------------------------------------------------------------------------

class DemandForecaster:
    """Production forecaster using ratio-based predictions."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_cols = FEATURES_RATIO
        self.model_name = ""
        self.mape = 0.0
        self.branch_map = {
            "Conut": 0, "Conut Main": 0,
            "Conut - Tyre": 1, "Conut Tyre": 1,
            "Conut Jnah": 2,
            "Main Street Coffee": 3, "Main St Coffee": 3,
        }
        # December actuals for Jan forecast lag
        self.dec_actuals = {}

    def fit(self, df, model, model_name, mape, dec_actuals=None):
        """Train on all rows with valid growth multiplier."""
        valid = df[df["growth_multiplier"].notna() & np.isfinite(df["growth_multiplier"])].copy()

        X = valid[self.feature_cols].values
        y = valid["growth_multiplier"].values

        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.model = clone(model)
        self.model.fit(X_scaled, y)
        self.model_name = model_name
        self.mape = mape

        if dec_actuals:
            self.dec_actuals = dec_actuals
        else:
            # Extract December actuals for January forecasting
            dec = df[df["Month_Num"] == 12]
            for _, row in dec.iterrows():
                self.dec_actuals[row.get("branch_name", "")] = row["MonthlySale"]

        print(f"[DemandForecaster V3] Trained {model_name} on {len(valid)} rows (ratio target)")
        return self

    def predict(self, branch_name, month, year=2026):
        """Predict demand with confidence interval."""
        # Resolve branch encoding
        branch_enc = self.branch_map.get(branch_name)
        if branch_enc is None:
            # Fuzzy match
            for k, v in self.branch_map.items():
                if branch_name.lower() in k.lower() or k.lower() in branch_name.lower():
                    branch_enc = v
                    break
            if branch_enc is None:
                branch_enc = 0

        # Get previous month's sales for back-conversion
        prev_sales = self.dec_actuals.get(branch_name, 800_000_000)
        # Also try fuzzy match on dec_actuals
        if branch_name not in self.dec_actuals:
            for k, v in self.dec_actuals.items():
                if branch_name.lower() in k.lower() or k.lower() in branch_name.lower():
                    prev_sales = v
                    break

        # Build features
        days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
                7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}.get(month, 30)
        weekend_days = 8
        holiday_month = 1 if month in [10, 12] else 0
        hi = HOLIDAY_INTENSITY.get(month, 0.0)
        is_coastal = 1 if branch_enc in [1, 2] else 0
        rdp = is_coastal * hi

        # Operational days ratio
        is_conut_main = (branch_enc == 0)
        if is_conut_main and month == 12:
            op_ratio, br_status = 0.05, 1
        else:
            op_ratio, br_status = 1.0, 0

        feature_vec = np.array([
            branch_enc, month,
            hi, op_ratio, rdp, br_status,
            days, weekend_days, holiday_month,
            weekend_days / days, int(weekend_days >= 10),
            6, 0, is_coastal,
            np.log1p(prev_sales),
            1.0,  # relative_size placeholder
        ]).reshape(1, -1)

        X_scaled = self.scaler.transform(feature_vec)

        # Predict growth multiplier
        std_dev = 0.0
        try:
            pred_ratio, pred_std = self.model.predict(X_scaled, return_std=True)
            std_dev = float(pred_std[0])
        except TypeError:
            pred_ratio = self.model.predict(X_scaled)

        ratio = float(pred_ratio[0])
        predicted_volume = max(0, ratio * prev_sales)

        # Confidence interval from model uncertainty or MAPE
        if std_dev > 0:
            upper_ratio = ratio + std_dev
            lower_ratio = max(0, ratio - std_dev)
            upper = upper_ratio * prev_sales
            lower = lower_ratio * prev_sales
        else:
            margin = predicted_volume * (self.mape / 100)
            lower = max(0, predicted_volume - margin)
            upper = predicted_volume + margin

        return {
            "branch": branch_name,
            "predicted_volume": round(predicted_volume, 2),
            "confidence_interval": f"{lower:,.0f} to {upper:,.0f}",
            "mape": round(self.mape, 1),
            "warning": f"Model operates with +/- {self.mape:.0f}% historical error rate.",
            "month": month,
            "year": year,
            "xai_drivers": {
                "model_type": self.model_name,
                "target_method": "growth_multiplier_ratio",
                "predicted_growth_ratio": f"{ratio:.3f}x",
                "previous_month_sales": f"{prev_sales/1e9:.2f}B",
                "Holiday_Intensity": str(hi),
                "Operational_Days_Ratio": str(op_ratio),
                "Is_Coastal": str(is_coastal),
                "Branch_Status": "Closed/Maintenance" if br_status else "Normal",
            },
            "model_type": f"{self.model_name} (V3 ratio-based, MAPE={self.mape:.1f}%)",
        }

    def save(self, path=None):
        if path is None:
            path = os.path.join(MODELS_DIR, "demand_forecaster.pkl")
        joblib.dump(self, path)
        print(f"[DemandForecaster V3] Saved to {path}")

    @staticmethod
    def load(path=None):
        """Load forecaster from disk. Handles path resolution for Uvicorn and module path issues."""
        import sys
        
        # Add unpickling helper to map __main__.DemandForecaster -> models.demand_forecaster.DemandForecaster
        # This handles models saved when running the script directly as __main__
        original_main = sys.modules.get('__main__')
        if 'models.demand_forecaster' in sys.modules:
            sys.modules['__main__'] = sys.modules['models.demand_forecaster']
        
        try:
            if path is None:
                # Try multiple path resolution strategies
                # Strategy 1: Use MODELS_DIR from module (works when run directly)
                path = os.path.join(MODELS_DIR, "demand_forecaster.pkl")
                
                # Strategy 2: If not found, try from current working directory
                if not os.path.exists(path):
                    cwd_path = os.path.join(os.getcwd(), "models", "demand_forecaster.pkl")
                    if os.path.exists(cwd_path):
                        path = cwd_path
                        print(f"[DemandForecaster] Using CWD path: {path}")
                
                # Strategy 3: Try from parent of current file (for Uvicorn from app/)
                if not os.path.exists(path):
                    # When running from app/, models is in parent
                    alt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                            "models", "demand_forecaster.pkl")
                    if os.path.exists(alt_path):
                        path = alt_path
                        print(f"[DemandForecaster] Using alternate path: {path}")
            
            if not os.path.exists(path):
                raise FileNotFoundError(f"DemandForecaster model not found at: {path}")
                
            return joblib.load(path)
        finally:
            # Restore original __main__
            if original_main is not None:
                sys.modules['__main__'] = original_main
        
    def fallback_predict(self, *args, **kwargs):
        """Dummy method to allow unpickling of older model versions that referenced this."""
        pass


# ---------------------------------------------------------------------------
# 5. MAIN
# ---------------------------------------------------------------------------

def train_and_save_model():
    print("=" * 70)
    print("  Project C.O.C.O. - Demand Forecaster V3")
    print("  Innovation: Ratio-Based Target Engineering")
    print("=" * 70)

    # 1. Load and engineer
    print("\n[1/4] Engineering features...")
    df = load_and_engineer()
    valid = df[df["growth_multiplier"].notna() & np.isfinite(df["growth_multiplier"])]
    print(f"  Total rows: {len(df)} | Valid (have prev month): {len(valid)}")
    print(f"  Growth multiplier range: {valid['growth_multiplier'].min():.3f} to {valid['growth_multiplier'].max():.3f}")

    # 2. Walk-forward CV
    print("\n[2/4] Walk-Forward Cross-Validation (5 models x 3 windows)...")
    models = build_models()
    results = walk_forward_cv(df, models, FEATURES_RATIO, "coco_demand_v3_ratio")

    # 3. Analyze results
    print("\n[3/4] Selecting best model...")

    # Build summary table
    summary = {}
    for r in results:
        key = r["model"]
        if key not in summary:
            summary[key] = {"mapes": [], "rmses": [], "r2s": []}
        summary[key]["mapes"].append(r["sales_mape"])
        summary[key]["rmses"].append(r["sales_rmse"])
        summary[key]["r2s"].append(r["sales_r2"])

    with open(os.path.join(BASE_DIR, "outputs", "v3_results.txt"), "w") as f:
        f.write("V3 RATIO-BASED DEMAND FORECASTER — RESULTS\n")
        f.write("=" * 70 + "\n\n")

        header = f"{'Model':18s} | {'Avg MAPE':>10s} | {'W1(Oct)':>10s} | {'W2(Nov)':>10s} | {'W3(Dec)':>10s} | {'Avg R2':>8s}"
        f.write(header + "\n")
        f.write("-" * 80 + "\n")
        print(f"\n  {header}")
        print(f"  {'-'*80}")

        ranked = sorted(summary.items(), key=lambda x: np.mean(x[1]["mapes"]))
        for name, data in ranked:
            mapes = data["mapes"]
            avg_mape = np.mean(mapes)
            avg_r2 = np.mean(data["r2s"])
            mape_strs = [f"{m:8.1f}%" for m in mapes]
            line = f"  {name:18s} | {avg_mape:8.1f}% | {' | '.join(mape_strs)} | {avg_r2:+.3f}"
            print(line)
            f.write(line + "\n")

        # Best model = lowest average MAPE across all windows
        best_name = ranked[0][0]
        best_avg_mape = np.mean(ranked[0][1]["mapes"])

        f.write(f"\nWINNER: {best_name} (Avg MAPE: {best_avg_mape:.1f}%)\n")
        print(f"\n  WINNER: {best_name} (Avg MAPE: {best_avg_mape:.1f}%)")

        # Print per-branch December predictions
        dec_results = [r for r in results if r["test_month"] == 12 and r["model"] == best_name]
        if dec_results:
            r = dec_results[0]
            f.write(f"\nDec Predictions ({best_name}):\n")
            print(f"\n  Dec Predictions ({best_name}):")
            for i in range(len(r["branches"])):
                actual = r["actual_sales"][i]
                pred = r["pred_sales"][i]
                err_pct = abs(pred - actual) / actual * 100
                line = f"    {r['branches'][i]:15s} | Actual: {actual/1e9:.3f}B | Pred: {pred/1e9:.3f}B | Err: {err_pct:.1f}%"
                f.write(line + "\n")
                print(line)

    # 4. Train final model and save
    print("\n[4/4] Training final model on all data...")
    best_model_template = models[best_name]
    forecaster = DemandForecaster()
    forecaster.fit(df, best_model_template, best_name, best_avg_mape)

    # Sample predictions
    print("\n  Sample January 2026 Predictions:")
    for branch in ["Conut Jnah", "Conut Tyre", "Main St Coffee", "Conut Main"]:
        r = forecaster.predict(branch, 1, 2026)
        print(f"    {branch:18s}: {r['predicted_volume']:>15,.0f} [{r['confidence_interval']}]")

    forecaster.save()

    print(f"\n{'='*70}")
    print(f"  Model saved. Run 'mlflow ui' in project root to explore results.")
    print(f"{'='*70}")
