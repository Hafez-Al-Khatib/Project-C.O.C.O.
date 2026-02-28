"""
Objective 4 — Shift Staffing Estimation (Probabilistic)
=========================================================
Estimates required employees per shift using predicted demand volume.
Uses GPR, BayesianRidge, and Quantile Regression with walk-forward CV.
All runs tracked via MLFlow under a unified file-based backend.
"""

import os
import warnings
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel, Matern
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.base import clone
import joblib

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Unified MLFlow backend (file-based, shared with Obj 2)
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")
os.makedirs(MLRUNS_DIR, exist_ok=True)
mlflow.set_tracking_uri("file:///" + MLRUNS_DIR.replace(os.sep, "/"))
mlflow.set_experiment("Objective4_Staffing_Estimation")

WINDOWS = [
    {"train": [8, 9],         "test": 10, "label": "W1_Oct"},
    {"train": [8, 9, 10],     "test": 11, "label": "W2_Nov"},
    {"train": [8, 9, 10, 11], "test": 12, "label": "W3_Dec"},
]


def get_models():
    kernel = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=2.5
    ) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))

    return {
        "GPR": GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10,
            normalize_y=True, random_state=42
        ),
        "BayesianRidge": BayesianRidge(
            alpha_1=1e-6, alpha_2=1e-6,
            lambda_1=1e-6, lambda_2=1e-6,
            compute_score=True, max_iter=500
        ),
        "QR_Q90": GradientBoostingRegressor(
            loss="quantile", alpha=0.9,
            n_estimators=200, max_depth=2,
            learning_rate=0.05, subsample=0.8, random_state=42
        ),
    }


class StaffingEstimator:
    def __init__(self):
        self.best_model_name = "GPR"
        self.best_model_instance = None
        self.branch_map = {}
        self.throughput_stats = {}
        self.scaler = StandardScaler()
        self.cv_results = []

    def prepare_data(self):
        labor_df = pd.read_parquet(os.path.join(CLEANED_DIR, "labor_hours.parquet"))
        monthly_sales = pd.read_parquet(os.path.join(CLEANED_DIR, "monthly_sales.parquet"))

        labor_df["date"] = pd.to_datetime(labor_df["date"])
        daily_staff = labor_df.groupby(["branch", "date"])["employee_id"].nunique().reset_index()
        daily_staff.rename(columns={"employee_id": "staff_count"}, inplace=True)

        daily_staff["month"] = daily_staff["date"].dt.month

        active_days = daily_staff.groupby(["branch", "month"])["date"].nunique().reset_index()
        active_days.rename(columns={"date": "active_days_in_month"}, inplace=True)

        monthly_sales = pd.merge(monthly_sales, active_days, on=["branch", "month"], how="inner")
        monthly_sales["avg_daily_volume"] = monthly_sales["total_sales"] / monthly_sales["active_days_in_month"]

        merged = pd.merge(
            daily_staff,
            monthly_sales[["branch", "month", "avg_daily_volume"]],
            on=["branch", "month"], how="inner"
        )
        merged.rename(columns={"avg_daily_volume": "daily_volume"}, inplace=True)

        merged["throughput"] = merged["daily_volume"] / merged["staff_count"]
        self.throughput_stats = merged.groupby("branch")["throughput"].median().to_dict()

        merged["day_of_week"] = merged["date"].dt.dayofweek
        merged["is_weekend"] = merged["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

        branches = merged["branch"].unique()
        self.branch_map = {b: i for i, b in enumerate(branches)}
        merged["branch_encoded"] = merged["branch"].map(self.branch_map)

        return merged

    def fit(self):
        df = self.prepare_data()

        features = ["daily_volume", "day_of_week", "is_weekend", "branch_encoded"]
        target = "staff_count"

        models = get_models()
        best_mae = float("inf")
        self.cv_results = []

        for window in WINDOWS:
            train_mask = df["month"].isin(window["train"])
            test_mask = df["month"] == window["test"]

            train_df = df[train_mask]
            test_df = df[test_mask]

            if len(train_df) == 0 or len(test_df) == 0:
                continue

            X_train = train_df[features].values
            y_train = train_df[target].values
            X_test = test_df[features].values
            y_test = test_df[target].values

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            for model_name, model_template in models.items():
                model = clone(model_template)
                run_name = f"Staffing_{model_name}_{window['label']}"

                with mlflow.start_run(run_name=run_name):
                    model.fit(X_train_s, y_train)

                    # Probabilistic prediction
                    if hasattr(model, "predict") and model_name in ("GPR", "BayesianRidge"):
                        y_pred, y_std = model.predict(X_test_s, return_std=True)
                        avg_ci_width = float(np.mean(y_std * 2))
                    else:
                        y_pred = model.predict(X_test_s)
                        y_std = np.zeros_like(y_pred)
                        avg_ci_width = 0.0

                    mae = mean_absolute_error(y_test, y_pred)
                    mape = mean_absolute_percentage_error(y_test, y_pred) * 100
                    r2 = r2_score(y_test, y_pred)

                    mlflow.log_param("model_type", model_name)
                    mlflow.log_param("window", window["label"])
                    mlflow.log_param("test_month", window["test"])
                    mlflow.log_param("features", str(features))
                    mlflow.log_param("n_train", len(X_train))
                    mlflow.log_param("n_test", len(X_test))
                    mlflow.log_metric("mae", round(mae, 4))
                    mlflow.log_metric("mape", round(mape, 2))
                    mlflow.log_metric("r2", round(r2, 4))
                    mlflow.log_metric("avg_ci_width", round(avg_ci_width, 4))

                    print(f"  [{window['label']}] {model_name:15s} | MAE: {mae:.2f} | MAPE: {mape:.1f}% | R²: {r2:+.3f} | CI Width: {avg_ci_width:.2f}")

                    self.cv_results.append({
                        "window": window["label"],
                        "model": model_name,
                        "mae": mae, "mape": mape, "r2": r2,
                        "avg_ci_width": avg_ci_width,
                    })

                    if mae < best_mae:
                        best_mae = mae
                        self.best_model_name = model_name
                        self.best_model_instance = model
                        self.scaler = scaler  # persist the fitted scaler

        # Retrain best model on ALL data for production deployment
        all_X = df[features].values
        all_y = df[target].values
        self.scaler = StandardScaler()
        all_X_s = self.scaler.fit_transform(all_X)

        final_model = clone(get_models()[self.best_model_name])
        final_model.fit(all_X_s, all_y)
        self.best_model_instance = final_model

        print(f"\n[StaffingEstimator] Best model: {self.best_model_name} (CV MAE: {best_mae:.2f})")
        print(f"[StaffingEstimator] Retrained on full dataset ({len(all_X)} rows) for production.")
        return self

    def predict(self, branch_name, predicted_volume, date=None):
        if not self.branch_map:
            raise ValueError("Model not trained yet.")

        branch_encoded = self.branch_map.get(branch_name, 0)

        if date:
            day_of_week = pd.to_datetime(date).dayofweek
        else:
            day_of_week = 4

        is_weekend = 1 if day_of_week >= 5 else 0

        if not predicted_volume:
            throughput = self.throughput_stats.get(
                branch_name, np.mean(list(self.throughput_stats.values()))
            )
            return {
                "branch": branch_name,
                "predicted_volume": 0,
                "recommended_staff": 3,
                "confidence_band": "N/A",
                "throughput_metric": round(throughput, 2),
                "xai_drivers": {
                    "Fallback": "No volume provided, returning baseline staff.",
                },
                "model_type": self.best_model_name,
            }

        X_infer = np.array([[predicted_volume, day_of_week, is_weekend, branch_encoded]])
        X_infer_s = self.scaler.transform(X_infer)

        # Probabilistic prediction with confidence band
        if self.best_model_name in ("GPR", "BayesianRidge"):
            raw_pred, raw_std = self.best_model_instance.predict(X_infer_s, return_std=True)
            raw_pred = raw_pred[0]
            raw_std = raw_std[0]
            lower = max(1, int(np.floor(raw_pred - 1.96 * raw_std)))
            upper = int(np.ceil(raw_pred + 1.96 * raw_std))
            confidence_band = f"{lower} to {upper} staff (95% CI)"
        else:
            raw_pred = self.best_model_instance.predict(X_infer_s)[0]
            raw_std = 0.0
            confidence_band = "Point estimate (no CI from Quantile model)"

        recommended_staff = max(1, int(np.ceil(raw_pred)))

        throughput = predicted_volume / recommended_staff if recommended_staff > 0 else 0
        bench_tp = self.throughput_stats.get(branch_name, 0)

        if bench_tp > 0:
            if throughput > bench_tp * 1.2:
                efficiency = f"HIGH stress ({throughput:.0f} vs benchmark {bench_tp:.0f})"
            elif throughput < bench_tp * 0.8:
                efficiency = f"LOW utilization ({throughput:.0f} vs benchmark {bench_tp:.0f})"
            else:
                efficiency = "OPTIMAL range"
        else:
            efficiency = "No historical benchmark available"

        day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        return {
            "branch": branch_name,
            "predicted_volume": predicted_volume,
            "recommended_staff": recommended_staff,
            "confidence_band": confidence_band,
            "throughput_metric": round(throughput, 2),
            "xai_drivers": {
                "Model": f"{self.best_model_name} with walk-forward cross-validation",
                "Inference Day": f"{day_names[day_of_week]} ({'Weekend' if is_weekend else 'Weekday'})",
                "Efficiency": efficiency,
                "Uncertainty": f"±{raw_std:.1f} staff (1σ)" if raw_std > 0 else "Deterministic estimate",
            },
            "model_type": self.best_model_name,
        }

    def save(self, path=None):
        if path is None:
            path = os.path.join(MODELS_DIR, "staffing_estimator.pkl")
        joblib.dump(self, path)
        print(f"[StaffingEstimator] Saved to {path}")

    @staticmethod
    def load(path=None):
        if path is None:
            path = os.path.join(MODELS_DIR, "staffing_estimator.pkl")
        return joblib.load(path)


if __name__ == "__main__":
    est = StaffingEstimator().fit()
    print("\n[Test Inference - Main Street Coffee]")
    print(est.predict("Main Street Coffee", 150000))
    est.save()
