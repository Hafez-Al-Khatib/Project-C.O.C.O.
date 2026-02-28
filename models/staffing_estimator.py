"""
Objective 4 — Shift Staffing Estimation
=========================================
Estimates required employees per shift using predicted demand volume and time-related operational data.
Trains and tracks RandomForest, XGBoost, and LightGBM using MLFlow.
"""

import os
import pandas as pd
import numpy as np
import mlflow
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Setup MLflow
mlflow.set_tracking_uri("sqlite:///mlruns.db")
mlflow.set_experiment("Objective4_Staffing_Estimation")

class StaffingEstimator:
    def __init__(self):
        self.models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42),
            "XGBoost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
            "LightGBM": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        }
        self.best_model_name = "RandomForest" # Default
        self.best_model_instance = self.models["RandomForest"]
        self.branch_map = {}
        self.throughput_stats = {}

    def prepare_data(self):
        labor_df = pd.read_parquet(os.path.join(CLEANED_DIR, "labor_hours.parquet"))
        monthly_sales = pd.read_parquet(os.path.join(CLEANED_DIR, "monthly_sales.parquet"))

        # 1. Aggregate Labor to daily staff count
        labor_df["date"] = pd.to_datetime(labor_df["date"])
        daily_staff = labor_df.groupby(["branch", "date"])["employee_id"].nunique().reset_index()
        daily_staff.rename(columns={"employee_id": "staff_count"}, inplace=True)
        
        # 2. Map monthly volume to daily proxy
        daily_staff["month"] = daily_staff["date"].dt.month
        
        # Count active days per month per branch
        active_days = daily_staff.groupby(["branch", "month"])["date"].nunique().reset_index()
        active_days.rename(columns={"date": "active_days_in_month"}, inplace=True)
        
        # Merge actual monthly sales with active days
        monthly_sales = pd.merge(monthly_sales, active_days, on=["branch", "month"], how="inner")
        monthly_sales["avg_daily_volume"] = monthly_sales["total_sales"] / monthly_sales["active_days_in_month"]
        
        # Merge proxy volume back to daily staff
        merged = pd.merge(daily_staff, monthly_sales[["branch", "month", "avg_daily_volume"]], on=["branch", "month"], how="inner")
        merged.rename(columns={"avg_daily_volume": "daily_volume"}, inplace=True)

        # Calculate throughput benchmark (Volume per employee)
        merged["throughput"] = merged["daily_volume"] / merged["staff_count"]
        self.throughput_stats = merged.groupby("branch")["throughput"].median().to_dict()

        # Time Features
        merged["day_of_week"] = merged["date"].dt.dayofweek
        merged["is_weekend"] = merged["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)

        # Encode Branch
        branches = merged["branch"].unique()
        self.branch_map = {b: i for i, b in enumerate(branches)}
        merged["branch_encoded"] = merged["branch"].map(self.branch_map)

        return merged

    def fit(self):
        df = self.prepare_data()
        
        features = ["daily_volume", "day_of_week", "is_weekend", "branch_encoded"]
        target = "staff_count"

        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        best_mae = float("inf")

        for model_name, model in self.models.items():
            with mlflow.start_run(run_name=f"Staffing_{model_name}"):
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                
                mae = mean_absolute_error(y_test, preds)
                mape = mean_absolute_percentage_error(y_test, preds) * 100
                
                mlflow.log_param("model_type", model_name)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("mape", mape)
                
                print(f"[StaffingEstimator] {model_name:15s} | MAE: {mae:.2f} employees | MAPE: {mape:.1f}%")
                
                if mae < best_mae:
                    best_mae = mae
                    self.best_model_name = model_name
                    self.best_model_instance = model
        
        print(f"[StaffingEstimator] Selected {self.best_model_name} as final production model.")
        return self

    def predict(self, branch_name, predicted_volume, date=None):
        if not self.branch_map:
            raise ValueError("Model not trained yet.")
            
        branch_encoded = self.branch_map.get(branch_name, 0)
        
        if date:
            day_of_week = pd.to_datetime(date).dayofweek
        else:
            day_of_week = 4 # Default to Friday
            
        is_weekend = 1 if day_of_week >= 5 else 0
        
        if not predicted_volume:
            throughput = self.throughput_stats.get(branch_name, np.mean(list(self.throughput_stats.values())))
            return {
                "branch": branch_name,
                "predicted_volume": 0,
                "recommended_staff": 3,
                "throughput_metric": round(throughput, 2),
                "xai_drivers": {"Fallback": "No volume provided, returning baseline staff."}
            }

        X_infer = pd.DataFrame([{
            "daily_volume": predicted_volume,
            "day_of_week": day_of_week,
            "is_weekend": is_weekend,
            "branch_encoded": branch_encoded
        }])

        raw_pred = self.best_model_instance.predict(X_infer)[0]
        recommended_staff = int(np.ceil(raw_pred))
        
        throughput = predicted_volume / recommended_staff if recommended_staff > 0 else 0
        bench_tp = self.throughput_stats.get(branch_name, 0)
        
        efficiency_note = "Expected throughput is "
        if throughput > bench_tp * 1.2:
            efficiency_note += f"HIGH (Stress warning: {throughput:.0f} vs norm {bench_tp:.0f})"
        elif throughput < bench_tp * 0.8:
            efficiency_note += f"LOW (Overstaffed: {throughput:.0f} vs norm {bench_tp:.0f})"
        else:
            efficiency_note += "OPTIMAL"

        return {
            "branch": branch_name,
            "predicted_volume": predicted_volume,
            "recommended_staff": max(1, recommended_staff),
            "throughput_metric": round(throughput, 2),
            "xai_drivers": {
                "Model Logic": f"{self.best_model_name} mapped expected demand volume to optimal daily headcount.",
                "Efficiency Analysis": efficiency_note
            },
            "model_type": self.best_model_name
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
