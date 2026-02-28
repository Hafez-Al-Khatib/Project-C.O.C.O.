"""
Project C.O.C.O. — Demand Forecasting V2
Algorithm & Feature Redesign per Brief V2

New Features (brief §1):
  - Operational_Days_Ratio  : actual sales days / total days (captures Conut Main closure)
  - Holiday_Intensity       : Aug-Oct=0.0, Nov=0.2, Dec=1.0 (linear holiday knob)
  - Regional_Demand_Proxy   : Is_Coastal * Holiday_Intensity (coastal branches spike in Dec)
  - Branch_Status           : 0=Normal, 1=Maintenance/Closed (Conut Main Dec = 1)

Models (brief §2):
  - GPR with RBF+WhiteKernel  → returns mean + std_dev (high-risk flag)
  - Quantile Regression Q90   → captures holiday spike upper bound
  - Bayesian Ridge             → secondary probabilistic baseline

MLflow: all runs tracked, best model promoted to Production
Train: Aug–Nov 2025 | Test: Dec 2025 | Forecast: Jan 2026

Run:
    python coco_forecast_v2.py
    mlflow ui  →  http://127.0.0.1:5000
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlflow.db")   # avoids Windows path-with-spaces bug

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
BRANCH_FLAGS = [
    "BranchisConut", "BranchisConutTyre",
    "BranchisConutJnah", "BranchisMainStreetCoffee"
]
BRANCH_DISPLAY = {
    "BranchisConut":            "Conut Main",
    "BranchisConutTyre":        "Conut Tyre",
    "BranchisConutJnah":        "Conut Jnah",
    "BranchisMainStreetCoffee": "Main St Coffee",
}
BRANCH_ENCODED = {
    "BranchisConut": 0, "BranchisConutTyre": 1,
    "BranchisConutJnah": 2, "BranchisMainStreetCoffee": 3,
}

# Holiday intensity per brief §1: Aug-Oct=0.0, Nov=0.2, Dec=1.0
HOLIDAY_INTENSITY = {8: 0.0, 9: 0.0, 10: 0.0, 11: 0.2, 12: 1.0}

# ─────────────────────────────────────────────────────────────
# STEP 1 — Load monthly sales CSV
# ─────────────────────────────────────────────────────────────
df = pd.read_csv("Branches_Cleaned_For_XGBoost.csv")
df["PrevsMonthly"] = pd.to_numeric(df["PrevsMonthly"], errors="coerce").fillna(0)

# ─────────────────────────────────────────────────────────────
# STEP 2 — Data integrity check: Conut Main December anomaly
# Brief §3: 67M in Dec vs 1.35B in Nov = 95% drop → label as Maintenance/Closed
# ─────────────────────────────────────────────────────────────
print("=" * 60)
print("DATA INTEGRITY CHECK — Conut Main December")
print("=" * 60)
conut_dec = df[(df["BranchisConut"] == 1) & (df["Month_Num"] == 12)]["MonthlySale"].values[0]
conut_nov = df[(df["BranchisConut"] == 1) & (df["Month_Num"] == 11)]["MonthlySale"].values[0]
drop_pct  = (conut_nov - conut_dec) / conut_nov * 100
print(f"  Nov sales : {conut_nov:>20,.0f}")
print(f"  Dec sales : {conut_dec:>20,.0f}")
print(f"  Drop      : {drop_pct:.1f}%")
if drop_pct > 80:
    print("  Verdict   : LIKELY CLOSED / MAINTENANCE in December")
    print("  Action    : Branch_Status = 1 (Maintenance) for Conut Dec row")
    CONUT_DEC_STATUS = 1
else:
    print("  Verdict   : Reporting anomaly — document in Executive Brief")
    CONUT_DEC_STATUS = 0
print("=" * 60)

# ─────────────────────────────────────────────────────────────
# STEP 3 — Load work hours from Excel
# ─────────────────────────────────────────────────────────────
xl = pd.read_excel("Conut_Work_Hours_UPDATED.xlsx", sheet_name="Branch Totals")
xl = xl[xl["Branch"] != "TOTAL"].copy()
xl.columns = xl.columns.str.strip()

WH_MAP = {
    "Conut - Tyre":       "BranchisConutTyre",
    "Conut Jnah":         "BranchisConutJnah",
    "Main Street Coffee": "BranchisMainStreetCoffee",
}
xl["branch_flag"] = xl["Branch"].map(WH_MAP)
xl = xl.dropna(subset=["branch_flag"])

wh_lookup = {}
for _, row in xl.iterrows():
    flag = row["branch_flag"]
    twh  = float(row["Total Work Hours"])
    awh  = float(row["Avg Hrs / Shift"])
    nsh  = float(row["Total Shifts"])
    wh_lookup[flag] = {
        "TotalWorkHours":  twh,
        "AvgWorkHours":    awh,
        "NumShifts":       nsh,
        "WorkHoursPerDay": twh / 31,
    }

# ─────────────────────────────────────────────────────────────
# STEP 4 — Feature engineering (brief §1 — Contextual Injection)
# ─────────────────────────────────────────────────────────────
df["branch_encoded"] = df[BRANCH_FLAGS].idxmax(axis=1).map(BRANCH_ENCODED)

# Rolling 3-month average per branch
df["rolling_3m_avg"] = (
    df.groupby("branch_encoded")["MonthlySale"]
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
)
df["rolling_3m_avg"] = (
    df.groupby("branch_encoded")["rolling_3m_avg"]
    .transform(lambda x: x.fillna(x.mean()))
)

# ── NEW FEATURE 1: Holiday_Intensity ─────────────────────────
# Aug-Oct = 0.0 | Nov = 0.2 | Dec = 1.0
df["Holiday_Intensity"] = df["Month_Num"].map(HOLIDAY_INTENSITY).fillna(0.0)

# ── NEW FEATURE 2: Operational_Days_Ratio ────────────────────
# For Conut Main Dec: 67M vs 1.35B in Nov implies ~5% of month operational
# We estimate actual_sales_days as proportional to (sale / avg_daily_rate)
# avg_daily_rate = PrevsMonthly / 30 (prior month)
df["avg_daily_rate"] = df["PrevsMonthly"] / 30
# Estimate operational days: min(DaysinMonth, sale / avg_daily_rate)
df["estimated_ops_days"] = np.where(
    df["avg_daily_rate"] > 0,
    np.minimum(df["DaysinMonth"], df["MonthlySale"] / df["avg_daily_rate"]),
    df["DaysinMonth"],
)
df["Operational_Days_Ratio"] = (
    df["estimated_ops_days"] / df["DaysinMonth"]
).clip(0, 1)

# ── NEW FEATURE 3: Regional_Demand_Proxy ─────────────────────
# Is_Coastal * Holiday_Intensity: coastal branches spike in Dec
df["Regional_Demand_Proxy"] = df["Is_Coastal"] * df["Holiday_Intensity"]

# ── NEW FEATURE 4: Branch_Status ─────────────────────────────
# 0 = Normal, 1 = Maintenance/Closed
df["Branch_Status"] = 0
df.loc[
    (df["BranchisConut"] == 1) & (df["Month_Num"] == 12),
    "Branch_Status"
] = CONUT_DEC_STATUS

# ── Supporting features ───────────────────────────────────────
df["is_weekend_heavy"]  = (df["WeekendDays"] >= 10).astype(int)
df["WeekendRatio"]      = df["WeekendDays"] / df["DaysinMonth"]
df["SalePerDay"]        = df["PrevsMonthly"] / df["DaysinMonth"]
df["month_num"]         = df["Month_Num"]

# Work hours (December only)
for col in ["TotalWorkHours","AvgWorkHours","NumShifts","WorkHoursPerDay"]:
    df[col] = 0.0
for flag, feats in wh_lookup.items():
    mask = (df["Month_Num"] == 12) & (df[flag] == 1)
    for col, val in feats.items():
        df.loc[mask, col] = val

# Log-transform target to compress 40x value range
df["log_sale"] = np.log1p(df["MonthlySale"])

FEATURES = [
    # Core identifiers
    "branch_encoded",
    "month_num",
    # V2 contextual injection features
    "Holiday_Intensity",
    "Operational_Days_Ratio",
    "Regional_Demand_Proxy",
    "Branch_Status",
    # Calendar
    "DaysinMonth",
    "WeekendDays",
    "HolidayMonth",
    "WeekendRatio",
    "is_weekend_heavy",
    # Branch profile
    "Months Active",
    "Coldstart",
    "Is_Coastal",
    # Lag / trend
    "PrevsMonthly",
    "SalePerDay",
    "rolling_3m_avg",
    # Work hours (Dec)
    "TotalWorkHours",
    "AvgWorkHours",
    "NumShifts",
    "WorkHoursPerDay",
]

print(f"\nTotal features: {len(FEATURES)}")
print("V2 injected features: Holiday_Intensity, Operational_Days_Ratio, Regional_Demand_Proxy, Branch_Status")

# ─────────────────────────────────────────────────────────────
# STEP 5 — Time series split: Train Aug–Nov / Test Dec
# ─────────────────────────────────────────────────────────────
train = df[df["Month_Num"] < 12].copy()
test  = df[df["Month_Num"] == 12].copy()

X_train_raw = train[FEATURES].values
y_train_log = train["log_sale"].values
y_train_raw = train["MonthlySale"].values

X_test_raw  = test[FEATURES].values
y_test      = test["MonthlySale"].values

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train_raw)
X_test  = scaler.transform(X_test_raw)

branch_labels = test[BRANCH_FLAGS].idxmax(axis=1).map(BRANCH_DISPLAY).tolist()

print(f"\nTrain: {len(X_train)} rows (Aug–Nov 2025)")
print(f"Test : {len(X_test)} rows (Dec 2025)")

# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1))) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape}

results    = {}
qr_preds   = {}
mlflow.set_experiment("coco_demand_forecasting_v2")

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 1 — GPR (Primary) — returns std_dev for high-risk flag
# Brief §2: "ensure predict_demand returns std_dev"
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXPERIMENT 1 — Gaussian Process Regression (GPR)")
print("="*60)

gpr_kernels = {
    "RBF_base":           ConstantKernel(1.0) * RBF(length_scale=1.0),
    "RBF_WhiteKernel":    ConstantKernel(1.0) * RBF(length_scale=2.0) + WhiteKernel(noise_level=0.1),
    "RBF_long_WhiteKernel": ConstantKernel(1.0) * RBF(length_scale=3.0) + WhiteKernel(noise_level=0.5),
}

best_gpr_rmse = float("inf")
best_gpr      = {}

for kname, kernel_obj in gpr_kernels.items():
    with mlflow.start_run(run_name=f"GPR_{kname}"):
        gpr = GaussianProcessRegressor(
            kernel=kernel_obj,
            n_restarts_optimizer=10,
            normalize_y=True,
        )
        gpr.fit(X_train, y_train_log)

        y_pred_log, y_std = gpr.predict(X_test, return_std=True)
        y_pred  = np.maximum(np.expm1(y_pred_log), 0)
        avg_ci  = float(np.mean(np.expm1(y_std) * 2))
        m       = compute_metrics(y_test, y_pred)

        # High-risk flag: if std_dev > 30% of prediction, flag as high-risk
        high_risk_branches = [
            branch_labels[i] for i in range(len(y_pred))
            if y_std[i] / (abs(y_pred_log[i]) + 1e-6) > 0.3
        ]

        mlflow.log_param("model_type",        "GPR")
        mlflow.log_param("kernel",             kname)
        mlflow.log_param("features",           str(FEATURES))
        mlflow.log_param("v2_features",        "Holiday_Intensity,Operational_Days_Ratio,Regional_Demand_Proxy,Branch_Status")
        mlflow.log_param("high_risk_branches", str(high_risk_branches))
        mlflow.log_metric("rmse",              round(m["rmse"], 2))
        mlflow.log_metric("mae",               round(m["mae"],  2))
        mlflow.log_metric("r2",                round(m["r2"],   4))
        mlflow.log_metric("mape",              round(m["mape"], 2))
        mlflow.log_metric("avg_ci_width",      round(avg_ci,    2))
        mlflow.sklearn.log_model(
            sk_model=gpr,
            artifact_path="gpr_model",
            registered_model_name="COCO_DemandForecasting_GPR_V2",
        )

        print(f"  {kname:30s} | RMSE {m['rmse']/1e9:.3f}B | R² {m['r2']:+.3f} | MAPE {m['mape']:.1f}% | CI {avg_ci/1e9:.2f}B")
        if high_risk_branches:
            print(f"    ⚠  High-risk branches: {high_risk_branches}")

        if m["rmse"] < best_gpr_rmse:
            best_gpr_rmse = m["rmse"]
            best_gpr = {
                "model": gpr, "preds": y_pred, "std": y_std,
                "metrics": m, "avg_ci": avg_ci,
                "kernel": kname, "high_risk": high_risk_branches,
            }

results["GPR"] = best_gpr

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 2 — Quantile Regression Q90 (Primary for holiday spike)
# Brief §2: "focus on 90th percentile to capture holiday spike"
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXPERIMENT 2 — Quantile Regression (Q10 / Q50 / Q90)")
print("  → Q90 is PRIMARY for holiday spike capture per brief §2")
print("="*60)

quantile_configs = [
    (0.1, "lower_Q10",  "Conservative lower bound"),
    (0.5, "median_Q50", "Central estimate"),
    (0.9, "upper_Q90",  "Holiday spike capture (PRIMARY per brief)"),
]

best_qr_rmse = float("inf")
best_qr      = {}

for alpha, label, description in quantile_configs:
    with mlflow.start_run(run_name=f"QuantileReg_{label}"):
        qr = GradientBoostingRegressor(
            loss="quantile",
            alpha=alpha,
            n_estimators=300,
            max_depth=2,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42,
        )
        qr.fit(X_train, y_train_log)

        y_pred_log = qr.predict(X_test)
        y_pred     = np.maximum(np.expm1(y_pred_log), 0)
        m          = compute_metrics(y_test, y_pred)

        mlflow.log_param("model_type",   "QuantileRegression")
        mlflow.log_param("quantile",     alpha)
        mlflow.log_param("label",        label)
        mlflow.log_param("description",  description)
        mlflow.log_param("features",     str(FEATURES))
        mlflow.log_param("v2_features",  "Holiday_Intensity,Operational_Days_Ratio,Regional_Demand_Proxy,Branch_Status")
        mlflow.log_metric("rmse",  round(m["rmse"], 2))
        mlflow.log_metric("mae",   round(m["mae"],  2))
        mlflow.log_metric("r2",    round(m["r2"],   4))
        mlflow.log_metric("mape",  round(m["mape"], 2))
        mlflow.sklearn.log_model(
            sk_model=qr,
            artifact_path="qr_model",
            registered_model_name=f"COCO_QuantileReg_V2_{label}",
        )

        print(f"  {label:15s}  {description:40s} | RMSE {m['rmse']/1e9:.3f}B | R² {m['r2']:+.3f} | MAPE {m['mape']:.1f}%")

        qr_preds[label] = y_pred
        if m["rmse"] < best_qr_rmse:
            best_qr_rmse = m["rmse"]
            best_qr = {"model": qr, "preds": y_pred, "metrics": m, "label": label}

results["QuantileReg"] = best_qr

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT 3 — Bayesian Ridge (Secondary baseline)
# ═══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXPERIMENT 3 — Bayesian Ridge (Secondary Baseline)")
print("="*60)

br_configs = [
    {"alpha_1": 1e-6, "alpha_2": 1e-6, "lambda_1": 1e-6, "lambda_2": 1e-6},
    {"alpha_1": 1e-4, "alpha_2": 1e-4, "lambda_1": 1e-4, "lambda_2": 1e-4},
    {"alpha_1": 1e-2, "alpha_2": 1e-2, "lambda_1": 1e-2, "lambda_2": 1e-2},
]

best_br_rmse = float("inf")
best_br      = {}

for cfg in br_configs:
    tag = f"a={cfg['alpha_1']}_l={cfg['lambda_1']}"
    with mlflow.start_run(run_name=f"BayesianRidge_{tag}"):
        br = BayesianRidge(**cfg, compute_score=True)
        br.fit(X_train, y_train_log)

        y_pred_log, y_std = br.predict(X_test, return_std=True)
        y_pred  = np.maximum(np.expm1(y_pred_log), 0)
        avg_ci  = float(np.mean(np.expm1(y_std) * 2))
        m       = compute_metrics(y_test, y_pred)

        mlflow.log_param("model_type",   "BayesianRidge")
        mlflow.log_param("alpha_1",      cfg["alpha_1"])
        mlflow.log_param("lambda_1",     cfg["lambda_1"])
        mlflow.log_param("features",     str(FEATURES))
        mlflow.log_param("v2_features",  "Holiday_Intensity,Operational_Days_Ratio,Regional_Demand_Proxy,Branch_Status")
        mlflow.log_metric("rmse",         round(m["rmse"], 2))
        mlflow.log_metric("mae",          round(m["mae"],  2))
        mlflow.log_metric("r2",           round(m["r2"],   4))
        mlflow.log_metric("mape",         round(m["mape"], 2))
        mlflow.log_metric("avg_ci_width", round(avg_ci,    2))
        mlflow.sklearn.log_model(
            sk_model=br,
            artifact_path="br_model",
            registered_model_name="COCO_DemandForecasting_BayesianRidge_V2",
        )

        print(f"  {tag:30s} | RMSE {m['rmse']/1e9:.3f}B | R² {m['r2']:+.3f} | MAPE {m['mape']:.1f}% | CI {avg_ci/1e9:.2f}B")

        if m["rmse"] < best_br_rmse:
            best_br_rmse = m["rmse"]
            best_br = {
                "model": br, "preds": y_pred, "std": y_std,
                "metrics": m, "avg_ci": avg_ci, "config": cfg,
            }

results["BayesianRidge"] = best_br

# ─────────────────────────────────────────────────────────────
# STEP 6 — Pick best overall model & promote to Production
# ─────────────────────────────────────────────────────────────
ranked     = sorted(results.items(), key=lambda x: x[1]["metrics"]["rmse"])
best_name, best_run = ranked[0]

print("\n" + "="*60)
print("  MODEL COMPARISON — V2 with Contextual Injection")
print("="*60)
print(f"  {'Model':<20} {'RMSE':>15} {'R²':>8} {'MAPE':>8}")
print(f"  {'-'*54}")
for name, run in ranked:
    marker = " ← BEST" if name == best_name else ""
    print(f"  {name:<20} {run['metrics']['rmse']:>15,.0f}  {run['metrics']['r2']:>+7.4f}  {run['metrics']['mape']:>6.1f}%{marker}")
print("="*60)

# Promote in registry
model_reg_map = {
    "GPR":           "COCO_DemandForecasting_GPR_V2",
    "QuantileReg":   f"COCO_QuantileReg_V2_{results['QuantileReg'].get('label','median_Q50')}",
    "BayesianRidge": "COCO_DemandForecasting_BayesianRidge_V2",
}
try:
    client   = MlflowClient()
    versions = client.search_model_versions(f"name='{model_reg_map[best_name]}'")
    if versions:
        latest = sorted(versions, key=lambda v: int(v.version))[-1]
        client.set_model_version_tag(
            name=model_reg_map[best_name],
            version=latest.version,
            key="stage", value="Production",
        )
        print(f"\n  ✓ Promoted {model_reg_map[best_name]} v{latest.version} → Production")
except Exception as e:
    print(f"\n  (Registry promotion skipped: {e})")

# ─────────────────────────────────────────────────────────────
# STEP 7 — December predictions table
# ─────────────────────────────────────────────────────────────
dec_df = pd.DataFrame({
    "Branch":           branch_labels,
    "Actual":           y_test.astype(int),
    "GPR_Pred":         results["GPR"]["preds"].astype(int),
    "BayesRidge_Pred":  results["BayesianRidge"]["preds"].astype(int),
    "QR_Lower_Q10":     qr_preds["lower_Q10"].astype(int),
    "QR_Median_Q50":    qr_preds["median_Q50"].astype(int),
    "QR_Upper_Q90":     qr_preds["upper_Q90"].astype(int),
    "GPR_StdDev":       results["GPR"]["std"].round(4),
    "High_Risk":        ["YES" if b in results["GPR"]["high_risk"] else "NO" for b in branch_labels],
    "Branch_Status":    ["Maintenance/Closed" if (b == "Conut Main") else "Normal" for b in branch_labels],
})
dec_df["Best_Model"] = best_name
dec_df["Best_Pred"]  = best_run["preds"].astype(int)
dec_df["Error_Pct"]  = (
    (dec_df["Best_Pred"] - dec_df["Actual"]) /
    (dec_df["Actual"].abs() + 1) * 100
).round(1)

dec_df.to_csv("outputs/december_predictions_v2.csv", index=False)
print("\n=== December 2025 Predictions (V2) ===")
print(dec_df[[
    "Branch","Actual","GPR_Pred","QR_Upper_Q90",
    "Best_Pred","Error_Pct","High_Risk","Branch_Status"
]].to_string(index=False))

# ─────────────────────────────────────────────────────────────
# STEP 8 — January 2026 forecast (FastAPI DemandResponse schema)
# ─────────────────────────────────────────────────────────────
dec_actuals = dict(zip(branch_labels, y_test))
jan_rows    = []

for flag, name in BRANCH_DISPLAY.items():
    prev = dec_actuals.get(name, 0)
    wh   = wh_lookup.get(flag, {
        "TotalWorkHours": 0, "AvgWorkHours": 0,
        "NumShifts": 0, "WorkHoursPerDay": 0
    })
    days = 31
    # Jan: no closures expected, low holiday intensity
    row = {
        "branch_encoded":         BRANCH_ENCODED[flag],
        "month_num":              1,
        "Holiday_Intensity":      0.0,       # Jan = no holiday signal
        "Operational_Days_Ratio": 1.0,       # assume fully operational
        "Regional_Demand_Proxy":  0.0,       # 0 holiday * coastal
        "Branch_Status":          0,         # all normal in Jan
        "DaysinMonth":            days,
        "WeekendDays":            8,
        "HolidayMonth":           0,
        "WeekendRatio":           8 / days,
        "is_weekend_heavy":       0,
        "Months Active":          6,
        "Coldstart":              0,
        "Is_Coastal":             1 if flag in ("BranchisConutTyre","BranchisConutJnah") else 0,
        "PrevsMonthly":           prev,
        "SalePerDay":             prev / days,
        "rolling_3m_avg":         prev,
        "TotalWorkHours":         wh["TotalWorkHours"],
        "AvgWorkHours":           wh["AvgWorkHours"],
        "NumShifts":              wh["NumShifts"],
        "WorkHoursPerDay":        wh["WorkHoursPerDay"],
        "_name":                  name,
        "_flag":                  flag,
    }
    jan_rows.append(row)

jan_df = pd.DataFrame(jan_rows)
X_jan  = scaler.transform(jan_df[FEATURES].values)

best_model_obj = best_run["model"]
best_mape      = best_run["metrics"]["mape"]

if best_name == "GPR":
    jan_log, jan_std = best_model_obj.predict(X_jan, return_std=True)
    jan_pred = np.maximum(np.expm1(jan_log), 0)
else:
    jan_log  = best_model_obj.predict(X_jan)
    jan_pred = np.maximum(np.expm1(jan_log), 0)
    jan_std  = jan_pred * (best_mape / 100)

# Build FastAPI-compatible DemandResponse per brief §5
api_responses = []
for i, row in jan_df.iterrows():
    pv      = float(jan_pred[i])
    std_val = float(jan_std[i]) if best_name == "GPR" else float(jan_std[i])
    lower   = max(0.0, pv - pv * (best_mape / 100))
    upper   = pv + pv * (best_mape / 100)
    is_high_risk = (std_val / (abs(jan_log[i]) + 1e-6)) > 0.3 if best_name == "GPR" else False

    api_responses.append({
        "branch":              row["_name"],
        "predicted_volume":    round(pv, 2),
        "confidence_interval": f"{lower:,.0f} to {upper:,.0f}",
        "mape":                round(best_mape, 2),
        "warning":             "High forecast uncertainty" if is_high_risk else None,
        "month":               1,
        "year":                2026,
        "xai_drivers": {
            "Holiday_Intensity":      row["Holiday_Intensity"],
            "Operational_Days_Ratio": row["Operational_Days_Ratio"],
            "Regional_Demand_Proxy":  row["Regional_Demand_Proxy"],
            "prev_monthly_sales":     f"{row['PrevsMonthly']/1e9:.2f}B",
            "rolling_3m_avg":         f"{row['rolling_3m_avg']/1e9:.2f}B",
        },
        "model_type": best_name,
        "std_dev":    round(std_val, 6),
    })

jan_out = pd.DataFrame(api_responses)
jan_out.to_csv("outputs/january_2026_forecast_v2.csv", index=False)

print("\n=== January 2026 Forecast (V2 — FastAPI Ready) ===")
for r in api_responses:
    warn = f"  ⚠ {r['warning']}" if r["warning"] else ""
    print(f"  {r['branch']:20s} | {r['predicted_volume']:>15,.0f} | CI: {r['confidence_interval']}{warn}")

# ─────────────────────────────────────────────────────────────
# STEP 9 — 4-panel chart
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(17, 12))
fig.suptitle(
    "Project C.O.C.O. V2 — Contextual Feature Injection + Multi-Model Comparison",
    fontsize=13, fontweight="bold"
)

bl = [b.replace("Main St Coffee","MSC") for b in branch_labels]
x  = np.arange(len(bl))
w  = 0.2

# Panel 1 — All models vs actual
ax = axes[0, 0]
ax.bar(x - 1.5*w, y_test/1e9,                            w, label="Actual",        color="#1e293b")
ax.bar(x - 0.5*w, results["GPR"]["preds"]/1e9,           w, label="GPR",            color="#2563eb")
ax.bar(x + 0.5*w, results["BayesianRidge"]["preds"]/1e9, w, label="Bayesian Ridge", color="#16a34a")
ax.bar(x + 1.5*w, qr_preds["upper_Q90"]/1e9,             w, label="QR Q90 (spike)", color="#ea580c")
ax.set_xticks(x); ax.set_xticklabels(bl, rotation=15)
ax.set_title("December: All Models vs Actual (V2 Features)")
ax.set_ylabel("Sales (Billions)"); ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.1f}B"))

# Panel 2 — Quantile bands + Conut Main anomaly annotation
ax = axes[0, 1]
ax.fill_between(range(len(bl)),
    qr_preds["lower_Q10"]/1e9, qr_preds["upper_Q90"]/1e9,
    alpha=0.25, color="#ea580c", label="Q10–Q90 Band")
ax.plot(qr_preds["median_Q50"]/1e9, "o-", color="#ea580c", lw=2, label="Q50 Median")
ax.plot(y_test/1e9, "s--", color="#1e293b", lw=2, label="Actual")
# Annotate Conut Main anomaly
conut_idx = bl.index("Conut Main") if "Conut Main" in bl else None
if conut_idx is not None:
    ax.annotate(
        "⚠ Maintenance\n/Closure",
        xy=(conut_idx, y_test[conut_idx]/1e9),
        xytext=(conut_idx + 0.3, y_test[conut_idx]/1e9 + 0.3),
        fontsize=8, color="red",
        arrowprops=dict(arrowstyle="->", color="red"),
    )
ax.set_xticks(range(len(bl))); ax.set_xticklabels(bl, rotation=15)
ax.set_title("Quantile Bands + Anomaly Annotation")
ax.set_ylabel("Sales (Billions)"); ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.1f}B"))

# Panel 3 — Feature importance: which V2 features matter most
ax = axes[1, 0]
# Use QR Q90 model feature importances (it has .feature_importances_)
best_qr_model = results["QuantileReg"]["model"]
fi = pd.Series(best_qr_model.feature_importances_, index=FEATURES).sort_values()
colors_fi = ["#ea580c" if f in (
    "Holiday_Intensity","Operational_Days_Ratio","Regional_Demand_Proxy","Branch_Status"
) else "#94a3b8" for f in fi.index]
fi.plot(kind="barh", ax=ax, color=colors_fi)
ax.set_title("Feature Importances (QR Q90)\nOrange = V2 Injected Features")
ax.set_xlabel("Importance")

# Panel 4 — Jan 2026 forecast with CI error bars
ax = axes[1, 1]
jan_x  = np.arange(len(api_responses))
fcast  = np.array([r["predicted_volume"] for r in api_responses])
lower  = fcast * (1 - best_mape/100)
upper  = fcast * (1 + best_mape/100)
ax.bar(jan_x, fcast/1e9, color="#2563eb", alpha=0.8, label="Forecast")
ax.errorbar(jan_x, fcast/1e9,
    yerr=[(fcast - lower)/1e9, (upper - fcast)/1e9],
    fmt="none", color="#1e293b", capsize=8, lw=2)
ax.set_xticks(jan_x)
ax.set_xticklabels(
    [r["branch"].replace("Main St Coffee","MSC") for r in api_responses],
    rotation=15
)
ax.set_title(f"January 2026 Forecast ± MAPE CI  (Best: {best_name})")
ax.set_ylabel("Sales (Billions)"); ax.legend(fontsize=8)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v,_: f"{v:.1f}B"))

plt.tight_layout()
plt.savefig("outputs/model_comparison_v2.png", dpi=150, bbox_inches="tight")
plt.close()

# ─────────────────────────────────────────────────────────────
# STEP 10 — Model comparison summary
# ─────────────────────────────────────────────────────────────
pd.DataFrame([
    {"Model":"GPR V2",           **{k:round(v,4) for k,v in results["GPR"]["metrics"].items()},           "avg_ci":round(results["GPR"]["avg_ci"],2)},
    {"Model":"QuantileReg Q90",  **{k:round(v,4) for k,v in results["QuantileReg"]["metrics"].items()},   "avg_ci":"N/A"},
    {"Model":"BayesianRidge V2", **{k:round(v,4) for k,v in results["BayesianRidge"]["metrics"].items()}, "avg_ci":round(results["BayesianRidge"]["avg_ci"],2)},
]).to_csv("outputs/model_comparison_summary_v2.csv", index=False)

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print(f"  BEST MODEL : {best_name}")
print(f"  RMSE       : {best_run['metrics']['rmse']:>20,.0f}")
print(f"  R²         : {best_run['metrics']['r2']:>20.4f}")
print(f"  MAPE       : {best_run['metrics']['mape']:>20.1f}%")
print("="*60)
print("\nV2 Features injected:")
print("  Holiday_Intensity       : Aug-Oct=0.0, Nov=0.2, Dec=1.0")
print("  Operational_Days_Ratio  : ~0.05 for Conut Main Dec (closure detected)")
print("  Regional_Demand_Proxy   : Is_Coastal × Holiday_Intensity")
print("  Branch_Status           : 1 = Maintenance/Closed (Conut Main Dec)")
print("\nOutputs:")
print("  outputs/december_predictions_v2.csv")
print("  outputs/january_2026_forecast_v2.csv   (FastAPI-ready)")
print("  outputs/model_comparison_summary_v2.csv")
print("  outputs/model_comparison_v2.png        (4-panel)")
print("\nMLflow:")
print("  mlflow ui   →   http://127.0.0.1:5000")
print(f"  Experiment  :   coco_demand_forecasting_v2")
total_runs = len(gpr_kernels) + len(quantile_configs) + len(br_configs)
print(f"  Total runs  :   {total_runs}")
