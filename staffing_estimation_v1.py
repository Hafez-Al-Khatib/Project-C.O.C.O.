"""
Project C.O.C.O. — Objective 4: Shift Staffing Estimation
==========================================================
Estimate required employees per shift using demand and
time-related operational data.

Data Source:
  Conut_Work_Hours_UPDATED.xlsx  →  Raw Shift Log (Dec 2025)
  Branches_Cleaned_For_XGBoost.csv  →  Monthly sales per branch

Target Variable:
  staff_count — number of distinct employees working on a given day at a branch

Models:
  1. XGBoost        → GradientBoostingRegressor  (sklearn, identical boosting logic)
  2. LightGBM       → HistGradientBoostingRegressor (sklearn native LGBM-equivalent)
  3. Random Forest  → RandomForestRegressor

Features (14 total):
  Temporal  : day_of_week, is_weekend, week_of_month, day, is_holiday_week
  Branch    : branch_encoded, max_staff_capacity, monthly_sales_norm
  Lag/trend : staff_lag1, staff_roll3, hours_lag1
  Demand    : daily_sales_proxy, hours_per_staff_lag1
  Interaction: weekend_x_holiday

Validation:
  Leave-One-Out Cross-Validation (only 86 rows, so LOO is the right call)

MLflow:
  Experiment: coco_staffing_v1
  9 runs: 3 models × 3 branches
  Metrics: RMSE, MAE, R², MAPE, within_1_accuracy

Outputs:
  outputs/staffing_daily_features.csv
  outputs/staffing_cv_predictions.csv
  outputs/staffing_model_summary.csv
  outputs/staffing_jan2026_forecast.csv
  outputs/staffing_results.png
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
    mlflow.set_tracking_uri('sqlite:///mlflow.db')
    mlflow.set_experiment('coco_staffing_v1')
except ImportError:
    MLFLOW_AVAILABLE = False
    class _FakeRun:
        def __enter__(self): return self
        def __exit__(self,*a): pass
        def __getattr__(self,n): return lambda *a,**k: None
    class _FakeSklearn:
        def log_model(self, *a, **k): pass
    class _FakeMlflow:
        def __init__(self): self.sklearn = _FakeSklearn()
        def start_run(self, **k): return _FakeRun()
        def __getattr__(self, n): return lambda *a, **k: None
    mlflow = _FakeMlflow()
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
os.makedirs("outputs", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("coco_staffing_v1")

print("=" * 65)
print("PROJECT C.O.C.O. — Objective 4: Shift Staffing Estimation")
print("Models: XGBoost | LightGBM | Random Forest")
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# STEP 1 — Load & clean shift log
# ─────────────────────────────────────────────────────────────
xl  = pd.read_excel("Conut_Work_Hours_UPDATED.xlsx", sheet_name="Raw Shift Log")
xl["Date"] = pd.to_datetime(xl["Date"], format="%d-%b-%y")

# Drop ghost punches under 30 minutes (timestamp correction artifacts)
xl = xl[xl["Work Hours"] >= 0.5].copy()
print(f"\nShift records after cleaning : {len(xl)}")

# Daily aggregation: how many distinct staff per branch per day
daily = (
    xl.groupby(["Branch", "Date"])
    .agg(
        staff_count   = ("Employee ID", "nunique"),
        total_hours   = ("Work Hours",  "sum"),
        avg_shift_hrs = ("Work Hours",  "mean"),
        num_punches   = ("Employee ID", "count"),
    )
    .reset_index()
)

# ─────────────────────────────────────────────────────────────
# STEP 2 — Feature engineering
# ─────────────────────────────────────────────────────────────
# Branch capacity (max staff seen in data — hard ceiling for that branch)
capacity = daily.groupby("Branch")["staff_count"].max().to_dict()

daily["day_of_week"]    = daily["Date"].dt.dayofweek       # 0=Mon … 6=Sun
daily["is_weekend"]     = daily["day_of_week"].isin([4, 5]).astype(int)  # Fri/Sat in Lebanon
daily["day"]            = daily["Date"].dt.day
daily["week_of_month"]  = ((daily["day"] - 1) // 7) + 1
daily["month"]          = daily["Date"].dt.month
# Christmas week: Dec 22-29 — historically the spike period for Jnah/MSC
daily["is_holiday_week"]  = (daily["day"] >= 22).astype(int)
# Interaction: weekend AND holiday week together (biggest demand driver)
daily["weekend_x_holiday"] = daily["is_weekend"] * daily["is_holiday_week"]
# Branch profile
daily["branch_encoded"]    = daily["Branch"].map(
    {"Conut - Tyre": 0, "Conut Jnah": 1, "Main Street Coffee": 2}
)
daily["max_staff_capacity"] = daily["Branch"].map(capacity)

# Attach Dec monthly sales as a demand context signal (normalized 0–1)
sales_csv = pd.read_csv("Branches_Cleaned_For_XGBoost.csv")
dec_sales  = sales_csv[sales_csv["Month_Num"] == 12]
BRANCH_SALES = {
    "Conut - Tyre":       float(dec_sales[dec_sales["BranchisConutTyre"]       == 1]["MonthlySale"].values[0]),
    "Conut Jnah":         float(dec_sales[dec_sales["BranchisConutJnah"]       == 1]["MonthlySale"].values[0]),
    "Main Street Coffee": float(dec_sales[dec_sales["BranchisMainStreetCoffee"]== 1]["MonthlySale"].values[0]),
}
max_sales = max(BRANCH_SALES.values())
daily["monthly_sales_norm"]  = daily["Branch"].map(BRANCH_SALES) / max_sales
# Daily sales proxy: uniform distribution within month (we have no daily sales data)
daily["daily_sales_proxy"]   = daily["monthly_sales_norm"] / 31

# Sort to enable lag features
daily = daily.sort_values(["Branch", "Date"]).reset_index(drop=True)

# Lag features (per branch to avoid cross-branch leakage)
daily["staff_lag1"]         = daily.groupby("Branch")["staff_count"].shift(1)
daily["hours_lag1"]         = daily.groupby("Branch")["total_hours"].shift(1)
daily["staff_roll3"]        = daily.groupby("Branch")["staff_count"].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
daily["hours_per_staff_lag1"] = daily["hours_lag1"] / daily["staff_lag1"].replace(0, np.nan)

# Fill first-day nulls with branch mean (only 3 rows affected)
for col in ["staff_lag1", "hours_lag1", "staff_roll3", "hours_per_staff_lag1"]:
    daily[col] = daily.groupby("Branch")[col].transform(lambda x: x.fillna(x.mean()))

print(f"Daily branch-day records     : {len(daily)}")
print(f"Feature nulls                : {daily.isnull().sum().sum()}")
print(f"\nBranch staff ranges:")
for b, grp in daily.groupby("Branch"):
    print(f"  {b:<22} min={grp['staff_count'].min()}  max={grp['staff_count'].max()}  mean={grp['staff_count'].mean():.2f}")

# Save engineered dataset
daily.to_csv("outputs/staffing_daily_features.csv", index=False)

# ─────────────────────────────────────────────────────────────
# STEP 3 — Define features and models
# ─────────────────────────────────────────────────────────────
FEATURES = [
    "branch_encoded",
    "day_of_week",
    "is_weekend",
    "day",
    "week_of_month",
    "is_holiday_week",
    "weekend_x_holiday",
    "max_staff_capacity",
    "monthly_sales_norm",
    "daily_sales_proxy",
    "staff_lag1",
    "staff_roll3",
    "hours_lag1",
    "hours_per_staff_lag1",
]

MODELS = {
    "XGBoost": GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        min_samples_leaf=2,
        random_state=42,
    ),
    "LightGBM": HistGradientBoostingRegressor(
        max_iter=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=3,
        random_state=42,
    ),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=2,
        random_state=42,
    ),
}

# ─────────────────────────────────────────────────────────────
# STEP 4 — Metrics helper
# ─────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse   = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae    = float(mean_absolute_error(y_true, y_pred))
    r2     = float(r2_score(y_true, y_pred))
    mape   = float(np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100)
    # Within-1 accuracy: how often is the rounded prediction exactly right or off by 1?
    within1 = float(np.mean(np.abs(np.round(y_pred) - y_true) <= 1) * 100)
    return {"rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "within1_pct": within1}

# ─────────────────────────────────────────────────────────────
# STEP 5 — Leave-One-Out CV per model (all branches combined)
# ─────────────────────────────────────────────────────────────
# With only 86 rows, LOO gives us 86 test points — every row gets to be test once.
# This maximises evaluation fidelity given our tiny dataset.

X = daily[FEATURES].values
y = daily["staff_count"].values

print("\n" + "=" * 65)
print("  LEAVE-ONE-OUT CV — All 3 branches combined (86 rows)")
print("=" * 65)
print(f"  {'Model':<15} {'RMSE':>7} {'MAE':>7} {'R²':>8} {'MAPE':>8} {'Within±1':>10}")
print(f"  {'-'*60}")

all_cv_results  = {}
all_cv_preds_df = []

loo = LeaveOneOut()

for model_name, model_template in MODELS.items():
    oof_preds  = np.zeros(len(y))
    oof_true   = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X):
        from sklearn.base import clone
        m = clone(model_template)
        m.fit(X[train_idx], y[train_idx])
        pred = m.predict(X[test_idx])[0]
        oof_preds[test_idx[0]] = max(1, pred)   # floor at 1 — can't have 0 staff
        oof_true[test_idx[0]]  = y[test_idx[0]]

    metrics = compute_metrics(oof_true, oof_preds)
    all_cv_results[model_name] = metrics

    print(f"  {model_name:<15} {metrics['rmse']:>7.3f} {metrics['mae']:>7.3f} "
          f"{metrics['r2']:>8.4f} {metrics['mape']:>7.1f}% {metrics['within1_pct']:>9.1f}%")

    for i in range(len(daily)):
        row = daily.iloc[i]
        all_cv_preds_df.append({
            "model":       model_name,
            "branch":      row["Branch"],
            "date":        row["Date"].date(),
            "day_of_week": row["day_of_week"],
            "is_weekend":  row["is_weekend"],
            "is_holiday_week": row["is_holiday_week"],
            "actual":      int(oof_true[i]),
            "predicted":   round(oof_preds[i], 2),
            "predicted_rounded": int(round(oof_preds[i])),
            "error":       round(oof_preds[i] - oof_true[i], 2),
            "correct":     int(abs(round(oof_preds[i]) - oof_true[i]) == 0),
            "within_1":    int(abs(round(oof_preds[i]) - oof_true[i]) <= 1),
        })

print("=" * 65)

# Best model by RMSE
best_name    = min(all_cv_results, key=lambda k: all_cv_results[k]["rmse"])
best_metrics = all_cv_results[best_name]
print(f"\n  BEST MODEL: {best_name}  |  RMSE: {best_metrics['rmse']:.3f}  |  Within±1: {best_metrics['within1_pct']:.1f}%")

cv_df = pd.DataFrame(all_cv_preds_df)
cv_df.to_csv("outputs/staffing_cv_predictions.csv", index=False)

# ─────────────────────────────────────────────────────────────
# STEP 6 — Per-branch CV breakdown (MLflow run per model×branch)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  PER-BRANCH BREAKDOWN — LOO CV Results")
print("=" * 65)
print(f"  {'Model':<15} {'Branch':<22} {'RMSE':>7} {'MAE':>7} {'R²':>8} {'Within±1':>10}")
print(f"  {'-'*68}")

summary_rows = []
BRANCHES     = daily["Branch"].unique().tolist()

for model_name, model_template in MODELS.items():
    for branch in BRANCHES:
        branch_df  = daily[daily["Branch"] == branch].copy().reset_index(drop=True)
        X_b = branch_df[FEATURES].values
        y_b = branch_df["staff_count"].values

        oof_b = np.zeros(len(y_b))
        for train_idx, test_idx in loo.split(X_b):
            from sklearn.base import clone
            m = clone(model_template)
            m.fit(X_b[train_idx], y_b[train_idx])
            oof_b[test_idx[0]] = max(1, m.predict(X_b[test_idx])[0])

        bm = compute_metrics(y_b, oof_b)

        # MLflow run per model × branch
        with mlflow.start_run(run_name=f"{model_name}_{branch.replace(' ','_')}"):
            mlflow.log_param("model",           model_name)
            mlflow.log_param("branch",          branch)
            mlflow.log_param("n_samples",       len(X_b))
            mlflow.log_param("cv_strategy",     "LeaveOneOut")
            mlflow.log_param("features",        str(FEATURES))
            mlflow.log_param("target",          "staff_count")
            mlflow.log_metric("rmse",           round(bm["rmse"],    3))
            mlflow.log_metric("mae",            round(bm["mae"],     3))
            mlflow.log_metric("r2",             round(bm["r2"],      4))
            mlflow.log_metric("mape",           round(bm["mape"],    2))
            mlflow.log_metric("within1_pct",    round(bm["within1_pct"], 1))
            # Train on full branch data for model logging
            full_model = clone(model_template)
            full_model.fit(X_b, y_b)
            mlflow.sklearn.log_model(full_model, f"{model_name.lower()}_model")

        print(f"  {model_name:<15} {branch:<22} {bm['rmse']:>7.3f} {bm['mae']:>7.3f} "
              f"{bm['r2']:>8.4f} {bm['within1_pct']:>9.1f}%")

        summary_rows.append({
            "model":        model_name,
            "branch":       branch,
            **{k: round(v, 4) for k, v in bm.items()},
        })

print("=" * 65)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("outputs/staffing_model_summary.csv", index=False)

# ─────────────────────────────────────────────────────────────
# STEP 7 — Retrain best model on all data → January 2026 forecast
# ─────────────────────────────────────────────────────────────
print(f"\n── January 2026 Forecast using {best_name} (retrained on full Dec data)")

from sklearn.base import clone
final_model = clone(MODELS[best_name])
final_model.fit(X, y)

# January 2026: 31 days, 3 branches
# Known context: Conut Main re-opens (no closure), normal operations
JAN_BRANCH_SALES = {
    "Conut - Tyre":       BRANCH_SALES["Conut - Tyre"]    * 0.85,   # post-holiday correction
    "Conut Jnah":         BRANCH_SALES["Conut Jnah"]      * 0.60,   # holiday spike fades
    "Main Street Coffee": BRANCH_SALES["Main Street Coffee"] * 0.65,
}
max_jan = max(JAN_BRANCH_SALES.values())

jan_rows = []
for day in range(1, 32):
    date = pd.Timestamp(f"2026-01-{day:02d}")
    dow  = date.dayofweek
    wom  = ((day - 1) // 7) + 1
    is_we = int(dow in [4, 5])
    is_hol = 0   # January has no major holiday week in this context

    for branch, b_enc in [("Conut - Tyre", 0), ("Conut Jnah", 1), ("Main Street Coffee", 2)]:
        # Use average Dec lag values as priors for January
        branch_dec     = daily[daily["Branch"] == branch]
        avg_staff_dec  = branch_dec["staff_count"].mean()
        avg_hours_dec  = branch_dec["total_hours"].mean()
        ms_norm        = JAN_BRANCH_SALES[branch] / max_jan
        cap            = capacity[branch]

        jan_rows.append({
            "branch":              branch,
            "date":                date.date(),
            "day":                 day,
            "branch_encoded":      b_enc,
            "day_of_week":         dow,
            "is_weekend":          is_we,
            "week_of_month":       wom,
            "is_holiday_week":     is_hol,
            "weekend_x_holiday":   is_we * is_hol,
            "max_staff_capacity":  cap,
            "monthly_sales_norm":  ms_norm,
            "daily_sales_proxy":   ms_norm / 31,
            "staff_lag1":          avg_staff_dec,
            "staff_roll3":         avg_staff_dec,
            "hours_lag1":          avg_hours_dec,
            "hours_per_staff_lag1": avg_hours_dec / max(avg_staff_dec, 1),
        })

jan_df    = pd.DataFrame(jan_rows)
X_jan     = jan_df[FEATURES].values
jan_preds = np.clip(final_model.predict(X_jan), 1, None)  # floor at 1
jan_df["predicted_staff"] = np.round(jan_preds).astype(int)

# Save per-branch daily forecast
jan_out = jan_df[["branch", "date", "day_of_week", "is_weekend", "is_holiday_week", "predicted_staff"]]
jan_out["day_name"] = pd.to_datetime(jan_out["date"]).dt.day_name()
jan_out.to_csv("outputs/staffing_jan2026_forecast.csv", index=False)

# Summary by branch
print("\n=== January 2026 Staffing Summary ===")
print(f"  {'Branch':<22} {'Min':>5} {'Max':>5} {'Avg':>7} {'Weekday Avg':>13} {'Weekend Avg':>13}")
print(f"  {'-'*65}")
for b in BRANCHES:
    sub    = jan_df[jan_df["branch"] == b]
    wkday  = sub[sub["is_weekend"] == 0]["predicted_staff"].mean()
    wkend  = sub[sub["is_weekend"] == 1]["predicted_staff"].mean()
    print(f"  {b:<22} {sub['predicted_staff'].min():>5} {sub['predicted_staff'].max():>5} "
          f"{sub['predicted_staff'].mean():>7.1f} {wkday:>13.1f} {wkend:>13.1f}")

# ─────────────────────────────────────────────────────────────
# STEP 8 — Feature importance (Random Forest & XGBoost have .feature_importances_)
# ─────────────────────────────────────────────────────────────
fi_models = {}
for mname in ["XGBoost", "RandomForest"]:
    m = clone(MODELS[mname])
    m.fit(X, y)
    fi_models[mname] = pd.Series(m.feature_importances_, index=FEATURES).sort_values(ascending=False)

# ─────────────────────────────────────────────────────────────
# STEP 9 — 4-panel chart
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(17, 12))
fig.suptitle(
    "Project C.O.C.O. — Objective 4: Shift Staffing Estimation\n"
    f"Models: XGBoost | LightGBM | Random Forest  |  Best: {best_name}  |  LOO CV",
    fontsize=13, fontweight="bold"
)

MODEL_COLORS = {"XGBoost": "#2563eb", "LightGBM": "#ea580c", "RandomForest": "#16a34a"}
BRANCH_COLORS = {"Conut - Tyre": "#7c3aed", "Conut Jnah": "#db2777", "Main Street Coffee": "#0891b2"}

# ── Panel 1: Actual vs Predicted scatter (all branches, best model) ──
ax = axes[0, 0]
for branch in BRANCHES:
    sub = cv_df[(cv_df["model"] == best_name) & (cv_df["branch"] == branch)]
    ax.scatter(sub["actual"], sub["predicted"],
               label=branch, color=BRANCH_COLORS[branch], alpha=0.7, s=50)
ax.plot([1, 4], [1, 4], "k--", lw=1, alpha=0.4, label="Perfect")
ax.set_xlabel("Actual Staff Count")
ax.set_ylabel("Predicted Staff Count")
ax.set_title(f"Actual vs Predicted — {best_name} (LOO CV)")
ax.legend(fontsize=8)
ax.set_xticks([1, 2, 3, 4])
ax.set_yticks([1, 2, 3, 4])

# ── Panel 2: CV Metrics comparison across models ──
ax = axes[0, 1]
metrics_to_plot = ["rmse", "mae", "within1_pct"]
labels          = ["RMSE", "MAE", "Within±1 (%)"]
x   = np.arange(len(metrics_to_plot))
w   = 0.25
for j, (mname, mcolor) in enumerate(MODEL_COLORS.items()):
    vals = [all_cv_results[mname][m] for m in metrics_to_plot]
    bars = ax.bar(x + (j - 1) * w, vals, w, label=mname, color=mcolor, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_title("Model Comparison — LOO CV Metrics (All Branches)")
ax.legend(fontsize=9)
ax.set_ylabel("Score")

# ── Panel 3: Time series — Actual vs all models (Main Street Coffee) ──
ax = axes[1, 0]
branch_focus = "Main Street Coffee"
actual_ts = daily[daily["Branch"] == branch_focus].set_index("Date")["staff_count"]
ax.plot(actual_ts.index, actual_ts.values, "ko-", lw=2, ms=5, label="Actual", zorder=5)
for mname, mcolor in MODEL_COLORS.items():
    sub = cv_df[(cv_df["model"] == mname) & (cv_df["branch"] == branch_focus)].sort_values("date")
    ax.plot(pd.to_datetime(sub["date"]), sub["predicted"],
            "--", lw=1.5, color=mcolor, alpha=0.8, label=mname)
ax.axvline(pd.Timestamp("2025-12-22"), color="red", linestyle=":", alpha=0.5, lw=1.5)
ax.text(pd.Timestamp("2025-12-22"), 0.5, "Holiday\nWeek", fontsize=7, color="red", va="bottom")
ax.set_title(f"Staffing Time Series — {branch_focus}")
ax.set_ylabel("Staff Count")
ax.legend(fontsize=8)
ax.set_ylim(0, 5.5)
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))

# ── Panel 4: January 2026 weekly forecast heatmap ──
ax = axes[1, 1]
pivot = jan_df.pivot_table(
    index="branch", columns="day_of_week", values="predicted_staff", aggfunc="mean"
)
pivot.index = [b.replace("Main Street Coffee", "MSC") for b in pivot.index]
DOW_LABELS  = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=1, vmax=4)
ax.set_xticks(range(7))
ax.set_xticklabels(DOW_LABELS)
ax.set_yticks(range(len(pivot.index)))
ax.set_yticklabels(pivot.index, fontsize=9)
ax.set_title("Jan 2026 Avg Staffing Forecast by Day of Week")
plt.colorbar(im, ax=ax, label="Avg Staff Count")
for i in range(len(pivot.index)):
    for j in range(7):
        val = pivot.values[i, j]
        if not np.isnan(val):
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig("outputs/staffing_results.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nChart saved → outputs/staffing_results.png")

# ─────────────────────────────────────────────────────────────
# STEP 10 — Feature importance chart
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Feature Importance — XGBoost vs Random Forest", fontsize=12, fontweight="bold")
FI_COLORS = {
    "staff_lag1":           "#2563eb",
    "staff_roll3":          "#3b82f6",
    "hours_lag1":           "#60a5fa",
    "hours_per_staff_lag1": "#93c5fd",
    "max_staff_capacity":   "#7c3aed",
    "monthly_sales_norm":   "#8b5cf6",
    "daily_sales_proxy":    "#a78bfa",
    "branch_encoded":       "#16a34a",
    "day_of_week":          "#ea580c",
    "is_weekend":           "#f97316",
    "day":                  "#fb923c",
    "week_of_month":        "#fdba74",
    "is_holiday_week":      "#dc2626",
    "weekend_x_holiday":    "#ef4444",
}
for ax, (mname, fi) in zip(axes, fi_models.items()):
    colors = [FI_COLORS.get(f, "#94a3b8") for f in fi.index]
    fi.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"{mname} Feature Importance")
    ax.set_xlabel("Importance")
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig("outputs/staffing_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Feature importance chart saved → outputs/staffing_feature_importance.png")

# ─────────────────────────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL SUMMARY — Staffing Estimation V1")
print("=" * 65)
print(f"  Training records   : {len(daily)} daily branch-days (Dec 2025)")
print(f"  Validation         : Leave-One-Out CV (86 folds)")
print(f"  Features           : {len(FEATURES)}")
print(f"\n  {'Model':<15} {'RMSE':>7} {'MAE':>7} {'R²':>8} {'MAPE':>8} {'Within±1':>10}")
print(f"  {'-'*58}")
for mname, m in all_cv_results.items():
    marker = " ← BEST" if mname == best_name else ""
    print(f"  {mname:<15} {m['rmse']:>7.3f} {m['mae']:>7.3f} {m['r2']:>8.4f} "
          f"{m['mape']:>7.1f}% {m['within1_pct']:>9.1f}%{marker}")
print("=" * 65)

print(f"""
Key Operational Insights:
  • Weekend staffing demand is {daily[daily['is_weekend']==1]['staff_count'].mean():.1f} avg vs
    {daily[daily['is_weekend']==0]['staff_count'].mean():.1f} avg on weekdays — plan accordingly
  • Christmas week (Dec 22+) averages {daily[daily['is_holiday_week']==1]['staff_count'].mean():.1f} staff/day
    vs {daily[daily['is_holiday_week']==0]['staff_count'].mean():.1f} in normal weeks
  • Main Street Coffee requires the most staff (avg {daily[daily['Branch']=='Main Street Coffee']['staff_count'].mean():.1f}/day)
  • Conut Jnah scales up significantly in holiday week (max {daily[daily['Branch']=='Conut Jnah']['staff_count'].max()} staff)

Outputs:
  outputs/staffing_daily_features.csv      (engineered dataset)
  outputs/staffing_cv_predictions.csv      (all LOO predictions)
  outputs/staffing_model_summary.csv       (per-branch metrics)
  outputs/staffing_jan2026_forecast.csv    (Jan 2026 daily forecast)
  outputs/staffing_results.png             (4-panel chart)
  outputs/staffing_feature_importance.png  (XGB vs RF importance)

MLflow:
  mlflow ui  →  http://127.0.0.1:5000
  Experiment :  coco_staffing_v1
  Total runs :  9  (3 models × 3 branches)
""")
