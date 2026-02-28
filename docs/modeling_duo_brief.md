# Project C.O.C.O. — Modeling Duo: Algorithm Redesign Brief
## Objective 2 — Demand Forecasting & Objective 4 — Staffing Estimation

**Prepared by:** Systems & Analytics Lead  
**Date:** 2026-02-28  
**Team:** Modeling Duo (Maram & Reem)

---

## 1. The Problem with XGBoost on This Dataset

The MLFlow run confirmed what the data already implied — XGBoost failed due to structural mismatch between the method and the data:

| Issue | Why it Breaks XGBoost |
|---|---|
| **Sparse temporal coverage** | Only ~6–12 monthly observations per branch. XGBoost requires hundreds of rows for reliable gradient boosting. |
| **No temporal autocorrelation** | Each branch has too few timestamps for lag features to carry statistical meaning. |
| **No seasonality signal** | With under 12 months of data, there is no repeating seasonal cycle for a boosted tree to learn. |
| **High feature-to-row ratio** | If you one-hot encoded branches + months, the dimensionality vastly exceeds the row count (overfitting trap). |

**Verdict:** XGBoost is the wrong tool for Few-Shot Temporal Inference. Stop treating this as an autoregressive sequence problem.

---

## 2. Recommended Alternative Approach: Few-Shot Probabilistic Inference

When temporal data is this sparse, the correct paradigm shift is:

> **Stop extrapolating. Start interpolating with uncertainty.**

Instead of asking "What will sales be next month?", the model should ask: "Given everything I know about branches with similar profiles, what is the probability distribution of sales for this branch next month?"

---

## 3. Recommended Model Stack

### 3A. Gaussian Process Regression (GPR) — **Primary Recommendation**

**Why it's ideal:**  
GPR is specifically designed for small datasets with high uncertainty. It outputs a **full probability distribution**, not just a single number, making confidence intervals mathematically native rather than hacked on.

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

# Kernel: seasonal smooth variation + noise
kernel = ConstantKernel(1.0) * RBF(length_scale=2.0) + WhiteKernel(noise_level=0.1)

gpr = GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    normalize_y=True
)

gpr.fit(X_train, y_train)

# Returns mean + std dev for confidence intervals natively
y_pred, y_std = gpr.predict(X_test, return_std=True)
```

**Feature Engineering for GPR:**
- `month_num` (integer 1–12)
- `branch_encoded` (label encode by revenue tier)
- `rolling_3m_avg` (imputed with branch mean if missing)
- `is_weekend_heavy` (binary flag from attendance data)

---

### 3B. Bayesian Ridge Regression — **Secondary Recommendation**

**Why it works here:**  
Like GPR, Bayesian Ridge natively produces posterior uncertainty estimates. It's computationally cheaper and works well when variance is mostly linear.

```python
from sklearn.linear_model import BayesianRidge

model = BayesianRidge(
    alpha_1=1e-6, alpha_2=1e-6,
    lambda_1=1e-6, lambda_2=1e-6,
    compute_score=True,
)
model.fit(X_train, y_train)

y_pred, y_std = model.predict(X_test, return_std=True)
```

---

### 3C. Quantile Regression — **Fallback / Ensemble Companion**

**Why it works here:**  
Quantile Regression directly estimates the 10th and 90th percentiles of the output distribution — ideal for generating conservative-vs-aggressive demand bounds without distributional assumptions.

```python
from sklearn.ensemble import GradientBoostingRegressor

lower_model = GradientBoostingRegressor(loss='quantile', alpha=0.1)
upper_model = GradientBoostingRegressor(loss='quantile', alpha=0.9)
median_model = GradientBoostingRegressor(loss='quantile', alpha=0.5)
```

---

## 4. MLFlow Tracking Protocol

Every experiment run **must** be tracked with the following MLFlow schema:

```python
import mlflow

mlflow.set_experiment("coco_demand_forecasting")

with mlflow.start_run(run_name="GPR_RBF_Kernel"):
    mlflow.log_param("model_type", "GPR")
    mlflow.log_param("kernel", "ConstantKernel * RBF + WhiteKernel")
    mlflow.log_param("features", str(feature_names))
    
    # Metrics
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mape", mape)
    mlflow.log_metric("r2", r2_score)
    mlflow.log_metric("avg_ci_width", np.mean(y_std * 2))  # Avg confidence band width
    
    # Register the model
    mlflow.sklearn.log_model(
        sk_model=gpr,
        artifact_path="gpr_model",
        registered_model_name="COCO_DemandForecasting_GPR",
    )
```

### Required Metrics Per Run

| Metric | Description | Target |
|---|---|---|
| `rmse` | Root Mean Squared Error | Minimize |
| `mape` | Mean Absolute Percentage Error | < 15% acceptable |
| `r2` | R-squared | > 0.70 acceptable |
| `avg_ci_width` | Average confidence interval width | Minimize |

---

## 5. FastAPI Contract (Unchanged)

The API stub in `app/main.py` is already waiting. Your final model output must conform to this schema in `app/schemas.py`:

```python
class DemandResponse(BaseModel):
    branch: str
    predicted_volume: float
    # NEW: Confidence interval support
    confidence_interval: Optional[str] = None  # e.g. "1,100 to 1,400"
    mape: Optional[float] = None               # From MLflow best run
    warning: Optional[str] = None
    month: int
    year: int
    xai_drivers: dict
    model_type: str
```

The FastAPI endpoint will automatically compute the confidence interval from the MAPE you log in MLFlow:

```python
error_margin = predicted_volume * (best_model_mape / 100)
lower_bound = predicted_volume - error_margin
upper_bound = predicted_volume + error_margin
```

---

## 6. Implementation Checklist

- [ ] Set up MLFlow tracking server (`mlflow ui` in project root)
- [ ] Build feature matrix: `month_num`, `branch_encoded`, `rolling_3m_avg`
- [ ] Run Experiment 1: GPR with RBF kernel — log all metrics
- [ ] Run Experiment 2: Bayesian Ridge — log all metrics
- [ ] Run Experiment 3: Quantile Regression (lower/upper/median) — log all metrics
- [ ] Use MLFlow Model Registry to promote the best model to `Production` stage
- [ ] Replace the stub in `app/main.py → predict_demand()` with the production model
- [ ] Repeat the same MLFlow workflow for Objective 4 (Staffing) using `labor_hours.parquet`

---

## 7. Data Files Available

| File | Description | Use For |
|---|---|---|
| `cleaned/monthly_sales.parquet` | Monthly revenue per branch | Demand forecasting feature matrix |
| `cleaned/sales_by_item.parquet` | Item-level sales per branch | Product mix features |
| `cleaned/labor_hours.parquet` | Shifts per employee per day | Staffing estimation |

---

## 8. Key Design Principle

> **Probabilistic outputs are not a weakness — they are a strength.**
>
> When OpenClaw tells a Chief of Operations "Sales will be 1.25M ± 15%", that is more valuable than a false point estimate. Every metric logged to MLFlow becomes part of the Executive Brief.
