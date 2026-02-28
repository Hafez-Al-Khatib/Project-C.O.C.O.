"""
Project C.O.C.O. — PyTest Configuration and Shared Fixtures
===========================================================
"""

import os
import sys
import json
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))


@pytest.fixture
def project_root():
    """Return project root directory."""
    return BASE_DIR


@pytest.fixture
def cleaned_dir(project_root):
    """Return cleaned data directory path."""
    return project_root / "cleaned"


@pytest.fixture
def models_dir(project_root):
    """Return models directory path."""
    return project_root / "models"


@pytest.fixture
def sample_transactions_df():
    """Create sample transactions DataFrame for testing."""
    return pd.DataFrame({
        "receipt_id": [1, 1, 2, 2, 3],
        "branch": ["Conut Tyre", "Conut Tyre", "Conut Jnah", "Conut Jnah", "Main St Coffee"],
        "customer": ["Person_001", "Person_001", "Person_002", "Person_002", "Person_003"],
        "qty": [2, 1, 3, 2, 1],
        "item": ["CAFFE LATTE", "CROISSANT", "CAFFE LATTE", "CHIMNEY CAKE", "AMERICANO"],
        "price": [45000, 32000, 45000, 28000, 38000]
    })


@pytest.fixture
def sample_sales_by_item_df():
    """Create sample sales by item DataFrame for testing."""
    return pd.DataFrame({
        "branch": ["Conut Tyre", "Conut Tyre", "Conut Jnah", "Main St Coffee"],
        "division": ["Beverages", "Bakery", "Beverages", "Beverages"],
        "group": ["Coffee", "Pastry", "Coffee", "Coffee"],
        "item": ["CAFFE LATTE", "CROISSANT", "CAFFE LATTE", "AMERICANO"],
        "qty": [150, 80, 200, 120],
        "total_amount": [6750000, 2560000, 9000000, 4560000]
    })


@pytest.fixture
def sample_monthly_sales_df():
    """Create sample monthly sales DataFrame for testing."""
    return pd.DataFrame({
        "branch": ["Conut Tyre", "Conut Tyre", "Conut Tyre", "Conut Jnah", "Conut Jnah"],
        "month": [8, 9, 10, 8, 9],
        "month_name": ["August", "September", "October", "August", "September"],
        "year": [2025, 2025, 2025, 2025, 2025],
        "total_sales": [150000000, 180000000, 220000000, 200000000, 240000000]
    })


@pytest.fixture
def sample_labor_hours_df():
    """Create sample labor hours DataFrame for testing."""
    return pd.DataFrame({
        "employee_id": ["1.0", "1.0", "2.0", "2.0"],
        "employee_name": ["Person_001", "Person_001", "Person_002", "Person_002"],
        "branch": ["Conut Tyre", "Conut Tyre", "Conut Jnah", "Conut Jnah"],
        "date": pd.to_datetime(["2025-12-01", "2025-12-02", "2025-12-01", "2025-12-02"]),
        "work_hours": [8.0, 7.5, 8.5, 8.0]
    })


@pytest.fixture
def sample_avg_sales_menu_df():
    """Create sample avg sales menu DataFrame for testing."""
    return pd.DataFrame({
        "item": ["CAFFE LATTE", "CROISSANT", "CHIMNEY CAKE", "AMERICANO"],
        "avg_price": [45000, 32000, 28000, 38000],
        "total_quantity": [500, 300, 250, 400]
    })


@pytest.fixture
def sample_demand_forecast_features():
    """Create sample features for demand forecasting."""
    return {
        "branch_encoded": 0,
        "Month_Num": 11,
        "Holiday_Intensity": 0.0,
        "Operational_Days_Ratio": 1.0,
        "Regional_Demand_Proxy": 0.0,
        "Branch_Status": 0,
        "DaysinMonth": 30,
        "WeekendDays": 8,
        "HolidayMonth": 0,
        "WeekendRatio": 0.267,
        "is_weekend_heavy": 0,
        "Months Active": 6,
        "Coldstart": 0,
        "Is_Coastal": 1,
        "log_prev": np.log1p(180000000),
        "relative_size": 1.0
    }


@pytest.fixture
def mock_mlflow(tmp_path):
    """Create temporary MLflow tracking directory."""
    mlruns_dir = tmp_path / "mlruns"
    mlruns_dir.mkdir(exist_ok=True)
    os.environ["MLFLOW_TRACKING_URI"] = f"file:///{mlruns_dir}"
    return mlruns_dir


@pytest.fixture
def temp_parquet_dir(tmp_path):
    """Create temporary directory for parquet files."""
    parquet_dir = tmp_path / "cleaned"
    parquet_dir.mkdir(exist_ok=True)
    return parquet_dir


@pytest.fixture(autouse=True)
def clean_mlflow_runs():
    """Clean up MLflow runs after each test."""
    yield
    # Cleanup logic if needed
    import mlflow
    mlflow.end_run()
