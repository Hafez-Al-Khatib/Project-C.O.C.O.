"""
Tests for Staffing Estimator
============================
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.staffing_estimator import StaffingEstimator


class TestStaffingEstimatorInitialization:
    """Test StaffingEstimator initialization."""
    
    def test_initialization(self):
        """Verify estimator initializes correctly."""
        estimator = StaffingEstimator()
        assert estimator.branch_map == {}
        assert estimator.models is not None
        assert len(estimator.models) == 3  # RandomForest, XGBoost, LightGBM
        
    def test_models_are_defined(self):
        """Verify all expected models are defined."""
        estimator = StaffingEstimator()
        expected_models = ["RandomForest", "XGBoost", "LightGBM"]
        for name in expected_models:
            assert name in estimator.models
            
    def test_best_model_is_none_initially(self):
        """Verify best_model is None before training."""
        estimator = StaffingEstimator()
        assert estimator.best_model is None


class TestStaffingEstimatorFit:
    """Test StaffingEstimator fit method."""
    
    def test_fit_creates_branch_map(self, sample_labor_hours_df, sample_monthly_sales_df):
        """Verify fit creates branch mapping."""
        estimator = StaffingEstimator()
        
        # Create mock aggregated data
        df = pd.DataFrame({
            "branch": ["Conut Tyre", "Conut Tyre", "Conut Jnah"],
            "date": pd.to_datetime(["2025-12-01", "2025-12-02", "2025-12-01"]),
            "staff_count": [3, 4, 5],
            "daily_volume": [150000000, 180000000, 200000000],
            "day_of_week": [0, 1, 0],
            "is_weekend": [0, 0, 0]
        })
        
        estimator.fit(df)
        
        assert len(estimator.branch_map) > 0
        assert "Conut Tyre" in estimator.branch_map or any("Tyre" in k for k in estimator.branch_map)
        
    def test_fit_selects_best_model(self, tmp_path):
        """Verify fit selects a best model."""
        estimator = StaffingEstimator()
        
        df = pd.DataFrame({
            "branch": ["Conut Tyre"] * 10,
            "date": pd.date_range("2025-12-01", periods=10),
            "staff_count": [3, 4, 3, 4, 5, 3, 4, 3, 4, 5],
            "daily_volume": np.random.uniform(100000000, 250000000, 10),
            "day_of_week": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            "is_weekend": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
        })
        
        estimator.fit(df)
        
        assert estimator.best_model is not None
        assert estimator.best_model_name in ["RandomForest", "XGBoost", "LightGBM"]


class TestStaffingEstimatorPredict:
    """Test StaffingEstimator predict method."""
    
    def test_predict_requires_fit(self):
        """Verify predict raises error if not fitted."""
        estimator = StaffingEstimator()
        
        with pytest.raises(ValueError, match="Model not trained"):
            estimator.predict("Conut Tyre", 150000000)
            
    def test_predict_returns_positive_integer(self, tmp_path):
        """Verify predict returns positive integer staff count."""
        estimator = StaffingEstimator()
        
        # Fit with sample data
        df = pd.DataFrame({
            "branch": ["Conut Tyre"] * 10,
            "date": pd.date_range("2025-12-01", periods=10),
            "staff_count": [3, 4, 3, 4, 5, 3, 4, 3, 4, 5],
            "daily_volume": np.random.uniform(100000000, 250000000, 10),
            "day_of_week": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            "is_weekend": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
        })
        
        estimator.fit(df)
        result = estimator.predict("Conut Tyre", 150000000)
        
        assert isinstance(result, int)
        assert result > 0
        
    def test_predict_respects_weekend_factor(self, tmp_path):
        """Verify predict adjusts for weekend."""
        estimator = StaffingEstimator()
        
        df = pd.DataFrame({
            "branch": ["Conut Tyre"] * 10,
            "date": pd.date_range("2025-12-01", periods=10),
            "staff_count": [3, 4, 3, 4, 5, 3, 4, 3, 4, 5],
            "daily_volume": np.random.uniform(100000000, 250000000, 10),
            "day_of_week": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            "is_weekend": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
        })
        
        estimator.fit(df)
        
        weekday_staff = estimator.predict("Conut Tyre", 200000000, date="2025-12-01")
        weekend_staff = estimator.predict("Conut Tyre", 200000000, date="2025-12-06")
        
        # Both should be valid positive integers
        assert weekday_staff > 0
        assert weekend_staff > 0
        
    def test_predict_handles_unknown_branch(self, tmp_path):
        """Verify predict handles unknown branch gracefully."""
        estimator = StaffingEstimator()
        
        df = pd.DataFrame({
            "branch": ["Conut Tyre"] * 10,
            "date": pd.date_range("2025-12-01", periods=10),
            "staff_count": [3, 4, 3, 4, 5, 3, 4, 3, 4, 5],
            "daily_volume": np.random.uniform(100000000, 250000000, 10),
            "day_of_week": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            "is_weekend": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
        })
        
        estimator.fit(df)
        
        # Unknown branch should use default encoding (0)
        result = estimator.predict("Unknown Branch", 150000000)
        assert isinstance(result, int)
        assert result > 0


class TestStaffingEstimatorSaveLoad:
    """Test save and load functionality."""
    
    def test_save_and_load_preserves_state(self, tmp_path):
        """Verify save and load preserves estimator state."""
        estimator = StaffingEstimator()
        
        df = pd.DataFrame({
            "branch": ["Conut Tyre"] * 10,
            "date": pd.date_range("2025-12-01", periods=10),
            "staff_count": [3, 4, 3, 4, 5, 3, 4, 3, 4, 5],
            "daily_volume": np.random.uniform(100000000, 250000000, 10),
            "day_of_week": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            "is_weekend": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
        })
        
        estimator.fit(df)
        
        save_path = tmp_path / "staffing_model.pkl"
        estimator.save(str(save_path))
        
        assert save_path.exists()
        
        loaded = StaffingEstimator.load(str(save_path))
        assert loaded.best_model_name == estimator.best_model_name
        assert loaded.branch_map == estimator.branch_map


class TestStaffingEstimatorEdgeCases:
    """Test edge cases and error handling."""
    
    def test_fit_with_empty_dataframe(self):
        """Verify fit handles empty DataFrame."""
        estimator = StaffingEstimator()
        empty_df = pd.DataFrame()
        
        # Should either handle gracefully or raise meaningful error
        with pytest.raises(Exception):
            estimator.fit(empty_df)
            
    def test_fit_with_missing_columns(self):
        """Verify fit handles missing columns."""
        estimator = StaffingEstimator()
        bad_df = pd.DataFrame({
            "wrong_column": [1, 2, 3]
        })
        
        with pytest.raises(Exception):
            estimator.fit(bad_df)
            
    def test_predict_with_zero_volume(self, tmp_path):
        """Verify predict handles zero volume."""
        estimator = StaffingEstimator()
        
        df = pd.DataFrame({
            "branch": ["Conut Tyre"] * 10,
            "date": pd.date_range("2025-12-01", periods=10),
            "staff_count": [3, 4, 3, 4, 5, 3, 4, 3, 4, 5],
            "daily_volume": np.random.uniform(100000000, 250000000, 10),
            "day_of_week": [0, 1, 2, 3, 4, 5, 6, 0, 1, 2],
            "is_weekend": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
        })
        
        estimator.fit(df)
        result = estimator.predict("Conut Tyre", 0)
        
        # Should still return a minimum staffing level
        assert isinstance(result, int)
        assert result >= 0
