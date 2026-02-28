"""
Tests for Staffing Estimator
===========================
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
        assert estimator.best_model_name == "GPR"
        assert estimator.best_model_instance is None
        assert estimator.branch_map == {}
        assert estimator.throughput_stats == {}
        assert estimator.cv_results == []
        
    def test_scaler_initialized(self):
        """Verify scaler is initialized."""
        from sklearn.preprocessing import StandardScaler
        estimator = StaffingEstimator()
        assert isinstance(estimator.scaler, StandardScaler)
        
    def test_best_model_is_GPR_by_default(self):
        """Verify default best model is GPR."""
        estimator = StaffingEstimator()
        assert estimator.best_model_name == "GPR"


class TestStaffingEstimatorFit:
    """Test fitting the staffing estimator."""
    
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_fit_creates_branch_map(self):
        """Verify fit creates branch mapping."""
        estimator = StaffingEstimator()
        estimator.fit()
        
        assert len(estimator.branch_map) > 0
        
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_fit_selects_best_model(self):
        """Verify fit selects the best model."""
        estimator = StaffingEstimator()
        estimator.fit()
        
        assert estimator.best_model_instance is not None


class TestStaffingEstimatorPredict:
    """Test predicting staffing requirements."""
    
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_predict_returns_positive_integer(self):
        """Verify predict returns positive integer."""
        estimator = StaffingEstimator()
        estimator.fit()
        
        result = estimator.predict("Conut Tyre", daily_volume=150000000, is_weekend=False)
        
        assert isinstance(result, int)
        assert result > 0
        
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_predict_respects_weekend_factor(self):
        """Verify predict respects weekend factor."""
        estimator = StaffingEstimator()
        estimator.fit()
        
        weekday = estimator.predict("Conut Tyre", daily_volume=100000000, is_weekend=False)
        weekend = estimator.predict("Conut Tyre", daily_volume=100000000, is_weekend=True)
        
        assert weekend >= weekday
        
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_predict_handles_unknown_branch(self):
        """Verify predict handles unknown branch gracefully."""
        estimator = StaffingEstimator()
        estimator.fit()
        
        result = estimator.predict("Unknown Branch", daily_volume=100000000, is_weekend=False)
        
        assert isinstance(result, int)
        assert result > 0


class TestStaffingEstimatorSaveLoad:
    """Test save and load functionality."""
    
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_save_and_load_preserves_state(self, tmp_path):
        """Verify save and load preserve model state."""
        estimator = StaffingEstimator()
        estimator.fit()
        
        save_path = tmp_path / "staffing_estimator.pkl"
        estimator.save(str(save_path))
        
        assert save_path.exists()
        
        loaded = StaffingEstimator.load(str(save_path))
        assert loaded.best_model_name == estimator.best_model_name
        assert loaded.branch_map is not None


class TestStaffingEstimatorEdgeCases:
    """Test edge cases."""
    
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_predict_with_zero_volume(self):
        """Verify predict handles zero volume gracefully."""
        estimator = StaffingEstimator()
        estimator.fit()
        
        result = estimator.predict("Conut Tyre", daily_volume=0, is_weekend=False)
        
        assert isinstance(result, int)
        assert result >= 1
        
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_predict_with_very_high_volume(self):
        """Verify predict handles very high volume."""
        estimator = StaffingEstimator()
        estimator.fit()
        
        result = estimator.predict("Conut Tyre", daily_volume=1000000000, is_weekend=False)
        
        assert isinstance(result, int)
        assert result > 0
        assert result < 100
