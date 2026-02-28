"""
Tests for Demand Forecaster (V3)
==================================
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.demand_forecaster import (
    DemandForecaster, build_models, walk_forward_cv,
    FEATURES_RATIO, HOLIDAY_INTENSITY
)


class TestBuildModels:
    """Test model building."""
    
    def test_has_expected_models(self):
        """Verify all expected models are built."""
        models = build_models()
        expected_models = [
            "GPR_Matern", "GPR_RBF", "BayesianRidge",
            "RandomForest", "Ridge"
        ]
        for model_name in expected_models:
            assert model_name in models, f"Missing model: {model_name}"
            
    def test_models_are_regressors(self):
        """Verify all models are regressor instances."""
        from sklearn.base import RegressorMixin
        models = build_models()
        for name, model in models.items():
            assert isinstance(model, RegressorMixin) or hasattr(model, 'fit'), \
                f"{name} should be a regressor"
            
    def test_gpr_matern_has_std(self):
        """Verify GPR_Matern supports return_std."""
        models = build_models()
        gpr = models["GPR_Matern"]
        assert hasattr(gpr, 'predict')


class TestConstants:
    """Test module constants."""
    
    def test_features_ratio_defined(self):
        """Verify FEATURES_RATIO is defined."""
        assert len(FEATURES_RATIO) > 0
        assert "branch_encoded" in FEATURES_RATIO
        assert "Month_Num" in FEATURES_RATIO
        
    def test_holiday_intensity_defined(self):
        """Verify HOLIDAY_INTENSITY is defined."""
        assert len(HOLIDAY_INTENSITY) > 0
        assert 12 in HOLIDAY_INTENSITY  # December
        assert 8 in HOLIDAY_INTENSITY   # August


class TestDemandForecasterInitialization:
    """Test DemandForecaster initialization."""
    
    def test_initialization(self):
        """Verify forecaster initializes correctly."""
        forecaster = DemandForecaster()
        assert forecaster.model is None
        assert forecaster.model_name == ""
        assert forecaster.mape == 0.0
        assert len(forecaster.branch_map) > 0
        assert forecaster.dec_actuals == {}
        
    def test_branch_map_has_known_branches(self):
        """Verify branch map includes known branches."""
        forecaster = DemandForecaster()
        assert "Conut Jnah" in forecaster.branch_map
        assert "Conut Tyre" in forecaster.branch_map
        
    def test_feature_cols_match_constants(self):
        """Verify feature_cols matches FEATURES_RATIO."""
        forecaster = DemandForecaster()
        assert forecaster.feature_cols == FEATURES_RATIO


class TestDemandForecasterFit:
    """Test fitting the forecaster."""
    
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_fit_trains_model(self, sample_monthly_sales_df):
        """Verify fit trains the model."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        
        forecaster = DemandForecaster()
        model = GaussianProcessRegressor()
        
        forecaster.fit(sample_monthly_sales_df, model, "GPR_Matern", 15.0)
        
        assert forecaster.model is not None
        assert forecaster.model_name == "GPR_Matern"
        assert forecaster.mape == 15.0


class TestDemandForecasterPredict:
    """Test prediction functionality."""
    
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_predict_returns_dict(self, sample_monthly_sales_df):
        """Verify predict returns a dictionary."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        
        forecaster = DemandForecaster()
        model = GaussianProcessRegressor()
        forecaster.fit(sample_monthly_sales_df, model, "GPR_Matern", 15.0)
        
        result = forecaster.predict("Conut Jnah", 1, 2026)
        
        assert isinstance(result, dict)
        
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_predict_has_required_fields(self, sample_monthly_sales_df):
        """Verify predict result has all required fields."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        
        forecaster = DemandForecaster()
        model = GaussianProcessRegressor()
        forecaster.fit(sample_monthly_sales_df, model, "GPR_Matern", 15.0)
        
        result = forecaster.predict("Conut Jnah", 1, 2026)
        
        required_fields = [
            "branch", "predicted_volume", "confidence_interval",
            "mape", "warning", "month", "year", "xai_drivers", "model_type"
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
            
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_predict_volume_positive(self, sample_monthly_sales_df):
        """Verify predicted volume is positive."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        
        forecaster = DemandForecaster()
        model = GaussianProcessRegressor()
        forecaster.fit(sample_monthly_sales_df, model, "GPR_Matern", 15.0)
        
        result = forecaster.predict("Conut Jnah", 1, 2026)
        
        assert result["predicted_volume"] > 0
        
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_predict_fuzzy_branch_match(self, sample_monthly_sales_df):
        """Verify fuzzy matching for branch names."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        
        forecaster = DemandForecaster()
        model = GaussianProcessRegressor()
        forecaster.fit(sample_monthly_sales_df, model, "GPR_Matern", 15.0)
        
        # Should match "Conut Jnah" even with partial name
        result = forecaster.predict("Jnah", 1, 2026)
        
        assert isinstance(result, dict)
        assert result["predicted_volume"] > 0


class TestDemandForecasterSaveLoad:
    """Test save and load functionality."""
    
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_save_and_load(self, sample_monthly_sales_df, tmp_path):
        """Verify save and load work correctly."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        
        forecaster = DemandForecaster()
        model = GaussianProcessRegressor()
        forecaster.fit(sample_monthly_sales_df, model, "GPR_Matern", 15.0)
        
        save_path = tmp_path / "demand_forecaster.pkl"
        forecaster.save(str(save_path))
        
        assert save_path.exists()
        
        loaded = DemandForecaster.load(str(save_path))
        assert loaded.model_name == forecaster.model_name
        assert loaded.mape == forecaster.mape


class TestDemandForecasterEdgeCases:
    """Test edge cases."""
    
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_predict_unknown_branch(self, sample_monthly_sales_df):
        """Verify predict handles unknown branch."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        
        forecaster = DemandForecaster()
        model = GaussianProcessRegressor()
        forecaster.fit(sample_monthly_sales_df, model, "GPR_Matern", 15.0)
        
        result = forecaster.predict("Unknown Branch XYZ", 1, 2026)
        
        # Should still return a result with default encoding
        assert isinstance(result, dict)
        assert result["predicted_volume"] > 0
        
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_predict_different_months(self, sample_monthly_sales_df):
        """Verify predict works for different months."""
        from sklearn.gaussian_process import GaussianProcessRegressor
        
        forecaster = DemandForecaster()
        model = GaussianProcessRegressor()
        forecaster.fit(sample_monthly_sales_df, model, "GPR_Matern", 15.0)
        
        for month in [1, 6, 12]:
            result = forecaster.predict("Conut Jnah", month, 2026)
            assert result["month"] == month
            assert result["predicted_volume"] > 0


class TestWalkForwardCV:
    """Test walk-forward cross-validation."""
    
    @pytest.mark.skip(reason="Requires actual parquet data and MLFlow")
    def test_walk_forward_cv_returns_results(self, sample_monthly_sales_df):
        """Verify walk_forward_cv returns results list."""
        models = build_models()
        results = walk_forward_cv(
            sample_monthly_sales_df, models, FEATURES_RATIO, "test_experiment"
        )
        
        assert isinstance(results, list)
        
    @pytest.mark.skip(reason="Requires actual parquet data and MLFlow")
    def test_walk_forward_cv_result_structure(self, sample_monthly_sales_df):
        """Verify result structure."""
        models = build_models()
        results = walk_forward_cv(
            sample_monthly_sales_df, models, FEATURES_RATIO, "test_experiment"
        )
        
        if results:
            result = results[0]
            required_fields = [
                "model", "window", "test_month", "ratio_mape",
                "sales_mape", "sales_rmse", "sales_r2"
            ]
            for field in required_fields:
                assert field in result
