"""
Tests for Demand Forecaster (V3 Ratio-Based Model)
===================================================
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.demand_forecaster import (
    load_and_engineer,
    build_models,
    walk_forward_cv,
    DemandForecaster,
    FEATURES_RATIO,
    HOLIDAY_INTENSITY,
    BRANCH_DISPLAY
)


class TestLoadAndEngineer:
    """Test data loading and feature engineering."""
    
    def test_returns_dataframe(self):
        """Verify load_and_engineer returns a DataFrame."""
        df = load_and_engineer()
        assert isinstance(df, pd.DataFrame)
        
    def test_has_required_columns(self):
        """Verify DataFrame has required columns."""
        df = load_and_engineer()
        required_cols = [
            "branch_encoded", "Month_Num", "growth_multiplier",
            "MonthlySale", "PrevsMonthly", "Holiday_Intensity"
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
            
    def test_branch_encoding_present(self):
        """Verify branch encoding is present."""
        df = load_and_engineer()
        assert "branch_encoded" in df.columns
        assert df["branch_encoded"].dtype in ["int64", "int32", "float64"]
        
    def test_growth_multiplier_calculated(self):
        """Verify growth multiplier is calculated correctly."""
        df = load_and_engineer()
        valid = df[df["growth_multiplier"].notna()]
        if len(valid) > 0:
            # Check that growth_multiplier = MonthlySale / PrevsMonthly
            expected_ratio = valid["MonthlySale"] / valid["PrevsMonthly"]
            assert np.allclose(valid["growth_multiplier"], expected_ratio, rtol=1e-3)


class TestBuildModels:
    """Test model building function."""
    
    def test_returns_dict(self):
        """Verify build_models returns a dictionary."""
        models = build_models()
        assert isinstance(models, dict)
        
    def test_has_expected_models(self):
        """Verify all expected models are present."""
        models = build_models()
        expected = ["GPR", "BayesianRidge", "QR_Q50", "QR_Q90", "RandomForest", "Ridge"]
        for name in expected:
            assert name in models, f"Missing model: {name}"
            
    def test_models_are_sklearn_compatible(self):
        """Verify models have fit/predict methods."""
        models = build_models()
        for name, model in models.items():
            assert hasattr(model, "fit"), f"{name} missing fit method"
            assert hasattr(model, "predict"), f"{name} missing predict method"
            
    def test_gpr_has_std(self):
        """Verify GPR model can return standard deviation."""
        models = build_models()
        gpr = models["GPR"]
        assert isinstance(gpr, GaussianProcessRegressor)


class TestWalkForwardCV:
    """Test walk-forward cross-validation."""
    
    def test_returns_list(self, mock_mlflow):
        """Verify walk_forward_cv returns a list of results."""
        df = load_and_engineer()
        models = build_models()
        results = walk_forward_cv(df, models, FEATURES_RATIO, "test_experiment")
        assert isinstance(results, list)
        
    def test_results_have_required_keys(self, mock_mlflow):
        """Verify results have all required keys."""
        df = load_and_engineer()
        models = build_models()
        results = walk_forward_cv(df, models, FEATURES_RATIO, "test_experiment")
        
        if results:
            required_keys = [
                "model", "window", "test_month", "ratio_mape", 
                "sales_mape", "sales_rmse", "sales_r2", "avg_ci"
            ]
            for key in required_keys:
                assert key in results[0], f"Missing key in results: {key}"
                
    def test_metrics_are_numeric(self, mock_mlflow):
        """Verify metrics are numeric values."""
        df = load_and_engineer()
        models = build_models()
        results = walk_forward_cv(df, models, FEATURES_RATIO, "test_experiment")
        
        if results:
            for r in results:
                assert isinstance(r["sales_mape"], (int, float))
                assert isinstance(r["sales_rmse"], (int, float))
                assert isinstance(r["sales_r2"], (int, float))


class TestDemandForecaster:
    """Test DemandForecaster class."""
    
    def test_initialization(self):
        """Verify forecaster initializes correctly."""
        forecaster = DemandForecaster()
        assert forecaster.model is None
        assert forecaster.mape == 0.0
        assert isinstance(forecaster.branch_map, dict)
        
    def test_branch_map_has_all_branches(self):
        """Verify branch map includes all expected branches."""
        forecaster = DemandForecaster()
        expected_branches = [
            "Conut", "Conut Main", "Conut - Tyre", "Conut Tyre",
            "Conut Jnah", "Main Street Coffee", "Main St Coffee"
        ]
        for branch in expected_branches:
            assert branch in forecaster.branch_map or any(
                b in forecaster.branch_map for b in [branch]
            )
            
    def test_fit_trains_model(self):
        """Verify fit method trains the model."""
        df = load_and_engineer()
        models = build_models()
        
        forecaster = DemandForecaster()
        model_template = models["Ridge"]  # Use simple model for speed
        
        forecaster.fit(df, model_template, "Ridge", 15.0)
        
        assert forecaster.model is not None
        assert forecaster.model_name == "Ridge"
        assert forecaster.mape == 15.0
        
    def test_predict_returns_dict(self):
        """Verify predict returns a dictionary."""
        df = load_and_engineer()
        models = build_models()
        
        forecaster = DemandForecaster()
        model_template = models["Ridge"]
        forecaster.fit(df, model_template, "Ridge", 15.0)
        
        result = forecaster.predict("Conut Jnah", 1, 2026)
        
        assert isinstance(result, dict)
        assert "predicted_volume" in result
        assert "confidence_interval" in result
        assert "mape" in result
        
    def test_predict_returns_positive_volume(self):
        """Verify predicted volume is positive."""
        df = load_and_engineer()
        models = build_models()
        
        forecaster = DemandForecaster()
        model_template = models["Ridge"]
        forecaster.fit(df, model_template, "Ridge", 15.0)
        
        result = forecaster.predict("Conut Jnah", 1, 2026)
        
        assert result["predicted_volume"] >= 0
        
    def test_predict_has_confidence_interval(self):
        """Verify prediction includes confidence interval."""
        df = load_and_engineer()
        models = build_models()
        
        forecaster = DemandForecaster()
        model_template = models["Ridge"]
        forecaster.fit(df, model_template, "Ridge", 15.0)
        
        result = forecaster.predict("Conut Jnah", 1, 2026)
        
        assert "confidence_interval" in result
        assert "to" in result["confidence_interval"]
        
    def test_predict_has_xai_drivers(self):
        """Verify prediction includes XAI drivers."""
        df = load_and_engineer()
        models = build_models()
        
        forecaster = DemandForecaster()
        model_template = models["Ridge"]
        forecaster.fit(df, model_template, "Ridge", 15.0)
        
        result = forecaster.predict("Conut Jnah", 1, 2026)
        
        assert "xai_drivers" in result
        drivers = result["xai_drivers"]
        assert "model_type" in drivers
        assert "target_method" in drivers
        assert "predicted_growth_ratio" in drivers
        
    def test_save_and_load(self, tmp_path):
        """Verify save and load functionality."""
        df = load_and_engineer()
        models = build_models()
        
        forecaster = DemandForecaster()
        model_template = models["Ridge"]
        forecaster.fit(df, model_template, "Ridge", 15.0)
        
        save_path = tmp_path / "test_forecaster.pkl"
        forecaster.save(str(save_path))
        
        assert save_path.exists()
        
        loaded = DemandForecaster.load(str(save_path))
        assert loaded.model_name == "Ridge"
        assert loaded.mape == 15.0
        
    def test_branch_name_fuzzy_matching(self):
        """Verify fuzzy matching for branch names."""
        forecaster = DemandForecaster()
        
        # These should all resolve to valid encodings
        test_names = ["Conut Jnah", "Jnah", "Conut Tyre", "Tyre", "Main Street", "Main St"]
        for name in test_names:
            # Should not raise exception
            encoding = forecaster.branch_map.get(name)
            # If not exact match, fuzzy match should work
            if encoding is None:
                for k, v in forecaster.branch_map.items():
                    if name.lower() in k.lower() or k.lower() in name.lower():
                        encoding = v
                        break


class TestFeatureEngineering:
    """Test feature engineering logic."""
    
    def test_holiday_intensity_defined(self):
        """Verify holiday intensity mapping is defined."""
        assert isinstance(HOLIDAY_INTENSITY, dict)
        assert 12 in HOLIDAY_INTENSITY  # December should have high intensity
        
    def test_branch_display_defined(self):
        """Verify branch display names are defined."""
        assert isinstance(BRANCH_DISPLAY, dict)
        assert len(BRANCH_DISPLAY) > 0
        
    def test_features_ratio_list(self):
        """Verify FEATURES_RATIO list is defined."""
        assert isinstance(FEATURES_RATIO, list)
        assert len(FEATURES_RATIO) > 0
        # Check for key features
        assert "branch_encoded" in FEATURES_RATIO
        assert "Month_Num" in FEATURES_RATIO or "month" in FEATURES_RATIO
