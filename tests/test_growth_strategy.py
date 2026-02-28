"""
Tests for Growth Strategy Analyzer
==================================
"""

import os
import sys
import pytest
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.growth_strategy import GrowthStrategyAnalyzer, COFFEE_DIVISIONS, SHAKE_DIVISIONS


class TestGrowthStrategyAnalyzerInitialization:
    """Test GrowthStrategyAnalyzer initialization."""
    
    def test_initialization(self):
        """Verify analyzer initializes correctly."""
        analyzer = GrowthStrategyAnalyzer()
        assert analyzer.sales_df is None
        assert analyzer.branch_coffee is None
        assert analyzer.branch_shakes is None
        assert analyzer.branch_totals is None
        
    def test_division_constants_defined(self):
        """Verify division constants are defined."""
        assert "Hot-Coffee Based" in COFFEE_DIVISIONS
        assert "Shakes" in SHAKE_DIVISIONS
        assert len(COFFEE_DIVISIONS) > 0
        assert len(SHAKE_DIVISIONS) > 0


class TestGrowthStrategyAnalyzerFit:
    """Test fitting the analyzer."""
    
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_fit_creates_branch_totals(self):
        """Verify fit creates branch totals."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit()
        
        assert analyzer.branch_totals is not None
        assert len(analyzer.branch_totals) > 0
        
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_fit_creates_coffee_data(self):
        """Verify fit creates coffee performance data."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit()
        
        assert analyzer.branch_coffee is not None
        
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_fit_creates_shake_data(self):
        """Verify fit creates shake performance data."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit()
        
        assert analyzer.branch_shakes is not None
        
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_fit_loads_sales_df(self):
        """Verify fit loads sales dataframe."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit()
        
        assert analyzer.sales_df is not None
        assert len(analyzer.sales_df) > 0


class TestGrowthStrategyAnalyzerGetStrategy:
    """Test getting growth strategy."""
    
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_get_strategy_returns_dict(self):
        """Verify get_strategy returns a dictionary for single branch."""
        analyzer = GrowthStrategyAnalyzer()
        result = analyzer.get_strategy("Conut Tyre")
        
        assert isinstance(result, dict)
        
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_get_strategy_has_branch(self):
        """Verify result has branch field."""
        analyzer = GrowthStrategyAnalyzer()
        result = analyzer.get_strategy("Conut Tyre")
        
        assert "branch" in result
        
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_get_strategy_has_metrics(self):
        """Verify result has metrics."""
        analyzer = GrowthStrategyAnalyzer()
        result = analyzer.get_strategy("Conut Tyre")
        
        assert "metrics" in result
        metrics = result["metrics"]
        assert "total_revenue" in metrics
        assert "coffee_ratio_actual" in metrics
        assert "shake_ratio_actual" in metrics
        
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_get_strategy_has_franchise_rank(self):
        """Verify result has franchise rank info."""
        analyzer = GrowthStrategyAnalyzer()
        result = analyzer.get_strategy("Conut Tyre")
        
        assert "franchise_rank" in result
        
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_get_strategy_has_best_selling_assets(self):
        """Verify result has best selling assets."""
        analyzer = GrowthStrategyAnalyzer()
        result = analyzer.get_strategy("Conut Tyre")
        
        assert "best_selling_assets" in result
        assets = result["best_selling_assets"]
        assert "top_coffees" in assets
        assert "top_shakes" in assets
        
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_get_strategy_has_status(self):
        """Verify result has status flags."""
        analyzer = GrowthStrategyAnalyzer()
        result = analyzer.get_strategy("Conut Tyre")
        
        assert "status" in result
        status = result["status"]
        assert "coffee_struggling" in status
        assert "shakes_struggling" in status
        
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_get_all_branches_returns_list(self):
        """Verify get_strategy with no branch returns list."""
        analyzer = GrowthStrategyAnalyzer()
        results = analyzer.get_strategy()
        
        assert isinstance(results, list)
        assert len(results) > 0


class TestGrowthStrategyAnalyzerEdgeCases:
    """Test edge cases."""
    
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_get_strategy_unknown_branch(self):
        """Verify get_strategy handles unknown branch gracefully."""
        analyzer = GrowthStrategyAnalyzer()
        result = analyzer.get_strategy("Unknown Branch")
        
        # Should return error info
        assert isinstance(result, dict)
        assert "error" in result or "branch" in result
        
    @pytest.mark.skip(reason="Requires actual parquet data")
    def test_auto_fit_on_get_strategy(self):
        """Verify get_strategy auto-fits if not already fitted."""
        analyzer = GrowthStrategyAnalyzer()
        # Don't call fit() explicitly
        
        result = analyzer.get_strategy("Conut Tyre")
        
        # Should have auto-fitted
        assert analyzer.sales_df is not None
        assert isinstance(result, dict)
