"""
Tests for Growth Strategy Analyzer
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

from models.growth_strategy import GrowthStrategyAnalyzer


class TestGrowthStrategyAnalyzerInitialization:
    """Test GrowthStrategyAnalyzer initialization."""
    
    def test_initialization(self):
        """Verify analyzer initializes correctly."""
        analyzer = GrowthStrategyAnalyzer()
        assert analyzer.branch_profiles == {}
        assert analyzer.item_performance == {}
        
    def test_division_keywords_defined(self):
        """Verify division keywords are defined."""
        analyzer = GrowthStrategyAnalyzer()
        assert "coffee" in analyzer.division_keywords
        assert "shake" in analyzer.division_keywords
        assert "pastry" in analyzer.division_keywords
        assert "food" in analyzer.division_keywords


class TestGrowthStrategyAnalyzerFit:
    """Test fitting the analyzer."""
    
    def test_fit_creates_branch_profiles(self, sample_sales_by_item_df):
        """Verify fit creates branch profiles."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        assert len(analyzer.branch_profiles) > 0
        
    def test_fit_creates_item_performance(self, sample_sales_by_item_df):
        """Verify fit creates item performance metrics."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        assert len(analyzer.item_performance) > 0
        
    def test_branch_profile_has_division_breakdown(self, sample_sales_by_item_df):
        """Verify branch profiles have division breakdown."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        for branch, profile in analyzer.branch_profiles.items():
            assert "total_revenue" in profile
            assert "coffee_revenue" in profile
            assert "shake_revenue" in profile
            assert "pastry_revenue" in profile
            assert "food_revenue" in profile
            
    def test_branch_profile_has_ratios(self, sample_sales_by_item_df):
        """Verify branch profiles have calculated ratios."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        for branch, profile in analyzer.branch_profiles.items():
            assert "coffee_ratio" in profile
            assert "shake_ratio" in profile
            assert "pastry_ratio" in profile
            assert "food_ratio" in profile
            
    def test_ratios_are_normalized(self, sample_sales_by_item_df):
        """Verify ratios are between 0 and 1."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        for branch, profile in analyzer.branch_profiles.items():
            assert 0 <= profile["coffee_ratio"] <= 1
            assert 0 <= profile["shake_ratio"] <= 1
            assert 0 <= profile["pastry_ratio"] <= 1
            assert 0 <= profile["food_ratio"] <= 1


class TestGrowthStrategyAnalyzerAnalyze:
    """Test analyzing branch growth strategy."""
    
    def test_analyze_returns_dict(self, sample_sales_by_item_df):
        """Verify analyze returns a dictionary."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        result = analyzer.analyze("Conut Tyre")
        
        assert isinstance(result, dict)
        
    def test_analyze_has_required_fields(self, sample_sales_by_item_df):
        """Verify analysis has all required fields."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        result = analyzer.analyze("Conut Tyre")
        
        required_fields = [
            "branch", "total_revenue", "coffee_ratio", "shake_ratio",
            "pastry_ratio", "food_ratio", "underperforming_divisions",
            "strengths", "top_items", "recommendations"
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
            
    def test_underperforming_divisions_list(self, sample_sales_by_item_df):
        """Verify underperforming divisions is a list."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        result = analyzer.analyze("Conut Tyre")
        
        assert isinstance(result["underperforming_divisions"], list)
        
    def test_strengths_list(self, sample_sales_by_item_df):
        """Verify strengths is a list."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        result = analyzer.analyze("Conut Tyre")
        
        assert isinstance(result["strengths"], list)
        
    def test_recommendations_list(self, sample_sales_by_item_df):
        """Verify recommendations is a list."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        result = analyzer.analyze("Conut Tyre")
        
        assert isinstance(result["recommendations"], list)
        
    def test_top_items_list(self, sample_sales_by_item_df):
        """Verify top items is a list."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        result = analyzer.analyze("Conut Tyre")
        
        assert isinstance(result["top_items"], list)
        
    def test_revenue_positive(self, sample_sales_by_item_df):
        """Verify total revenue is positive."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        result = analyzer.analyze("Conut Tyre")
        
        assert result["total_revenue"] > 0


class TestGrowthStrategyAnalyzerCoffeePerformance:
    """Test coffee-specific performance analysis."""
    
    def test_coffee_below_threshold_identified(self, sample_sales_by_item_df):
        """Verify branches with low coffee ratio are identified."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        result = analyzer.analyze("Conut Tyre")
        
        # If coffee ratio is below 20%, should be in underperforming
        if result["coffee_ratio"] < 0.20:
            assert "Coffee" in result["underperforming_divisions"] or \
                   any("coffee" in d.lower() for d in result["underperforming_divisions"])
                   
    def test_coffee_revenue_positive(self, sample_sales_by_item_df):
        """Verify coffee revenue is non-negative."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        result = analyzer.analyze("Conut Tyre")
        
        assert result.get("coffee_revenue", 0) >= 0


class TestGrowthStrategyAnalyzerShakePerformance:
    """Test shake-specific performance analysis."""
    
    def test_shake_below_threshold_identified(self, sample_sales_by_item_df):
        """Verify branches with low shake ratio are identified."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        result = analyzer.analyze("Conut Tyre")
        
        # If shake ratio is below 15%, should be in underperforming
        if result["shake_ratio"] < 0.15:
            assert "Shakes" in result["underperforming_divisions"] or \
                   any("shake" in d.lower() for d in result["underperforming_divisions"])
                   
    def test_shake_revenue_positive(self, sample_sales_by_item_df):
        """Verify shake revenue is non-negative."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        result = analyzer.analyze("Conut Tyre")
        
        assert result.get("shake_revenue", 0) >= 0


class TestGrowthStrategyAnalyzerGetAllProfiles:
    """Test getting all branch profiles."""
    
    def test_get_all_profiles_returns_list(self, sample_sales_by_item_df):
        """Verify get_all_profiles returns a list."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        profiles = analyzer.get_all_profiles()
        
        assert isinstance(profiles, list)
        
    def test_profiles_have_required_fields(self, sample_sales_by_item_df):
        """Verify profiles have required fields."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        profiles = analyzer.get_all_profiles()
        
        if profiles:
            required_fields = [
                "branch", "total_revenue", "coffee_ratio", "shake_ratio",
                "coffee_revenue", "shake_revenue"
            ]
            for field in required_fields:
                assert field in profiles[0], f"Missing field: {field}"


class TestGrowthStrategyAnalyzerGetItemPerformance:
    """Test getting item performance metrics."""
    
    def test_get_item_performance_returns_list(self, sample_sales_by_item_df):
        """Verify get_item_performance returns a list."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        items = analyzer.get_item_performance(top_n=5)
        
        assert isinstance(items, list)
        assert len(items) <= 5
        
    def test_items_sorted_by_revenue(self, sample_sales_by_item_df):
        """Verify items are sorted by revenue."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        items = analyzer.get_item_performance(top_n=5)
        
        if len(items) > 1:
            revenues = [item["total_revenue"] for item in items]
            assert revenues == sorted(revenues, reverse=True)


class TestGrowthStrategyAnalyzerEdgeCases:
    """Test edge cases."""
    
    def test_analyze_unknown_branch(self, sample_sales_by_item_df):
        """Verify analyze handles unknown branch gracefully."""
        analyzer = GrowthStrategyAnalyzer()
        analyzer.fit(sample_sales_by_item_df)
        
        result = analyzer.analyze("Unknown Branch")
        
        # Should return default/empty analysis
        assert isinstance(result, dict)
        assert result["branch"] == "Unknown Branch"
        
    def test_fit_with_empty_data(self):
        """Verify fit handles empty data."""
        analyzer = GrowthStrategyAnalyzer()
        empty_df = pd.DataFrame(columns=[
            "branch", "division", "group", "item", "qty", "total_amount"
        ])
        
        analyzer.fit(empty_df)
        
        assert len(analyzer.branch_profiles) == 0
        assert len(analyzer.item_performance) == 0
        
    def test_fit_with_missing_columns(self):
        """Verify fit handles missing columns."""
        analyzer = GrowthStrategyAnalyzer()
        bad_df = pd.DataFrame({"wrong_column": [1, 2, 3]})
        
        with pytest.raises(Exception):
            analyzer.fit(bad_df)
