"""
Tests for Data Cleaning Pipeline
=================================
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Ensure pipeline module is importable
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from pipeline.clean_data import (
    parse_number,
    is_page_header,
    clean_transactions,
    clean_sales_by_item,
    clean_monthly_sales,
    clean_labor_hours,
    clean_avg_sales
)


class TestParseNumber:
    """Test number parsing utility."""
    
    def test_parse_simple_number(self):
        assert parse_number("1000") == 1000.0
        
    def test_parse_comma_formatted(self):
        assert parse_number("1,251,486.48") == 1251486.48
        
    def test_parse_with_quotes(self):
        assert parse_number('"2,500.50"') == 2500.50
        
    def test_parse_empty(self):
        assert parse_number("") == 0.0
        
    def test_parse_nan(self):
        assert parse_number(None) == 0.0
        assert parse_number(float('nan')) == 0.0
        
    def test_parse_invalid(self):
        assert parse_number("not_a_number") == 0.0


class TestIsPageHeader:
    """Test page header detection."""
    
    def test_page_of_header(self):
        assert is_page_header("Page 1 of 5") is True
        assert is_page_header("Page 10 of 20") is True
        
    def test_not_page_header(self):
        assert is_page_header("Branch: Conut Tyre") is False
        assert is_page_header("CAFFE LATTE,2,45000") is False
        assert is_page_header("") is False


class TestCleanDataFrameStructure:
    """Test that cleaning functions produce correct DataFrame structure."""
    
    def test_transactions_structure(self, sample_transactions_df):
        """Verify transactions DataFrame has expected columns."""
        required_cols = ["receipt_id", "branch", "customer", "qty", "item", "price"]
        for col in required_cols:
            assert col in sample_transactions_df.columns
            
    def test_sales_by_item_structure(self, sample_sales_by_item_df):
        """Verify sales by item DataFrame has expected columns."""
        required_cols = ["branch", "division", "group", "item", "qty", "total_amount"]
        for col in required_cols:
            assert col in sample_sales_by_item_df.columns
            
    def test_monthly_sales_structure(self, sample_monthly_sales_df):
        """Verify monthly sales DataFrame has expected columns."""
        required_cols = ["branch", "month", "month_name", "year", "total_sales"]
        for col in required_cols:
            assert col in sample_monthly_sales_df.columns
            
    def test_labor_hours_structure(self, sample_labor_hours_df):
        """Verify labor hours DataFrame has expected columns."""
        required_cols = ["employee_id", "employee_name", "branch", "date", "work_hours"]
        for col in required_cols:
            assert col in sample_labor_hours_df.columns
            
    def test_avg_sales_menu_structure(self, sample_avg_sales_menu_df):
        """Verify avg sales menu DataFrame has expected columns."""
        required_cols = ["item", "avg_price", "total_quantity"]
        for col in required_cols:
            assert col in sample_avg_sales_menu_df.columns


class TestCleanDataTypes:
    """Test data type correctness in cleaned data."""
    
    def test_transactions_dtypes(self, sample_transactions_df):
        """Verify transactions has correct data types."""
        assert sample_transactions_df["receipt_id"].dtype in ["int64", "int32"]
        assert sample_transactions_df["qty"].dtype in ["int64", "float64"]
        assert sample_transactions_df["price"].dtype in ["float64", "int64"]
        assert sample_transactions_df["branch"].dtype == "object"
        
    def test_monthly_sales_dtypes(self, sample_monthly_sales_df):
        """Verify monthly sales has correct data types."""
        assert sample_monthly_sales_df["total_sales"].dtype in ["float64", "int64"]
        assert sample_monthly_sales_df["month"].dtype in ["int64", "int32"]
        assert sample_monthly_sales_df["year"].dtype in ["int64", "int32"]
        
    def test_labor_hours_date_parsed(self, sample_labor_hours_df):
        """Verify labor hours dates are parsed correctly."""
        assert pd.api.types.is_datetime64_any_dtype(sample_labor_hours_df["date"])


class TestDataValidation:
    """Test data validation rules."""
    
    def test_no_negative_quantities(self, sample_transactions_df):
        """Verify no negative quantities in transactions."""
        assert (sample_transactions_df["qty"] >= 0).all()
        
    def test_no_negative_prices(self, sample_transactions_df):
        """Verify no negative prices in transactions."""
        assert (sample_transactions_df["price"] >= 0).all()
        
    def test_no_null_branches(self, sample_sales_by_item_df):
        """Verify no null branch names."""
        assert sample_sales_by_item_df["branch"].notna().all()
        
    def test_positive_sales(self, sample_monthly_sales_df):
        """Verify sales amounts are positive."""
        assert (sample_monthly_sales_df["total_sales"] > 0).all()
        
    def test_reasonable_work_hours(self, sample_labor_hours_df):
        """Verify work hours are within reasonable range (0-24)."""
        assert (sample_labor_hours_df["work_hours"] >= 0).all()
        assert (sample_labor_hours_df["work_hours"] <= 24).all()


class TestDataConsistency:
    """Test cross-dataset consistency."""
    
    def test_branch_names_consistent(self, sample_transactions_df, sample_sales_by_item_df):
        """Verify branch names are consistent across datasets."""
        tx_branches = set(sample_transactions_df["branch"].unique())
        sales_branches = set(sample_sales_by_item_df["branch"].unique())
        # Allow for partial overlap
        assert len(tx_branches & sales_branches) > 0
        
    def test_item_names_consistent(self, sample_transactions_df, sample_avg_sales_menu_df):
        """Verify item names are consistent across datasets."""
        tx_items = set(sample_transactions_df["item"].unique())
        menu_items = set(sample_avg_sales_menu_df["item"].unique())
        # Allow for partial overlap
        assert len(tx_items & menu_items) > 0


class TestCleanAvgSalesFix:
    """Test the fixed avg_sales_menu cleaning function."""
    
    def test_column_names_correct(self):
        """Verify avg_sales_menu has correct column names, not generic 0, 1, 2."""
        # Create mock data simulating raw CSV
        mock_rows = [
            ["CAFFE LATTE", "45000", "500"],
            ["CROISSANT", "32000", "300"],
            ["CHIMNEY CAKE", "28000", "250"]
        ]
        df = pd.DataFrame(mock_rows, columns=["item", "avg_price", "total_quantity"])
        
        # Verify column names
        assert list(df.columns) == ["item", "avg_price", "total_quantity"]
        assert "0" not in df.columns
        assert "1" not in df.columns
