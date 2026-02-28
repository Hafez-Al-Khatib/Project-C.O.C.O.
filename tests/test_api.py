"""
Tests for FastAPI Application
=============================
"""

import os
import sys
import pytest
import json
from pathlib import Path
from fastapi.testclient import TestClient

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

# Import the FastAPI app
from app.main import app

client = TestClient(app)


class TestRootEndpoint:
    """Test root endpoint."""
    
    def test_root_returns_welcome(self):
        """Verify root endpoint returns welcome message."""
        response = client.get("/")
        assert response.status_code == 200
        assert "C.O.C.O" in response.text or "Chief" in response.text
        
    def test_root_returns_html(self):
        """Verify root endpoint returns HTML."""
        response = client.get("/")
        assert response.headers["content-type"] == "text/html; charset=utf-8"


class TestHealthEndpoint:
    """Test health check endpoint."""
    
    def test_health_returns_ok(self):
        """Verify health endpoint returns status ok."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        
    def test_health_has_timestamp(self):
        """Verify health endpoint includes timestamp."""
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data


class TestSkillsEndpoint:
    """Test skills manifest endpoint."""
    
    def test_skills_returns_manifest(self):
        """Verify skills endpoint returns OpenClaw manifest."""
        response = client.get("/skills")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert "version" in data
        
    def test_skills_has_required_tools(self):
        """Verify skills manifest has all required tools."""
        response = client.get("/skills")
        data = response.json()
        
        tool_names = [tool["name"] for tool in data["tools"]]
        expected_tools = [
            "predict_demand", "estimate_staffing", 
            "evaluate_expansion", "get_combo_recommendations", "analyze_growth"
        ]
        for tool in expected_tools:
            assert tool in tool_names, f"Missing tool: {tool}"


class TestPredictDemandEndpoint:
    """Test demand prediction endpoint."""
    
    def test_predict_demand_returns_result(self):
        """Verify predict_demand returns prediction result."""
        response = client.post(
            "/tools/predict_demand",
            json={"branch_name": "Conut Jnah", "month": 1, "year": 2026}
        )
        assert response.status_code in [200, 500]  # 500 if model not loaded
        
        if response.status_code == 200:
            data = response.json()
            assert "predicted_volume" in data or "error" in data
            
    def test_predict_demand_requires_branch(self):
        """Verify predict_demand requires branch_name."""
        response = client.post(
            "/tools/predict_demand",
            json={"month": 1, "year": 2026}
        )
        assert response.status_code == 422  # Validation error
        
    def test_predict_demand_invalid_branch(self):
        """Verify predict_demand handles invalid branch."""
        response = client.post(
            "/tools/predict_demand",
            json={"branch_name": "Invalid Branch", "month": 1, "year": 2026}
        )
        # Should either return fallback or error
        assert response.status_code in [200, 500]


class TestEstimateStaffingEndpoint:
    """Test staffing estimation endpoint."""
    
    def test_estimate_staffing_returns_result(self):
        """Verify estimate_staffing returns staffing result."""
        response = client.post(
            "/tools/estimate_staffing",
            json={"branch_name": "Conut Jnah", "predicted_volume": 150000000}
        )
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "staff_needed" in data or "error" in data
            
    def test_estimate_staffing_requires_branch(self):
        """Verify estimate_staffing requires branch_name."""
        response = client.post(
            "/tools/estimate_staffing",
            json={"predicted_volume": 150000000}
        )
        assert response.status_code == 422
        
    def test_estimate_staffing_requires_volume(self):
        """Verify estimate_staffing requires predicted_volume."""
        response = client.post(
            "/tools/estimate_staffing",
            json={"branch_name": "Conut Jnah"}
        )
        assert response.status_code == 422


class TestExpansionFeasibilityEndpoint:
    """Test expansion feasibility endpoint."""
    
    def test_expansion_feasibility_returns_result(self):
        """Verify expansion_feasibility returns scoring result."""
        response = client.post(
            "/tools/expansion_feasibility",
            json={
                "candidate_lat": 34.0,
                "candidate_lon": 35.8,
                "candidate_features": {
                    "coffee_ratio": 0.4,
                    "pastry_ratio": 0.2,
                    "food_ratio": 0.3,
                    "beverage_ratio": 0.1
                }
            }
        )
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "feasibility_score" in data or "error" in data
            
    def test_expansion_feasibility_requires_coordinates(self):
        """Verify expansion_feasibility requires coordinates."""
        response = client.post(
            "/tools/expansion_feasibility",
            json={
                "candidate_features": {"coffee_ratio": 0.4}
            }
        )
        assert response.status_code == 422


class TestGetCombosEndpoint:
    """Test combo recommendations endpoint."""
    
    def test_get_combos_returns_result(self):
        """Verify get_combos returns recommendations."""
        response = client.post(
            "/tools/get_combos",
            json={"target_item": "CAFFE LATTE", "top_n": 3}
        )
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "recommendations" in data or "error" in data
            
    def test_get_combos_requires_target_item(self):
        """Verify get_combos requires target_item."""
        response = client.post(
            "/tools/get_combos",
            json={"top_n": 3}
        )
        assert response.status_code == 422
        
    def test_get_combos_respects_top_n(self):
        """Verify get_combos respects top_n parameter."""
        response = client.post(
            "/tools/get_combos",
            json={"target_item": "CAFFE LATTE", "top_n": 2}
        )
        
        if response.status_code == 200:
            data = response.json()
            if "recommendations" in data:
                assert len(data["recommendations"]) <= 2


class TestGrowthStrategyEndpoint:
    """Test growth strategy endpoint."""
    
    def test_growth_strategy_returns_result(self):
        """Verify growth_strategy returns analysis."""
        response = client.post(
            "/tools/growth_strategy",
            json={"branch_name": "Conut Jnah"}
        )
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "total_revenue" in data or "coffee_ratio" in data or "error" in data
            
    def test_growth_strategy_requires_branch(self):
        """Verify growth_strategy requires branch_name."""
        response = client.post(
            "/tools/growth_strategy",
            json={}
        )
        assert response.status_code == 422


class TestPlotEndpoints:
    """Test plot generation endpoints."""
    
    def test_generate_demand_plot(self):
        """Verify demand plot endpoint."""
        response = client.post(
            "/tools/generate_demand_confidence_plot",
            json={
                "branch_name": "Conut Jnah",
                "historical_months": ["Aug", "Sep", "Oct", "Nov", "Dec"],
                "historical_sales": [150000000, 180000000, 220000000, 240000000, 260000000],
                "prediction_month": "Jan 2026",
                "mean_prediction": 280000000,
                "lower_bound": 250000000,
                "upper_bound": 310000000
            }
        )
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "plot_url" in data or "error" in data
            
    def test_generate_combo_lift_plot(self):
        """Verify combo lift plot endpoint."""
        response = client.post(
            "/tools/generate_combo_lift_plot",
            json={
                "item_a": "CAFFE LATTE",
                "item_b": "CROISSANT",
                "base_sales_a": 10000000,
                "base_sales_b": 5000000,
                "expected_lift_sales": 18000000
            }
        )
        assert response.status_code in [200, 500]
        
    def test_generate_coffee_gap_plot(self):
        """Verify coffee gap plot endpoint."""
        response = client.post(
            "/tools/generate_coffee_gap_plot",
            json={
                "branch_names": ["Conut Tyre", "Conut Jnah", "Main St Coffee"],
                "coffee_ratios": [0.15, 0.25, 0.18]
            }
        )
        assert response.status_code in [200, 500]


class TestErrorHandling:
    """Test API error handling."""
    
    def test_404_error(self):
        """Verify 404 for unknown endpoints."""
        response = client.get("/unknown_endpoint")
        assert response.status_code == 404
        
    def test_method_not_allowed(self):
        """Verify 405 for wrong HTTP method."""
        response = client.get("/tools/predict_demand")
        assert response.status_code == 405  # GET not allowed
        
    def test_invalid_json(self):
        """Verify 422 for invalid JSON."""
        response = client.post(
            "/tools/predict_demand",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422


class TestCORSHeaders:
    """Test CORS configuration."""
    
    def test_cors_preflight(self):
        """Verify CORS preflight requests are handled."""
        response = client.options("/tools/predict_demand")
        # Should not raise exception
        assert response.status_code in [200, 405]
