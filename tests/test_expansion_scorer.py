"""
Tests for Expansion Scorer
==========================
"""

import os
import sys
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.expansion_scorer import ExpansionScorer, OSMClient


class TestOSMClientInitialization:
    """Test OSMClient initialization."""
    
    def test_initialization(self):
        """Verify client initializes correctly."""
        client = OSMClient()
        assert client.cache == {}
        
    def test_initialization_with_cache_file(self, tmp_path):
        """Verify client loads cache from file."""
        cache_file = tmp_path / "test_cache.json"
        cache_file.write_text('{"34.0_35.8": {"foot_traffic_index": 0.8}}')
        
        client = OSMClient(str(cache_file))
        assert "34.0_35.8" in client.cache


class TestOSMClientGetSpatialFeatures:
    """Test fetching spatial features from OSM."""
    
    @patch('models.expansion_scorer.requests.get')
    def test_successful_api_call(self, mock_get):
        """Verify successful API call returns features."""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "elements": [
                {"tags": {"amenity": "restaurant"}},
                {"tags": {"amenity": "cafe"}},
                {"tags": {"amenity": "university"}}
            ]
        }
        mock_get.return_value = mock_response
        
        client = OSMClient()
        features = client.get_spatial_features(34.0, 35.8)
        
        assert "foot_traffic_index" in features
        assert "commercial_density" in features
        assert "university_proximity" in features
        assert 0 <= features["foot_traffic_index"] <= 1
        
    @patch('models.expansion_scorer.requests.get')
    def test_api_error_fallback(self, mock_get):
        """Verify graceful fallback on API error."""
        mock_get.side_effect = Exception("Network error")
        
        client = OSMClient()
        features = client.get_spatial_features(34.0, 35.8)
        
        # Should return fallback values
        assert features["foot_traffic_index"] == 0.5
        assert features["commercial_density"] == 0.5
        assert features["university_proximity"] == 0.1
        
    @patch('models.expansion_scorer.requests.get')
    def test_caching(self, mock_get):
        """Verify results are cached."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"elements": []}
        mock_get.return_value = mock_response
        
        client = OSMClient()
        
        # First call
        client.get_spatial_features(34.0, 35.8)
        
        # Second call should use cache
        client.get_spatial_features(34.0, 35.8)
        
        # API should only be called once
        assert mock_get.call_count == 1


class TestExpansionScorerInitialization:
    """Test ExpansionScorer initialization."""
    
    def test_initialization(self):
        """Verify scorer initializes correctly."""
        scorer = ExpansionScorer()
        assert scorer.osm_client is not None
        assert len(scorer.reference_branches) == 0
        
    def test_osm_client_type(self):
        """Verify OSM client is correct type."""
        scorer = ExpansionScorer()
        assert isinstance(scorer.osm_client, OSMClient)


class TestExpansionScorerFit:
    """Test fitting the expansion scorer."""
    
    def test_fit_creates_reference_profiles(self, sample_sales_by_item_df):
        """Verify fit creates reference branch profiles."""
        scorer = ExpansionScorer()
        scorer.fit(sample_sales_by_item_df)
        
        assert len(scorer.reference_branches) > 0
        
    def test_fit_calculates_ratios(self, sample_sales_by_item_df):
        """Verify fit calculates category ratios."""
        scorer = ExpansionScorer()
        scorer.fit(sample_sales_by_item_df)
        
        for branch, profile in scorer.reference_branches.items():
            assert "coffee_ratio" in profile
            assert "pastry_ratio" in profile
            assert "food_ratio" in profile
            assert "beverage_ratio" in profile
            
    def test_ratios_sum_to_one(self, sample_sales_by_item_df):
        """Verify category ratios sum to approximately 1."""
        scorer = ExpansionScorer()
        scorer.fit(sample_sales_by_item_df)
        
        for branch, profile in scorer.reference_branches.items():
            total = profile["coffee_ratio"] + profile["pastry_ratio"] + \
                   profile["food_ratio"] + profile["beverage_ratio"]
            assert np.isclose(total, 1.0, rtol=1e-2)


class TestExpansionScorerScoreCandidate:
    """Test scoring expansion candidates."""
    
    @patch('models.expansion_scorer.OSMClient.get_spatial_features')
    def test_score_returns_dict(self, mock_get_features, sample_sales_by_item_df):
        """Verify score_candidate returns a dictionary."""
        mock_get_features.return_value = {
            "foot_traffic_index": 0.7,
            "commercial_density": 0.6,
            "university_proximity": 0.3
        }
        
        scorer = ExpansionScorer()
        scorer.fit(sample_sales_by_item_df)
        
        candidate_features = {
            "coffee_ratio": 0.4,
            "pastry_ratio": 0.2,
            "food_ratio": 0.3,
            "beverage_ratio": 0.1
        }
        
        result = scorer.score_candidate(34.0, 35.8, candidate_features)
        
        assert isinstance(result, dict)
        
    @patch('models.expansion_scorer.OSMClient.get_spatial_features')
    def test_score_has_required_fields(self, mock_get_features, sample_sales_by_item_df):
        """Verify score has all required fields."""
        mock_get_features.return_value = {
            "foot_traffic_index": 0.7,
            "commercial_density": 0.6,
            "university_proximity": 0.3
        }
        
        scorer = ExpansionScorer()
        scorer.fit(sample_sales_by_item_df)
        
        candidate_features = {
            "coffee_ratio": 0.4,
            "pastry_ratio": 0.2,
            "food_ratio": 0.3,
            "beverage_ratio": 0.1
        }
        
        result = scorer.score_candidate(34.0, 35.8, candidate_features)
        
        required_fields = [
            "feasibility_score", "spatial_score", "profile_match_score",
            "foot_traffic_index", "commercial_density", "university_proximity",
            "best_match_branch", "similarity_to_best"
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
            
    @patch('models.expansion_scorer.OSMClient.get_spatial_features')
    def test_score_in_valid_range(self, mock_get_features, sample_sales_by_item_df):
        """Verify feasibility score is in valid range."""
        mock_get_features.return_value = {
            "foot_traffic_index": 0.7,
            "commercial_density": 0.6,
            "university_proximity": 0.3
        }
        
        scorer = ExpansionScorer()
        scorer.fit(sample_sales_by_item_df)
        
        candidate_features = {
            "coffee_ratio": 0.4,
            "pastry_ratio": 0.2,
            "food_ratio": 0.3,
            "beverage_ratio": 0.1
        }
        
        result = scorer.score_candidate(34.0, 35.8, candidate_features)
        
        assert 0 <= result["feasibility_score"] <= 1
        assert 0 <= result["spatial_score"] <= 1
        assert 0 <= result["profile_match_score"] <= 1
        
    @patch('models.expansion_scorer.OSMClient.get_spatial_features')
    def test_similarity_calculation(self, mock_get_features, sample_sales_by_item_df):
        """Verify similarity is calculated correctly."""
        mock_get_features.return_value = {
            "foot_traffic_index": 0.7,
            "commercial_density": 0.6,
            "university_proximity": 0.3
        }
        
        scorer = ExpansionScorer()
        scorer.fit(sample_sales_by_item_df)
        
        # Candidate with same profile as existing branch
        existing_branch = list(scorer.reference_branches.keys())[0]
        candidate_features = scorer.reference_branches[existing_branch].copy()
        
        result = scorer.score_candidate(34.0, 35.8, candidate_features)
        
        # Should have high similarity to itself
        assert result["similarity_to_best"] > 0.9


class TestExpansionScorerFindSimilarBranches:
    """Test finding similar branches."""
    
    def test_find_similar_returns_list(self, sample_sales_by_item_df):
        """Verify find_similar_branches returns a list."""
        scorer = ExpansionScorer()
        scorer.fit(sample_sales_by_item_df)
        
        similar = scorer.find_similar_branches(
            {"coffee_ratio": 0.5, "pastry_ratio": 0.2, "food_ratio": 0.2, "beverage_ratio": 0.1},
            top_n=3
        )
        
        assert isinstance(similar, list)
        
    def test_find_similar_sorted_by_similarity(self, sample_sales_by_item_df):
        """Verify results are sorted by similarity."""
        scorer = ExpansionScorer()
        scorer.fit(sample_sales_by_item_df)
        
        similar = scorer.find_similar_branches(
            {"coffee_ratio": 0.5, "pastry_ratio": 0.2, "food_ratio": 0.2, "beverage_ratio": 0.1},
            top_n=3
        )
        
        if len(similar) > 1:
            similarities = [s["similarity"] for s in similar]
            assert similarities == sorted(similarities, reverse=True)


class TestExpansionScorerEdgeCases:
    """Test edge cases."""
    
    def test_fit_with_empty_data(self):
        """Verify fit handles empty data."""
        scorer = ExpansionScorer()
        empty_df = pd.DataFrame(columns=["branch", "division", "group", "item", "qty", "total_amount"])
        
        scorer.fit(empty_df)
        
        assert len(scorer.reference_branches) == 0
        
    def test_score_with_empty_reference(self):
        """Verify score handles empty reference branches."""
        scorer = ExpansionScorer()
        
        with pytest.raises(ValueError):
            scorer.score_candidate(34.0, 35.8, {"coffee_ratio": 0.5})
            
    def test_score_with_missing_features(self, sample_sales_by_item_df):
        """Verify score handles missing features."""
        scorer = ExpansionScorer()
        scorer.fit(sample_sales_by_item_df)
        
        # Missing some features
        incomplete_features = {"coffee_ratio": 0.5}
        
        with pytest.raises(Exception):
            scorer.score_candidate(34.0, 35.8, incomplete_features)
