"""
Tests for Expansion Scorer
==========================
"""

import os
import sys
import pytest
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from models.expansion_scorer import ExpansionScorer, OSMClient, cosine_similarity


class TestOSMClientInitialization:
    """Test OSMClient initialization."""
    
    def test_initialization_default_radius(self):
        """Verify OSMClient initializes with default radius."""
        client = OSMClient()
        assert client.radius == 1000
        assert client.cache is not None
        
    def test_initialization_custom_radius(self):
        """Verify OSMClient initializes with custom radius."""
        client = OSMClient(radius=500)
        assert client.radius == 500


class TestExpansionScorerInitialization:
    """Test ExpansionScorer initialization."""
    
    def test_initialization_default_reference(self):
        """Verify ExpansionScorer initializes with default reference."""
        scorer = ExpansionScorer()
        assert scorer.reference_branch == "Conut Jnah"
        assert scorer.profiles_df is None
        assert scorer.osm_client is not None
        assert len(scorer.feature_cols) > 0
        
    def test_initialization_custom_reference(self):
        """Verify ExpansionScorer initializes with custom reference."""
        scorer = ExpansionScorer(reference_branch="Conut Tyre")
        assert scorer.reference_branch == "Conut Tyre"
        
    def test_feature_cols_defined(self):
        """Verify feature columns are defined."""
        scorer = ExpansionScorer()
        expected_cols = [
            "coffee_ratio", "pastry_ratio", "drinks_ratio", "shakes_ratio",
            "foot_traffic_index", "commercial_density", "university_proximity"
        ]
        for col in expected_cols:
            assert col in scorer.feature_cols


class TestExpansionScorerFit:
    """Test fitting the expansion scorer."""
    
    @pytest.mark.skip(reason="Requires actual parquet data and OSM API")
    def test_fit_creates_profiles(self):
        """Verify fit creates branch profiles."""
        scorer = ExpansionScorer()
        scorer.fit()
        
        assert scorer.profiles_df is not None
        assert len(scorer.profiles_df) > 0
        
    @pytest.mark.skip(reason="Requires actual parquet data and OSM API")
    def test_fit_validates_reference_branch(self):
        """Verify fit validates reference branch exists."""
        scorer = ExpansionScorer(reference_branch="NonExistent")
        scorer.fit()
        
        # Should auto-select a valid reference
        assert scorer.reference_branch in scorer.profiles_df.index


class TestExpansionScorerScore:
    """Test scoring expansion candidates."""
    
    @pytest.mark.skip(reason="Requires actual parquet data and OSM API")
    def test_score_returns_dict(self):
        """Verify score returns a dictionary."""
        scorer = ExpansionScorer()
        scorer.fit()
        
        result = scorer.score(candidate_branch="Conut Tyre")
        
        assert isinstance(result, dict)
        
    @pytest.mark.skip(reason="Requires actual parquet data and OSM API")
    def test_score_has_required_fields(self):
        """Verify score result has all required fields."""
        scorer = ExpansionScorer()
        scorer.fit()
        
        result = scorer.score(candidate_branch="Conut Tyre")
        
        required_fields = [
            "reference_branch", "candidate", "similarity_score",
            "recommendation", "reference_profile", "candidate_profile", "gaps"
        ]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
            
    @pytest.mark.skip(reason="Requires actual parquet data and OSM API")
    def test_score_similarity_in_valid_range(self):
        """Verify similarity score is between 0 and 1."""
        scorer = ExpansionScorer()
        scorer.fit()
        
        result = scorer.score(candidate_branch="Conut Tyre")
        
        assert 0 <= result["similarity_score"] <= 1
        
    @pytest.mark.skip(reason="Requires actual parquet data and OSM API")
    def test_score_with_custom_features(self):
        """Verify score works with custom features."""
        scorer = ExpansionScorer()
        scorer.fit()
        
        features = {
            "coffee_ratio": 0.45,
            "pastry_ratio": 0.35,
            "drinks_ratio": 0.15,
            "shakes_ratio": 0.05,
            "foot_traffic_index": 0.5,
            "commercial_density": 0.5,
            "university_proximity": 0.5
        }
        
        result = scorer.score(candidate_features=features)
        
        assert isinstance(result, dict)
        assert "similarity_score" in result
        
    @pytest.mark.skip(reason="Requires actual parquet data and OSM API")
    def test_score_with_coordinates(self):
        """Verify score works with lat/lon coordinates."""
        scorer = ExpansionScorer()
        scorer.fit()
        
        features = {
            "coffee_ratio": 0.45,
            "pastry_ratio": 0.35,
            "drinks_ratio": 0.15,
            "shakes_ratio": 0.05
        }
        
        result = scorer.score(
            candidate_features=features,
            lat=34.4346,
            lon=35.8362
        )
        
        assert isinstance(result, dict)
        assert result.get("osm_data_fetched") is True


class TestExpansionScorerRankAll:
    """Test ranking all branches."""
    
    @pytest.mark.skip(reason="Requires actual parquet data and OSM API")
    def test_rank_all_returns_list(self):
        """Verify rank_all_branches returns a list."""
        scorer = ExpansionScorer()
        scorer.fit()
        
        results = scorer.rank_all_branches()
        
        assert isinstance(results, list)
        
    @pytest.mark.skip(reason="Requires actual parquet data and OSM API")
    def test_rank_all_sorted_by_similarity(self):
        """Verify results are sorted by similarity descending."""
        scorer = ExpansionScorer()
        scorer.fit()
        
        results = scorer.rank_all_branches()
        
        if len(results) > 1:
            scores = [r["similarity_score"] for r in results]
            assert scores == sorted(scores, reverse=True)


class TestCosineSimilarity:
    """Test cosine similarity utility."""
    
    def test_identical_vectors(self):
        """Verify identical vectors have similarity 1."""
        v1 = np.array([[1, 2, 3]])
        v2 = np.array([[1, 2, 3]])
        
        result = cosine_similarity(v1, v2)
        assert result[0][0] == 1.0
        
    def test_orthogonal_vectors(self):
        """Verify orthogonal vectors have similarity 0."""
        v1 = np.array([[1, 0, 0]])
        v2 = np.array([[0, 1, 0]])
        
        result = cosine_similarity(v1, v2)
        assert abs(result[0][0]) < 0.001


class TestExpansionScorerEdgeCases:
    """Test edge cases."""
    
    @pytest.mark.skip(reason="Requires actual parquet data and OSM API")
    def test_score_with_empty_features(self):
        """Verify score handles empty features gracefully."""
        scorer = ExpansionScorer()
        scorer.fit()
        
        result = scorer.score(candidate_features={})
        
        assert isinstance(result, dict)
        
    @pytest.mark.skip(reason="Requires actual parquet data and OSM API")
    def test_score_with_no_candidate(self):
        """Verify score handles missing candidate gracefully."""
        scorer = ExpansionScorer()
        scorer.fit()
        
        result = scorer.score()
        
        # Should return error
        assert "error" in result
