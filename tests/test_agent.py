"""
Tests for ReAct Agent
=====================
"""

import os
import sys
import pytest
import json
import duckdb
from pathlib import Path
from unittest.mock import patch, MagicMock

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

from agent.react_agent import (
    sql_engine,
    model_inference,
    growth_strategy,
    PARQUET_TABLES
)


class TestSQLTool:
    """Test SQL engine tool."""
    
    def test_sql_engine_select_query(self, temp_parquet_dir, sample_monthly_sales_df):
        """Verify SQL engine executes SELECT query."""
        # Create test parquet file
        test_file = temp_parquet_dir / "monthly_sales.parquet"
        sample_monthly_sales_df.to_parquet(test_file, index=False)
        
        # Temporarily override the parquet path
        original_path = PARQUET_TABLES.get("monthly_sales")
        PARQUET_TABLES["monthly_sales"] = str(test_file)
        
        try:
            result = sql_engine.invoke({"query": "SELECT * FROM monthly_sales LIMIT 5"})
            assert "Conut Tyre" in result or "branch" in result or "error" not in result.lower()
        finally:
            PARQUET_TABLES["monthly_sales"] = original_path
            
    def test_sql_engine_rejects_non_select(self):
        """Verify SQL engine rejects non-SELECT queries."""
        result = sql_engine.invoke({"query": "DROP TABLE monthly_sales"})
        assert "Error" in result or "Only SELECT" in result
        
    def test_sql_engine_rejects_forbidden_operations(self):
        """Verify SQL engine rejects forbidden operations."""
        forbidden_queries = [
            "DELETE FROM monthly_sales",
            "INSERT INTO monthly_sales VALUES (1,2,3)",
            "UPDATE monthly_sales SET branch='test'",
            "ALTER TABLE monthly_sales ADD COLUMN test INT",
            "CREATE TABLE test (id INT)",
            "TRUNCATE TABLE monthly_sales"
        ]
        
        for query in forbidden_queries:
            result = sql_engine.invoke({"query": query})
            assert "Error" in result or "not permitted" in result.lower()
            
    def test_sql_engine_handles_invalid_table(self):
        """Verify SQL engine handles invalid table gracefully."""
        result = sql_engine.invoke({"query": "SELECT * FROM nonexistent_table"})
        # Should return error message
        assert "Error" in result or "not found" in result.lower()


class TestModelInferenceTool:
    """Test model inference tool."""
    
    @patch('agent.react_agent.requests.post')
    def test_model_inference_demand(self, mock_post):
        """Verify model_inference calls demand endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "predicted_volume": 250000000,
            "confidence_interval": "200M to 300M",
            "mape": 15.0
        }
        mock_post.return_value = mock_response
        
        result = model_inference.invoke({
            "model_name": "demand",
            "params": '{"branch_name": "Conut Jnah", "month": 1, "year": 2026}'
        })
        
        assert "predicted_volume" in result
        assert "250000000" in result or "250M" in result
        
    @patch('agent.react_agent.requests.post')
    def test_model_inference_staffing(self, mock_post):
        """Verify model_inference calls staffing endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"staff_needed": 4}
        mock_post.return_value = mock_response
        
        result = model_inference.invoke({
            "model_name": "staffing",
            "params": '{"branch_name": "Conut Jnah", "predicted_volume": 150000000}'
        })
        
        assert "staff_needed" in result
        assert "4" in result
        
    @patch('agent.react_agent.requests.post')
    def test_model_inference_expansion(self, mock_post):
        """Verify model_inference calls expansion endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"feasibility_score": 0.75}
        mock_post.return_value = mock_response
        
        result = model_inference.invoke({
            "model_name": "expansion",
            "params": '{"candidate_lat": 34.0, "candidate_lon": 35.8, "candidate_features": {}}'
        })
        
        assert "feasibility_score" in result
        
    @patch('agent.react_agent.requests.post')
    def test_model_inference_combos(self, mock_post):
        """Verify model_inference calls combos endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"recommendations": [{"item": "CROISSANT"}]}
        mock_post.return_value = mock_response
        
        result = model_inference.invoke({
            "model_name": "combos",
            "params": '{"target_item": "CAFFE LATTE", "top_n": 3}'
        })
        
        assert "recommendations" in result
        
    def test_model_inference_invalid_json(self):
        """Verify model_inference handles invalid JSON."""
        result = model_inference.invoke({
            "model_name": "demand",
            "params": "invalid json"
        })
        
        assert "Error" in result
        
    def test_model_inference_unknown_model(self):
        """Verify model_inference handles unknown model."""
        result = model_inference.invoke({
            "model_name": "unknown",
            "params": '{}'
        })
        
        assert "Unknown" in result or "Error" in result


class TestGrowthStrategyTool:
    """Test growth strategy tool."""
    
    @patch('agent.react_agent.requests.post')
    def test_growth_strategy_success(self, mock_post):
        """Verify growth_strategy returns analysis."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "total_revenue": 500000000,
            "coffee_ratio": 0.25,
            "underperforming_divisions": ["Shakes"],
            "recommendations": ["Promote milkshakes"]
        }
        mock_post.return_value = mock_response
        
        result = growth_strategy.invoke({"branch_name": "Conut Jnah"})
        
        assert "total_revenue" in result
        assert "500000000" in result or "500M" in result
        
    @patch('agent.react_agent.requests.post')
    def test_growth_strategy_error(self, mock_post):
        """Verify growth_strategy handles errors."""
        mock_post.side_effect = Exception("Connection error")
        
        result = growth_strategy.invoke({"branch_name": "Conut Jnah"})
        
        assert "error" in result.lower()


class TestToolSchemas:
    """Test tool schema definitions."""
    
    def test_sql_engine_description(self):
        """Verify SQL engine has proper description."""
        assert sql_engine.description is not None
        assert "SELECT" in sql_engine.description
        assert "monthly_sales" in sql_engine.description
        
    def test_model_inference_description(self):
        """Verify model_inference has proper description."""
        assert model_inference.description is not None
        assert "demand" in model_inference.description
        assert "staffing" in model_inference.description
        
    def test_growth_strategy_description(self):
        """Verify growth_strategy has proper description."""
        assert growth_strategy.description is not None
        assert "growth" in growth_strategy.description.lower()


class TestParquetTables:
    """Test parquet table paths."""
    
    def test_parquet_tables_defined(self):
        """Verify all parquet tables are defined."""
        expected_tables = [
            "monthly_sales", "sales_by_item", "labor_hours",
            "transactions", "avg_sales_menu"
        ]
        for table in expected_tables:
            assert table in PARQUET_TABLES, f"Missing table: {table}"
            
    def test_parquet_paths_exist_or_not_required(self):
        """Verify parquet paths point to valid locations."""
        for table, path in PARQUET_TABLES.items():
            assert path is not None
            assert len(path) > 0


class TestBuildReactAgent:
    """Test ReAct agent builder."""
    
    def test_build_react_agent_returns_callable(self):
        """Verify build_react_agent returns a callable agent."""
        from agent.react_agent import build_react_agent
        
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.bind_tools.return_value = MagicMock()
        
        agent = build_react_agent(mock_llm)
        
        assert agent is not None
        
    def test_agent_has_tools(self):
        """Verify agent has access to tools."""
        from agent.react_agent import ALL_TOOLS
        
        assert len(ALL_TOOLS) >= 3
        tool_names = [t.name for t in ALL_TOOLS]
        assert "sql_engine" in tool_names
        assert "model_inference" in tool_names


class TestStreamLLMReact:
    """Test LLM streaming function."""
    
    @pytest.mark.asyncio
    async def test_stream_llm_react_requires_api_key(self):
        """Verify stream_llm_react requires API key."""
        from agent.react_agent import stream_llm_react
        
        # Temporarily clear env var
        orig_key = os.environ.pop("GEMINI_API_KEY", None)
        
        try:
            results = []
            async for event in stream_llm_react([{"role": "user", "content": "Hello"}]):
                results.append(event)
                
            assert len(results) > 0
            assert any("GEMINI_API_KEY is missing" in r for r in results)
        finally:
            if orig_key:
                os.environ["GEMINI_API_KEY"] = orig_key
