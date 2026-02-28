"""
Project C.O.C.O. — ReAct Agent (LangGraph)
============================================
A Chief of Operations Copilot powered by LangGraph's ReAct framework.
Uses three tools:
  1. SQL Engine  — DuckDB read-only queries on cleaned parquet files
  2. Model Inference — Calls trained ML models (Demand, Staffing, Expansion, Combos)
  3. Growth Strategy — Retrieves branch-level growth analysis

The agent follows: Thought → Action → Observation → Thought → ... → Final Answer
"""

import os
import json
import duckdb
import logging
from typing import Annotated, Sequence, TypedDict, Any

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")

# ──────────────────────────────────────────────
# TOOL 1: SQL Engine (DuckDB on Parquet files)
# ──────────────────────────────────────────────

PARQUET_TABLES = {
    "monthly_sales": os.path.join(CLEANED_DIR, "monthly_sales.parquet"),
    "sales_by_item": os.path.join(CLEANED_DIR, "sales_by_item.parquet"),
    "labor_hours": os.path.join(CLEANED_DIR, "labor_hours.parquet"),
    "transactions": os.path.join(CLEANED_DIR, "transactions.parquet"),
    "avg_sales_menu": os.path.join(CLEANED_DIR, "avg_sales_menu.parquet"),
}


@tool
def sql_engine(query: str) -> str:
    """Execute a read-only SQL query against the operational database.
    
    Available tables:
    - monthly_sales: columns [branch, month, total_sales]
    - sales_by_item: columns [branch, item, quantity, revenue]
    - labor_hours: columns [branch, date, employee_id, hours]
    - transactions: columns [branch, date, transaction_id, total]
    - avg_sales_menu: columns [item, avg_price, total_quantity]

    Only SELECT statements are allowed. Use table names directly (e.g. SELECT * FROM monthly_sales LIMIT 5).
    """
    query_upper = query.strip().upper()
    if not query_upper.startswith("SELECT"):
        return "Error: Only SELECT queries are permitted (read-only access)."

    forbidden = ["DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "CREATE", "TRUNCATE"]
    for word in forbidden:
        if word in query_upper:
            return f"Error: '{word}' operations are not permitted."

    try:
        conn = duckdb.connect(":memory:")
        for table_name, path in PARQUET_TABLES.items():
            if os.path.exists(path):
                conn.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet('{path.replace(os.sep, '/')}')")

        result = conn.execute(query).fetchdf()
        conn.close()

        if len(result) > 50:
            result = result.head(50)
            return result.to_string(index=False) + "\n... (truncated to 50 rows)"
        return result.to_string(index=False)
    except Exception as e:
        return f"SQL Error: {str(e)}"


# ──────────────────────────────────────────────
# TOOL 2: Model Inference
# ──────────────────────────────────────────────

@tool
def model_inference(model_name: str, params: str) -> str:
    """Run inference on a trained ML model.

    model_name: one of 'demand', 'staffing', 'expansion', 'combos'
    params: JSON string with the required parameters for that model.

    Examples:
      model_inference('demand', '{"branch_name": "Conut Jnah", "month": 11, "year": 2026}')
      model_inference('staffing', '{"branch_name": "Conut Jnah", "predicted_volume": 1250}')
      model_inference('expansion', '{"candidate_lat": 34.43, "candidate_lon": 35.83, "candidate_features": {"coffee_ratio": 0.45}}')
      model_inference('combos', '{"target_item": "CAFFE LATTE", "top_n": 3}')
    """
    import requests

    try:
        p = json.loads(params)
    except json.JSONDecodeError as e:
        return f"Error parsing params JSON: {e}"

    endpoints = {
        "demand": "http://localhost:8000/tools/predict_demand",
        "staffing": "http://localhost:8000/tools/estimate_staffing",
        "expansion": "http://localhost:8000/tools/expansion_feasibility",
        "combos": "http://localhost:8000/tools/get_combos",
    }

    url = endpoints.get(model_name)
    if not url:
        return f"Unknown model: '{model_name}'. Available: {list(endpoints.keys())}"

    try:
        resp = requests.post(url, json=p, timeout=30)
        return json.dumps(resp.json(), indent=2)
    except Exception as e:
        return f"Inference error: {str(e)}"


# ──────────────────────────────────────────────
# TOOL 3: Growth Strategy Analyzer
# ──────────────────────────────────────────────

@tool
def growth_strategy(branch_name: str) -> str:
    """Retrieve the growth strategy analysis for a specific branch.
    Returns product performance insights, growth recommendations,
    and intervention suggestions from the strategic analyzer.
    
    branch_name: e.g. 'Conut Jnah', 'Conut Tyre', 'Main Street Coffee'
    """
    import requests
    try:
        resp = requests.post(
            "http://localhost:8000/tools/growth_strategy",
            json={"branch_name": branch_name},
            timeout=15
        )
        return json.dumps(resp.json(), indent=2)
    except Exception as e:
        return f"Growth strategy error: {str(e)}"


# ──────────────────────────────────────────────
# LANGGRAPH ReAct Agent
# ──────────────────────────────────────────────

ALL_TOOLS = [sql_engine, model_inference, growth_strategy]


class AgentState(TypedDict):
    messages: Sequence[BaseMessage]


def build_react_agent(llm):
    """Build a LangGraph ReAct agent with the three operational tools."""
    
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    def should_continue(state: AgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"
        return END

    def call_model(state: AgentState):
        messages = state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    tool_node = ToolNode(ALL_TOOLS)

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")

    return graph.compile()


# ──────────────────────────────────────────────
# Deterministic ReAct Runner (No LLM Required)
# ──────────────────────────────────────────────

def run_deterministic_react(query: str, context: dict = None) -> list:
    """
    A deterministic ReAct chain that doesn't require an LLM API key.
    Follows: Thought → Action → Observation for each tool invocation.
    Returns the full reasoning trace as a list of steps.
    """
    trace = []
    ctx = context or {}
    branch = ctx.get("branch_name", "Conut Jnah")
    month = ctx.get("month", 11)
    year = ctx.get("year", 2026)

    # Step 1: SQL — gather historical context
    trace.append({
        "type": "thought",
        "content": f"I need to understand the historical performance of {branch} before making predictions. Let me query the sales database."
    })
    sql_q = f"SELECT branch, month, total_sales FROM monthly_sales WHERE branch LIKE '%{branch.split(' ')[-1]}%' ORDER BY month"
    trace.append({
        "type": "action",
        "tool": "sql_engine",
        "input": sql_q
    })
    sql_result = sql_engine.invoke(sql_q)
    trace.append({
        "type": "observation",
        "content": sql_result
    })

    # Step 2: Demand forecast
    trace.append({
        "type": "thought",
        "content": f"Now I have the historical sales data. Let me use the GPR demand forecasting model to predict {branch}'s volume for month {month}/{year}."
    })
    demand_params = json.dumps({"branch_name": branch, "month": month, "year": year})
    trace.append({
        "type": "action",
        "tool": "model_inference",
        "input": f"model_name='demand', params={demand_params}"
    })
    demand_result = model_inference.invoke({"model_name": "demand", "params": demand_params})
    trace.append({
        "type": "observation",
        "content": demand_result
    })

    try:
        demand_data = json.loads(demand_result)
        predicted_vol = demand_data.get("predicted_volume", 1250)
    except Exception:
        predicted_vol = 1250

    # Step 3: Staffing — chain from demand
    trace.append({
        "type": "thought",
        "content": f"The demand model predicts {predicted_vol:,.0f} transactions. I'll now chain this into the staffing estimator to determine how many crew members are needed."
    })
    staffing_params = json.dumps({
        "branch_name": branch,
        "predicted_volume": predicted_vol,
        "date": f"{year}-{str(month).zfill(2)}-15"
    })
    trace.append({
        "type": "action",
        "tool": "model_inference",
        "input": f"model_name='staffing', params={staffing_params}"
    })
    staffing_result = model_inference.invoke({"model_name": "staffing", "params": staffing_params})
    trace.append({
        "type": "observation",
        "content": staffing_result
    })

    try:
        staffing_data = json.loads(staffing_result)
    except Exception:
        staffing_data = {}

    # Step 4: Expansion feasibility
    trace.append({
        "type": "thought",
        "content": "Let me also evaluate a candidate expansion location in Tripoli using live OSM spatial data."
    })
    expansion_params = json.dumps({
        "candidate_lat": 34.4346, "candidate_lon": 35.8362,
        "candidate_features": {"coffee_ratio": 0.45, "pastry_ratio": 0.35, "drinks_ratio": 0.15, "shakes_ratio": 0.05}
    })
    trace.append({
        "type": "action",
        "tool": "model_inference",
        "input": f"model_name='expansion', params={expansion_params}"
    })
    expansion_result = model_inference.invoke({"model_name": "expansion", "params": expansion_params})
    trace.append({
        "type": "observation",
        "content": expansion_result
    })

    # Step 5: Combo optimization
    trace.append({
        "type": "thought",
        "content": "Finally, let me check what product combos would maximize basket size for the top-selling item."
    })
    combo_params = json.dumps({"target_item": "CAFFE LATTE", "top_n": 3})
    trace.append({
        "type": "action",
        "tool": "model_inference",
        "input": f"model_name='combos', params={combo_params}"
    })
    combo_result = model_inference.invoke({"model_name": "combos", "params": combo_params})
    trace.append({
        "type": "observation",
        "content": combo_result
    })

    # Step 6: Final synthesis
    staff_count = staffing_data.get("recommended_staff", "?")
    confidence = staffing_data.get("confidence_band", "N/A")
    model_type = staffing_data.get("model_type", "unknown")

    try:
        exp_data = json.loads(expansion_result)
        exp_score = exp_data.get("similarity_score", 0)
        exp_rec = exp_data.get("recommendation", "pending")
    except Exception:
        exp_score = 0
        exp_rec = "pending"

    try:
        combo_data = json.loads(combo_result)
        combo_item = combo_data.get("recommended_combo", "None")
        combo_reason = combo_data.get("business_reason", "")
    except Exception:
        combo_item = "None"
        combo_reason = ""

    synthesis = (
        f"Executive Brief for {branch} — {month}/{year}:\n\n"
        f"1. DEMAND: Predicted {predicted_vol:,.0f} transactions for the period.\n"
        f"2. STAFFING: Recommend {staff_count} crew members ({confidence}). Model: {model_type}.\n"
        f"3. EXPANSION: Tripoli site scored {exp_score:.2f}/1.00 — {exp_rec}.\n"
        f"4. MENU: Best pairing for Caffè Latte: {combo_item}. {combo_reason}\n\n"
        f"The demand-to-staffing chain is fully connected: predicted volume directly drives headcount allocation, "
        f"ensuring operational efficiency is optimized for the forecasted period."
    )

    trace.append({
        "type": "thought",
        "content": "I now have all the data needed to synthesize an executive operational brief."
    })
    trace.append({
        "type": "final_answer",
        "content": synthesis
    })

    return trace
