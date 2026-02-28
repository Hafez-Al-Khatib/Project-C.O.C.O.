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
# True ReAct LLM Runner (LangGraph)
# ──────────────────────────────────────────────

def run_llm_react(query: str, api_key: str = None) -> list:
    """
    Runs the true LangGraph ReAct agent using an LLM.
    Parses the trajectory into the visual Thought→Action→Observation trace.
    """
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        return [
            {"type": "error", "content": "OPENAI_API_KEY is missing. Please provide it in the UI or set it as an environment variable."}
        ]

    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
        app = build_react_agent(llm)
        
        trace = []
        messages = [HumanMessage(content=query)]
        
        # Use stream to capture intermediate steps
        for chunk in app.stream({"messages": messages}):
            if "agent" in chunk:
                msg = chunk["agent"]["messages"][0]
                if getattr(msg, "tool_calls", None):
                    # Agent wants to act
                    if msg.content:
                        trace.append({"type": "thought", "content": msg.content})
                    else:
                        trace.append({"type": "thought", "content": f"Invoking tool to gather data..."})
                        
                    for tc in msg.tool_calls:
                        trace.append({
                            "type": "action",
                            "tool": tc["name"],
                            "input": json.dumps(tc["args"])
                        })
                else:
                    # Final answer
                    if msg.content:
                        trace.append({"type": "final_answer", "content": msg.content})
                        
            elif "tools" in chunk:
                # Observations from tool execution
                for msg in chunk["tools"]["messages"]:
                    content_str = msg.content
                    if len(content_str) > 2000:
                        content_str = content_str[:2000] + "\n... (truncated)"
                    trace.append({
                        "type": "observation",
                        "content": content_str
                    })
        
        return trace
        
    except Exception as e:
        logger.error(f"LLM Agent Error: {e}")
        return [
            {"type": "error", "content": f"Agent execution failed: {str(e)}"}
        ]
