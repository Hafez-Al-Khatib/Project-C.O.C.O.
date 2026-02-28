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


@tool
def generate_demand_confidence_plot(branch_name: str, historical_months: list[str], historical_sales: list[float], prediction_month: str, mean_prediction: float, lower_bound: float, upper_bound: float) -> str:
    """Generates a line chart showing historical sales and a GPR prediction with confidence bounds. Use this when the user asks for a visual forecast, chart, or graph of demand."""
    import requests
    try:
        resp = requests.post(
            "http://localhost:8000/tools/generate_demand_confidence_plot",
            json={"branch_name": branch_name, "historical_months": historical_months, "historical_sales": historical_sales, "prediction_month": prediction_month, "mean_prediction": mean_prediction, "lower_bound": lower_bound, "upper_bound": upper_bound},
            timeout=15
        )
        return json.dumps(resp.json(), indent=2)
    except Exception as e:
        return f"Plot error: {str(e)}"

@tool
def generate_combo_lift_plot(item_a: str, item_b: str, base_sales_a: float, base_sales_b: float, expected_lift_sales: float) -> str:
    """Generates a bar chart showing expected sales lift from bundling two items. Use this when the user asks for a combo lift chart or graph."""
    import requests
    try:
        resp = requests.post(
            "http://localhost:8000/tools/generate_combo_lift_plot",
            json={"item_a": item_a, "item_b": item_b, "base_sales_a": base_sales_a, "base_sales_b": base_sales_b, "expected_lift_sales": expected_lift_sales},
            timeout=15
        )
        return json.dumps(resp.json(), indent=2)
    except Exception as e:
        return f"Plot error: {str(e)}"

@tool
def generate_coffee_gap_plot(branch_names: list[str], coffee_ratios: list[float]) -> str:
    """Generates a horizontal bar chart of branch coffee ratio performance vs 20% target. Use this when the user asks to visualize coffee performance across branches."""
    import requests
    try:
        resp = requests.post(
            "http://localhost:8000/tools/generate_coffee_gap_plot",
            json={"branch_names": branch_names, "coffee_ratios": coffee_ratios},
            timeout=15
        )
        return json.dumps(resp.json(), indent=2)
    except Exception as e:
        return f"Plot error: {str(e)}"

# ──────────────────────────────────────────────
# LANGGRAPH ReAct Agent
# ──────────────────────────────────────────────

ALL_TOOLS = [sql_engine, model_inference, growth_strategy, generate_demand_confidence_plot, generate_combo_lift_plot, generate_coffee_gap_plot]

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent


def _sanitize_messages_for_gemini(messages):
    """Patch empty AIMessage content values before they reach the Gemini SDK.
    
    Gemini's protobuf converter rejects AIMessages with content='' or content=[].
    This occurs internally when LangGraph feeds tool-call-only messages back.
    """
    sanitized = []
    for m in messages:
        if isinstance(m, AIMessage):
            content = m.content
            if not content or (isinstance(content, list) and len(content) == 0):
                m = m.copy(update={"content": " "})
        sanitized.append(m)
    return sanitized


def build_react_agent(llm):
    """Build a LangGraph ReAct agent with operational tools and Gemini-safe message handling."""
    
    sys_prompt = (
        "You are an elite AI Copilot assisting the human Chief of Operations for Conut bakery. "
        "You do NOT make final decisions; you provide rigorous, tool-backed strategic recommendations.\n\n"
        "RULES:\n"
        "1. Always Use Tools: Never guess numbers. If asked about sales, staffing, combos, or expansion, you MUST execute the relevant API tools.\n"
        "2. Chain of Thought: If a user asks a complex question, use multiple tools in sequence before answering.\n"
        "3. Explain the Risk: Always report the mathematical uncertainty or confidence bounds returned by the models.\n"
        "4. Actionable Advice: End every response with a clear, data-driven recommendation for the human executive to approve. Use Markdown formatting.\n"
        "5. Plotting & BI Visuals: When the user asks for a chart or graph, call the appropriate plotting tool. Include the returned markdown_image string in your response.\n"
    )
    
    # Monkey-patch the LLM's _generate to sanitize messages before hitting the Gemini SDK
    original_generate = llm._generate
    
    def patched_generate(messages, stop=None, run_manager=None, **kwargs):
        messages = _sanitize_messages_for_gemini(messages)
        return original_generate(messages, stop=stop, run_manager=run_manager, **kwargs)
    
    llm._generate = patched_generate
    
    # Also patch async _agenerate if it exists
    if hasattr(llm, '_agenerate'):
        original_agenerate = llm._agenerate
        
        async def patched_agenerate(messages, stop=None, run_manager=None, **kwargs):
            messages = _sanitize_messages_for_gemini(messages)
            return await original_agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        
        llm._agenerate = patched_agenerate
    
    # Also patch _stream
    if hasattr(llm, '_stream'):
        original_stream = llm._stream
        
        def patched_stream(messages, stop=None, run_manager=None, **kwargs):
            messages = _sanitize_messages_for_gemini(messages)
            return original_stream(messages, stop=stop, run_manager=run_manager, **kwargs)
        
        llm._stream = patched_stream
    
    # Also patch _astream  
    if hasattr(llm, '_astream'):
        original_astream = llm._astream
        
        async def patched_astream(messages, stop=None, run_manager=None, **kwargs):
            messages = _sanitize_messages_for_gemini(messages)
            async for chunk in original_astream(messages, stop=stop, run_manager=run_manager, **kwargs):
                yield chunk
        
        llm._astream = patched_astream
    
    return create_react_agent(llm, ALL_TOOLS, prompt=sys_prompt)


# ──────────────────────────────────────────────
# True ReAct LLM Runner (LangGraph)
# ──────────────────────────────────────────────
import json
import os
import logging
from langchain_core.messages import HumanMessage, AIMessage

logger = logging.getLogger(__name__)

async def stream_llm_react(messages_data: list[dict], api_key: str = None):
    """
    Runs the true LangGraph ReAct agent using Gemini.
    Yields Server-Sent Events (SSE) formatting:
      - data: {"type":"trace", "step": {...}}   -> For Reasoning (Thought/Action/Observation)
      - data: {"type":"token", "content": "..."}-> For Final Answer streaming
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        yield f"data: {json.dumps({'type': 'error', 'content': 'GEMINI_API_KEY is missing. Please provide it in the UI.'})}\n\n"
        return

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
        app = build_react_agent(llm)
        
        # Hydrate messages safely for Gemini's strict role checks
        # Instead of sending all messages back (which fails if tools are skipped or roles dont alternate),
        # we collapse the past history into the latest prompt context.
        if not messages_data:
            langchain_msgs = [HumanMessage(content="Hello")]
        else:
            # Extract out the last user message
            last_query = messages_data[-1].get("content", "Hello").strip()
            if not last_query:
                last_query = "Hello"
            
            # Build conversation history for context
            history_str = ""
            for m in messages_data[:-1]:
                content = m.get("content", "").strip()
                if not content: continue
                role = "User" if m.get("role") == "user" else "Copilot"
                history_str += f"{role}: {content}\n\n"
            
            if history_str:
                final_content = f"PREVIOUS CONVERSATION CONTEXT:\n{history_str}\n\nACTUAL QUERY:\n{last_query}"
            else:
                final_content = last_query
                
            langchain_msgs = [HumanMessage(content=final_content)]

        # Use astream_events to catch both tool executions and streamed tokens
        async for event in app.astream_events({"messages": langchain_msgs}, version="v2"):
            kind = event["event"]
            
            # --- Stream the Agent's Final Answer Tokens ---
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content and not chunk.tool_calls:
                    content_to_yield = chunk.content
                    if isinstance(content_to_yield, list):
                        # Sometimes Gemini streams list chunks like [{"text": "..."}]
                        content_to_yield = "".join([c.get("text", "") for c in content_to_yield if isinstance(c, dict) and "text" in c])
                        
                    if isinstance(content_to_yield, str) and content_to_yield:
                        yield f"data: {json.dumps({'type': 'token', 'content': content_to_yield})}\n\n"

            # --- Capture Tool Calls (Action) ---
            elif kind == "on_tool_start":
                yield f"data: {json.dumps({'type': 'trace', 'step': {'type': 'action', 'tool': event['name'], 'input': json.dumps(event['data'].get('input'))}})}\n\n"

            # --- Capture Tool Outputs (Observation) ---
            elif kind == "on_tool_end":
                output = event['data'].get('output')
                if output:
                    if hasattr(output, 'content'):
                        content_str = str(output.content)
                    else:
                        content_str = str(output)
                    
                    if len(content_str) > 2000:
                        content_str = content_str[:2000] + "\n... (truncated)"
                    
                    yield f"data: {json.dumps({'type': 'trace', 'step': {'type': 'observation', 'content': content_str}})}\n\n"

    except Exception as e:
        import traceback
        logger.error(f"Agent streaming failed:\n{traceback.format_exc()}")
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
