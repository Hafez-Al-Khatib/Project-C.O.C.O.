"""
Project C.O.C.O. - FastAPI Application

Agent Gateway serving as the Private Skills Registry for OpenClaw.
Every endpoint has graceful error fallbacks.
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import matplotlib.pyplot as plt
import uuid
import logging

logging.basicConfig(level=logging.ERROR)

# Ensure project root is on path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from app.schemas import (
    ComboRequest, ComboResponse,
    ExpansionRequest, ExpansionResponse,
    GrowthRequest, GrowthResponse,
    DemandRequest, DemandResponse,
    StaffingRequest, StaffingResponse,
)

# Globals for loaded models
combo_optimizer = None
expansion_scorer = None
growth_analyzer = None
demand_forecaster = None
staffing_estimator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global combo_optimizer, expansion_scorer, growth_analyzer, demand_forecaster, staffing_estimator

    print("[C.O.C.O.] Loading models...")
    try:
        from models.combo_optimizer import ComboOptimizer
        combo_optimizer = ComboOptimizer.load()
        print("[C.O.C.O.] Combo Optimizer loaded")
    except Exception as e:
        print(f"[C.O.C.O.] Combo Optimizer failed: {e}")

    try:
        from models.expansion_scorer import ExpansionScorer
        expansion_scorer = ExpansionScorer.load()
        print("[C.O.C.O.] Expansion Scorer loaded")
    except Exception as e:
        print(f"[C.O.C.O.] Expansion Scorer failed: {e}")

    try:
        from models.growth_strategy import GrowthStrategyAnalyzer
        growth_analyzer = GrowthStrategyAnalyzer.load()
        print("[C.O.C.O.] Growth Strategy Analyzer loaded")
    except Exception as e:
        print(f"[C.O.C.O.] Growth Strategy Analyzer failed: {e}")

    try:
        from models.demand_forecaster import DemandForecaster
        demand_forecaster = DemandForecaster.load()
        print(f"[C.O.C.O.] Demand Forecaster loaded ({demand_forecaster.model_name}, MAPE={demand_forecaster.mape:.1f}%)")
    except Exception as e:
        print(f"[C.O.C.O.] Demand Forecaster failed: {e}")

    try:
        from models.staffing_estimator import StaffingEstimator
        staffing_estimator = StaffingEstimator.load()
        print(f"[C.O.C.O.] Staffing Estimator loaded ({getattr(staffing_estimator, 'best_model_name', 'Unknown')})")
    except Exception as e:
        print(f"[C.O.C.O.] Staffing Estimator failed: {e}")

    print("[C.O.C.O.] All models loaded. Server ready.")
    yield
    print("[C.O.C.O.] Shutting down.")


app = FastAPI(
    title="Project C.O.C.O: Chief of Operations Conut Optimizer",
    description=(
        "AI-driven decision-support API for Conut bakery operations. "
        "Provides combo optimization, demand forecasting, expansion feasibility, "
        "staffing estimation, and coffee/milkshake growth strategy."
    ),
    version="1.0.0",
    lifespan=lifespan,
)
app = FastAPI(title="C.O.C.O. Agent Backend Gateway", lifespan=lifespan)

# Define exact ports OpenClaw and your SvelteKit dashboard use locally
# In production, this would be read from os.environ.get("ALLOWED_ORIGINS")
ALLOWED_ORIGINS = [
    "http://localhost:3000",  # OpenClaw frontend
    "http://localhost:5173",  # SvelteKit frontend
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(os.path.join(BASE_DIR, "static", "plots"), exist_ok=True)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "C.O.C.O. Agent Backend Gateway is running."}

# Health Check

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "Project C.O.C.O.",
        "models_loaded": {
            "combo_optimizer": combo_optimizer is not None,
            "expansion_scorer": expansion_scorer is not None,
            "growth_analyzer": growth_analyzer is not None,
        },
    }


# Combo Optimization

@app.post("/tools/get_combos", response_model=ComboResponse)
def get_combos(req: ComboRequest):
    """
    Get product combo recommendations using graph-based co-purchase analysis.
    Uses Louvain community detection to find natural product groupings.
    """
    try:
        if combo_optimizer is None:
            raise RuntimeError("Combo optimizer not loaded")
        result = combo_optimizer.predict(req.target_item, req.top_n)
        return result
    except Exception as e:
        # 1. Log the dangerous raw error internally
        logging.error(f"Combo prediction failed: {str(e)}")
        
        # 2. Return a SAFE error to the external world
        return ComboResponse(
            target_item=req.target_item,
            recommended_combo="CLASSIC CHIMNEY",
            confidence_weight="N/A (fallback)",
            business_reason="Default recommendation based on top-selling item.",
            error="Internal model evaluation error. Proceeding with fallback.",
        )


@app.get("/tools/combo_stats")
def combo_stats():
    """Return graph statistics and community overview."""
    try:
        if combo_optimizer is None:
            raise RuntimeError("Combo optimizer not loaded")
        return {
            "graph_stats": combo_optimizer.get_graph_stats(),
            "communities": combo_optimizer.get_all_communities(),
        }
    except Exception as e:
        logging.error(f"Combo stats failed: {str(e)}")
        return {"error": "Internal server error retrieving combo stats."}

# Demand Forecasting

@app.post("/tools/predict_demand", response_model=DemandResponse)
def predict_demand(req: DemandRequest):
    """
    Predict demand for a branch in a given month.
    Uses the MLFlow-tracked GPR/Bayesian model with contextual features.
    """
    try:
        if demand_forecaster is not None:
            result = demand_forecaster.predict(req.branch_name, req.month, req.year)
            return DemandResponse(**result)
    except Exception as e:
        logging.error(f"DemandForecaster inference failed: {str(e)}")

    # Fallback if model is not loaded or fails
    predicted_volume = 1250.0
    best_model_mape = 15.0
    error_margin = predicted_volume * (best_model_mape / 100)
    return DemandResponse(
        branch=req.branch_name,
        predicted_volume=predicted_volume,
        confidence_interval=f"{predicted_volume - error_margin:,.0f} to {predicted_volume + error_margin:,.0f}",
        mape=best_model_mape,
        warning="Using fallback. DemandForecaster model not loaded.",
        month=req.month,
        year=req.year,
        xai_drivers={"status": "fallback"},
        model_type="fallback",
    )

# Expansion Feasibility

@app.post("/tools/expansion_feasibility", response_model=ExpansionResponse)
def expansion_feasibility(req: ExpansionRequest):
    """
    Score expansion candidates against top-performing branch using cosine similarity.
    """
    try:
        if expansion_scorer is None:
            raise RuntimeError("Expansion scorer not loaded")

        result = expansion_scorer.score(
            candidate_branch=req.candidate_branch,
            candidate_features=req.candidate_features,
            lat=req.candidate_lat,
            lon=req.candidate_lon,
        )
        if "error" in result:
            raise ValueError(result["error"])
        return result
    except Exception as e:
        logging.error(f"Expansion feasibility failed: {str(e)}")
        return ExpansionResponse(
            reference_branch="Conut Jnah",
            candidate=req.candidate_branch or "Unknown",
            similarity_score=0.0,
            recommendation="Internal model evaluation error. Manual evaluation needed.",
            reference_profile={},
            candidate_profile={},
            gaps={},
            error="Internal server error. Proceeding with fallback.",
        )


@app.get("/tools/branch_rankings")
def branch_rankings():
    """Rank all branches by similarity to the reference branch."""
    try:
        if expansion_scorer is None:
            raise RuntimeError("Expansion scorer not loaded")
        return {"rankings": expansion_scorer.rank_all_branches()}
    except Exception as e:
        logging.error(f"Branch rankings failed: {str(e)}")
        return {"error": "Internal server error retrieving branch rankings."}


# Staffing Estimation


@app.post("/tools/estimate_staffing", response_model=StaffingResponse)
def estimate_staffing(req: StaffingRequest):
    """
    Estimate required staffing based on predicted demand using the trained model.
    """
    try:
        if staffing_estimator is None:
            raise RuntimeError("Staffing estimator not loaded")

        result = staffing_estimator.predict(
            branch_name=req.branch_name,
            predicted_volume=req.predicted_volume,
            date=req.date
        )
        return StaffingResponse(**result)
    except Exception as e:
        logging.error(f"Staffing estimation failed: {str(e)}")
        # Fallback
        volume = req.predicted_volume or 1250
        return StaffingResponse(
            branch=req.branch_name,
            predicted_volume=volume,
            recommended_staff=max(1, int(volume / 200)),
            throughput_metric=200,
            xai_drivers={"demand_level": "moderate", "historical_avg": "5 staff", "error": str(e)},
            model_type="fallback triggered",
        )


# Coffee and Milkshake Growth Strategy

@app.post("/tools/growth_strategy", response_model=GrowthResponse)
def growth_strategy(req: GrowthRequest):
    """
    Analyze coffee and milkshake performance and generate growth interventions.
    """
    try:
        if growth_analyzer is None:
            raise RuntimeError("Growth strategy analyzer not loaded")

        if req.branch_name:
            result = growth_analyzer.get_strategy(req.branch_name)
        else:
            # Return first branch if none specified
            result = growth_analyzer.get_strategy()
            if isinstance(result, list):
                result = result[0] if result else {"branch": "Unknown", "error": "No data"}

        return result
    except Exception as e:
        logging.error(f"Growth strategy failed: {str(e)}")
        return GrowthResponse(
            branch=req.branch_name or "Unknown",
            error="Internal model evaluation error. Proceeding with fallback.",
        )



# OpenClaw Skills Manifest
@app.get("/skills")
async def skills():
    """
    OpenClaw Skills Manifest. Describes all available tools for the AI agent to invoke.
    """
    return {
        "name": "Project C.O.C.O.",
        "description": "AI-driven Chief of Operations Agent for Conut bakery",
        "tools": [
            {
                "name": "get_combos",
                "description": "Get product combo recommendations based on co-purchase graph analysis. Returns the best item to bundle with a given menu item.",
                "endpoint": "/tools/get_combos",
                "method": "POST",
                "parameters": {
                    "target_item": {"type": "string", "description": "Menu item name (e.g., 'CHIMNEY THE ONE')", "required": True},
                    "top_n": {"type": "integer", "description": "Number of recommendations", "required": False, "default": 5},
                },
                "example_response": {
                    "target_item": "CHIMNEY THE ONE",
                    "recommended_combo": "CLASSIC CHIMNEY",
                    "confidence_weight": "28%",
                    "business_reason": "Co-purchased 15 times. Same product community.",
                },
            },
            {
                "name": "predict_demand",
                "description": "Forecast demand volume for a branch in a given month. Returns predicted sales volume with XAI drivers.",
                "endpoint": "/tools/predict_demand",
                "method": "POST",
                "parameters": {
                    "branch_name": {"type": "string", "description": "Branch name", "required": True},
                    "month": {"type": "integer", "description": "Month number (1-12)", "required": True},
                    "year": {"type": "integer", "description": "Year", "required": False, "default": 2026},
                },
            },
            {
                "name": "expansion_feasibility",
                "description": "Evaluate whether a new branch location is feasible by comparing its profile against the top-performing branch using cosine similarity.",
                "endpoint": "/tools/expansion_feasibility",
                "method": "POST",
                "parameters": {
                    "candidate_branch": {"type": "string", "description": "Existing branch to evaluate", "required": False},
                    "candidate_features": {"type": "object", "description": "Custom feature dict for a new location", "required": False},
                },
            },
            {
                "name": "estimate_staffing",
                "description": "Estimate required employees per shift based on predicted demand volume.",
                "endpoint": "/tools/estimate_staffing",
                "method": "POST",
                "parameters": {
                    "branch_name": {"type": "string", "description": "Branch name", "required": True},
                    "predicted_volume": {"type": "number", "description": "Predicted demand volume", "required": False},
                },
            },
            {
                "name": "growth_strategy",
                "description": "Analyze coffee and milkshake sales at a branch and generate targeted marketing interventions to boost low-performing categories.",
                "endpoint": "/tools/growth_strategy",
                "method": "POST",
                "parameters": {
                    "branch_name": {"type": "string", "description": "Branch to analyze. Omit for all branches.", "required": False},
                },
            },
        ],
    }


# ─────────────────────────────────────────────
# OpenClaw Interface (ReAct Agent Endpoint)
# ─────────────────────────────────────────────

from pydantic import BaseModel as PydanticBaseModel

from fastapi.responses import StreamingResponse

class MessageInput(PydanticBaseModel):
    role: str
    content: str

class OpenClawRequest(PydanticBaseModel):
    messages: list[MessageInput]
    gemini_api_key: str = None
    context: dict = {}

@app.post("/openclaw")
async def openclaw_endpoint(req: OpenClawRequest):
    """
    OpenClaw ingestion endpoint.
    Accepts a conversation history (messages array),
    runs the true LangGraph ReAct agent using Gemini,
    and streams back Server-Sent Events (SSE) for both
    tool traces and the final token response.
    """
    from agent.react_agent import stream_llm_react
    
    messages_data = [{"role": m.role, "content": m.content} for m in req.messages]
    
    return StreamingResponse(
        stream_llm_react(messages_data, req.gemini_api_key),
        media_type="text/event-stream"
    )

class DemandPlotRequest(PydanticBaseModel):
    branch_name: str
    historical_months: list[str]
    historical_sales: list[float]
    prediction_month: str
    mean_prediction: float
    lower_bound: float
    upper_bound: float

@app.post("/tools/generate_demand_confidence_plot")
async def generate_demand_confidence_plot(req: DemandPlotRequest):
    """Generates a line chart showing historical sales and a GPR prediction with confidence bounds."""
    fig, ax = plt.subplots(figsize=(7, 4))
    
    # Plot historical
    ax.plot(req.historical_months, req.historical_sales, marker='o', color='#3b82f6', label='Historical Sales')
    
    # Plot prediction point
    ax.plot([req.prediction_month], [req.mean_prediction], marker='*', markersize=12, color='#8b5cf6', label='GPR Forecast')
    
    # Connect last historical point to prediction
    if req.historical_months and req.historical_sales:
        ax.plot([req.historical_months[-1], req.prediction_month], 
                [req.historical_sales[-1], req.mean_prediction], 
                linestyle='--', color='#8b5cf6')
        
    # Plot confidence band (we mock the fill for the last point just for visual flavor)
    if req.historical_months:
        ax.fill_between([req.historical_months[-1], req.prediction_month],
                        [req.historical_sales[-1], req.lower_bound],
                        [req.historical_sales[-1], req.upper_bound],
                        color='#8b5cf6', alpha=0.2, label='95% Confidence Interval')

    ax.set_ylabel('LBP')
    ax.set_title(f'Demand Forecast & Uncertainty: {req.branch_name}')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x/1e9:.1f}B"))
    ax.legend(loc='upper left', fontsize='small')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plot_id = str(uuid.uuid4())[:8]
    filepath = f"static/plots/demand_{plot_id}.png"
    plt.savefig(os.path.join(BASE_DIR, filepath))
    plt.close()
    
    image_url = f"http://localhost:8000/{filepath}"
    markdown_image = f"![Demand Forecast for {req.branch_name}]({image_url})"
    return {"status": "success", "markdown_image": markdown_image}

class ComboPlotRequest(PydanticBaseModel):
    item_a: str
    item_b: str
    base_sales_a: float
    base_sales_b: float
    expected_lift_sales: float

@app.post("/tools/generate_combo_lift_plot")
async def generate_combo_lift_plot(req: ComboPlotRequest):
    """Generates a bar chart showing expected sales lift from bundling two items."""
    fig, ax = plt.subplots(figsize=(6, 4))
    categories = [f"{req.item_a}\n(Standalone)", f"{req.item_b}\n(Standalone)", "Bundled\n(Expected)"]
    values = [req.base_sales_a, req.base_sales_b, req.expected_lift_sales]
    colors = ['#cbd5e1', '#cbd5e1', '#10b981']
    
    bars = ax.bar(categories, values, color=colors)
    ax.set_ylabel('Sales Volume (Qty)')
    ax.set_title(f'Apriori Combo Lift:\n{req.item_a} + {req.item_b}')
    
    # Add value labels
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{int(bar.get_height())}',
                ha='center', va='bottom')
                
    plt.tight_layout()
    
    plot_id = str(uuid.uuid4())[:8]
    filepath = f"static/plots/combo_{plot_id}.png"
    plt.savefig(os.path.join(BASE_DIR, filepath))
    plt.close()
    
    image_url = f"http://localhost:8000/{filepath}"
    markdown_image = f"![Combo Lift Plot]({image_url})"
    return {"status": "success", "markdown_image": markdown_image}

class CoffeeGapRequest(PydanticBaseModel):
    branch_names: list[str]
    coffee_ratios: list[float]

@app.post("/tools/generate_coffee_gap_plot")
async def generate_coffee_gap_plot(req: CoffeeGapRequest):
    """Generates a horizontal bar chart of branch coffee ratio performance vs 20% target."""
    fig, ax = plt.subplots(figsize=(7, max(4, len(req.branch_names) * 0.5)))
    
    y_pos = range(len(req.branch_names))
    
    colors = ['#ef4444' if r < 0.20 else '#10b981' for r in req.coffee_ratios]
    ax.barh(y_pos, req.coffee_ratios, color=colors)
    
    # Target line
    ax.axvline(x=0.20, color='#f59e0b', linestyle='--', linewidth=2, label='20% Core Coffee Target')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(req.branch_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    ax.set_xlabel('Coffee Sales Ratio')
    ax.set_title('Coffee Expansion Gap Analysis (Target: 20%)')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: f"{x*100:.0f}%"))
    ax.legend(loc='lower right', fontsize='small')
    plt.tight_layout()
    
    plot_id = str(uuid.uuid4())[:8]
    filepath = f"static/plots/coffee_{plot_id}.png"
    plt.savefig(os.path.join(BASE_DIR, filepath))
    plt.close()
    
    image_url = f"http://localhost:8000/{filepath}"
    markdown_image = f"![Coffee Gap Plot]({image_url})"
    return {"status": "success", "markdown_image": markdown_image}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
