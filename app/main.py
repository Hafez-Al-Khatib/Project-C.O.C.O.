"""
Project C.O.C.O. — FastAPI Application
========================================
Agent Gateway serving as the Private Skills Registry for OpenClaw.
Every endpoint has graceful error fallbacks.
"""

import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global combo_optimizer, expansion_scorer, growth_analyzer

    print("[C.O.C.O.] Loading models...")
    try:
        from models.combo_optimizer import ComboOptimizer
        combo_optimizer = ComboOptimizer.load()
        print("[C.O.C.O.] ✓ Combo Optimizer loaded")
    except Exception as e:
        print(f"[C.O.C.O.] ✗ Combo Optimizer failed: {e}")

    try:
        from models.expansion_scorer import ExpansionScorer
        expansion_scorer = ExpansionScorer.load()
        print("[C.O.C.O.] ✓ Expansion Scorer loaded")
    except Exception as e:
        print(f"[C.O.C.O.] ✗ Expansion Scorer failed: {e}")

    try:
        from models.growth_strategy import GrowthStrategyAnalyzer
        growth_analyzer = GrowthStrategyAnalyzer.load()
        print("[C.O.C.O.] ✓ Growth Strategy Analyzer loaded")
    except Exception as e:
        print(f"[C.O.C.O.] ✗ Growth Strategy Analyzer failed: {e}")

    print("[C.O.C.O.] All models loaded. Server ready.")
    yield
    print("[C.O.C.O.] Shutting down.")


app = FastAPI(
    title="Project C.O.C.O. — Chief of Operations Conut Optimizer",
    description=(
        "AI-driven decision-support API for Conut bakery operations. "
        "Provides combo optimization, demand forecasting, expansion feasibility, "
        "staffing estimation, and coffee/milkshake growth strategy."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HEALTH CHECK
# ============================================================================

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


# ============================================================================
# OBJECTIVE 1: COMBO OPTIMIZATION
# ============================================================================

@app.post("/tools/get_combos", response_model=ComboResponse)
async def get_combos(req: ComboRequest):
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
        # Graceful fallback: return a default recommendation
        return ComboResponse(
            target_item=req.target_item,
            recommended_combo="CLASSIC CHIMNEY",
            confidence_weight="N/A (fallback)",
            business_reason=f"Model unavailable ({str(e)}). Default recommendation based on top-selling item.",
            error=str(e),
        )


@app.get("/tools/combo_stats")
async def combo_stats():
    """Return graph statistics and community overview."""
    try:
        if combo_optimizer is None:
            raise RuntimeError("Combo optimizer not loaded")
        return {
            "graph_stats": combo_optimizer.get_graph_stats(),
            "communities": combo_optimizer.get_all_communities(),
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# OBJECTIVE 2: DEMAND FORECASTING (STUB — Modeling Duo)
# ============================================================================

@app.post("/tools/predict_demand", response_model=DemandResponse)
async def predict_demand(req: DemandRequest):
    """
    Predict demand for a branch in a given month.
    NOTE: This is a stub endpoint. The Modeling Duo will replace with
    XGBoost-based predictions.
    """
    # Stub: return data-contract mock output
    return DemandResponse(
        branch=req.branch_name,
        predicted_volume=1250,
        month=req.month,
        year=req.year,
        xai_drivers={"weekend": "45%", "prev_day_sales": "30%"},
        model_type="stub — awaiting Modeling Duo XGBoost implementation",
    )


# ============================================================================
# OBJECTIVE 3: EXPANSION FEASIBILITY
# ============================================================================

@app.post("/tools/expansion_feasibility", response_model=ExpansionResponse)
async def expansion_feasibility(req: ExpansionRequest):
    """
    Score expansion candidates against top-performing branch using cosine similarity.
    """
    try:
        if expansion_scorer is None:
            raise RuntimeError("Expansion scorer not loaded")

        result = expansion_scorer.score(
            candidate_branch=req.candidate_branch,
            candidate_features=req.candidate_features,
        )
        if "error" in result:
            raise ValueError(result["error"])
        return result
    except Exception as e:
        return ExpansionResponse(
            reference_branch="Conut Jnah",
            candidate=req.candidate_branch or "Unknown",
            similarity_score=0.0,
            recommendation=f"Model unavailable ({str(e)}). Manual evaluation needed.",
            reference_profile={},
            candidate_profile={},
            gaps={},
            error=str(e),
        )


@app.get("/tools/branch_rankings")
async def branch_rankings():
    """Rank all branches by similarity to the reference branch."""
    try:
        if expansion_scorer is None:
            raise RuntimeError("Expansion scorer not loaded")
        return {"rankings": expansion_scorer.rank_all_branches()}
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# OBJECTIVE 4: STAFFING ESTIMATION (STUB — Modeling Duo)
# ============================================================================

@app.post("/tools/estimate_staffing", response_model=StaffingResponse)
async def estimate_staffing(req: StaffingRequest):
    """
    Estimate required staffing based on predicted demand.
    NOTE: This is a stub endpoint. The Modeling Duo will replace with
    real throughput calculations.
    """
    volume = req.predicted_volume or 1250
    return StaffingResponse(
        branch=req.branch_name,
        predicted_volume=volume,
        recommended_staff=max(1, int(volume / 200)),
        throughput_metric=200,
        xai_drivers={"demand_level": "moderate", "historical_avg": "5 staff"},
        model_type="stub — awaiting Modeling Duo implementation",
    )


# ============================================================================
# OBJECTIVE 5: COFFEE & MILKSHAKE GROWTH STRATEGY
# ============================================================================

@app.post("/tools/growth_strategy", response_model=GrowthResponse)
async def growth_strategy(req: GrowthRequest):
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
        return GrowthResponse(
            branch=req.branch_name or "Unknown",
            error=str(e),
            interventions=[{
                "category": "General",
                "severity": "N/A",
                "finding": "Model unavailable.",
                "action": "Review coffee and milkshake sales manually.",
            }],
        )


# ============================================================================
# OPENCLAW SKILLS MANIFEST
# ============================================================================

@app.get("/skills")
async def skills():
    """
    OpenClaw Skills Manifest — describes all available tools
    for the AI agent to invoke.
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


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
