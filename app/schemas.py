"""
Project C.O.C.O. - Pydantic Request/Response Schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


# Combo Optimization
class ComboRequest(BaseModel):
    target_item: str = Field(..., description="Menu item to find combo recommendations for")
    top_n: int = Field(5, description="Number of recommendations to return")


class ComboResponse(BaseModel):
    target_item: str
    recommended_combo: Optional[str] = None
    confidence_weight: str = "0%"
    business_reason: str = ""
    community_id: int = -1
    all_recommendations: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    available_items: Optional[List[str]] = None


# Expansion Feasibility
class ExpansionRequest(BaseModel):
    candidate_branch: Optional[str] = Field(None, description="Existing branch name to evaluate")
    candidate_features: Optional[Dict[str, float]] = Field(
        None, description="Custom feature vector for a new location"
    )
    candidate_lat: Optional[float] = Field(None, description="Latitude for the candidate location (e.g. 33.8966)")
    candidate_lon: Optional[float] = Field(None, description="Longitude for the candidate location (e.g. 35.4815)")



class ExpansionResponse(BaseModel):
    reference_branch: str
    candidate: str
    similarity_score: float
    recommendation: str
    reference_profile: Dict[str, Any]
    candidate_profile: Dict[str, Any]
    gaps: Dict[str, float]
    error: Optional[str] = None


# Growth Strategy
class GrowthRequest(BaseModel):
    branch_name: Optional[str] = Field(None, description="Branch to analyze. Omit for all branches.")

class GrowthResponse(BaseModel):
    branch: str
    metrics: Optional[Dict[str, Any]] = None
    franchise_rank: Optional[Dict[str, Any]] = None
    best_selling_assets: Optional[Dict[str, Any]] = None
    status: Optional[Dict[str, bool]] = None
    error: Optional[str] = None


# Demand Forecast (Objective 2)
class DemandRequest(BaseModel):
    branch_name: str = Field(..., description="Branch to predict demand for")
    month: int = Field(..., ge=1, le=12)
    year: int = Field(2026)


class DemandResponse(BaseModel):
    branch: str
    predicted_volume: float
    confidence_interval: Optional[str] = None  # e.g. "1,062 to 1,437"
    mape: Optional[float] = None               # From MLFlow best run
    warning: Optional[str] = None
    month: int
    year: int
    xai_drivers: Dict[str, str] = {}
    model_type: str = "production"



# ---- Staffing Estimation (Objective 4) ----

class StaffingRequest(BaseModel):
    branch_name: str = Field(..., description="Branch to estimate staffing for")
    predicted_volume: Optional[float] = Field(None, description="Predicted demand volume")
    date: Optional[str] = Field(None, description="Date for inference context (ISO format, e.g. 2026-11-15)")


class StaffingResponse(BaseModel):
    branch: str
    predicted_volume: float
    recommended_staff: int
    confidence_band: Optional[str] = None
    throughput_metric: float = 0
    xai_drivers: Dict[str, str] = {}
    model_type: str = "production"
