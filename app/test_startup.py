"""Reproduce the exact Uvicorn startup loading of demand_forecaster."""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

# Exactly as in main.py lifespan
demand_forecaster = None

try:
    from models.demand_forecaster import DemandForecaster
    demand_forecaster = DemandForecaster.load()
    print(f"[OK] Demand Forecaster loaded ({demand_forecaster.model_name}, MAPE={demand_forecaster.mape:.1f}%)")
except Exception as e:
    import traceback
    print(f"[FAIL] Demand Forecaster failed: {e}")
    traceback.print_exc()

# Check path
import models.demand_forecaster as dm
print(f"\nModels dir: {dm.MODELS_DIR}")
print(f"PKL path: {os.path.join(dm.MODELS_DIR, 'demand_forecaster.pkl')}")
print(f"Exists: {os.path.exists(os.path.join(dm.MODELS_DIR, 'demand_forecaster.pkl'))}")

# Simulate the endpoint
if demand_forecaster is not None:
    result = demand_forecaster.predict("Conut Jnah", 1, 2026)
    print(f"\n[ENDPOINT WOULD RETURN] Volume: {result['predicted_volume']}, Model: {result['model_type']}")
else:
    print("\n[ENDPOINT WOULD RETURN] FALLBACK: demand_forecaster is None!")
