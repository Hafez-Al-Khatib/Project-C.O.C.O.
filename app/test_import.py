"""Simulate the exact import path used by main.py's lifespan function."""
import os
import sys

# Mimic main.py's path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

print(f"BASE_DIR: {BASE_DIR}")
print(f"sys.path[0]: {sys.path[0]}")

# This is exactly what main.py does:
try:
    from models.demand_forecaster import DemandForecaster
    demand_forecaster = DemandForecaster.load()
    print(f"SUCCESS: {demand_forecaster.model_name}, MAPE={demand_forecaster.mape:.1f}%")
    r = demand_forecaster.predict("Conut Jnah", 1, 2026)
    print(f"Predicted: {r['predicted_volume']}")
except Exception as e:
    import traceback
    print(f"FAILED: {e}")
    traceback.print_exc()
