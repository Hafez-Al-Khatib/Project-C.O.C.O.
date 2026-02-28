import sys
sys.path.insert(0, '.')
from models.demand_forecaster import DemandForecaster

try:
    df = DemandForecaster.load()
    print(f"Model: {df.model_name}, MAPE: {df.mape}")
    r = df.predict("Conut Jnah", 1, 2026)
    print(f"Prediction: {r['predicted_volume']}")
    print(f"CI: {r['confidence_interval']}")
    print(f"Model type: {r['model_type']}")
    print(f"Warning: {r.get('warning', 'NONE')}")
except Exception as e:
    import traceback
    traceback.print_exc()
