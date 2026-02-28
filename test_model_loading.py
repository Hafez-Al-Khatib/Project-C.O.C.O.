import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=== TESTING DemandForecaster.load() ===")
try:
    from models.demand_forecaster import DemandForecaster
    df = DemandForecaster.load()
    print(f"SUCCESS: model={df.model_name}, mape={df.mape}")
    r = df.predict("Conut Jnah", 11, 2023)
    print(f"Prediction: {r}")
except Exception as e:
    import traceback
    traceback.print_exc()

print("\n=== TESTING StaffingEstimator.load() ===")
try:
    from models.staffing_estimator import StaffingEstimator
    se = StaffingEstimator.load()
    print(f"SUCCESS: model={getattr(se, 'best_model_name', 'Unknown')}")
except Exception as e:
    import traceback
    traceback.print_exc()

print("\n=== TESTING ComboOptimizer.load() ===")
try:
    from models.combo_optimizer import ComboOptimizer
    co = ComboOptimizer.load()
    print(f"SUCCESS")
    r = co.predict("CHIMNEY CAKE", 3)
    print(f"Combo: {r}")
except Exception as e:
    import traceback
    traceback.print_exc()

print("\n=== TESTING ExpansionScorer.load() ===")
try:
    from models.expansion_scorer import ExpansionScorer
    es = ExpansionScorer.load()
    print(f"SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()

print("\n=== TESTING GrowthStrategyAnalyzer.load() ===")
try:
    from models.growth_strategy import GrowthStrategyAnalyzer
    gs = GrowthStrategyAnalyzer.load()
    print(f"SUCCESS")
except Exception as e:
    import traceback
    traceback.print_exc()
