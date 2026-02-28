"""
Project C.O.C.O. - Train All Models
====================================
Fits and saves all analytical models.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from models.combo_optimizer import ComboOptimizer
from models.expansion_scorer import ExpansionScorer
from models.growth_strategy import GrowthStrategyAnalyzer


def main():
    print("=" * 60)
    print("  Project C.O.C.O. - Model Training")
    print("=" * 60)

    print("\n[1/4] Training Combo Optimizer (Objective 1)...")
    combo = ComboOptimizer().fit()
    combo.save()

    print("\n[2/4] Training Expansion Scorer (Objective 3)...")
    expansion = ExpansionScorer().fit()
    expansion.save()

    print("\n[3/4] Training Growth Strategy Analyzer (Objective 5)...")
    growth = GrowthStrategyAnalyzer().fit()
    growth.save()

    print("\n[4/4] Training Demand Forecaster (Objective 2) with MLFlow...")
    try:
        from models.demand_forecaster import (
            load_and_engineer_features, run_mlflow_experiment, DemandForecaster
        )
        df = load_and_engineer_features()
        best_name, best_info, _ = run_mlflow_experiment(df)
        forecaster = DemandForecaster()
        forecaster.fit(df, best_info["model"], best_name,
                       best_info["uses_log"], best_info["mape"])
        forecaster.save()
    except Exception as e:
        print(f"  [WARN] DemandForecaster training failed: {e}")

    print("\n" + "=" * 60)
    print("  All models trained and saved to ./models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
