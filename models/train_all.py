"""
Project C.O.C.O. — Train All Models
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
    print("  Project C.O.C.O. — Model Training")
    print("=" * 60)

    print("\n[1/3] Training Combo Optimizer (Objective 1)...")
    combo = ComboOptimizer().fit()
    combo.save()

    print("\n[2/3] Training Expansion Scorer (Objective 3)...")
    expansion = ExpansionScorer().fit()
    expansion.save()

    print("\n[3/3] Training Growth Strategy Analyzer (Objective 5)...")
    growth = GrowthStrategyAnalyzer().fit()
    growth.save()

    print("\n" + "=" * 60)
    print("  All models trained and saved to ./models/")
    print("=" * 60)


if __name__ == "__main__":
    main()
