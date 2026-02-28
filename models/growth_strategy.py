"""
Objective 5 - Coffee & Milkshake Growth Strategy
Analyzes branch performance metrics for coffee and milkshake categories,
cross-references with top-selling item data, and packages the intelligence
for the OpenClaw LLM Strategist to generate hyper-specific interventions.
"""

import os
import pandas as pd
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "models")


COFFEE_DIVISIONS = [
    "Hot-Coffee Based", "Frappes", "CONUT'S FAVORITE", "CONUT''S FAVORITE"
]
SHAKE_DIVISIONS = ["Shakes"]
BEVERAGE_KEYWORDS = [
    "COFFEE", "LATTE", "ESPRESSO", "CAPPUCCINO", "MOCHA", "AMERICANO",
    "FRAPPE", "FLAT WHITE", "AFFOGATO", "MATCHA LATTE", "HOT CHOCOLATE"
]
SHAKE_KEYWORDS = [
    "MILKSHAKE", "SHAKE"
]


class GrowthStrategyAnalyzer:
    """Analyzes coffee and milkshake performance across branches."""

    def __init__(self):
        self.sales_df = None
        self.branch_coffee = None
        self.branch_shakes = None
        self.branch_totals = None

    def fit(self):
        """Load and analyze sales data."""
        self.sales_df = pd.read_parquet(os.path.join(CLEANED_DIR, "sales_by_item.parquet"))

        # Calculate per-branch totals
        self.branch_totals = self.sales_df.groupby("branch")["total_amount"].sum()

        # Coffee performance per branch
        coffee_mask = self.sales_df["division"].str.contains(
            "|".join(COFFEE_DIVISIONS), case=False, na=False
        )
        self.branch_coffee = self.sales_df[coffee_mask].groupby("branch").agg(
            coffee_revenue=("total_amount", "sum"),
            coffee_qty=("qty", "sum"),
            coffee_items=("item", "nunique"),
        )

        # Shake performance per branch
        shake_mask = self.sales_df["division"].str.contains(
            "|".join(SHAKE_DIVISIONS), case=False, na=False
        )
        self.branch_shakes = self.sales_df[shake_mask].groupby("branch").agg(
            shake_revenue=("total_amount", "sum"),
            shake_qty=("qty", "sum"),
            shake_items=("item", "nunique"),
        )

        print(f"[GrowthStrategy] Analyzed {len(self.branch_totals)} branches")
        return self

    def _get_branch_category_items(self, branch, category="coffee"):
        """Get detailed item breakdown for a branch in a category."""
        if category == "coffee":
            mask = self.sales_df["division"].str.contains(
                "|".join(COFFEE_DIVISIONS), case=False, na=False
            )
        else:
            mask = self.sales_df["division"].str.contains(
                "|".join(SHAKE_DIVISIONS), case=False, na=False
            )

        branch_mask = self.sales_df["branch"] == branch
        items = self.sales_df[mask & branch_mask].groupby("item").agg(
            qty=("qty", "sum"),
            revenue=("total_amount", "sum"),
        ).sort_values("revenue", ascending=False)

        return items.head(10).to_dict("index")

    def get_strategy(self, branch_name=None):
        """
        Generate growth strategy for a specific branch or all branches.
        Returns actionable interventions.
        """
        if self.sales_df is None:
            self.fit()

        if branch_name:
            return self._branch_strategy(branch_name)

        # Rank all branches
        results = []
        for branch in self.branch_totals.index:
            results.append(self._branch_strategy(branch))
        return results

    def _branch_strategy(self, branch):
        """Analyze branch data and package it for the LLM Strategist."""
        total = self.branch_totals.get(branch, 0)
        if total == 0:
            return {"branch": branch, "error": "No sales data found"}

        # Coffee metrics
        coffee_rev = self.branch_coffee.loc[branch, "coffee_revenue"] if branch in self.branch_coffee.index else 0
        coffee_ratio = coffee_rev / total if total > 0 else 0

        # Shake metrics
        shake_rev = self.branch_shakes.loc[branch, "shake_revenue"] if branch in self.branch_shakes.index else 0
        shake_ratio = shake_rev / total if total > 0 else 0

        # Absolute Industry Thresholds
        # A healthy cafe should have > 20% coffee revenue and > 10% shake revenue
        coffee_target = 0.20
        shake_target = 0.10

        coffee_gap = round(coffee_target - coffee_ratio, 3)  # positive = below target
        shake_gap = round(shake_target - shake_ratio, 3)

        # Rank within franchise for context (simple rank, not percentile)
        all_coffee_ratios = {
            b: (self.branch_coffee.loc[b, "coffee_revenue"] / self.branch_totals.get(b, 1))
            if b in self.branch_coffee.index else 0
            for b in self.branch_totals.index
        }
        coffee_rank = sorted(all_coffee_ratios, key=all_coffee_ratios.get).index(branch) + 1
        n_branches = len(self.branch_totals.index)

        # Get top items so OpenClaw can create real, specific combos
        top_coffees = list(self._get_branch_category_items(branch, "coffee").keys())[:3]
        top_shakes = list(self._get_branch_category_items(branch, "shakes").keys())[:3]

        return {
            "branch": branch,
            "metrics": {
                "total_revenue": round(total, 2),
                "coffee_revenue": round(coffee_rev, 2),
                "coffee_ratio_actual": round(coffee_ratio, 3),
                "coffee_ratio_target": coffee_target,
                "coffee_gap": coffee_gap,
                "shake_revenue": round(shake_rev, 2),
                "shake_ratio_actual": round(shake_ratio, 3),
                "shake_ratio_target": shake_target,
                "shake_gap": shake_gap,
            },
            "franchise_rank": {
                "coffee_rank": coffee_rank,
                "out_of": n_branches,
                "label": "Lowest" if coffee_rank == 1 else ("Highest" if coffee_rank == n_branches else f"#{coffee_rank}")
            },
            "best_selling_assets": {
                "top_coffees": top_coffees,
                "top_shakes": top_shakes,
            },
            "status": {
                "coffee_struggling": coffee_ratio < coffee_target,
                "shakes_struggling": shake_ratio < shake_target,
            }
        }

    def save(self, path=None):
        if path is None:
            path = os.path.join(MODELS_DIR, "growth_strategy.pkl")
        joblib.dump(self, path)
        print(f"[GrowthStrategy] Saved to {path}")

    @staticmethod
    def load(path=None):
        if path is None:
            path = os.path.join(MODELS_DIR, "growth_strategy.pkl")
        return joblib.load(path)


if __name__ == "__main__":
    analyzer = GrowthStrategyAnalyzer().fit()

    for branch in analyzer.branch_totals.index:
        result = analyzer._branch_strategy(branch)
        print(f"\n{'='*55}")
        print(f"Branch: {result.get('branch')}")
        if "error" in result:
            print(f"  ERROR: {result['error']}")
            continue
        m = result["metrics"]
        print(f"  Coffee: {m['coffee_ratio_actual']*100:.1f}% (target {m['coffee_ratio_target']*100:.0f}%, gap {m['coffee_gap']*100:+.1f}%)")
        print(f"  Shakes: {m['shake_ratio_actual']*100:.1f}% (target {m['shake_ratio_target']*100:.0f}%, gap {m['shake_gap']*100:+.1f}%)")
        print(f"  Franchise Coffee Rank: {result['franchise_rank']['label']} of {result['franchise_rank']['out_of']}")
        print(f"  Top Coffees: {result['best_selling_assets']['top_coffees']}")
        print(f"  Struggling: coffee={result['status']['coffee_struggling']}, shakes={result['status']['shakes_struggling']}")

    analyzer.save()
