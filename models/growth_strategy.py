"""
Objective 5 — Coffee & Milkshake Growth Strategy
=================================================
Identifies branches struggling with coffee and milkshake sales,
cross-references with combo data, and generates targeted
marketing interventions.
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
        """Generate strategy for a single branch."""
        total = self.branch_totals.get(branch, 0)
        if total == 0:
            return {"branch": branch, "error": "No sales data found"}

        # Coffee metrics
        coffee_rev = self.branch_coffee.loc[branch, "coffee_revenue"] if branch in self.branch_coffee.index else 0
        coffee_qty = self.branch_coffee.loc[branch, "coffee_qty"] if branch in self.branch_coffee.index else 0
        coffee_ratio = coffee_rev / total if total > 0 else 0

        # Shake metrics
        shake_rev = self.branch_shakes.loc[branch, "shake_revenue"] if branch in self.branch_shakes.index else 0
        shake_qty = self.branch_shakes.loc[branch, "shake_qty"] if branch in self.branch_shakes.index else 0
        shake_ratio = shake_rev / total if total > 0 else 0

        # Calculate percentile rank among branches
        all_coffee_ratios = []
        all_shake_ratios = []
        for b in self.branch_totals.index:
            bt = self.branch_totals.get(b, 0)
            cr = self.branch_coffee.loc[b, "coffee_revenue"] / bt if (b in self.branch_coffee.index and bt > 0) else 0
            sr = self.branch_shakes.loc[b, "shake_revenue"] / bt if (b in self.branch_shakes.index and bt > 0) else 0
            all_coffee_ratios.append(cr)
            all_shake_ratios.append(sr)

        coffee_percentile = sum(1 for r in all_coffee_ratios if r <= coffee_ratio) / len(all_coffee_ratios)
        shake_percentile = sum(1 for r in all_shake_ratios if r <= shake_ratio) / len(all_shake_ratios)

        # Generate interventions
        interventions = []

        if coffee_percentile <= 0.25:
            interventions.append({
                "category": "Coffee",
                "severity": "HIGH",
                "finding": f"Coffee contributes only {coffee_ratio*100:.1f}% of revenue — bottom quartile among branches.",
                "action": f"Implement a morning coffee combo bundle. Target pastry buyers who currently skip coffee.",
            })
        elif coffee_percentile <= 0.50:
            interventions.append({
                "category": "Coffee",
                "severity": "MEDIUM",
                "finding": f"Coffee at {coffee_ratio*100:.1f}% of revenue — below median.",
                "action": "Introduce loyalty stamps for coffee purchases. Cross-promote with chimney cake combos.",
            })

        if shake_percentile <= 0.25:
            interventions.append({
                "category": "Milkshakes",
                "severity": "HIGH",
                "finding": f"Milkshakes contribute only {shake_ratio*100:.1f}% of revenue — bottom quartile.",
                "action": "Launch seasonal milkshake flavors. Bundle with ice cream bowl at discounted price.",
            })
        elif shake_percentile <= 0.50:
            interventions.append({
                "category": "Milkshakes",
                "severity": "MEDIUM",
                "finding": f"Milkshakes at {shake_ratio*100:.1f}% of revenue — below median.",
                "action": "Feature milkshakes prominently in menu boards. Introduce afternoon milkshake happy hour.",
            })

        if not interventions:
            interventions.append({
                "category": "Overall",
                "severity": "LOW",
                "finding": "Branch performs above median in both coffee and milkshakes.",
                "action": "Maintain current strategy. Consider premium upsells.",
            })

        return {
            "branch": branch,
            "total_revenue": total,
            "coffee_revenue": coffee_rev,
            "coffee_share": f"{coffee_ratio*100:.1f}%",
            "coffee_percentile": f"{coffee_percentile*100:.0f}th",
            "coffee_qty": coffee_qty,
            "shake_revenue": shake_rev,
            "shake_share": f"{shake_ratio*100:.1f}%",
            "shake_percentile": f"{shake_percentile*100:.0f}th",
            "shake_qty": shake_qty,
            "interventions": interventions,
            "top_coffee_items": self._get_branch_category_items(branch, "coffee"),
            "top_shake_items": self._get_branch_category_items(branch, "shakes"),
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
        print(f"\n{'='*50}")
        print(f"Branch: {result['branch']}")
        print(f"  Coffee: {result['coffee_share']} ({result['coffee_percentile']} percentile)")
        print(f"  Shakes: {result['shake_share']} ({result['shake_percentile']} percentile)")
        for inv in result["interventions"]:
            print(f"  [{inv['severity']}] {inv['category']}: {inv['action']}")

    analyzer.save()
