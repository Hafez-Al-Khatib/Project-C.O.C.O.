"""
Objective 3 — Expansion Feasibility (Spatial Signature Scoring)
===============================================================
Profiles branch performance and uses cosine similarity to score
potential expansion locations against the best-performing branch.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def build_branch_profiles():
    """
    Build a feature vector for each branch based on:
    - Total revenue
    - Product mix ratios (coffee %, pastry %, drinks %, shakes %)
    - Order volume
    - Revenue per month
    """
    sales_df = pd.read_parquet(os.path.join(CLEANED_DIR, "sales_by_item.parquet"))
    monthly_df = pd.read_parquet(os.path.join(CLEANED_DIR, "monthly_sales.parquet"))

    profiles = {}

    for branch in sales_df["branch"].dropna().unique():
        branch_data = sales_df[sales_df["branch"] == branch]

        # Aggregate by division
        div_totals = branch_data.groupby("division")["total_amount"].sum()
        total_revenue = div_totals.sum()

        if total_revenue == 0:
            continue

        # Key category ratios
        coffee_rev = 0
        pastry_rev = 0
        drinks_rev = 0
        shakes_rev = 0

        for div, amount in div_totals.items():
            div_lower = str(div).lower() if div else ""
            if "coffee" in div_lower or "frappes" in div_lower:
                coffee_rev += amount
            elif "item" in div_lower:
                pastry_rev += amount
            elif "hot and cold" in div_lower or "tea" in div_lower:
                drinks_rev += amount
            elif "shake" in div_lower:
                shakes_rev += amount

        # Monthly sales data
        branch_monthly = monthly_df[monthly_df["branch"].str.contains(branch.split(" - ")[0], case=False, na=False)]
        avg_monthly = branch_monthly["total_sales"].mean() if len(branch_monthly) > 0 else 0
        n_months = len(branch_monthly)

        profiles[branch] = {
            "total_revenue": total_revenue,
            "coffee_ratio": coffee_rev / total_revenue if total_revenue > 0 else 0,
            "pastry_ratio": pastry_rev / total_revenue if total_revenue > 0 else 0,
            "drinks_ratio": drinks_rev / total_revenue if total_revenue > 0 else 0,
            "shakes_ratio": shakes_rev / total_revenue if total_revenue > 0 else 0,
            "avg_monthly_revenue": avg_monthly,
            "n_months_active": n_months,
            "n_items_sold": branch_data["item"].nunique(),
            "total_qty": branch_data["qty"].sum(),
        }

    return pd.DataFrame(profiles).T


class ExpansionScorer:
    """Scores expansion candidates against the best-performing branch profile."""

    def __init__(self, reference_branch="Conut Jnah"):
        self.reference_branch = reference_branch
        self.profiles_df = None
        self.feature_cols = [
            "coffee_ratio", "pastry_ratio", "drinks_ratio", "shakes_ratio",
            "n_items_sold",
        ]

    def fit(self):
        """Build branch profiles and set reference."""
        self.profiles_df = build_branch_profiles()
        print(f"[ExpansionScorer] Built profiles for {len(self.profiles_df)} branches")
        print(f"[ExpansionScorer] Reference branch: {self.reference_branch}")

        # Find the actual best branch by revenue if reference not found
        if self.reference_branch not in self.profiles_df.index:
            self.reference_branch = self.profiles_df["total_revenue"].idxmax()
            print(f"[ExpansionScorer] Auto-selected reference: {self.reference_branch}")

        return self

    def score(self, candidate_branch=None, candidate_features=None):
        """
        Score a candidate location against the reference branch.
        If candidate_branch is provided, use existing branch data.
        If candidate_features is a dict, create a synthetic profile.
        """
        if self.profiles_df is None:
            self.fit()

        ref_vector = self.profiles_df.loc[
            self.reference_branch, self.feature_cols
        ].values.reshape(1, -1)

        if candidate_branch and candidate_branch in self.profiles_df.index:
            cand_vector = self.profiles_df.loc[
                candidate_branch, self.feature_cols
            ].values.reshape(1, -1)
            cand_profile = self.profiles_df.loc[candidate_branch].to_dict()
        elif candidate_features:
            cand_vector = np.array(
                [candidate_features.get(c, 0) for c in self.feature_cols]
            ).reshape(1, -1)
            cand_profile = candidate_features
        else:
            return {"error": "Provide either candidate_branch or candidate_features"}

        similarity = cosine_similarity(ref_vector, cand_vector)[0][0]

        # Generate recommendation
        if similarity >= 0.9:
            recommendation = "STRONG — Profile closely matches our best branch. High expansion potential."
        elif similarity >= 0.7:
            recommendation = "MODERATE — Reasonable match. Consider adjusting product mix to align with winning formula."
        elif similarity >= 0.5:
            recommendation = "CAUTIOUS — Significant differences from top performer. Targeted menu adaptation needed."
        else:
            recommendation = "WEAK — Very different profile. High risk without major operational changes."

        ref_profile = self.profiles_df.loc[self.reference_branch].to_dict()

        return {
            "reference_branch": self.reference_branch,
            "candidate": candidate_branch or "Custom Profile",
            "similarity_score": round(similarity, 4),
            "recommendation": recommendation,
            "reference_profile": {
                "total_revenue": ref_profile["total_revenue"],
                "coffee_ratio": round(ref_profile["coffee_ratio"], 3),
                "pastry_ratio": round(ref_profile["pastry_ratio"], 3),
                "shakes_ratio": round(ref_profile["shakes_ratio"], 3),
            },
            "candidate_profile": {
                k: round(v, 3) if isinstance(v, float) else v
                for k, v in cand_profile.items()
            },
            "gaps": {
                col: round(ref_profile[col] - cand_profile.get(col, 0), 3)
                for col in self.feature_cols
            },
        }

    def rank_all_branches(self):
        """Rank all branches by similarity to reference."""
        results = []
        for branch in self.profiles_df.index:
            if branch == self.reference_branch:
                continue
            result = self.score(candidate_branch=branch)
            results.append(result)
        results.sort(key=lambda x: x["similarity_score"], reverse=True)
        return results

    def save(self, path=None):
        if path is None:
            path = os.path.join(MODELS_DIR, "expansion_scorer.pkl")
        joblib.dump(self, path)
        print(f"[ExpansionScorer] Saved to {path}")

    @staticmethod
    def load(path=None):
        if path is None:
            path = os.path.join(MODELS_DIR, "expansion_scorer.pkl")
        return joblib.load(path)


if __name__ == "__main__":
    scorer = ExpansionScorer().fit()
    print("\nBranch profiles:")
    print(scorer.profiles_df.to_string())

    print("\nAll branch rankings:")
    for r in scorer.rank_all_branches():
        print(f"  {r['candidate']}: similarity={r['similarity_score']}, {r['recommendation']}")

    scorer.save()
