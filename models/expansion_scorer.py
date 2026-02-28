"""
Objective 3 — Expansion Feasibility (Spatial Signature Scoring)
Profiles branch performance and uses cosine similarity to score
potential expansion locations against the best-performing branch.

V2 Upgrades: Live OSM (OpenStreetMap) API integration for foot traffic,
commercial density, and university proximity to replace hardcoded proxies.
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CACHE_FILE = os.path.join(MODELS_DIR, "osm_cache.json")

# Approximate coordinates for the branches
BRANCH_COORDS = {
    "Conut Jnah": {"lat": 33.8646, "lon": 35.4852},
    "Main Street Coffee": {"lat": 33.8966, "lon": 35.4815},
    "Conut - Tyre": {"lat": 33.2721, "lon": 35.1966},
    "Conut Main": {"lat": 33.8825, "lon": 35.4930},
}


class OSMClient:
    """Fetches and caches Points of Interest from OpenStreetMap Overpass API."""
    
    def __init__(self, radius=1000):
        self.radius = radius
        self.cache = self._load_cache()
        
    def _load_cache(self):
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}
        
    def _save_cache(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(CACHE_FILE, "w") as f:
            json.dump(self.cache, f)

    def get_spatial_features(self, lat, lon):
        cache_key = f"{round(lat, 3)}_{round(lon, 3)}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        print(f"[OSMClient] Fetching live OSM data for {lat}, {lon}...")
        overpass_url = "http://overpass-api.de/api/interpreter"
        query = f'''
        [out:json];
        (
          node["amenity"](around:{self.radius},{lat},{lon});
          node["shop"](around:{self.radius},{lat},{lon});
          node["office"](around:{self.radius},{lat},{lon});
        );
        out body;
        '''

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(overpass_url, params={'data': query}, timeout=15)
                if response.status_code == 200:
                    elements = response.json().get('elements', [])

                    foot_traffic_tags = ['restaurant', 'cafe', 'fast_food', 'mall', 'supermarket', 'convenience', 'clothes']
                    commercial_tags = ['office', 'bank', 'company']
                    uni_tags = ['university', 'college']

                    ft_count = 0
                    com_count = 0
                    uni_count = 0

                    for e in elements:
                        tags = e.get('tags', {})
                        amenity = tags.get('amenity', '')
                        shop = tags.get('shop', '')
                        office = tags.get('office', '')

                        if amenity in foot_traffic_tags or shop in foot_traffic_tags:
                            ft_count += 1
                        if amenity in commercial_tags or office:
                            com_count += 1
                        if amenity in uni_tags:
                            uni_count += 1

                    features = {
                        "foot_traffic_index": min(1.0, ft_count / 50.0),
                        "commercial_density": min(1.0, com_count / 30.0),
                        "university_proximity": min(1.0, uni_count / 3.0),
                        "raw_ft_count": ft_count,
                        "raw_com_count": com_count,
                        "raw_uni_count": uni_count,
                        "osm_status": "live"
                    }

                    self.cache[cache_key] = features
                    self._save_cache()
                    time.sleep(1)
                    return features
                else:
                    print(f"[OSMClient] HTTP {response.status_code} on attempt {attempt + 1}/{max_retries + 1}")
            except Exception as e:
                print(f"[OSMClient] Attempt {attempt + 1}/{max_retries + 1} failed: {e}")

            if attempt < max_retries:
                backoff = 2 ** attempt
                print(f"[OSMClient] Retrying in {backoff}s...")
                time.sleep(backoff)

        print("[OSMClient] Circuit breaker triggered — all retries exhausted, using fallback values.")
        return {
            "foot_traffic_index": 0.5,
            "commercial_density": 0.5,
            "university_proximity": 0.1,
            "raw_ft_count": 0,
            "raw_com_count": 0,
            "raw_uni_count": 0,
            "osm_status": "fallback (API unreachable after retries)"
        }


def build_branch_profiles():
    """
    Build a feature vector for each branch based on Revenue, Product Mix ratios, and Spatial Features (OSM).
    """
    sales_df = pd.read_parquet(os.path.join(CLEANED_DIR, "sales_by_item.parquet"))
    monthly_df = pd.read_parquet(os.path.join(CLEANED_DIR, "monthly_sales.parquet"))
    
    osm_client = OSMClient()
    profiles = {}

    for branch in sales_df["branch"].dropna().unique():
        branch_data = sales_df[sales_df["branch"] == branch]

        div_totals = branch_data.groupby("division")["total_amount"].sum()
        total_revenue = div_totals.sum()

        if total_revenue == 0:
            continue

        coffee_rev = 0
        pastry_rev = 0
        drinks_rev = 0
        shakes_rev = 0

        for div, amount in div_totals.items():
            div_str = str(div).lower() if div else ""
            if any(k in div_str for k in ["coffee", "frappe", "espresso"]):
                coffee_rev += amount
            elif any(k in div_str for k in ["chimney", "conut", "pastry", "bakery"]):
                pastry_rev += amount
            elif any(k in div_str for k in ["shake", "smoothie"]):
                shakes_rev += amount
            elif any(k in div_str for k in ["drink", "tea", "water", "juice"]):
                drinks_rev += amount
            else:
                pass

        branch_monthly = monthly_df[monthly_df["branch"].str.contains(branch.split(" - ")[0], case=False, na=False)]
        avg_monthly = branch_monthly["total_sales"].mean() if len(branch_monthly) > 0 else 0
        n_months = len(branch_monthly)

        # OSM Live Location Data
        coords = BRANCH_COORDS.get(branch)
        if coords:
            spatial_feats = osm_client.get_spatial_features(coords["lat"], coords["lon"])
        else:
            # Fallback for unknown branches
            spatial_feats = osm_client.get_spatial_features(33.88, 35.49)

        profiles[branch] = {
            "total_revenue": total_revenue,
            "coffee_ratio": coffee_rev / total_revenue if total_revenue > 0 else 0,
            "pastry_ratio": pastry_rev / total_revenue if total_revenue > 0 else 0,
            "drinks_ratio": drinks_rev / total_revenue if total_revenue > 0 else 0,
            "shakes_ratio": shakes_rev / total_revenue if total_revenue > 0 else 0,
            "foot_traffic_index": spatial_feats["foot_traffic_index"],
            "commercial_density": spatial_feats["commercial_density"],
            "university_proximity": spatial_feats["university_proximity"],
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
            "foot_traffic_index", "commercial_density", "university_proximity"
        ]
        self.osm_client = OSMClient()

    def fit(self):
        """Build branch profiles and set reference."""
        self.profiles_df = build_branch_profiles()
        print(f"[ExpansionScorer] Built profiles for {len(self.profiles_df)} branches using live OSM data.")
        print(f"[ExpansionScorer] Reference branch: {self.reference_branch}")

        if self.reference_branch not in self.profiles_df.index:
            self.reference_branch = self.profiles_df["total_revenue"].idxmax()
            print(f"[ExpansionScorer] Auto-selected reference: {self.reference_branch}")

        return self

    def score(self, candidate_branch=None, candidate_features=None, lat=None, lon=None):
        if self.profiles_df is None:
            self.fit()

        ref_vector = self.profiles_df.loc[
            self.reference_branch, self.feature_cols
        ].values.reshape(1, -1)

        osm_info = ""

        if candidate_branch and candidate_branch in self.profiles_df.index:
            cand_vector = self.profiles_df.loc[
                candidate_branch, self.feature_cols
            ].values.reshape(1, -1)
            cand_profile = self.profiles_df.loc[candidate_branch].to_dict()
        elif candidate_features and lat and lon:
            # Inject real OSM data into the candidate features
            spatial_feats = self.osm_client.get_spatial_features(lat, lon)
            candidate_features["foot_traffic_index"] = spatial_feats["foot_traffic_index"]
            candidate_features["commercial_density"] = spatial_feats["commercial_density"]
            candidate_features["university_proximity"] = spatial_feats["university_proximity"]
            osm_info = f"OSM data derived from ({lat}, {lon})"
            
            cand_vector = np.array(
                [candidate_features.get(c, 0) for c in self.feature_cols]
            ).reshape(1, -1)
            cand_profile = candidate_features
        elif candidate_features:
            cand_vector = np.array(
                [candidate_features.get(c, 0) for c in self.feature_cols]
            ).reshape(1, -1)
            cand_profile = candidate_features
        else:
            return {"error": "Provide either candidate_branch or candidate_features"}

        similarity = cosine_similarity(ref_vector, cand_vector)[0][0]

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
            "osm_data_fetched": bool(lat and lon),
            "similarity_score": round(similarity, 4),
            "recommendation": recommendation,
            "reference_profile": {
                "total_revenue": ref_profile["total_revenue"],
                "coffee_ratio": round(ref_profile["coffee_ratio"], 3),
                "pastry_ratio": round(ref_profile["pastry_ratio"], 3),
                "shakes_ratio": round(ref_profile["shakes_ratio"], 3),
                "foot_traffic_index": round(ref_profile["foot_traffic_index"], 3),
                "commercial_density": round(ref_profile["commercial_density"], 3),
                "university_proximity": round(ref_profile["university_proximity"], 3),
            },
            "candidate_profile": {
                k: round(v, 3) if isinstance(v, float) else v
                for k, v in cand_profile.items() if k in self.feature_cols
            },
            "gaps": {
                col: round(ref_profile[col] - cand_profile.get(col, 0), 3)
                for col in self.feature_cols
            },
        }

    def rank_all_branches(self):
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
        
    print("\nTesting New Location Fetch (Tripoli, Lebanon)")
    res = scorer.score(
        candidate_features={
            "coffee_ratio": 0.45,
            "pastry_ratio": 0.35,
            "drinks_ratio": 0.15,
            "shakes_ratio": 0.05
        },
        lat=34.4346, lon=35.8362
    )
    print(f"  Custom Tripoli Location: similarity={res['similarity_score']}, {res['recommendation']}")
    print(f"  Fetched OSM Spatial: FT={res['candidate_profile']['foot_traffic_index']}, COM={res['candidate_profile']['commercial_density']}, UNI={res['candidate_profile']['university_proximity']}")

    scorer.save()
