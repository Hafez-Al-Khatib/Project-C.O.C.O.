"""
Objective 1 — Combo Optimization via Graph Clustering
======================================================
Builds a co-purchase graph from transaction data and runs Louvain
community detection to identify natural product combos.
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
import community as community_louvain
from itertools import combinations
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_DIR = os.path.join(BASE_DIR, "cleaned")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def build_copurchase_graph():
    """
    Build a weighted co-purchase graph:
      - Nodes = menu items (with price > 0)
      - Edges = co-occurrence count (items bought together by same customer)
    """
    df = pd.read_parquet(os.path.join(CLEANED_DIR, "transactions_products.parquet"))

    # Group items into baskets per customer per branch per receipt
    if "receipt_id" in df.columns:
        baskets = df.groupby(["branch", "customer", "receipt_id"])["item"].apply(list).reset_index()
    else:
        baskets = df.groupby(["branch", "customer"])["item"].apply(list).reset_index()

    # Build the graph
    G = nx.Graph()

    for _, row in baskets.iterrows():
        items = list(set(row["item"]))  # unique items in this basket
        if len(items) < 2:
            continue
        for item_a, item_b in combinations(items, 2):
            if G.has_edge(item_a, item_b):
                G[item_a][item_b]["weight"] += 1
            else:
                G.add_edge(item_a, item_b, weight=1)

    # Add node attributes (total qty & revenue)
    item_stats = df.groupby("item").agg(
        total_qty=("qty", "sum"),
        total_revenue=("price", "sum"),
        n_customers=("customer", "nunique"),
    ).to_dict("index")

    for node in list(G.nodes()):
        stats = item_stats.get(node, {})
        G.nodes[node]["total_qty"] = stats.get("total_qty", 0)
        G.nodes[node]["total_revenue"] = stats.get("total_revenue", 0)
        G.nodes[node]["n_customers"] = stats.get("n_customers", 0)

    # Prune statistical noise: Remove edges where items were bought together less than 3 times
    edges_to_remove = [(u, v) for u, v, data in G.edges(data=True) if data.get('weight', 0) < 3]
    G.remove_edges_from(edges_to_remove)
    
    # Remove any nodes that became completely isolated after pruning
    nodes_to_remove = [node for node, degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(nodes_to_remove)

    return G


def detect_communities(G):
    """Run Louvain community detection on the co-purchase graph."""
    if len(G.nodes) == 0:
        return {}, {}

    partition = community_louvain.best_partition(G, weight="weight", random_state=42)

    # Group items by community
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    return partition, communities


def get_combo_recommendations(G, partition, target_item, top_n=5):
    """
    Get combo recommendations for a given item based on:
    1. Edge weight (co-purchase frequency)
    2. Same community bonus
    3. Attach Rate Probability
    """
    if target_item not in G.nodes:
        # Fuzzy match: try partial matching
        matches = [n for n in G.nodes if target_item.upper() in n.upper()]
        if matches:
            target_item = matches[0]
        else:
            return {
                "target_item": target_item,
                "error": "Item not found in graph",
                "available_items": sorted(list(G.nodes))[:20],
            }

    neighbors = G[target_item]
    target_comm = partition.get(target_item, -1)

    recommendations = []
    for neighbor, attrs in neighbors.items():
        weight = attrs["weight"]
        neighbor_comm = partition.get(neighbor, -1)
        same_community = target_comm == neighbor_comm

        # Attach Rate = edge weight / max degree of either node
        max_degree = max(G.degree(target_item, weight="weight"),
                         G.degree(neighbor, weight="weight"))
        attach_rate = weight / max_degree if max_degree > 0 else 0

        # Give a 20% mathematical boost for being in the same natural cluster
        combo_score = attach_rate * (1.2 if same_community else 1.0)

        recommendations.append({
            "recommended_combo": neighbor,
            "co_purchase_count": weight,
            "attach_rate": round(attach_rate, 3),
            "same_community": same_community,
            "community_id": neighbor_comm,
            "score": combo_score
        })

    # Sort by the new weighted score instead of the boolean flag
    recommendations.sort(key=lambda x: x["score"], reverse=True)
    top_recs = recommendations[:top_n]

    if top_recs:
        best = top_recs[0]
        return {
            "target_item": target_item,
            "recommended_combo": best["recommended_combo"],
            "confidence_weight": f"{best['attach_rate']*100:.1f}%",
            "business_reason": (
                f"Co-purchased {best['co_purchase_count']} times. "
                f"{'Same product community — naturally paired.' if best['same_community'] else 'Cross-community upsell opportunity.'}"
            ),
            "community_id": partition.get(target_item, -1),
            "all_recommendations": top_recs,
        }
    return {
        "target_item": target_item,
        "recommended_combo": None,
        "confidence_weight": "0%",
        "business_reason": "No co-purchase data found for this item.",
    }


class ComboOptimizer:
    """Encapsulates the combo optimization model for API use."""

    def __init__(self):
        self.G = None
        self.partition = None
        self.communities = None

    def fit(self):
        """Build graph and detect communities."""
        self.G = build_copurchase_graph()
        self.partition, self.communities = detect_communities(self.G)
        print(f"[ComboOptimizer] Graph: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")
        print(f"[ComboOptimizer] Detected {len(self.communities)} communities")
        return self

    def predict(self, target_item, top_n=5):
        """Get combo recommendations for a target item."""
        return get_combo_recommendations(self.G, self.partition, target_item, top_n)

    def get_all_communities(self):
        """Return all detected communities with their items."""
        result = {}
        for comm_id, items in self.communities.items():
            result[comm_id] = {
                "items": items,
                "size": len(items),
            }
        return result

    def get_graph_stats(self):
        """Return graph statistics."""
        return {
            "n_nodes": len(self.G.nodes),
            "n_edges": len(self.G.edges),
            "n_communities": len(self.communities),
            "density": round(nx.density(self.G), 4),
            "top_items_by_degree": sorted(
                [(n, d) for n, d in self.G.degree(weight="weight")],
                key=lambda x: x[1], reverse=True
            )[:10],
        }

    def save(self, path=None):
        if path is None:
            path = os.path.join(MODELS_DIR, "combo_optimizer.pkl")
        joblib.dump(self, path)
        print(f"[ComboOptimizer] Saved to {path}")

    @staticmethod
    def load(path=None):
        if path is None:
            path = os.path.join(MODELS_DIR, "combo_optimizer.pkl")
        return joblib.load(path)


if __name__ == "__main__":
    optimizer = ComboOptimizer().fit()
    print("\nGraph stats:", optimizer.get_graph_stats())
    print("\nCommunities:", optimizer.get_all_communities())

    # Test a recommendation
    test_items = ["CHIMNEY THE ONE", "CLASSIC CHIMNEY", "CAFFE LATTE"]
    for item in test_items:
        print(f"\nCombo for '{item}':")
        result = optimizer.predict(item)
        print(result)

    optimizer.save()
