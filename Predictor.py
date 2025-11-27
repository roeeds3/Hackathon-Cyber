import json
import time
import io
from math import sqrt
from typing import Dict, List, Any, Optional

import pandas as pd
import networkx as nx
from sklearn.preprocessing import LabelEncoder
import hdbscan
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for web API
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


# -----------------------------------------------------------------------------
# Node Store
# -----------------------------------------------------------------------------

class NodeStore:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}

    def update_node(self, node_json: dict):
        self.nodes[str(node_json["ID"])] = node_json

    def get_all_nodes(self) -> pd.DataFrame:
        if not self.nodes:
            return pd.DataFrame()
        df = pd.DataFrame(self.nodes.values())
        # Ensure severity column exists even if all values are None
        # This prevents issues when some nodes have severity and others don't
        if "severity" not in df.columns:
            df["severity"] = None
        return df

    def get_attacked_nodes(self) -> pd.DataFrame:
        df = self.get_all_nodes()
        if df.empty:
            return df
        return df[df["Is_attacked"] == True]


# -----------------------------------------------------------------------------
# Attack Cluster Detector
# -----------------------------------------------------------------------------

class AttackClusterDetector:
    def __init__(self, distance_threshold=300, min_cluster_size=3):
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.provider_encoder = LabelEncoder()

    # -------------------------------------------------------------------------
    # HDBSCAN clustering
    # -------------------------------------------------------------------------

    def hdbscan_clusters(self, attacked_df: pd.DataFrame) -> pd.DataFrame:
        if attacked_df.empty or len(attacked_df) < self.min_cluster_size:
            attacked_df["cluster_id"] = -1
            return attacked_df

        df = attacked_df.copy()

        df["provider_encoded"] = self.provider_encoder.fit_transform(
            df["provider"].astype(str)
        )

        features = df[
            ["loc_x", "loc_y", "provider_encoded"]
        ].astype(float)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=1
        )

        labels = clusterer.fit_predict(features)
        df["cluster_id"] = labels

        return df

    # -------------------------------------------------------------------------
    # At-Risk Node Detection
    # -------------------------------------------------------------------------

    def identify_at_risk_nodes(self, all_nodes: pd.DataFrame, attacked_with_clusters: pd.DataFrame) -> pd.DataFrame:
        """
        Find safe nodes that would cluster together with attacked nodes according to HDBSCAN.
        Uses the same features as the attack clustering.
        """
        if all_nodes.empty or attacked_with_clusters.empty:
            return pd.DataFrame()

        safe_nodes = all_nodes[all_nodes["Is_attacked"] == False].copy()
        if safe_nodes.empty:
            return pd.DataFrame()

        # Re-fit encoder on all providers (attacked + safe)
        all_providers = pd.concat([
            attacked_with_clusters["provider"],
            safe_nodes["provider"]
        ])
        self.provider_encoder.fit(all_providers.astype(str))

        # Encode providers for both attacked and safe nodes
        attacked_copy = attacked_with_clusters.copy()
        attacked_copy["provider_encoded"] = self.provider_encoder.transform(
            attacked_copy["provider"].astype(str)
        )
        
        safe_nodes["provider_encoded"] = self.provider_encoder.transform(
            safe_nodes["provider"].astype(str)
        )

        # Combine attacked and safe nodes for clustering
        combined = pd.concat([
            attacked_copy[["ID", "loc_x", "loc_y", "provider_encoded", "cluster_id", "Is_attacked", "provider"]],
            safe_nodes[["ID", "loc_x", "loc_y", "provider_encoded", "Is_attacked", "provider"]].assign(cluster_id=-999)
        ], ignore_index=True)

        features = combined[
            ["loc_x", "loc_y", "provider_encoded"]
        ].astype(float)

        # Re-cluster with all nodes
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=1
        )

        labels = clusterer.fit_predict(features)
        combined["new_cluster_id"] = labels

        # Find safe nodes that got assigned to actual attack clusters (not noise)
        attacked_cluster_ids = set(attacked_copy[attacked_copy["cluster_id"] != -1]["cluster_id"].unique())
        at_risk = combined[
            (combined["Is_attacked"] == False) & 
            (combined["new_cluster_id"].isin(attacked_cluster_ids)) &
            (combined["new_cluster_id"] != -1)  # Exclude noise clusters
        ]

        return at_risk[["ID", "loc_x", "loc_y", "new_cluster_id", "provider"]].rename(
            columns={"new_cluster_id": "cluster_id"}
        ).reset_index(drop=True)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------

    def visualize(self,
                  all_nodes: pd.DataFrame,
                  attacked_with_clusters: pd.DataFrame,
                  at_risk_nodes: pd.DataFrame,
                  node_store: Optional[NodeStore] = None,
                  return_image: bool = False) -> Optional[bytes]:
        """
        Creates a visualization of all nodes + clusters + at-risk nodes.
        
        Args:
            all_nodes: DataFrame with all nodes
            attacked_with_clusters: DataFrame with attacked nodes and cluster IDs
            at_risk_nodes: DataFrame with at-risk nodes
            return_image: If True, returns image bytes instead of showing plot
        
        Returns:
            Image bytes if return_image=True, None otherwise
        """
        plt.figure(figsize=(10, 8))

        # Plot safe nodes in grey circles
        if not all_nodes.empty:
            safe_nodes = all_nodes[all_nodes["Is_attacked"] == False]
            if not safe_nodes.empty:
                plt.scatter(
                    safe_nodes["loc_x"],
                    safe_nodes["loc_y"],
                    c="gray",
                    label="Safe",
                    s=100,
                    alpha=0.7,
                    marker="o",
                    edgecolor="none"
                )

        # Plot at-risk safe nodes in yellow circles
        if not at_risk_nodes.empty:
            plt.scatter(
                at_risk_nodes["loc_x"],
                at_risk_nodes["loc_y"],
                c="orange",
                label="At Risk",
                s=100,
                alpha=0.7,
                marker="o",
                edgecolor="none"
            )

        # Color-map for HDBSCAN clusters - colored circles with severity-based intensity
        if not attacked_with_clusters.empty:
            # Debug: Check if severity column exists
            has_severity_col = "severity" in all_nodes.columns if not all_nodes.empty else False
            
            # Create severity lookup dictionary from all_nodes
            severity_lookup = {}
            if not all_nodes.empty and has_severity_col:
                for _, row in all_nodes.iterrows():
                    node_id = str(row["ID"])
                    severity_val = row["severity"]
                    if pd.notna(severity_val):
                        severity_lookup[node_id] = float(severity_val)
                    else:
                        severity_lookup[node_id] = 0.0
            elif not all_nodes.empty and node_store is not None:
                # If severity column doesn't exist in DataFrame, get it from the original nodes dict
                # This handles cases where DataFrame conversion might drop None values
                for node_id, node_data in node_store.nodes.items():
                    if "severity" in node_data:
                        severity_val = node_data["severity"]
                        if severity_val is not None:
                            severity_lookup[str(node_id)] = float(severity_val)
                        else:
                            severity_lookup[str(node_id)] = 0.0
                    else:
                        severity_lookup[str(node_id)] = 0.0
            
            clusters = attacked_with_clusters["cluster_id"].unique()
            # Filter out noise points (cluster_id == -1)
            actual_clusters = [c for c in clusters if c != -1]
            cluster_colors = cm.tab10(np.linspace(0, 1, len(actual_clusters)))

            for color, cluster_id in zip(cluster_colors, actual_clusters):
                subset = attacked_with_clusters[
                    attacked_with_clusters["cluster_id"] == cluster_id
                ]
                label = f"Attacked Cluster {cluster_id}"
                
                # Get severity scores for nodes in this cluster using lookup
                severities = []
                for _, row in subset.iterrows():
                    node_id = str(row["ID"])
                    severity_val = severity_lookup.get(node_id, 0.0)
                    severities.append(severity_val)
                severities = np.array(severities)
                
                # Normalize severity to alpha values (0.3 to 1.0 for maximum visibility range)
                # Higher severity = darker/more opaque (higher alpha)
                if len(severities) > 0 and severities.max() > 0:
                    # Scale severity to alpha range [0.3, 1.0] for better contrast
                    alpha_values = 0.3 + 0.7 * (severities / 100.0)
                    alpha_values = np.clip(alpha_values, 0.3, 1.0)
                else:
                    # Default alpha if no severity data
                    alpha_values = np.full(len(subset), 0.7)
                
                # Convert color to RGB tuple for matplotlib
                if isinstance(color, np.ndarray):
                    base_rgb = np.array(color[:3])
                elif isinstance(color, tuple):
                    base_rgb = np.array(color[:3])
                else:
                    base_rgb = np.array([0.5, 0.5, 0.5])
                
                # Plot points individually with per-point alpha and color intensity based on severity
                for idx, (_, row) in enumerate(subset.iterrows()):
                    alpha = alpha_values[idx]
                    severity = severities[idx] if len(severities) > idx else 0.0
                    
                    # Darken color based on severity (higher severity = darker color)
                    # Mix base color with black based on severity
                    darkness_factor = severity / 100.0  # 0 to 1
                    darkened_rgb = base_rgb * (1 - 0.5 * darkness_factor)  # Darken up to 50% for more contrast
                    darkened_rgb = np.clip(darkened_rgb, 0, 1)
                    
                    # Create RGBA color tuple with alpha included
                    rgba_color = (*darkened_rgb, alpha)
                    
                    # Only add label for first point in cluster
                    # Use facecolors parameter to ensure RGBA is applied correctly
                    plt.scatter(
                        row["loc_x"],
                        row["loc_y"],
                        s=100,
                        marker="o",
                        edgecolor="none",
                        label=label if idx == 0 else "",
                        facecolors=[rgba_color]
                    )
                    
                    # Add severity score as text label next to the point
                    plt.annotate(
                        f'{severity:.1f}',
                        xy=(row["loc_x"], row["loc_y"]),
                        xytext=(5, 5),  # Offset text slightly to the right and up
                        textcoords='offset points',
                        fontsize=8,
                        color='black',
                        weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray', linewidth=0.5)
                    )
                
                # Draw circle around each cluster
                if len(subset) > 0:
                    center_x = subset["loc_x"].mean()
                    center_y = subset["loc_y"].mean()
                    # Calculate radius as max distance from center to any point in cluster
                    distances = np.sqrt((subset["loc_x"] - center_x)**2 + (subset["loc_y"] - center_y)**2)
                    radius = distances.max() + 20  # Add padding
                    circle = plt.Circle((center_x, center_y), radius, color=color, fill=False, linewidth=2, alpha=0.5)
                    plt.gca().add_patch(circle)

            # Plot noise points separately with severity-based intensity
            noise = attacked_with_clusters[attacked_with_clusters["cluster_id"] == -1]
            if not noise.empty:
                # Get severity scores for noise points using lookup
                severities = []
                for _, row in noise.iterrows():
                    node_id = str(row["ID"])
                    severity_val = severity_lookup.get(node_id, 0.0)
                    severities.append(severity_val)
                severities = np.array(severities)
                
                # Normalize severity to alpha values (0.3 to 1.0 for maximum visibility range)
                if len(severities) > 0 and severities.max() > 0:
                    # Scale severity to alpha range [0.3, 1.0] for better contrast
                    alpha_values = 0.3 + 0.7 * (severities / 100.0)
                    alpha_values = np.clip(alpha_values, 0.3, 1.0)
                else:
                    # Default alpha if no severity data
                    alpha_values = np.full(len(noise), 0.7)
                
                # Plot noise points individually with per-point alpha and color intensity based on severity
                for idx, (_, row) in enumerate(noise.iterrows()):
                    alpha = alpha_values[idx]
                    severity = severities[idx] if len(severities) > idx else 0.0
                    
                    # Darken red color based on severity (higher severity = darker red)
                    darkness_factor = severity / 100.0  # 0 to 1
                    red_intensity = 1.0 - 0.5 * darkness_factor  # Darken red up to 50% for more contrast
                    red_intensity = max(0.0, min(1.0, red_intensity))
                    
                    # Create RGBA color tuple with alpha included
                    rgba_red = (red_intensity, 0.0, 0.0, alpha)
                    
                    # Only add label for first noise point
                    # Use facecolors parameter to ensure RGBA is applied correctly
                    plt.scatter(
                        row["loc_x"],
                        row["loc_y"],
                        s=100,
                        marker="o",
                        edgecolor="none",
                        label="Attacked" if idx == 0 else "",
                        facecolors=[rgba_red]
                    )
                    
                    # Add severity score as text label next to the point
                    plt.annotate(
                        f'{severity:.1f}',
                        xy=(row["loc_x"], row["loc_y"]),
                        xytext=(5, 5),  # Offset text slightly to the right and up
                        textcoords='offset points',
                        fontsize=8,
                        color='black',
                        weight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7, edgecolor='gray', linewidth=0.5)
                    )

        plt.xlabel("X Location")
        plt.ylabel("Y Location")
        plt.title("EV Charger Attack Cluster Visualization")
        plt.legend()
        plt.grid(True)
        
        if return_image:
            # Save to bytes buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            image_bytes = buf.read()
            buf.close()
            plt.close()  # Close figure to free memory
            return image_bytes
        else:
            plt.show()
            return None

    # -------------------------------------------------------------------------
    # Full detection pipeline
    # -------------------------------------------------------------------------

    def detect(self, all_nodes: pd.DataFrame, attacked_df: pd.DataFrame) -> Dict[str, Any]:
        if attacked_df.empty:
            return {
                "num_attacked": 0,
                "hdbscan_clusters": [],
                "connected_components": [],
                "clustered_df": pd.DataFrame(),  # Return empty DataFrame for consistency
                "at_risk_nodes": pd.DataFrame()
            }

        clustered = self.hdbscan_clusters(attacked_df)
        at_risk = self.identify_at_risk_nodes(all_nodes, clustered)

        return {
            "num_attacked": len(attacked_df),
            "clustered_df": clustered,
            "at_risk_nodes": at_risk
        }


# -----------------------------------------------------------------------------
# Streaming Monitor
# -----------------------------------------------------------------------------

class StreamingAttackMonitor:
    def __init__(self, distance_threshold=300, min_cluster_size=3):
        self.store = NodeStore()
        self.detector = AttackClusterDetector(
            distance_threshold=distance_threshold,
            min_cluster_size=min_cluster_size
        )

    def process_json(self, node_json: dict, visualize=True, return_image=False) -> Dict[str, Any]:
        self.store.update_node(node_json)
        all_nodes = self.store.get_all_nodes()
        attacked = self.store.get_attacked_nodes()
        results = self.detector.detect(all_nodes, attacked)

        if visualize:
            image_bytes = self.detector.visualize(
                all_nodes,
                results["clustered_df"],
                results["at_risk_nodes"],
                node_store=self.store,
                return_image=return_image
            )
            if return_image and image_bytes:
                results["visualization_image"] = image_bytes

        return results


# -----------------------------------------------------------------------------
# Demo
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import os

    monitor = StreamingAttackMonitor()

    # Default JSON file path
    json_file = "data.json" # {ID:str, Is_attacked:bool, loc_x:int, lox_y:int,provider:str}

    # Try to load from JSON file, fall back to example data
    example_updates = None
    if os.path.exists(json_file):
        try:
            with open(json_file, 'r') as f:
                example_updates = json.load(f)
            print(f"Loaded data from {json_file}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            print("Using example data instead")

    # Use example data if no file was loaded
    if example_updates is None:
        example_updates = [
            {"ID": "A1", "Is_attacked": True,  "loc_x": 100, "loc_y": 200, "provider": "EVGO"},
            {"ID": "A2", "Is_attacked": True,  "loc_x": 120, "loc_y": 210, "provider": "EVGO"},
            {"ID": "A3", "Is_attacked": False, "loc_x": 500, "loc_y": 500, "provider": "Tesla"},
            {"ID": "B1", "Is_attacked": True,  "loc_x": 900, "loc_y": 900, "provider": "ChargeNet"},
            {"ID": "A2", "Is_attacked": True,  "loc_x": 125, "loc_y": 220, "provider": "EVGO"},
            {"ID": "B2", "Is_attacked": True,  "loc_x": 880, "loc_y": 890, "provider": "ChargeNet"},
            {"ID": "C1", "Is_attacked": True,  "loc_x": 250, "loc_y": 150, "provider": "EVGO"},
            {"ID": "C2", "Is_attacked": True,  "loc_x": 270, "loc_y": 160, "provider": "EVGO"},
            {"ID": "C3", "Is_attacked": True,  "loc_x": 280, "loc_y": 170, "provider": "Tesla"},
            {"ID": "D1", "Is_attacked": True,  "loc_x": 700, "loc_y": 300, "provider": "ChargeNet"},
            {"ID": "D2", "Is_attacked": True,  "loc_x": 720, "loc_y": 310, "provider": "ChargeNet"},
            {"ID": "E1", "Is_attacked": False, "loc_x": 50,  "loc_y": 50,  "provider": "EVGO"},
            {"ID": "E2", "Is_attacked": True,  "loc_x": 400, "loc_y": 400, "provider": "Tesla"},
            {"ID": "F1", "Is_attacked": True,  "loc_x": 150, "loc_y": 180, "provider": "EVGO"},
            {"ID": "F2", "Is_attacked": False, "loc_x": 600, "loc_y": 600, "provider": "ChargeNet"},
        ]

    for upd in example_updates:
        print("\nUpdate:", upd)
        results = monitor.process_json(upd, visualize=True)
        print("Detected:", json.dumps({
            "num_attacked": results["num_attacked"],
            "clusters": [
                {"ID": r["ID"], "cluster_id": r["cluster_id"]}
                for r in results["clustered_df"][["ID", "cluster_id"]].to_dict("records")
            ]
        }, indent=2))
        time.sleep(0.5)
