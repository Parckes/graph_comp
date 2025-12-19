"""
Module for result visualization
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, Tuple, Any
import os

from config.config import (
    FIGURE_DPI, OUTPUT_DIR, PLOT_STYLE, PLOT_CONTEXT, PLOT_PALETTE,
    COLOR_ER, COLOR_BA
)

# Set plot style
sns.set(style=PLOT_STYLE, context=PLOT_CONTEXT, palette=PLOT_PALETTE)


def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_degree_distribution(
    er_degree_dist: Tuple[list, list],
    ba_degree_dist: Tuple[list, list],
    er_p: float = 0.005,
    ba_m: int = 5
) -> None:
    """
    Plot degree distribution
    
    Args:
        er_degree_dist: Tuple (k_values, pk_values) for ER network
        ba_degree_dist: Tuple (k_values, pk_values) for BA network
        er_p: Parameter p for ER network (for label)
        ba_m: Parameter m for BA network (for label)
    """
    ensure_output_dir()
    
    plt.figure(figsize=(8, 5))
    
    k_er, pk_er = er_degree_dist
    plt.bar(k_er, pk_er, alpha=0.6, label=f"ER (p={er_p})", color=COLOR_ER)
    
    k_ba, pk_ba = ba_degree_dist
    # Filter points for visualization (degree >= 3)
    k_ba_filtered = [k for k in k_ba if k >= 3]
    pk_ba_filtered = [pk_ba[i] for i, k in enumerate(k_ba) if k >= 3]
    plt.plot(k_ba_filtered, pk_ba_filtered, 'o-', color=COLOR_BA,
             label=f"BA (m={ba_m})", markersize=5)
    
    plt.xlabel("Degree k")
    plt.ylabel("P(k)")
    plt.title("Degree Distribution")
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "degree_distribution.png")
    plt.savefig(output_path, dpi=FIGURE_DPI)
    plt.close()
    
    print(f"  ✓ {output_path}")


def plot_clustering_comparison(
    er_clustering: Tuple[float, float],
    ba_clustering: Tuple[float, float]
) -> None:
    """
    Plot clustering coefficient comparison
    
    Args:
        er_clustering: Tuple (mean, std) for ER network
        ba_clustering: Tuple (mean, std) for BA network
    """
    ensure_output_dir()
    
    plt.figure(figsize=(6, 4))
    
    models = ['ER', 'BA']
    C_values = [er_clustering[0], ba_clustering[0]]
    errors = [er_clustering[1], ba_clustering[1]]
    
    plt.bar(models, C_values, yerr=errors, capsize=8,
            color=[COLOR_ER, COLOR_BA])
    plt.ylabel("Average Clustering Coefficient")
    plt.title("Clustering Coefficient Comparison")
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "clustering_comparison.png")
    plt.savefig(output_path, dpi=FIGURE_DPI)
    plt.close()
    
    print(f"  ✓ {output_path}")


def plot_average_path_length(
    er_path_lengths: Dict[float, Tuple[float, float]],
    ba_path_lengths: Dict[int, Tuple[float, float]],
    n_nodes: int = 1000
) -> None:
    """
    Plot average path length vs mean degree
    
    Args:
        er_path_lengths: Dictionary {p: (mean, std)} for ER networks
        ba_path_lengths: Dictionary {m: (mean, std)} for BA networks
        n_nodes: Number of nodes in network
    """
    ensure_output_dir()
    
    plt.figure(figsize=(7, 5))
    
    # Compute mean degrees for each configuration
    er_configs = {}  # {mean_degree: path_length}
    for p, (mean, _) in er_path_lengths.items():
        mean_k = p * (n_nodes - 1)
        er_configs[mean_k] = mean
    
    ba_configs = {}  # {mean_degree: path_length}
    for m, (mean, _) in ba_path_lengths.items():
        mean_k = 2 * m  # Approximate mean degree for BA
        ba_configs[mean_k] = mean
    
    # Match to target mean degrees: 3, 5, 10
    target_degrees = [3, 5, 10]
    L_er = []
    L_ba = []
    
    for target_k in target_degrees:
        # Find closest ER configuration
        er_closest_mean = min(er_configs.keys(), key=lambda x: abs(x - target_k))
        L_er.append(er_configs[er_closest_mean])
        
        # Find closest BA configuration
        ba_closest_mean = min(ba_configs.keys(), key=lambda x: abs(x - target_k))
        L_ba.append(ba_configs[ba_closest_mean])
    
    plt.plot(target_degrees, L_er, 'o--', label='ER', color=COLOR_ER)
    plt.plot(target_degrees, L_ba, 's-', label='BA', color=COLOR_BA)
    plt.xlabel("Mean Degree <k>")
    plt.ylabel("Average Path Length L")
    plt.title("Average Path Length vs Mean Degree")
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "average_path_length.png")
    plt.savefig(output_path, dpi=FIGURE_DPI)
    plt.close()
    
    print(f"  ✓ {output_path}")


def plot_robustness_attack(
    er_robustness: Tuple[np.ndarray, np.ndarray],
    ba_robustness: Tuple[np.ndarray, np.ndarray]
) -> None:
    """
    Plot robustness to targeted attack
    
    Args:
        er_robustness: Tuple (fractions, robustness) for ER network
        ba_robustness: Tuple (fractions, robustness) for BA network
    """
    ensure_output_dir()
    
    if er_robustness is None or ba_robustness is None:
        print("Warning: robustness data is missing")
        return
    
    plt.figure(figsize=(7, 5))
    
    frac_removed_er, largest_component_er = er_robustness
    frac_removed_ba, largest_component_ba = ba_robustness
    
    plt.plot(frac_removed_er * 100, largest_component_er, 'o--',
             label='ER', color=COLOR_ER)
    plt.plot(frac_removed_ba * 100, largest_component_ba, 's-',
             label='BA', color=COLOR_BA)
    plt.xlabel("Fraction of Nodes Removed (%)")
    plt.ylabel("Relative Size of Largest Component")
    plt.title("Network Robustness under Targeted Attack")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, "robustness_attack.png")
    plt.savefig(output_path, dpi=FIGURE_DPI)
    plt.close()
    
    print(f"  ✓ {output_path}")

