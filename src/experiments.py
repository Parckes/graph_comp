"""
Module for running experiments
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional
import networkx as nx

from config.config import (
    N_NODES, N_RUNS, ER_PROBS, BA_MS,
    ROBUSTNESS_FRACTIONS, ROBUSTNESS_MAX_FRACTION
)

from src.network_generation import (
    generate_multiple_er_networks,
    generate_multiple_ba_networks
)

from src.metrics import (
    compute_average_degree_distribution,
    compute_average_clustering_coefficient,
    compute_average_path_length_stats,
    compute_average_robustness
)


def run_experiments() -> Dict[str, Any]:
    """
    Run all experiments and collect results
    
    Returns:
        Dictionary with all experiment results
    """
    print("=" * 60)
    print("ER and BA Network Comparative Analysis")
    print("=" * 60)
    print(f"Parameters: N={N_NODES}, Runs={N_RUNS}")
    print()
    
    # Results for degree distribution
    er_degree_dist = {}  # {p: (k_values, pk_values)}
    ba_degree_dist = {}  # {m: (k_values, pk_values)}
    
    # Results for clustering coefficient
    er_clustering = {}  # {p: (mean, std)}
    ba_clustering = {}  # {m: (mean, std)}
    
    # Results for average path length
    er_path_lengths = {}  # {p: (mean, std)}
    ba_path_lengths = {}  # {m: (mean, std)}
    
    # Results for robustness to attacks
    er_robustness: Optional[Tuple[np.ndarray, np.ndarray]] = None  # For p=0.005
    ba_robustness: Optional[Tuple[np.ndarray, np.ndarray]] = None  # For m=5
    
    # Generate ER networks
    print("Generating ER networks...")
    for p in ER_PROBS:
        print(f"  p = {p}", end=" ... ")
        
        graphs = generate_multiple_er_networks(N_NODES, p, N_RUNS)
        
        # Degree distribution
        er_degree_dist[p] = compute_average_degree_distribution(graphs)
        
        # Clustering coefficient
        er_clustering[p] = compute_average_clustering_coefficient(graphs)
        
        # Average path length
        er_path_lengths[p] = compute_average_path_length_stats(graphs)
        
        # Robustness to attacks only for p=0.005
        if p == 0.005:
            fracs = np.linspace(0, ROBUSTNESS_MAX_FRACTION, ROBUSTNESS_FRACTIONS)
            er_robustness = compute_average_robustness(graphs, fracs)
        
        print("✓")
    
    print()
    
    # Generate BA networks
    print("Generating BA networks...")
    for m in BA_MS:
        print(f"  m = {m}", end=" ... ")
        
        graphs = generate_multiple_ba_networks(N_NODES, m, N_RUNS)
        
        # Degree distribution
        ba_degree_dist[m] = compute_average_degree_distribution(graphs)
        
        # Clustering coefficient
        ba_clustering[m] = compute_average_clustering_coefficient(graphs)
        
        # Average path length
        ba_path_lengths[m] = compute_average_path_length_stats(graphs)
        
        # Robustness to attacks only for m=5
        if m == 5:
            fracs = np.linspace(0, ROBUSTNESS_MAX_FRACTION, ROBUSTNESS_FRACTIONS)
            ba_robustness = compute_average_robustness(graphs, fracs)
        
        print("✓")
    
    print()
    print("Experiments completed!")
    print()
    
    return {
        'er_degree_dist': er_degree_dist,
        'ba_degree_dist': ba_degree_dist,
        'er_clustering': er_clustering,
        'ba_clustering': ba_clustering,
        'er_path_lengths': er_path_lengths,
        'ba_path_lengths': ba_path_lengths,
        'er_robustness': er_robustness,
        'ba_robustness': ba_robustness
    }

