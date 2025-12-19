"""
Main script for running ER and BA network comparative analysis
"""

import sys
import os

# Add root directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiments import run_experiments
from src.visualization import (
    plot_degree_distribution,
    plot_clustering_comparison,
    plot_average_path_length,
    plot_robustness_attack
)
from config.config import ER_PROBS, BA_MS, N_NODES


def main():
    """Main function"""
    print("Starting network comparative analysis...")
    print()
    
    # Run experiments
    results = run_experiments()
    
    # Generate plots
    print("Generating plots...")
    print()
    
    # Degree distribution
    plot_degree_distribution(
        results['er_degree_dist'][0.005],
        results['ba_degree_dist'][5],
        er_p=0.005,
        ba_m=5
    )
    
    # Clustering coefficient
    plot_clustering_comparison(
        results['er_clustering'][0.005],
        results['ba_clustering'][5]
    )
    
    # Average path length
    plot_average_path_length(
        results['er_path_lengths'],
        results['ba_path_lengths'],
        n_nodes=N_NODES
    )
    
    # Robustness to attacks
    plot_robustness_attack(
        results['er_robustness'],
        results['ba_robustness']
    )
    
    print()
    print("=" * 60)
    print("All plots successfully saved to 'results/' directory")
    print("=" * 60)


if __name__ == "__main__":
    main()

