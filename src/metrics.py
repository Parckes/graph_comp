"""
Module for computing network metrics
"""

import networkx as nx
import numpy as np
from typing import List, Tuple, Dict
from collections import Counter


def compute_degree_distribution(graph: nx.Graph) -> Tuple[List[int], List[float]]:
    """
    Compute degree distribution of nodes
    
    Args:
        graph: NetworkX graph
        
    Returns:
        Tuple (k_values, pk_values) - degree values and their probabilities
    """
    degrees = [d for n, d in graph.degree()]
    degree_counts = Counter(degrees)
    k_values = sorted(degree_counts.keys())
    pk_values = [degree_counts[k] / len(graph.nodes()) for k in k_values]
    return k_values, pk_values


def compute_average_degree_distribution(graphs: List[nx.Graph]) -> Tuple[List[int], List[float]]:
    """
    Compute averaged degree distribution across multiple graphs
    
    Args:
        graphs: List of graphs
        
    Returns:
        Tuple (k_values, pk_values) - averaged values
    """
    degree_dists = [compute_degree_distribution(G) for G in graphs]
    
    # Collect all unique degrees
    all_k = set()
    for k_vals, _ in degree_dists:
        all_k.update(k_vals)
    
    # Average probabilities
    avg_pk = {}
    for k in all_k:
        pk_sum = 0.0
        count = 0
        for k_vals, pk_vals in degree_dists:
            if k in k_vals:
                idx = k_vals.index(k)
                pk_sum += pk_vals[idx]
                count += 1
        avg_pk[k] = pk_sum / count if count > 0 else 0.0
    
    k_sorted = sorted(avg_pk.keys())
    pk_sorted = [avg_pk[k] for k in k_sorted]
    return k_sorted, pk_sorted


def compute_clustering_coefficient(graph: nx.Graph) -> float:
    """
    Compute average clustering coefficient
    
    Args:
        graph: NetworkX graph
        
    Returns:
        Average clustering coefficient
    """
    return nx.average_clustering(graph)


def compute_average_clustering_coefficient(graphs: List[nx.Graph]) -> Tuple[float, float]:
    """
    Compute average clustering coefficient and standard deviation
    
    Args:
        graphs: List of graphs
        
    Returns:
        Tuple (mean, std) - mean value and standard deviation
    """
    clusterings = [compute_clustering_coefficient(G) for G in graphs]
    return np.mean(clusterings), np.std(clusterings)


def compute_average_path_length(graph: nx.Graph) -> float:
    """
    Compute average path length
    
    Args:
        graph: NetworkX graph
        
    Returns:
        Average path length
    """
    if not nx.is_connected(graph):
        # If graph is disconnected, compute average path length only for largest component
        largest_cc = max(nx.connected_components(graph), key=len)
        subgraph = graph.subgraph(largest_cc)
        if len(subgraph) < 2:
            return 0.0
        return nx.average_shortest_path_length(subgraph)
    return nx.average_shortest_path_length(graph)


def compute_average_path_length_stats(graphs: List[nx.Graph]) -> Tuple[float, float]:
    """
    Compute average path length statistics
    
    Args:
        graphs: List of graphs
        
    Returns:
        Tuple (mean, std) - mean value and standard deviation
    """
    path_lengths = [compute_average_path_length(G) for G in graphs]
    return np.mean(path_lengths), np.std(path_lengths)


def compute_robustness_targeted_attack(graph: nx.Graph, fractions: np.ndarray) -> List[float]:
    """
    Compute robustness to targeted attack (remove nodes in decreasing degree order)
    
    Args:
        graph: NetworkX graph
        fractions: Array of node removal fractions (0.0 - 1.0)
        
    Returns:
        List of relative sizes of largest component after each removal
    """
    results = []
    
    # Sort nodes by degree in original graph
    nodes_by_degree = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
    total_nodes = len(graph.nodes())
    
    for frac in fractions:
        n_to_remove = int(frac * total_nodes)
        
        # Create copy of graph for each removal fraction
        G_attacked = graph.copy()
        
        if n_to_remove > 0:
            # Remove nodes with highest degrees
            nodes_to_remove = [node for node, _ in nodes_by_degree[:n_to_remove]]
            G_attacked.remove_nodes_from(nodes_to_remove)
        
        if len(G_attacked.nodes()) == 0:
            results.append(0.0)
        else:
            largest_cc = max(nx.connected_components(G_attacked), key=len)
            results.append(len(largest_cc) / total_nodes)
    
    return results


def compute_average_robustness(graphs: List[nx.Graph], fractions: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute averaged robustness to targeted attack
    
    Args:
        graphs: List of graphs
        fractions: Array of node removal fractions
        
    Returns:
        Tuple (fractions, robustness_mean) - fractions and averaged robustness values
    """
    robustness_runs = [compute_robustness_targeted_attack(G, fractions) for G in graphs]
    robustness_mean = np.mean(robustness_runs, axis=0)
    return fractions, robustness_mean

