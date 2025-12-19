"""
Module for generating ER and BA networks
"""

import networkx as nx
from typing import Dict, Tuple, List


def generate_er_network(n: int, p: float) -> nx.Graph:
    """
    Generate an Erdos-Renyi network
    
    Args:
        n: Number of nodes
        p: Connection probability
        
    Returns:
        Generated NetworkX graph
    """
    return nx.erdos_renyi_graph(n, p)


def generate_ba_network(n: int, m: int) -> nx.Graph:
    """
    Generate a Barabasi-Albert network
    
    Args:
        n: Number of nodes
        m: Number of edges attached when a new node is added
        
    Returns:
        Generated NetworkX graph
    """
    return nx.barabasi_albert_graph(n, m)


def generate_multiple_er_networks(n: int, p: float, n_runs: int) -> List[nx.Graph]:
    """
    Generate multiple ER network instances
    
    Args:
        n: Number of nodes
        p: Connection probability
        n_runs: Number of runs
        
    Returns:
        List of generated graphs
    """
    return [generate_er_network(n, p) for _ in range(n_runs)]


def generate_multiple_ba_networks(n: int, m: int, n_runs: int) -> List[nx.Graph]:
    """
    Generate multiple BA network instances
    
    Args:
        n: Number of nodes
        m: Number of edges per new node
        n_runs: Number of runs
        
    Returns:
        List of generated graphs
    """
    return [generate_ba_network(n, m) for _ in range(n_runs)]

