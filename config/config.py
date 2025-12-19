"""
Configuration for ER and BA network comparison experiments
"""

# Network parameters
N_NODES = 1000
N_RUNS = 10  # Number of runs per configuration

# Erdos-Renyi model parameters
ER_PROBS = [0.003, 0.005, 0.01]

# Barabasi-Albert model parameters (number of edges per new node)
BA_MS = [3, 5, 10]

# Robustness analysis parameters
ROBUSTNESS_FRACTIONS = 11  # Number of points on robustness plot
ROBUSTNESS_MAX_FRACTION = 0.2  # Maximum fraction of removed nodes

# Plot parameters
FIGURE_DPI = 300
OUTPUT_DIR = "results"

# Visualization parameters
PLOT_STYLE = "whitegrid"
PLOT_CONTEXT = "talk"
PLOT_PALETTE = "muted"

# Plot colors
COLOR_ER = 'skyblue'
COLOR_BA = 'salmon'

