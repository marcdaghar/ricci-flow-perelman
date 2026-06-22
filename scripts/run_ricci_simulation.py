#!/usr/bin/env python3
"""
Run Ricci flow and Perelman entropy simulations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pickle
import networkx as nx
from src.ricci_model import RicciFlowModel
from src.perelman_entropy import PerelmanEntropy
from src.graph_utils import GraphGenerator
from src.visualization import FigureGenerator

def run_ricci_simulation():
    """
    Run cognitive Ricci flow simulation.
    """
    print("=" * 60)
    print("Running Cognitive Ricci Flow Simulation...")
    print("=" * 60)
    
    # Generate initial graph
    print("\nGenerating graph...")
    G = GraphGenerator.community_graph(n_nodes=100, n_communities=5,
                                       p_in=0.3, p_out=0.02)
    
    print(f"Graph generated:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")
    print(f"  Density: {nx.density(G):.4f}")
    
    # Initialize model
    model = RicciFlowModel(lambda_=0.1, mu=0.01, tau=1.0)
    
    # Define loss function
    def loss_function(G):
        # Simple loss: negative average curvature (want to maximize curvature)
        curvatures = model.compute_ollivier_ricci_curvature(G)
        mean_curv = np.mean(list(curvatures.values()))
        return -mean_curv  # Minimize negative curvature = maximize curvature
    
    # Run Ricci flow
    print("\nRunning Ricci flow...")
    history = model.evolve_ricci_flow(G, n_steps=100, dt=0.01,
                                      loss_function=loss_function)
    
    print(f"\nRicci flow results:")
    final_curv = history['mean_curvature'][-1]
    initial_curv = history['mean_curvature'][0]
    print(f"  Initial mean curvature: {initial_curv:.4f}")
    print(f"  Final mean curvature: {final_curv:.4f}")
    print(f"  Change: {final_curv - initial_curv:.4f}")
    
    # Save results
    os.makedirs('data', exist_ok=True)
    with open('data/ricci_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    
    return history, G

def run_perelman_entropy_simulation(G):
    """
    Run Perelman entropy simulation.
    """
    print("\n" + "=" * 60)
    print("Running Perelman Entropy Simulation...")
    print("=" * 60)
    
    # Initialize entropy model
    entropy_model = PerelmanEntropy(tau=1.0, n_dim=2)
    
    # Compute entropy evolution
    print("\nComputing entropy evolution...")
    entropy_history = entropy_model.compute_entropy_evolution(G, n_steps=100, dt=0.01)
    
    # Compute derivatives
    derivative = np.diff(entropy_history) / 0.01
    positive_count = np.sum(derivative >= 0)
    total_count = len(derivative)
    
    print(f"\nEntropy results:")
    print(f"  Initial entropy: {entropy_history[0]:.6f}")
    print(f"  Final entropy: {entropy_history[-1]:.6f}")
    print(f"  Monotonicity: {positive_count}/{total_count} steps ({(positive_count/total_count)*100:.1f}%)")
    
    # Save results
    with open('data/entropy_history.pkl', 'wb') as f:
        pickle.dump(entropy_history, f)
    
    return entropy_history

def run_economic_entropy_simulation():
    """
    Simulate economic entropy for correlation.
    """
    print("\n" + "=" * 60)
    print("Running Economic Entropy Simulation...")
    print("=" * 60)
    
    # Simulate economic entropy (from Article 4)
    n_steps = 100
    economic_entropy = np.zeros(n_steps)
    
    # Start with some entropy and converge to stable value
    S0 = 10.0
    S_stable = 20.0
    tau_econ = 0.1
    
    for t in range(n_steps):
        economic_entropy[t] = S_stable + (S0 - S_stable) * np.exp(-tau_econ * t)
        # Add small noise
        economic_entropy[t] += 0.05 * np.random.randn()
    
    # Normalize
    economic_entropy = (economic_entropy - np.min(economic_entropy)) / (np.max(economic_entropy) - np.min(economic_entropy))
    
    # Save results
    with open('data/economic_entropy.pkl', 'wb') as f:
        pickle.dump(economic_entropy, f)
    
    return economic_entropy

def run_unified_stability(G, entropy_history, economic_entropy):
    """
    Run unified stability simulation.
    """
    print("\n" + "=" * 60)
    print("Running Unified Stability Simulation...")
    print("=" * 60)
    
    # Align lengths
    min_len = min(len(entropy_history), len(economic_entropy))
    entropy_sample = entropy_history[:min_len]
    econ_sample = economic_entropy[:min_len]
    
    # Compute unified stability
    stability = np.array(entropy_sample) + np.array(econ_sample)
    
    # Check convergence
    if len(stability) > 10:
        last_vals = stability[-10:]
        stable_value = np.mean(last_vals)
        std_dev = np.std(last_vals)
        
        print(f"\nStability results:")
        print(f"  Steady state value: {stable_value:.4f}")
        print(f"  Std deviation (last 10): {std_dev:.4f}")
        print(f"  Convergence: {std_dev / stable_value:.4f}% variation")
    
    # Save results
    with open('data/stability_history.pkl', 'wb') as f:
        pickle.dump(stability, f)
    
    return stability

def main():
    """
    Main execution function.
    """
    # Run simulations
    history, G_initial = run_ricci_simulation()
    entropy_history = run_perelman_entropy_simulation(G_initial)
    economic_entropy = run_economic_entropy_simulation()
    stability_history = run_unified_stability(G_initial, entropy_history, economic_entropy)
    
    print("\n" + "=" * 60)
    print("All simulations complete. Results saved.")
    print("=" * 60)
    
    # Generate figures
    print("\nGenerating figures...")
    fig_gen = FigureGenerator()
    
    G_final = history['final_graph'] if 'final_graph' in history else None
    
    fig_gen.figure_all(history, entropy_history, economic_entropy,
                       stability_history, G_initial, G_final)
    
    print("\nDone!")

if __name__ == "__main__":
    main()
