#!/usr/bin/env python3
"""
Generate all figures from saved results.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
from src.visualization import FigureGenerator

def main():
    """
    Load saved results and generate figures.
    """
    print("Loading saved results...")
    
    with open('data/ricci_history.pkl', 'rb') as f:
        history = pickle.load(f)
    
    with open('data/entropy_history.pkl', 'rb') as f:
        entropy_history = pickle.load(f)
    
    with open('data/economic_entropy.pkl', 'rb') as f:
        economic_entropy = pickle.load(f)
    
    with open('data/stability_history.pkl', 'rb') as f:
        stability_history = pickle.load(f)
    
    # Get graphs if available
    G_initial = None
    G_final = history.get('final_graph', None)
    
    print("Generating figures...")
    fig_gen = FigureGenerator()
    fig_gen.figure_all(history, entropy_history, economic_entropy,
                       stability_history, G_initial, G_final)
    
    print("All figures generated successfully!")

if __name__ == "__main__":
    main()
