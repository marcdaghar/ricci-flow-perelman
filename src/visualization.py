"""
Visualization functions for the Ricci Flow and Perelman Entropy article.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib import rcParams
from matplotlib.patches import Circle, Rectangle

# Set publication-ready style
plt.style.use('seaborn-v0-8-whitegrid')
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['legend.fontsize'] = 10
rcParams['figure.dpi'] = 300

class FigureGenerator:
    """
    Generate figures for the Ricci Flow and Perelman Entropy article.
    """
    
    def __init__(self, output_dir='figures'):
        self.output_dir = output_dir
        import os
        os.makedirs(output_dir, exist_ok=True)
    
    def figure_ricci_flow_convergence(self, history):
        """
        Figure 1: Convergence of the cognitive Ricci flow.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Panel 1: Mean curvature evolution
        ax = axes[0, 0]
        mean_curvature = history['mean_curvature']
        ax.plot(mean_curvature, linewidth=2.5, color='darkblue')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Mean curvature', fontsize=12)
        ax.set_title('Mean Curvature Evolution', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Curvature distribution over time
        ax = axes[0, 1]
        # Extract curvature distributions at different times
        curvatures = history['curvatures']
        times = [0, len(curvatures)//4, len(curvatures)//2, -1]
        colors = ['blue', 'green', 'orange', 'red']
        
        for idx, time_idx in enumerate(times):
            if time_idx < 0:
                time_idx = len(curvatures) - 1
            if time_idx < len(curvatures):
                vals = list(curvatures[time_idx].values())
                ax.hist(vals, bins=20, alpha=0.3, color=colors[idx], 
                       label=f't={time_idx}')
        
        ax.set_xlabel('Curvature', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Curvature Distribution Over Time', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Weight evolution (select edges)
        ax = axes[1, 0]
        weights_history = history['weights']
        if len(weights_history) > 1:
            weights_array = np.array(weights_history)
            for i in range(min(5, weights_array.shape[1])):
                ax.plot(weights_array[:, i], linewidth=1.5, alpha=0.7,
                       label=f'Edge {i+1}')
        ax.set_xlabel('Snapshot (x10 iterations)', fontsize=12)
        ax.set_ylabel('Edge weight', fontsize=12)
        ax.set_title('Edge Weight Evolution', fontsize=14)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Loss function evolution
        ax = axes[1, 1]
        if 'loss' in history and len(history['loss']) > 0:
            ax.plot(history['loss'], linewidth=2.5, color='darkgreen')
            ax.set_xlabel('Iteration', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('Loss Function Evolution', fontsize=14)
        else:
            ax.text(0.5, 0.5, 'Loss data not available', 
                    ha='center', va='center', transform=ax.transAxes)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/ricci_flow_convergence.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/ricci_flow_convergence.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_perelman_entropy_evolution(self, entropy_history):
        """
        Figure 2: Evolution of Perelman entropy.
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Panel 1: Entropy evolution
        ax = axes[0]
        ax.plot(entropy_history, linewidth=2.5, color='purple')
        
        # Add trend line
        x = np.arange(len(entropy_history))
        coeff = np.polyfit(x, entropy_history, 1)
        trend = coeff[0] * x + coeff[1]
        ax.plot(x, trend, '--', linewidth=1.5, color='red',
                label=f'Trend: slope = {coeff[0]:.4f}')
        
        # Mark monotonicity
        diff = np.diff(entropy_history)
        positive_diff = np.sum(diff > 0)
        total_diff = len(diff)
        monotonic_ratio = positive_diff / total_diff
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Perelman entropy W', fontsize=12)
        ax.set_title(f'Perelman Entropy Evolution (monotonicity: {monotonic_ratio:.1%})', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Entropy derivative (should be >= 0)
        ax = axes[1]
        derivative = np.diff(entropy_history) / 0.01  # Approximate derivative
        
        ax.plot(derivative, linewidth=2, color='darkorange')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        ax.fill_between(range(len(derivative)), 0, derivative, 
                       where=(derivative >= 0), color='green', alpha=0.2,
                       label='dW/dt >= 0')
        ax.fill_between(range(len(derivative)), derivative, 0, 
                       where=(derivative < 0), color='red', alpha=0.2,
                       label='dW/dt < 0 (should not occur)')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('dW/dt', fontsize=12)
        ax.set_title('Entropy Derivative (Monotonicity Check)', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/perelman_entropy_evolution.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/perelman_entropy_evolution.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_curvature_distribution(self, history):
        """
        Figure 3: Distribution of Ollivier-Ricci curvature over time.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Panel 1: Heatmap of curvature distribution
        ax = axes[0]
        
        # Extract curvatures at different times
        curvatures = history['curvatures']
        n_times = min(50, len(curvatures))
        time_indices = np.linspace(0, len(curvatures)-1, n_times, dtype=int)
        
        curvature_matrix = []
        for idx in time_indices:
            vals = list(curvatures[idx].values())
            curvature_matrix.append(vals)
        
        # Pad to equal length
        max_len = max([len(vals) for vals in curvature_matrix])
        for i in range(len(curvature_matrix)):
            if len(curvature_matrix[i]) < max_len:
                curvature_matrix[i] = curvature_matrix[i] + [np.nan] * (max_len - len(curvature_matrix[i]))
        
        curvature_matrix = np.array(curvature_matrix)
        
        im = ax.imshow(curvature_matrix, aspect='auto', cmap='RdBu_r', 
                      vmin=-1, vmax=1)
        ax.set_xlabel('Edge index', fontsize=12)
        ax.set_ylabel('Time iteration', fontsize=12)
        ax.set_title('Curvature Evolution Heatmap', fontsize=14)
        plt.colorbar(im, ax=ax, label='Curvature')
        
        # Panel 2: Curvature statistics
        ax = axes[1]
        
        # Compute statistics over time
        mean_curv = history['mean_curvature']
        
        # Compute percentiles
        all_curvatures = []
        for curv in curvatures:
            all_curvatures.extend(list(curv.values()))
        
        if all_curvatures:
            percentiles = np.percentile(all_curvatures, [25, 50, 75])
            
            ax.hist(all_curvatures, bins=30, color='purple', alpha=0.7, 
                   edgecolor='black', density=True)
            
            ax.axvline(percentiles[0], color='blue', linestyle='--', 
                      linewidth=1.5, label='25th percentile')
            ax.axvline(percentiles[1], color='green', linestyle='--', 
                      linewidth=1.5, label='50th percentile (median)')
            ax.axvline(percentiles[2], color='red', linestyle='--', 
                      linewidth=1.5, label='75th percentile')
            
            ax.set_xlabel('Curvature', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title('Final Curvature Distribution', fontsize=14)
            ax.legend(loc='upper left')
        else:
            ax.text(0.5, 0.5, 'No curvature data available', 
                    ha='center', va='center', transform=ax.transAxes)
        
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/curvature_distribution.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/curvature_distribution.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_entropy_economic_correlation(self, entropy_history, economic_entropy_history):
        """
        Figure 4: Correlation between Perelman entropy and economic entropy.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Panel 1: Both entropies over time
        ax = axes[0]
        x = np.arange(len(entropy_history))
        
        # Normalize both
        ent_norm = (entropy_history - np.min(entropy_history)) / (np.max(entropy_history) - np.min(entropy_history))
        econ_norm = (economic_entropy_history - np.min(economic_entropy_history)) / (np.max(economic_entropy_history) - np.min(economic_entropy_history))
        
        ax.plot(x, ent_norm, linewidth=2.5, color='purple', 
                label='Perelman entropy (normalized)')
        ax.plot(x, econ_norm, linewidth=2.5, color='darkblue', 
                label='Economic entropy (normalized)')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Normalized entropy', fontsize=12)
        ax.set_title('Entropy Correspondence', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Correlation scatter plot
        ax = axes[1]
        
        # Resample to equal length
        min_len = min(len(entropy_history), len(economic_entropy_history))
        ent_sample = entropy_history[:min_len]
        econ_sample = economic_entropy_history[:min_len]
        
        ax.scatter(ent_sample, econ_sample, alpha=0.5, s=10, color='darkgreen')
        
        # Fit linear regression
        coeff = np.polyfit(ent_sample, econ_sample, 1)
        x_fit = np.linspace(np.min(ent_sample), np.max(ent_sample), 100)
        y_fit = coeff[0] * x_fit + coeff[1]
        
        ax.plot(x_fit, y_fit, '--', linewidth=2, color='red',
                label=f'$S_{{total}} = {coeff[0]:.3f} \cdot W + {coeff[1]:.3f}$')
        
        # Compute correlation
        corr = np.corrcoef(ent_sample, econ_sample)[0, 1]
        
        ax.set_xlabel('Perelman entropy W', fontsize=12)
        ax.set_ylabel('Economic entropy S_total', fontsize=12)
        ax.set_title(f'Correlation: $r = {corr:.3f}$', fontsize=14)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/entropy_economic_correlation.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/entropy_economic_correlation.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_unified_stability(self, stability_history):
        """
        Figure 5: Unified stability: L_econ + W convergence.
        """
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        
        # Panel 1: Stability measure
        ax = axes[0]
        ax.plot(stability_history, linewidth=2.5, color='darkgreen')
        
        # Add convergence line
        if len(stability_history) > 10:
            last_vals = stability_history[-10:]
            stable_value = np.mean(last_vals)
            ax.axhline(y=stable_value, color='red', linestyle='--', linewidth=1.5,
                      label=f'Steady state: {stable_value:.4f}')
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Unified stability measure', fontsize=12)
        ax.set_title('Unified Stability: $\mathcal{L}_{econ} + \mathcal{W}$', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Convergence rate (log scale)
        ax = axes[1]
        
        # Compute convergence to steady state
        if len(stability_history) > 10:
            diff = np.abs(np.array(stability_history) - stable_value)
            ax.semilogy(diff, linewidth=2.5, color='darkred')
            
            # Fit exponential
            if len(diff) > 20:
                log_diff = np.log(diff[10:] + 1e-10)
                x = np.arange(len(log_diff))
                coeff = np.polyfit(x, log_diff, 1)
                rate = -coeff[0]
                ax.set_title(f'Convergence Rate: $\lambda = {rate:.4f}$', fontsize=14)
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('|Stability - Steady State|', fontsize=12)
        ax.set_title('Convergence Rate (Log Scale)', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/unified_stability.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/unified_stability.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_network_visualization(self, G_initial, G_final):
        """
        Additional figure: Network visualization before and after Ricci flow.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Panel 1: Initial network
        ax = axes[0]
        pos = nx.spring_layout(G_initial, seed=42)
        
        # Edge weights for color mapping
        weights_initial = [G_initial[u][v].get('weight', 1.0) for u, v in G_initial.edges()]
        
        nx.draw_networkx_nodes(G_initial, pos, ax=ax, node_size=100, 
                              node_color='lightblue', edgecolors='black')
        nx.draw_networkx_edges(G_initial, pos, ax=ax, 
                              edge_color=weights_initial, edge_cmap=plt.cm.Blues,
                              edge_vmin=0, edge_vmax=1, width=2)
        ax.set_title('Initial Network', fontsize=14)
        ax.set_aspect('equal')
        
        # Panel 2: Final network
        ax = axes[1]
        pos = nx.spring_layout(G_final, seed=42)
        
        weights_final = [G_final[u][v].get('weight', 1.0) for u, v in G_final.edges()]
        
        nx.draw_networkx_nodes(G_final, pos, ax=ax, node_size=100,
                              node_color='lightgreen', edgecolors='black')
        nx.draw_networkx_edges(G_final, pos, ax=ax,
                              edge_color=weights_final, edge_cmap=plt.cm.Greens,
                              edge_vmin=0, edge_vmax=1, width=2)
        ax.set_title('Final Network (After Ricci Flow)', fontsize=14)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/network_evolution.pdf', format='pdf')
        plt.savefig(f'{self.output_dir}/network_evolution.png', format='png', dpi=300)
        plt.close()
        
        return fig
    
    def figure_all(self, history, entropy_history, economic_entropy_history,
                   stability_history, G_initial=None, G_final=None):
        """
        Generate all figures for the article.
        """
        print("Generating Figure 1: Ricci flow convergence...")
        self.figure_ricci_flow_convergence(history)
        
        print("Generating Figure 2: Perelman entropy evolution...")
        self.figure_perelman_entropy_evolution(entropy_history)
        
        print("Generating Figure 3: Curvature distribution...")
        self.figure_curvature_distribution(history)
        
        print("Generating Figure 4: Entropy-economic correlation...")
        self.figure_entropy_economic_correlation(entropy_history, economic_entropy_history)
        
        print("Generating Figure 5: Unified stability...")
        self.figure_unified_stability(stability_history)
        
        if G_initial is not None and G_final is not None:
            print("Generating Figure 6: Network visualization...")
            self.figure_network_visualization(G_initial, G_final)
        
        print("All figures generated successfully!")
