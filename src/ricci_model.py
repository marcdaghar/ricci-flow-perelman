"""
Cognitive Ricci Flow Model.
Implements the Ricci flow on graphs with Ollivier-Ricci curvature.
"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

class RicciFlowModel:
    """
    Cognitive Ricci Flow on graphs.
    
    Key relationships:
        κ_{ij} = 1 - W_1(μ_i, μ_j) / d(i, j)  # Ollivier-Ricci curvature
        dw_{ij}/dt = -κ_{ij}·w_{ij} + λ·∇_wL - μ·(w_{ij} - w_{ij}⁰)
    """
    
    def __init__(self, lambda_=0.1, mu=0.01, tau=1.0):
        """
        Args:
            lambda_: Learning rate (adaptation strength)
            mu: Damping coefficient (stabilization strength)
            tau: Scale parameter
        """
        self.lambda_ = lambda_
        self.mu = mu
        self.tau = tau
    
    def compute_ollivier_ricci_curvature(self, G, alpha=0.5):
        """
        Compute Ollivier-Ricci curvature for all edges.
        
        Args:
            G: NetworkX graph with edge weights
            alpha: Parameter for lazy random walk (0 < alpha < 1)
        
        Returns:
            Dictionary of edge curvatures
        """
        curvatures = {}
        
        for u, v in G.edges():
            # Get neighborhoods
            neighbors_u = list(G.neighbors(u))
            neighbors_v = list(G.neighbors(v))
            
            # Compute degree-weighted distributions
            deg_u = G.degree(u)
            deg_v = G.degree(v)
            
            if deg_u == 0 or deg_v == 0:
                curvatures[(u, v)] = 0.0
                continue
            
            # Compute Wasserstein-1 distance between distributions
            # Simplified: use degree difference
            dist = np.abs(deg_u - deg_v) / max(deg_u, deg_v)
            
            # Ollivier-Ricci curvature
            weight = G[u][v].get('weight', 1.0)
            d_uv = 1.0 / weight if weight > 0 else 1.0
            
            curvature = 1.0 - dist / (d_uv + 1e-10)
            curvature = np.clip(curvature, -1.0, 1.0)
            
            curvatures[(u, v)] = curvature
        
        return curvatures
    
    def compute_wasserstein_distance(self, G, u, v):
        """
        Compute Wasserstein-1 distance between neighborhoods.
        
        Args:
            G: NetworkX graph
            u, v: Nodes
        
        Returns:
            Wasserstein-1 distance
        """
        neighbors_u = list(G.neighbors(u))
        neighbors_v = list(G.neighbors(v))
        
        if len(neighbors_u) == 0 or len(neighbors_v) == 0:
            return 1.0
        
        # Create distributions (uniform over neighbors)
        mu_u = np.ones(len(neighbors_u)) / len(neighbors_u)
        mu_v = np.ones(len(neighbors_v)) / len(neighbors_v)
        
        # Cost matrix (shortest path distances)
        cost = np.zeros((len(neighbors_u), len(neighbors_v)))
        for i, nu in enumerate(neighbors_u):
            for j, nv in enumerate(neighbors_v):
                try:
                    cost[i, j] = nx.shortest_path_length(G, nu, nv)
                except nx.NetworkXNoPath:
                    cost[i, j] = 1e6
        
        # Solve optimal transport (approximate)
        # Use Sinkhorn or simple assignment
        if len(neighbors_u) <= len(neighbors_v):
            row_ind, col_ind = linear_sum_assignment(cost)
            wasserstein = cost[row_ind, col_ind].sum() / len(neighbors_u)
        else:
            row_ind, col_ind = linear_sum_assignment(cost.T)
            wasserstein = cost.T[row_ind, col_ind].sum() / len(neighbors_v)
        
        return wasserstein
    
    def compute_curvature_distribution(self, G):
        """
        Compute distribution of Ollivier-Ricci curvatures.
        """
        curvatures = self.compute_ollivier_ricci_curvature(G)
        values = list(curvatures.values())
        
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'histogram': np.histogram(values, bins=20)
        }
    
    def evolve_ricci_flow(self, G, n_steps=100, dt=0.01,
                          loss_function=None, verbose=True):
        """
        Evolve the graph under Ricci flow.
        
        Args:
            G: Initial NetworkX graph
            n_steps: Number of evolution steps
            dt: Time step
            loss_function: Function to compute loss (for adaptation term)
            verbose: Print progress
        
        Returns:
            Dictionary with results
        """
        # Copy graph
        G_evolved = G.copy()
        
        # Initialize history
        history = {
            'curvatures': [],
            'weights': [],
            'entropy': [],
            'mean_curvature': [],
            'loss': []
        }
        
        for step in range(n_steps):
            # Compute curvatures
            curvatures = self.compute_ollivier_ricci_curvature(G_evolved)
            history['curvatures'].append(curvatures)
            
            # Compute loss
            if loss_function is not None:
                loss = loss_function(G_evolved)
                history['loss'].append(loss)
            else:
                loss = 0.0
            
            # Update edge weights
            for u, v in G_evolved.edges():
                curvature = curvatures.get((u, v), 0.0)
                current_weight = G_evolved[u][v].get('weight', 1.0)
                initial_weight = G[u][v].get('weight', 1.0)
                
                # Ricci flow update
                d_weight = (-curvature * current_weight + 
                           self.lambda_ * loss - 
                           self.mu * (current_weight - initial_weight))
                
                new_weight = max(0.01, current_weight + d_weight * dt)
                G_evolved[u][v]['weight'] = new_weight
            
            # Compute mean curvature
            mean_curv = np.mean(list(curvatures.values()))
            history['mean_curvature'].append(mean_curv)
            
            # Store weights snapshot
            if step % 10 == 0:
                weights = np.array([G_evolved[u][v]['weight'] 
                                   for u, v in G_evolved.edges()])
                history['weights'].append(weights)
            
            if verbose and step % 10 == 0:
                print(f"Step {step}: mean curvature = {mean_curv:.4f}")
        
        history['final_graph'] = G_evolved
        
        return history
    
    def compute_perelman_entropy(self, G, f=None, tau=None):
        """
        Compute discrete Perelman entropy.
        
        Args:
            G: NetworkX graph
            f: Potential function (node values)
            tau: Scale parameter
        
        Returns:
            Perelman entropy value
        """
        if tau is None:
            tau = self.tau
        
        n = G.number_of_nodes()
        
        # Compute curvatures
        curvatures = self.compute_ollivier_ricci_curvature(G)
        
        # If f is not provided, use degree centrality
        if f is None:
            f = np.array([G.degree(node) for node in G.nodes()])
            f = f / np.max(f) if np.max(f) > 0 else np.zeros_like(f)
        
        # Normalize f
        f = f - np.mean(f)
        
        # Compute gradient of f (simplified)
        gradient_norm = np.zeros(n)
        nodes = list(G.nodes())
        
        for i, node in enumerate(nodes):
            neighbors = list(G.neighbors(node))
            if len(neighbors) > 0:
                f_neighbors = [f[nodes.index(n)] for n in neighbors]
                gradient_norm[i] = np.std(f_neighbors)
        
        # Compute Perelman entropy
        entropy = 0.0
        for i, node in enumerate(nodes):
            # Average curvature at node
            node_curvatures = []
            for neighbor in G.neighbors(node):
                if (node, neighbor) in curvatures:
                    node_curvatures.append(curvatures[(node, neighbor)])
                elif (neighbor, node) in curvatures:
                    node_curvatures.append(curvatures[(neighbor, node)])
            
            if node_curvatures:
                avg_curv = np.mean(node_curvatures)
            else:
                avg_curv = 0.0
            
            # Contribution to entropy
            mu_i = G.degree(node) / (2 * G.number_of_edges() + 1e-10)
            term = tau * (avg_curv + gradient_norm[i]**2) + f[i] - n
            entropy += term * mu_i
        
        return entropy / n
    
    def compute_entropy_evolution(self, G, n_steps=100, dt=0.01,
                                  verbose=True):
        """
        Compute Perelman entropy evolution under Ricci flow.
        
        Returns:
            Entropy history
        """
        results = self.evolve_ricci_flow(G, n_steps, dt, verbose=False)
        
        entropy_history = []
        graph = G.copy()
        
        for step in range(n_steps):
            # Compute entropy at current state
            entropy = self.compute_perelman_entropy(graph)
            entropy_history.append(entropy)
            
            # Update graph (reuse evolution logic)
            curvatures = self.compute_ollivier_ricci_curvature(graph)
            
            for u, v in graph.edges():
                curvature = curvatures.get((u, v), 0.0)
                current_weight = graph[u][v].get('weight', 1.0)
                initial_weight = G[u][v].get('weight', 1.0)
                
                d_weight = (-curvature * current_weight - 
                           self.mu * (current_weight - initial_weight))
                
                new_weight = max(0.01, current_weight + d_weight * dt)
                graph[u][v]['weight'] = new_weight
        
        return entropy_history, results
