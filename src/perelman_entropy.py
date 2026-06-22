"""
Perelman Entropy for Discrete Graphs.
Implements the Perelman entropy functional and its properties.
"""

import numpy as np
import networkx as nx
from scipy.spatial.distance import cdist

class PerelmanEntropy:
    """
    Perelman entropy on discrete graphs.
    
    Key relationships:
        W(g,f,τ) = Σ_i [τ(κ̄_i + |∇f_i|²) + f_i - n] · μ_i
        dW/dt ≥ 0  # Monotonicity
    """
    
    def __init__(self, tau=1.0, n_dim=2):
        """
        Args:
            tau: Scale parameter
            n_dim: Dimension of the manifold (embedding dimension)
        """
        self.tau = tau
        self.n_dim = n_dim
    
    def compute_curvature_scalar(self, G):
        """
        Compute scalar curvature (average curvature at each node).
        
        Returns:
            Array of scalar curvatures at each node
        """
        # Compute Ollivier-Ricci curvatures
        curvatures = self._compute_ollivier_ricci(G)
        
        # Average curvature at each node
        scalar_curvature = []
        nodes = list(G.nodes())
        
        for node in nodes:
            node_curvatures = []
            for neighbor in G.neighbors(node):
                if (node, neighbor) in curvatures:
                    node_curvatures.append(curvatures[(node, neighbor)])
                elif (neighbor, node) in curvatures:
                    node_curvatures.append(curvatures[(neighbor, node)])
            
            if node_curvatures:
                scalar_curvature.append(np.mean(node_curvatures))
            else:
                scalar_curvature.append(0.0)
        
        return np.array(scalar_curvature)
    
    def _compute_ollivier_ricci(self, G):
        """
        Compute Ollivier-Ricci curvature for all edges.
        """
        curvatures = {}
        
        for u, v in G.edges():
            deg_u = G.degree(u)
            deg_v = G.degree(v)
            
            if deg_u == 0 or deg_v == 0:
                curvatures[(u, v)] = 0.0
                continue
            
            # Wasserstein-1 distance (approximated)
            dist = np.abs(deg_u - deg_v) / max(deg_u, deg_v)
            
            # Edge weight
            weight = G[u][v].get('weight', 1.0)
            d_uv = 1.0 / weight if weight > 0 else 1.0
            
            curvature = 1.0 - dist / (d_uv + 1e-10)
            curvature = np.clip(curvature, -1.0, 1.0)
            
            curvatures[(u, v)] = curvature
        
        return curvatures
    
    def compute_gradient_norm(self, G, f):
        """
        Compute gradient norm of potential function f.
        
        Args:
            G: NetworkX graph
            f: Potential function (array of node values)
        
        Returns:
            Array of gradient norms at each node
        """
        nodes = list(G.nodes())
        gradient_norm = np.zeros(len(nodes))
        
        for i, node in enumerate(nodes):
            neighbors = list(G.neighbors(node))
            if len(neighbors) > 0:
                f_neighbors = [f[nodes.index(n)] for n in neighbors]
                # Use variance as proxy for gradient norm
                gradient_norm[i] = np.std(f_neighbors)
            else:
                gradient_norm[i] = 0.0
        
        return gradient_norm
    
    def compute_measure(self, G):
        """
        Compute normalized measure μ_i = deg(i) / (2|E|).
        
        Returns:
            Array of measures at each node
        """
        n_edges = G.number_of_edges()
        if n_edges == 0:
            return np.ones(G.number_of_nodes()) / G.number_of_nodes()
        
        measures = np.array([G.degree(node) for node in G.nodes()])
        measures = measures / (2 * n_edges)
        
        return measures
    
    def compute_entropy(self, G, f=None):
        """
        Compute Perelman entropy W(g, f, τ).
        
        Args:
            G: NetworkX graph
            f: Potential function (if None, use degree centrality)
        
        Returns:
            Perelman entropy value
        """
        nodes = list(G.nodes())
        n = len(nodes)
        
        # Scalar curvature
        R = self.compute_curvature_scalar(G)
        
        # Potential function
        if f is None:
            # Use degree centrality as potential
            f = np.array([G.degree(node) for node in nodes])
            f = f / np.max(f) if np.max(f) > 0 else np.zeros_like(f)
        
        # Normalize f
        f = f - np.mean(f)
        
        # Gradient norm
        gradient_norm = self.compute_gradient_norm(G, f)
        
        # Measure
        mu = self.compute_measure(G)
        
        # Entropy
        entropy = 0.0
        for i in range(n):
            term = self.tau * (R[i] + gradient_norm[i]**2) + f[i] - self.n_dim
            entropy += term * mu[i]
        
        return entropy
    
    def compute_entropy_derivative(self, G, f=None):
        """
        Compute derivative of Perelman entropy.
        
        Returns:
            dW/dt value (should be >= 0 for valid Ricci flow)
        """
        # This is a simplified implementation
        # In practice, we would compute the full expression
        
        entropy_prev = self.compute_entropy(G, f)
        
        # Perturb graph slightly
        G_perturbed = G.copy()
        for u, v in G_perturbed.edges():
            current_weight = G_perturbed[u][v].get('weight', 1.0)
            G_perturbed[u][v]['weight'] = current_weight * (1 + 1e-4)
        
        entropy_next = self.compute_entropy(G_perturbed, f)
        
        derivative = (entropy_next - entropy_prev) / 1e-4
        
        return derivative
    
    def compute_entropy_evolution(self, G, n_steps=100, dt=0.01):
        """
        Compute entropy evolution under Ricci flow.
        
        Returns:
            Entropy history
        """
        entropy_history = []
        graph = G.copy()
        f = None
        
        for step in range(n_steps):
            # Compute current entropy
            entropy = self.compute_entropy(graph, f)
            entropy_history.append(entropy)
            
            # Update graph (simple Ricci flow)
            curvatures = self._compute_ollivier_ricci(graph)
            
            for u, v in graph.edges():
                curvature = curvatures.get((u, v), 0.0)
                current_weight = graph[u][v].get('weight', 1.0)
                
                d_weight = -curvature * current_weight
                new_weight = max(0.01, current_weight + d_weight * dt)
                graph[u][v]['weight'] = new_weight
        
        return entropy_history
    
    def compute_unified_stability(self, G, economic_entropy, n_steps=100, dt=0.01):
        """
        Compute unified stability measure: L_econ + W.
        
        Args:
            G: NetworkX graph
            economic_entropy: Economic entropy function (from Article 4)
            n_steps: Number of steps
            dt: Time step
        
        Returns:
            Unified stability history
        """
        stability_history = []
        graph = G.copy()
        
        for step in range(n_steps):
            # Perelman entropy
            perelman = self.compute_entropy(graph)
            
            # Economic entropy (simulated)
            econ = economic_entropy(graph) if callable(economic_entropy) else 0.0
            
            # Unified stability
            stability = econ + perelman
            stability_history.append(stability)
            
            # Update graph
            curvatures = self._compute_ollivier_ricci(graph)
            
            for u, v in graph.edges():
                curvature = curvatures.get((u, v), 0.0)
                current_weight = graph[u][v].get('weight', 1.0)
                
                d_weight = -curvature * current_weight
                new_weight = max(0.01, current_weight + d_weight * dt)
                graph[u][v]['weight'] = new_weight
        
        return stability_history
