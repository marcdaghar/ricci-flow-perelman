"""
Graph utilities for generating synthetic and real networks.
"""

import numpy as np
import networkx as nx

class GraphGenerator:
    """
    Generate synthetic graphs for Ricci flow simulations.
    """
    
    @staticmethod
    def small_world_graph(n_nodes=100, k=10, p=0.1):
        """
        Generate Watts-Strogatz small-world graph.
        """
        G = nx.watts_strogatz_graph(n_nodes, k, p)
        
        # Add random weights
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.5, 1.0)
        
        return G
    
    @staticmethod
    def scale_free_graph(n_nodes=100, m=5):
        """
        Generate Barabási-Albert scale-free graph.
        """
        G = nx.barabasi_albert_graph(n_nodes, m)
        
        # Add random weights
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.5, 1.0)
        
        return G
    
    @staticmethod
    def erdos_renyi_graph(n_nodes=100, p=0.05):
        """
        Generate Erdős-Rényi random graph.
        """
        G = nx.erdos_renyi_graph(n_nodes, p)
        
        # Add random weights
        for u, v in G.edges():
            G[u][v]['weight'] = np.random.uniform(0.5, 1.0)
        
        return G
    
    @staticmethod
    def community_graph(n_nodes=100, n_communities=5, p_in=0.3, p_out=0.02):
        """
        Generate graph with community structure.
        """
        G = nx.Graph()
        
        # Create communities
        community_size = n_nodes // n_communities
        communities = []
        
        for c in range(n_communities):
            nodes = list(range(c * community_size, (c + 1) * community_size))
            communities.append(nodes)
            
            # Add nodes
            for node in nodes:
                G.add_node(node)
            
            # Intra-community edges
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    if np.random.random() < p_in:
                        G.add_edge(nodes[i], nodes[j], weight=np.random.uniform(0.5, 1.0))
        
        # Inter-community edges
        for c1 in range(n_communities):
            for c2 in range(c1 + 1, n_communities):
                for u in communities[c1]:
                    for v in communities[c2]:
                        if np.random.random() < p_out:
                            G.add_edge(u, v, weight=np.random.uniform(0.3, 0.6))
        
        return G
    
    @staticmethod
    def load_real_network(filename):
        """
        Load a real network from file.
        """
        G = nx.read_edgelist(filename)
        
        # Add weights if not present
        for u, v in G.edges():
            if 'weight' not in G[u][v]:
                G[u][v]['weight'] = 1.0
        
        return G
    
    @staticmethod
    def generate_social_network(n_nodes=100, n_cliques=10, clique_size=10):
        """
        Generate a graph resembling a social network.
        """
        G = nx.Graph()
        
        # Add nodes
        G.add_nodes_from(range(n_nodes))
        
        # Create cliques (communities)
        for c in range(n_cliques):
            start = c * clique_size
            end = min((c + 1) * clique_size, n_nodes)
            clique_nodes = list(range(start, end))
            
            for i in range(len(clique_nodes)):
                for j in range(i + 1, len(clique_nodes)):
                    G.add_edge(clique_nodes[i], clique_nodes[j], weight=np.random.uniform(0.7, 1.0))
        
        # Add random connections between cliques
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if not G.has_edge(i, j):
                    if np.random.random() < 0.02:
                        G.add_edge(i, j, weight=np.random.uniform(0.2, 0.4))
        
        return G
