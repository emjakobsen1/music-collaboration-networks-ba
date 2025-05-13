import networkx as nx
from collections import defaultdict
import csv

SUPERGENRES = [
    'r&b', 'folk-world', 'rock', 'jazz-blues', 'pop', 'hip-hop', 'electronic', 
    'classical', 'religious', 'other', 'caribbean-latin-america', 'country', 'asian'
]

def compute_weighted_betweenness(G, year):
    """
    Computes approximate betweenness centrality and returns metrics for export.
    """
    if not isinstance(G, nx.Graph):
        raise ValueError("Graph must be an undirected networkx.Graph")

    Gcc_nodes = max(nx.connected_components(G), key=len)
    Gcc = G.subgraph(Gcc_nodes).copy()
    G_sub = G

    print(f"[{year}] Full Graph Nodes: {G_sub.number_of_nodes()}, Edges: {G_sub.number_of_edges()}")
    print(f"[{year}] Giant Component Nodes: {Gcc.number_of_nodes()}, Edges: {Gcc.number_of_edges()}")

    centrality = nx.betweenness_centrality(G_sub, weight='weight', k=400, seed=61)
    nx.set_node_attributes(G_sub, centrality, 'centrality')

    genre_centrality = defaultdict(list)
    for u, v, data in G_sub.edges(data=True):
        genre = data.get('genre', 'other')
        genre_centrality[genre].append(G_sub.nodes[u].get('centrality', 0.0))
        genre_centrality[genre].append(G_sub.nodes[v].get('centrality', 0.0))

    row = {
        'Year': year,
        'Nodes': G_sub.number_of_nodes(),
        'Edges': G_sub.number_of_edges(),
        'GC Nodes': Gcc.number_of_nodes(),
        'GC Edges': Gcc.number_of_edges()
    }

    for genre in SUPERGENRES:
        values = genre_centrality.get(genre, [])
        avg = sum(values) / len(values) if values else 0.0
        row[genre] = f"{avg:.10f}"

    return row

def compute_top_betweenness_summary(G, year, top_n=10):
    """
    Computes betweenness centrality for full graph and returns summary with:
    - Graph stats
    - Avg centrality (full + GC)
    - Top-N nodes by centrality
    """
    if not isinstance(G, nx.Graph):
        raise ValueError("Graph must be an undirected networkx.Graph")

    Gcc_nodes = max(nx.connected_components(G), key=len)
    Gcc = G.subgraph(Gcc_nodes).copy()

    print(f"[{year}] Full Graph — Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"[{year}] Giant Component — Nodes: {Gcc.number_of_nodes()}, Edges: {Gcc.number_of_edges()}")

    print("Calculating betweenness centrality on full graph...")
    full_bc = nx.betweenness_centrality(G, weight='weight', k=1500, seed=61)

    print("Calculating betweenness centrality on giant component...")
    #gc_bc = nx.betweenness_centrality(Gcc, weight='weight', k=None, seed=61)

    
    top_nodes = sorted(full_bc.items(), key=lambda x: x[1], reverse=True)[:top_n]

    
    avg_full = sum(full_bc.values()) / len(full_bc)
    #avg_gc = sum(gc_bc.values()) / len(gc_bc)

    row = {
        'Year': year,
        'Nodes': G.number_of_nodes(),
        'Edges': G.number_of_edges(),
        'GC Nodes': Gcc.number_of_nodes(),
        'GC Edges': Gcc.number_of_edges(),
        'Avg BC (Full)': f"{avg_full:.10f}",
        #'Avg BC (GC)': f"{avg_gc:.10f}"
    }

    print(f"\n🎯 Top {top_n} Nodes by Betweenness Centrality ({year}):")
    for i, (node_id, score) in enumerate(top_nodes, 1):
        name = G.nodes[node_id].get('name', f'Node {node_id}')
        row[f'Top {i}'] = name
        print(f"{i:2}. {name:<30} | BC: {score:.6f}")

    return row



import os

def export_to_csv(results, filepath):
    fieldnames = ['Year', 'Nodes', 'Edges', 'GC Nodes', 'GC Edges'] + SUPERGENRES
    file_exists = os.path.isfile(filepath)

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  

        for row in results:
            writer.writerow(row)

def export_top_bc_summary(results, filepath, top_n=10):
    fieldnames = [
        'Year', 'Nodes', 'Edges', 'GC Nodes', 'GC Edges',
        'Avg BC (Full)', 'Avg BC (GC)'
    ] + [f'Top {i}' for i in range(1, top_n + 1)]

    file_exists = os.path.isfile(filepath)

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  

        for row in results:
            writer.writerow(row)