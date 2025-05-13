import pandas as pd
import networkx as nx
from visualize import build_artist_collab_graph2, load_data_from_files


data_folder = '../data_processed'
data = load_data_from_files(data_folder)

rows = []

for year in range(1986, 2017):
    print(f"Processing year {year}...")
    G = build_artist_collab_graph2(data, year_range=(year, year))

    
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_deg = sum(dict(G.degree()).values()) / num_nodes if num_nodes > 0 else 0

    
    if nx.is_connected(G):
        G_gcc = G
    else:
        gcc_nodes = max(nx.connected_components(G), key=len)
        G_gcc = G.subgraph(gcc_nodes).copy()

    gcc_nodes_count = G_gcc.number_of_nodes()
    gcc_edges_count = G_gcc.number_of_edges()
    gcc_avg_deg = sum(dict(G_gcc.degree()).values()) / gcc_nodes_count if gcc_nodes_count > 0 else 0

    # Add to table
    rows.append({
        "Year": year,
        "Nodes": num_nodes,
        "Edges": num_edges,
        "GCC Nodes": gcc_nodes_count,
        "GCC Edges": gcc_edges_count,
        "Avg. Degree": round(avg_deg, 3),
        "GCC Avg. Degree": round(gcc_avg_deg, 3)
    })


df = pd.DataFrame(rows)
df.to_csv("graph_summary_by_year.csv", index=False)

