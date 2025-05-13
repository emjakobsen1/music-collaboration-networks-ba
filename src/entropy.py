import pandas as pd
import networkx as nx
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import math

def compute_entropy(G, min_degree=0):
    node_entropies = []

    sum_j_h = Counter()

    
    for node in G.nodes():
        if G.degree[node] < min_degree:
            continue

        neighborhood_genres = []
        for neighbor in G.neighbors(node):
            neighborhood_genres.extend(G[node][neighbor].get("genres", []))

        h_j = Counter(neighborhood_genres)
        sum_j_h.update(h_j)

    
    for node in G.nodes():
        if G.degree[node] < min_degree:
            continue

        neighborhood_genres = []
        for neighbor in G.neighbors(node):
            neighborhood_genres.extend(G[node][neighbor].get("genres", []))

        h_j = Counter(neighborhood_genres)

        bar_h_j = {genre: count / sum_j_h.get(genre, 1) for genre, count in h_j.items()}
        sum_bar = sum(bar_h_j.values())
        P_j = {genre: val / sum_bar for genre, val in bar_h_j.items()} if sum_bar else {}
        entropy = -sum(p * math.log(p) for p in P_j.values() if p > 0)

        node_entropies.append(entropy)

    return node_entropies


def analyze_mean_entropy_over_years(data, start_year=1986, end_year=2016, min_degree=1):
    results = []

    for year in range(start_year, end_year + 1):
        print(f"Processing year {year}...")
        G = build_artist_collab_graph2(data, year_range=(year, year))
        Gcc_nodes = max(nx.connected_components(G), key=len)
        Gcc = G.subgraph(Gcc_nodes).copy()
        G = Gcc
        entropies = compute_entropy(G, min_degree=min_degree)
        mean_entropy = sum(entropies) / len(entropies) if entropies else None

        results.append({"Year": year, "MeanEntropy": mean_entropy})

    df = pd.DataFrame(results)
    df.to_csv("mean_entropy_over_years.csv", index=False)
    print("Saved to mean_entropy_over_years.csv")



def plot_mean_entropy(csv_path="mean_entropy_over_years.csv"):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(12, 6))
    plt.plot(df['Year'], df['MeanEntropy'], marker='o', linewidth=2, color='darkblue')

    plt.title("Mean Genre Entropy Over Time", fontsize=14)
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Mean Entropy", fontsize=12)
    plt.xticks(df['Year'], rotation=45)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()





if __name__ == "__main__":
    from visualize import build_artist_collab_graph2, load_data_from_files

    data_folder = '../data_processed'
    data = load_data_from_files(data_folder)

    analyze_mean_entropy_over_years(data)
    plot_mean_entropy()
    
