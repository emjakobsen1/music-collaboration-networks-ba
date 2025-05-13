import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import re
import random
import math
import copy
import numpy as np
import seaborn as sns
from collections import Counter
from collections import defaultdict
from genre_classifier import classify_genre, print_classification_summary, plot_genre_treemap, classification_stats
from betweenness_centrality import compute_weighted_betweenness, export_to_csv,compute_top_betweenness_summary,export_top_bc_summary

def load_data_from_files(data_folder):
    data = []
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.json'):
            file_path = os.path.join(data_folder, file_name)
            print(f"Loading file: {file_name}")  
            with open(file_path, 'r', encoding='utf-8') as f:
                data.extend(json.load(f))  
    return data

def build_artist_collab_graph2(data, year_range):
    G = nx.Graph()
    start_year, end_year = year_range  

    total_songs = 0
    collab_songs = 0
    artist_raw_genres = {}  

    
    excluded_artists = {
        "Johann Sebastian Bach", "Wolfgang Amadeus Mozart", "Ludwig van Beethoven",
        "Franz Schubert", "Joseph Haydn", "Pyotr Ilyich Tchaikovsky",
        "Frédéric Chopin", "Antonio Vivaldi", "George Frideric Handel",
        "Igor Stravinsky", "Claude Debussy", "Richard Wagner",
        "Johannes Brahms", "Sergei Rachmaninoff", "Robert Schumann",
        "Giacomo Puccini", "Gustav Mahler", "Franz Liszt",
        "Antonín Dvořák", "Felix Mendelssohn", "Dmitri Shostakovich",
        "Sergei Prokofiev", "Camille Saint-Saëns", "Maurice Ravel",
        "Hector Berlioz"
    }

    for entry in data:
        release_year = entry.get("release_year")
        try:
            release_year = int(release_year)
        except (TypeError, ValueError):
            continue  

        if not (start_year <= release_year <= end_year):
            continue  

        artists = entry.get("artists", [])
        if len(artists) > 5:
            continue  

        total_songs += 1
        song_name = entry["track_name"]
        raw_genres = entry.get("genres", [])  

        
        classified_genres = [classify_genre(g) for g in raw_genres]
        main_genre = Counter(classified_genres).most_common(1)[0][0] if classified_genres else "other"

        artist_nodes = {artist["artist_id"]: artist["artist_name"] for artist in artists}

        for artist_id, artist_name in artist_nodes.items():
            if artist_name in excluded_artists:
                continue  

            if artist_id not in G:
                G.add_node(artist_id, name=artist_name, genre=main_genre, genres=[])  
            artist_raw_genres.setdefault(artist_id, []).extend(raw_genres)

        
        artist_ids = [aid for aid, aname in artist_nodes.items() if aname not in excluded_artists]

        if len(artist_ids) > 1:
            collab_songs += 1  
            for i in range(len(artist_ids)):
                for j in range(i + 1, len(artist_ids)):
                    artist1, artist2 = artist_ids[i], artist_ids[j]

                    if G.has_edge(artist1, artist2):
                        G[artist1][artist2]["songs"].add(song_name)
                        G[artist1][artist2]["weight"] += 1
                        G[artist1][artist2]["genre"].append(main_genre)
                        G[artist1][artist2]["genres"].extend(raw_genres)  
                    else:
                        G.add_edge(
                            artist1, artist2,
                            songs={song_name},
                            weight=1,
                            genre=[main_genre],
                            genres=raw_genres[:]  
                        )

    collab_percentage = (collab_songs / total_songs * 100) if total_songs > 0 else 0
    print("Collab percentage:", collab_percentage)

    
    for u, v, data in G.edges(data=True):
        genre_counts = Counter(data["genre"])
        G[u][v]["genre"] = genre_counts.most_common(1)[0][0]

    for node in G.nodes():
        raw = artist_raw_genres.get(node, [])
        G.nodes[node]["genres"] = raw

        if raw:
            classified = [classify_genre(g) for g in raw]
            G.nodes[node]["genre"] = Counter(classified).most_common(1)[0][0]
        else:
            G.nodes[node]["genre"] = "other"

    return G










def print_random_samples(G, sample_size=50, genre_filter=None):
    """
    Print a random sample of nodes (artists) and their connected artists,
    showing name, main genre, and raw genre list.

    Parameters:
    - G: The artist collaboration graph.
    - sample_size: Number of nodes to sample.
    - genre_filter: Optional filter by main genre.
    """
    print("\n--- Random Sample of Nodes (Artists) ---")

    # Filter nodes
    if genre_filter:
        filtered_nodes = [n for n in G.nodes() if G.nodes[n].get("genre") == genre_filter]
        print(f"🎯 Filtering nodes by main genre: '{genre_filter}' ({len(filtered_nodes)} nodes found)")
    else:
        filtered_nodes = list(G.nodes())

    if not filtered_nodes:
        print("⚠️ No nodes match the genre filter.")
        return

    nodes_sample = random.sample(filtered_nodes, min(sample_size, len(filtered_nodes)))

    for node in nodes_sample:
        node_name = G.nodes[node].get("name", "Unknown")
        node_main_genre = G.nodes[node].get("genre", "unknown")
        node_genres = G.nodes[node].get("genres", [])

        connected_artists = [
            G.nodes[neighbor].get("name", "Unknown") for neighbor in G.neighbors(node)
        ]

        print(f"🎤 Node: {node_name}")
        print(f"   ➤ Main Genre: {node_main_genre}")
        print(f"   ➤ All Genres (raw): {node_genres if node_genres else 'None'}")
        print(f"   ➤ Connected Artists: {', '.join(connected_artists) if connected_artists else 'None'}\n")



def print_nodes_by_entropy_range(G, entropy_data, min_entropy=0.0, max_entropy=1.0, sample_size=50):
    """
    Print a sample of nodes with entropy within a specified range.

    Parameters:
    - G: The artist collaboration graph.
    - entropy_data: Dict from compute_genre_entropy() function
    - min_entropy: Minimum entropy (inclusive)
    - max_entropy: Maximum entropy (inclusive)
    - sample_size: How many nodes to print
    """
    print(f"\n Nodes with entropy between {min_entropy} and {max_entropy}:\n")

    filtered_nodes = [
        node for node, data in entropy_data.items()
        if min_entropy <= data.get("entropy", 0) <= max_entropy
    ]

    if not filtered_nodes:
        print("No nodes match the entropy filter.")
        return

    nodes_sample = random.sample(filtered_nodes, min(sample_size, len(filtered_nodes)))

    for node in nodes_sample:
        node_name = G.nodes[node].get("name", "Unknown")
        node_genres = G.nodes[node].get("genres", [])
        node_main_genre = G.nodes[node].get("genre", "unknown")
        entropy = entropy_data[node].get("entropy", 0)

        connected_artists = []
        connected_artists_genres = []

        for neighbor in G.neighbors(node):
            connected_artists.append(G.nodes[neighbor].get("name", "Unknown"))
            edge_genres = G[node][neighbor].get("genres", [])
            connected_artists_genres.extend(edge_genres)

        print(f"Node: {node_name}")
        print(f"   Main Genre: {node_main_genre}")
        print(f"   Entropy: {entropy:.3f}")
        print(f"   All Genres (raw): {node_genres if node_genres else 'None'}")
        print(f"   Connected Artists: {', '.join(connected_artists) if connected_artists else 'None'}")
        print(f"   Edge Genres (raw, from all connections): {connected_artists_genres if connected_artists_genres else 'None'}")
        print(f"   Raw Edge Genre Count: {len(connected_artists_genres)}\n")



def count_total_songs(data):
    print(f"Total song entries: {len(data)}")

def count_unique_songs_and_artists(data):
    song_ids = set()
    artist_ids = set()
    
    for entry in data:
        song_ids.add(entry["track_id"])  
        for artist in entry.get("artists", []):
            artist_ids.add(artist["artist_id"])  
    
    return len(song_ids), len(artist_ids)

def get_genre_statistics(data):
    genre_counter = Counter()
    unique_genres_set = set()
   
    for entry in data:
        for genre in entry.get("genres", []):
            genre_counter[genre] += 1  
            unique_genres_set.add(genre)
    
    unique_genres = len(genre_counter)
    sorted_genres = genre_counter.most_common()
    
    print(f"Total unique genres: {unique_genres}")
    return genre_counter

def get_release_year_statistics(data):
    year_counter = Counter()
    
    for entry in data:
        release_year = entry.get("release_year")
        if release_year:  
            year_counter[release_year] += 1  
    
    sorted_years = year_counter.most_common()
    
    for year, count in sorted_years[:50]:
        print(f"{year}: {count}")
    
    return year_counter

def graph_songs_to_artists (data):
    song_graph = nx.Graph()


    for entry in data:
        song_id = entry["track_id"]
        artists = entry.get("artists", [])

        song_graph.add_node(song_id)  
        for artist in artists:
            artist_id = artist["artist_id"]
            song_graph.add_edge(song_id, artist_id) 
    
    print(f"Number of nodes: {len(song_graph.nodes)}")
    print(f"Number of edges: {len(song_graph.edges)}")
    return song_graph

def export_graphML(G, _filename):

    
   
    for u, v, data in G.edges(data=True):
        for key, value in data.items():
            if isinstance(value, (set, list)):  
                data[key] = ", ".join(map(str, value))

    for node, data in G.nodes(data=True):
        for key, value in data.items():
            if isinstance(value, (set, list)):  
                data[key] = ", ".join(map(str, value))


    filename = f"../cytographs/graph{_filename}.graphml"
    nx.write_graphml(G, filename)
    print(f"Graph saved as {filename}")


def frequency_plot (genre_counts):
    sorted_genres = genre_counts.most_common()  
    genres = [genre for genre, count in sorted_genres]
    counts = [count for genre, count in sorted_genres]
    plt.figure(figsize=(10, 6))
    plt.barh(genres, counts, color='skyblue')

    plt.xlabel('Frequency')
    plt.ylabel('Genres')
    plt.title('Genre Frequency')

    
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=4)  
    plt.yticks(range(0, len(genres), 5), [genres[i] for i in range(0, len(genres), 5)])
    plt.tight_layout()
    plt.show()

#Redistribute genres randomly
def apply_null_model(G):
    G_null = copy.deepcopy(G)  

    all_genres = []
    for u, v in G.edges():
        all_genres.extend(G[u][v].get("genres", []))

    
    random.shuffle(all_genres)

    
    genre_iter = iter(all_genres)
    for u, v in G_null.edges():
        num_genres = len(G[u][v].get("genres", []))
        G_null[u][v]["genres"] = [next(genre_iter) for _ in range(num_genres)]

    return G_null

def compute_genre_entropy(G, min_degree=0):
    node_entropy_data = {}
    sum_j_h = Counter()

    
    for node in G.nodes():
        if G.degree[node] < min_degree:
            continue  

        neighborhood_genres = []
        for neighbor in G.neighbors(node):
            neighborhood_genres.extend(G[node][neighbor].get("genres", []))

        h_j = Counter(neighborhood_genres)
        sum_j_h.update(h_j)

        node_entropy_data[node] = {
            "h": h_j
        }

   
    for node in node_entropy_data:
        h_j = node_entropy_data[node]["h"]

        # bar_h_j(c) = h_j(c) / sum_j h_j(c)
        bar_h_j = {}
        for genre, count in h_j.items():
            total_count_c = sum_j_h.get(genre, 0)
            if total_count_c > 0:
                bar_h_j[genre] = count / total_count_c

        node_entropy_data[node]["bar_h"] = bar_h_j

        # Compute P_j(c)
        sum_bar = sum(bar_h_j.values())
        P_j = {}
        if sum_bar > 0:
            for genre, val in bar_h_j.items():
                P_j[genre] = val / sum_bar
        else:
            P_j = {}

        
        entropy = -sum(p * math.log(p) for p in P_j.values() if p > 0)

        
        node_entropy_data[node]["P"] = P_j
        node_entropy_data[node]["entropy"] = entropy

    
    entropy_values = [data["entropy"] for data in node_entropy_data.values() if "entropy" in data]


    return node_entropy_data

def inspect_artists_by_genre(G, genre_filter):
    """
    Prints detailed info for nodes matching a genre, including name, genres,
    collaborators, and size of their connected component.
    """
    print(f"\n Inspecting artists with genre '{genre_filter}':\n")

    # Build component size lookup
    component_sizes = {}
    for component in nx.connected_components(G):
        size = len(component)
        for node in component:
            component_sizes[node] = size

    match_count = 0
    for node, data in G.nodes(data=True):
        main_genre = data.get('genre', '').lower()
        if genre_filter.lower() in main_genre:
            name = data.get('name', '<unknown>')
            raw_genres = data.get('genres', [])
            neighbors = list(G.neighbors(node))
            collab_names = [G.nodes[n].get('name', '<unknown>') for n in neighbors]
            comp_size = component_sizes.get(node, 1)

            print(f"Artist: {name}")
            print(f"Main Genre: {main_genre}")
            print(f"All Genres: {', '.join(raw_genres) if raw_genres else 'N/A'}")
            print(f"Collaborators ({len(collab_names)}): {', '.join(collab_names) if collab_names else 'None'}")
            print(f"Component Size: {comp_size}\n")

            match_count += 1

    if match_count == 0:
        print("No artists matched.")
    else:
        print(f"Found {match_count} artist(s) with genre '{genre_filter}'.")

def print_nodes_in_entropy_range(G, node_entropy_data, min_entropy, max_entropy, max_examples=5):
    print(f"\n🔍 Nodes with entropy between {min_entropy:.3f} and {max_entropy:.3f}:\n")
    
    count = 0
    for node, data in node_entropy_data.items():
        entropy = data["entropy"]
        if min_entropy <= entropy <= max_entropy:
            name = G.nodes[node].get("name", "Unknown")
            genres = G.nodes[node].get("genres", [])

            print(f"🎧 Node ID: {node}")
            print(f"  🎤 Name: {name}")
            print(f"  🎶 Genres (node attribute): {genres}")
            print(f"  📊 Entropy: {entropy:.4f}")
            print(f"  📍 h_j (local counts): {dict(data['h'])}")
            print(f"  📉 bar_h_j (normalized by global): {data['bar_h']}")
            print(f"  🧪 P_j (probability): {data['P']}")
            print("-" * 50)
            
            count += 1
            if count >= max_examples:
                break

    if count == 0:
        print("  (No nodes found in this range.)")

def get_top_entropy_nodes_per_year(G_by_year, min_degree=4, top_n=10):
    yearly_top_nodes = {}

    for year, G in G_by_year.items():
        entropy_data = compute_genre_entropy(G, min_degree=min_degree)
        sorted_nodes = sorted(entropy_data.items(), key=lambda x: x[1].get("entropy", 0), reverse=True)
        top_nodes = sorted_nodes[:top_n]
        
        yearly_top_nodes[year] = [{
            "id": node,
            "name": G.nodes[node].get("name", ""),
            "entropy": data["entropy"],
            "degree": G.degree[node],
            "genres": G.nodes[node].get("genres", []),
            "P": data["P"],
        } for node, data in top_nodes]

    return yearly_top_nodes

if __name__ == "__main__":
    data_folder = '../data_processed'  
    data = load_data_from_files(data_folder)
    

    num_songs, num_artists = count_unique_songs_and_artists(data)
    print(f"Unique songs: {num_songs}, Unique artists: {num_artists}")

    total_songs = count_total_songs(data)
    
    #get_release_year_statistics(data)

    genre_counts = get_genre_statistics(data)
    #frequency_plot(genre_counts)
    YEAR = 2016

    artist_graph = build_artist_collab_graph2(data, (1996, 2016))
    # # # print_random_samples(artist_graph, genre_filter="hip-hop")
    entropy_data = compute_genre_entropy(artist_graph)
    entropies = [d["entropy"] for d in entropy_data.values()]
    G_null = apply_null_model(artist_graph)
    node_entropy_null = compute_genre_entropy(G_null, min_degree=0)
    null_entropies = [d["entropy"] for d in node_entropy_null.values()]
    ### EARLY TOP 10 ENTROPY ARTISTS ###
    # yearly_top_entropy = {}

    # for year in range(1996, 2017):  # note: use range(start, end+1)
    #     print(f"Processing year: {year}")
    #     graph = build_artist_collab_graph2(data, (year, year))  
    #     top_nodes = get_top_entropy_nodes_per_year({year: graph}, min_degree=4, top_n=10)
    #     yearly_top_entropy.update(top_nodes)

    # for year in sorted(yearly_top_entropy):
    #     print(f"{year}:")
    #     for idx, artist in enumerate(yearly_top_entropy[year], start=1):
    #         name = artist["name"]
    #         entropy = artist["entropy"]
    #         degree = artist["degree"]
    #         print(f"   {idx}. {name} — Entropy: {entropy:.4f}, Degree: {degree}")
    #     print()  

    # # centrality = compute_weighted_betweenness(artist_graph)

    ### BC csv export plots
    # results = []
    
    ### YEARLY TOP 10 BC ARTISTS ###

    # yearly_top_bc = {}

    # for year in range(2007, 2017):
    #     print(f"\n===== Processing {year} =====")
    #     G = build_artist_collab_graph2(data, (year, year))
    #     result = compute_top_betweenness_summary(G, year, top_n=10)

        
        

    # row = compute_weighted_betweenness(artist_graph, YEAR)
    # results.append(row)
   
    #export_to_csv(results,"betweenness_centrality_summary.csv")
    #export_top_bc_summary(results, "betweenness_centrality_no_genre_summary.csv")
    #inspect_artists_by_genre(artist_graph, "folk-world")
    
    plt.figure(figsize=(12, 6))
   

    sns.set_style("whitegrid")

    
    plt.hist(null_entropies, bins=200, edgecolor='white', color='#69b3a2', alpha=0.85)

    
    plt.xlabel("Entropy $H_j$", fontsize=13)
    plt.ylabel("Artist Count", fontsize=13)
    plt.title("Distribution of Genre Entropy Among Artists", fontsize=15, weight='bold')

    
    sns.despine()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    
    plt.tight_layout()
    plt.show()
    
    