import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import re
from collections import Counter
from collections import defaultdict
from genre_classifier import classify_genre, print_classification_summary, plot_genre_treemap, classification_stats

def load_data_from_files(data_folder):
    data = []
    for file_name in os.listdir(data_folder):
        if file_name.endswith('.json'):
            file_path = os.path.join(data_folder, file_name)
            print(f"Loading file: {file_name}")  
            with open(file_path, 'r', encoding='utf-8') as f:
                data.extend(json.load(f))  
    return data



def build_artist_collab_graph(data, year_range):

    G = nx.Graph()
    start_year, end_year = year_range  
    
    total_songs = 0
    collab_songs = 0
    artist_genres = {}  # Temporary storage for artist genres

    for entry in data:
        release_year = entry.get("release_year")
        
        try:
            release_year = int(release_year)
        except (TypeError, ValueError):
            continue  

        if not (start_year <= release_year <= end_year):
            continue  

        total_songs += 1
        
        song_name = entry["track_name"]
        artists = entry.get("artists", [])
        raw_genres = entry.get("genres", [])  

        classified_genres = [classify_genre(g) for g in raw_genres]
        
        
        if classified_genres:
            genre_counts = Counter(classified_genres)
            main_genre = genre_counts.most_common(1)[0][0]  # Most frequent genre
        else:
            main_genre = "other"
        
        artist_nodes = {artist["artist_id"]: artist["artist_name"] for artist in artists}

        for artist_id, artist_name in artist_nodes.items():
            if artist_id not in G:
                G.add_node(artist_id, name=artist_name, genre=main_genre) 
            artist_genres.setdefault(artist_id, []).append(main_genre)  

        artist_ids = list(artist_nodes.keys())
        if len(artist_ids) > 1:
            collab_songs += 1  
            for i in range(len(artist_ids)):
                for j in range(i + 1, len(artist_ids)):
                    artist1, artist2 = artist_ids[i], artist_ids[j]

                    if G.has_edge(artist1, artist2):
                        G[artist1][artist2]["songs"].add(song_name)  
                        G[artist1][artist2]["weight"] += 1  
                        G[artist1][artist2]["genre"].append(main_genre)  
                    else:
                        G.add_edge(artist1, artist2, songs={song_name}, weight=1, genre=[main_genre])

    collab_percentage = (collab_songs / total_songs * 100) if total_songs > 0 else 0
    print("Collab percentage:", collab_percentage)
    print("Nodes before removing isolates:", len(G.nodes))

    
    for u, v, data in G.edges(data=True):
        genre_counts = Counter(data["genre"])
        G[u][v]["genre"] = genre_counts.most_common(1)[0][0]  

    
    for node in G.nodes():
        edge_genres = [G[node][neighbor]["genre"] for neighbor in G.neighbors(node)]
        if edge_genres:
            genre_counts = Counter(edge_genres)
            G.nodes[node]["genre"] = genre_counts.most_common(1)[0][0]  
        else:
           
            if node in artist_genres:
                genre_counts = Counter(artist_genres[node])
                G.nodes[node]["genre"] = genre_counts.most_common(1)[0][0]
            else:
                G.nodes[node]["genre"] = "other"  

    print_classification_summary()
    return G




def build_artist_collab_graph_with_release_date(data, year_range):
    """
    Builds a collaboration graph of artists within a specified year interval and includes the release date.
    
    Parameters:
    - data (list of dicts): The dataset containing track information.
    - year_range (tuple): A tuple (start_year, end_year) defining the year filter.
    
    Returns:
    - G (networkx.Graph): A NetworkX graph with artist collaborations.
    """
    G = nx.Graph()
    start_year, end_year = year_range  
    total_songs = 0
    collab_songs = 0

    for entry in data:
        release_date = entry.get("release_date")  # Get exact release date

        if not release_date:
            continue  # Skip if no release_date is provided

        # Extract the year from the release_date (assuming 'YYYY' or 'YYYY-MM-DD' format)
        try:
            release_year = int(release_date.split("-")[0])  # Take the year part
        except (TypeError, ValueError):
            continue  # Skip if the release_date format is invalid

        if not (start_year <= release_year <= end_year):
            continue  # Skip songs outside the desired year range

        total_songs += 1
        song_id = entry["track_id"]
        song_name = entry["track_name"]
        artists = entry.get("artists", [])
        raw_genres = entry.get("genres", [])  

        # Classify genres
        classified_genres = [classify_genre(g) for g in raw_genres]
        main_genre = max(set(classified_genres), key=classified_genres.count) if classified_genres else "other"
        
        # Create artist nodes
        artist_nodes = {artist["artist_id"]: artist["artist_name"] for artist in artists}

        for artist_id, artist_name in artist_nodes.items():
            if artist_id not in G:
                G.add_node(artist_id, name=artist_name, genre=main_genre)

        artist_ids = list(artist_nodes.keys())
        if len(artist_ids) > 1:
            collab_songs += 1  
            for i in range(len(artist_ids)):
                for j in range(i + 1, len(artist_ids)):
                    artist1, artist2 = artist_ids[i], artist_ids[j]
                    print(f"Release Date for song {song_name}: {release_date}")
                    
                    # If an edge exists between the two artists, update it
                    if G.has_edge(artist1, artist2):
                        G[artist1][artist2]["songs"].add(song_name)  
                        G[artist1][artist2]["release_dates"].add(release_date)  # Add release date
                        G[artist1][artist2]["weight"] += 1  
                        G[artist1][artist2]["genre"] = main_genre  
                    else:
                        # If no edge exists, create a new one
                        G.add_edge(
                            artist1, artist2,
                            songs={song_name},
                            release_dates={release_date},  # Add release date
                            weight=1,
                            genre=main_genre
                        )

    # Calculate collaboration percentage
    collab_percentage = (collab_songs / total_songs * 100) if total_songs > 0 else 0
    print("Collab percentage:", collab_percentage)
    print("Nodes before removing isolates", len(G.nodes))
    
    # Remove isolated nodes (nodes with no edges)
    G.remove_nodes_from(list(nx.isolates(G)))
    
    return G


def visualize_full_graph(G):
   
    plt.figure(figsize=(12, 10))

    pos = nx.spring_layout(G, seed=42)  

    nx.draw(
        G, pos, 
        node_size=20,  
        node_color="blue",
        edge_color="gray", 
        alpha=0.3,  
        width=0.5  
    )

    plt.title("Full Artist Collaboration Graph")
    plt.show()

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

def export_graph_to_json(G, output_folder="../jsonexport/", filename="artist_collab.json"):
   
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  
    
    graph_data = {
        "nodes": [{"id": n, "name": G.nodes[n]["name"], "genre": G.nodes[n]["genre"]} for n in G.nodes()],
        "links": [
            {
                "source": u, 
                "target": v, 
                "songs": list(d["songs"]),
                "release_dates": list(d["release_dates"]), 
                "weight": d["weight"],
                "genre": d["genre"]
            }
            for u, v, d in G.edges(data=True)
        ],
    }

    output_path = os.path.join(output_folder, filename)
    with open(output_path, "w") as f:
        json.dump(graph_data, f, indent=4)

    print(f"Graph exported successfully to {output_path}")

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


def print_supergenre_edge_density_summary(G):
    
    genre_edges = defaultdict(list)

    
    for u, v, data in G.edges(data=True):
        genre = data.get("genre", "other")  
        genre_edges[genre].append((u, v))

    print("\nSupergenre Edge Density Summary:")
    for genre, edges in genre_edges.items():
        subgraph = G.edge_subgraph(edges)  
        density = nx.density(subgraph)  
        
        total_weight = sum(G[u][v].get("weight", 1) for u, v in edges)
        avg_weight = total_weight / len(edges) if edges else 0

        print(f"  - {genre}: Density = {density:.4f} | Avg Edge Weight = {avg_weight:.2f} | Nodes: {len(subgraph.nodes)}, Edges: {len(edges)}")


if __name__ == "__main__":
    data_folder = '../data_processed'  
    data = load_data_from_files(data_folder)
    

    num_songs, num_artists = count_unique_songs_and_artists(data)
    print(f"Unique songs: {num_songs}, Unique artists: {num_artists}")

    total_songs = count_total_songs(data)
    
    #song_graph = graph_songs_to_artists(data)
    #get_release_year_statistics(data)

    genre_counts = get_genre_statistics(data)
    #frequency_plot(genre_counts)

    artist_graph = build_artist_collab_graph(data, (1998, 2000))
    print(f" whole graph density: {nx.density(artist_graph)}")
    print_supergenre_edge_density_summary(artist_graph)
    #plot_genre_treemap(classification_stats)
    # print(f"G nodes: {len(artist_graph.nodes)}")
    # print(f"G edges: {len(artist_graph.edges)}")

    #jsonExport = build_artist_collab_graph_with_release_date(data, (1998, 1998))


    #export_graph_to_json(jsonExport)
    
    # print("\nSample collaborations:")
    # for artist1, artist2, edge_data in list(artist_graph.edges(data=True))[:30]:
    #     print(f"{artist_graph.nodes[artist1]['name']} ↔ {artist_graph.nodes[artist2]['name']} (Number of Songs: {len(edge_data['songs'])})")
    #export_graphML(artist_graph, "1998-2000")

    #visualize_full_graph(artist_graph)