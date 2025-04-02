import requests
import json
import os
import glob
import multiprocessing

# Load Spotify API credentials
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "d6439164eb6746c8ac5bb20a2f708fe4")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "6c209eabbd824444aed24feec03b7766")

# Spotify API endpoints
AUTH_URL = "https://accounts.spotify.com/api/token"
TRACKS_URL = "https://api.spotify.com/v1/tracks"
ARTISTS_URL = "https://api.spotify.com/v1/artists"

# Function to get an access token
def get_spotify_token():
    auth_response = requests.post(
        AUTH_URL,
        data={"grant_type": "client_credentials"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
    )
    auth_response_data = auth_response.json()
    return auth_response_data.get("access_token")

# Function to fetch track details
def fetch_track_info(track_ids, token):
    headers = {"Authorization": f"Bearer {token}"}
    track_info_dict = {}

    for i in range(0, len(track_ids), 50):  
        batch = track_ids[i:i + 50]
        response = requests.get(f"{TRACKS_URL}?ids={','.join(batch)}", headers=headers)
        if response.status_code != 200:
            print(f"Error fetching tracks: {response.json()}")
            continue

        for track in response.json().get("tracks", []):
            if not track:
                continue

            track_id = track["id"]
            track_info_dict[track_id] = {
                "track_id": track_id,
                "track_name": track["name"],
                "album_name": track["album"]["name"],
                "release_year": track["album"]["release_date"][:4],
                "duration_ms": track["duration_ms"],
                "popularity": track["popularity"],
                "explicit": track["explicit"],
                "artists": [{"artist_name": artist["name"], "artist_id": artist["id"]} for artist in track["artists"]],
                "genres": []
            }

    return list(track_info_dict.values())

# Function to fetch artist genres
def fetch_artist_genres(track_info, token):
    headers = {"Authorization": f"Bearer {token}"}
    all_artist_ids = {aid["artist_id"] for track in track_info for aid in track["artists"]}
    artist_genre_map = {}

    for i in range(0, len(all_artist_ids), 50):  
        batch = list(all_artist_ids)[i:i + 50]
        response = requests.get(f"{ARTISTS_URL}?ids={','.join(batch)}", headers=headers)
        if response.status_code != 200:
            print(f"Error fetching artists: {response.json()}")
            continue

        for artist in response.json().get("artists", []):
            artist_genre_map[artist["id"]] = artist.get("genres", [])

    for track in track_info:
        genres = set()
        for artist in track["artists"]:
            genres.update(artist_genre_map.get(artist["artist_id"], []))
        track["genres"] = list(genres)

    return track_info

# Function to process a single playlist file
def process_file(json_file):
    token = get_spotify_token()
    if not token:
        print(f"Failed to retrieve Spotify token for {json_file}")
        return

    output_file = json_file.replace("data/", "data2/").replace(".json", ".processed.json")
    
    # Skip processing if already processed
    if os.path.exists(output_file):
        print(f"✅ Skipping {json_file}, already processed.")
        return

    # Load track IDs from JSON
    with open(json_file, "r", encoding="utf-8") as f:
        playlist_data = json.load(f)

    track_ids = set()  
    for playlist in playlist_data["playlists"]:  
        for track in playlist["tracks"]:
            track_ids.add(track["track_uri"].split(":")[-1])  # Extract the track ID from Spotify URI

    print(f"Processing {json_file} with {len(track_ids)} unique tracks...")

    track_info = fetch_track_info(list(track_ids), token)
    track_info = fetch_artist_genres(track_info, token)

    # Save to output JSON
    os.makedirs("data2", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(track_info, f, indent=4)

    print(f"✅ Processed {json_file} → {output_file}")

# Main function for parallel processing
def main():
    files = glob.glob("data/*.json")  # Find all JSON files in 'data/' folder
    num_workers = min(10, multiprocessing.cpu_count())  

    print(f"🚀 Processing {len(files)} files using {num_workers} workers...")

    with multiprocessing.Pool(num_workers) as pool:
        pool.map(process_file, files)

    print("✅ All files processed!")

if __name__ == "__main__":
    main()
