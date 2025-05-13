import requests
import json
import os
from dotenv import load_dotenv
load_dotenv()
# Load Spotify API credentials
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")

# Spotify API endpoints
AUTH_URL = "https://accounts.spotify.com/api/token"
TRACKS_URL = "https://api.spotify.com/v1/tracks"
ARTISTS_URL = "https://api.spotify.com/v1/artists"

def get_spotify_token():
    auth_response = requests.post(
        AUTH_URL,
        data={"grant_type": "client_credentials"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
    )
    auth_response_data = auth_response.json()
    return auth_response_data.get("access_token")

def fetch_track_info(track_ids, token):
    headers = {"Authorization": f"Bearer {token}"}
    track_info_dict = {}

    for i in range(0, len(track_ids), 50):  
        batch = track_ids[i:i + 50]
        response = requests.get(f"{TRACKS_URL}?ids={','.join(batch)}", headers=headers)

        
        if response.status_code != 200:
            print(f"Error fetching tracks: {response.status_code}, {response.text}")
            
            
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                print(f"Retry-After header: Retry after {retry_after} seconds.")
            
            continue

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"Error decoding JSON response for tracks: {response.text}")
            continue

        for track in response_data.get("tracks", []):
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


def process_file(json_file):
    token = get_spotify_token()
    if not token:
        print(f"Failed to retrieve Spotify token for {json_file}")
        return

    output_file = json_file.replace("data/", "data_processed/").replace(".json", ".processed.json")
    

    if os.path.exists(output_file):
        print(f"✅ Skipping {json_file}, already processed.")
        return

    
    with open(json_file, "r", encoding="utf-8") as f:
        playlist_data = json.load(f)

    track_ids = set()  
    for playlist in playlist_data["playlists"]:  
        for track in playlist["tracks"]:
            track_ids.add(track["track_uri"].split(":")[-1])  

    print(f"Processing {json_file} with {len(track_ids)} tracks...")

    track_info = fetch_track_info(list(track_ids), token)
    track_info = fetch_artist_genres(track_info, token)

    
    os.makedirs("data_processed", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(track_info, f, indent=4)

    print(f"✅ Processed {json_file} → {output_file}")


def main():
    
    file_to_process = "../data/mpd.slice.159000-159999.json"  
    
    if not os.path.exists(file_to_process):
        print(f"File {file_to_process} not found.")
        return

    print(f"🚀 Processing file: {file_to_process}")
    process_file(file_to_process)

    print("File processed.")

if __name__ == "__main__":
    main()
