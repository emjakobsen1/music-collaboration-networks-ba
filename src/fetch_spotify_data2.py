import requests
import json
import os
from datetime import datetime

# Spotify API credentials
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "ab9abd363a564edba95c95be11525e1e")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "a6b3f628a3774f8c9eb435fed1c1a86c")

# API endpoints
AUTH_URL = "https://accounts.spotify.com/api/token"
TRACKS_URL = "https://api.spotify.com/v1/tracks"

# Year to filter (Set the year you want to process)
TARGET_YEAR = "1998"  # Change this to the desired year

def get_spotify_token():
    """Fetches a Spotify API token."""
    auth_response = requests.post(
        AUTH_URL,
        data={"grant_type": "client_credentials"},
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        auth=(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET),
    )
    return auth_response.json().get("access_token")

def fetch_track_info(track_ids, token):
    """Fetches detailed track info from Spotify, including exact release date."""
    headers = {"Authorization": f"Bearer {token}"}
    updated_tracks = []

    for i in range(0, len(track_ids), 50):  # Process in batches of 50
        batch = track_ids[i:i + 50]
        response = requests.get(f"{TRACKS_URL}?ids={','.join(batch)}", headers=headers)

        if response.status_code != 200:
            print(f"⚠️ Error fetching tracks: {response.status_code}, {response.text}")
            continue

        try:
            response_data = response.json()
        except requests.exceptions.JSONDecodeError:
            print(f"⚠️ JSON decode error: {response.text}")
            continue

        for track in response_data.get("tracks", []):
            if not track:
                continue

            track_id = track["id"]
            release_date = track["album"]["release_date"]  # YYYY-MM-DD or YYYY format

            # Ensure all release dates are in YYYY-MM-DD format
            if len(release_date) == 4:  
                release_date += "-01-01"  # Default to Jan 1 if only the year is available

            updated_tracks.append({
                "track_id": track_id,
                "track_name": track["name"],
                "album_name": track["album"]["name"],
                "release_date": release_date,  # Updated field
                "duration_ms": track["duration_ms"],
                "popularity": track["popularity"],
                "explicit": track["explicit"],
                "artists": [{"artist_name": artist["name"], "artist_id": artist["id"]} for artist in track["artists"]],
                "genres": []  # We keep genres as-is for now
            })

    return updated_tracks

def process_file(json_file):
    """Processes a single JSON file, filtering by year and updating the release date."""
    token = get_spotify_token()
    if not token:
        print("❌ Failed to retrieve Spotify token.")
        return

    output_file = json_file.replace("../data2/", "data3/").replace(".json", ".processed.json")
    if os.path.exists(output_file):
        print(f"✅ Skipping {json_file}, already processed.")
        return

    with open(json_file, "r", encoding="utf-8") as f:
        track_data = json.load(f)

    # Filter tracks that match the target year
    track_ids = [t["track_id"] for t in track_data if t["release_year"] == TARGET_YEAR]

    if not track_ids:
        print(f"⚠️ No tracks found for year {TARGET_YEAR} in {json_file}")
        return

    print(f"🚀 Processing {json_file} with {len(track_ids)} tracks for {TARGET_YEAR}...")

    updated_tracks = fetch_track_info(track_ids, token)

    os.makedirs("data3", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(updated_tracks, f, indent=4)

    print(f"✅ Processed {json_file} → {output_file}")

def main():
    """Processes all relevant JSON files in ../data2/."""
    json_files = [f for f in os.listdir("../data2") if f.endswith(".json")]

    if not json_files:
        print("❌ No files found in data2/")
        return

    for file in json_files:
        process_file(os.path.join("../data2", file))

if __name__ == "__main__":
    main()
