import json
import os
import shutil

def clean_dataset(input_folder, output_folder):
    """Goes through all JSON files, keeps only the first occurrence of each song, and saves a cleaned version."""

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)  # Create the cleaned data folder

    unique_songs = set()  # To track songs we've already seen
    total_songs = 0
    removed_duplicates = 0

    for file in os.listdir(input_folder):
        if file.endswith(".json"):
            input_path = os.path.join(input_folder, file)
            output_path = os.path.join(output_folder, file)

            with open(input_path, "r") as f:
                data = json.load(f)

            cleaned_data = []
            for entry in data:
                song_id = entry["track_id"]
                total_songs += 1

                if song_id not in unique_songs:  # Keep only the first occurrence
                    unique_songs.add(song_id)
                    cleaned_data.append(entry)
                else:
                    removed_duplicates += 1

            # Save cleaned data to new folder
            with open(output_path, "w") as f:
                json.dump(cleaned_data, f, indent=4)

    print(f"Total song entries processed: {total_songs}")
    print(f"Unique songs kept: {len(unique_songs)}")
    print(f"Duplicate entries removed: {removed_duplicates}")

# Paths
input_folder = "../data2"
output_folder = "../data2"

# Run cleaning process
clean_dataset(input_folder, output_folder)
