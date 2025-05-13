import pandas as pd
import matplotlib.pyplot as plt

def plot_genre_centrality_from_csv(csv_path):
    # Load the CSV data
    df = pd.read_csv(csv_path)
    
    # Convert 'Year' column to int if it's not already
    df['Year'] = df['Year'].astype(int)
    
    # Identify genre columns (everything after 'GC Edges')
    start_col = list(df.columns).index('GC Edges') + 1
    genre_columns = df.columns[start_col:]

    # Plot each genre as a line
    plt.figure(figsize=(14, 7))
    
    for genre in genre_columns:
        plt.plot(df['Year'], df[genre].astype(float), label=genre, linewidth=2)

    # Force x-axis ticks at every integer year
    plt.xticks(df['Year'], rotation=45)

    plt.title("Mean Betweenness Centrality over Years")
    plt.xlabel("Year")
    plt.ylabel("Mean Betweenness Centrality")
    plt.legend(title="Genre", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

plot_genre_centrality_from_csv("betweenness_centrality_summary.csv")
