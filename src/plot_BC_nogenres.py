import pandas as pd
import matplotlib.pyplot as plt

def plot_overall_centrality_from_csv(csv_path):
    """
    Plots average betweenness centrality (full graph and GC) over time from a CSV summary.
    Assumes columns: 'Year', 'Avg BC (Full)', 'Avg BC (GC)'
    """
    df = pd.read_csv(csv_path)
    df['Year'] = df['Year'].astype(int)
    df.sort_values('Year', inplace=True)

    # Convert centrality columns to float if they are stored as strings
    df['Avg BC (Full)'] = df['Avg BC (Full)'].astype(float)
    df['Avg BC (GC)'] = df['Avg BC (GC)'].astype(float)

    plt.figure(figsize=(12, 6))
    plt.plot(df['Year'], df['Avg BC (Full)'], label='Avg BC (Full Graph)', linewidth=2)
    plt.plot(df['Year'], df['Avg BC (GC)'], label='Avg BC (Giant Component)', linewidth=2)

    plt.title("Average Betweenness Centrality Over Time")
    plt.xlabel("Year")
    plt.ylabel("Average Betweenness Centrality")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

plot_overall_centrality_from_csv("betweenness_centrality_no_genre_summary.csv")