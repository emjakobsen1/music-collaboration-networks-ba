import networkx as nx
import matplotlib.pyplot as plt
import collections
import powerlaw

def compute_degree_distribution(G):
    degrees = [d for n, d in G.degree()]
    degree_count = collections.Counter(degrees)
    deg, cnt = zip(*sorted(degree_count.items()))
    return degrees, deg, cnt

def analyze_degree_distributions(data, start_year=1986, end_year=2016):
    import powerlaw

    years = []
    alpha_full = []
    alpha_gcc = []

    for year in range(start_year, end_year + 1):
        print(f"Processing year {year}...")
        G = build_artist_collab_graph2(data, year_range=(year, year))
        degrees_full = [d for _, d in G.degree() if d > 0]

        # Fit on full graph
        fit_full = powerlaw.Fit(degrees_full, verbose=False)
        alpha_f = fit_full.power_law.alpha
        alpha_full.append(alpha_f)

        # Fit on GCC
        if not nx.is_connected(G):
            gcc_nodes = max(nx.connected_components(G), key=len)
            G_gcc = G.subgraph(gcc_nodes).copy()
        else:
            G_gcc = G

        degrees_gcc = [d for _, d in G_gcc.degree() if d > 0]
        fit_gcc = powerlaw.Fit(degrees_gcc, verbose=False)
        alpha_g = fit_gcc.power_law.alpha
        alpha_gcc.append(alpha_g)

        print(f"  α (Full): {alpha_f:.2f} | α (GCC): {alpha_g:.2f}")
        years.append(year)

    
    plt.figure(figsize=(9, 6))
    plt.plot(years, alpha_full, marker='o', label='Full Graph α', linestyle='--')
    plt.plot(years, alpha_gcc, marker='o', label='GCC α', linestyle='-')
    plt.xlabel("Year")
    plt.ylabel("Power-law exponent α")
    plt.title("Power-law exponent over time: Full Graph vs. GCC")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import collections
import networkx as nx
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_degree_distributions_by_decades(data, start_year=1986, end_year=2016):
    years = list(range(start_year, end_year + 1))
    
    
    decade_colors = {
        1980: cm.Blues,
        1990: cm.Greens,
        2000: cm.Oranges,
        2010: cm.Purples,
    }

    
    fig_full, ax_full = plt.subplots(figsize=(10, 7))

    for year in years:
        G = build_artist_collab_graph2(data, year_range=(year, year))
        degrees = [d for _, d in G.degree()]
        degree_count = collections.Counter(degrees)

        if not degree_count:
            continue  

        deg, cnt = zip(*sorted(degree_count.items()))
        decade = (year // 10) * 10
        cmap = decade_colors.get(decade, cm.Greys)
        color = cmap((year - decade) / 10)

        ax_full.plot(deg, cnt, color=color, label=str(year), linewidth=1)

    ax_full.set_xscale('log')
    ax_full.set_yscale('log')
    ax_full.set_xlabel("Degree")
    ax_full.set_ylabel("Frequency")
    ax_full.set_title("Degree Distribution")
    ax_full.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax_full.legend(title="Year", fontsize=8, loc='best', ncol=2)
    plt.tight_layout()
    plt.show()

    # GCC plot
    fig_gcc, ax_gcc = plt.subplots(figsize=(10, 7))

    for year in years:
        G = build_artist_collab_graph2(data, year_range=(year, year))
        if not nx.is_connected(G):
            gcc_nodes = max(nx.connected_components(G), key=len)
            G_gcc = G.subgraph(gcc_nodes).copy()
        else:
            G_gcc = G

        degrees_gcc = [d for _, d in G_gcc.degree()]
        degree_count_gcc = collections.Counter(degrees_gcc)

        if not degree_count_gcc:
            continue

        deg_gcc, cnt_gcc = zip(*sorted(degree_count_gcc.items()))
        decade = (year // 10) * 10
        cmap = decade_colors.get(decade, cm.Greys)
        color = cmap((year - decade) / 10)

        ax_gcc.plot(deg_gcc, cnt_gcc, color=color, label=str(year), linewidth=1)

    ax_gcc.set_xscale('log')
    ax_gcc.set_yscale('log')
    ax_gcc.set_xlabel("Degree")
    ax_gcc.set_ylabel("Frequency")
    ax_gcc.set_title("Degree Distribution in the Giant Component")
    ax_gcc.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax_gcc.legend(title="Year", fontsize=8, loc='best', ncol=2)
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    from visualize import build_artist_collab_graph2, load_data_from_files

    data_folder = '../data_processed'
    data = load_data_from_files(data_folder)

    #analyze_degree_distributions(data)
    plot_degree_distributions_by_decades(data, start_year=1986, end_year=2016)
