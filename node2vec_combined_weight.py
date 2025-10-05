import pickle
from os.path import exists
import networkx as nx
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
from node2vec import Node2Vec
import plotly.express as px
from resources.platnets_constants import plants_original_combined_weight_graph_pickel_file, \
    planets_combined_node2vec_pkl, inputfile, planets_combined_node2vec_tsne_pkl, node2vec_combined_keys, \
    node2_vec_combined_labels, node2vec_combined_radius_list, node2vec_combined_embedding


# --- Helper Functions ---

def read_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_as_pickle(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)


def similarity(value_planet, value_earth, max_range):
    """Calculate similarity score scaled between 0 and 1."""
    if pd.isna(value_planet) or pd.isna(value_earth):
        return 0
    diff = abs(value_planet - value_earth)
    sim = max(0, 1 - diff / max_range)
    return sim


def create_original_graph_combined_weight(source_nd, target_nd, radii, periods, temps):
    G = nx.Graph()

    # Earth baseline values and normalization ranges
    earth_radius = 1.0  # Earth radii
    earth_period = 365.25  # days (Earth's orbital period)
    earth_temp = 288  # Kelvin (Earth average temp)

    max_radius_range = 10.0  # max radius difference for scaling
    max_period_range = 1000.0  # max orbital period diff
    max_temp_range = 1000.0  # max temp difference

    for src, trg, rad, per, temp in zip(source_nd, target_nd, radii, periods, temps):
        if pd.isna(trg):
            continue

        sim_radius = similarity(rad, earth_radius, max_radius_range)
        sim_period = similarity(per, earth_period, max_period_range)
        sim_temp = similarity(temp, earth_temp, max_temp_range)

        combined_weight = (sim_radius + sim_period + sim_temp) / 3.0

        if combined_weight > 0:  # Add edge only if positive similarity
            G.add_edge(src, trg, weight=combined_weight)

    return G


def embed_node2vec_planets(G, labels, radius_list):
    if exists(planets_combined_node2vec_pkl) == False:
        model = Node2Vec(G, dimensions=20, walk_length=10, num_walks=50, p=1, q=0.5, workers=1)
        model = model.fit()
        save_as_pickle(planets_combined_node2vec_pkl, model)
    else:
        model = read_pickle(planets_combined_node2vec_pkl)

    if exists(planets_combined_node2vec_tsne_pkl) == False:
        key_embedding = []
        keys = []

        for key in model.wv.index2entity:
            key_embedding.append(model.wv.get_vector(key))
            keys.append(key)
        tsne = TSNE(n_components=2, perplexity=25, random_state=42)
        embeddings_2d = tsne.fit_transform(np.array(key_embedding))
        save_as_pickle(planets_combined_node2vec_tsne_pkl, embeddings_2d)
        save_as_pickle(node2vec_combined_keys, keys)
    else:
        embeddings_2d = read_pickle(planets_combined_node2vec_tsne_pkl)
        keys = read_pickle(node2vec_combined_keys)

    # Create a DataFrame for Plotly, add node names for hover info
    df = pd.DataFrame({
        'X': embeddings_2d[:, 0],
        'Y': embeddings_2d[:, 1],
        'label': labels,  # categorical labels for color
        'node': keys,  # node names to show on hover
        'radius': radius_list
    })

    fig = px.scatter(
        df, x='X', y='Y',
        color='label',
        hover_name='node',  # Show node info on hover
        title="TSNE 2D Projection With Combined Weights Only",
        width=800, height=800,
        color_discrete_map={
            "CONFIRMED": "green",
            "FALSE POSITIVE": "red",
            "CANDIDATE": "yellow"
        }
    )

    fig.show()
    fig.write_html(node2vec_combined_embedding)


# --- Main Execution ---

if __name__ == '__main__':
    print("Loading data...")
    node_content = pd.read_csv(inputfile)

    if exists(plants_original_combined_weight_graph_pickel_file):
        print("Loading saved graph from pickle...")
        G = read_pickle(plants_original_combined_weight_graph_pickel_file)
    else:
        print("Creating graph with combined similarity weights...")
        target_nodes = node_content["kepoi_name"]
        source_nodes = ["earth"] * len(target_nodes)
        radii = node_content["koi_prad"]
        periods = node_content["koi_period"]
        temps = node_content["koi_teq"]

        G = create_original_graph_combined_weight(source_nodes, target_nodes, radii, periods, temps)
        save_as_pickle(plants_original_combined_weight_graph_pickel_file, G)
        print("Graph saved.")

    if exists(node2_vec_combined_labels) == False:
        # Create a lookup dictionary: {kepoi_name: koi_disposition}
        labels_lookup = dict(zip(node_content["kepoi_name"], node_content["koi_disposition"]))
        target_nodes_distance_lookup = dict(zip(node_content["kepoi_name"], node_content["koi_prad"]))
        radius_list = []
        # Assign labels based on node names in G
        labels = []
        for node in G.nodes():
            if node == "earth":
                labels.append("")
                radius_list.append("")
            else:
                # Use .get() to return "" if the node is not found in the dictionary
                labels.append(labels_lookup.get(node, ""))
                radius_list.append(target_nodes_distance_lookup.get(node, ""))

        save_as_pickle(node2_vec_combined_labels, labels)
        save_as_pickle(node2vec_combined_radius_list, radius_list)
    else:
        labels = read_pickle(node2_vec_combined_labels)
        radius_list = read_pickle(node2vec_combined_radius_list)
    embed_node2vec_planets(G, labels, radius_list)
