import copy
import pickle
from os.path import exists
import networkx as nx
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
from resources.platnets_constants import plants_original_graph_pickel_file, inputfile, planets_poincare_pkl, \
    planets_node2vec_pkl, planets_node2vec_tsne_pkl, node2_vec_labels, node2vec_keys, node2vec_radius_list, \
    node2vec_embedding
from node2vec import Node2Vec
import plotly.express as px
# --- Helper Functions --- #

def read_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def save_as_pickle(filename, data):
    with open(filename, 'wb') as output:
        pickle.dump(data, output)


def embed_node2vec_planets(G, labels, radius_list):
    if exists(planets_node2vec_pkl) == False:
        model = Node2Vec(G, dimensions=20, walk_length=10, num_walks=50, p=1, q=0.5, workers=1)
        model = model.fit()
        save_as_pickle(planets_node2vec_pkl, model)
    else:
        model = read_pickle(planets_node2vec_pkl)


    if exists(planets_node2vec_tsne_pkl) == False:
        key_embedding = []
        keys = []

        for key in model.wv.index2entity:
            key_embedding.append(model.wv.get_vector(key))
            keys.append(key)
        tsne = TSNE(n_components=2, perplexity=25, random_state=42)
        embeddings_2d = tsne.fit_transform(np.array(key_embedding))
        save_as_pickle(planets_node2vec_tsne_pkl, embeddings_2d)
        save_as_pickle(node2vec_keys, keys)
    else:
        embeddings_2d = read_pickle(planets_node2vec_tsne_pkl)
        keys = read_pickle(node2vec_keys)

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
        title="TSNE 2D Projection With Kepler Object of Interest Planetary Radius(koi_prad) Only",
        width=800, height=800,
        color_discrete_map={
            "CONFIRMED": "green",
            "FALSE POSITIVE": "red",
            "CANDIDATE": "yellow"
        }
    )

    fig.show()
    fig.write_html(node2vec_embedding)


# --- Create Graph from CSV with Weights --- #

def create_original_graph(source_nd, target_nd, rating):
    G = nx.Graph()
    edges = []
    nodes = set()
    # Using set for faster lookup
    for i in range(len(source_nd)):
        if not pd.isna(target_nd[i]) and not pd.isna(rating[i]):
           # Correct NaN check
           edges.append((source_nd[i], target_nd[i], rating[i]))
           nodes.add(source_nd[i])
           nodes.add(target_nd[i])
           G.add_nodes_from(nodes)
           G.add_weighted_edges_from(edges)
    return G

# --- Simulate Weights by Scaling and Repeating Edges --- #

def create_weighted_relations(G, weight_multiplier=10):
    weighted_relations = []
    for u, v, data in G.edges(data=True):
        raw_weight = data.get("weight", 1.0)
        scaled_weight = int(max(1, round(raw_weight * weight_multiplier)))
        weighted_relations.extend([(u, v)] * scaled_weight)
    return weighted_relations

# --- Train Poincare Embedding --- #

'''def poincareEmbedding(G, weight_multiplier=10):
    weighted_relations = create_weighted_relations(G, weight_multiplier)
    model = PoincareModel(weighted_relations, size=2, negative=1)
    model.train(epochs=100)
    save_as_pickle(planets_poincare_pkl, model)
    return model'''

# --- Visualization --- #

'''def plot_poincare_2d_visualization(model, pairs):
    fig = poincare_2d_visualization(model, pairs, "Poincare Hierarchy", show_node_labels=pairs)
    fig_source = copy.deepcopy(fig)
    fig_target = copy.deepcopy(fig)

    for trace in fig.data:
        trace.marker.colorbar = None  # Hide colorbar
    fig.show()'''

# --- Main Entry Point --- #

if __name__ == '__main__':
    # Read node data from CSV
    node_content = pd.read_csv(inputfile)
    # Load or create the graph
    if exists(plants_original_graph_pickel_file):
        G = read_pickle(plants_original_graph_pickel_file)
    else:
        target_nodes_distance = node_content["koi_prad"]
        target_nodes = node_content["kepoi_name"]
        source_nodes = ["earth"] * len(target_nodes)
        # List of "earth"
        # Create graph
        G = create_original_graph(source_nodes, list(target_nodes), list(target_nodes_distance))
        # Save graph
        save_as_pickle(plants_original_graph_pickel_file, G)

    if exists(node2_vec_labels) == False:
        # Create a lookup dictionary: {kepoi_name: koi_disposition}
        labels_lookup = dict(zip(node_content["kepoi_name"], node_content["koi_disposition"]))
        target_nodes_distance_lookup = dict(zip(node_content["kepoi_name"], node_content["koi_prad"]))
        # Assign labels based on node names in G
        labels = []
        radius_list = []
        for node in G.nodes():
            if node == "earth":
                labels.append("")
                radius_list.append("")
            else:
                # Use .get() to return "" if the node is not found in the dictionary
                labels.append(labels_lookup.get(node, ""))
                radius_list.append(target_nodes_distance_lookup.get(node,""))
        save_as_pickle(node2vec_radius_list, radius_list)
        save_as_pickle(node2_vec_labels, labels)
    else:
        labels = read_pickle(node2_vec_labels)
        radius_list = read_pickle(node2vec_radius_list)
    embed_node2vec_planets(G, labels, radius_list)