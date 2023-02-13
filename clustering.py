import umap, hdbscan
import numpy as np


'''
Generates HDBSCAN cluster object after UMAP dimension reduction of embeddings
=============================================================================
Parameters:
    - embeddings: np.array-like structure of sentence embeddings
    - n_neighbors: the parameter 'k' in the k-neighbour graph of the UMAP algorithm
    - n_components: reduced dimension of the UMAP embeddings
    - min_cluster_size: minimum no. of elements in HDBSCAN cluster
    - random_state: for reproducibility of results
'''


def generate_clusters(embeddings, n_neighbors, n_components, min_cluster_size, random_state=None):
    dim_reduce = (
        umap
        .UMAP(n_neighbors=n_neighbors,
              n_components=n_components,
              metric='cosine',
              random_state=random_state)
        .fit(embeddings)
    )

    umap_embeddings = dim_reduce.transform(embeddings)

    clusters = (
        hdbscan
        .HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric='euclidean',
            cluster_selection_method='eom',
            prediction_data=True
        )
        .fit(umap_embeddings)
    )

    return clusters, dim_reduce


"""
Returns the label count and cost of a given HDBSCAN cluster object
========================================================================
Parameters:
    - clusters: HDBSCAN cluster object
    - prob_threshold: Confidence level at which cost reflects percentage of data having label confidence below 
    this specified threshold
"""


def score_clusters(clusters, prob_threshold=0.05):
    label_count = len(np.unique(clusters.labels_))
    total_num = len(clusters.labels_)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold) / total_num)

    return label_count, cost
