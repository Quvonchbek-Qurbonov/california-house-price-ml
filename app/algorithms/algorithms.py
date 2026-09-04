import numpy as np

from app.services.preprocessor import X_processed_df

from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AgglomerativeClustering,
    DBSCAN,
    OPTICS,
    MeanShift,
    Birch,
    SpectralClustering,
    AffinityPropagation,
)

from sklearn.mixture import GaussianMixture
from sklearn.neighbors import kneighbors_graph


X = X_processed_df.values

N_CLUSTERS = 10


# ---------------------------------------------------------
# K-Means
# ---------------------------------------------------------

kmeans = KMeans(
    n_clusters=N_CLUSTERS,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=42,
)


# ---------------------------------------------------------
# Mini-Batch K-Means
# ---------------------------------------------------------

mini_batch_kmeans = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    init="k-means++",
    n_init=10,
    max_iter=300,
    batch_size=2048,
    random_state=42,
)


# ---------------------------------------------------------
# Agglomerative
# ---------------------------------------------------------

connectivity = kneighbors_graph(
    X,
    n_neighbors=10,
    mode="connectivity",
    include_self=False,
)

agg = AgglomerativeClustering(
    n_clusters=N_CLUSTERS,
    metric="euclidean",
    linkage="ward",
    connectivity=connectivity,
)


# ---------------------------------------------------------
# DBSCAN
# ---------------------------------------------------------

dbscan = DBSCAN(
    eps=3,
    min_samples=10,
    metric="euclidean",
    n_jobs=-1,
)


# ---------------------------------------------------------
# OPTICS
# ---------------------------------------------------------

optics = OPTICS(
    min_samples=10,
    max_eps=5.0,
    cluster_method="xi",
    xi=0.05,
    min_cluster_size=50,
    n_jobs=-1,
)


# ---------------------------------------------------------
# Gaussian Mixture Model
# ---------------------------------------------------------

gmm = GaussianMixture(
    n_components=N_CLUSTERS,
    covariance_type="full",
    n_init=3,
    max_iter=300,
    init_params="kmeans",
    reg_covar=1e-6,
    random_state=42,
)


# ---------------------------------------------------------
# Spectral Clustering
# ---------------------------------------------------------

spectral = SpectralClustering(
    n_clusters=N_CLUSTERS,
    affinity="nearest_neighbors",
    n_neighbors=10,
    assign_labels="kmeans",
    n_init=5,
    random_state=42,
)


# ---------------------------------------------------------
# Mean Shift
# ---------------------------------------------------------

meanshift = MeanShift(
    bandwidth=2.0,
    bin_seeding=True,
    min_bin_freq=50,
    cluster_all=True,
    n_jobs=-1,
)


# ---------------------------------------------------------
# BIRCH
# ---------------------------------------------------------

birch = Birch(
    threshold=1.5,
    branching_factor=50,
    n_clusters=N_CLUSTERS,
)


# ---------------------------------------------------------
# Affinity Propagation
# ---------------------------------------------------------

affinity_propagation = AffinityPropagation(
    damping=0.9,
    max_iter=300,
    convergence_iter=15,
    random_state=42,
)
