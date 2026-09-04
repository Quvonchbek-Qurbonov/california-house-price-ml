import numpy as np

from app.algorithms.algorithms import (
    kmeans,
    mini_batch_kmeans,
    agg,
    dbscan,
    optics,
    gmm,
    spectral,
    meanshift,
    birch,
    affinity_propagation,
)

from app.services.preprocessor import X_processed_df, df


models = {
    "KMeans": kmeans,
    "MiniBatchKMeans": mini_batch_kmeans,
    "Agglomerative": agg,
    "DBSCAN": dbscan,
    "OPTICS": optics,
    "GMM": gmm,
    "Spectral": spectral,
    "MeanShift": meanshift,
    "Birch": birch,
    "AffinityPropagation": affinity_propagation,
}


def apply_clustering(algo_name: str) -> dict:

    if algo_name not in models:
        raise ValueError(
            f"Unknown algorithm: {algo_name}"
        )

    cluster_labels = models[algo_name].fit_predict(X_processed_df)

    actual = df.loc[X_processed_df.index, "Class"].to_numpy()

    fraud_rates = {}

    for cluster in np.unique(cluster_labels):
        mask = cluster_labels == cluster
        fraud_rates[cluster] = actual[mask].mean()

    fraud_cluster = max(fraud_rates, key=fraud_rates.get)

    # Treat that cluster as "fraud"
    predicted = (cluster_labels == fraud_cluster).astype(int)

    # Confusion matrix values
    tp = np.sum((actual == 1) & (predicted == 1))
    fp = np.sum((actual == 0) & (predicted == 1))
    fn = np.sum((actual == 1) & (predicted == 0))

    # Metrics
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0
    )

    return {
        "algorithm": algo_name,
        "fraud_cluster": int(fraud_cluster),
        "fraud_detected": int(tp),
        "fraud_missed": int(fn),
        "false_alarms": int(fp),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }
