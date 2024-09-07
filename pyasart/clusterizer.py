import numpy as np

__all__ = ['kmeans']


def init_centroids(k, samples, seed=0):
    rng = np.random.default_rng(seed)
    return rng.choice(samples, k)


def closest_centroid(x, centroids, k):
    # print(centroids)
    distances = np.linalg.norm(centroids[np.newaxis, :] - x[:, np.newaxis], axis=-1)
    return np.argmin(distances, axis=-1)


def compute_means(cluster_idx, k, x):
    centroids = np.empty((k,) + x.shape[1:])
    for i in range(k):
        points = x[cluster_idx == i]
        centroids[i] = np.mean(points, axis=0)
    return centroids


def kmeans_centroids(x, k, epochs):
    centroids = init_centroids(k, x)
    for _ in range(epochs):
        clusters = closest_centroid(x, centroids, k)
        previous_centroids = centroids
        centroids = compute_means(clusters, k, x)
        if np.any(previous_centroids - centroids):
            break
    return centroids
