import numpy as np

# SSE = Sum of Squared Errors
# SSE = \Sum_{i=1}^{k} \Sum_{x \in C_i} ||x_j - c_i||^2

"""
sse: calculates the sum of squared errors for each cluster
centroids: list of centroids
clusters: list of arrays, one per cluster

for each cluster:
for each point in the cluster:
        calculate the distance between the point and the centroid
        add the squared distance to the total sum of squared errors
        if the cluster is empty, skip

distance = ||x_j - c_i||^2 => normalize by 2 => squared distance
add the squared distance to the total sum of squared errors

return: float -> total sum of squared errors
"""
def sse(centroids, clusters):
    total_sse = 0.0
    centroids = np.array(centroids)
    for i, cluster_points in enumerate(clusters):
        cluster_points = np.atleast_2d(cluster_points)
        if cluster_points.shape[0] == 0:
            continue
        distances = np.linalg.norm(cluster_points - centroids[i], axis=1) ** 2
        total_sse += np.sum(distances)
    return total_sse