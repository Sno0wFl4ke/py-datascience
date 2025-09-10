import numpy as np

import generator as gen
import matplotlib.pyplot as plt
import src.kmeans as km
import src.sse as sse
import src.k_iterate as ki
from src.k_iterate import k_means_iterative


def run():
    """
    array_points contains the points of the three clouds, N=3 = number of clouds, n = number of points per cloud, mu = mean, covariance = covariance
    """
    #array_points = gen.N_clouds(3, 500, [[5.0, 5.0], [5.0, 0.0], [0.0, 5.0]],
                                #[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    array_points2 = gen.N_clouds(3, 500, [[2.0, 2.0], [2.0, 0.0], [0.0, 2.0]], [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    
    ki.k_means_iterative(10, array_points2)

    
    label, centroids = km.k_means(4, array_points2)

    # build clusters: list of arrays, one per cluster
    clusters = [array_points2[label == i] for i in range(len(centroids))]

    sse_value = sse.sse(centroids, clusters)

    print("Centroids:", centroids)
    print("Clusters:", clusters)
    print("SSE:", sse_value)

    plt.scatter(array_points2[:, 0], array_points2[:, 1], c=label, cmap='tab10', s=50)

    # plt.scatter(array_points2[:, 0], array_points2[:, 1], c=label, cmap='tab10', s=50)

    """
    Labelling the cluster visualisation
    """
    plt.xlabel("x-Axis")
    plt.ylabel("y-Axis")
    ki.k_means_iterative(10, array_points2)
    plt.title("Cluster")
    plt.show()

    pass

"""
Run N times and return the plot with the lowest SSE
N = number of times to run the k-means algorithm
k = number of clusters

Within the for loop we run the k-means algorithm and calculate the SSE for each run.
We then save it in a dictionary and add it to a list.
At the end we sort the list and return the lowest SSE result.
From the lowest SSE result we can get the points & build the cluster visualisation.
"""
def run_n_times(N: int = 10, k: int = 10):
    sse_list = []
    all_results = []
    # array => kmeans[]
    # map => sse_time + array[i]
    # ==> Check -> lowest SSE => print
    array_points = gen.N_clouds(3, 500, [[5.0, 5.0], [5.0, 0.0], [0.0, 5.0]],
                          [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])

    for i in range(N):
        result = {
            "sse": 0.0,
            "label": 0.0
        }

        label, centroids = km.k_means(k, array_points)

        # build clusters: list of arrays, one per cluster
        clusters = [array_points[label == i] for i in range(len(centroids))]

        sse_value = sse.sse(centroids, clusters)

        result["sse"] = float(sse_value)
        result["label"] = label
        all_results.append(result)

        sse_list.append(sse_value)

    all_results = sorted(all_results, key=lambda x: x["sse"])
    print("Lowest SSE result:", all_results[0])

    plt.scatter(array_points[:, 0], array_points[:, 1], c=all_results[0]["label"], cmap='tab10', s=50)

    plt.xlabel("x-Axis")
    plt.ylabel("y-Axis")
    plt.title("Smallest SSE Cluster (" + str(all_results[0]["sse"]) + ")")
    plt.show()


if __name__ == "__main__":
    run()