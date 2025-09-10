import generator as gen
import matplotlib.pyplot as plt
import src.kmeans as km
import src.sse as sse
import src.k_iterate as ki


def run():
    """
    array_points contains the points of the three clouds, N=3 = number of clouds, n = number of points per cloud, mu = mean, covariance = covariance
    """
    #array_points = gen.N_clouds(3, 500, [[5.0, 5.0], [5.0, 0.0], [0.0, 5.0]],
                                #[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    array_points2 = gen.N_clouds(3, 500, [[2.0, 2.0], [2.0, 0.0], [0.0, 2.0]], [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])
    
    ki.k_means_iterative(10, array_points2)
    
    """
    the SSE desreases with increasing k
    
    both datasets have very similar courves
    """
    
    label, centroids = km.k_means(4, array_points2)

    # build clusters: list of arrays, one per cluster
    clusters = [array_points2[label == i] for i in range(len(centroids))]

    sse_value = sse.sse(centroids, clusters)

    print("Centroids:", centroids)
    print("Clusters:", clusters)
    print("SSE:", sse_value)

    plt.scatter(array_points2[:, 0], array_points2[:, 1], c=label, cmap='tab10', s=50)

    """
    no k-means does not always provide the same result for this data set
    """

    # plt.scatter(array_points2[:, 0], array_points2[:, 1], c=label, cmap='tab10', s=50)
    """
    the clouds overlap more here, so the k-means algorithm has more problems to cluster them correctly
    """

    """
    Labelling the cluster visualisation
    """
    plt.xlabel("x-Axis")
    plt.ylabel("y-Axis")
    plt.title("Cluster")
    plt.show()

    pass


if __name__ == "__main__":
    run()