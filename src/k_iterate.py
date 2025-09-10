import src.kmeans as km
import numpy as np
import matplotlib.pyplot as plt
import src.sse as sse

"""
It runs k-means for different values of k (from 1 to k_max) and plots the SSE (sum of squared errors) for each k.
"""

def k_means_iterative(k_max:int ,points: np.ndarray):
    SSEs = []
    for i in range(1, k_max + 1):
        centerpoints, centroids = km.k_means(i, points)
        clusters = [points[centerpoints == i] for i in range(len(centroids))]
        
        sse_value = sse.sse(centroids, clusters)
        SSEs.append(sse_value)
    x = range(1, k_max + 1)
    plt.plot(x, SSEs, marker='o', linestyle='-', color='blue')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('SSE to k')
    plt.grid(True)
    plt.show()
