import src.kmeans as km
import numpy as np
import matplotlib.pyplot as plt
#import src.sse as sse

def k_means_iterative(k_max:int ,points: np.ndarray):
    SSEs = []
    for i in range(1, k_max + 1):
        centerpoints, centroids = km.k_means(i, points)
        #sse = sse.sse(centroids, centerpoints)
        #SSEs.append(sse)
    x = range(1, k_max + 1)
    plt.plot(x, SSEs, marker='o', linestyle='-', color='blue')
    plt.xlabel('k')
    plt.ylabel('SSE')
    plt.title('SSE zu K')
    plt.grid(True)
    plt.show()
