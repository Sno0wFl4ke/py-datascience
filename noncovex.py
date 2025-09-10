import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.kmeans as km

df = pd.read_csv("data\clustering_non_convex (1).csv", header=None, names=["x", "y", "cluster"])

worth = (df[["x", "y"]].to_numpy())



#if you were to draw a line from one of the clusters points to another in the same 
#cluster you would cross the other cluster

#there a re 2 clusters

#no the results are not matching

def eucledean_distances(x: np.array):
    matrix = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x[i])):
            matrix[i, j] = np.linalg.norm(x[i] - x[j])


def adjacency_matrix_knn(D: np.ndarray,i:int, k: int):
    n = D.shape[0]
    A = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        distances = D[i]
        neighbors = np.argsort(distances)
        neighbors = neighbors[neighbors != i]  
        neighbors = neighbors[:k]
        
        A[i, neighbors] = 1  
    
    return A

def adjacency_matrix_radius(D: np.ndarray,i: int, r: float):
    
    n = D.shape[0]
    A = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        neighbors = np.where((D[i] <= r) & (D[i] > 0))[0] 
        A[i, neighbors] = 1
    
    return A

def laplacian(W: np.ndarray) -> np.ndarray:
   
    D = np.diag(W.sum(axis=1)) 
    print (D) 
    L = D - W
    return L

rad = adjacency_matrix_radius(worth,0,10)
knn = adjacency_matrix_knn(worth,0,8)

print(laplacian(rad))

centerpoints, y = km.k_means(2, worth)
plt.scatter(worth[:, 0],worth[:, 1], c=centerpoints, cmap='tab10', s=50)
plt.show()


