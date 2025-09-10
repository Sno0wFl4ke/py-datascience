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


def adjacency_matrix_knn(D: np.ndarray, k: int):
    
    n = D.shape[0]
    A = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        distances = D[i]
        neighbors = np.argsort(distances)
        neighbors = neighbors[neighbors != i]  # sich selbst ausschließen
        neighbors = neighbors[:k]
        A[i, neighbors] = 1
        
    return A

def adjacency_matrix_radius(D: np.ndarray, r: float):
    
    n = D.shape[0]
    A = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        neighbors = np.where((D[i] <= r) & (D[i] > 0))[0]  # sich selbst ausschließen
        A[i, neighbors] = 1
    
    return A

def laplacian(W: np.ndarray) -> np.ndarray:
   
    D = np.diag(W.sum(axis=1)) 
    print (D) 
    L = D - W
    return L

rad = adjacency_matrix_radius(worth,10)
knn = adjacency_matrix_knn(worth,8)

Lrad = laplacian(rad)
Lknn = laplacian(knn)

eigvals_knn, eigvecs_knn = np.linalg.eig(Lrad)
eigvals_rad, eigvecs_rad = np.linalg.eig(Lknn)

for i in eigvals_knn:
    if i < 0:
        print("Negative eigenvalue in knn:", i)

for i in eigvals_rad:
    if i < 0:
        print("Negative eigenvalue in radius:", i)

M = 4


idx_knn = np.argsort(eigvals_knn)
Y_knn = eigvecs_knn[:, idx_knn[:M]]  # N x M


idx_rbf = np.argsort(eigvals_rad)
Y_rbf = eigvecs_rad[:, idx_rbf[:M]]


k = 2
centerpoints_knn, centroids_knn = km.k_means(k, Y_knn)
centerpoints_rbf, centroids_rbf = km.k_means(k, Y_rbf)
plt.scatter(Y_knn[:, 0], Y_knn[:, 1],c=centerpoints_knn, cmap='tab10', s=50)
plt.scatter(Y_rbf[:, 0], Y_rbf[:, 1],c=centerpoints_rbf, cmap='tab10', s=50)
plt.show()

centerpoints, y = km.k_means(2, worth)
plt.scatter(worth[:, 0],worth[:, 1], c=centerpoints, cmap='tab10', s=50)
plt.show()


