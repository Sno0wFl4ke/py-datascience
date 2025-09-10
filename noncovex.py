import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import src.kmeans as km

df = pd.read_csv("data\clustering_non_convex (1).csv", header=None, names=["x", "y", "cluster"])

worth = (df[["x", "y"]].to_numpy())

"""
The euclidean distance between two points is defined as the length of the vector between the points.
x: 2D array_like, shape (N, 2)
"""
def eucledean_distances(x: np.array):
    matrix = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x[i])):
            matrix[i, j] = np.linalg.norm(x[i] - x[j])


"""
Adjacency matrix:
A[i, j] = 1 if point i and point j are neighbors, 0 otherwise.
k: number of neighbors
D: distance matrix"""
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

"""
Adjacency matrix:
A[i, j] = 1 if point i and point j are neighbors, 0 otherwise.
r: radius
D: distance matrix
"""
def adjacency_matrix_radius(D: np.ndarray, r: float):
    
    n = D.shape[0]
    A = np.zeros((n, n), dtype=int)
    
    for i in range(n):
        neighbors = np.where((D[i] <= r) & (D[i] > 0))[0]  # sich selbst ausschließen
        A[i, neighbors] = 1
    
    return A

"""
Laplacian graph: 
L = D - W
D: diagonal matrix with the sum of the weights of each node as the diagonal entries
W: adjacency matrix
L: laplacian matrix

"""
def laplacian(W: np.ndarray) -> np.ndarray:
   
    D = np.diag(W.sum(axis=1)) 
    print (D) 
    L = D - W
    return L

"""
Spectral clustering analysis:
Tries to analyse clusters by finding neighbors in the space.

data: 2D array_like, shape (N, 2)
radius: radius of the neighbors
k_neighbors: number of neighbors
n_clusters: number of clusters
n_eigvecs: number of eigenvectors to use
"""
def spectral_clustering_analysis(data, radius=10, k_neighbors=8, n_clusters=2, n_eigvecs=4):
    rad = adjacency_matrix_radius(data, radius)
    knn = adjacency_matrix_knn(data, k_neighbors)

    Lrad = laplacian(rad)
    Lknn = laplacian(knn)

    # Eigen decomposition
    eigvals_knn, eigvecs_knn = np.linalg.eig(Lrad)
    eigvals_rad, eigvecs_rad = np.linalg.eig(Lknn)

    # Check for negative eigenvalues
    for i in eigvals_knn:
        if i < 0:
            print("Negative eigenvalue in knn:", i)
    for i in eigvals_rad:
        if i < 0:
            print("Negative eigenvalue in radius:", i)

    idx_knn = np.argsort(eigvals_knn)
    Y_knn = eigvecs_knn[:, idx_knn[:n_eigvecs]]
    idx_rbf = np.argsort(eigvals_rad)
    Y_rbf = eigvecs_rad[:, idx_rbf[:n_eigvecs]]

    centerpoints_knn, centroids_knn = km.k_means(n_clusters, Y_knn)
    centerpoints_rbf, centroids_rbf = km.k_means(n_clusters, Y_rbf)

    plt.scatter(Y_knn[:, 0], Y_knn[:, 1], c=centerpoints_knn, cmap='tab10', s=50)
    plt.scatter(Y_rbf[:, 0], Y_rbf[:, 1], c=centerpoints_rbf, cmap='tab10', s=50)
    plt.show()

    centerpoints, y = km.k_means(n_clusters, data)
    plt.scatter(data[:, 0], data[:, 1], c=centerpoints, cmap='tab10', s=50)
    plt.show()



