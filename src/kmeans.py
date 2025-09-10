import numpy as np
import matplotlib.pyplot as plt


def generate_data(my=np.array([0.0, 0.0]), covariance=np.array([[1.0, 0.0], [0.0, 1.0]]), n=None) -> np.ndarray:
    # μ = mean, Σ = covariance
    mean = my  # n length 1D array_like /
    cov = covariance  # (N, N) array_like 2D /
    size = n  # int or tuple of ints, optional
    check_valid = 'warn'  # 'warn', 'raise', or 'ignore', optional
    tol = 1e-8  # float, optional
    #print(np.random.multivariate_normal(mean, cov, size=size, check_valid=check_valid, tol=tol))
    return np.random.multivariate_normal(mean, cov, size=size, check_valid=check_valid, tol=tol)


def is_diagonal_psd (matrix: np.ndarray) -> bool:
    if not np.allclose(matrix, np.diag(np.diagonal(matrix))):
        return False
    return np.all(np.diagonal(matrix) >= 0)


def draw():
    plt.scatter(generate_data())
    plt.show()
    
def N_clouds(N: int, n:int, my=[], covariance=[]):
    all_points = []
    for i in range(N):
        all_points.append(generate_data(my[i],covariance[i], n))
    return all_points

array_points = N_clouds(3,500,[[5.0, 5.0],[5.0, 0.0],[0.0, 5.0]],[[[1.0, 0.0], [0.0, 1.0]],[[1.0, 0.0], [0.0, 1.0]],[[1.0, 0.0], [0.0, 1.0]]])

def k_means(k: int, points: np.ndarray):
    rand = []
    for i in range(k):
        x = np.random.randint(0,len(points))
        rand.append(x)
    centroids = points[rand]
    centerpoints = []
    for point in points:
        i = 0
        for centroid in centroids:
            dist = np.sqrt((point[0]-centroid[0])**2 + (point[1]-centroid[1])**2)
            
            if dist < smalldist or smalldist is None:
                smalldist = dist
                centerpoints[i] = centroid
        i += 1
            
    
    
if __name__ == "__main__":
    
    for i in array_points:
        plt.scatter(i[:,0],i[:,1])
    plt.show()