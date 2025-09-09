import numpy as np
import matplotlib.pyplot as plt


def generate_data(my=np.array([0.0, 0.0]), covariance=np.array([[1.0, 0.0], [0.0, 1.0]]), n=None) -> np.ndarray:
    # μ = mean, Σ = covariance
    mean = my  # n length 1D array_like /
    cov = covariance  # (N, N) array_like 2D /
    size = n  # int or tuple of ints, optional
    check_valid = 'warn'  # 'warn', 'raise', or 'ignore', optional
    tol = 1e-8  # float, optional

    return np.random.multivariate_normal(mean, cov, size=size, check_valid=check_valid, tol=tol)


def is_diagonal_psd (matrix: np.ndarray) -> bool:
    if not np.allclose(matrix, np.diag(np.diagonal(matrix))):
        return False
    return np.all(np.diagonal(matrix) >= 0)


def draw():
    plt.scatter(generate_data())
    plt.show()

if __name__ == "__main__":
    draw()