import numpy as np

"""
This function generates N random points with the given mean and covariance.
Similar to the Gaussian distribution.
Math formula in TeX: p(x) = \frac{1}{\sigma\sqrt{2\pi}}exp(-\frac{(x-\mu)^2}{2\sigma^2})
"""

def generate_data(my=np.array([0.0, 0.0]), covariance=np.array([[1.0, 0.0], [0.0, 1.0]]), n=None) -> np.ndarray:
    # μ = mean, Σ = covariance
    mean = my  # n length 1D array_like /
    cov = covariance  # (N, N) array_like 2D /
    size = n  # int or tuple of ints, optional
    check_valid = 'warn'  # 'warn', 'raise', or 'ignore', optional
    tol = 1e-8  # float, optional
    # print(np.random.multivariate_normal(mean, cov, size=size, check_valid=check_valid, tol=tol))
    return np.random.multivariate_normal(mean, cov, size=size, check_valid=check_valid, tol=tol)

"""
This function generates N random clouds of points with the given mean and covariance.
N: number of clouds
n: number of points per cloud
mu μ: list of mean vectors
covariance: list of covariance matrices
"""

def N_clouds(N: int, n: int, mu=[], covariance=[]):
    all_points = []
    for i in range(N):
        all_points.append(generate_data(mu[i], covariance[i], n))
    return np.vstack(all_points)
