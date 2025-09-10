import numpy as np
import pandas as pd

"""
This functions calculates the Euclidean distance between two points.
a: 1D array_like, shape (N,)
b: 1D array_like, shape (N,)
"""


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
