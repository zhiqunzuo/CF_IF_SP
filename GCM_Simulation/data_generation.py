import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from typing import Tuple

np.random.seed(42)

def sample(N: int, d_u: int, d_x: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.random.choice([1, 2], p=[0.5, 0.5], size=N)
    
    mean_u = np.zeros(d_u)
    cov_u = np.eye(d_u)
    u = np.random.multivariate_normal(mean_u, cov_u, N)
    
    w_u = np.random.randn(d_x, d_u)
    w_a = np.random.randn(d_x, 1)
    
    b = np.random.randn(d_x)
    mean_x = (np.dot(w_u, u.T) + np.dot(w_a, a.reshape(1, -1)) + b[:, np.newaxis]).T
    cov_x = np.random.randn(d_x, d_x)
    cov_x = np.dot(cov_x, cov_x.T)
    
    x = np.zeros((N, d_x))
    for i in range(N):
        data = np.random.multivariate_normal(mean_x[i], cov_x, 1)
        x[i] = data
    
    w_x = np.random.randn(1, d_x)
    b_x = np.random.randn()
    y = (np.dot(w_x, x.T) + b_x).T
    
    return a, x, y, w_u, cov_x, w_a, b