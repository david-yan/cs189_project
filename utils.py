import numpy as np
from itertools import combinations_with_replacement

def generate_Y(n, degree, dim, X, terms, c_s, noise_Y):
    Y = np.zeros((n, 1))
    for i in range(n):
        x = X[i]
        y = 0
        for j in range(len(terms)):
            c = c_s[j]
            term = 1
            for k in terms[j]:
                if k == dim:
                    continue
                term *= x[k]
            y += c * term
        Y[i, 0] = y
    Y = Y + noise_Y * np.random.randn(n, 1)
    return Y

def generate_data(n, degree, dim, x_range, c_range, noise_X=0, noise_Y=0):
    X = np.random.rand(n, dim) * x_range - (x_range / 2)
    terms = list(combinations_with_replacement(range(dim+1), degree))
    c_s = np.random.rand(len(terms)) * c_range - (c_range / 2)
    Y = generate_Y(n, degree, dim, X, terms, c_s, noise_Y)
    X = X + noise_X * np.random.randn(n, dim)
    return X, Y, c_s

def generate_data_from_c(n, degree, dim, x_range, c_s, noise_X=0, noise_Y=0):
    X = np.random.rand(n, dim) * x_range - (x_range / 2)
    terms = list(combinations_with_replacement(range(dim+1), degree))
    Y = generate_Y(n, degree, dim, X, terms, c_s, noise_Y)
    X = X + noise_X * np.random.randn(n, dim)
    return X, Y

def polynomial_coefficients(degree):
    coeffs = {
        0: [1],
        1: [1, 1],
        2: [1, -4, 4],
        3: [1, -1, -9, 9],
        4: [1, 1, -11, -9, 18],
        5: [1, -1, -13, 13, 36, -36],
        6: [2, 1, -29, -13, 111, 36, -108]
    }
    return coeffs[degree]
