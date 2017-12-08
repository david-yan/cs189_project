import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
import utils

class PolynomialRegressor:
    def __init__(self, dim, degree, lmbda=0):
        self.dim = dim
        self.degree = degree
        self.lmbda = lmbda
        self.w = None

    def train(self, X, y):
        assert(self.dim == X.shape[1])
        terms = list(combinations_with_replacement(range(self.dim+1), self.degree))
        A = np.zeros((X.shape[0], len(terms)))
        for i in range(X.shape[0]):
            x = X[i]
            for j in range(len(terms)):
                term = 1
                for k in terms[j]:
                    if k == self.dim:
                        continue
                    term *= x[k]
                A[i, j] = term
        self.w = np.linalg.solve(A.T.dot(A)+self.lmbda*np.identity(A.shape[1]), A.T.dot(y))

    def predict(self, X):
        assert(self.dim == X.shape[1])
        terms = list(combinations_with_replacement(range(self.dim+1), self.degree))
        A = np.zeros((X.shape[0], len(terms)))
        for i in range(X.shape[0]):
            x = X[i]
            for j in range(len(terms)):
                term = 1
                for k in terms[j]:
                    if k == self.dim:
                        continue
                    term *= x[k]
                A[i, j] = term
        return A.dot(self.w)

    def mse(self, X, y):
        assert(self.dim == X.shape[1])
        terms = list(combinations_with_replacement(range(self.dim+1), self.degree))
        A = np.zeros((X.shape[0], len(terms)))
        for i in range(X.shape[0]):
            x = X[i]
            for j in range(len(terms)):
                term = 1
                for k in terms[j]:
                    if k == self.dim:
                        continue
                    term *= x[k]
                A[i, j] = term
        y_hat = A.dot(self.w)
        return np.linalg.norm(y - y_hat) ** 2 / X.shape[0]

def evaluate_regressor_over_varying_degree(regressor, n_train, n_test, x_range, noise_Y=0, noise_X=0):
    train_errors = []
    test_errors = []
    plt.figure(1, (20, 20))
    x = np.linspace(-1*x_range/2, x_range/2)
    for degree in range(1, 7):
        c_true = utils.polynomial_coefficients(degree)
        x_train, y_train = utils.generate_data_from_c(n_train, degree, dim, x_range, c_true, noise_Y=noise_Y, noise_X=noise_X)
        x_test, y_test = utils.generate_data_from_c(n_test, degree, dim, x_range, c_true, noise_Y=noise_Y, noise_X=noise_X)
        regressor.train(x_train, y_train)
        train_errors.append(regressor.mse(x_train, y_train))
        test_errors.append(regressor.mse(x_test, y_test))

        terms = list(combinations_with_replacement(range(2), degree))
        y_true = utils.generate_Y(len(x), degree, 1, np.array([x]).T, terms, c_true, 0)
        y_hat = regressor.predict(np.array([x]).T)

        plt.subplot(3, 2, degree)
        plt.title("Degree: "+str(degree))
        plt.plot(x, y_true, label='true model')
        plt.plot(x, y_hat.T[0], label='predicted model')
        plt.scatter(x_train, y_train, label='training data')
        plt.legend()
    plt.show()

    degrees = np.arange(1, 7)
    plt.title("MSE over degree")
    plt.plot(degrees, train_errors, label='training MSE')
    plt.plot(degrees, test_errors, label='test MSE')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    degree = 6
    dim = 1

    n_train = 100
    n_test = 20
    x_range = 7

    num_input = 1
    num_output = 1

    # x_train, y_train, c_true = utils.generate_data(n_train, degree, dim, x_range, c_range, noise_Y=2)
    # x_test, y_test = utils.generate_data_from_c(n_test, degree, dim, x_range, c_true, noise_Y=2)
    #
    # plt.scatter(x_train.T[0], y_train.T[0])
    # plt.show()

    regressor = PolynomialRegressor(dim=dim, degree=degree)
    evaluate_regressor_over_varying_degree(regressor, n_train, n_test, x_range, noise_Y=2)
