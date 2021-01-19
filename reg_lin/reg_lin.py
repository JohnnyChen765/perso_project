import numpy as np


def custom_reg_lin(X, y):  # X and y have to be numpy matrixes
    # np.matrix([np.ones(house_data.shape[0]), house_data['surface']).T
    # (T(X) * X)^-1 * T(X) * Y
    return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)


def custom_reg_ridge(X, y, lamda=1):  # X and y have to be numpy matrixes
    _, p = X.shape
    return np.linalg.inv(X.T.dot(X) + lamda * np.ones(p)).dot(X.T).dot(y)
