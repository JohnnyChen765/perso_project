import helper
import matplotlib.pyplot as plt
import sklearn.metrics
import numpy as np

# cercle_points = helper.random_circle(100)

# cercle_points = helper.random_square(100)
# plt.plot(cercle_points["x"], cercle_points["y"], "ro", markersize=4)
# observation_matrix = np.matrix(cercle_points["x"]).T

# # Kernel matrix accepting 2D array
# K = sklearn.metrics.pairwise.polynomial_kernel(observation_matrix, degree=2)

# ridge_coeff = 10
# results = helper.kernel_reg(K, np.matrix(cercle_points["y"]).T, ridge_coeff)
# # results is our vector of alphas

# y_hat_train = K.dot(results)
# mse_train = helper.mse(y_hat_train, np.matrix(cercle_points["y"]).T)

# # Prediction is just KernelMatrix_to_predict * alphas
# x_predict = np.matrix(helper.random_points(100, -2, 2)).T
# K_predict = sklearn.metrics.pairwise.polynomial_kernel(
#     np.matrix(cercle_points["x"]).T, Y=x_predict, degree=2
# )
# y_predict = K_predict.T.dot(results)
# mse = helper.mse(y_predict, np.matrix(cercle_points["y"]).T)
# plt.plot(x_predict, y_predict, "yo", markersize=4)
# plt.legend(mse)

# plt.grid()
# plt.show()


def ridge_rkhs(x_train, y_train, x_test, y_test, kernelMatrix_callback, ridge_coeff=10):
    plt.plot(x_train, y_train, "ro", markersize=4)
    observation_matrix = np.matrix(x_train).T
    y_train_matrix = np.matrix(y_train).T
    x_test_matrix = np.matrix(x_test).T
    y_test_matrix = np.matrix(y_test).T

    # Kernel matrix accepting 2D array
    # K = sklearn.metrics.pairwise.polynomial_kernel(observation_matrix, degree=2)
    K = kernelMatrix_callback(observation_matrix)

    results = helper.kernel_reg(K, y_train_matrix, ridge_coeff)
    # results is our vector of alphas

    y_hat_train = K.dot(results)
    mse_train = helper.mse(y_hat_train, y_train_matrix)

    # Prediction is just KernelMatrix_to_predict * alphas
    K_predict = kernelMatrix_callback(observation_matrix, Y=x_test_matrix)
    y_predict = K_predict.T.dot(results)
    mse = helper.mse(y_predict, y_test_matrix)

    plt.plot(x_test, y_predict, "yo", markersize=4)
    plt.legend(mse)

    plt.grid()
    plt.show()


x_limit = 10
dataset_function = lambda n_point: helper.random_function(
    lambda x: np.sin(x) / x, number_points=n_point, min=-x_limit, max=x_limit
)
# dataset_function = lambda n_point: helper.random_function(
#     lambda x: np.sin(x) ** 2 + np.cos(x) ** 2,
#     number_points=n_point,
#     min=-x_limit,
#     max=x_limit,
# )

# dataset_function = lambda n_point: helper.random_circle(n_point)

dataset = dataset_function(1000)
x_test = np.matrix(helper.random_points(100, -x_limit, x_limit)).T
dataset_test = dataset_function(100)
# kernelMatrix = lambda x, Y=None: sklearn.metrics.pairwise.rbf_kernel(x, Y=Y, gamma=1)
kernelMatrix = lambda x, Y=None: sklearn.metrics.pairwise.linear_kernel(x, Y=Y)

ridge_rkhs(
    dataset["x"],
    dataset["y"],
    dataset_test["x"],
    dataset_test["y"],
    kernelMatrix,
    ridge_coeff=1,
)

# from sklearn.kernel_ridge import KernelRidge

# clf = KernelRidge(
#     alpha=1.0, coef0=1, degree=3, gamma=None, kernel="rbf", kernel_params=None
# )
# clf.fit(np.matrix(dataset["x"]).T, np.matrix(dataset["y"]).T)
# results = clf.predict(np.matrix(x_test).T)
# plt.plot(dataset["x"], dataset["y"])
# plt.plot(dataset["x"], results)
# plt.grid()
# plt.show()

