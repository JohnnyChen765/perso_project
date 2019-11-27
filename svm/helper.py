import numpy as np
import pandas as pd

def random_circle(number_points, noise = True, rayon = 1, x_offset = 0, y_offset = 0):
    noise_x = noise_normal(number_points, rayon / 20)
    noise_y = noise_normal(number_points, rayon / 20)
    random_alphas = 2 * np.pi * np.random.rand(1, number_points)[0]
    
    x = rayon * np.cos(random_alphas) + noise_x + x_offset
    y = rayon * np.sin(random_alphas) + noise_y + y_offset
    dataframe = pd.DataFrame({'x': x, 'y': y}, index=[i for i in range(0, len(x))])
    return dataframe

def noise_normal(number_points, variance = 1):
    return np.random.normal(0, variance, number_points)

def random_points(number_points, min = 0, max = 1):
    return (max - min) * np.random.rand(1, number_points)[0] + min

def mse(y_hat, y):
    error = y_hat - y
    n = np.shape(y)[0]
    return error.T.dot(error) / n