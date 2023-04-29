# Libraries...
import numpy as np

x_train = np.array([1.0, 1.7, 2.0, 2.5, 3.0, 3.2,])
y_train = np.array([250, 300, 480, 430, 630, 730,])

m = x_train.shape[0]
m = len(x_train)

# Cost function for univariable linear regression
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (y[i] - f_wb) ** 2
        cost_sum += cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost

# Gradient function for univariable linear regression
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw_i = (f_wb - y[i]) * x[i]
        dj_db_i = f_wb - y[i]
        dj_dw += dj_dw_i
        dj_db += dj_db_i
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

# Gradient descent function for univariable linear regression
