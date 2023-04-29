# Libraries...
import numpy as np
import copy

# Example variebles...
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])

#Prediction function for multivariable linear regression
def predict(x, w, b):
    prediction = np.dot(x, w) + b
    return prediction

# Cost function for multivariable linear regression
def compute_cost(X, y, w, b):
    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        cost_i = (y[i] - f_wb) ** 2
        cost += cost_i
    cost = cost / (2 * m)
    return cost

# Gradient function for multivariable linear regression
def compute_gradient(X, y, w, b):
    m,n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):
        cost = (np.dot(X[i], w) + b) - y[i]
        for j in range(n):
            dj_dw[j] += cost * X[i, j]
        dj_db += cost
    dj_dw /= m
    dj_db /= m
    return dj_db, dj_dw


def gradient_descent(X, y, w_in, b_in, alpha, num_iters, gradient_function):
    w = copy.deepcopy(w_in)
    b = b_in
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
    return w, b