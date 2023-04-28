# Libraries...
import numpy as np

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

m = x_train.shape[0]
m = len(x_train)

def compute_cost(x, y, w, b):
    m = x.shape[0]

    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (y[i] - f_wb) ** 2
        cost_sum += cost
    total_cost = (1 / (2 * m)) * cost_sum

    return total_cost

