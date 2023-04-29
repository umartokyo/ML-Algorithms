# Libraries...
import numpy as np

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
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, gradient_function):
    b = b_in
    w = w_in
    for i in range(num_iters):
        dj_dw, dj_db = gradient_function(x, y, w, b)
        b = b - alpha * dj_db
        w = w - alpha * dj_dw
    return w, b

def example():
    x_example = np.array([1.0, 2.0]) 
    y_example = np.array([300.0, 500.0])
    w_init = 0
    b_init = 0
    iterations = 10000
    alpha_init = 1.0e-2
    w_final, b_final = gradient_descent(x_example, y_example, w_init, b_init, alpha_init, iterations, compute_gradient)
    print(f"\nThe final function: f_wb = {w_final:0.4f} * x + {b_final:0.4f}")
    print("\nPredictions:")
    print(f"1000 sqm house prediction {w_final * 1.0 + b_final:0.1f} ")
    print(f"1200 sqm house prediction {w_final * 1.2 + b_final:0.1f} ")
    print(f"2000 sqm house prediction {w_final * 2.0 + b_final:0.1f} \n")

example()