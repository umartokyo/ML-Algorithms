# Libraries...
import numpy as np
import copy

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

# Gradient descent function for multivariable linear regression
def gradient_descent(X, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    w = copy.deepcopy(w_in)
    b = b_in
    print("\nRunning gradient_descent...")
    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i % (num_iters // 10) == 0:
            print(f"Progress: {int(i / num_iters * 100)}%, Iteration: {i}, Cost: {cost_function(X, y, w, b):0.2f}")
    return w, b

def example():
    # Paramenters: X: [Size in sqm, num of bedrooms, num of floors, age] y: price of the house
    X_example = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
    y_example = np.array([460, 232, 178])
    w_init = np.zeros(X_example.shape[1])
    b_init = 0.
    iterations = 1000 # Cost = 53 with iterations = 10000000
    alpha_init = 5.0e-7
    w_final, b_final = gradient_descent(X_example, y_example, w_init, b_init, alpha_init, iterations, compute_cost, compute_gradient)
    print(f"\nThe final function: X * {w_final} + {b_final}")
    print("\nPredictions:")
    for i in range(X_example.shape[0]):
        print(f"prediction: {np.dot(X_example[i], w_final) + b_final:0.2f}, target value: {y_example[i]}, ")
    print(f"Cost: {compute_cost(X_example, y_example, w_final, b_final):0.2f}\n")

example()