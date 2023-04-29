# Libraries...
import numpy as np

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

