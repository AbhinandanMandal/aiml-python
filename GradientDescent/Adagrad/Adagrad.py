
"""
Adagrad is also built on the top of SDG
"""

import numpy as np

np.random.seed(42)
x = np.linspace(0, 10, 200)
y = 4 * x + 7 + np.random.normal(0, 3, 200)


def adaptive_gradient_descent(x,y,learning_rate=0.01,epochs=1000,epsilon=1e-8):
    n = len(x)
    m = 0.0
    b = 0.0

    # Accumulated squared gradients
    G_m = 0.0
    G_b = 0.0
    history = {"m": [],"b": [],"loss": []}

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(n)
        for i in indices:
            x_i = x[i]
            y_i = y[i]
            y_pred = m * x_i + b
            error = y_i - y_pred
            dm = -2 * error * x_i
            db = -2 * error

            # Updating accumulated squared gradient
            G_m += dm ** 2
            G_b += db ** 2

            # Adagrad parameter update
            m -= (learning_rate* dm/ np.sqrt(G_m + epsilon))
            b -= (learning_rate* db/ np.sqrt(G_b + epsilon))

        y_pred = m * x + b
        mse = np.mean((y - y_pred) ** 2)
        history["m"].append(m)
        history["b"].append(b)
        history["loss"].append(mse)

    return m, b, history
