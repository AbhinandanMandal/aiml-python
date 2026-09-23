
import numpy as np

np.random.seed(42)
x = np.linspace(0, 10, 200)
y = 4 * x + 7 + np.random.normal(0, 3, 200)


def stochastic_gradient_descent(x, y, learning_rate=0.01, epochs=1000):
    n = len(x)
    m = 0.0
    b = 0.0
    history = {"m": [], "b": [], "loss": []}

    for epoch in range(epochs):
        # For stochastic gradient descent, shuffling matters for random datapoints
        indices = np.random.permutation(n)
        for i in indices:
            x_i = x[i]  # Selecting random sample
            y_i = y[i]
            y_pred = m * x_i + b
            error = y_i - y_pred
            dm = -2 * error * x_i
            db = -2 * error
            m -= learning_rate * dm
            b -= learning_rate * db

        # Calculate full-dataset loss after epoch
        y_pred = m * x + b
        mse = np.mean((y - y_pred) ** 2)
        history["m"].append(m)
        history["b"].append(b)
        history["loss"].append(mse)
    return m, b, history
