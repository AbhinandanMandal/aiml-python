
import numpy as np
np.random.seed(42)

x = np.linspace(0, 10, 200)
y = (4 * x + 7 + np.random.normal(0, 3, 200))


def batch_gradient_descent(x, y, learning_rate=0.01, epochs=1000):
    n = len(x)
    m = 0.0
    b = 0.0
    history = {"m": [], "b": [], "loss": []}

    for _ in range(epochs):
        y_pred = m * x + b
        mse = np.mean((y - y_pred) ** 2)
        dm = (-2 / n * np.sum((y - y_pred) * x))
        db = (-2 / n * np.sum(y - y_pred))

        history["m"].append(m)
        history["b"].append(b)
        history["loss"].append(mse)

        m -= learning_rate * dm
        b -= learning_rate * db

    return m, b, history
