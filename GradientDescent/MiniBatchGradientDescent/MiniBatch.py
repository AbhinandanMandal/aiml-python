
import numpy as np

np.random.seed(42)
x = np.linspace(0, 10, 200)
y = 4 * x + 7 + np.random.normal(0, 3, 200)


def mini_batch_gradient_descent(x, y, learning_rate=0.01, epochs=1000, batch_size=32):
    n = len(x)
    m = 0.0
    b = 0.0
    history = {"m": [], "b": [], "loss": []}

    for epoch in range(epochs):

        # Shuffle the dataset at the beginning of every epoch
        indices = np.random.permutation(n)
        x_shuffled = x[indices]  # Shuffling data
        y_shuffled = y[indices]

        # Process data in mini-batches
        for start in range(0, n, batch_size):
            end = start + batch_size
            # Selecting current mini batch
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            y_pred = m * x_batch + b
            error = y_batch - y_pred
            dm = (-2 / len(x_batch) * np.sum(error * x_batch))
            db = (-2 / len(x_batch) * np.sum(error))
            m -= learning_rate * dm
            b -= learning_rate * db

        y_pred = m * x + b
        mse = np.mean((y - y_pred) ** 2)
        history["m"].append(m)
        history["b"].append(b)
        history["loss"].append(mse)

    return m, b, history
