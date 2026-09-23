
import numpy as np

np.random.seed(42)
x = np.linspace(0, 10, 200)
y = 4 * x + 7 + np.random.normal(0, 3, 200)


def nesterov_gradient_descent(x, y, learning_rate=0.001, momentum=0.9, epochs=1000):
    n = len(x)
    m = 0.0
    b = 0.0
    v_m = 0.0
    v_b = 0.0
    history = {"m": [], "b": [], "loss": []}

    for epoch in range(epochs):
        indices = np.random.permutation(n)  # Will be on SDG

        for i in indices:
            x_i = x[i]
            y_i = y[i]

            """ 
            With NAG, we first make a look-ahed step using previous velocity and then compute 
            gradient there. 
            Take NAG like a smart ball that actually know where it's going https://arxiv.org/pdf/1609.04747 
            """
            # Move temporarily in the direction suggested by the previous velocity
            m_lookahead = m - momentum * v_m
            b_lookahead = b - momentum * v_b

            y_pred = (m_lookahead * x_i + b_lookahead)
            error = y_i - y_pred
            dm = -2 * error * x_i
            db = -2 * error

            # Momentum update
            v_m = (momentum * v_m + learning_rate * dm)
            v_b = (momentum * v_b + learning_rate * db)

            m -= v_m
            b -= v_b

        y_pred = m * x + b
        mse = np.mean((y - y_pred) ** 2)

        history["m"].append(m)
        history["b"].append(b)
        history["loss"].append(mse)

    return m, b, history


# Train
m, b, history = nesterov_gradient_descent(
    x,
    y,
    learning_rate=0.001,
    momentum=0.9,
    epochs=1000
)

print(f"Learned slope     : {m:.4f}")
print(f"Learned intercept : {b:.4f}")
print(f"Final MSE         : {history['loss'][-1]:.4f}")
