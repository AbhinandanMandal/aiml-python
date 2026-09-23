
import numpy as np

np.random.seed(42)
x = np.linspace(0, 10, 200)
y = 4 * x + 7 + np.random.normal(0, 3, 200)


def momentum_gradient_descent(x,y,learning_rate=0.01,momentum=0.9,epochs=1000):
    n = len(x)
    m = 0.0
    b = 0.0

    # For momentum, we have to initialize velocity also
    v_m=0.0
    v_b=0.0
    history = {"m": [],"b": [],"loss": []}
    for epoch in range(epochs):
        indices=np.random.permutation(n)
        for i in indices:
            x_i=x[i]
            y_i=y[i]
            y_pred=m*x_i+b
            error=y_i-y_pred
            dm=-2*error*x_i
            db=-2*error

            # Momentum update
            """ 
            Momentum updating formula is
            v_t = gamma*v_t-1 + learning_rate*gradient
            weights = weights-v_t
            """
            v_m=momentum*v_m+learning_rate*dm 
            v_b=momentum*v_b+learning_rate*db 
            m-=v_m 
            b-=v_b 

        # For full dataset
        y_pred=m*x+b 
        mse = np.mean((y - y_pred) ** 2)
        history["m"].append(m)
        history["b"].append(b)
        history["loss"].append(mse)
    return m,b,history
    


