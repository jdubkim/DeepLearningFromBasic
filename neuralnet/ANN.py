import numpy as np

class ANN:

    def __init__(self, input_size : int, hidden_size : int, output_size : int):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.random.randn(1, self.hidden_size)

        self.W2 = np.random.randn(self.hidden_size, self.hidden_size)
        self.b2 = np.random.randn(1, self.hidden_size)

        self.W3 = np.random.randn(self.hidden_size, self.hidden_size)
        self.b3 = np.random.randn(1, self.hidden_size)

        self.W4 = np.random.randn(self.hidden_size, self.output_size)
        self.b4 = np.random.randn(1, self.output_size)

    def feed_forward(self, X):
        self.a1 = np.dot(X, self.W1) + self.b1
        self.z1 = self.sigmoid(self.a1)
        self.a2 = np.dot(self.z1, self.W2) + self.b2
        self.z2 = self.sigmoid(self.a2)
        self.a3 = np.dot(self.z2, self.W3) + self.b3
        self.z3 = self.sigmoid(self.a3)
        self.a4 = np.dot(self.z3, self.W4) + self.b4

        return self.softmax(self.a4)

    
    def sigmoid(self, z : np.ndarray):
        return (1 / (1 + np.exp(z)))

    def RELU(self, z:np.ndarray):
        return np.maximum(z, 0)

    def softmax(self, z : np.ndarray):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def mean_squared_error(self, y : np.array, t : np.array):
        return 0.5 * np.sum((y - t) ** 2)

    def cross_entropy_error(self, y : np.array, t : np.array):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)

        batch_size = y.shape[0]
        return -np.sum(t * np.log(y)) / batch_size

    def numerical_gradient(self, f, x):
        h = 1e-4
        grad = np.zeros_like(x)

        for i in range(x.size):
            # f(x + h)
            init_x = x[i]
            x[i] = init_x + h
            fx_plus_h = f(x)

            x[i] = init_x - h
            fx_minus_h = f(x)

            grad[i] = (fx_plus_h - fx_minus_h) / (2 * h)
            x[i] = init_x

        return grad

    def gradient_descent(self, x, w, y_true):
        pass

