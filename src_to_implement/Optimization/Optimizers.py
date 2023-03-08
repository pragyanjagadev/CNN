import numpy as np

class Sgd:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.result = np.array

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.result = weight_tensor - self.learning_rate * gradient_tensor
        return self.result

class SgdWithMomentum:
    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v_prev = 0.

    def calculate_update(self, weight_tensor, gradient_tensor):
        v = (self.momentum_rate * self.v_prev) - (self.learning_rate * gradient_tensor)
        self.v_prev = v
        weight_tensor = weight_tensor + v
        return weight_tensor


class Adam:
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v_prev = 0.
        self.r_prev = 0.
        self.k = 1

    def calculate_update(self, weight_tensor, gradient_tensor):

        v = (self.mu * self.v_prev) + (1-self.mu) * gradient_tensor
        r = (self.rho * self.r_prev) + (1 - self.rho) * (gradient_tensor ** 2)
        v_hat = v/(1 - self.mu ** self.k)
        r_hat = r / (1 - self.rho ** self.k)
        weight_tensor = weight_tensor - (self.learning_rate * (v_hat / (np.sqrt(r_hat)) + np.finfo(float).eps))

        self.v_prev = v
        self.k = self.k + 1
        self.r_prev = r
        return weight_tensor