import numpy as np
from matplotlib import pyplot as plt


class Sgd:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        # self.output = np.array

    def calculate_update(self, weight_tensor, gradient_tensor):
        result = weight_tensor - self.learning_rate * gradient_tensor
        return result
