import numpy as np
from Optimization.Optimizers import Sgd
from Layers.Base import BaseLayer


class FullyConnected(BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        # print("Fullycon======")
        self.__gradient_weights = None
        self.input_size = input_size  # columns
        self.output_size = output_size
        self.trainable = True
        self.batch_size = 9  # rows
        self.weights = np.random.uniform(0., 1., (self.input_size + 1 ,self.output_size))
        self.biases = np.zeros(self.output_size)
        self.optimizer = None

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, val):
        self.__optimizer = val

    @property
    def gradient_weights(self):
        return self.__gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, val):
        self.__gradient_weights = val

    def forward(self, input_tensor):
        bias_list = [1] * input_tensor.shape[0]
        bias_array = np.array(bias_list).reshape(input_tensor.shape[0], 1)
        self.input_tensor = np.concatenate((input_tensor, bias_array), axis=1)
        output_tensor = np.dot(self.input_tensor, self.weights)
        return output_tensor

    def backward(self, error_tensor):
        self.gradient_weights = np.dot(self.input_tensor.T, error_tensor)
        self.output = np.dot(error_tensor, self.weights[:-1].T)

        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.weights, self.gradient_weights)
        return self.output

    def initialize(self, weights_initializer, bias_initializer):
        self.weights[:-1] = weights_initializer.initialize(self.weights[:-1].shape, self.input_size, self.output_size)
        self.weights[-1] = bias_initializer.initialize(self.weights[-1].shape, self.input_size, self.output_size)

