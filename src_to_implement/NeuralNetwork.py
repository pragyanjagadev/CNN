from copy import deepcopy

# from Layers import *
# from src_to_implement.Optimization import Loss
from joblib.numpy_pickle_utils import xrange


class NeuralNetwork:
    def __init__(self, optimizer, weights_initializer, bias_initializer):
        self.optimizer = optimizer
        self.loss = list()
        self.layers = list()
        self.data_layer = None
        self.loss_layer = None
        self.input_tensor = None
        self.label_tensor = None
        self.out = None
        self.weights_initializer = weights_initializer
        self.bias_initializer = bias_initializer

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()
        x = self.input_tensor.copy()
        for layer in self.layers:
            x = layer.forward(x)
        self.out = self.loss_layer.forward(x, self.label_tensor)
        return self.out

    def backward(self):
        loss_grad = self.loss_layer.backward(self.label_tensor)
        for layer in self.layers[::-1]:
            loss_grad = layer.backward(loss_grad)

    def append_layer(self, layer):
        if layer.trainable:
            layer.initialize(self.weights_initializer, self.bias_initializer)
            layer.optimizer = deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)
            self.backward()

    def test(self, input_tensor):
        x = input_tensor
        for layer in self.layers:
            x = layer.forward(x)
        return x
