import numpy as np

class ReLU():
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        self.input = input_tensor
        return np.maximum(input_tensor, 0, input_tensor)

    def backward(self,  error_tensor):
        self.output = error_tensor * (self.input > 0)
        return self.output

