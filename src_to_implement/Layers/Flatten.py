

from Layers.Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.b = None
        self.w = None
        self.h = None
        self.c = None

    def forward(self, input_tensor):
        self.b, self.w, self.h, self.c = input_tensor.shape
        flat = self.w * self.h * self.c
        return input_tensor.reshape(self.b, flat)

    def backward(self, error_tensor):
        return error_tensor.reshape(self.b, self.w, self.h, self.c)
