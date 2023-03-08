import numpy as np
from joblib.numpy_pickle_utils import xrange


class SoftMax:
    def __init__(self):
        self.trainable = False

    def forward(self, input_tensor):
        expA = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        self.probs  = expA / np.sum(expA, axis=1, keepdims=True)
        self.input = input_tensor
        return self.probs

    def backward(self,  error_tensor):
        self.error_tensor = error_tensor
        self.output = self.probs * (self.error_tensor - (self.error_tensor * self.probs).sum(axis=1)[:, None])
        return self.output



