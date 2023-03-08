import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.trainable = False
        self.pred_tensor = np.ndarray

    def forward(self, prediction_tensor, label_tensor):
        self.pred_tensor = prediction_tensor
        return -np.sum(label_tensor * np.log(prediction_tensor + np.finfo(float).eps))

    def backward(self, label_tensor):
        loss = -label_tensor / (self.pred_tensor + np.finfo(float).eps)
        return loss

