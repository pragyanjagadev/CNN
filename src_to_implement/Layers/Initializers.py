import numpy as np

class Constant():

    def __init__(self, init_weight):
        self.init_weight = init_weight

    def initialize(self, weights_shape, fan_in=None, fan_out=None):
        return self.init_weight * np.ones(weights_shape)

class UniformRandom():

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        return np.random.uniform(0, 1, size= weights_shape)

class Xavier():

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        scale = np.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(scale=scale, size=weights_shape)

class He():

    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        scale = sigma = np.sqrt(2 / (fan_in))
        init_tensor = np.random.normal(scale=scale, size=weights_shape)
        return init_tensor

