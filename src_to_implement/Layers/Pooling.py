import numpy as np
from Layers.Base import BaseLayer

class Pooling(BaseLayer):

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.last_input = None
        self.stride = stride_shape
        self.pooling_shape = pooling_shape
        #self.trainable = True

    def forward(self, input_tensor):
        self.last_input = input_tensor
        # print(input_tensor.shape)
        batch_size, num_channels, h_prev, w_prev = input_tensor.shape
        h = int((h_prev - self.pooling_shape[1]) / self.stride[0]) + 1
        w = int((w_prev - self.pooling_shape[0]) / self.stride[1]) + 1

        downsampled = np.zeros((batch_size, num_channels, h, w))

        for i in range(batch_size):
            for j in range(num_channels):
                curr_y = out_y = 0
                while curr_y + self.pooling_shape[1] <= h_prev:
                    curr_x = out_x = 0
                    while curr_x + self.pooling_shape[0] <= w_prev:
                        patch = input_tensor[i, j, curr_y:curr_y + self.pooling_shape[1],
                                curr_x:curr_x + self.pooling_shape[0]]
                        downsampled[i, j, out_y, out_x] = np.max(patch)
                        curr_x += self.stride[1]
                        out_x += 1
                    curr_y += self.stride[0]
                    out_y += 1

        return downsampled

    def backward(self, error_tensor):
        batch_size, num_channels, h_prev, w_prev = self.last_input.shape

        output = np.zeros(self.last_input.shape)
        for b in range(batch_size):
            for c in range(num_channels):
                tmp_y = out_y = 0
                while tmp_y + self.pooling_shape[1] <= h_prev:
                    tmp_x = out_x = 0
                    while tmp_x + self.pooling_shape[0] <= w_prev:
                        patch = self.last_input[b, c, tmp_y:tmp_y + self.pooling_shape[1],
                                tmp_x:tmp_x + self.pooling_shape[0]]
                        (x, y) = np.unravel_index(np.argmax(patch), patch.shape)
                        output[b, c, tmp_y + x, tmp_x + y] += error_tensor[b, c, out_y, out_x]
                        tmp_x += self.stride[1]
                        out_x += 1
                    tmp_y += self.stride[0]
                    out_y += 1

        return output