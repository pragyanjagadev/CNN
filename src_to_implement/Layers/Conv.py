import numpy as np
import math
from scipy.signal import correlate2d, convolve2d
from Layers.Base import BaseLayer
import copy


class Conv(BaseLayer):

    def __init__(self, stride_shape, convolution_shape, num_kernels):

        super().__init__()

        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.convolution_shape_list = list(self.convolution_shape)
        self.convolution_shape_list.insert(0, self.num_kernels)
        self.weights = np.random.uniform(low=0.0, high=1.0, size=tuple(self.convolution_shape_list))
        self.bias = np.random.uniform(low=0.0, high=1.0, size=(num_kernels, 1))
        self.input_tensor = None
        self.gradient_weights = None
        self.gradient_bias = None
        self.optimizer = None

    @property
    def gradient_weights(self):
        return self.__gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, val):
        self.__gradient_weights = val

    @property
    def gradient_bias(self):
        return self.__gradient_bias

    @gradient_bias.setter
    def gradient_bias(self, val):
        self.__gradient_bias = val

    @property
    def optimizer(self):
        return self.__optimizer

    @optimizer.setter
    def optimizer(self, val):
        self.__optimizer = val

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = None
        fan_out = None
        #handling 1D
        if len(self.convolution_shape) == 2:
            fan_in = self.convolution_shape[0] * self.convolution_shape[1]
            fan_out = self.num_kernels * self.convolution_shape[1]
        elif len(self.convolution_shape) == 3:
            fan_in = self.convolution_shape[0] * self.convolution_shape[1] * self.convolution_shape[2]
            fan_out = self.num_kernels * self.convolution_shape[1] * self.convolution_shape[2]

        self.weights = weights_initializer.initialize(tuple(self.convolution_shape_list), fan_in, fan_out)
        self.bias = bias_initializer.initialize((self.num_kernels), fan_in, fan_out)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = None
        stride_x = 1
        stride_y = 1

        if (len(self.stride_shape) == 1) and (len(input_tensor.shape) > 3):
            stride_x = self.stride_shape[0]
            stride_y = self.stride_shape[0]

        elif len(self.stride_shape) == 2:
            stride_x = self.stride_shape[0]
            stride_y = self.stride_shape[1]

        elif (len(input_tensor.shape) == 3):
            stride_x = self.stride_shape[0]

        if len(input_tensor.shape) == 4:
            output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, input_tensor.shape[2], input_tensor.shape[3]))

        elif len(input_tensor.shape) == 3:
            output_tensor = np.zeros((input_tensor.shape[0], self.num_kernels, input_tensor.shape[2]))

        for num in range(input_tensor.shape[0]):
            for filter_num in range(self.num_kernels):
                current_filter = self.weights[filter_num]
                current_bias = self.bias[filter_num]
                #print((current_bias.shape))

                if len(input_tensor.shape) == 3:
                    output_tensor[num, filter_num] = np.sum(
                        correlate2d(input_tensor[num], current_filter, mode='same')) + current_bias
                else:
                    output_channel = 0
                    for channel_num in range(input_tensor.shape[1]):
                        output_channel = output_channel + correlate2d(input_tensor[num, channel_num],
                                                                      current_filter[channel_num], mode='same')

                    output_tensor[num, filter_num, :, :] = output_channel + current_bias

        final_out = output_tensor.copy()

        if len(input_tensor.shape) == 3:
            final_out = final_out[:, :, ::stride_x]
        elif len(input_tensor.shape) == 4:
            final_out = final_out[:, :, ::stride_x, ::stride_y]

        return final_out

    def backward(self, error_tensor):

        if len(self.input_tensor.shape) == 3:

            input_tensor = np.expand_dims(self.input_tensor, axis=2)
            error_tensor = np.expand_dims(error_tensor, axis=2)
            weights = np.expand_dims(self.weights, axis=2)
            convolution_shape = tuple([self.convolution_shape[0], 1, self.convolution_shape[1]])
            stride_x, stride_y = 1, self.stride_shape[0]

        else:
            input_tensor = self.input_tensor
            weights = self.weights
            convolution_shape = self.convolution_shape
            stride_x, stride_y = self.stride_shape

        num, channel_num, X, Y = input_tensor.shape
        gradient_input = np.zeros(input_tensor.shape)

        new_weights = []

        for i in range(channel_num):
            new_weights.append((weights[:, i]))

        new_weights = np.array(new_weights)

        error_tensor_upsampled = np.zeros((num, self.num_kernels, X, Y))

        for batch_num in range(num):
            for filter_num in range(self.num_kernels):
                for i in range(error_tensor.shape[2]):
                    for j in range(error_tensor.shape[3]):
                        error_tensor_upsampled[batch_num, filter_num, i * stride_x, j * stride_y] = error_tensor[batch_num, filter_num, i, j]

        for batch_num in range(num):
            for channel in range(channel_num):
                for filter_num in range(self.num_kernels):
                    gradient_input[batch_num, channel] += convolve2d(error_tensor_upsampled[batch_num, filter_num],
                                                                     new_weights[channel, filter_num],
                                                                     mode='same',
                                                                     fillvalue=0.0)

        if gradient_input.shape[2] == 1:
            gradient_input = np.squeeze(gradient_input, axis=2)

        self.gradient_weights = np.zeros(weights.shape)

        conv_x, conv_y = convolution_shape[1:]

        for batch_num in range(num):
            for channel in range(channel_num):
                pad_x = (conv_x - 1)
                pad_y = (conv_y - 1)

                if pad_x % 2 == 0:
                    pad_x_left = pad_x // 2
                    pad_x_right = pad_x // 2

                else:
                    pad_x_left = pad_x // 2
                    pad_x_right = int(math.ceil(pad_x / 2))

                if pad_y % 2 == 0:
                    pad_y_up = pad_y // 2
                    pad_y_down = pad_y // 2

                else:
                    pad_y_up = pad_y // 2
                    pad_y_down = int(math.ceil(pad_y / 2))

                padded_input = np.pad(input_tensor[batch_num, channel],
                                      ((pad_x_left, pad_x_right), (pad_y_up, pad_y_down)))

                for filter_num in range(self.num_kernels):
                    self.gradient_weights[filter_num, channel] += correlate2d(padded_input, error_tensor_upsampled[
                        batch_num, filter_num],
                                                                              mode='valid')

        self.gradient_bias = np.zeros((self.bias.shape))

        for batch_num in range(error_tensor.shape[0]):
            for filter_num in range(error_tensor.shape[1]):
                self.gradient_bias[filter_num] += np.sum(error_tensor[batch_num, filter_num])

        if self.optimizer != None:
            weights_optimizer = copy.deepcopy(self.optimizer)
            bias_optimizer = copy.deepcopy(self.optimizer)

            self.weights = weights_optimizer.calculate_update(self.weights, self.gradient_weights)
            self.bias = bias_optimizer.calculate_update(self.bias, self.gradient_bias)

        return gradient_input