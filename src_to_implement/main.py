from matplotlib import pyplot as plt

from Optimizers import Sgd
from Layers.ReLU import ReLU
from Layers.FullyConnected import FullyConnected
import numpy as np
from Optimization.Loss import CrossEntropyLoss

from NeuralNetworkTests import L2Loss
#from Layers.Helpers import Helpers
from src_to_implement import Optimizers, NeuralNetwork
from src_to_implement.Layers import Helpers
from src_to_implement.Layers.SoftMax import SoftMax
from src_to_implement.Optimization import Loss

if __name__ == '__main__':
    #optimizer = Sgd(1.)
    #result = optimizer.calculate_update(1., 1.)
    #print(result)
    #result1 = optimizer.calculate_update(result, 1.)
    #print(result1)

    fully_conn = FullyConnected(4, 3)
    #print(fully_conn.input_tensor)
    output_tensor = fully_conn.forward(fully_conn.input_tensor)
    #output = fully_conn.backward(output_tensor)

    # layer = FullyConnected(4,3)
    # optimizer = Optimizers.Sgd(1)
    # for _ in range(10):
    #     output_tensor = layer.forward(layer.input_tensor)
    #     error_tensor = np.zeros([9, 3])
    #     error_tensor -= output_tensor
    #     # print(error_tensor.shape)
    #     layer.backward(error_tensor)
    #     new_output_tensor = layer.forward(layer.input_tensor)
    #     print(np.sum(np.power(output_tensor, 2)))
    #     print(np.sum(np.power(new_output_tensor, 2)))

    # input_tensor = np.random.rand(9, 4)

    # categories = 9
    # label_tensor = np.zeros([9, categories])
    # input_tensor = np.zeros([9, 4])
    # layer = FullyConnected(4, 3)
    # layer.optimizer = Optimizers.Sgd(1)
    # for _ in range(10):
    #     output_tensor = layer.forward(input_tensor)
    #     error_tensor = np.zeros([9, 4])
    #     error_tensor -= output_tensor
    #     layer.backward(error_tensor)
    #     new_output_tensor = layer.forward(input_tensor)
    #     print(np.sum(np.power(output_tensor, 2)))
    #
    #     print(np.sum(np.power(new_output_tensor, 2)))

    #self.assertLessEqual(np.sum(difference), 1e-5)

    #layer.test_gradient()

    #output_tensor = layer.backward(self.input_tensor * 2)

    # net = NeuralNetwork.NeuralNetwork(Optimizers.Sgd(1e-3))
    # categories = 3
    # input_size = 4
    # net.data_layer = Helpers.IrisData(50)
    # net.loss_layer = Loss.CrossEntropyLoss()
    #
    # fcl_1 = FullyConnected(input_size, categories)
    # net.append_layer(fcl_1)
    # net.append_layer(ReLU())
    # fcl_2 = FullyConnected(categories, categories)
    # net.append_layer(fcl_2)
    # net.append_layer(SoftMax())
    #
    # net.train(10)
    # plt.figure('Loss function for a Neural Net on the Iris dataset using SGD')
    # plt.plot(net.loss, '-x')
    # plt.show()

    # self.assertNotEqual(out, out2)

