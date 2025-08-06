from allison.nn.models import NeuralNetwork
import numpy as np

class SGD:

    def __init__(self, learning_rate:float = 0.001):
        self.learning_rate = learning_rate

    def update(self, network:NeuralNetwork):
        for layer in network.layers.values():
            layer.weights -= self.learning_rate * layer.DL
            layer.bias -= self.learning_rate * layer.Db


class SGDMomentum:

    def __init__(self, learning_rate:float = 0.001, momentum:float = 0.1):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.vts = {}

    def update(self, network:NeuralNetwork):

        for i, layer in network.layers.items():
            if i not in self.vts:
                self.vts[i] = {
                    'Vw': np.zeros(layer.weights.shape),
                    'Vb': np.zeros(layer.bias.shape)
                }

            self.vts[i]['Vw'] = self.momentum * self.vts[i]['Vw'] + layer.DL
            self.vts[i]['Vb'] = self.momentum * self.vts[i]['Vb'] + layer.Db

            layer.weights -= self.learning_rate * self.vts[i]['Vw']
            layer.bias -= self.learning_rate * self.vts[i]['Vb']