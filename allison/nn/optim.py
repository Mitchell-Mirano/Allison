from allison.nn.tensor import Tensor
import numpy as np

class SGD:
    def __init__(self, parameters: list[Tensor], learning_rate=1e-3):
        self.parameters = parameters
        self.lr = learning_rate

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)

    def step(self):
        for param in self.parameters:
            param.data -= self.lr * param.grad

class SGDMomentum:
    def __init__(self, parameters: list[Tensor], learning_rate=1e-3, momentum=0.9):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.vts = {}

    def zero_grad(self):
        for param in self.parameters:
            param.grad = np.zeros_like(param.grad)

    def step(self):
        for i, param in enumerate(self.parameters):
            if i not in self.vts:
                self.vts[i] = np.zeros_like(param.data)
            self.vts[i] = self.momentum * self.vts[i] + param.grad
            param.data -= self.learning_rate * self.vts[i]
