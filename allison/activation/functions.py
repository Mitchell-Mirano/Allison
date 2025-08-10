from allison.base.tensor import Tensor
import numpy as np

class Relu:
    def __call__(self, other: Tensor):
        out = Tensor(np.maximum(0, other.data), (other,), 'ReLU')
        def _backward():
            # Usar other.data para claridad
            other.grad += out.grad * (other.data > 0)
        out._backward = _backward
        return out

class Linear:
    def __init__(self, features: int, neurons: int):
        self.std_dev = np.sqrt(2.0 / features)  # He init para ReLU
        self.W = Tensor(np.random.normal(0, self.std_dev, size=(features, neurons)))
        self.b = Tensor(np.zeros((1, neurons)))  # Bias inicializado en 0

    def __call__(self, other: Tensor):
        return other @ self.W + self.b