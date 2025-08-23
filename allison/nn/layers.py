from allison.nn.tensor import Tensor
import numpy as np
from allison import _cupy_available

if _cupy_available:
    import cupy as cp


class Relu:
    def __call__(self, X: Tensor):

        xp = cp if X.device == 'gpu' else np

        out = Tensor(xp.maximum(0, X.data), (X,), 'ReLU',device=X.device,requires_grad=X.requires_grad)
        
        def _backward():
            # Usar other.data para claridad
            X.grad += out.grad * (X.data > 0)
        out._backward = _backward
        return out
    

class Linear:
    def __init__(self, features: int, neurons: int):

        self.std_dev = np.sqrt(2.0 / features)  # He init para ReLU
        self.W = Tensor(np.random.normal(0, self.std_dev, size=(features, neurons)),requires_grad=True)
        self.b = Tensor(np.zeros((1, neurons)),requires_grad=True)  # Bias inicializado en 0

    def __call__(self, X: Tensor):

        return X @ self.W + self.b
    
    def to(self, device):
        self.W = self.W.to(device)
        self.b = self.b.to(device)