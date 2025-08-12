import numpy as np


_autograd_enabled = True

class no_grad:
    def __enter__(self):
        global _autograd_enabled
        self.prev = _autograd_enabled
        _autograd_enabled = False
    def __exit__(self, exc_type, exc_val, exc_tb):
        global _autograd_enabled
        _autograd_enabled = self.prev

def _noop():
    """Función vacía para usar como backward por defecto."""
    return None

class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = data if isinstance(data, np.ndarray) else np.array(data)
        self.grad = np.zeros_like(self.data)
        self._backward = _noop
        global _autograd_enabled
        self._prev = set(_children) if _autograd_enabled else set()
        self._op = _op if _autograd_enabled else ''

    def __getstate__(self) -> object:
        return {'data': self.data}
    
    def __setstate__(self, state):
        self.data = state['data']
        self.grad = np.zeros_like(self.data)
        self._backward = _noop
        self._prev = set()
        self._op = ''
 

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        global _autograd_enabled

        if not _autograd_enabled:
            return Tensor(self.data + other.data)

        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            grad_self = Tensor._match_shape(out.grad, self.data.shape)
            grad_other = Tensor._match_shape(out.grad, other.data.shape)
            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        global _autograd_enabled

        if not _autograd_enabled:
            return Tensor(self.data - other.data)
        
        out = Tensor(self.data - other.data, (self, other), '-')

        def _backward():
            grad_self = Tensor._match_shape(out.grad, self.data.shape)
            grad_other = Tensor._match_shape(out.grad, other.data.shape)
            self.grad += grad_self
            other.grad -= grad_other

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        global _autograd_enabled

        if not _autograd_enabled:
            return Tensor(self.data * other.data)
        
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            grad_self = Tensor._match_shape(other.data * out.grad, self.data.shape)
            grad_other = Tensor._match_shape(self.data * out.grad, other.data.shape)
            self.grad += grad_self
            other.grad += grad_other

        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        global _autograd_enabled

        if not _autograd_enabled:
            return Tensor(self.data @ other.data)
        
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            grad_self = out.grad @ other.data.T
            grad_other = self.data.T @ out.grad
            # Ajustar dimensiones para broadcast
            self.grad += Tensor._match_shape(grad_self, self.data.shape)
            other.grad += Tensor._match_shape(grad_other, other.data.shape)

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += out.grad * other * (self.data**(other-1))

        out._backward = _backward
        return out
    
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)
        self.grad = np.ones_like(self.data)

        for node in reversed(topo):
            node._backward()

    
    def mean(self):

        global _autograd_enabled
        if not _autograd_enabled:
            return Tensor(np.mean(self.data))
        
        out = Tensor(np.mean(self.data), (self,), 'mean')

        def _backward():
            self.grad += out.grad * np.ones_like(self.data) / self.data.size
        out._backward = _backward
        return out

    @staticmethod
    def _match_shape(grad, shape):
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        for axis, dim in enumerate(shape):
            if dim == 1 and grad.shape[axis] != 1:
                grad = grad.sum(axis=axis, keepdims=True)

        return grad
    

    def __str__(self) -> str:
        return f"Tensor(\n{self.data}, shape={self.data.shape})"

    def __repr__(self):
        return self.__str__()
    
    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def to_numpy(self):
        return self.data
