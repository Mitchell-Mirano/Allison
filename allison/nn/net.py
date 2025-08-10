from allison.nn.tensor import Tensor
from allison.nn.layers import Linear

class NeuralNetwork:
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Debes implementar forward en la subclase.")

    def parameters(self):
        params = []

        def _gather_params(obj):
            # Recorre atributos
            if isinstance(obj, Linear):
                params.extend([obj.W, obj.b])
            elif isinstance(obj, NeuralNetwork):  # Si es un submodelo
                params.extend(obj.parameters())
            elif isinstance(obj, (list, tuple)):
                for layer in obj:
                    _gather_params(layer)
            elif isinstance(obj, dict):
                for layer in obj.values():
                    _gather_params(layer)

        for layer in self.__dict__.values():
            _gather_params(layer)

        return params

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)