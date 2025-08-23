from allison.nn.tensor import Tensor
from allison.nn.layers import Linear

class NeuralNetwork:
    def __init__(self):
        super().__init__()
        self.device = None

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError("Debes implementar forward en la subclase.")

    def parameters(self):
        """Conserva tu implementación actual si te resulta útil para optimizadores,
        pero ya no dependemos de esto para mover a device."""
        params = []

        def _gather_params(obj):
            if isinstance(obj, Linear):
                if getattr(obj, "W", None) is not None: params.append(obj.W)
                if getattr(obj, "b", None) is not None: params.append(obj.b)
            elif isinstance(obj, NeuralNetwork):
                params.extend(obj.parameters())
            elif isinstance(obj, (list, tuple)):
                for layer in obj: _gather_params(layer)
            elif isinstance(obj, dict):
                for layer in obj.values(): _gather_params(layer)

        for layer in self.__dict__.values():
            _gather_params(layer)
        return params

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def to(self, device):
        """Mueve TODOS los tensores encontrados en el grafo de atributos al device."""
        def _apply(obj):
            # 1) Tensors: retornar el tensor movido
            if isinstance(obj, Tensor):
                return obj.to(device)

            # 2) Capas Linear: mover pesos y bias y devolver el propio objeto
            if isinstance(obj, Linear):
                if getattr(obj, "W", None) is not None:
                    obj.W = _apply(obj.W)
                if getattr(obj, "b", None) is not None:
                    obj.b = _apply(obj.b)
                return obj

            # 3) Submodelos
            if isinstance(obj, NeuralNetwork):
                for k, v in obj.__dict__.items():
                    # Evitar ciclos si quieres, pero en general está bien:
                    setattr(obj, k, _apply(v))
                obj.device = device
                return obj

            # 4) Estructuras compuestas
            if isinstance(obj, list):
                return [ _apply(v) for v in obj ]
            if isinstance(obj, tuple):
                return tuple(_apply(v) for v in obj)
            if isinstance(obj, dict):
                return { k: _apply(v) for k, v in obj.items() }

            # 5) Otros tipos: devolver tal cual
            return obj

        _apply(self)
        self.device = device
        return self  # para encadenar llamadas

    def weights(self):

        if self.device == 'cpu':
            return self.__dict__
        return self.to('cpu').__dict__

    def load_weights(self, weights):
        self.__dict__.update(weights)
