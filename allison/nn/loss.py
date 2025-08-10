import numpy as np
from allison.nn.tensor import Tensor

class MSELoss:
    def __call__(self, Y, y_pred):
        return ((Y - y_pred)**2).mean()
    

class CrossEntropyLoss:
    def __init__(self, one_hot=False):
        self.one_hot = one_hot

    def __call__(self, Y: Tensor, y_pred: Tensor) -> Tensor:
        # Paso 1: Softmax estable
        exp_logits = np.exp(y_pred.data - np.max(y_pred.data, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        batch_size = Y.data.shape[0]

        # Paso 2: Calcular la pérdida
        if self.one_hot:
            log_probs = np.log(probs + 1e-9)
            loss_val = -np.mean(np.sum(Y.data * log_probs, axis=-1))
        else:
            Y_indices = Y.data.flatten().astype(int)
            correct_log_probs = -np.log(probs[np.arange(batch_size), Y_indices] + 1e-9)
            loss_val = np.mean(correct_log_probs)
        
        # Paso 3: Crear el Tensor de pérdida para el backpropagation
        out = Tensor(loss_val, (y_pred,), 'CrossEntropyLoss')

        # Paso 4: Unificar el backpropagation
        def _backward():
            # Crear las etiquetas one-hot, independientemente del formato original
            if self.one_hot:
                Y_one_hot = Y.data
            else:
                Y_one_hot = np.zeros_like(probs)
                Y_one_hot[np.arange(batch_size), Y.data.flatten().astype(int)] = 1
            
            # La derivada combinada es la misma para ambos casos
            grad_combined = (probs - Y_one_hot) / batch_size
            y_pred.grad += out.grad * grad_combined
            
        out._backward = _backward
        return out