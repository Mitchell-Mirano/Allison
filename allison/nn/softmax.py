from allison.nn.tensor import Tensor
import numpy as np


def softmax(X: Tensor | np.ndarray) -> Tensor | np.ndarray:

    exp_logits = np.exp(X.data - np.max(X.data, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    if isinstance(X, Tensor):
        return Tensor(probs)
    
    return probs
