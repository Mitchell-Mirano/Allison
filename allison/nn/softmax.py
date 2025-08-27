from allison.tensor.tensor import tensor
import numpy as np


def softmax(X: tensor | np.ndarray) -> tensor | np.ndarray:

    exp_logits = np.exp(X.data - np.max(X.data, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    if isinstance(X, tensor):
        return tensor(probs)
    
    return probs
