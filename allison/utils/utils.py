from allison.tensor.tensor import tensor
import numpy as np
from allison.cupy.cupy import _cupy_available

if _cupy_available:
    import cupy as cp


def sigmoid(X: tensor | np.ndarray| cp.ndarray) -> tensor | np.ndarray:

    if isinstance(X, tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        probs = 1 / (1 + xp.exp(-X.data))
    else:
        xp = cp if isinstance(X, cp.ndarray) else np

        probs = 1 / (1 + xp.exp(-X))

    if isinstance(X, tensor):
        return tensor(probs)
    
    return 1 / (1 + xp.exp(-X.data))


def softmax(X: tensor | np.ndarray | cp.ndarray, axis=1, keepdims=True) -> tensor | np.ndarray:

    if isinstance(X, tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        exp_logits = xp.exp(X.data - xp.max(X.data, axis=axis, keepdims=keepdims))
        probs = exp_logits / xp.sum(exp_logits, axis=axis, keepdims=keepdims)
    else:
        xp = cp if isinstance(X, cp.ndarray) else np
        exp_logits = xp.exp(X - xp.max(X, axis=axis, keepdims=keepdims))
        probs = exp_logits / xp.sum(exp_logits, axis=axis, keepdims=keepdims)

    if isinstance(X, tensor):
        return tensor(probs)
    
    return probs


def argmax(X: tensor | np.ndarray | cp.ndarray, axis=1, keepdims=True) -> tensor | np.ndarray:

    if isinstance(X, tensor):
        xp = cp if X.device == 'gpu' and _cupy_available else np
        return tensor(X.data.argmax(axis=axis, keepdims=keepdims),device=X.device)
    else:
        xp = cp if isinstance(X, cp.ndarray) else np
        return X.argmax(axis=axis, keepdims=keepdims)