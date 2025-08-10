import numpy as np


def r2_score(Y, y_pred):
    sr = np.mean((y_pred.data - Y.data)**2)
    sy = np.mean((Y.data - np.mean(Y.data))**2)

    return 1-(sr/sy)


def accuracy(preds,labels):
    
    if labels.ndim==1:
        return np.mean(preds==labels)
    return np.mean(np.sum(preds==labels,axis=1)/labels.shape[1])

