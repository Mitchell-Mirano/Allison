import numpy as np
from utils.metrics import accuracy, recall,precision, r2_score
from utils.functions.loss import mean_square_error, binary_cross_entropy

def test():

    labels=np.array([1,1,1,1,0,0,0])
    predictions=np.array([0.9,0.8,0.95,0.15,0.20,0.30,0.89])

    mse=mean_square_error(labels, predictions)
    bce=binary_cross_entropy(labels,predictions)
    r2=r2_score(labels, predictions)
    return mse, bce, r2

if __name__=='__main__':
    print(test())