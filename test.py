import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.metrics import accuracy, recall,precision, r2_score
from utils.functions.loss import mean_square_error, binary_cross_entropy

from linear_models import LogisticRegression
from linear_models import LinearRegression


if __name__=='__main__':
    x=np.linspace(2,20,100)
    y=2*x +1 +2 *np.random.randn(100)

    plt.scatter(x,y)
    plt.show()

    model=LinearRegression()
    model.optimizers(mean_square_error,0.001,r2_score)
    model.train(30,x,y)

    plt.scatter(x,y)
    plt.plot(x,model.predict(x))
    plt.show()

    