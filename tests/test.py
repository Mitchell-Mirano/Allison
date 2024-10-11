import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from allison.linear_models.linear_regression import LinearRegression
from allison.utils.functions.loss import mean_square_error
from allison.utils.metrics import r2_score

if __name__=='__main__':
    
    x=np.linspace(2,20,100)
    y=2*x + 1 + 2*np.sin(x) + np.random.randn(100)

    model=LinearRegression()

    model.optimizers(loss_function=mean_square_error,
                    learning_rate=0.001, 
                    metric=r2_score)
    
    hist_train=model.train(n_iters=10,
            features=x, 
            labels=y, 
            callbacks_period=2)
    
    plt.figure(figsize=(12,8))
    plt.scatter(x,y,s=50)
    plt.plot(x,model.predict(x), lw=3, c='red')
    plt.show()