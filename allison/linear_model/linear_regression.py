import pandas as pd
import numpy as np

import allison
from allison.nn.loss import MSELoss
from allison.nn.layers import Linear
from allison.optim.optim import RMSprop,Adam
from allison.tensor.tensor import tensor
from allison.metrics import r2_score


class LinearRegression:
    def __init__(self,optimizer='RMSprop',lr=0.001,epsilon=1e-6):
        self._linear: Linear = None
        self._optimizer_name = optimizer
        self._optimizer = None
        self._loss = MSELoss()
        self._features_names = None
        self.lr = lr
        self.epsilon = epsilon


    def __str__(self):

        return f"LinearRegression()"
    
    def __repr__(self):
        return self.__str__()

    def _to_tensor(self, X):

        if isinstance(X, tensor):
            return X

        if isinstance(X, (pd.DataFrame,pd.Series, np.ndarray, list)):

            if isinstance(X, pd.DataFrame):
                self._features_names = list(X.columns)
                X = X.to_numpy()
            elif isinstance(X, pd.Series):
                self._features_names = X.name
                X = X.to_numpy().reshape(-1, 1)
            elif isinstance(X, np.ndarray):
                if X.ndim > 1:
                    self._features_names = [f"F{i}" for i in range(X.shape[1])]
                else:
                    self._features_names = ["F0"]
                    X = X.reshape(-1, 1)
            elif isinstance(X, list):
                X = np.array(X)
                if X.ndim > 1:
                    self._features_names = [f"F{i}" for i in range(X.shape[1])]
                else:
                    self._features_names = ["F0"]
                    X = X.reshape(-1, 1)
            else:
                raise TypeError("La entrada debe ser un ndarray de NumPy o un DataFrame de Pandas o una lista.")
            
            return tensor(X)

    def fit(self, X, Y,iters=10000,verbose=False):
        
        X = self._to_tensor(X)
        Y = self._to_tensor(Y)

        self._linear = Linear(X.shape[1], 1)
        
        if self._optimizer_name == 'Adam':
            self._optimizer = Adam(self._linear.parameters(), lr=self.lr)
        elif self._optimizer_name == 'RMSprop':
            self._optimizer = RMSprop(self._linear.parameters(), lr=self.lr)


        prev_loss = np.inf

        for itr in range(iters + 1):
            self._optimizer.zero_grad()
            y_pred = self._linear(X)
            loss = self._loss(y_pred, Y)
            loss.backward()
            self._optimizer.step()

            if itr % 100 == 0:
                if (prev_loss - loss.data) < self.epsilon:
                    break
                if verbose:
                    print(f"Epoch: {itr}, Loss: {loss.data}")
            prev_loss = loss.data
                
                

    def predict(self, X):
        
        X = self._to_tensor(X)
        with allison.no_grad():
            return self._linear(X).data.flatten()
    
    def score(self, X, Y):
        return r2_score(Y, self.predict(X))

    @property
    def features_names(self):
        return self._features_names
    
    @property
    def coef_(self):
        if self._linear.W.data.shape[0] > 1:
            return self._linear.W.data.flatten()
        else:
            return self._linear.W.data[0]

    @property
    def intercept_(self):
        if self._linear.b.data.shape[0] > 1:
            return self._linear.b.data.flatten()
        else:
            return self._linear.b.data[0][0]