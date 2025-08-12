# -*- coding: utf-8 -*-
from typing import Callable
import numpy as np
from allison.utils.metrics import predict_labels
from allison.regression.base import BaseRegressor
from allison.utils.functions.activation import sigmoid
from typing import Union
import pandas as pd


class LogisticRegression(BaseRegressor):

    def __init__(self, 
                loss_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
                metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
                lr: float):
        super().__init__(loss_function, metric, lr)
        self.linear_convination:np.ndarray = None
        self.function_of_activation:Callable = sigmoid


    def init_weights(self,n_features: int):

        # init for sigmoid

        std_dev = np.sqrt(2.0 / (n_features +1))

        self.weights = np.random.normal(0, std_dev, n_features)


    def _foward(self, features: np.ndarray):

        if features.ndim == 1:
            self.linear_convination = self.bias + features*self.weights
        else:
            self.linear_convination = self.bias + features@self.weights

        prediction = self.function_of_activation(self.linear_convination)
        return prediction

    def _bacward(self, labels: np.ndarray, predictions: np.ndarray, features: np.ndarray):

        gradient = self.loss_function(labels, predictions, True)\
                   *self.function_of_activation(self.linear_convination, True)
        gradient = np.mean(gradient)


        if features.ndim == 1:
            gradient_weights = gradient*np.mean(features)
        else:
            gradient_weights = gradient*np.mean(features, axis=0)
        
        self.bias = self.bias-self.lr*gradient
        self.weights = self.weights-self.lr*gradient_weights


    def train(self,
            features: Union[np.ndarray, pd.DataFrame],
            labels: Union[np.ndarray, pd.Series],
            n_iters: int,
            callbacks_period: int = 1,
            history_train: bool = False):
    
        """
        Method to train the model

        Args:
            features (Union[np.ndarray, pd.DataFrame]): features
            labels (Union[np.ndarray, pd.Series]): labels
            n_iters (int): number of iterations
            callbacks_period (int, optional): period of callbacks. The default is 1.
            history_train (bool, optional): save history of train. The default is False.
        """
    
        features, labels = self._init_params(features,labels)
        
        if callbacks_period == 1:
            callbacks_period = np.max([1, int(n_iters/10)])

        if history_train:
            self.history_train = {
                'iter': [],
                'loss': [],
                'precision': [],
                'params': [{
                    'bias': self.bias,
                    'weights': self.weights
                }]
            }



        for i in range(n_iters):
            predictions = self._foward(features)
            self._bacward(labels, predictions, features) 


            if history_train and (i+1) % callbacks_period == 0:
                
                score = self.metric(labels, np.where(predictions<=0.5,0,1))
                loss = self.loss_function(labels, predictions)
                self.history_train['iter'].append(i+1)
                self.history_train['loss'].append(loss)
                self.history_train['precision'].append(score)
                self.history_train['params'].append({
                    'bias': self.bias,
                    'weights': self.weights
                })

                print(f"Iter:\t{i+1}\t{50*'='+'>'}\t {self.loss_function.__name__}: {loss:.3f}\t {self.metric.__name__}: {100*score:.2f}% \n")


    def predict(self, features: np.ndarray,probs:bool=False) -> np.ndarray:

        features = features.to_numpy() if isinstance(features, pd.DataFrame) else features

        if probs:
            return self._foward(features)
        else:
            return np.where(self._foward(features)<=0.5,0,1)

