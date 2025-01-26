# -*- coding: utf-8 -*-
from typing import Callable
import numpy as np
from allison.utils.metrics import predict_labels
from allison.linear_models.linear_model import LinearModel
from allison.utils.functions.activation import sigmoid
from typing import Union
import pandas as pd


class LogisticRegression(LinearModel):

    def __init__(self):
        super().__init__()
        self.linear_convination:np.array = None
        self.function_of_activation:Callable = sigmoid


    def _foward(self, features: np.array):

        if features.ndim == 1:
            self.linear_convination = self.bias + features*self.weights
        else:
            self.linear_convination = self.bias + features@self.weights

        prediction = self.function_of_activation(self.linear_convination)
        return prediction

    def _bacward(self, labels: np.array, predictions: np.array, features: np.array):

        gradient = self.loss_function(labels, predictions, True)\
                   *self.function_of_activation(self.linear_convination, True)
        gradient = np.mean(gradient)


        if features.ndim == 1:
            gradient_weights = gradient*np.mean(features)
        else:
            gradient_weights = gradient*np.mean(features, axis=0)
        
        self.bias = self.bias-self.learning_rate*gradient
        self.weights = self.weights-self.learning_rate*gradient_weights


    def train(self,
            features: Union[np.array, pd.DataFrame],
            labels: Union[np.array, pd.Series],
            n_iters: int,
            callbacks_period: int = 1,
            history_train: bool = False):
    
        """
        Method to train the model

        Args:
            features (Union[np.array, pd.DataFrame]): features
            labels (Union[np.array, pd.Series]): labels
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

                print(f"Iter:\t{i+1}\t{50*'='+'>'}\t {self.loss_function.__name__}: {loss:.3f}\t {self.metric.__name__}: {score:.2f}% \n")


    def predict(self, features: np.array,probs:bool=False) -> np.array:

        features = features.to_numpy() if isinstance(features, pd.DataFrame) else features

        if probs:
            return self._foward(features)
        else:
            return np.where(self._foward(features)<=0.5,0,1)

