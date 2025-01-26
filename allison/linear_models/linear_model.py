import numpy as np
from pathlib import Path
from typing import Callable
import pandas as pd
from typing import Union

class LinearModel:

    """
    Base class for linear models

    Attributes:
        bias: float
        weights: np.array
        loss_function: Callable[[np.array, np.array], np.array]
        metric: Callable[[np.array, np.array], np.array]
        learning_rate: float
        history_train: dict
    """

    def __init__(self):
        self.bias: float = None
        self.weights: np.array = None
        self.loss_function: Callable[[np.array, np.array], np.array] = None
        self.metric: Callable[[np.array, np.array], np.array] = None
        self.learning_rate: float = None
        self.history_train: dict = None

    def optimizers(self,
                   loss_function: Callable[[np.array, np.array], np.array],
                   metric: Callable[[np.array, np.array], np.array],
                   learning_rate: float):
        
        """
        Method to set loss function, metric and learning rate

        Args:
            loss_function (Callable[[np.array, np.array], np.array]): loss function
            metric (Callable[[np.array, np.array], np.array]): metric
            learning_rate (float): learning rate
        """

        self.loss_function = loss_function
        self.metric = metric
        self.learning_rate = learning_rate

    def _init_params(self, features: Union[np.array, pd.DataFrame], labels: Union[np.array, pd.Series]):

        features = features.to_numpy() if isinstance(features, pd.DataFrame) else features
        labels = labels.to_numpy() if isinstance(labels, pd.Series) else labels

        self.bias = np.random.rand(1)

        if features.ndim == 1:
            self.weights = np.random.rand(1)
        else:
            self.weights = np.random.rand(len(features[0]))

        return features, labels

    def _foward(self, features: np.array):
        pass

    def _bacward(self, labels: np.array, predictions: np.array, features: np.array):

        pass

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
                print(labels,predictions)
                score = self.metric(labels, predictions)
                loss = self.loss_function(labels, predictions)
                self.history_train['iter'].append(i+1)
                self.history_train['loss'].append(loss)
                self.history_train['precision'].append(score)
                self.history_train['params'].append({
                    'bias': self.bias,
                    'weights': self.weights
                })

                print(f"Iter:\t{i+1}\t{50*'='+'>'}\t {self.loss_function.__name__}: {loss:.3f}% \n")

    def predict(self, features: Union[np.array, pd.DataFrame]) -> np.array:

        """
        Method to predict the labels

        Args:
            features (Union[np.array, pd.DataFrame]): features

        Returns:
            np.array: predictions
        """
        
        features = features.to_numpy() if isinstance(features, pd.DataFrame) else features
        predictions = self._foward(features)
        return predictions

    def evaluate(self,
                 features_test: Union[np.array, pd.DataFrame],
                 labels_test:Union[np.array, pd.Series]) -> float:
        
        """
        Method to evaluate the model

        Args:
            labels_test (Union[np.array, pd.Series]): labels
            features_test (Union[np.array, pd.DataFrame]): features

        Returns:
            float: score
        """

        labels_test = labels_test.to_numpy() if isinstance(labels_test, pd.Series) else labels_test
        features_test = features_test.to_numpy() if isinstance(features_test, pd.DataFrame) else features_test

        return self.metric(labels_test, self.predict(features_test))
    

    def __str__(self) -> str:
        text = f"""
        model: {self.__class__.__name__} \n
        model_bias: {self.bias} \n
        model_weights: {self.weights} \n
        """
        return text
    
    def __repr__(self) -> str:
        return self.__str__()