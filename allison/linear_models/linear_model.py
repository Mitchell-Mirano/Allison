import numpy as np
import pandas as pd
from typing import Union
from typing import Callable

class LinearModel:

    def __init__(self, 
                loss_function: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                metric: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                lr: float):
        
        """
        Args:
            loss_function (Callable[[np.ndarray, np.ndarray], np.ndarray]): loss function
            metric (Callable[[np.ndarray, np.ndarray], np.ndarray]): metric
            lr (float): learning ratio
        """
        
        self.bias: float = None
        self.weights: np.ndarray = None
        self.features_names:list = None
        self.loss_function: Callable[[np.ndarray, np.ndarray], np.ndarray] = loss_function
        self.metric: Callable[[np.ndarray, np.ndarray], np.ndarray] = metric
        self.lr: float = lr
        self.history_train: dict = None
        

    def _init_params(self, features: Union[np.ndarray, pd.DataFrame], labels: Union[np.ndarray, pd.Series]):

        if isinstance(features, pd.DataFrame):
            self.features_names = features.columns.to_list()

        features = features.to_numpy() if isinstance(features, pd.DataFrame) else features
        labels = labels.to_numpy() if isinstance(labels, pd.Series) else labels



        self.bias = np.array([0.0])

        if features.ndim == 1:
            std_dev = np.sqrt(2.0 / 1)  # He init para ReLU
            self.weights = np.random.normal(0, std_dev, 1)
        else:
            std_dev = np.sqrt(2.0 / len(features[0]))  # He init para ReLU
            self.weights = np.random.normal(0, std_dev, len(features[0]))

        return features, labels

    def _foward(self, features: np.ndarray):
        if features.ndim == 1:
            prediction = self.bias + features*self.weights
        else:
            prediction = self.bias + features@self.weights

        return prediction

    def _bacward(self, labels: np.ndarray, predictions: np.ndarray, features: np.ndarray):

        gradient = self.loss_function(labels, predictions, derivative=True)

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
                loss = self.loss_function(labels, predictions)
                score = self.metric(labels, predictions)
                self.history_train['iter'].append(i+1)
                self.history_train['loss'].append(loss)
                self.history_train['precision'].append(score)
                self.history_train['params'].append({
                    'bias': self.bias,
                    'weights': self.weights
                })

                print(f"Iter:\t{i+1}\t{50*'='+'>'}\t {self.loss_function.__name__}: {loss:.3f}% \t {self.metric.__name__}: {100*score:.2f}% \n")

    def predict(self, features: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:

        """
        Method to predict the labels

        Args:
            features (Union[np.ndarray, pd.DataFrame]): features

        Returns:
            np.ndarray: predictions
        """
        
        features = features.to_numpy() if isinstance(features, pd.DataFrame) else features
        return self._foward(features)

    def evaluate(self,
                 features_test: Union[np.ndarray, pd.DataFrame],
                 labels_test:Union[np.ndarray, pd.Series]) -> float:
        
        """
        Method to evaluate the model

        Args:
            labels_test (Union[np.ndarray, pd.Series]): labels
            features_test (Union[np.ndarray, pd.DataFrame]): features

        Returns:
            float: score
        """

        labels_test = labels_test.to_numpy() if isinstance(labels_test, pd.Series) else labels_test
        features_test = features_test.to_numpy() if isinstance(features_test, pd.DataFrame) else features_test

        return self.metric(labels_test, self.predict(features_test))
    

    def __str__(self) -> str:
        text = f"""
        model: {self.__class__.__name__} \n
        bias: {self.bias} \n
        weights: {self.weights} \n
        features_names: {self.features_names} \n
        """
        return text
    
    def __repr__(self) -> str:
        return self.__str__()