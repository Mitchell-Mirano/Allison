import numpy as np
from allison.linear_models.linear_model import LinearModel
from typing import Callable
import pandas as pd
from typing import Union
import numpy as np



class PolinomicRegression(LinearModel):
    """
    Polinomic Regression
    """

    def __init__(self,
                 loss_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 lr: float,
                 n_grade: int):
        """

        Args:
            loss_function (Callable[[np.ndarray, np.ndarray], np.ndarray]): loss function
            metric (Callable[[np.ndarray, np.ndarray], np.ndarray]): metric
            lr (float): learning ratio
            n_grade (int): grade of the polynomial
        """
        super().__init__(loss_function, metric, lr)
        self.n_grade = n_grade
        

    def calculate_kernels(self,features:np.ndarray):

        kernels = features

        for i in range(2, self.n_grade + 1):
            kernels = np.column_stack((kernels, np.power(features, i)))
               
        return kernels

    def _init_params(self,
                     features: Union[np.ndarray, pd.DataFrame],
                     labels: Union[np.ndarray, pd.Series]):
        """
        Initialize the parameters

        Args:
            features (Union[np.ndarray, pd.DataFrame]): features
            labels (Union[np.ndarray, pd.Series]): labels

        Returns:
            np.ndarray, np.ndarray: features, labels
        """
        if isinstance(features, pd.DataFrame):
            self.features_names = features.columns.to_list()

        features = features.to_numpy() if isinstance(features, pd.DataFrame) else features
        labels = labels.to_numpy() if isinstance(labels, pd.Series) else labels


        self.bias = np.random.rand(1)

        if features.ndim == 1:
            self.weights = np.random.rand(self.n_grade)
        else:
            self.weights = np.random.rand(len(features[0]))

        features = self.calculate_kernels(features)

        return features, labels


    def predict(self, features:Union[pd.DataFrame, np.ndarray]) -> np.ndarray:

        features = features.to_numpy() if isinstance(features, pd.DataFrame) else features
        features = self.calculate_kernels(features)
        return self._foward(features)
    

    def evaluate(self,
                 features_test: Union[np.ndarray, pd.DataFrame],
                 labels_test:Union[np.ndarray, pd.Series]) -> float:
        
        labels_test = labels_test.to_numpy() if isinstance(labels_test, pd.Series) else labels_test

        return self.metric(labels_test,self.predict(features_test))

