import numpy as np
from allison.regression.base import BaseRegressor
from typing import Callable,Union
import pandas as pd

class BaseFunctions:

	def sin(x):
		return np.sin(x)

	def cos(x):
		return np.cos(x)

	def log(x):
		return np.log(x)

	def linear(x):
		return x

	def polynomial(x,grade):
		return x**grade


functions={
	
	'sinx': BaseFunctions.sin,
	'cosx': BaseFunctions.cos,
	'lnx' : BaseFunctions.log,
	'x'   : BaseFunctions.linear,
	'polynomial': BaseFunctions.polynomial
}


class FunctionalRegression(BaseRegressor):

    """
	Functional Regression
    """

    def __init__(self,
                 base_functions: list[str],
                 loss_function: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 metric: Callable[[np.ndarray, np.ndarray], np.ndarray],
                 lr: float):
        """

        Args:
            loss_function (Callable[[np.ndarray, np.ndarray], np.ndarray]): loss function
            metric (Callable[[np.ndarray, np.ndarray], np.ndarray]): metric
            lr (float): learning ratio
            n_grade (int): grade of the polynomial
        """
        super().__init__(loss_function, metric, lr)

        self.base_functions = base_functions
        

    def calculate_kernels(self,features:np.ndarray):

        kernels = features

        for function in self.base_functions:
            kernels = np.column_stack((kernels, functions[function](features)))

            
        return kernels[:,1:]

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

        features = self.calculate_kernels(features)

        n_features = 1
        if features.ndim == 2:
            n_features = features.shape[1]

        self.init_weights(n_features)


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

