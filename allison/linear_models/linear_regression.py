import numpy as np
from pathlib import Path
from typing import Callable
from allison.linear_models.linear_model import LinearModel

class LinearRegression(LinearModel):

    def __init__(self):
        super().__init__()

    def _foward(self, features: np.array):
        if features.ndim == 1:
            prediction = self.bias + features*self.weights
        else:
            prediction = self.bias + features@self.weights

        return prediction

    def _bacward(self, labels: np.array, predictions: np.array, features: np.array):

        gradient = self.loss_function(labels, predictions, derivative=True)

        if features.ndim == 1:
            gradient_weights = gradient*np.mean(features)
        else:
            gradient_weights = gradient*np.mean(features, axis=0)

        self.bias = self.bias-self.learning_rate*gradient
        self.weights = self.weights-self.learning_rate*gradient_weights
