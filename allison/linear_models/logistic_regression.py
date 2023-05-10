# -*- coding: utf-8 -*-
from typing import Callable
import numpy as np
from allison.utils.metrics import predict_labels
from allison.linear_models.linear_model import LinearModel
from allison.utils.functions.activation import Sigmoid

class LogisticRegression(LinearModel):

    def __init__(self):
        super().__init__()
        self.linear_convination:np.array = None
        self.function_of_activation:Callable = Sigmoid


    def foward(self, features):

        if features.ndim == 1:
            self.linear_convination = self.bias + features*self.weights
        else:
            self.linear_convination = self.bias + features@self.weights

        prediction = self.function_of_activation(self.linear_convination)
        return prediction

    def bacward(self, labels, predictions, features):

        gradient = self.loss_function(labels, predictions, True)\
                   *self.function_of_activation(self.linear_convination, True)
        gradient = np.mean(gradient)

        if features.ndim == 1:
            gradient_weights = gradient*np.mean(features)
        else:
            gradient_weights = gradient*np.mean(features, axis=0)

        self.bias = self.bias-self.learning_rate*gradient
        self.weights = self.weights-self.learning_rate*gradient_weights

    def predict(self, features: np.array) -> np.array:
        return predict_labels(super().predict(features))

    def __str__(self)->str:
        text = \
            f"model: Logistic Regression \n" +\
            f"model_bias: {self.bias} \n" +\
            f"model_weights: {self.weights} \n"
        return text
