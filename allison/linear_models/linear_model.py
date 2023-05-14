import numpy as np
from pathlib import Path
from typing import Callable


class LinearModel:

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

        self.loss_function = loss_function
        self.metric = metric
        self.learning_rate = learning_rate

    def init_params(self, features: np.array):
        self.bias = np.random.rand(1)
        if features.ndim == 1:
            self.weights = np.random.rand(1)
        else:
            self.weights = np.random.rand(len(features[0]))

    def foward(self, features: np.array):
        pass

    def bacward(self, labels: np.array, predictions: np.array, features: np.array):

        pass

    def train(self,
              features: np.array,
              labels: np.array,
              n_iters: int,
              callbacks_period: int = 1,
              history_train: bool = False):

        if history_train:
            self.history_train = {
                'iter': [],
                'loss': [],
                'precision': []
            }

        if callbacks_period == 1:
            callbacks_period = np.max([1, int(n_iters/10)])

        self.init_params(features)

        for i in range(n_iters):
            predictions = self.foward(features)
            self.bacward(labels, predictions, features)

            if (i+1) % callbacks_period == 0:
                score = self.metric(labels, predictions)
                loss = self.loss_function(labels, predictions)
                if history_train:
                    self.history_train['iter'].append(i+1)
                    self.history_train['loss'].append(loss)
                    self.history_train['precision'].append(score)
                print(f"Iter:\t{i+1}\t{50*'='+'>'}\t precision: {score:.3f}% \n")

    def predict(self, features: np.array) -> np.array:
        predictions = self.foward(features)
        return predictions

    def evaluate(self, labels_test: np.array, features_test: np.array) -> float:
        return self.metric(labels_test, self.predict(features_test))

    def save_weights(self, path: Path) -> None:
        with open(path, 'wb') as f:
            np.save(f, self.bias)
            np.save(f, self.weights)

    def load_weights(self, path: Path) -> None:
        with open(path, 'rb') as f:
            bias = np.load(f)
            weights = np.load(f)
            self.bias = bias
            self.weights = weights

    
    def __repr__(self) -> str:
        return self.__str__()