import numpy as np
from pathlib import Path
from typing import Callable
from allison.regression.base import BaseRegressor

class LinearRegression(BaseRegressor):

    def __init__(self, 
                loss_function: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                metric: Callable[[np.ndarray, np.ndarray], np.ndarray], 
                lr: float):
        super().__init__(loss_function, metric, lr)
