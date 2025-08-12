import numpy as np
import pandas as pd
from typing import Union, Optional

class BaseScaler:
    """Clase base para todos los escaladores, implementando métodos comunes."""
    def __init__(self):
        """Inicializa los parámetros del escalador."""
        pass

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        """Ajusta el escalador a los datos de entrada."""
        raise NotImplementedError("El método fit debe ser implementado por subclases.")

    def transform(self, X: Union[np.ndarray, pd.DataFrame]):
        """Transforma los datos usando los parámetros ajustados."""
        raise NotImplementedError("El método transform debe ser implementado por subclases.")

    def fit_transform(self, X: Union[np.ndarray, pd.DataFrame]):
        """Ajusta el escalador y transforma los datos en un solo paso."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]):
        """Invierte la transformación de los datos."""
        raise NotImplementedError("El método inverse_transform debe ser implementado por subclases.")

    def _validate_input(self, X: Union[np.ndarray, pd.DataFrame]):
        """Valida que la entrada sea un array de NumPy o un DataFrame de Pandas."""
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise TypeError("La entrada debe ser un ndarray de NumPy o un DataFrame de Pandas.")
        return X

    def __repr__(self) -> str:
        """Representación de cadena de la clase."""
        params = ", ".join(f"{k}={v}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({params})"

#---

class MinMaxScaler(BaseScaler):
    """
    Escala las características a un rango dado, típicamente [0, 1].

    La fórmula es: X_scaled = (X - X.min) / (X.max - X.min)
    """
    def __init__(self):
        super().__init__()
        self.min: Optional[np.ndarray] = None
        self.max: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        X = self._validate_input(X)
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]):
        X = self._validate_input(X)
        if self.min is None or self.max is None:
            raise ValueError("El escalador debe ser ajustado con fit() antes de transformar.")
        
        # Evitar división por cero
        denominador = self.max - self.min
        denominador[denominador == 0] = 1e-9  # Pequeño valor para evitar el error
        
        return (X - self.min) / denominador

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]):
        X = self._validate_input(X)
        if self.min is None or self.max is None:
            raise ValueError("El escalador debe ser ajustado con fit() antes de invertir.")
        return X * (self.max - self.min) + self.min

#---

class StandardScaler(BaseScaler):
    """
    Estandariza las características eliminando la media y escalando a la varianza unitaria.

    La fórmula es: X_scaled = (X - X.mean) / X.std
    """
    def __init__(self):
        super().__init__()
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        X = self._validate_input(X)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]):
        X = self._validate_input(X)
        if self.mean is None or self.std is None:
            raise ValueError("El escalador debe ser ajustado con fit() antes de transformar.")
        
        # Evitar división por cero
        std_safe = self.std.copy()
        std_safe[std_safe == 0] = 1e-9
        
        return (X - self.mean) / std_safe

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]):
        X = self._validate_input(X)
        if self.mean is None or self.std is None:
            raise ValueError("El escalador debe ser ajustado con fit() antes de invertir.")
        return X * self.std + self.mean

#---

class RobustScaler(BaseScaler):
    """
    Escala las características utilizando estadísticas robustas a valores atípicos.

    La fórmula es: X_scaled = (X - mediana) / (Q3 - Q1)
    """
    def __init__(self):
        super().__init__()
        self.median: Optional[np.ndarray] = None
        self.q1: Optional[np.ndarray] = None
        self.q3: Optional[np.ndarray] = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame]):
        X = self._validate_input(X)
        self.median = np.quantile(X, 0.5, axis=0)
        self.q1 = np.quantile(X, 0.25, axis=0)
        self.q3 = np.quantile(X, 0.75, axis=0)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]):
        X = self._validate_input(X)
        if self.median is None or self.q1 is None or self.q3 is None:
            raise ValueError("El escalador debe ser ajustado con fit() antes de transformar.")
        
        # Evitar división por cero
        iqr = self.q3 - self.q1
        iqr[iqr == 0] = 1e-9
        
        return (X - self.median) / iqr

    def inverse_transform(self, X: Union[np.ndarray, pd.DataFrame]):
        X = self._validate_input(X)
        if self.median is None or self.q1 is None or self.q3 is None:
            raise ValueError("El escalador debe ser ajustado con fit() antes de invertir.")
        return X * (self.q3 - self.q1) + self.median