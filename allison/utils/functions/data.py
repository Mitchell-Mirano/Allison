import pandas as pd
import numpy as np

def train_test_split(X: pd.DataFrame,
                     y: pd.Series,
                     test_size: float = 0.2,
                     random_state: int = 42,
                     shuffle: bool = True) -> tuple:
    """
    Method to split the data into train and test sets.

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Labels.
        test_size (float, optional): Proportion of the dataset to include in the test split. Default is 0.2.
        random_state (int, optional): Random state for reproducibility. Default is 42.
        shuffle (bool, optional): Whether to shuffle the data before splitting. Default is True.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    
    # Validate test_size
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1.")

    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Shuffle the data if specified
    if shuffle:
        indices = np.random.permutation(len(X))
        X = X.iloc[indices]
        y = y.iloc[indices]

    # Calculate the split index
    split_index = int(len(X) * (1 - test_size))

    # Split the data
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]

    return X_train, X_test, y_train, y_test
