import pandas as pd
import numpy as np

def train_test_split(df:pd.DataFrame,p_train:float=0.8):
    strainer = np.random.rand(len(df))<p_train
    train= df[strainer]
    test = df[~strainer]
    return train, test