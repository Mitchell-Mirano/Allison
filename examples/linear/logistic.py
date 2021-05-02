
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


from ...linear_models import LogisticRegression

if __name__=='__main__':
    data=pd.read_csv('./data/Iris.csv')
    print(data.head(10))