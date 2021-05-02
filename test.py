import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from clustering import Kmeans
from linear_models import LogisticRegression
from linear_models import LinearRegression


if __name__=='__main__':

    data=pd.read_csv('./data/Iris.csv')
    df=data[data['labels']<2]
    features=df[['sepal width (cm)',  'petal length (cm)']]
    labels=df['labels']

    model=LogisticRegression('multiple')

    model.train(0.01,100, features, labels)

    plt.scatter(features['sepal width (cm)'], features['petal length (cm)'],c=model.predict(features))
    plt.show()