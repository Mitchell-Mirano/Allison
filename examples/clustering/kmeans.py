import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ...clustering import Kmeans

if __name__=='__main__':

    data=pd.read_csv('./data/Iris.csv')
    model=Kmeans()
    model.train(data[['sepal length (cm)','sepal width (cm)']],4,0.1,False)

    plt.scatter(data['sepal length (cm)'],data['sepal width (cm)'], s=50, c=model.labels)
    for centroid in model.centroids:
        plt.scatter(centroid[0],centroid[1],s=150)
    plt.show()