
from utils.functions.activation import Sigmoid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.metrics import accuracy, predict_labels, recall,precision, r2_score
from utils.functions.loss import mean_square_error, binary_cross_entropy

from linear_models import LogisticRegression
from linear_models import LinearRegression

def numeric_labels(categorical_labels):
    
    labels=list(set(categorical_labels))
    numeric_labels=[]
    
    for cat_label in categorical_labels:
        if cat_label in labels:
            numeric_labels.append(labels.index(cat_label))
    
    return sorted(numeric_labels)

if __name__=='__main__':
   
   data=pd.read_csv('./data/iris.csv')
   data['labels']=numeric_labels(data['variety'])
   data=data.drop('variety', axis=1)
   data=data[data['labels']<2]
   labels=data.pop('labels')
   features=data.pop('sepal.length')

   model=LogisticRegression()
   model.optimizers(function_of_activation=Sigmoid,loss_function=binary_cross_entropy,lr=0.001,metrics=accuracy)
   model.train(n_iters=100,features=features,labels=labels, callbacks_period=10)

   plt.figure(figsize=(12,8))
   plt.scatter(features, labels, s=80, c=labels)
   plt.plot(features,model.foward(features), lw=3, c='red')
   plt.show()

