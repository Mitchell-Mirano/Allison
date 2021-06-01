
from utils.functions.activation import Sigmoid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.metrics import accuracy, predict_labels, recall,precision, r2_score
from utils.functions.loss import mean_square_error, binary_cross_entropy

from linear_models import LogisticRegression

def numeric_labels(categorical_labels):
    
    labels=list(set(categorical_labels))
    numeric_labels=[]
    
    for cat_label in categorical_labels:
        if cat_label in labels:
            numeric_labels.append(labels.index(cat_label))
    
    return sorted(numeric_labels)

if __name__=='__main__':
   
   x=np.linspace(-10,10,100)
   y=np.sin(x)

   plt.plot(x,y, lw=3)
   plt.show()