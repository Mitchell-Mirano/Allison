import numpy as np

def mean_squared_error(labels, predictions,derivative=False):
  if derivative==False:
    mse=np.mean((labels-predictions)**2)
    return mse
  else:
    dmse=-2*np.mean(labels-predictions)
    return dmse

def binary_cross_entropy(labels,predictions, derivative=False):
  if derivative==False:
    bce=-np.sum(labels*np.log(predictions) + (1-labels)*np.log(1-predictions))
    return bce 
  else:
    dbce=-np.sum(labels/predictions + (labels-1)/(1-predictions))
    return dbce

def categorical_cross_entropy(labels:np.array,predictions:np.array, derivative=False):
    
    if derivative==False:
      return 1
    else:
      return predictions-labels 


