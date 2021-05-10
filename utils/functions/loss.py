import numpy as np

def mean_square_error(labels, predictions,derivative=False):
  if derivative==False:
    mse=np.mean((labels-predictions)**2)
    return mse
  else:
    dmse=-2*np.mean(labels-predictions)
    return dmse

def binary_cross_entropy(labels,predictions, derivative=False):
  if derivative==False:
    bce=-np.sum(labels*np.log(predictions) +
    return bce (1-labels)*np.log(1-predictions))
  else:
    dbce=-np.sum((labels-predictions)/(predictions*(1-predictions)))
    return dbce

def categorical_cross_entropy(labels, predictions):
  pass

