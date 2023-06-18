import numpy as np

def sigmoid(sumatory,derivative=False):
    if derivative==False:
        return 1/(1+np.exp(-sumatory))
    else :
        return np.exp(-sumatory)/((1+ np.exp(-sumatory)))**2

def softmax(predictions:np.array):
    predictions = np.exp(predictions)
    return predictions/np.sum(predictions,axis=1,keepdims=True)

def relu(feature,derivative=False):
  if derivative==False:
    return feature*(feature>0)
  else :
    return 1*(feature>0)

def tanh(feature,derivative=False):
  if derivative==False:
    return np.tanh(feature)
  else :
    return 1/(np.cosh(feature))**2