import numpy as np


def linear(feature,derivative=False):
  if derivative==False:
    return feature
  else :
    return np.ones(feature.shape)


def sigmoid(sumatory,derivative=False):
    salida = 1/(1+np.exp(-sumatory))
    if derivative==False:
        return salida
    else :
        return salida*(1-salida)

def softmax(predictions:np.array, derivative=False):
    if derivative==False:
      predictions = np.exp(predictions)
      return predictions/np.sum(predictions,axis=1,keepdims=True)
    else:
      return 1

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