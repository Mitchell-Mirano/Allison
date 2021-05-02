class Activation():

  def Sigmoid(feature,derivative=False):
    if derivative==False:
      return 1/(1+np.exp(-feature))
    else :
      return np.exp(-feature)/((1+ np.exp(-feature)))**2

  def ReLu(feature,derivative=False):
    if derivative==False:
      return feature*(feature>0)
    else :
      return 1*(feature>0)

  def Tanh(feature,derivative=False):
    if derivative==False:
      return np.tanh(feature)
    else :
      return 1/(np.cosh(feature))**2


class Loss():

  def mean_square_error(prediction, labels, features):
    pass  

  def binary_cross_entropy(predictions, labels, features):
    pass
    
  def categorical_cross_entropy(predictions, labels, features):
    pass

