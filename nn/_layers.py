import numpy as np

class LayerDense():

  def __init__(self,n_neurons,function_of_activation,imput_shape):
    self.n_neurons=n_neurons
    self.function_of_activation=function_of_activation
    self.imput_shape=imput_shape
    self.bias=np.random.randn(n_neurons,1)
    self.weights=np.random.randn(self.n_neurons,self.imput_shape[0])
    self.linear_convination=None
    self.activation=None

  def foward_layer_dense(self,features):
    if self.imput_shape[0]==1:
      self.linear_convination=self.bias + self.weights*features
    else:
      self.linear_convination=self.bias + self.weights@features
    self.activation=self.function_of_activation(self.linear_convination)
    return self.activation

  def bacward_layer_dense(self, learning_ratio, gradient_next_layer,weights_next_layer=0,activation_previous_layer=0):
    if type(weights_next_layer)==int and type(activation_previous_layer)==int:
      this_layer_gradient=np.mean(gradient_next_layer*self.function_of_activation(self.linear_convination, derivative=True))
      self.bias = self.bias -learning_ratio*this_layer_gradient
      self.weights=self.weights-learning_ratio*this_layer_gradient*np.mean(activation_previous_layer)
      return this_layer_gradient
    else :
      this_layer_gradient=np.mean(gradient_next_layer*weights_next_layer@self.function_of_activation(self.linear_convination, derivative=True))
      self.bias = self.bias -learning_ratio*this_layer_gradient
      self.weights=self.weights-learning_ratio*this_layer_gradient*np.mean(activation_previous_layer)
      return this_layer_gradient