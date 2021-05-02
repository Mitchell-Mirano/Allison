class LayerDense():

  def __init__(self,n_neurons,function_of_activation,imput_shape):
    self.n_neurons=n_neurons
    self.function_of_activation=function_of_activation
    self.imput_shape=imput_shape
    self.weigths=np.random.randn(self.n_neurons,self.imput_shape[0]+1)

  def neuron(self, features, weights):
    linear_covination=weights[0] + np.matmul(weights[1:],features)
    predict=self.function_of_activation(linear_covination)
    return predict


  def foward_layer_dense(self,features):
    predictions=None
    for i in range(self.n_neurons):
      predict=self.neuron(features, self.weights[i])
      if predictions is None:
        predictions=predict
      else :
        predictions=np.append(prediction, predict)
    
    return predictions.reshape(self.n_neurons,self.imput_shape[1])

  def gradients_layer_dense(self,gradientes_next_layer):
    pass

  def bacward_layer_dense(self,learning_ratio, gradientes_next_layer):
    gradients=self.gradients_layer_dense(gradientes_next_layer)
    self.weights=self.weights-learning_ratio*gradients