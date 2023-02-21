class NeuralNetwork ():
  def __init__(self):
    self.layers=[]
    self.n_layers=0
    self.learning_ratio=None
    self.loss_function=None
    self.metric=None

  def add_layer(self,layer):
    self.n_layers=self.n_layers +1
    self.layers.append(layer)

  def optimizers(self,learning_ratio,loss_function,metric):
    self.learning_ratio=learning_ratio
    self.loss_function=loss_function
    self.metric=metric


  def foward(self, features):
    for layer in self.layers:
      features=layer.foward_layer_dense(features)
    return features

  def bacward(self,predictions, labels, features):

    index_layers=len(self.layers)-1
    for i, layer in enumerate(reversed(self.layers)):
      if i==0:
        gradient_final_layer=self.loss_function(predictions,labels, derivative=True)
        activation_previous_layer=self.layers[index_layers-1-i].activation
        gradient_next_layer=layer.bacward_layer_dense(self.learning_ratio,gradient_final_layer)
      else:
        weigths_next_layer=self.layers[index_layers+1-i].weights
        if i==index_layers:
          activation_previous_layer=features
        activation_previous_layer=self.layers[index_layers-i-1].activation
        gradient_next_layer=layer.bacward_layer_dense(self.learning_ratio,gradient_next_layer,weigths_next_layer,activation_previous_layer)




  def train(self,n_iters,features, labels, explicit=True):

    history_train={
        'loss':[]
    }
      
    for i in range(n_iters):
      labels_pred=self.foward(features)
      loss=self.loss_function(labels,labels_pred)
      self.bacward(labels_pred, labels, features)
      history_train['loss'].append(loss)
      if explicit==True and i%100==0:
        loss=self.loss_function(labels,labels_pred)
        history_train['loss'].append(loss)
        r2_score=self.metric(labels,labels_pred)
        print('Iter {}: \t {} \t Loss: {:.2f} \t r2_score: {:.2f}%'.format(i,50*'='+'>',loss,r2_score))
    return history_train


  def summary(self):
    total_params=0
    print(50*'=')
    print('layer \t neurons \t weights')
    layers_count=0
    neurons_count=0
    for layer in self.layers:
      layers_count=layers_count +1
      neurons, weights, =layer.weights.shape[0],layer.weights.shape[1]
      neurons_count=neurons_count + neurons
      total_weights=neurons*weights + neurons
      total_params=total_params + total_weights
      print('  {}\t   {}\t           {}'.format(layers_count,layer.n_neurons,total_weights))
    print('Total Layers: \t',layers_count)
    print('Total Neurons:\t',neurons_count)
    print('Total Weights: \t',total_params)
    print(50*'=')