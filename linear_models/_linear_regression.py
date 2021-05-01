import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class LinearRegression:
  
  def __init__(self,type_regression='simple',functions=['lineal'],regularization=None):
    self.type_regression=type_regression
    self.functions=functions
    self.regularization=regularization
    self.weights=None

  def data_preproscessing_train(self, features, labels):
    if self.type_regression=='simple':
      self.weights=np.random.randn(2)
      features_train=features.to_numpy()

    if self.type_regression=='multiple':
      features_train=features.to_numpy()
      self.weights=np.random.randn(len(features_train[0])+1)

    return features_train, labels.to_numpy()

  def data_preproscessing_predict(self, features):
    features_train=features.to_numpy()
    return features_train


  def foward(self, features):
    if self.type_regression=='simple':
      predictions=self.weights[0] + features*self.weights[1] 

    if self.type_regression=='multiple':
      predictions=self.weights[0] + np.sum(features*self.weights[1:], axis=1)

    return predictions


  def decm_dw(self, predictions, labels, features):
    n=len(labels)
    gradients=np.zeros(len(self.weights))
    if self.type_regression=='simple':
      gradients[0]=(-2/n)*np.sum(labels-predictions)
      gradients[1]=(-2/n)*np.dot(labels-predictions,features)

    if self.type_regression=='multiple':
      gradients[0]=(-2/n)*np.sum(labels-predictions)
      for  i in range(1,len(self.weights)-1):
        gradients[i]=(-2/n)*np.dot(labels-predictions,features.T[i])
  
    return gradients


  def bacward(self,gradients,lr):
    self.weights=self.weights-lr*gradients
  

  def train(self,lr,n_iters, features, labels):
    features_train, labels_train=self.data_preproscessing_train(features, labels)
    for i in range(n_iters):
      predictions=self.foward(features_train)
      gradients=self.decm_dw(predictions,labels_train, features_train)
      self.bacward(gradients,lr)
      score=self.r2_score(labels_train,predictions)
      if (i+1)%2==0:
        print('Iter:\t{}\t{}\tr2_score:\t{:.2f}%'.format(i+1,50*'='+'>',score))

    if self.type_regression=='simple':
      plt.figure(figsize=(12,6))
      plt.scatter(features,labels)
      plt.plot(features,predictions,c='r', lw=3)
      plt.show()

  def r2_score(self, labels, predictions):
    errors=np.sum((labels-predictions)**2)
    varianza=labels.var()*len(labels)
    return (1-(errors/varianza))*100

  def predict (self,features):
    data_predict=self.data_preproscessing_predict(features)
    labels_predict=self.foward(data_predict)
    return labels_predict
  
  def save_weights(self,path):
    weights=pd.DataFrame(self.weights)
    weights.to_csv(path,index=False)
  
  def load_weights(self,path):
    data=pd.read_csv(path)
    weights_charge=data['0']
    self.weights=weights_charge.to_numpy()
