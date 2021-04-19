import numpy as np
import pandas as pd

class LinearRegresion:
    
  def __init__(self,features,labels):
      self.features=np.append([np.ones(features.to_numpy().T.shape[1])], features.to_numpy().T, axis=0)
      self.weights=np.random.randn(len(self.features))
      self.labels=labels.to_numpy().T

  def foward(self):
    Linear_convination=np.array([self.weights[i]*self.features[i] for i in range(len(self.weights))])
    predictions=np.sum(Linear_convination,axis=0)
    return predictions

  def decm_dw(self, predictions):
    gradients=np.zeros(len(self.weights))
    for  i in range(len(self.weights)):
      gradients[i]=(-2/self.features.shape[1])*np.dot(self.labels-predictions,self.features[i])
    return gradients
  
  def bacward(self,gradients,lr):
    new_weigths=self.weights-lr*gradients
    self.weights=new_weigths

  def train(self,lr,n_iters):
    for i in range(n_iters):
      predictions=self.foward()
      gradients=self.decm_dw(predictions)
      self.bacward(gradients,lr)

  def r2_score(self):
    errors=np.sum((self.labels-self.foward())**2)
    varianza=self.labels.var()*self.labels.shape[1]
    return 1-(errors/varianza)

  def predict (self,features):
    data=np.append([np.ones(features.to_numpy().T.shape[1])], features.to_numpy().T, axis=0)
    Linear_convination=np.array([self.weights[i]*data[i] for i in range(len(self.weights))])
    predictions=np.sum(Linear_convination,axis=0)
    return predictions
  
  def save_weights(self,path):
    weights=pd.DataFrame(self.weights)
    weights.to_csv(path,index=False)
    