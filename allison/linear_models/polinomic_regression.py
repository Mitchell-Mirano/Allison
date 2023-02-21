import numpy as np

class PolinomicRegression():

  def __init__(self, regularization=None):
    self.weights=None
    self.bias=np.random.rand(1)
    self.n_grade=None
    self.regularization=None
    self.loss_function=None
    self.learning_ratio=None
    self.metrics=None

  def optimizers(self, n_grade,loss_function,lr,metrics,regularization=None):
    self.n_grade=n_grade
    self.loss_function=loss_function
    self.learning_ratio=lr
    self.metrics=metrics
    self.regularization=regularization

  def init_params(self,features,predict=False):

    if 'pandas' in str(type(features)):
      features=features.to_numpy()
    
    kernels=features

    if predict==False:
      if features.ndim==1:
        self.weights=np.random.rand(self.n_grade)
      else :
        self.weights=np.random.rand(len(features[0]))

    for i in range(2,self.n_grade+1):
      kernels=np.append(kernels,features**i)

    kernels=kernels.reshape(self.n_grade,len(features)).T

    return kernels

  def foward(self,features):
    if features.ndim==1:
      prediction=self.bias + features*self.weights
    else:
      prediction=self.bias + features@self.weights
  
    return prediction

  def  bacward(self,labels,predictions, features):

    gradient=self.loss_function(labels, predictions,derivative=True)

    if features.ndim==1:
      gradient_weights=gradient*np.mean(features)
    else:
      gradient_weights=gradient*np.mean(features, axis=0)

    self.bias =self.bias-self.learning_ratio*gradient
    self.weights= self.weights-self.learning_ratio*gradient_weights



  def train(self,n_iters,features, labels, callbacks_period=2):

    features=self.init_params(features)

    history_train={
        'iter':[],
        'loss':[],
        'r2_score':[]
    }

    for i in range(n_iters):
      predictions=self.foward(features)
      self.bacward(labels, predictions, features)

      if (i+1)%callbacks_period==0:
        score=self.metrics(labels,predictions)
        loss=self.loss_function(labels, predictions)
        history_train['loss'].append(loss)
        history_train['r2_score'].append(score)
        history_train['iter'].append(i+1)
        print('Iter:\t{}\t{}\t r2_score:\t{:.2f}% \n\n'.format(i+1,50*'='+'>',score))

    return history_train


  def predict (self,features):
    features=self.init_params(features, predict=True)
    predictions=self.foward(features)
    return predictions
  
  def save_weights(self,path):
    with open(path, 'wb') as f:
      np.save(f,self.bias)
      np.save(f,self.weights)

  
  def load_weights(self,path):
    with open(path, 'rb') as f:
      bias = np.load(f)
      weights = np.load(f)
    self.bias= bias
    self.weights=weights