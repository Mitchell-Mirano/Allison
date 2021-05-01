# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class LogisticRegression():

    def __init__(self,type_regression='simple', regularization=None):
        self.type_regression=type_regression
        self.regularization=regularization
        self.weights=None
        self.labels=None

    def data_preproscessing_train(self, features, labels):
        np.random.seed(100)
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

            predictions=1/(1+np.exp(-predictions))

        if self.type_regression=='multiple':
            predictions=self.weights[0] + np.sum(features*self.weights[1:], axis=1)
            predictions=1/(1+np.exp(-predictions))
        
        return predictions
    
    def dbce_dw(self, predictions, labels, features):
        n=len(labels)
        gradients=np.zeros(len(self.weights))
        if self.type_regression=='simple':
            gradients[0]=(-1/n)*np.sum(labels-predictions)
            gradients[1]=(-1/n)*np.dot(labels-predictions,features)

        if self.type_regression=='multiple':
            gradients[0]=(-1/n)*np.sum(labels-predictions)
            for  i in range(1,len(self.weights)-1):
                gradients[i]=(-1/n)*np.dot(labels-predictions,features.T[i])
    
        return gradients

    def bacward(self,gradients,lr):
        self.weights=self.weights-lr*gradients


    def train(self,lr,n_iters, features, labels):

        features_train, labels_train=self.data_preproscessing_train(features, labels)

        for i in range(n_iters):
            predictions=self.foward(features_train)
            gradients=self.dbce_dw(predictions,labels_train, features_train)
            self.bacward(gradients,lr)
            accuracy=self.accuracy(labels_train,predictions)
            if (i+1)%5==0:
                print('Iter:\t{}\t{}\t Accuracy: \t{:.2f}%'.format(i+1,50*'='+'>',accuracy))
                
        if self.type_regression=='simple':
            plt.figure(figsize=(12,6))
            plt.scatter(features,labels)
            plt.plot(features_train,predictions, c='red', lw=3)
            plt.show()
    
    def accuracy(self, labels, predictions):
        labels_predicted=[]
        for label in predictions:
            if label<=0.5:
                labels_predicted.append(0)
            else:
                labels_predicted.append(1)
        
        count_correct_prediction=0

        for label_true, label_pred in zip(labels, labels_predicted):
            if label_pred==label_true:
                count_correct_prediction=count_correct_prediction +1 
            else:
                count_correct_prediction=count_correct_prediction +0

        return (count_correct_prediction/len(predictions))*100

    

    def predict (self,features):
        data_predict=self.data_preproscessing_predict(features)
        labels_predict=self.foward(data_predict)
        predictions=[]
        for pred in labels_predict:
            if pred<=0.5:
                predictions.append(0)
            else:
                predictions.append(1)
        return np.array(predictions)
    
    def save_weights(self,path):
        weights=pd.DataFrame(self.weights)
        weights.to_csv(path,index=False)
    
    def load_weights(self,path):
        data=pd.read_csv(path)
        weights_charge=data['0']
        self.weights=weights_charge.to_numpy()
        
        
    
