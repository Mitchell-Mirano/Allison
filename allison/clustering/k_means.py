# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


class Kmeans:

  def __init__(self):
    self.centroids=None
    self.history_train={}
    self.labels=None

  def data_preprocessing_train(self,features,n_centroids, history_train):

    features_train=features.to_numpy()
    self.centroids=features_train[np.random.randint(0, len(features_train),n_centroids )]
    if history_train==True:
      self.history_train[0]=self.centroids
    return features_train


  def distances(self,features, centroids):

    distances_features_centroides=[]
    for feature in features:
      distances_feature=[]
      for centroid in centroids:
        distances_feature.append(np.linalg.norm(centroid-feature))
      distances_features_centroides.append(distances_feature)
    return np.array(distances_features_centroides)


  def new_labels(self,distances):

    new_labels=np.array([np.where(distance==np.min(distance))[0][0] for distance in distances])
    return new_labels


  def new_centroids(self, features, labels):

    new_centroids=[]
    for label in set(labels):
      filter=np.where(labels==label)
      vectors=features[filter]
      media_vectors=vectors.sum(axis=0)/len(vectors)
      new_centroids.append(media_vectors)
    return np.array(new_centroids)
  

  def moviment(self,centroids_beafore, centroids_after):

    moviments=[]
    for centroid_bf, centroid_af in zip(centroids_beafore,centroids_after):
      moviments.append(np.linalg.norm(centroid_bf-centroid_af))
    return np.mean(moviments)


  def train(self,features,n_centroids,moviment_limit=0.1, history_train=True):

    features_train=self.data_preprocessing_train(features,n_centroids, history_train)
    iter=0
    while True:
      iter=iter+1
      distances=self.distances(features_train,self.centroids)
      self.labels=self.new_labels(distances)
      centroids_before=self.centroids
      self.centroids=self.new_centroids(features_train, self.labels)
      if history_train==True:
        self.history_train[iter]=self.centroids
      moviment=self.moviment(centroids_before,self.centroids)
      print('Iter: {} \t {} \t moviment: {:.2f}'.format(iter,50*'='+'>',moviment))
      if moviment< moviment_limit:
        break

  def save_labels(self,path):
    weights=pd.DataFrame(self.labels)
    weights.to_csv(path,index=False)

 