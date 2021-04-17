# -*- coding: utf-8 -*-
import numpy as np 

class Kmeans:
    
    def __init__(self,features,n_centroides):
        self.features=features.to_numpy()
        self.n_centroides=n_centroides
        self.centroides_init=self.features[np.random.choice(len(self.features),self.n_centroides)]
        self.labels=np.zeros(len(self.features))
        self.centroides=np.empty([self.n_centroides,self.features.shape[1]])

    def distances(self,features, centroides):
      distances_features_to_centroides=[]
      for feature in features:
        distances=[]
        for centroide in centroides:
          distances.append(np.linalg.norm(feature-centroide))
        distances_features_to_centroides.append(distances)
      return np.array(distances_features_to_centroides)

    def update_values(self,new_array,old_array):
      for i in range(len(new_array)):
        old_array[i]=new_array[i]

    def new_centroides(self,labels,features):
      new_centroides=[]
      for label in set(labels):
        filter=np.where(labels==label)
        vectors=features[filter]
        media_vectors=vectors.sum(axis=0)/len(vectors)
        new_centroides.append(media_vectors)

      return new_centroides

    def train(self, error=0.01):
      movement_init=10000
      iter=0
      while movement_init>error:
        iter=iter+1
        if iter==1:
          distances=self.distances(self.features,self.centroides_init)
        else :
          distances=self.distances(self.features,self.centroides)

        new_labels=np.array([np.where(distance==np.min(distance))[0][0] for distance in distances])
        self.update_values(new_labels,self.labels)
        new_centroides=self.new_centroides(self.labels,self.features)

        if iter==1:
          movement=np.mean(np.array([np.linalg.norm(centroide-new_centroide) for centroide, new_centroide in zip(self.centroides_init,np.array(new_centroides))]))
        else:
          movement=np.mean(np.array([np.linalg.norm(centroide-new_centroide) for centroide, new_centroide in zip(self.centroides,np.array(new_centroides))]))
        
        self.update_values(new_centroides,self.centroides)

        movement_init=movement


 