import numpy as np
from allison.nn.layers import LayerDense

class NeuralNetwork:

    def __init__(self,loss_function,metric):
        self.layers: dict[int,LayerDense]={}
        self.n_layers = 0
        self.loss_function = loss_function
        self.metric = metric

    def add_layer(self,layer):
        self.n_layers += 1
        self.layers[self.n_layers] = layer 

    def forward(self,features):
        for key in self.layers.keys():
            features = self.layers[key].foward(features)
        return features

    def backward(self,activation,labels,features):

        DcDz = None

        DcDa = self.loss_function(labels,activation,True)
            
        for i,layer in reversed(self.layers.items()):
            if i == self.n_layers:
                Activation_l_1 = self.layers[i-1].activation
                DcDz = layer.backward_final_layer(DcDa,Activation_l_1)
            else:
                Weights_l = self.layers[i+1].weights
                if i == 1:
                    Activation_l_1 = features
                    layer.backward_first_layer(DcDz,Activation_l_1,Weights_l)
                else:
                    Activation_l_1 = self.layers[i-1].activation
                    DcDz = layer.backward(DcDz,Activation_l_1,Weights_l)
                    
    
    def predict(self,features)->np.array:
        out = self.forward(features)

        if self.loss_function.__name__ in ["binary_cross_entropy", "categorical_cross_entropy"]:
            if out.shape[1] == 1:
                return (out > 0.5).astype(int).squeeze()
            return (out == out.max(axis=1, keepdims=True)).astype(int)

        
        if self.loss_function.__name__ == "mean_squared_error":
            if out.shape[1] == 1:
                return out.squeeze()
            return out
    

    def evaluate(self,features,labels)->float:
        return self.metric(self.predict(features),labels)
    

    def summary(self)->None:
        total_layers = len(self.layers.keys())
        for i,layer in enumerate(self.layers.values()):
            layer.summary(i+1)
        total_neurons = np.sum([layer.n_neurones for layer in self.layers.values()])
        total_weights = np.sum([layer.n_features*layer.n_neurones for layer in self.layers.values()])
        print(f"Total -> Layers:{total_layers}, neurons:{total_neurons}, weights:{total_weights}, bias:{total_neurons} params:{total_neurons+total_weights} \n")
        print(f"Loss function: {self.loss_function.__name__} \n")
        print(f"Metric: {self.metric.__name__} \n")

    def __str__(self) -> str:
        layers = self.n_layers
        loss = self.loss_function.__name__
        metric = self.metric.__name__

        return f"NeuralNetwork(layers:{layers},loss:{loss},metric:{metric})"
    
    def __repr__(self) -> str:
        return self.__str__()