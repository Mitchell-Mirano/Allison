import numpy as np
from allison.nn.layers import LayerDense

class NeuralNetwork:

    def __init__(self,loss_function,metric,learning_rate):
        self.layers: dict[int,LayerDense]={}
        self.n_layers = 0
        self.loss_function = loss_function
        self.learning_rate = learning_rate
        self.metric = metric

    def add_layer(self,layer):
        self.n_layers += 1
        self.layers[self.n_layers] = layer 

    def foward(self,features):
        for key in self.layers.keys():
            features = self.layers[key].foward(features)
        return features

    def bacward(self,activation,labels,features):

        loss_function_name = self.loss_function.__name__

        DcDz = None

        if loss_function_name == "categorical_cross_entropy":
            DcDa = activation - labels
            
        if loss_function_name == "binary_cross_entropy":
            DcDa = self.loss_function(labels,activation,True)
            
        for i,layer in reversed(self.layers.items()):
            if i == self.n_layers:
                Activation_l_1 = self.layers[i-1].activation
                DcDz = layer.backward_final_layer(self.learning_rate,DcDa,Activation_l_1,loss_function_name)
            else:
                Weights_l = self.layers[i+1].weights
                if i == 1:
                    Activation_l_1 = features
                    layer.backward_first_layer(self.learning_rate,DcDz,Activation_l_1,Weights_l)
                else:
                    Activation_l_1 = self.layers[i-1].activation
                    DcDz = layer.backward(self.learning_rate,DcDz,Activation_l_1,Weights_l)
                    
    def train(self,features, labels,iters,verbose=True)->None:
        steps = int(iters/10)
        for i  in range(1,iters+1):
            activation = self.foward(features)
            self.bacward(activation,labels,features)
            if  i%steps == 0 and verbose:
                error=self.loss_function(labels,activation)
                accuracy=self.metric(activation,labels)
                print(f"Iter:{i:.2f} \t Error:{error:.6f} \t Accuracy:{accuracy:.6f}%")
    
    def predict(self,features)->np.array:
        predictions=self.foward(features)
        condition=predictions==np.max(predictions, axis = 1,keepdims=True)
        labels = condition.astype(int)
        return labels
    

    def evaluate(self,features,labels)->float:
        predictions = self.foward(features)
        return self.metric(predictions,labels)
    

    def summary(self)->None:
        total_layers = len(self.layers.keys())
        for i,layer in enumerate(self.layers.values()):
            layer.summary(i+1)
        total_neurons = np.sum([layer.n_neurones for layer in self.layers.values()])
        total_weights = np.sum([layer.n_features*layer.n_neurones for layer in self.layers.values()])
        print(f"Total -> Layers:{total_layers}, neurons:{total_neurons}, weights:{total_weights}, bias:{total_neurons} params:{total_neurons+total_weights} \n")
        print(f"Loss function: {self.loss_function.__name__} \n")
        print(f"Metric: {self.metric.__name__} \n")
        print(f"Learning Rate: {self.learning_rate} \n")

    def __str__(self) -> str:
        layers = self.n_layers
        loss = self.loss_function.__name__
        metric = self.metric.__name__
        lr = self.learning_rate
        return f"NeuralNetwork(layers:{layers},loss:{loss},metric:{metric},lr:{lr})"
    
    def __repr__(self) -> str:
        return self.__str__()