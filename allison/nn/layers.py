import numpy as np

class LayerDense:
    
    def __init__(self,n_features:int,n_neurones:int,activation_function:str):
        self.n_features = n_features
        self.n_neurones = n_neurones
        self.activation_function = activation_function

        if self.activation_function.__name__ in ['tanh','sigmoid']:
            self.std_dev = np.sqrt(2.0 / (n_features + n_neurones))
        elif self.activation_function.__name__ in ['relu']:
            self.std_dev = np.sqrt(2.0 / n_features)
        else:
            self.std_dev = np.sqrt(2.0 / (n_features + n_neurones))

        self.weights=np.random.normal(loc=0,scale=self.std_dev,size=(n_features,n_neurones))
        self.bias=np.random.normal(loc=0,scale=self.std_dev,size=(1,n_neurones))
        self.z:np.array = None
        self.activation=None

    def foward(self,features):
        self.z = features@self.weights + self.bias
        self.activation  = self.activation_function(self.z)
        return self.activation


    def backward_final_layer(self,lr,Delta_l,activation,activation_name):

        if activation_name in ["binary_cross_entropy", "mean_squared_error"]:
            Delta_l = Delta_l*self.activation_function(self.z,True)
            
        if activation_name == "categorical_cross_entropy":
            pass
        
        DcDw=activation.T@Delta_l
        DcDb=Delta_l.sum(axis=0)
    
        self.weights = self.weights - lr*DcDw
        self.bias = self.bias - lr*DcDb

        return Delta_l

    def backward(self,lr,Delta_l,Activation_l_1,Weights_l):

        Delta_l = Delta_l@Weights_l.T*self.activation_function(self.z,True)

        DcDw=Activation_l_1.T@Delta_l
        DcDb=Delta_l.sum(axis=0)

        self.weights = self.weights - lr*DcDw
        self.bias = self.bias - lr*DcDb

        return Delta_l

    def backward_first_layer(self,lr,Delta_l,Activation_l_1,Weights_l):

        Delta_l = Delta_l@Weights_l.T*self.activation_function(self.z,True)
    
        DcDw=Activation_l_1.T@Delta_l
        DcDb=Delta_l.sum(axis=0)

        self.weights = self.weights - lr*DcDw
        self.bias = self.bias - lr*DcDb

    def summary(self,layer_index):
        weights = f"({self.n_features},{self.n_neurones})"
        activation_name = self.activation_function.__name__
        print(f"Layer:{layer_index}, neurons:{self.n_neurones}, input:(n,{self.n_features}), weights:{weights}, output:(n,{self.n_neurones}), activation:{activation_name} \n")

    def __str__(self) -> str:
        return f"LayerDense(neurones={self.n_neurones},activation={self.activation_function.__name__})"
    
    def __repr__(self) -> str:
        return self.__str__()