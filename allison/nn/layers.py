import numpy as np

class LayerDense:
    
    def __init__(self,n_neurones:int,n_features:int,activation_function:str):
        self.n_neurones = n_neurones
        self.n_features = n_features
        self.weights=np.random.randn(self.n_features,self.n_neurones)**2
        self.bias=np.random.randn(1,self.n_neurones)**2
        self.z:np.array = None
        self.activation_function = activation_function
        self.activation=None

    def foward(self,features):
        self.z = features@self.weights + self.bias
        self.activation  = self.activation_function(self.z)
        return self.activation


    def backward_final_layer(self,lr,Delta_l,activation,activation_name):

        if activation_name == "binary_cross_entropy":
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