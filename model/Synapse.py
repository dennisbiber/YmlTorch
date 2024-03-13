from torch import nn
from .Modules import Squeeze, Unsqueeze

class SynapseFactory:
    @staticmethod
    def Synapse_Droupout(config):
        dropout = config["dropout"]
        inplace = config["inplace"]
        dropout_synapse = nn.Dropout(p=dropout, inplace=inplace)
        return dropout_synapse
    
    @staticmethod
    def Synapse_Activation(activator, dim=1):
        activator = activator["activator"]
        if activator == "sigmoid":
            return nn.Sigmoid()
        elif activator == "relu":
            return nn.ReLU()
        elif activator == "tanh":
            return nn.Tanh()
        elif activator == "softmax":
            # Note: The dimension argument (`dim`) for the softmax activation function 
            # should be chosen based on the shape of your input tensor and the desired behavior. 
            # For example, in the case of a batch of probability distributions, 
            # `dim=1` is commonly used to apply softmax across the classes within each sample in the batch. 
            # However, the appropriate dimension may vary depending on the specific use case, 
            # so it's important to verify the correct dimension based on your data.
            return nn.Softmax(dim=dim)
        else:
            raise ValueError(f"Activation function '{activator}' not supported.")

    @staticmethod
    def Synapse_Dense(config):
        input_size = config["input_size"]
        output_size = config["output_size"]
        dense_layer = nn.Linear(input_size, output_size)
        return dense_layer

    @staticmethod
    def Synapse_Squeeze(config):
        # Apply squeeze operation to remove singleton dimensions
        dim = config["dim"]
        return Squeeze(dim)

    @staticmethod
    def Synapse_Unsqueeze(config):
        # Apply unsqueeze operation to add a singleton dimension
        dim = config["dim"]
        return Unsqueeze(dim)
