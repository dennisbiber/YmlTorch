import torch
import torch.nn as nn
from .Transmitter import TransmitterFactory
from .Synapse import SynapseFactory


class NetworkBuilder(nn.Module):
    def __init__(self):
        super(NetworkBuilder, self).__init__()
        # Dictionary to store different types of input processing mechanisms
        self.transmitter_dict = nn.ModuleDict()
        self.empty_layers_sequence()
        
    def forward(self, x, input_type):
        # Retrieve the appropriate transmitter based on input type
        transmitter = self.transmitter_dict[input_type]
        
        # Forward pass through the selected transmitter
        return transmitter(x)
    
    def empty_layers_sequence(self):
        self.sequence = []
    
    def build_sequence(self, config, layer):
        if layer.lower() == "conv":
            seq = TransmitterFactory.CNN(config)
        elif layer.lower() == "pool":
            seq = TransmitterFactory.Pooling(config)
        elif layer.lower() == "activator":
            seq = SynapseFactory.Synapse_Activation(config)
        self.sequence.append(seq)
    
    def add_transmitter(self, name, params, predefined=None, mockData=False):
        # Add a new transmitter to the dictionary
        if predefined == "LSTM":
            layer = TransmitterFactory.LSTM(params)
        elif predefined == "Embedding":
            layer = TransmitterFactory.Embedding(params)
        elif predefined == "GRU":
            layer = TransmitterFactory.GRU(params)
        elif predefined == "Upsample":
            layer = TransmitterFactory.Upsample(params)
        elif predefined == "Linear":
            layer = TransmitterFactory.Linear(params)
        elif predefined == "Cosine_sim":
            layer = TransmitterFactory.Cosine_Sim(params)
        elif predefined == "Positional_encode":
            layer = TransmitterFactory.Positional_encoder(params)
        elif predefined == "TransformerDecoder":
            decoder = TransmitterFactory.TransformerDecoder(params)
            layer = TransmitterFactory.DecoderMemory(params, decoder)
        elif predefined   == "Dropout":
            layer = SynapseFactory.Synapse_Droupout(params)
        elif predefined == "Dense":
            layer = SynapseFactory.Synapse_Dense(params)
        elif predefined == "Activator":
            layer = SynapseFactory.Synapse_Activation(params)
        elif predefined == "Squeeze":
            layer = SynapseFactory.Synapse_Squeeze(params)
        elif predefined == "Sequence":
            params = params["params"]
            for param, idx in zip(params, range(len(params))):
                self.build_sequence(param[idx], param[idx]["type"])
            layer = TransmitterFactory.Sequencer(self.sequence)
            self.empty_layers_sequence()
        self.transmitter_dict[name] = layer
            
    def fetch_layers(self):
        return self.transmitter_dict

