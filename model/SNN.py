from torch import nn
import torch

class SpikingNeuronLayer(nn.Module):
    def __init__(self, num_neurons, num_inputs, 
                 tau_mem=10.0, threshold=1.0, reset=0.0, 
                 init_synaptic_weight=None, synaptic_decay=None, 
                 connectivity='dense', learning_rule=None):
        super(SpikingNeuronLayer, self).__init__()
        
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        
        # Membrane potential parameters
        self.tau_mem = tau_mem
        self.membrane_potential = torch.zeros(num_neurons)
        
        # Spike generation parameters
        self.threshold = threshold
        self.reset = reset
        self.spike = torch.zeros(num_neurons)
        
        # Synaptic parameters
        self.connectivity = connectivity
        if init_synaptic_weight is None:
            self.synaptic_weight = nn.Parameter(torch.randn(num_neurons, num_inputs))
        else:
            self.synaptic_weight = nn.Parameter(init_synaptic_weight.clone().detach())
        if synaptic_decay is None:
            self.synaptic_decay = nn.Parameter(torch.ones(num_neurons, num_inputs))
        else:
            self.synaptic_decay = nn.Parameter(synaptic_decay.clone().detach())
        
        # Learning parameters
        self.learning_rule = learning_rule
    
    def forward(self, x):
        # Membrane potential dynamics (leaky integration)
        self.membrane_potential = torch.exp(-1.0 / self.tau_mem) * self.membrane_potential + torch.mm(self.synaptic_weight, x)
        
        # Spike generation (reset if threshold reached)
        self.spike = torch.zeros(self.num_neurons)
        self.spike[self.membrane_potential >= self.threshold] = 1.0
        self.membrane_potential[self.membrane_potential >= self.threshold] = self.reset
        
        # Update synaptic weights based on learning rule
        if self.learning_rule:
            self.learning_rule(self)
        
        return self.spike