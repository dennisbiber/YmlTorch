from model.Network import NetworkBuilder
from model.Forward import Forward
import torchviz

class NetworkManager:
    def __init__(self, config):
        self._config = config["model"]
        self._order = config["forward"]
        self._builder = NetworkBuilder()
        self._forward = None
        
    def layer(self):
        for transmitter in self._config:
            conf = self._config[transmitter]
            layerType = conf["type"]
            self._builder.add_transmitter(transmitter, conf, layerType)
            
    def build_forward(self):
        if self._forward is None:
            module_dict = self._builder.fetch_layers()
            self._forward = Forward(module_dict, self._order["order"])
        else:
            print("Forward pass already built.")
            
    def forward(self, inputs):
        if self._forward is None:
            print("Forward pass not yet built. Please call build_forward() first.")
        else:
            return self._forward(inputs)
            
    def view_mnager(self):
        return self._builder.fetch_layers()
    
    def visualize_network(self, inputs):
        import torch
        if self._forward is None:
            print("Forward pass not yet built. Please call build_forward() first.")
        else:
            outputs = self.forward(inputs)
            print(type(outputs))
            torchviz.make_dot(outputs).render("network_graph", format="png")
            print("Network visualization saved as 'network_graph.png'")