import torch
import torch.nn as nn
import torch.nn.functional as F

class Forward(nn.Module):
    def __init__(self, module_dict, order):
        super(Forward, self).__init__()
        self.module_dict = module_dict
        self.order = order
        
    def forward(self, inputs):
        outputs = {}
        if isinstance(inputs, list):  # Check if inputs is a list
            if "sequence" in self.order[0]:  # Check if the first task is a sequence
                inputs, outputs = self.handle_sequence(inputs, outputs)
        for task in self.order:
            if isinstance(task, dict):
                if "sequence" in task:
                    continue  # Already handled in the previous loop
            else:
                print(task)
                module = self.module_dict[task]
                outputs[task] = module(*inputs)
                inputs = [outputs[task]]
        return inputs[0]
    
    def handle_sequence(self, inputs, outputs):
        sequence = self.order[0]["sequence"]
        for i, input_data in enumerate(inputs):
            sequence_inputs = input_data
            for seg in sequence:
                module = self.module_dict[seg]
                if "lstm" in seg.lower():
                    sequence_outputs = self.lstm_return(module, sequence_inputs, return_type="hidden_state")
                else:
                    sequence_outputs = module(sequence_inputs)
                outputs[seg + "_input_" + str(i)] = sequence_outputs
                sequence_inputs = sequence_outputs
            inputs[i] = sequence_inputs
        return inputs, outputs

    def lstm_return(self, layer, input, return_type=None):
        if return_type == "hidden_state":
            _, (h_n, _) = layer(input)
            return h_n
        else:
            output, (h_n, c_n) = layer(input)
            return output