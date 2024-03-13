import torch.nn as nn
import torch
import math

class Squeeze(nn.Module):
    def __init__(self, dim):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(self.dim)

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super(Unsqueeze, self).__init__()
        self.dim = dim

    def forward(self, x):
        return x.unsqueeze(self.dim)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return x


class TransformerMemory(nn.Module):
    def __init__(self, transformer_decoder, memory_dim):
        super(TransformerMemory, self).__init__()
        self.transformer_decoder = transformer_decoder
        self.memory_dim = memory_dim
        self.memory = None  # Initialize memory

    def forward(self, tgt):
        # tgt: target sequence
        # If memory is not initialized, use zeros as initial memory
        if self.memory is None:
            batch_size = tgt.size(1)
            self.memory = torch.zeros(1, batch_size, self.memory_dim, device=tgt.device)
        
        # Forward pass through the transformer decoder
        output = self.transformer_decoder(tgt, self.memory)

        # Update memory for next step
        self.memory = output.clone()

        return output