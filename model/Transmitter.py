from torch import nn
from .Modules import PositionalEncoding, TransformerMemory

class TransmitterFactory:
    @staticmethod
    def Sequencer(layers: list):
        sequence = nn.Sequential(
            *layers
        )
        return sequence
    
    @staticmethod
    def CNN(config):
        num_inputs = config["num_inputs"]
        out_channels = config["out_channels"]
        kernel_size = config["kernel_size"]
        stride = config["stride"]
        padding = config["padding"]
        conv = nn.Conv2d(in_channels=num_inputs, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        return conv
    
    @staticmethod
    def Pooling(config):
        kernel_size = config["kernel_size"]
        stride = config["stride"]
        pool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        return pool
    
    @staticmethod
    def LSTM(config):
        embedding_dim = config["embedding_dim"]
        hidden_dim = config["hidden_dim"]
        batch_first = config["batch_first"] # bool
        lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=batch_first)
        return lstm
    
    @staticmethod
    def Embedding(config):
        num_embeddings = config["num_embeddings"] # int such as vocab size
        embedding_dim = config["embedding_dim"] # int
        scale_grad_by_freq = config["scale_grad_by_freq"] # bool
        sparse = config["sparse"] # bool
        embedding = nn.Embedding(num_embeddings, embedding_dim, scale_grad_by_freq=scale_grad_by_freq, sparse=sparse)
        return embedding
    
    @staticmethod
    def GRU(config):
        input_size = config["input_size"] # int
        hidden_size = config["hidden_size"] # int
        num_layers = config["num_layers"] # int
        batch_first = config["batch_first"] # Bool
        gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=batch_first)
        return gru
    
    @staticmethod
    def TransformerDecoder(config):
        embedding_dim = config["embedding_dim"]
        num_heads = config["num_heads"]
        num_layers = config["num_layers"]
        decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=num_heads)
        decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        return decoder
    
    @staticmethod
    def DecoderMemory(config, decoder):
        memory_dim = config["memory_dim"]
        memory = TransformerMemory(decoder, memory_dim)
        return memory
    
    @staticmethod
    def TransformerEncoder(config):
        embedding_dim = config["embedding_dim"]
        num_heads = config["num_heads"]
        num_layers = config["num_layers"]
        decoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads)
        decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        return decoder
    
    @staticmethod
    def Linear(config):
        inFeats = config["input_features"]
        outFeats = config["output_features"]
        bias = config["bias"] # bool
        output = nn.Linear(inFeats, outFeats, bias=bias)
        return output
    
    @staticmethod
    def Upsample(config):
        scale_factor = config["scale_factor"]
        mode = config["mode"]
        align_corners = config["align_corners"]
        upsample = nn.Upsample(scale_factor=scale_factor, mode=mode, align_corners=align_corners)
        return upsample
    
    @staticmethod
    def Cosine_Sim(config):
        dim = config["dim"]
        epsilon = config["epsilon"]
        sim = nn.CosineSimilarity(dim=dim, eps=epsilon)
        return sim
    
    @staticmethod
    def Positional_encoder(config):
        d_model = config["d_model"]
        max_len = config["max_len"]
        encoder = PositionalEncoding(d_model, max_len=max_len)
        return encoder