import torch

def generate_mock_input(config, batch_size=16):
    # Extract the number of inputs needed from the configuration
    num_inputs = config["forward"]["inputs"]
    
    # Extract the params of the first layer in the forward pass sequence
    layer1 = config["forward"]["order"][0]
    if type(layer1) == dict:
        if "sequence" in layer1.keys():
            first_layer_params = config["model"][config["forward"]["order"][0]["sequence"][0]]
    else:
        first_layer_params = config["model"][config["forward"]["order"][0]]
    
    # Initialize a list to store mock inputs
    mock_inputs = []
    
    # Generate mock inputs based on the type of the first layer
    if first_layer_params["type"] == "Embedding":
        # Generate random indices for embedding
        for _ in range(num_inputs):
            mock_input = torch.randint(0, first_layer_params["num_embeddings"], (batch_size, first_layer_params["embedding_dim"]))
            mock_inputs.append(mock_input)
    elif first_layer_params["type"] == "LSTM":
        # Generate random input tensors for LSTM
        for _ in range(num_inputs):
            mock_input = torch.randn(batch_size, 1, first_layer_params["embedding_dim"], first_layer_params["hidden_dim"])
            mock_inputs.append(mock_input)
    return mock_inputs