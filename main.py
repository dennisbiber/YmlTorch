from manager import NetworkManager
from mock_data import generate_mock_input
import argparse
import yaml


def Parser():
    parser = argparse.ArgumentParser(description="Load text data from a file")
    parser.add_argument("config", type=str, help="Path to the input text file")
    args = parser.parse_args()
    return args


def ConfigLoader(path):
    with open(path, "rb") as file:
        config = yaml.safe_load(file)
    return config


def main(args):
    config = ConfigLoader(args.config)
    manager = NetworkManager(config)
    manager.layer()
    print(manager.view_mnager())
    mock = generate_mock_input(config)
    manager.build_forward()
    manager.visualize_network(mock)
    
    
    
if __name__ == "__main__":
    args = Parser()
    main(args)