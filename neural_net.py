import numpy as np

class NeuralNetwork: 
    def __init__(self, input_size, hidden_size, output_size):
        #Layer 1 weights and biases 
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        #layer 2 weights and biases 
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

if __name__ == "__main__":
        nn = NeuralNetwork(784, 128, 10)
        print(f"W1 shape: {nn.W1.shape}")
        print(f"b1 shape: {nn.b1.shape}")
        print(f"W2 shape: {nn.W2.shape}")
        print(f"b2 shape: {nn.b2.shape}")
