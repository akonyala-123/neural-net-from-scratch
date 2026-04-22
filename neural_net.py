import numpy as np

class NeuralNetwork: 
    def __init__(self, input_size, hidden_size, output_size):
        #Layer 1 weights and biases 
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))

        #layer 2 weights and biases 
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    #Activation function - determines how strongly a neuron should fire 
    def relu(self, x): 
        return np.maximum(0, x)

    #Softmax will run on the final layer and convert activations to probablities
    def softmax(self, x):
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def forward(self, X):
        #Compute activation for neurons in layer 1
        self.z1 = X @ self.W1 + self.b1 
        self.a1 = self.relu(self.z1)
        #Compute activation for neurons in layer 2 
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

if __name__ == "__main__":
        nn = NeuralNetwork(784, 128, 10)
        #This is a random image 
        X = np.random.randn(1, 784)
        output = nn.forward(X)
        print(f"output shape: {output.shape}")
        print(f"output: {output}")
        print(f"sum: {output.sum():.4f}")
        print(f"predicted digit: {np.argmax(output)}")
