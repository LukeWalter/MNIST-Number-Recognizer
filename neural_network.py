import numpy as np
import idx2numpy
        
class NeuralNet:

    def sigmoid(n):
        ex = np.pow(np.e, n)
        return ex / (1 + ex)

    def __init__(self, in_size, hl_size, hl_count, out_size):
        
        self.in_size = in_size
        self.hl_size = hl_size
        self.hl_count = hl_count
        self.out_size = out_size
        
        self.hidden_layers = np.zeros((2, self.hl_count, self.hl_size), dtype=float)

        self.input_data = np.zeros((1, self.in_size), dtype=float)
        self.in_to_hl_weights = np.random.uniform(low=-1, high=1, size=(self.in_size, self.hl_size))
        self.hl_weight_rows = (self.hl_count - 1) * self.hl_size + self.out_size
        self.hl_weights = np.random.uniform(low=-1, high=1, size=(self.hl_weight_rows, self.hl_size))
        self.out_biases = np.zeros((1, self.out_size), dtype=float)
        self.out_data = np.zeros((1, self.out_size), dtype=float)
        
    def forward_prop(self, in_data):

        # Add a check here to see if the input data has the correct shape (based on in_size)
        self.input_data = np.array([(in_data / 255).flatten()])

        self.hidden_layers[0, 0] = NeuralNet.sigmoid(self.input_data @ self.in_to_hl_weights + self.hidden_layers[1, 0])

        for i in range(self.hl_count - 1):
            self.hidden_layers[0, i+1] = NeuralNet.sigmoid(self.hidden_layers[0, 0] @ self.hl_weights[i*self.hl_size:(i+1)*self.hl_size] + self.hidden_layers[1, (i+1)])

        self.out_data = NeuralNet.sigmoid(np.array([self.hidden_layers[0, self.hl_count - 1]]) @ self.hl_weights[(self.hl_count-1)*self.hl_size:(self.hl_count-1)*self.hl_size+self.out_size].T + self.out_biases)
        return self.out_data
        
    def back_prop(self, out_real, out_ideal):

        learn_rate = -0.01

        error = out_ideal - out_real
        self.hl_weights[(self.hl_count-1)*self.hl_size:(self.hl_count-1)*self.hl_size+self.out_size] -= learn_rate * (np.array([self.hidden_layers[0, self.hl_count - 1]]).T @ error).T
        self.out_biases -= learn_rate * error
        h = self.hidden_layers[0, self.hl_count - 1]
        derivative = np.array([h * (1 - h)])
        delta_h = error @ self.hl_weights[(self.hl_count-1)*self.hl_size:(self.hl_count-1)*self.hl_size+self.out_size] * derivative

        for i in range(self.hl_count - 1):
            
            layer = (self.hl_count - 2) - i
            
            self.hl_weights[layer*self.hl_size:(layer+1)*self.hl_size] -= learn_rate * (np.array([self.hidden_layers[0, layer]]).T @ delta_h).T
            self.hidden_layers[1, (layer+1)] -= learn_rate * delta_h[0]
            h = self.hidden_layers[0, layer]
            derivative = np.array([h * (1 - h)])
            delta_h = delta_h @ self.hl_weights[layer*self.hl_size:(layer+1)*self.hl_size] * derivative

        self.in_to_hl_weights -= learn_rate * self.input_data.T @ delta_h
        self.hidden_layers[1, 0] -=  learn_rate * delta_h[0]
        
    def train_on_input(self, training_input, training_output):
        
        out_real = self.forward_prop(training_input)
        out_ideal = np.zeros(out_real.shape, dtype=float)
        out_ideal[0, training_output] = 1
        
        self.back_prop(out_real, out_ideal)

# Training / testing data from MNIST
train_images = idx2numpy.convert_from_file('train-images.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
test_images = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

nn = NeuralNet(784, 10, 3, 10)
# nn.train_on_input(training_input, training_output)


# Write an algorithm to train the neural network on each training example in the MNIST database
# Loop through each training example
# Call the train method


total_correct = 0
print(test_images.shape)

for i in range(test_images.shape[0]):
    testing_input = train_images[i]
    testing_output = train_labels[i]
    output = nn.forward_prop(testing_input)
    print(f"Ideal: {testing_output} | Real: {np.argmax(output)}")
    
    if testing_output == np.argmax(output):
        total_correct += 1
        
print(f"{total_correct / test_images.shape[0] * 100}% correct")

# nn2 = NeuralNet(4, 5, 3, 2)
# nn.train_on_input(np.array([0, 1, 1, 0]), 1)