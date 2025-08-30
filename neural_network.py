import numpy as np
import idx2numpy

def sigmoid(n):
    ex = np.pow(np.e, n)
    return ex / (1 + ex)

# Training / testing data from MNIST
train_images = idx2numpy.convert_from_file('train-images.idx3-ubyte')
train_labels = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
test_images = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
test_labels = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')

in_size = 784
hl_size = 10
hl_count = 3
out_size = 5

input_data = np.array([(train_images[0] / 255).flatten()])

hidden_layers = np.zeros((2, hl_count, hl_size), dtype=float)

in_to_hl_weights = np.random.uniform(low=-1, high=1, size=(in_size, hl_size))
hl_weight_rows = (hl_count - 1) * hl_size + out_size
hl_weights = np.random.uniform(low=-1, high=1, size=(hl_weight_rows, hl_size))
# print(hl_weights.shape)

# Forward propagation

hidden_layers[0, 0] = sigmoid(input_data @ in_to_hl_weights + hidden_layers[1, 0])

for i in range(hl_count - 1):
    hidden_layers[0, i+1] = sigmoid(hidden_layers[0, 0] @ hl_weights[i*hl_size:(i+1)*hl_size] + hidden_layers[1, (i+1)])

out_biases = np.zeros((1, out_size), dtype=float)
out_data = sigmoid(np.array([hidden_layers[0, hl_count - 1]]) @ hl_weights[(hl_count-1)*hl_size:(hl_count-1)*hl_size+out_size].T + out_biases)
# print(out_data)
print(out_biases)

# Back propagation
learn_rate = -0.01

out_ideal = np.zeros(out_data.shape, dtype=float)
out_ideal[0, train_labels[1]] = 1

# output layer to hidden layers
error = out_ideal - out_data
# hl_weights[(hl_count-1)*hl_size:(hl_count-1)*hl_size+out_size] -= learn_rate * (np.array([hidden_layers[0, 2]]).T @ error).T
hl_weights[20:30] -= learn_rate * (np.array([hidden_layers[0, 2]]).T @ error).T
out_biases -= learn_rate * error
h = hidden_layers[0, 2]
derivative = np.array([h * (1 - h)])
# delta_h = error @ hl_weights[(hl_count-1)*hl_size:(hl_count-1)*hl_size+out_size] * derivative
delta_h = error @ hl_weights[20:30] * derivative
# print(delta_h[0])

# hidden layers
hl_weights[10:20] -= learn_rate * (np.array([hidden_layers[0, 1]]).T @ delta_h).T
# print(hidden_layers[1, 2])
hidden_layers[1, 2] -= learn_rate * delta_h[0]
h = hidden_layers[0, 1]
derivative = derivative = np.array([h * (1 - h)])
delta_h = delta_h @ hl_weights[10:20] * derivative
# print(delta_h)
# print(hidden_layers[1, 2])

hl_weights[0:10] -= learn_rate * (np.array([hidden_layers[0, 0]]).T @ delta_h).T
hidden_layers[1, 1] -=  learn_rate * delta_h[0]
h = hidden_layers[0, 0]
derivative = np.array([h * (1 - h)])
delta_h = delta_h @ hl_weights[0:10] * derivative
# print(delta_h)

# hidden layers to input layer
# print(input_data)
# print(input_data.shape)
# print((input_data.T @ delta_h).shape)
# print(in_to_hl_weights.shape)
print(in_to_hl_weights)
in_to_hl_weights -= 1000 * input_data.T @ delta_h
print(in_to_hl_weights)
