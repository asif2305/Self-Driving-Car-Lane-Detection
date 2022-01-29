"""
Creates a basic neuron with 3 inputs.
"""
inputs = [1.2, 5.1, 2.1]
weights = [3.1, 2.1, 8.7]
bias = 3
output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + bias
# print(output)

"""
coding a layer
Creates a simple layer of neurons, with 4 inputs.
THere are 1 hidden layer with 3 nodes
"""
inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 1.0]
weights2 = [0.5, -0.91, 0.26, -0.5]
weights3 = [-0.26, -0.27, 0.17, 0.87]
bias1 = 2
bias2 = 3
bias3 = 0.5
output = [inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + bias1,
          inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + bias2,
          inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3] + bias3]
# print(output)

"""
The Dot Product
"""
inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

layer_output = []  # Output of current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0  # Output of given neuron
    for n_input, weight in zip(input, neuron_weights):
        neuron_output += n_input * weight
    neuron_output += neuron_bias
    layer_output.append(neuron_output)
print(layer_output)