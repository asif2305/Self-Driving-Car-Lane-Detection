import numpy as np


# loss function
def cross_entropy(target, prediction):
    return -(target * np.log10(prediction) + (1-target) * (np.log10(1-prediction)))


# Activation function
def sigmoid(w_sum):
    return 1/(1 + np.exp(-w_sum))

# get model output
def percepton(features, weights, bias):
    return np.dot(features, weights) + bias

# data
#features = np.array(([0.1, 0.5, 0.2], [0.2, 0.3, 0.1],[0.7,0.4,0.2],[0.1,0.4,0.3]))
features = np.array(([0.1, 0.5, 0.2], [0.2, 0.3, 0.1]))
targets = np.array([0, 1])
weights = np.array([0.4, 0.2, 0.6])
bias = 0.5
learning_rate = 0.1


for input,target in zip(features, targets):
    w_sum=percepton(input,weights,bias)
    print("Weighted_Sum: ",w_sum)
    predicted_value = sigmoid(w_sum)
    print("predicted value: ",predicted_value)
    print("Loss :", cross_entropy(target,predicted_value))


