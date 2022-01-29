import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# update weights
def gradient_decent(input, weights, target, predicted_value, learning_rate, bias):
    # find new weights abd bias
    new_weights = []
    """
    Rule : new_bias = old_bias + learning_rate *(target- prediction)
    """
    new_bias = bias + learning_rate * (target - predicted_value)
    """
    Rule : new_weight = old_weight + learning_rate * (target- prediction) * x[i]
    """
    for x, w in zip(input, weights):
        new_w = w + learning_rate * (target - predicted_value) * x
        new_weights.append(new_w)
    return new_weights, new_bias


# loss function
def cross_entropy(target, prediction):
    return -(target * np.log10(prediction) + (1 - target) * (np.log10(1 - prediction)))


# Activation function
def sigmoid(w_sum):
    return 1 / (1 + np.exp(-w_sum))


# get model output
def percepton(features, weights, bias):
    return np.dot(features, weights) + bias


# data
features = np.array(([0.1, 0.5, 0.2], [0.2, 0.3, 0.1], [0.7, 0.4, 0.2], [0.1, 0.4, 0.3]))
# features = np.array(([0.1, 0.5, 0.2], [0.2, 0.3, 0.1]))
targets = np.array([0, 1, 0, 1])
weights = np.array([0.4, 0.2, 0.6])
bias = 0.5
learning_rate = 0.1
epoch_loss = []
for epoch in range(2):
    for input, target in zip(features, targets):
        w_sum = percepton(input, weights, bias)
        # print("Weighted_Sum: ", w_sum)
        predicted_value = sigmoid(w_sum)
        # print("predicted value: ", predicted_value)
        # print("Loss :", cross_entropy(target, predicted_value))
        weights, bias = gradient_decent(input, weights, target, predicted_value, learning_rate, bias)
        # print("Weight :", weights, "updated bias: ", bias)
    predictions = sigmoid(percepton(features, weights, bias))
    # print("Predictions: ", predictions)
    average_loss = np.mean(cross_entropy(targets, predictions))
    epoch_loss.append(average_loss)
    print("*****************************")
    print("EPOCH", str(epoch))
    print("*****************************")
    print("Average Loss: ", average_loss)

# plot the average loss
df = pd.DataFrame(epoch_loss)
df_plot = df.plot(kind="line", grid = True).get_figure()
df_plot.savefig("Training_Loss.pdf")
#df_plot.show()
plt.show()

