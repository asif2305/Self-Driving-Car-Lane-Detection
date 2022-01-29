import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# dwaw the line
def draw(x1, x2):
    ln = plt.plot(x1, x2, '-')
    plt.pause(0.0001)
   # time.sleep(.3)
    ln[0].remove()


# activation function
def sigmoid(prediction_score):
    return 1 / (1 + np.exp(-prediction_score))

# loss error
def cross_entropy(line_parameter, all_point, target):
    # n is the total number of the point
    n = all_point.shape[0]
    # print(n)
    prediction = sigmoid(all_point * line_parameter)
    # print(prediction)
    #  cross_entropy=-(1/n)*(np.log(p).T*y + np.log(1-p).T*(1-y))
    #  return -(target * np.log10(prediction) + (1 - target) * (np.log10(1 - prediction)))
    error_loss = -(1 / n) * (np.log(prediction).T * target + (np.log(1 - prediction).T * (1 - target)))
    # print(np.log10(prediction))
    # print(target)
    return error_loss

# update weight
def gradient_decent(line_parameter, points, target_label, learning_rate):
    epoch_loss = []
    for i in range(2000):
        # points * (prediction. target) * (1/ total_points)
        total_point = points.shape[0]
        prediction = sigmoid(points * line_parameter)
        gradient = (points.T * (prediction - target_label)) * (learning_rate / total_point)
        line_parameter = line_parameter - gradient
        w1 = line_parameter.item(0)
        w2 = line_parameter.item(1)
        bias = line_parameter.item(2)
        x1 = np.array([points[:, 0].min(), points[:, 0].max()])
        x2 = -bias / w2 + x1 * (-w1 / w2)
        draw(x1, x2)
        average_loss = np.mean(cross_entropy(line_parameter, all_point, target))
        epoch_loss.append(average_loss)
        print(average_loss)
    return epoch_loss

# loss
def loss(target, prediction):
    return -(target * np.log(prediction) + (1 - target) * (np.log(1 - prediction)))


# top region

n_pts = 100
np.random.seed(0)
bias = np.ones(n_pts)
# top region data point
top_region = np.array([np.random.normal(10, 2, n_pts), np.random.normal(12, 2, n_pts), bias]).T

# bottom region data point
bottom_region = np.array([np.random.normal(5, 2, n_pts), np.random.normal(6, 2, n_pts), bias]).T
all_point = np.vstack((top_region, bottom_region))

# w1 = -0.2
# w2 = -0.35
# b= 3.5
# transpose
# line_parameter = np.matrix([w1,w2,b]).T
line_parameter = np.matrix([np.zeros(3)]).T
# x1 = np.array([bottom_region[:,0].min(),top_region[:,0].max()])

# w1x1 + w2x2 + b= 0
# x2 = -b / w2 + x1 * (-w1 / w2)

# details

# x3 = -b / w2 + x1[0] * (-w1 / w2)
# x4 = -b / w2 + x1[1] * (-w1 / w2)
# get the weights
linear_combination = all_point * line_parameter
# prediction using sigmoid
prediction = sigmoid(linear_combination)
# probabilities
# target label
target = np.array([np.zeros(n_pts), np.ones(n_pts)]).reshape(n_pts * 2, 1)
# target


print(cross_entropy(line_parameter, all_point, target))

# print(prediction.T)
# target = target
# print(target)

# converting the matrix into array
# target = (np.asarray(target)).flatten()
# prediction = (np.asarray(prediction)).flatten()

# print(target,prediction)
# print(prediction)
# error_loss= loss(target, prediction)
# print(np.mean(error_loss))

# gradient decent

_, ax = plt.subplots(figsize=(4, 4))
ax.scatter(top_region[:, 0], top_region[:, 1], color='r')
ax.scatter(bottom_region[:, 0], bottom_region[:, 1], color='b')
epoch_loss = gradient_decent(line_parameter, all_point, target, 0.06)
df = pd.DataFrame(epoch_loss)
#df_plot = df.plot(kind="line", grid=True).get_figure()
#df_plot.savefig("Training_Loss.pdf")
plt.show()
