import numpy as np
import matplotlib.pyplot as plt


def cost_func(x, y, h):
    return (1 / (2 * np.size(x))) * (np.sum((h - y)**2))


# x are our features.
x = np.arange(20)*(-2)
# y are our output.
y = np.arange(20) * 0.5316 + 3
# initializing our hypothesis h ( theta1, theta2)
theta1 = 0
theta0 = 0
h = theta1 * x + theta0
learning_rate = 0.003
b = 1
iterations = 0
while b > 0.0000001:
    cost_func1 = cost_func(x, y, h)
    theta1 = theta1 - learning_rate * (1 / np.size(x)) * np.sum((h - y) * x)
    theta0 = theta0 - learning_rate * (1 / np.size(x)) * np.sum(h - y)
    h = theta1 * x + theta0
    b = cost_func1 - cost_func(x, y, h)
    iterations += 1
    print('iteration {}, theta1 {}, theta0 {}'.format(iterations, theta1, theta0))
    print('cost function {}'.format(cost_func(x, y, h)))



# Plotting
plt.plot(x, h, 'r')
plt.scatter(x, y)
plt.show()
