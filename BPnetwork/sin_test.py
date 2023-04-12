import pickle
import math
import numpy as np
from matplotlib import pyplot as plt
from sinx_Network import BPNetwork

with open('models/sinNet', 'rb') as f:
    bp_sinx = pickle.load(f)

x = np.linspace(-math.pi, math.pi, 3000)
x_list = []
for x_ in x:
    x_list.append(np.array([[x_]]))
x_list = np.array(x_list)

y_list = []
for i in range(3000):
    y_list.append(bp_sinx.forward(x_list[i]))
y_list = np.array(y_list)

x_list = np.squeeze(x_list)
y_list = np.squeeze(y_list)

plt.axis([-5, 5, -1.5, 1.5])
plt.plot(x_list, y_list)
plt.show()