from sinx_Network import BPNetwork
import matplotlib.pyplot as plt
import numpy as np
import math
import pickle


data = np.linspace(-math.pi, math.pi, 3000)
np.random.shuffle(data)
expected_result = np.sin(data)
loss = "MSE"
bp_sin = BPNetwork(neuron_nums=[1, 30, 30, 1], learning_rate=0.01, data = data, expect = expected_result, loss_func = loss)
bp_sin.check_accuracy(True)
x = []
y = []
for epoch in range(0, 3001):
    bp_sin.train()
    if epoch % 500 == 0:
        print("it is epoch" + str(epoch))
        x.append(epoch)
        y.append(bp_sin.check_accuracy(epoch, True)[0][0])
plt.plot(x, y)
plt.show()

f = open('models/sinNet', 'wb')
pickle.dump(bp_sin, f)
f.close()
print("模型存储完毕")