import math
import random
import numpy
from Layer import Layer
import matplotlib.pyplot as plt

class BPNetwork(object):
    def __init__(self, neuron_nums, learning_rate, data, expect, loss_func, scale=0.15):
        self.learning_rate = learning_rate
        self.neuron_nums = neuron_nums
        self.training_data = data
        self.expect = expect
        self.loss_func = loss_func
        # init layers
        self.layers = []
        for i in range(0, len(neuron_nums) - 1):
            if i == len(neuron_nums) - 2:
                self.layers.append(Layer(neuron_nums[i], neuron_nums[i + 1], scale, True, self.loss_func))
            else:
                self.layers.append(Layer(neuron_nums[i], neuron_nums[i + 1], scale, False, self.loss_func))
    
        # init test data
        self.test_data = []

    def forward(self, crt_input):
        for layer in self.layers:
            crt_input = layer.forward(crt_input)
        return crt_input

    def backward(self, loss):
        for layer in reversed(self.layers):
            loss = layer.backward(loss)

    def change_weight(self, learning_rate):
        for layer in self.layers:
            layer.change_weight(learning_rate)

    def train(self, batch_size=20):
        batch_num = 0
        for i in range(0, len(self.training_data)):
            cur_predict = self.forward(numpy.array([[self.training_data[i]]]))
            cur_loss = cur_predict - numpy.array([[self.expect[i]]])
            self.backward(cur_loss)
            batch_num += 1
            if batch_num == batch_size:
                batch_num = 0
                self.change_weight(self.learning_rate)

    def draw(self, test_result, epoch_num):
        test_data = self.test_data
        plt.plot(test_data, numpy.sin(test_data))
        plt.plot(test_data, test_result)
        plt.title("epoch" + str(epoch_num))
        plt.show()

    def get_test(self):
        self.test_data = []
        for i in range(0, 500):
            self.test_data.append(random.uniform(-math.pi, math.pi))
        self.test_data.sort()
        return self.test_data

    def check_accuracy(self, epoch_num, draw=False):
        accumulated_loss = 0
        test = self.get_test()
        test_results = []
        for i in range(0, len(self.test_data)):
            cur_predict = self.forward(numpy.array([[test[i]]]))
            test_results.append(cur_predict[0][0])
            accumulated_loss += abs(numpy.sin(test[i]) - cur_predict)
        if draw:
            self.draw(test_results, epoch_num)
        print("test accuracy,在 %d 次中产生了 %f 误差" % (len(test_results), accumulated_loss))
        avg_loss = accumulated_loss / len(test_results)
        print("平均误差是" + str(avg_loss))
        return avg_loss
