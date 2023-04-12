from Layer import Layer
import random
import numpy as np

class Network(object):
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
        tmp = 0
        for i in range(0, len(self.training_data)):
            cur_predict = self.forward(np.array(self.training_data[i]))
            cur_loss = cur_predict - np.array(self.expect[i])
            tmp += (self.expect[i] * np.log(cur_predict)).sum()
            self.backward(cur_loss)
            batch_num += 1
            if batch_num == batch_size:
                batch_num = 0
                self.change_weight(self.learning_rate)


    def check_accuracy(self, epoch_num, test_data, test_expected_result):
        accumulated_loss = 0
        test = test_data
        for i in range(0, len(test_data)):
            cur_predict = self.forward(np.array(test[i]))
            cur_predict = cur_predict.reshape((1,-1)).squeeze()
            expect = test_expected_result[i]
            expect = expect.reshape((1,-1)).squeeze()
            predict = np.argmax(cur_predict)
            real = np.argmax(expect)
            if predict != real:
                accumulated_loss = accumulated_loss + 1
        print("test accuracy,在 %d 次中产生了 %f 误差" % (len(test_data), accumulated_loss))
        avg_loss = accumulated_loss / len(test_data)
        print("分类准确率是" + str(1 - avg_loss))
        return avg_loss