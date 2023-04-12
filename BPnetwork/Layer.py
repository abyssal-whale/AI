import math
import random
import numpy as np

def sigmoid(x): # 一点也不好用
    return 1.0 / (1 + np.exp(-x))

def sigmoid_derive(x):
    return sigmoid(x)*(1 - sigmoid(x))

def tanh(x):
    exp = np.exp(x)
    return (exp - 1 / exp) / (exp + 1 / exp)

def tanh_derive(x):
    return 1 - tanh(x) ** 2


class Layer(object):
    def __init__(self, input_size, output_size, random_scale, is_last, loss_func):
        self.bias = np.random.normal(scale=random_scale, size=(output_size, 1))
        self.weight = np.random.normal(scale=random_scale, size=(input_size, output_size))
        self.input = np.zeros((input_size, 1))
        self.sums = np.zeros((output_size, 1))
        self.result = np.zeros((output_size, 1))
        # 暂时存储一个batch内的改变
        self.batch_delta_weight = np.zeros(self.weight.shape)
        self.batch_delta_bias = np.zeros(self.bias.shape)
        self.crt_batch_cnt = 0
        self.is_last = is_last
        self.loss_func = loss_func

    def calc_sum(self, crt_input):
        self.input = crt_input
        input_sum = np.dot(self.weight.T,self.input) + self.bias
        self.sums = input_sum
        return self.sums

    def activate(self,):
        if self.loss_func == "MSE":
            if not self.is_last:
                self.result = tanh(self.sums)
            else:
                self.result = self.sums
        else:
            if not self.is_last:
                self.result = tanh(self.sums)
            else:
                self.result = self.Softmax(self.sums)
        return self.result

    def Softmax(self, x):
        y = np.exp(x)
        z = y.sum()
        return y / z

    def save_batch(self, weight, bias):
        self.batch_delta_weight += weight
        self.batch_delta_bias += bias
        self.crt_batch_cnt += 1

    def change_weight(self, learning_rate):
        delta_weight = learning_rate * self.batch_delta_weight / self.crt_batch_cnt
        delta_bias = learning_rate * self.batch_delta_bias / self.crt_batch_cnt
        self.crt_batch_cnt = 0
        self.batch_delta_bias = np.zeros(self.bias.shape)
        self.batch_delta_weight = np.zeros(self.weight.shape)
        self.weight += delta_weight
        self.bias += delta_bias

    def forward(self, crt_input):
        self.calc_sum(crt_input)
        self.activate()
        return self.result

    def backward(self, loss):
        if self.is_last:
            if self.loss_func == "MSE":
                bias_gradient = loss * tanh_derive(self.sums)
            else:
                bias_gradient = loss
        else:
            bias_gradient = loss * tanh_derive(self.sums)
        weight_gradient = np.dot(self.input,bias_gradient.T)
        self.save_batch(-weight_gradient, -bias_gradient)
        loss_next = np.dot(self.weight, bias_gradient)
        return loss_next