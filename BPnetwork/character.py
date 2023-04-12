from character_Network import Network
import numpy as np
import os
from ImageLoader import ImageLoader
import pickle


imageLoader = ImageLoader("train", train_ratio=0.8, shuffle=True, expand=4)
train_data_list = imageLoader.data["train"]
validation_data_list = imageLoader.data["validation"]


n_valid = len(validation_data_list)
n_train = len(train_data_list)
loss = "CE"
epoch = 50

data = np.array([d[0] for d in train_data_list])
expected_result = np.array([d[1] for d in train_data_list])
test_data = np.array([d[0] for d in validation_data_list])
test_expected_result = np.array([d[1] for d in validation_data_list])
classifier = Network(neuron_nums=[28*28, 100, 100, 12], learning_rate=0.01, data = data, expect = expected_result, loss_func = loss)
for i in range(0, epoch):
    classifier.train()
    classifier.check_accuracy(epoch, test_data, test_expected_result)

f = open('models/classifyNet', 'wb')
pickle.dump(classifier, f)
f.close()
print("模型存储完毕")
