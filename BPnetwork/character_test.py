import pickle
import math
import numpy as np
from matplotlib import pyplot as plt
from character_Network import Network
from ImageLoader import ImageLoader

with open('models/classifyNet', 'rb') as f:
    classifier = pickle.load(f)

imageLoader = ImageLoader("train", train_ratio=0.8)
validation_data_list = imageLoader.data["validation"]

total = len(validation_data_list)
acc = 0
for x,y in validation_data_list:
    y_pred_value = classifier.forward(x)
    y_pred = np.argmax(y_pred_value)
    y_real = np.argmax(y)
    if y_pred == y_real:
        acc += 1
print("acc:",acc/total)
