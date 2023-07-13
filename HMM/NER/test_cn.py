import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn import metrics
from itertools import chain
import pickle
from HMM_cn import HMM_model

def load_data(data_path: str):
    sentences = []
    with open(data_path, 'r', encoding='utf-8') as fp:
        sentence = []
        text_data = []
        label_data = []
        for line in fp:
            if len(line) > 1:
                text = line[0]
                label = line[2:-1]
                text_data.append(text) 
                label_data.append(label)
            else:
                sentence.append(text_data)
                sentence.append(label_data)
                sentences.append(sentence)
                sentence = []
                text_data = []
                label_data = []
    return sentences

def save(path, y_pred, y_gold):
    f = open(path, "w", encoding="utf-8")
    for i in range(len(y_pred)):
        sentence = y_pred[i]
        for j in range(len(sentence)): # find the key
            f.write(y_gold[i][j] + " " + y_pred[i][j] + "\n")
        f.write("\n")
    f.close()


if __name__ == "__main__":
    valid_data = load_data("./Chinese/validation.txt")
    with open('./model/cn_model', 'rb') as f:
        model = pickle.load(f)
    model.load_paramters()
    y_pred = model.valid(valid_data)
    y_true = [data[1] for data in valid_data]
    y_gold = [data[0] for data in valid_data]
    save("./example_data/example_my_result.txt", y_pred, y_gold)  