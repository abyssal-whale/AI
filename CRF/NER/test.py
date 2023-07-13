import numpy as np
from easydict import EasyDict
from collections import OrderedDict
from collections import defaultdict
from sklearn_crfsuite import CRF
import pickle
from TrainCRF import CRFModel

cn_train_path = "./Chinese/train.txt"
cn_val_path = "./Chinese/validation.txt"
en_train_path = "./English/train.txt"
en_val_path = "./English/validation.txt"
gold_path = "./example_data/example_my_result.txt"


def GetWordDict(path_lists):
    word_dict = OrderedDict()
    for path in path_lists:
        with open(path, "r", encoding="utf-8") as f:
            annotations = f.readlines()
        for annotation in annotations:
            splited_string = annotation.strip(" ").split(" ")
            if len(splited_string)<=1:
                continue
            word = splited_string[0]
            if word not in word_dict:
                word_dict[word] = len(word_dict)
    return word_dict

def GetCnDict():
    label_classes = ['NAME', 'CONT', 'EDU', 'TITLE', 'ORG', 'RACE', 'PRO', 'LOC']
    tag2id = defaultdict()
    tag2id['O'] = 0
    count = 1
    for label in label_classes:
        tag2id['B-' + label] = count
        count += 1
        tag2id['M-' + label] = count
        count += 1
        tag2id['E-' + label] = count
        count += 1
        tag2id['S-' + label] = count
        count += 1
    return dict(tag2id)

def GetEnDict():
    label_classes = ['PER', 'ORG', 'LOC','MISC']
    tag2id = defaultdict()
    tag2id['O'] = 0
    count = 1
    for label in label_classes:
        tag2id['B-' + label] = count
        count += 1
        tag2id['I-' + label] = count
        count += 1
    return dict(tag2id)


def GetData(path):
    tot_raw_words = [] 
    tot_raw_tags = []
    raw_words = []
    raw_tags = []
    with open(path, "r", encoding="utf-8") as f:
        annotations = f.readlines()
    for annotation in annotations:
        splited_string = annotation.strip(" ").strip("\n").split(" ")
        if len(splited_string)<=1:
            tot_raw_words.append(raw_words)
            tot_raw_tags.append(raw_tags)
            raw_tags = []
            raw_words = []
            continue
        word = splited_string[0]
        tag = splited_string[1]
        raw_words.append(word)
        raw_tags.append(tag)
    tot_raw_words.append(raw_words)
    tot_raw_tags.append(raw_tags)
    # print(tot_raw_words, tot_raw_tags)
    return tot_raw_words, tot_raw_tags

if __name__ == "__main__":
    flag = "cn"
    # flag = "en"
    if flag == "cn":
        word_dict = GetWordDict([cn_train_path, cn_val_path])
        tag_dict = GetCnDict()
        train_words, train_tags = GetData(cn_train_path)
        val_words, val_tags = GetData(cn_val_path)
        with open('./model/cn_model', 'rb') as f:
            crf = pickle.load(f)
        crf.val(val_words, word_dict, tag_dict, gold_path)
    else:
        word_dict = GetWordDict([en_train_path, en_val_path])
        tag_dict = GetEnDict()
        train_words, train_tags = GetData(en_train_path)
        val_words, val_tags = GetData(en_val_path)
        with open('./model/en_model', 'rb') as f:
            crf = pickle.load(f)
        crf.val(val_words, word_dict, tag_dict, gold_path)