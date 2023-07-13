import numpy as np
from easydict import EasyDict
from collections import OrderedDict
from collections import defaultdict
from sklearn_crfsuite import CRF
import pickle

# flag = "cn"
flag = "en"

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

def cn_word2features(sent, i):
    word = sent[i]
    prev_word = '<s>' if i == 0 else sent[i-1]
    next_word = '</s>' if i == (len(sent)-1) else sent[i+1]
    features = {
        'w': word,
        'w-1': prev_word,
        'w+1': next_word,
        'w-1:w': prev_word+word,
        'w:w+1': word+next_word,
        'bias': 1,
        'word.isdigit()': word.isdigit(),
    }
    return features

def en_word2features(sent, i):
    word = sent[i]
    # postag = sent[1][i]
    prev_word = '<s>' if i == 0 else sent[i-1]
    next_word = '</s>' if i == (len(sent)-1) else sent[i+1]
    # prev_tag = 'BOS' if i == 0 else sent[1][i-1]
    # next_tag = 'EOS' if i == (len(sent[1])-1) else sent[1][i+1]
    features = {
        'w': word,
        'w-1': prev_word,
        'w+1': next_word,
        'w-1:w': prev_word+word,
        'w:w+1': word+next_word,
        'bias': 1,
        # 't': postag,
        # 't-1': prev_tag + postag,
        # 't+1': postag + next_tag,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.isdigit()': word.isdigit(),
        'prevword.lower()': prev_word.lower(),
        'prevword.isupper()': prev_word.isupper(),
        'prevword.isdigit()': prev_word.isdigit(),
        'nextword.lower()': next_word.lower(),
        'nextword.isupper()': next_word.isupper(),
        'nextword.isdigit()': next_word.isdigit()
    }
    return features

def sent2features(sent):
    if flag == 'cn':
        return [cn_word2features(sent, i) for i in range(len(sent))]
    else:
        return [en_word2features(sent, i) for i in range(len(sent))]


class CRFModel(object):
    def __init__(self, algorithm='lbfgs', c1=0.1, c2=0.1, 
                 max_iterations=100, all_possible_transitions=False):
        self.crf = CRF(algorithm=algorithm,
                         c1=c1,
                         c2=c2,
                         max_iterations=max_iterations,
                         all_possible_transitions=all_possible_transitions)

    def train(self, train_words, train_tags):
        features = [sent2features(s) for s in train_words]
        self.crf.fit(features, train_tags)

    def val(self, val_words, word_dict, tag_dict, out_path):
        f = open(out_path, "w", encoding="utf-8")
        features = [sent2features(s) for s in val_words]
        preds = self.crf.predict(features)
        for i, words in enumerate(val_words):
            for j in range(len(words)): # find the key
                f.write(words[j] + " " + preds[i][j] + "\n")
            if i!=len(val_words)-1:
                f.write("\n")
        f.close()


if __name__ == "__main__":
    if flag == "cn":
        word_dict = GetWordDict([cn_train_path, cn_val_path])
        tag_dict = GetCnDict()
        train_words, train_tags = GetData(cn_train_path)
        val_words, val_tags = GetData(cn_val_path)
        crf = CRFModel()
        crf.train(train_words, train_tags)
        f = open('./model/cn_model', 'wb')
        pickle.dump(crf, f)
        f.close()
        crf.val(val_words, word_dict, tag_dict, gold_path)
    else:
        word_dict = GetWordDict([en_train_path, en_val_path])
        tag_dict = GetEnDict()
        train_words, train_tags = GetData(en_train_path)
        val_words, val_tags = GetData(en_val_path)
        crf = CRFModel()
        crf.train(train_words, train_tags)
        f = open('./model/en_model', 'wb')
        pickle.dump(crf, f)
        f.close()
        crf.val(val_words, word_dict, tag_dict, gold_path)




    

    
    

