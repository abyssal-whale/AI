import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn import metrics
from itertools import chain
from easydict import EasyDict
from collections import OrderedDict
import matplotlib.pyplot as plt
import pickle
from MyDataset import MyDataset
from BiLSTM_CRF import BiLSTM_CRF

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# kind = "Cn" #### 
kind = "En" #### 
train_path = "./Chinese/train.txt" if kind == "Cn" else "./English/train.txt"
val_path = "./Chinese/validation.txt" if kind == "Cn" else "./English/validation.txt"
val1_path = "./Chinese/validation0.txt" if kind == "Cn" else "./English/validation0.txt"
out_path = "./Chinese/out.txt" if kind == "Cn" else "./English/out.txt"
batch_size = 32
epochs = 50
embedding_size = 35 if kind == "Cn" else 11
hidden_dim = 768
is_load = True
load_path = "./model/BiLSTM_CRF01.pth" if kind == "Cn" else "./model/BiLSTM_CRF02.pth"
save_path = "./model/BiLSTM_CRF01.pth" if kind == "Cn" else "./model/BiLSTM_CRF02.pth"

cn_tag_dict = {'O': 0, 'B-NAME': 1, 'M-NAME': 2, 'E-NAME': 3, 
      'S-NAME': 4, 'B-CONT': 5, 'M-CONT': 6, 'E-CONT': 7, 'S-CONT': 8,
      'B-EDU': 9, 'M-EDU': 10, 'E-EDU': 11, 'S-EDU': 12, 'B-TITLE': 13, 
      'M-TITLE': 14, 'E-TITLE': 15, 'S-TITLE': 16, 'B-ORG': 17, 
      'M-ORG': 18, 'E-ORG': 19, 'S-ORG': 20, 'B-RACE': 21, 'M-RACE': 22, 
      'E-RACE': 23, 'S-RACE': 24, 'B-PRO': 25, 'M-PRO': 26, 'E-PRO': 27, 
      'S-PRO': 28, 'B-LOC': 29, 'M-LOC': 30, 'E-LOC': 31, 'S-LOC': 32,
      "<s>": 33, "</s>": 34}

en_tag_dict = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3, 
      "I-ORG": 4, "B-LOC": 5, "I-LOC": 6, "B-MISC": 7, "I-MISC": 8,
      "<s>": 9, "</s>": 10}

sorted_labels_eng= ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC", "B-MISC" , "I-MISC"]

sorted_labels_chn = [
'O',
'B-NAME', 'M-NAME', 'E-NAME', 'S-NAME'
, 'B-CONT', 'M-CONT', 'E-CONT', 'S-CONT'
, 'B-EDU', 'M-EDU', 'E-EDU', 'S-EDU'
, 'B-TITLE', 'M-TITLE', 'E-TITLE', 'S-TITLE'
, 'B-ORG', 'M-ORG', 'E-ORG', 'S-ORG'
, 'B-RACE', 'M-RACE', 'E-RACE', 'S-RACE'
, 'B-PRO', 'M-PRO', 'E-PRO', 'S-PRO'
, 'B-LOC', 'M-LOC', 'E-LOC', 'S-LOC'
]

def GetDict(path_lists):
    word_dict = OrderedDict()
    word_dict["_PAD"] = 0
    word_dict["_UNKNOW"] = 1
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


def eval(model, word_dict, tag_dict):
    model.eval()
    model.state = 'eval'
    all_label = []
    all_pred = []
    # word_dict_rev = {k:v for v,k in word_dict.items()}
    tag_dict_rev = {k:v for v,k in tag_dict.items()}
    print("******Evaluating******\n")
    eval_dataset = MyDataset(path = val_path,
                              word_dict = word_dict,
                              tag_dict = tag_dict, )
    eval_loader = DataLoader(dataset = eval_dataset, 
                              shuffle = True, 
                              batch_size = batch_size,
                              drop_last = False,
                              collate_fn = eval_dataset.collect_fn)
    for i, batch in enumerate(eval_loader):
        words_batch, tags_batch, leng = batch # torch.tensor
        words_batch = words_batch.to(device)
        tags_batch = tags_batch.to(device)
        leng = leng.to(device)
        with torch.no_grad():
            batch_tag = model(words_batch, leng, tags_batch)
            all_label.extend([[tag_dict_rev[t] for t in l[:leng[i]].tolist()] for i, l in enumerate(tags_batch)])
            all_pred.extend([[tag_dict_rev[t] for t in l] for l in batch_tag])
    all_label = list(chain.from_iterable(all_label))
    all_pred = list(chain.from_iterable(all_pred))
    print(metrics.classification_report(all_label, all_pred, labels=sorted_labels_eng[1:], digits=4))


if __name__ == "__main__":
    with open('./model/final_en', 'rb') as f:
        crf = pickle.load(f)
    tag_dict = cn_tag_dict if kind == "Cn" else en_tag_dict
    word_dict = GetDict([train_path, val1_path])
    eval(crf, word_dict, tag_dict)