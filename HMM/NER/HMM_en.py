import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from collections import defaultdict
from sklearn import metrics
from itertools import chain
import pickle

label_classes = ['PER', 'ORG', 'LOC', 'MISC']

tag2idx = defaultdict()
tag2idx['O'] = 0
count = 1
for label in label_classes:
    tag2idx['B-' + label] = count
    count += 1
    tag2idx['I-' + label] = count
    count += 1

def GetDict(path_lists):
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
    return dict(word_dict)

def load_data(data_path: str):
    sentences = []
    with open(data_path, 'r') as fp:
        sentence = []
        text_data = []
        label_data = []
        for line in fp:
            if len(line) > 1:
                splited_string = line[:-1].split(" ")
                text = splited_string[0]
                label = splited_string[1]
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

sentences = load_data("./English/train.txt")

class HMM_model:
    def __init__(self, tag2idx):
        self.tag2idx = tag2idx  # tag2idx字典
        self.n_tag = len(self.tag2idx)  # 标签个数
        self.n_char = 65535  # 所有字符的Unicode编码个数，包括汉字
        self.epsilon = 1e-100  # 无穷小量，防止归一化时分母为0
        self.idx2tag = dict(zip(self.tag2idx.values(), self.tag2idx.keys()))  # idx2tag字典
        self.A = np.zeros((self.n_tag, self.n_tag))  # 状态转移概率矩阵, shape:(21, 21)
        self.B = np.zeros((self.n_tag, self.n_char))  # 观测概率矩阵, shape:(21, 65535)
        self.pi = np.zeros(self.n_tag)  # 初始隐状态概率,shape：(21,)

    def train(self, train_data, word_dict):
        print('开始训练数据：')
        for i in tqdm(range(len(train_data))):  # 几组数据
            for j in range(len(train_data[i][0])):  # 每组数据中几个字符
                cur_char = train_data[i][0][j]  # 取出当前字符
                cur_tag = train_data[i][1][j]  # 取出当前标签
                self.B[self.tag2idx[cur_tag]][word_dict[cur_char]] += 1  # 对B矩阵中标签->字符的位置加一
                if j == 0:
                    # 若是文本段的第一个字符，统计pi矩阵
                    self.pi[self.tag2idx[cur_tag]] += 1
                    continue
                pre_tag = train_data[i][1][j - 1]  # 记录前一个字符的标签
                self.A[self.tag2idx[pre_tag]][self.tag2idx[cur_tag]] += 1  # 对A矩阵中前一个标签->当前标签的位置加一

        # 防止数据下溢,对数据进行对数归一化
        self.A[self.A == 0] = self.epsilon
        self.A = np.log(self.A) - np.log(np.sum(self.A, axis=1, keepdims=True))
        self.B[self.B == 0] = self.epsilon
        self.B = np.log(self.B) - np.log(np.sum(self.B, axis=1, keepdims=True))
        self.pi[self.pi == 0] = self.epsilon
        self.pi = np.log(self.pi) - np.log(np.sum(self.pi))

        # 将A，B，pi矩阵保存到本地
        np.savetxt('./English/A.txt', self.A)
        np.savetxt('./English/B.txt', self.B)
        np.savetxt('./English/pi.txt', self.pi)
        print('训练完毕！')

    # 载入A，B，pi矩阵参数
    def load_paramters(self, A='./English/A.txt', B='./English/B.txt', pi='./English/pi.txt'):
        self.A = np.loadtxt(A)
        self.B = np.loadtxt(B)
        self.pi = np.loadtxt(pi)

    # 使用维特比算法进行解码
    def viterbi(self, s, word_dict):
        # 计算初始概率，pi矩阵+第一个字符对应各标签概率
        delta = self.pi + self.B[:, word_dict[s[0]]]
        # 前向传播记录路径
        path = []
        for i in range(1, len(s)):
            # 广播机制，重复加到A矩阵每一列
            tmp = delta.reshape(-1, 1) + self.A
            # 取最大值作为节点值，并加上B矩阵
            delta = np.max(tmp, axis=0) + self.B[:, word_dict[s[i]]]
            # 记录当前层每一个节点的最大值来自前一层哪个节点
            path.append(np.argmax(tmp, axis=0))

        # 回溯，先找到最后一层概率最大的索引
        index = np.argmax(delta)
        results = [self.idx2tag[index]]
        # 逐层回溯，沿着path找到起点
        while path:
            tmp = path.pop()
            index = tmp[index]
            results.append(self.idx2tag[index])
        # 序列翻转
        results.reverse()
        return results

    def predict(self, s):
        results = self.viterbi(s)
        for i in range(len(s)):
            print(s[i] + results[i], end=' | ')

    def valid(self, valid_data, word_dict):
        y_pred = []
        # 遍历验证集每一条数据，使用维特比算法得到预测序列，并加到列表中
        for i in range(len(valid_data)):
            y_pred.append(self.viterbi(valid_data[i][0], word_dict))
        return y_pred
    
def save(path, y_pred, y_gold):
    f = open(path, "w")
    for i in range(len(y_pred)):
        sentence = y_pred[i]
        for j in range(len(sentence)): # find the key
            f.write(y_gold[i][j] + " " + y_pred[i][j] + "\n")
        f.write("\n")
    f.close()

if __name__=="__main__":
    word_dict = GetDict(["./English/train.txt","./English/validation.txt"])
    print(word_dict)
    train_data = load_data("./English/train.txt")
    valid_data = load_data("./English/validation.txt")
    model = HMM_model(tag2idx)
    model.train(train_data, word_dict)
    f = open('./model/en_model', 'wb')
    pickle.dump(model, f)
    f.close()
    model.load_paramters()
    y_pred = model.valid(valid_data, word_dict)
    y_true = [data[1] for data in valid_data]
    y_gold = [data[0] for data in valid_data]
    save("./example_data/example_my_result.txt", y_pred, y_gold)
 