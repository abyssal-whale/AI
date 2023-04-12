import os
import numpy as np
from PIL import Image, ImageFilter


class ImageLoader():
    def __init__(self, data_path, train_ratio=0.9, shuffle=True, image_mode="L", expand=1):
        """
        imageloader的初始化
        :param data_path: 图片文件根路径
        :param train_ratio: 训练集所占比例
        :param shuffle: 是否将训练集打乱
        :param image_mode: 图像转换的方式，默认为"L"，即灰度图
        :param expand: 是否扩张数据集，扩张方式待补充
        """
        self.data_path = data_path
        self.expand = expand
        self.train_ratio = train_ratio
        self.shuffle = shuffle
        self.image_mode = image_mode
        self.data = {
            "train": [],
            "validation": []
        }
        self.load()
        if self.shuffle:
            np.random.shuffle(self.data['train'])

    def load(self):
        """
        根据文件路径划分验证集与训练集，并对图像进行预处理，这里进行的是高斯模糊与归一化数据的反转(如：0.3->0.7)
        :return: None
        """
        self.tags = os.listdir(self.data_path)
        self.n_class = len(self.tags)
        self.tag_to_id = {}
        self.id_to_tag = {}

        for i in range(self.n_class):
            self.tag_to_id[self.tags[i]] = i
            self.id_to_tag[i] = self.tags[i]

        for tag in self.tags:
            tag_path = self.data_path + "/" + tag
            images = os.listdir(tag_path)
            n_imgs = len(images)
            t_v_split = int(n_imgs * self.train_ratio)

            for img in images[0:t_v_split]:
                img_name = tag_path + "/" + img
                img_data = Image.open(img_name)
                img_data = img_data.convert(self.image_mode)
                img_data = img_data.filter(ImageFilter.GaussianBlur(1))
                img_data = np.array(img_data)
                img_data = img_data.flatten()
                img_data = img_data / 255
                img_data = img_data.reshape(img_data.size, 1)
                img_data = np.abs(img_data - 1)
                self.data["train"].append((img_data, self.y_generator(self.tag_to_id[tag])))

            for img in images[t_v_split:]:
                img_name = tag_path + "/" + img
                img_data = Image.open(img_name)
                img_data = img_data.convert(self.image_mode)
                img_data = img_data.filter(ImageFilter.GaussianBlur(1))
                img_data = np.array(img_data)
                img_data = img_data.flatten()
                img_data = img_data / 255
                img_data = img_data.reshape(img_data.size, 1)
                img_data = np.abs(img_data - 1)
                self.data["validation"].append((img_data, self.y_generator(self.tag_to_id[tag])))

    def y_generator(self, y: int):
        """
        由于结果一般是one-hot编码，因此我们这里用来产生相应的one-hot数据
        :param y: 第y位为1
        :return: 返回的one-hot编码
        """
        result = np.zeros((self.n_class, 1))
        result[y, 0] = 1
        return result
