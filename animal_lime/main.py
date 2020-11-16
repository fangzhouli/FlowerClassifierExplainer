# -*- encoding: utf-8 -*-
"""
"""

from os import path, listdir
import cv2
import numpy as np
from tensorflow.keras import layers, models
import tensorflow as tf

DATA_PATH = path.abspath(path.dirname(__name__)) + '/data/raw-img'


class BinaryModel(tf.keras.Model):
    """
    """

    def __init__(self):
        super().__init__()
        self.conv2D = layers.Conv2D(4, (3, 3), activation='relu')
        self.max_pool = layers.MaxPooling2D((2, 2))
        self.flat = layers.Flatten()
        self.dense_1 = layers.Dense(8)
        self.dense_2 = layers.Dense(2)

    def call(self, inputs):
        x = self.conv2D(inputs)
        x = self.max_pool(x)
        x = self.flat(x)
        x = self.dense_1(x)
        return self.dense_2(x)


if __name__ == '__main__':
    """
    """
    # preprocess
    n_samples = 1000
    res_height = 200
    res_width = 200
    train_ratio = 0.7

    data_cat_path = DATA_PATH + '/gatto'
    data_dog_path = DATA_PATH + '/cane'
    data_cat_files = listdir(data_cat_path)
    data_dog_files = listdir(data_dog_path)

    img_cat = [cv2.imread(data_cat_path + '/' + file)
               for file in data_cat_files[:n_samples]]
    img_dog = [cv2.imread(data_dog_path + '/' + file)
               for file in data_dog_files[:n_samples]]

    img_cat_resized = [cv2.resize(
        img, (res_height, res_width), interpolation=cv2.INTER_LINEAR)
        for img in img_cat]
    img_dog_resized = [cv2.resize(
        img, (res_height, res_width), interpolation=cv2.INTER_LINEAR)
        for img in img_dog]

    n_train = int(n_samples * train_ratio)
    X_train = np.asarray(img_cat_resized[:n_train] + img_dog_resized[:n_train]) / 255
    y_train = np.asarray([1] * n_train + [0] * n_train)
    X_test = np.asarray(img_cat_resized[n_train:] + img_dog_resized[n_train:]) / 255
    y_test = np.asarray([1] * (n_samples - n_train) +
                        [0] * (n_samples - n_train))

    # Model
    model = BinaryModel()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])
    model.fit(x=X_train, y=y_train, epochs=10,
              validation_data=(X_test, y_test))


    # model = models.Sequential()
    # model.add(layers.Conv2D(4, (3, 3), activation='relu',
    #                         input_shape=(res_height, res_width, 3)))
    # model.add(layers.MaxPooling2D((2, 2)))
    # model.add(layers.Conv2D(8, (3, 3), activation='relu'))
    # model.add(layers.Flatten())
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dense(2))

    model.save(path.abspath(path.dirname(__name__)) + '/models/test')
