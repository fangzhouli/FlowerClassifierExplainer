#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 23:25:00 2020

@author: xiawei
"""

import cv2
import numpy as np
import os
import random
from sklearn.model_selection import train_test_split


def preprocess_data(folder_path="../data/raw-img/", img_size=200,
                    test_size=0.2):
    """
    input:
        folder_path (str): local path to the folder of raw images of animals-10
            dataset, the folder contains 10 subfolders
        img_size (int): desired size of output images
        test_size (float): proportion of the dataset that are split into test
            set
    output:
        x_train: standardized gray-scale images in training set, (none, img_size, img_size,1), np.array, float32
        y_train: labels of x_train, (none,), np.array, int64
        x_test: standardized gray-scale images in test_set, (none, img_size, img_size,1), np.array, float32
        y_test: labels of x_test, (none,), np.array, int64    
    """

    # Create dataset that contains images and corresponding labels, labels are translated into English
    categories = {'cane': 'dog', "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly",
                  "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "scoiattolo": "squirrel", "ragno": "spider"}
    data = []
    animals = ["dog", "horse", "elephant", "butterfly",
               "chicken", "cat", "cow", "sheep", "squirrel", "spider"]
    for category, translation in categories.items():
        path = folder_path + category
        label = animals.index(translation)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(
                    path, img), cv2.IMREAD_GRAYSCALE)
                new_img_array = cv2.resize(img_array, (img_size, img_size))
                data.append([new_img_array, label])
            except Exception as e:
                pass
    # Split dataset into training and test set
    random.shuffle(data)
    x = []
    y = []
    for img, label in data:
        x.append(img)
        y.append(label)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size)
    # Resize and standardize images
    x_train = np.array(x_train).reshape(-1, img_size, img_size, 1)
    x_test = np.array(x_train).reshape(-1, img_size, img_size, 1)
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = preprocess_data(
        "./example-10/raw-img/", 100, 0.3)
