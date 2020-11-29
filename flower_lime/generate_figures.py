# -*- coding: utf-8 -*-
"""Generates figures for the paper.

This script generates figures of myLIME results. The purpose of these figures
is comparing different local explainers on different models.

Example:
        $ python generate_figures.py

Authors:
    Fangzhou Li - https://github.com/fangzhouli
    Chengyang Wang - https://github.com/cyywang-git

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

"""
from os import path
import pathlib
import matplotlib.pyplot as plt

from skimage.segmentation import mark_boundaries
import numpy as np
import tensorflow as tf

from lime import lime_image


def load_x_per_class():
    """
    """
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/"\
                  "example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file(
        'flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)

    ds_val = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(180, 180),
        batch_size=32)

    x_per_class = [[] for _ in range(5)]
    count = 0
    for x_batch, y_batch in ds_val:
        for x, y in zip(x_batch, y_batch):
            x_per_class[y].append(x)
            count += 1
        if count >= 1000:
            break
    return x_per_class


if __name__ == '__main__':

    PATH_DIR_MODELS = path.abspath(path.dirname(__file__)) + '/models'
    class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    model_regressors = ['linear', 'logistic', 'tree']

    # Load a good classifier
    model_good = tf.keras.models.load_model(PATH_DIR_MODELS + '/good_model')
    x_per_class = load_x_per_class()
    explainer = lime_image.LimeImageExplainer()
    n_exp = 10

    for i in range(len(x_per_class)):
        class_name = class_names[i]
        count = 0

        for x in x_per_class[i]:
            x = np.array([x])
            y_pred = np.argmax(model_good.predict(x)[0])
            if y_pred == i:
                for regressor in model_regressors:
                    explanation = explainer.explain_instance(
                        x[0].astype('double'),
                        model_good.predict,
                        model_regressor=regressor,
                        top_labels=5,
                        hide_color=0,
                        num_samples=500)
                    temp, mask = explanation.get_image_and_mask(
                        explanation.top_labels[0],
                        positive_only=True,
                        num_features=5,
                        hide_rest=False)
                    plt.imshow(mark_boundaries(temp / (255 * 2) + 0.4, mask))
                    plt.savefig("./figs/explanations/{}_{}_{}.png".format(
                        class_names[i], count + 1, regressor))
                count += 1

                if count == n_exp:
                    break
