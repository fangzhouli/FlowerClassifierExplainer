# -*- coding: utf-8 -*-
"""

TODO:
    - Finish the arguments

"""

import cv2

from lime import lime_image
from tensorflow.keras import layers, models
from skimage.segmentation import mark_boundaries
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from animal_lime.preprocessing import crop, get_img


class BinarySmallModel(tf.keras.Model):
    """
    """

    def __init__(self):
        super().__init__()
        self.conv2D = layers.Conv2D(3, (3, 3), activation='relu')
        self.max_pool = layers.MaxPooling2D((2, 2))
        self.flat = layers.Flatten()
        self.dense_1 = layers.Dense(4)
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
    training = True
    model_name = "binary_small"
    img_size = 200
    optimizer = 'adam'
    metrics = ['accuracy']
    epochs = 10
    class_names = ['dog', 'cat']
    training_labels = {'dog': 0, 'cat': 1}

    img_train = get_img(classes=class_names, num_img=1000,
                        random_state=-1)
    X = []
    y = []
    for label, img_files in img_train.items():
        X += [crop(cv2.resize(
            cv2.imread(img_file),
            (img_size, img_size),
            interpolation=cv2.INTER_LINEAR)) for img_file in img_files]
        y += [training_labels[label] for _ in range(len(img_files))]
    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1)

    if training:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train / 255, y_train, test_size=0.25, random_state=2)

        model = BinarySmallModel()
        model.compile(optimizer=optimizer,
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(
                          from_logits=True),
                      metrics=metrics)
        model.fit(x=X_train, y=y_train, epochs=epochs,
                  validation_data=(X_val, y_val))
        model.save(model_name)

    model = models.load_model(model_name)
    explainer = lime_image.LimeImageExplainer()

    for img_test in X_test[:10]:
        score = tf.nn.softmax(model.predict(np.array([img_test]) / 255))
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
        )

        explanation = explainer.explain_instance(
            image=np.array(img_test, dtype=np.double),
            classifier_fn=model.predict,
            top_labels=1,
            hide_color=0,
            num_features=20,
            num_samples=500)

        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=5,
            hide_rest=False)
        plt.figure()
        plt.imshow(mark_boundaries(temp / (255 * 2) + 0.4, mask))

        # temp, mask = explanation.get_image_and_mask(
        #     explanation.top_labels[0],
        #     positive_only=False,
        #     num_features=10,
        #     hide_rest=False)
        # plt.figure()
        # plt.imshow(mark_boundaries(temp / (255 * 2) + 0.4, mask))

        # temp, mask = explanation.get_image_and_mask(
        #     explanation.top_labels[0],
        #     positive_only=False, num_features=1000,
        #     hide_rest=False,
        #     min_weight=0.1)
        # plt.figure()
        # plt.imshow(mark_boundaries(temp / (255 * 2) + 0.4, mask))
        plt.show()
