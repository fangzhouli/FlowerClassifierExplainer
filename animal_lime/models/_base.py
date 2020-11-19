# -*- coding: utf-8 -*-
"""A one line summary of the module or program, terminated by a period.

This module demonstrates documentation as specified by the `Google Python
Style Guide`_. Docstrings may extend over multiple lines. Sections are created
with a section header and a colon followed by a block of indented text.

Example:
    Examples can be given using either the ``Example`` or ``Examples``
    sections. Sections support any reStructuredText formatting, including
    literal blocks::

        $ python example_google.py

Attributes:
    attribute_1 (int): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Authors:
    Fangzhou Li - https://github.com/fangzhouli

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension

"""

import cv2
import numpy as np
import tensorflow as tf
from animal_lime.utils.image import center


class AnimalsClassifierDataGenerator(tf.keras.utils.Sequence):
    """

    """

    def __init__(self, data_files, labels, n_classes, img_size, n_channels,
                 batch_size=32, shuffle=True):
        self.data_files = data_files
        self.labels = labels
        self.n_classes = n_classes
        self.img_size = img_size
        self.n_channels = n_channels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._on_epoch_end()

    def __len__(self):
        return len(self.data_files) // self.batch_size

    def __getitem__(self, index):
        indices = self._indices[index * self.batch_size:(index + 1)
                                * self.batch_size]
        files = [self.data_files[i] for i in indices]
        labels = [self.labels[i] for i in indices]
        X, y = self._generate_data(files, labels)
        return X, y

    def _on_epoch_end(self):
        self._indices = np.arange(len(self.data_files))
        if self.shuffle:
            np.random.shuffle(self._indices)

    def _generate_data(self, files, labels):
        X = np.empty(
            (self.batch_size, self.img_size, self.img_size, self.n_channels),
            dtype=np.float32)
        for i in range(len(files)):
            X[i] = center(cv2.imread(files[i]))
        return X, tf.keras.utils.to_categorical(labels, self.n_classes)

# from animal_lime.utils.image import load_files

# data_files = load_files(['dog'], 10, -1)['dog']
# data_generator = AnimalsClassifierDataGenerator(
#     data_files, [1, 1, 1, 0, 0, 1, 1, 1, 0, 1], 2, 200, 3, shuffle=False)
# data_generator[0]
