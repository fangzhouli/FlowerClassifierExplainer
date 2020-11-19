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

from tensorflow.keras import Model, layers
import tensorflow as tf


class BinarySmallModel(Model):
    """
    """

    def __init__(self):
        super().__init__()
        self.conv2D = layers.Conv2D(32, (3, 3), activation='relu',
                                    padding='same',
                                    input_shape=(200, 200, 3))
        self.max_pool = layers.MaxPooling2D((2, 2))
        self.flat = layers.Flatten()
        self.dense_1 = layers.Dense(10, activation='relu')
        self.dense_2 = layers.Dense(2, activation='softmax')

    def call(self, inputs):
        # print(inputs)
        print(inputs.shape)
        x = self.conv2D(inputs)
        # print(x.shape)
        x = self.max_pool(x)
        x = self.flat(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x
