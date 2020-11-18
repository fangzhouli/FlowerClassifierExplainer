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


def center(img, size=200):
    """Extract the center portion of an image.

    Args:
        img (np.array): M * N * 3 pixels.
        size (int): The length of the centered image.

    Returns:
        img_centered (np.array): size * size * 3

    """
    len_expand = int(size * 1.2)
    len_margin = int(size * 0.2 / 2)
    if(img.shape[0] > img.shape[1]):
        tile_size = (int(img.shape[1] * len_expand / img.shape[0]), len_expand)
    else:
        tile_size = (len_expand, int(img.shape[0] * len_expand / img.shape[1]))
    img_resized = cv2.resize(img, dsize=tile_size)

    img_centered = np.zeros((len_expand, len_expand, 3), dtype=np.uint8)
    row = (len_expand - img_resized.shape[0]) // 2
    col = (len_expand - img_resized.shape[1]) // 2
    img_centered[row:(row + img_resized.shape[0]),
                 col:(col + img_resized.shape[1])] = img_resized
    return img_centered[len_margin:len_margin + size,
                        len_margin:len_margin + size]
