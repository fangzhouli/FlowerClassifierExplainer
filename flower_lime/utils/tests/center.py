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

from os import path
import cv2
from animal_lime.utils.image import center


def test_center():
    path_dir = path.abspath(path.dirname(__file__))
    filename_img = 'OIP--1QXriWyOTJg-9fEwbznmgHaI4.jpeg'
    path_img = path_dir + '/../../data/raw-img/cane/' + filename_img

    img = cv2.imread(path_img)
    img_centered = center(img)
    cv2.imshow('img', img)
    cv2.imshow('img_resized', img_centered)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_center()
