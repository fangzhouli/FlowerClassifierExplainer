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
    PATH_DATA (str): Module level variables may be documented in
        either the ``Attributes`` section of the module docstring, or in an
        inline docstring immediately following the variable.

        Either form is acceptable, but the two should not be mixed. Choose
        one convention to document module level variables and be consistent
        with it.

Authors:
    Fangzhou Li - https://github.com/fangzhouli

Todo:
    * Module docstring
    * Func docstring
    * get_img:random_state

"""

from os import listdir, path
from animal_lime.data.translate import translate

PATH_DATA = path.abspath(path.dirname(__file__)) + '/../data/raw-img'


def get_img(classes=['dog', 'cat'], num_img=1000, random_state=None):
    """TODO

    """
    for class_ in set(classes):
        if class_ not in translate.keys():
            raise ValueError("Classes contain invalid name.")

    img = {}

    for class_ in classes:
        for translated_class, v in translate.items():
            if class_ == v:
                break

        if random_state == -1:
            img_files = listdir(PATH_DATA + '/' + translated_class)
            img[class_] = [path.join(PATH_DATA, translated_class, img_file)
                           for img_file in img_files[:num_img]]
    return img
