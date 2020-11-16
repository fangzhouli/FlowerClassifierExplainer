# -*- coding: utf-8 -*-
"""A preprocessing module for cropping an image.

This module contains a crop function for extracting only the center pixels of
an image.

Authors:
    Fangzhou Li - https://github.com/fangzhouli

"""


def crop(img, window_size=256):
    """Crop a given image matrix by a window.

    Args:
        img (np.array): A M*N*3 matrix with pixel intensity values.
        window_size (float): A crop window on the center of the image.

    Returns:
        (np.array::3d): A cropped image.

    """
    row = int((window_size - img.shape[0]) / 2)
    col = int((window_size - img.shape[1]) / 2)
    margin = int(len(img) * (1 - window_size) / 2)
    return img[margin:-margin, margin:-margin, :]
