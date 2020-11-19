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

from sklearn.model_selection import train_test_split
import tensorflow as tf
from animal_lime.utils.image import load_files
from animal_lime.models._base import AnimalsClassifierDataGenerator
from animal_lime.models.tiny_2 import BinarySmallModel

# from lime import lime_image
# from skimage.segmentation import mark_boundaries
# import matplotlib.pyplot as plt


def train(classes, epochs, n_samples, img_size, model_name):
    """Train an animal classifier.

    Args:
        TODO

    Returns:
        TODO

    """
    # Label class with one-hot encoded numetical number.
    label_map = {}
    for i in range(len(classes)):
        label_map[classes[i]] = i

    # Define training and validation datasets.
    data_files = []
    labels = []
    data_files_dict = load_files(classes, n_samples, -1)
    for class_, files in data_files_dict.items():
        data_files += files
        labels += [label_map[class_]] * len(files)
    data_files_train, data_files_val, labels_train, \
        labels_val = train_test_split(
            data_files, labels, test_size=0.1)

    # Train the model.
    model = BinarySmallModel()
    data_generator = AnimalsClassifierDataGenerator(
        data_files=data_files,
        labels=labels,
        n_classes=len(classes),
        img_size=img_size,
        n_channels=3)
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True))
    model.fit(
        data_generator,
        epochs=epochs,
        validation_data=(data_files_val, labels_val))
    model.save(model_name)


train(['dog', 'cat'], 10, 1000, 200, 'tiny_2')

# if __name__ == '__main__':
#     """
#     """
#     training = True
#     model_name = "binary_small"
#     img_size = 200
#     optimizer = 'adam'
#     metrics = ['accuracy']
#     epochs = 10
#     class_names = ['dog', 'cat']
#     training_labels = {'dog': 0, 'cat': 1}

#     img_train = load_files(classes=class_names, num_img=1000,
#                         random_state=-1)
#     X = []
#     y = []
#     for label, img_files in img_train.items():
#         X += [center(cv2.resize(
#             cv2.imread(img_file),
#             (img_size, img_size),
#             interpolation=cv2.INTER_LINEAR)) for img_file in img_files]
#         y += [training_labels[label] for _ in range(len(img_files))]
#     X = np.array(X)
#     y = np.array(y)
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=1)

#     if training:
#         X_train, X_val, y_train, y_val = train_test_split(
#             X_train / 255, y_train, test_size=0.25, random_state=2)

#         model = BinarySmallModel()
#         model.compile(optimizer=optimizer,
#                       loss=tf.keras.losses.SparseCategoricalCrossentropy(
#                           from_logits=True),
#                       metrics=metrics)
#         model.fit(x=X_train, y=y_train, epochs=epochs,
#                   validation_data=(X_val, y_val))
#         model.save(model_name)

#     model = models.load_model(model_name)
#     explainer = lime_image.LimeImageExplainer()

#     for img_test in X_test[:10]:
#         score = tf.nn.softmax(model.predict(np.array([img_test]) / 255))
#         print(
#             "This image most likely belongs to {} with a {:.2f} percent confidence."
#             .format(class_names[np.argmax(score)], 100 * np.max(score))
#         )

#         explanation = explainer.explain_instance(
#             image=np.array(img_test, dtype=np.double),
#             classifier_fn=model.predict,
#             top_labels=1,
#             hide_color=0,
#             num_features=20,
#             num_samples=500)

#         temp, mask = explanation.get_image_and_mask(
#             explanation.top_labels[0],
#             positive_only=True,
#             num_features=5,
#             hide_rest=False)
#         plt.figure()
#         plt.imshow(mark_boundaries(temp / (255 * 2) + 0.4, mask))

#         # temp, mask = explanation.get_image_and_mask(
#         #     explanation.top_labels[0],
#         #     positive_only=False,
#         #     num_features=10,
#         #     hide_rest=False)
#         # plt.figure()
#         # plt.imshow(mark_boundaries(temp / (255 * 2) + 0.4, mask))

#         # temp, mask = explanation.get_image_and_mask(
#         #     explanation.top_labels[0],
#         #     positive_only=False, num_features=1000,
#         #     hide_rest=False,
#         #     min_weight=0.1)
#         # plt.figure()
#         # plt.imshow(mark_boundaries(temp / (255 * 2) + 0.4, mask))
#         plt.show()
