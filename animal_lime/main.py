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

import argparse
import sys
import textwrap
from animal_lime.models import train


def parse_args(args):
    """

    """
    parser = argparse.ArgumentParser(
        description="Train a animal image classification model")
    parser.add_argument(
        '--epochs', '-E',
        nargs=1,
        type=int,
        default=10,
        help="")
    parser.add_argument(
        '--classes', '-C',
        nargs=1,
        type=str,
        default='dog,cat',
        help="")
    parser.add_argument(
        '--n', '-N',
        nargs=1,
        type=int,
        default=1000,
        help="")
    parser.add_argument(
        '--size', '-S',
        nargs=1,
        type=int,
        default=200,
        help="")
    parser.add_argument(
        '--log-level', '-L',
        choices=[10, 20, 30, 40, 50],
        default=10,
        type=int,
        help=textwrap.dedent("""\
        The specified log level:
        - 50: CRITICAL
        - 40: ERROR
        - 30: WARNING
        - 20: INFO
        - 10: DEBUG"""))
    return parser.parse_args(args)


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])

    train(
        classes=args.classes.split(','),
        epochs=args.epochs,
        n_samples=args.n,
        img_size=args.size)
