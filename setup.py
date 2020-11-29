# -*- coding: utf-8 -*-
"""setup.py description.

This is a setup.py template for any project.

"""

from setuptools import setup, find_packages

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='flower-lime',
    version='0.1.0',
    description='',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/fangzhouli/ECS289G3_DeepLearning',
    author='Fangzhou Li',
    author_email='fzli@ucdavis.edu',
    classifiers=[
        'Development Status :: 1 - Planning',
        # 'Environment ::',
        # 'Framework ::',
        # 'Intended Audience ::',
        # 'License ::',
        # 'Natural Language ::',
        # 'Operating System ::',
        # 'Programming Language ::',
        # 'Topic ::',
    ],
    keywords='',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'lime>=0.2.0',
        'tensorflow>=2.2.0',
        'scikit-learn>=0.23.2'
    ]
)
