# ECS289 Deep Learning Project: Investigation of Interpretability of Different Local Explainers for Deep Neural Network

## Introduction

This project investigates LIME [1], model explanation method, with different explainers. We implemented MyLIME to test the LIME with the logistic regression explainer and the decision tree explainer on the flower classification model. Our experiment result shows that the LIME with tree explainer outperforms the LIME with the linear explainer and the logistic explainer.

## How to start

### Installation

In order to reproduce the result, you need to git clone two repositories: This repository, and [MyLime](https://github.com/fangzhouli/mylime) repository. After cloning, perform the installation as below:
```console
pip install ./ECS289G3_DeepLearning  # This will overwrite original LIME if
                                     #   you have already installed.
pip install ./mylime
```

You are welcome to try out different models. If you want to get the exact same outcome as we have, please follow our report.

## How to use

```console
cd flower_lime
python cnn_model_training.py  # Purposely generate a bad model.
python generate_figures.py
```
## References

- [1] Marco Tulio Ribeiro, et al. 2016. "Why Should I Trust You?": Explaining the Predictions of Any Classifier.

## Contributers

Thanks for the contribution of:
- [Chengyang Wang](https://github.com/cyywang-git)
- [Xiawei Wang](https://github.com/Xiawei29)
