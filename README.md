# ECS289G3_DeepLearning

## How to install

If you have already installed LIME, uninstall it first.

Then, perform the installation as below:
```
pip install ./animal_lime/mylime
```

## How to use

```python
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(
    img,  # A M-by-N-by-3 array.
    model.predict,  # A classifier function
    model_regressor='logistic',
    top_labels=1,
    hide_color=0,
    num_samples=500)

print("The prediction: {}".format(class_names[explanation.top_labels[0]]))
temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=20,
    hide_rest=True)
plt.figure()
plt.imshow(img)
plt.figure()
plt.imshow(mark_boundaries(temp, mask))
plt.show()
```
