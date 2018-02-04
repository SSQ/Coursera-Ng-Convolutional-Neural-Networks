# Keras tutorial - the Happy House

# Goal
- Learn to use Keras, a high-level neural networks API (programming framework), written in Python and capable of running on top of several lower-level frameworks including TensorFlow and CNTK.
- See how you can in a couple of hours build a deep learning algorithm.

# File Description
- `.ipynb` file is the solution of Week 2 program assignment 1
  - Keras+-+Tutorial+-+Happy+House+v2.ipynb
- `.html` file is the html version of `.ipynb` file.
  - Keras+-+Tutorial+-+Happy+House+v2.html
- `.py` file
  - Keras+-+Tutorial+-+Happy+House+v2.py
- file
  - Keras+-+Tutorial+-+Happy+House+v2.md
  
# Snapshot
- **Recommend** read `.ipynb` file via [nbviewer](https://nbviewer.jupyter.org/)
- computer view. open .html file via brower for quick look.
- brower view. Keras+-+Tutorial+-+Happy+House+v2.md


# Implementation

```python
def model(input_shape):
    # Define the input placeholder as a tensor with shape input_shape. Think of this as your input image!
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0')(X)
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool')(X)

    # FLATTEN X (means convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X, name='HappyModel')

    return model
```

# What you should remember from this notebook:
- Keras is a tool we recommend for rapid prototyping. It allows you to quickly try out different model architectures. Are there any applications of deep learning to your daily life that you'd like to implement using Keras?
- Remember how to code a model in Keras and the four steps leading to the evaluation of your model on the test set. Create->Compile->Fit/Train->Evaluate/Test.
