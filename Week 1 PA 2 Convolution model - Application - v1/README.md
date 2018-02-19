# Convolutional Neural Networks: Application
<img src="images/SIGNS.png" width=50% />

# Goal
- Implement helper functions that you will use when implementing a TensorFlow model
- Implement a fully functioning ConvNet using TensorFlow

# Data and accuracy
```
number of training examples = 1080
number of test examples = 120
X_train shape: (1080, 64, 64, 3)
Y_train shape: (1080, 6)
X_test shape: (120, 64, 64, 3)
Y_test shape: (120, 6)

print (X_train_orig.shape, Y_train_orig.shape, X_test_orig.shape, Y_test_orig.shape, classes.shape)
(1080, 64, 64, 3) (1, 1080) (120, 64, 64, 3) (1, 120) (6,)
```

```
Train Accuracy: 0.940741
Test Accuracy: 0.783333
CPU times: user 7min 53s, sys: 1min 1s, total: 8min 54s
Wall time: 5min 34s
```

# File Description
- `.ipynb` file is the solution of Week 1 program assignment 2
  - Convolution+model+-+Application+-+v1.ipynb
- `.html` file is the html version of `.ipynb` file.
  - Convolution+model+-+Application+-+v1.html
- `.py` file
  - Convolution+model+-+Application+-+v1.py
- file
  - Convolution+model+-+Application+-+v1.md
  
# Snapshot
- **Recommend** read `.ipynb` file via [nbviewer](https://nbviewer.jupyter.org/)
- computer view. open .html file via brower for quick look.
- brower view. Convolution+model+-+Application+-+v1.md


# Implementation
- create placeholders
- initialize parameters
- forward propagate
- compute the cost
- create an optimizer


# What you should remember from this notebook:
- Build and train a ConvNet in TensorFlow for a classification problem
