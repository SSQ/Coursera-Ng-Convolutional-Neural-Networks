# Residual Networks

# Goal
- Implement the basic building blocks of ResNets.
- Put together these building blocks to implement and train a state-of-the-art neural network for image classification.

# File Description
- `.ipynb` file is the solution of Week 2 program assignment 2
  - Residual+Networks+-+v2.ipynb
- `.html` file is the html version of `.ipynb` file.
  - Residual+Networks+-+v2.html
- `.py` file
  - Residual+Networks+-+v2.py
- file
  - Residual+Networks+-+v2.md
  
# Snapshot
- **Recommend** read `.ipynb` file via [nbviewer](https://nbviewer.jupyter.org/)
- computer view. open .html file via brower for quick look.
- brower view. Residual+Networks+-+v2.md


# Implementation
- The identity block
- The convolutional block
- Building your first ResNet model (50 layers)


# What you should remember from this notebook:
- Very deep "plain" networks don't work in practice because they are hard to train due to vanishing gradients.
- The skip-connections help to address the Vanishing Gradient problem. They also make it easy for a ResNet block to learn an identity function.
- There are two main type of blocks: The identity block and the convolutional block.
- Very deep Residual Networks are built by stacking these blocks together.
