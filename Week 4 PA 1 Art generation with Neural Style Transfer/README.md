# Deep Learning & Art: Neural Style Transfer
<img src="images/louvre_generated.png" width=50% />

# Goal
- Implement the neural style transfer algorithm
- Generate novel artistic images using your algorithm

# File Description
- `.ipynb` file is the solution of Week 4 program assignment 1
  - Art+Generation+with+Neural+Style+Transfer+-+v2.ipynb
- `.html` file is the html version of `.ipynb` file.
  - Art+Generation+with+Neural+Style+Transfer+-+v2.html
- `.py` file
  - Art+Generation+with+Neural+Style+Transfer+-+v2.py
  - nst_utils.py
- file
  - Art+Generation+with+Neural+Style+Transfer+-+v2.md
 
# Snapshot
- **Recommend** read `.ipynb` file via [nbviewer](https://nbviewer.jupyter.org/)
- computer view. open .html file via brower for quick look.
- brower view Art+Generation+with+Neural+Style+Transfer+-+v2.md

# How to implement Neural Style Transfer!

1. Create an Interactive Session
1. Load the content image
1. Load the style image
1. Randomly initialize the image to be generated
1. Load the VGG16 model
1. Build the TensorFlow graph:
    - Run the content image through the VGG16 model and compute the content cost
    - Run the style image through the VGG16 model and compute the style cost
    - Compute the total cost
    - Define the optimizer and the learning rate
7. Initialize the TensorFlow graph and run it for a large number of iterations, updating the generated image at every step.


# What you should remember:

- The content cost takes a hidden layer activation of the neural network, and measures how different $a^{(C)}$ and $a^{(G)}$ are. 
- When we minimize the content cost later, this will help make sure $G$ has similar content as $C$.
  
- The style of an image can be represented using the Gram matrix of a hidden layer's activations. However, we get even better results combining this representation from multiple different layers. This is in contrast to the content representation, where usually using just a single hidden layer is sufficient.
- Minimizing the style cost will cause the image $G$ to follow the style of the image $S$. 

- The total cost is a linear combination of the content cost $J_{content}(C,G)$ and the style cost $J_{style}(S,G)$
- $\alpha$ and $\beta$ are hyperparameters that control the relative weighting between content and style

- Neural Style Transfer is an algorithm that given a content image C and a style image S can generate an artistic image
- It uses representations (hidden layer activations) based on a pretrained ConvNet.
- The content cost function is computed using one hidden layer's activations.
- The style cost function for one layer is computed using the Gram matrix of that layer's activations. The overall style cost function is obtained using several hidden layers.
- Optimizing the total cost function results in synthesizing new images.

# Statement
Sentences are extracted and reorganized from the original programming assignment in Coursera just for quick search
