# Car detection with YOLOv2
![](images/output_47_1.png)

```
Found 7 boxes for test.jpg
car 0.60 (925, 285) (1045, 374)
car 0.66 (706, 279) (786, 350)
bus 0.67 (5, 266) (220, 407)
car 0.70 (947, 324) (1280, 705)
car 0.74 (159, 303) (346, 440)
car 0.80 (761, 282) (942, 412)
car 0.89 (367, 300) (745, 648)
```

# Goal
- Use object detection on a car detection dataset
- Deal with bounding boxes

# File Description
- `.ipynb` file is the solution of Week 3 program assignment 1
  - Autonomous+driving+application+-+Car+detection+-+v1.ipynb
- `.html` file is the html version of `.ipynb` file.
  - Autonomous+driving+application+-+Car+detection+-+v1.html
- `.py` file
  - Autonomous+driving+application+-+Car+detection+-+v1.py
  - yolo_utils.py
- file
  - Autonomous+driving+application+-+Car+detection+-+v1.md
  
# Snapshot
- **Recommend** read `.ipynb` file via [nbviewer](https://nbviewer.jupyter.org/)
- computer view. open .html file via brower for quick look.
- brower view Autonomous+driving+application+-+Car+detection+-+v1.md


# Implementation
- Filtering with a threshold on class scores
- Non-max suppression

# Summary for YOLO
- Input image (608, 608, 3)
- The input image goes through a CNN, resulting in a (19,19,5,85) dimensional output.
- After flattening the last two dimensions, the output is a volume of shape (19, 19, 425):
  - Each cell in a 19x19 grid over the input image gives 425 numbers.
  - 425 = 5 x 85 because each cell contains predictions for 5 boxes, corresponding to 5 anchor boxes, as seen in lecture.
  - 85 = 5 + 80 where 5 is because  (pc,bx,by,bh,bw)(pc,bx,by,bh,bw)  has 5 numbers, and and 80 is the number of classes we'd like to detect
- You then select only few boxes based on:
  - Score-thresholding: throw away boxes that have detected a class with a score less than the threshold
  - Non-max suppression: Compute the Intersection over Union and avoid selecting overlapping boxes
- This gives you YOLO's final output.

# What you should remember from this notebook:
- YOLO is a state-of-the-art object detection model that is fast and accurate
- It runs an input image through a CNN which outputs a 19x19x5x85 dimensional volume.
- The encoding can be seen as a grid where each of the 19x19 cells contains information about 5 boxes.
- You filter through all the boxes using non-max suppression. Specifically:
  - Score thresholding on the probability of detecting a class to keep only accurate (high probability) boxes
  - Intersection over Union (IoU) thresholding to eliminate overlapping boxes
- Because training a YOLO model from randomly initialized weights is non-trivial and requires a large dataset as well as lot of computation, we used previously trained model parameters in this exercise. If you wish, you can also try fine-tuning the YOLO model with your own dataset, though this would be a fairly non-trivial exercise.

# Model structure 
```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
input_1 (InputLayer)             (None, 608, 608, 3)   0                                            
____________________________________________________________________________________________________
conv2d_1 (Conv2D)                (None, 608, 608, 32)  864         input_1[0][0]                    
____________________________________________________________________________________________________
batch_normalization_1 (BatchNorm (None, 608, 608, 32)  128         conv2d_1[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_1 (LeakyReLU)        (None, 608, 608, 32)  0           batch_normalization_1[0][0]      
____________________________________________________________________________________________________
max_pooling2d_1 (MaxPooling2D)   (None, 304, 304, 32)  0           leaky_re_lu_1[0][0]              
____________________________________________________________________________________________________
conv2d_2 (Conv2D)                (None, 304, 304, 64)  18432       max_pooling2d_1[0][0]            
____________________________________________________________________________________________________
batch_normalization_2 (BatchNorm (None, 304, 304, 64)  256         conv2d_2[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_2 (LeakyReLU)        (None, 304, 304, 64)  0           batch_normalization_2[0][0]      
____________________________________________________________________________________________________
max_pooling2d_2 (MaxPooling2D)   (None, 152, 152, 64)  0           leaky_re_lu_2[0][0]              
____________________________________________________________________________________________________
conv2d_3 (Conv2D)                (None, 152, 152, 128) 73728       max_pooling2d_2[0][0]            
____________________________________________________________________________________________________
batch_normalization_3 (BatchNorm (None, 152, 152, 128) 512         conv2d_3[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_3 (LeakyReLU)        (None, 152, 152, 128) 0           batch_normalization_3[0][0]      
____________________________________________________________________________________________________
conv2d_4 (Conv2D)                (None, 152, 152, 64)  8192        leaky_re_lu_3[0][0]              
____________________________________________________________________________________________________
batch_normalization_4 (BatchNorm (None, 152, 152, 64)  256         conv2d_4[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_4 (LeakyReLU)        (None, 152, 152, 64)  0           batch_normalization_4[0][0]      
____________________________________________________________________________________________________
conv2d_5 (Conv2D)                (None, 152, 152, 128) 73728       leaky_re_lu_4[0][0]              
____________________________________________________________________________________________________
batch_normalization_5 (BatchNorm (None, 152, 152, 128) 512         conv2d_5[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_5 (LeakyReLU)        (None, 152, 152, 128) 0           batch_normalization_5[0][0]      
____________________________________________________________________________________________________
max_pooling2d_3 (MaxPooling2D)   (None, 76, 76, 128)   0           leaky_re_lu_5[0][0]              
____________________________________________________________________________________________________
conv2d_6 (Conv2D)                (None, 76, 76, 256)   294912      max_pooling2d_3[0][0]            
____________________________________________________________________________________________________
batch_normalization_6 (BatchNorm (None, 76, 76, 256)   1024        conv2d_6[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_6 (LeakyReLU)        (None, 76, 76, 256)   0           batch_normalization_6[0][0]      
____________________________________________________________________________________________________
conv2d_7 (Conv2D)                (None, 76, 76, 128)   32768       leaky_re_lu_6[0][0]              
____________________________________________________________________________________________________
batch_normalization_7 (BatchNorm (None, 76, 76, 128)   512         conv2d_7[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_7 (LeakyReLU)        (None, 76, 76, 128)   0           batch_normalization_7[0][0]      
____________________________________________________________________________________________________
conv2d_8 (Conv2D)                (None, 76, 76, 256)   294912      leaky_re_lu_7[0][0]              
____________________________________________________________________________________________________
batch_normalization_8 (BatchNorm (None, 76, 76, 256)   1024        conv2d_8[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_8 (LeakyReLU)        (None, 76, 76, 256)   0           batch_normalization_8[0][0]      
____________________________________________________________________________________________________
max_pooling2d_4 (MaxPooling2D)   (None, 38, 38, 256)   0           leaky_re_lu_8[0][0]              
____________________________________________________________________________________________________
conv2d_9 (Conv2D)                (None, 38, 38, 512)   1179648     max_pooling2d_4[0][0]            
____________________________________________________________________________________________________
batch_normalization_9 (BatchNorm (None, 38, 38, 512)   2048        conv2d_9[0][0]                   
____________________________________________________________________________________________________
leaky_re_lu_9 (LeakyReLU)        (None, 38, 38, 512)   0           batch_normalization_9[0][0]      
____________________________________________________________________________________________________
conv2d_10 (Conv2D)               (None, 38, 38, 256)   131072      leaky_re_lu_9[0][0]              
____________________________________________________________________________________________________
batch_normalization_10 (BatchNor (None, 38, 38, 256)   1024        conv2d_10[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_10 (LeakyReLU)       (None, 38, 38, 256)   0           batch_normalization_10[0][0]     
____________________________________________________________________________________________________
conv2d_11 (Conv2D)               (None, 38, 38, 512)   1179648     leaky_re_lu_10[0][0]             
____________________________________________________________________________________________________
batch_normalization_11 (BatchNor (None, 38, 38, 512)   2048        conv2d_11[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_11 (LeakyReLU)       (None, 38, 38, 512)   0           batch_normalization_11[0][0]     
____________________________________________________________________________________________________
conv2d_12 (Conv2D)               (None, 38, 38, 256)   131072      leaky_re_lu_11[0][0]             
____________________________________________________________________________________________________
batch_normalization_12 (BatchNor (None, 38, 38, 256)   1024        conv2d_12[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_12 (LeakyReLU)       (None, 38, 38, 256)   0           batch_normalization_12[0][0]     
____________________________________________________________________________________________________
conv2d_13 (Conv2D)               (None, 38, 38, 512)   1179648     leaky_re_lu_12[0][0]             
____________________________________________________________________________________________________
batch_normalization_13 (BatchNor (None, 38, 38, 512)   2048        conv2d_13[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_13 (LeakyReLU)       (None, 38, 38, 512)   0           batch_normalization_13[0][0]     
____________________________________________________________________________________________________
max_pooling2d_5 (MaxPooling2D)   (None, 19, 19, 512)   0           leaky_re_lu_13[0][0]             
____________________________________________________________________________________________________
conv2d_14 (Conv2D)               (None, 19, 19, 1024)  4718592     max_pooling2d_5[0][0]            
____________________________________________________________________________________________________
batch_normalization_14 (BatchNor (None, 19, 19, 1024)  4096        conv2d_14[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_14 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_14[0][0]     
____________________________________________________________________________________________________
conv2d_15 (Conv2D)               (None, 19, 19, 512)   524288      leaky_re_lu_14[0][0]             
____________________________________________________________________________________________________
batch_normalization_15 (BatchNor (None, 19, 19, 512)   2048        conv2d_15[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_15 (LeakyReLU)       (None, 19, 19, 512)   0           batch_normalization_15[0][0]     
____________________________________________________________________________________________________
conv2d_16 (Conv2D)               (None, 19, 19, 1024)  4718592     leaky_re_lu_15[0][0]             
____________________________________________________________________________________________________
batch_normalization_16 (BatchNor (None, 19, 19, 1024)  4096        conv2d_16[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_16 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_16[0][0]     
____________________________________________________________________________________________________
conv2d_17 (Conv2D)               (None, 19, 19, 512)   524288      leaky_re_lu_16[0][0]             
____________________________________________________________________________________________________
batch_normalization_17 (BatchNor (None, 19, 19, 512)   2048        conv2d_17[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_17 (LeakyReLU)       (None, 19, 19, 512)   0           batch_normalization_17[0][0]     
____________________________________________________________________________________________________
conv2d_18 (Conv2D)               (None, 19, 19, 1024)  4718592     leaky_re_lu_17[0][0]             
____________________________________________________________________________________________________
batch_normalization_18 (BatchNor (None, 19, 19, 1024)  4096        conv2d_18[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_18 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_18[0][0]     
____________________________________________________________________________________________________
conv2d_19 (Conv2D)               (None, 19, 19, 1024)  9437184     leaky_re_lu_18[0][0]             
____________________________________________________________________________________________________
batch_normalization_19 (BatchNor (None, 19, 19, 1024)  4096        conv2d_19[0][0]                  
____________________________________________________________________________________________________
conv2d_21 (Conv2D)               (None, 38, 38, 64)    32768       leaky_re_lu_13[0][0]             
____________________________________________________________________________________________________
leaky_re_lu_19 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_19[0][0]     
____________________________________________________________________________________________________
batch_normalization_21 (BatchNor (None, 38, 38, 64)    256         conv2d_21[0][0]                  
____________________________________________________________________________________________________
conv2d_20 (Conv2D)               (None, 19, 19, 1024)  9437184     leaky_re_lu_19[0][0]             
____________________________________________________________________________________________________
leaky_re_lu_21 (LeakyReLU)       (None, 38, 38, 64)    0           batch_normalization_21[0][0]     
____________________________________________________________________________________________________
batch_normalization_20 (BatchNor (None, 19, 19, 1024)  4096        conv2d_20[0][0]                  
____________________________________________________________________________________________________
space_to_depth_x2 (Lambda)       (None, 19, 19, 256)   0           leaky_re_lu_21[0][0]             
____________________________________________________________________________________________________
leaky_re_lu_20 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_20[0][0]     
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 19, 19, 1280)  0           space_to_depth_x2[0][0]          
                                                                   leaky_re_lu_20[0][0]             
____________________________________________________________________________________________________
conv2d_22 (Conv2D)               (None, 19, 19, 1024)  11796480    concatenate_1[0][0]              
____________________________________________________________________________________________________
batch_normalization_22 (BatchNor (None, 19, 19, 1024)  4096        conv2d_22[0][0]                  
____________________________________________________________________________________________________
leaky_re_lu_22 (LeakyReLU)       (None, 19, 19, 1024)  0           batch_normalization_22[0][0]     
____________________________________________________________________________________________________
conv2d_23 (Conv2D)               (None, 19, 19, 425)   435625      leaky_re_lu_22[0][0]             
====================================================================================================
Total params: 50,983,561
Trainable params: 50,962,889
Non-trainable params: 20,672
____________________________________________________________________________________________________
```

# References
The ideas presented in this notebook came primarily from the two YOLO papers. The implementation here also took significant inspiration and used many components from Allan Zelener's github repository. The pretrained weights used in this exercise came from the official YOLO website.
- Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi - [You Only Look Once: Unified, Real-Time Object Detection (2015)](https://arxiv.org/abs/1506.02640)
- Joseph Redmon, Ali Farhadi - [YOLO9000: Better, Faster, Stronger (2016)](https://arxiv.org/abs/1612.08242)
- Allan Zelener - [YAD2K: Yet Another Darknet 2 Keras](https://github.com/allanzelener/YAD2K)
- The official YOLO website (https://pjreddie.com/darknet/yolo/)
