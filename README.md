# BasketBall Dunk Action Recognition 
The basketball dunk action recognition is to detect whether a action in a image or video is a dunk or not. It is based on the object detection with classical computer vision features.  It could used as a baseline algorithm.

## Table of content

- [Installation](#installation)
- [Dataset](#dataset)
- [Algorithm Outline](#algorithm-flow)
- [Result](#result)

## Installation

- Numpy
- Python3.5 or more
- Opencv 3.4 or more
- Sklearn

#Datasets

used for training and testing:

- KTH human activity - Boxing, Hand clapping, Running, Walking
- Weizmann - Bending, One hand waving


Methodology:
------------
Method-1 : 
HOG feature vecots from n consecutive video frames are analyzed to generate HOGPH (history of HOG features over past frames). HOGPH feature vectors are used to train the multi-class SVM classifier model for all activities.
For testing, HOGPH vector is generated for each sample video and SVM used for prediction of the class.

Method-2 : Video Matching using PCA and SVD

Results:
--------
Method 1 does not account for motion information and not suitable for large dataset.
Method 2, on the other hand, is easy to implement but time consuming. Method 2 accuracy found out to be around 63%. 
           


## Algorithm Overview

Method-1 : 
HOG feature vecots from n consecutive video frames are analyzed to generate HOGPH (history of HOG features over past frames). HOGPH feature vectors are used to train the multi-class SVM classifier model for all activities.
For testing, HOGPH vector is generated for each sample video and SVM used for prediction of the class.

- Detect balls and the basket and the person around  contour edge based on the histogram of oriented gradients (HOG) feature and Haar feature
 

- Use object detection training in cascade trainer to detect balls, baskets, person

- Use the postion of ball, basketball and person position to decide it is a dunk or not


Method-2 : Video Matching using PCA and SVD


Results:

Method 1 does not take into account the motion information and not suitable for large dataset in videos.

Method 2, on the other hand, is easy to implement but time consuming. Method 2 accuracy found out to be around 63%. 
           
