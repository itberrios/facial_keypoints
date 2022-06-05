# Facial Keypoint Detection

[//]: # (Image References)


<p align="center">
  <img src='images/key_pts_example.png' width=80% height=80% />
</p>


<p align="center">
  <img src='images/landmarks_numbered.jpg' width=50% height=50% />
</p>



## Project Overview
For this project we will use the youtube faces dataset to train a model to learn 68 distinct landmarks upon a face. The trained model is then used to create some interesting face filters, using the locations of the eyes, nose, mouth, and jawline. 


## Approach

Some early experimenting with Convolutional Neural Networks (CNNs) trained from scratch revealed that this is a fairly challenging problem. The next approach was to experiment with transfer learning. After some more experimenting, Resnet50 seemed to provide the best results, Resnet101 did not seem to provide any improvement. The training data was augmented with rotations and color jitter to help the model generalize better, then the images were resized and randomly cropped to 224x224. To help the network train betterm the images were normalized with Image Net mean and standard deviation, however the keypoint locations were not normalized. Experimenting with normalized keypoints may lead to better results, but this approach was not persued. 


## Results
Some validation results from the trained model are shown below:

![image3](images/keypoint_results/result_1.PNG)
![image4](images/keypoint_results/result_2.PNG)
[image5](images/keypoint_results/result_3.PNG)

The trained model is able to identify facial keypoints in the validation set fairly well. Even when the face is not directly captured, the model is able to line up the key points with decent accuracy.

The results from a webcam capture are shown below

![image6](/images/keypoints.gif)

In this case, the model tends to extend the jawline outside of the face, but follows the face as it moves throughout the frame.

## Face filters

An basic 'snapchat' googly orange filter was built using the key points which can be seen below

![image7](/images/orange.gif)
