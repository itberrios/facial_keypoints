# Facial Keypoint Detection

[//]: # (Image References)


<p align="center">
  <img src='images/key_pts_example.png' width=80% height=80% />
</p>


<p align="center">
  <img src='images/landmarks_numbered.jpg' width=50% height=50% />
</p>



## Project Overview
This project is an extension of the Facial Keypoint project from Udacity's COmputer Vision Nanodegree. For this project we will use the youtube faces dataset to train a model to learn 68 distinct landmarks upon a face. The trained model is then used to create some interesting face filters, using the locations of the eyes, nose, mouth, and jawline. 


## Approach

Some early experimenting with Convolutional Neural Networks (CNNs) trained from scratch provided meager results and revealed that this is a fairly challenging problem. The next approach was to experiment with transfer learning. After some more experimenting, Resnet50 seemed to provide the best results, Resnet101 did not seem to provide any improvement. The only type of model used was a resnet wince it was able to provide adequate results on validation data and out of sample data from a webcam capture. The final resnet had it's fully connected layer completly replaced with 136 outputs for each of the keypoints. During training, layer 4, avg pool, and the fully connected layer were the only ones updated. Experimenting with other layer updates didn't seem to increase or decrease performance on the validation set.


The training data was augmented with rotations and color jitter to help the model generalize better, then the images were resized and randomly cropped to 224x224. To help the network train better, the images were normalized with Image Net mean and standard deviation, however the keypoint locations were not normalized. Experimenting with normalized keypoints may lead to better results and/or faster training, but this approach was not persued. 

The first 10 epochs of training were on lightly augmented data (i.e. minor color jitter, minimal random rotations, resize and crop transforms unlikely to remove portions of faces).

The final 50 epochs were trained on more heavily augmented data (i.e. larger color jitter, +/1 180 degree random rotations, resize and crop transforms likely to remove portions of faces).

Future work could involve fine tuning the model on some even more heavily augmented data. Image blurring and Image warping (along with keypoint transformation) would be good additions to the already augmented data. Another approach would be to modify the pipeline by first using a Haar face detector to crop the face from each image, and then apply the facial keypoint model to the cropped image. This would provide a more consistent distribution/space/domain for the model to learn instead of having to learn keypoints at a variety of distances. This consistent domain for the model to learn may increase accuracy and consistency (i.e. less random flucuations of keypoint locations). However the pipeline now involves a Haar face detector that requires parameter tuning. An alternative approach would be to use the opencv2 Face dnn instead.

Another simple update to the pipline would be to add histogram equaliztion (via YUV color space) to better account for different lighting conditions.


## Results
Some validation results from the trained model are shown below:

![image3](images/keypoint_results/result_1.PNG)
![image4](images/keypoint_results/result_2.PNG)
![image5](images/keypoint_results/result_3.PNG)
![image7](images/keypoint_results/result_4.PNG)
![image7](images/keypoint_results/result_5.PNG)
![image8](images/keypoint_results/result_6.PNG)
![image9](images/keypoint_results/result_7.PNG)
![image10](images/keypoint_results/result_8.PNG)


The truth keypoints are in green and the predicted keypoints are in purple

The trained model is able to consistently identify facial keypoints in the validation images under a variety of lighting conditions. Even when the face is not directly captured, the model is able to line up the key points with decent accuracy, although it isn't very consistent with this. 

The results from a webcam capture are shown below

![image11](images/keypoints.gif)

In this case, the model tends to extend the jawline outside of the face, but follows the face as it moves throughout the frame. The glasses don't seem to have an effect on the model as it seems like it can find the eyes whether they are present or not. 

## Face filters

An basic 'snapchat' filter of a googly orange was built using the key points which can be seen below

![image12](images/orange.gif)


#### Sources
- Getting Started: https://github.com/udacity/P1_Facial_Keypoints
- Transfer Learning: https://debuggercafe.com/advanced-facial-keypoint-detection-with-pytorch/
