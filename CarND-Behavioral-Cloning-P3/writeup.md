# **Behavioral Cloning**

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/cnn-architecture-624x890.png "Model Visualization"
[image2]: ./images/placeholder.png "Grayscaling"
[image3]: ./images/center_1.jpg "Recovery Image"
[image4]: ./images/center_2.jpg "Recovery Image"
[image5]: ./images/center_3.jpg "Recovery Image"
[image6]: ./images/center_3.jpg "Normal Image"
[image7]: ./images/flip_center_3.jpg "Flipped Image"
[image8]: ./images/history.png "Convergence"
[image9]: ./images/center.jpg "Center"
[image10]: ./images/right.jpg "Right"
[image11]: ./images/left.jpg "Left"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy1

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network reproducing the  
[architecture of the Nvidia Self-Driving car team](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) which starts by a normalization layer followed by different convolution layers with 5x5 and 3x3 filters sizes and depths between 24 and 64 (model.py lines 82-104). The output of this convolutional stack is then flatten and fed to different fully connected  layers until a unique output is obtained (model.py lines 104-120).

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (model.py line 83).

#### 2. Attempts to reduce overfitting in the model

The model contains BatchNormalization layers in order to reduce overfitting. These BatchNormalization layers are applied just before the Activation function (see in model.py at lines 87, 91, 95, 99, 103, 109, 113 and 117).

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 72-74 where the splitting happens). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 122).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (data provided by default) and two additional datasets collected by my own, a dataset of recovering from the left and right sides of the road and a dataset of driving the wrong direction.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was a trial and error strategy.

My very first step was to use a very very simple fully connected layer with a RELU activation to ensure that the whole pipeline was working correctly.

I then moved the convolution neural network model [of the Nvidia Self-Driving car team](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) as suggested by the Udacity teaching team. This model has been proved to work particularly well on self-driving task which made me think this model might be appropriate.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by incorporating batch normalization which make the input of each activation (see the [batch normalization original paper](https://arxiv.org/pdf/1502.03167.pdf)). Batch normalization helps reducing overfitting which leads to better generalization (less gap between training and validation/test performance).

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior, I added two additional datasets of my own.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 82-120) consisted of a convolution neural network with the following layers and layer sizes :
- Normalization layer x: x-127.5 - 1 to ensure zero mean input
- Convolution layer with 5x5 filters sizes, depth 24 and strides (2,2) (followed by BatchNormalization and RELU activation)
- Convolution layer with 5x5 filters sizes, depth 36 and strides (2,2)  (followed by BatchNormalization and RELU activation)
- Convolution layer with 5x5 filters sizes, depth 48 and strides (2,2)  (followed by BatchNormalization and RELU activation)
- Convolution layer with 3x3 filters sizes, depth 64 and strides (1,1)  (followed by BatchNormalization and RELU activation)
- Convolution layer with 3x3 filters sizes, depth 64 and strides (1,1)  (followed by BatchNormalization and RELU activation)
- Flattening layer
- A fully connected layer with 100 output units (followed by BatchNormalization and RELU)
- A fully connected layer with 50 output units (followed by BatchNormalization and RELU)
- A fully connected layer with 10 output units (followed by BatchNormalization and RELU)
- A fully connected layer with 1 output, the final output

Here is a visualization of the architecture:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

The collection of additional datasets was actually the hardest part of the project. A lag between my local computer and the remotely controlled computer on which was running the simulation made it very difficult to have a smooth and nice driving. After several trials, I finally managed to collect two additional datasets.

I first recorded two laps on track one where I tried to combine center lane driving and recovering from the left and right sides of the road back to the center so that the vehicle could learn how to react when it's not centered anymore. These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Second, I recorded one lap on track one where I was driving wrong direction to prevent the model to simply learn "by heart" how to drive on track one.

After the collection process, I had roughly 10000 data points. For each data point:

- I also flipped images and angles to mitigate the fact that on track one there is a privileged turning direction. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

- I used the center image but also left and right camera image with a correction on the steering to augment the dataset. Here are the center, right and left images:

![alt text][image9]
![alt text][image10]
![alt text][image11]


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the graph below which plots training and validation performance:

![alt text][image8]


 I used an adam optimizer so that manually training the learning rate wasn't necessary.
