# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

General instructions
---

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

To meet specifications, the project will require submitting five files:

* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

The Project
---
The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior (we have also provided sample data that you can optionally use to help train your model.)
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires the Udacity self-driving car simulator. You can find more information about this simulator [here](https://github.com/udacity/self-driving-car-sim/blob/master/README.md). You can directly download the zip file, extract it and run the executable version of the simulator with one of the links below

Version 2, 2/07/17

[Linux](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983558_beta-simulator-linux/beta-simulator-linux.zip)
[Mac](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983385_beta-simulator-mac/beta-simulator-mac.zip)
[Windows](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/58983318_beta-simulator-windows/beta-simulator-windows.zip)


 Details About Files In This Directory
---
### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

Accomplished work
---

The most part of the accomplished work can be found in the file [model.py](./model.py). This file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.


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



### About the model

My model consists of a convolution neural network reproducing the  
[architecture of the Nvidia Self-Driving car team](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) which starts by a normalization layer followed by different convolution layers with 5x5 and 3x3 filters sizes and depths between 24 and 64 (model.py lines 82-104). The output of this convolutional stack is then flatten and fed to different fully connected  layers until a unique output is obtained (model.py lines 104-120).

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (model.py line 83).

Unlike the Nvidia model, the proposed model contains BatchNormalization layers in order to reduce overfitting. These BatchNormalization layers are applied just before the Activation function (see in model.py at lines 87, 91, 95, 99, 103, 109, 113 and 117).

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py lines 72-74 where the splitting happens). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 122).

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving (data provided by default) and two additional datasets collected by my own, a dataset of recovering from the left and right sides of the road and a dataset of driving the wrong direction.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### Solution Design Approach

The overall strategy for deriving a model architecture was a trial and error strategy.

My very first step was to use a very very simple fully connected layer with a RELU activation to ensure that the whole pipeline was working correctly.

I then moved the convolution neural network model [of the Nvidia Self-Driving car team](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) as suggested by the Udacity teaching team. This model has been proved to work particularly well on self-driving task which made me think this model might be appropriate.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by incorporating batch normalization which make the input of each activation (see the [batch normalization original paper](https://arxiv.org/pdf/1502.03167.pdf)). Batch normalization helps reducing overfitting which leads to better generalization (less gap between training and validation/test performance).

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior, I added two additional datasets of my own.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### Final Model Architecture

The final model architecture (model.py lines 82-120) consisted of a convolution neural network with the following layers and layer sizes:

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

#### Creation of the Training Set & Training Process


I first recorded two laps on track one where I tried to combine center lane driving and recovering from the left and right sides of the road back to the center so that the vehicle could learn how to react when it's not centered anymore. These images show what a recovery looks like starting from left to center:

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
 

### Results

You can see a video of the car driving autonomously in the file [successfull.mp4](./successful.mp4).

If you want to try by yourself, you can. Provided that you have on your machine the simulator as well as some necessary packages (in particular Keras, Tensorflow, eventlet and socketio), launch the simulator in autonomous mode and and run
```
python drive.py model.h5
```
in a terminal.