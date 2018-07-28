# **Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./class_diversity.jpg "Class diversity"
[image2]: ./sample_diversity.jpg "Sample diversity"
[image3]: ./class_balancing.jpg' "Class balancing"
[image4]: ./data_augmentation.jpg "Original data and fake data"
[image5]: ./balanced_classes.jpg "Class distribution after oversampling"
[image6]: ./sign0_.jpg "Web image 1"
[image7]: ./sign1_.jpg "Web image 2"
[image8]: ./sign2_.jpg "Web image 3"
[image9]: ./sign3_.jpg "Web image 4"
[image10]: ./sign4_.jpg "Web image 5"
[image11]: ./sign0_prediction.jpg "Web image 1"
[image12]: ./sign1_prediction.jpg "Web image 2"
[image13]: ./sign2_prediction.jpg "Web image 3"
[image14]: ./sign3_prediction.jpg "Web image 4"
[image15]: ./sign4_prediction.jpg "Web image 5"

---
### Writeup / README

### Data Set Summary & Exploration

#### 1. Basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799?
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


#### 2. Include an exploratory visualization of the dataset.

I explored the data regarding three aspects:
- The variability between classes by plotting a sample for each class
![alt text][image1]
- The variability within a class by plotting several sample for a given class
![alt text][image2]
- The class distribution across training, validation and testing dataset
![alt text][image3]

Based on the plots above, we can say that:

- Classes have the same distribution across the different datasets. In other words, we are training our model on the same kind of dataset that we will evaluate it on.
- Some classes, in particular the ones between 0 and 10, such as the speed limit signs, have more samples than others. Training our model on such unbalanced dataset could lead to better performance on certain classes that other. This can be tackled using techniques such as oversampling.
- For a given class, samples seem to show a variety of resolution, lightness but less rotation. To fix this I used data augmentation to generate fake data. However, we can note that some techniques in data augmentation would not make sense on our particular dataset. For example, using a horizontal flip would not help and even worse could harm our classifier since it can change the class of the sample (see for example "Go straight or right" sign that would become a "Go straight or left"). We will therefore only consider mild rotation with zoom.



### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)


I decided to stick with the RGB color space as I wanted to keep the color information.

I added a simple and quick normalization step which consists in using the transform x-> (x-128)/128 to normalize each chanel between -1 and 1.

I also decided to generate additional data to tackle two of the issues I mentionned above.

- Data augmentation with mild rotation to help the model better learn. Here is an example of a traffic sign image after such rotation.
![alt text][image4]

- Using oversampling (`imblearn` library) to obtain balanced class distribution
![alt text][image5]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I decided to keep the LeNet-5 architecture as it seemed to work fine on my first test. I also wanted to see how additionnal ingredients, rather than change in the architecture, could help to improve the accuracy. My final model is very close to the original one, but I added a preprocessing layer and two types of layer:

- dropout layers for regularization (hopefully will reduce overfitting and help to the generalization) using [`tf.layers.dropout`](https://www.tensorflow.org/api_docs/python/tf/layers/dropout)
- Batch-normalization layers introduced [in the paper](https://arxiv.org/pdf/1502.03167.pdf). This idea is to normalize the inputs to layers within the network. I will use for that the high-level API provided by tensorflow [`tf.layers.batch_normalization `](https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization)


| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Normalization  		| x->(x-128)/128 on each chanel     			|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| Batch-normalization 	|												|
| RELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| Batch-normalization 	|												|
| RELU					|												|
| Dropout				|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16    				|
| Flatten	         	| outputs 400                    				|
| Fully connected		| outputs 120, no activation     				|
| Batch-normalization 	|												|
| RELU					|												|
| Dropout				|												|
| Fully connected		| outputs 84, no activation     				|
| Batch-normalization 	|												|
| RELU					|												|
| Dropout				|												|
| Fully connected		| outputs 43, noactivation                      |
| Softmax				| 												|



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used first the Adam optimizer but I noticed large fluctuation of the validation accuracy during the training, I for instance could get 0.92 and suddenly got 0.8. This made the tuning of other parameters quite challenging. I then moved to the Adagrad optimizer which solved this issue. However I notice that this optimizer was a little bit slower, I therefore increased the learning rate. My final setting is as follows:
- Adagrad optimizer
- learning rate 0.05
- batch size 128
- epochs 10

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.986
* validation set accuracy of 0.94
* test set accuracy of 0.937


If a well known architecture was chosen:
* What architecture was chosen?
As I mentioned earlier, I used the LeNet-5 architecture as a starting since I wanted to see how other elements such as dropout and batch-normalization could help a neural network to learn. Also, since I was able to get almost 89% of accuracy with this model at the beginning, I was confident in this model.

If an iterative approach was chosen:
At first I went for L2 regularization to avoid overfitting. Using L2 regularization was not satisfactory in my case, I had hard time seeing the benefits of it and tuning it. I then decided to move to dropout layers.

Without dropout layers, but same architecture, my results were:
 - Train accuracy: 1.000
 - Validation accuracy: 0.928
 - Test accuracy 0.920
which is illustrative of the fact that there is overfitting (it is also illustrative of the benefits of batch-normalization layers).

I then tried several values of dropout probability. Using 0.3 I was able to reduce the gap between train and validation accuracy. Higher values were too penalizing for the network.

Overall, without changing the architecture in depth, I was able to increase the accuracy of 5%.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web. I changed their resolution to 32x32.

![alt text][image6] ![alt text][image7] ![alt text][image8]
![alt text][image9] ![alt text][image10]

The web image quality seems to be similar to the train image quality, I therefore expected the model to be successful.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| Stop sign   									|
| Right of way at the.. | Right of way at the..      					|
| No entry  			| No entry  									|
| General caution  		| No passing					 				|
| Priority road			| Priority road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares defavorably to the accuracy on the test set of 0.937.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located at the end of the Ipython notebook.

For the Stop Sign / Right of way at the next intersection / No entry / Priority road, the model is pretty confident in its prediction (probability close to 1) as illustrated below:

![alt text][image11]
![alt text][image12]
![alt text][image13]
![alt text][image15]

For the fourth image, General caution, for reasons I don't understand, the model fails to predict it's a general caution sign. However, the model is not confident at all in its prediction as you can see below:

![alt text][image14]
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
