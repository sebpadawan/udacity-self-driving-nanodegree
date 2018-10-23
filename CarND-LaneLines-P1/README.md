# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

<img src="examples/laneLines_thirdPass.jpg" width="480" alt="Combined Image" />


[//]: # (Image References)

[image1]: ./test_images_output/solidWhiteCurvefirst.jpg "Initial pipeline"
[image2]: ./test_images_output/solidWhiteCurveimproved.jpg "Improved pipeline"



General instructions
---

When we drive, we use our eyes to decide where to go.  The lines on the road that show us where the lanes are act as our constant reference for where to steer the vehicle.  Naturally, one of the first things we would like to do in developing a self-driving car is to automatically detect lane lines using an algorithm.

In this project you will detect lane lines in images using Python and OpenCV.  OpenCV means "Open-Source Computer Vision", which is a package that has many useful tools for analyzing images.  

To complete the project, two files will be submitted: a file containing project code and a file containing a brief write up explaining your solution. We have included template files to be used both for the [code](https://github.com/udacity/CarND-LaneLines-P1/blob/master/P1.ipynb) and the [writeup](https://github.com/udacity/CarND-LaneLines-P1/blob/master/writeup_template.md).


Accomplished work
---

Results and code referenced below can be found in the companion jupyter notebook [P1.ipynb](./P1.ipynb).


### Implemented pipeline

My pipeline consisted of 5 steps: 

- I convert the images to grayscale
- I smooth out the grayscale image by using a Gaussian kernel of size 3
- I detect edges by using a Canny edge detection algorithm with a low threshold of 50 and a high threshold of 180
- I apply a mask on the binary edge image
- I  use Hough transform to detect lines in the masked binary image with a distance resolution of 3 pixels, an angular resolution of 1°, a minimum intersections number of 20, a minimum number of pixels making up a line of 40 and a maximum gap of 10 pixels in the line

This first approach gave me the following results:

![alt text][image1]

More available in the folder [test\_images\_output](./test_images_output) with images ending with 'first'.

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by 

- First, sorting the x,y pairs that where associated with potentialy a right line and a left line. To do so, I initialize four lists empty list, x\_right, y\_right, x\_left and y\_lest. Then for each tuple (x1,y1,x2,y2) given by the Hough transform, I computed the slope (y2-y1)/(x2-x1). If the slope was positive, I consider this line as a candidate for a right line and I append (x1, x2) to the list x\_right and (y1, y2) to y\_right. If the slope is negative, I append them to x\_left and y\_left respectively. 
- Then I try to convert the collection x\_right, y\_right into a single right line (and conversely x\_left, y\_left into a left line). To do so, I try to find the line that best describes x\_right, y\_right by using a least square regression of y\_right vs x\_right. That gives me a pair (a,b), with a line defined by y=a.x+b), that best describes the points x\_right, y\_right. I do the same for the left points.
- Finally, I plot the best right and left lines starting from the bottom of the image up to 60% of the height of the image.

This second approach gave me the following results:

![alt text][image1]

More available in the folder [test\_images\_output](./test_images_output) with images ending with 'improved'.


### Limits of the pipeline

The potential shortcomings of this approach are: 

- The definition of area of interest in the masking function. This area was defined to work on the given images but might not be applicable on other images. 
- I noticed that this approach (at least with my parameter settings) badly handles situations of individual short line (in the case of a dashed line). In this case the estimation of the best line is noisy and I notice large fluctuations frame to frame.
- Weird behaviors can arise if a line with a positive occurs in the area where left lines are expected. Such line will be an outlier and greatly influence the best left line fit.

These shortcomings are true even for "easy" images. Even more shortcomings can be pointed out in the case of more complex images like the "Challenge" one that shows that 

- sharp material transitions in the road can be deteted as lines and then disturb our pipline
- curved line like when the car is turning right or left cannot be well detected by this approach


### Possible improvements 

A possible improvement would be to filter out the outliers mentionned just above. One approach could be to define two sides left and right and when a positive slope occurs on the left side, it is not taken into account in the best line fit. 

Another potential improvement could be to encode the fact that lines are likely to smoothly evolve across the frames. Currently each frame is treated individually without consideration for the previous frame. That results into fluctuations frame to frame, fluctuations that can be potentially be large. One approach could be to replace the current least square regression by a ridge regression (using the previous best fit line) to enforce smoothness frame to frame.
