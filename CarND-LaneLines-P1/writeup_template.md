# **Finding Lane Lines on the Road** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps: 

- I convert the images to grayscale
- I smooth out the grayscale image by using a Gaussian kernel of size 3
- I detect edges by using a Canny edge detection algorithm with a low threshold of 50 and a high threshold of 180
- I apply a mask on the binary edge image
- I  use Hough transform to detect lines in the masked binary image with a distance resolution of 3 pixels, an angular resolution of 1°, a minimum intersections number of 20, a minimum number of pixels making up a line of 40 and a maximum gap of 10 pixels in the line

This first approach gave me the following results:
![alt text][test_image_output/solidWhiteCurvefirst.jpg]
More available in the folder "test_image_output" (see "...first.jpg")

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by 

- First, sorting the x,y pairs that where associated with potentialy a right line and a left line. To do so, I initialize four lists empty list, x_right, y_right, x_left and y_lest. Then for each tuple (x1,y1,x2,y2) given by the Hough transform, I computed the slope (y2-y1)/(x2-x1). If the slope was positive, I consider this line as a candidate for a right line and I append (x1, x2) to the list x_right and (y1, y2) to y_right. If the slope is negative, I append them to x_left and y_left respectively. 
- Then I try to convert the collection x_right, y_right into a single right line (and conversely x_left, y_left into a left line). To do so, I try to find the line that best describes x_right, y_right by using a least square regression of y_right vs x_right. That gives me a pair (a,b), with a line defined by y=a.x+b), that best describes the points x_right, y_right. I do the same for the left points.
- Finally, I plot the best right and left lines starting from the bottom of the image up to 60% of the height of the image.

This second approach gave me the following results:
![alt text][test_image_output/solidWhiteCurveimproved.jpg]
More available in the folder "test_image_output" (see "...improved.jpg")

### 2. Identify potential shortcomings with your current pipeline


The potential shortcomings of this approach are: 
- The definition of area of interest in the masking function. This area was defined to work on the given images but might not be applicable on other images. 
- I noticed that this approach (at least with my parameter settings) badly handles situations of individual short line (in the case of a dashed line). In this case the estimation of the best line is noisy and I notice large fluctuations frame to frame.
- Weird behaviors can arise if a line with a positive occurs in the area where left lines are expected. Such line will be an outlier and greatly influence the best left line fit.

These shortcomings are true even for "easy" images. Even more shortcomings can be pointed out in the case of more complex images like the "Challenge" one that shows that 
- sharp material transitions in the road can be deteted as lines and then disturb our pipline
- curved line like when the car is turning right or left cannot be well detected by this approach


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to filter out the outliers mentionned just above. One approach could be to define two sides left and right and when a positive slope occurs on the left side, it is not taken into account in the best line fit. 

Another potential improvement could be to encode the fact that lines are likely to smoothly evolve across the frames. Currently each frame is treated individually without consideration for the previous frame. That results into fluctuations frame to frame, fluctuations that can be potentially be large. One approach could be to replace the current least square regression by a ridge regression (using the previous best fit line) to enforce smoothness frame to frame.
