## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration1result.jpg "After/Before distortion"
[image2]: ./output_images/test4undistortion.jpg "After/Before distorsion - road picture"
[image3]: ./output_images/test1binary.jpg "Binary Example"
[image35]: ./source_points.png "Source points"
[image4]: ./output_images/straight_lines1bird.jpg "Warp Example"
[image5]: ./output_images/find_lane_without_prior.jpg "Lane search without prior"
[image6]: ./output_images/find_lane_with_prior.jpg "Lane search with prior"
[image7]: ./output_images/test1final.jpg "Annotated image"
[video1]: ./project_video.mp4 "Video"



## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is the Section 1 the IPython notebook located in "Advanced-line-finding-project.ipynb".  

In subsection "Collect data", I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection. 

In subsection, "Computation of camera matrix and distorsion coefficients", I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  

Finally, in subsection "Test image calibration" I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this kind of results.

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.


Once the Camera Calibration is done, I wrap up it in function in the subsection "Save camera matrix and distortion coefficients / undistortion method definition". To demonstrate it, I apply the distortion correction to one of the test images like this one:

![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In the section Section "Binary image pipeline" I define a pipeline to generate a binary image. I start by defining in the subsection "Useful functions" several useful functions to compute the gradient in a single direction, the magnitude of the bidirectionnal gradient as well as its direction. I used in the function `binary_pipeline` a combination of color and gradient thresholds to generate a binary image in subsection "Pipeline for binarising an image".  In more details, I applied a color thresholding on the saturation chanel and a thresholding on the magnitude/direction of the gradient of the lightness chanel.

In subsection "Testing the pipeline", I define the different thresholds and I test my pipeline on the test images. Here's an example of my output for this step:

![alt text][image3]

On the left, the original image, in the middle the binary output and the right the contributions of the gradient tresholding on the lightness channel (blue) and the thresholding on the saturation channel (green).

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

In the third section "Perspective transform, I compute the perspective transform matrix. I start by defining a set of source points (`src`) and a set of destination (`dst`) points

```python
src = np.float32([[527, 470],
                  [757, 470],
                 [1250, 690],
                  [50, 690]
                 ])
offset, img_size = 0, (raw.shape[1], raw.shape[0])

dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 527, 470      | 0, 0          | 
| 757, 470      | 1280, 0       |
| 1250, 690     | 1280, 720     |
| 50, 690       | 0, 720        |

These points are as following: 

![alt text][image35]

Using these points I compute the transform matrix `M` and its inverse `Minv` that I save for later use.

In subsection "Visual checking of the transform", I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In the fourth Section "Line finding pipeline", I define an intermediary pipeline that takes a binary image from a bird's eye perspective and provide a binary image where the lane lines stand out clearly. This pipeline is defined in the function `find_lane`. This function also takes a `prior` inputs which defines the way of operating the search line. Namely:

- `prior` is `None`: Seach from scratch using a histogram and sliding window method
- `prior` is a dictionnary: Search the line using the prior line to narrow down the search area

It then provides the following outputs
- `results` dictionnary with keys
    - `fit`: contains relevant information about the fit of the left and right lines (coefficients)
    - `road`: contains revelant information about the road (its curvature, its width, the position of the car)
- `visualization` 3 chanels image for visualization

Here are the lane search without prior:

![alt text][image5]

and with prior:

![alt text][image6]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in the function `find_lane` through the two subfunctions:

```python
def fit_polynomial(img_shape, lefty, leftx, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    fit_line = {}
    fit_line['left_fit'], fit_line['right_fit'] = left_fit, right_fit
    fit_line['left_fitx'], fit_line['right_fitx'] = left_fitx, right_fitx
    fit_line['ploty'] = ploty
    return fit_line
    
def curvature_in_meters(ploty, fit):
    A = xm_per_pix*fit[0]/(ym_per_pix**2)
    B = xm_per_pix*fit[1]/ym_per_pix
    y_eval = np.max(ploty)
    curvature = ((1 + (2*A*y_eval + B)**2)**1.5) / np.absolute(2*A)
    return curvature 
```
that respectively performs a polynomial fit and computes the curvature of the line.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


In Section "Final pipeline", I implemented the final pipeline that goes from the raw image to the lane aera identification in the function `process_image`. This function uses all the previous blocks, namely undistortion, thresholding, bird's perspective transform and line detection as well as the class `Line()` that will enable us to receive the characteristics of each line detection. This class is defined by

```python
class Line():
    def __init__(self, n = 15):
        #was the line detected in the last iteration?
        self.detected = False  
        #polynomial coefficients over the last n iterations (NOT AVERAGED)
        self.previous_fits = deque(maxlen=n)
        #polynomial coefficients for the most recent fit
        self.current_fit = None  
        #Number of frames where the algorithms failed
        self.fail = 0
        #Previous estimated curvature
        self.previous_curvatures = deque(maxlen=n)
        #Previous estimated offset
        self.previous_offsets = deque(maxlen=n)
```

In this function `process_image`, starting from line 19 till line 30, I added some logic to define a prior that makes sense. Using a right and left Line object, I check if lines were detected on the previous frame, I use them to define my prior and narrow down the search. If not, I used previous successfull (at most 15) that I average to define my prior. If no successful fit are kept in memory in the Line objects, I start a fresh search line without prior. This is implemented as follows:

```python
if rightline.detected & leftline.detected : # If lines were detected on the previous frame
        prior = {} # Then use the previous parabola to narrow down the search of the pixels 
        prior['left_fit'], prior['right_fit'] = leftline.current_fit, rightline.current_fit # 
    else : # If not
        if len(leftline.previous_fits)==0 & len(rightline.previous_fits)==0: # If no detection was successful before
            prior  = None # Perform a line search from scratch using the sliding window method
        else: # If detections were successful before, use the average fit as prior
            averaged_left_fit = np.mean(np.array(leftline.previous_fits), axis = 0)
            averaged_right_fit = np.mean(np.array(rightline.previous_fits), axis = 0)
            prior = {}
            prior['left_fit'], prior['right_fit'] = averaged_left_fit, averaged_right_fit
```

Once line were detected,  I perform a sanity check from line 45 to 57 as follows:

```python
#B- SANITY CHECK 
    #- Right and left lanes have similar curvature
    threshold_curvature = 6000
    delta_curvature = abs(left_curvature-right_curvature)
    ok_curvature = delta_curvature < threshold_curvature
    #- Good distance between the two lines
    threshold_distance = 2
    delta_distance = abs(distance-3.7)
    ok_distance = delta_distance < threshold_distance
    #- Parallel 
    ok_parallel = True
    #- Final check 
    check = ok_curvature & ok_distance & ok_parallel
```

where I check that curvature of left and right line are not too different, that they yield a road width that makes sense. I didn't implement a parallelism checking. 

Once this sanity check is done, I update this line objects accordingly as follows:

```python
if check:
        #Left line
        leftline.detected = True
        leftline.previous_fits.append(left_fit)
        leftline.current_fit = left_fit
        leftline.previous_curvatures.append(left_curvature)
        leftline.previous_offsets.append(offset)
        
        #Right line
        rightline.detected = True
        rightline.previous_fits.append(right_fit)
        rightline.current_fit = right_fit     
        rightline.previous_curvatures.append(right_curvature)
        rightline.previous_offsets.append(offset)
    else: 
        leftline.detected = False
        rightline.detected = False
        leftline.fail +=1
        rightline.fail +=1
        
        if leftline.fail > fail_max:
            leftline.previous_fits = deque(maxlen=ndeque)
            rightline.previous_fits = deque(maxlen=ndeque)
            leftline.previous_curvatures = deque(maxlen=ndeque)
            rightline.previous_curvatures = deque(maxlen=ndeque)
            leftline.previous_offsets = deque(maxlen=ndeque)
            rightline.previous_offsets = deque(maxlen=ndeque)
            leftline.fail = 0
            rightline.fail = 0
```            
            
The rest of the code takes care of annotating the result frame using averaged fits and averaged curvatures. An example of output is given below  


![alt text][image7]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The implemented pipeline seems to work fine on the project video, although I noticed a gap between the expected curvature and the estimated one. However this pipeline seems to fail on more complex video, as it is the case on challenge videos where vehicules crossing the road, change in road material and shadow seem to disturb the pipeline.
