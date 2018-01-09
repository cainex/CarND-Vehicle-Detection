**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[dataset]: ./output_images/dataset_sample.jpg
[hog_images]: ./output_images/hog_images.jpg
[windows]: ./output_images/test1_search_windows.jpg
[hot_windows]: ./output_images/hot_windows.jpg
[vehicle_detect]: ./output_images/vehicle_detect.jpg
[labels]: ./output_images/Video14_labels.jpg
[all_heat]: ./output_images/all_heat_maps.jpg
[final]: ./output_images/Video14_final.jpg
[video1]: ./video_out.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the create_classifier function of utils.py[lines 215-258]. This function calls the extract_features function of utils.py[lines 119-138]. 

The extract features functions takes several parameters: list of images to extract features from, colorspace conversion, spatial size (for spatial features), number of histogram bins (for color features), number of orientation bins for HOG gradient histogram, pixels per cell for HOG extraction, cells per block for HOG extraction, color channels to use for HOG extraction, and which features to extract. This function will then first perform any color space conversion specified, then extract the requested features (spatial, color, HOG) and return the feature vector for each image.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][dataset]

I then did online research and explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). Using the supplied test images, I ran the feature extraction and detection on those images to determine which parameters provided the best results with the least compute.

Here is an example using the the final chosen parameters:`YUV` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(16, 16)` and `cells_per_block=(2, 2)`:

![alt text][hog_images]

#### 2. Explain how you settled on your final choice of HOG parameters.

I first implemented a similar mechanism to analyze both single images and full video (with the option of using subclips) by reusing code from the previous project (Advanced Lane Finding). With this framework implemented, I then implemented the SVM training and classification. After this, I was able to experiment with different HOG parameters (including spatial and color features) over both the test images and full video. 

I first started with the parameters from the class lectures, and tried several color spaces including LUV, HSV, YUV and YCrCb, along with choosing to use 'ALL' channels and single channels. Having settled on the YUV colorspace, I then experimented with different orientation bins, pixels per cell and cells per block. Using color_space=YUV, orient=11, pixels_per_cell=(16,16) cells_per_block=(2,2) yeilded satisfactory results, and was my final parameter set.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Initially I started with just a single classifier, an SVM trained using the spatial, color and HOG parameters. This classifier was trained using GridSearchCV with the grid parameters of {'kernel':('linear', 'rbf'), 'C':[1,10]}.

However, the training times were very long and the inference times were also very long, limiting the number of sliding windows I was able to use for classification.

This led me to re-use the Convolutional Neural Network that I implemented in the Traffic Sign Detection project by adjusting the input image size and the final logit count to identify car vs. not-car. The convolutional network classifier worked two orders of magnitude faster (due to the use of a GPU for inferencing and training), allowing me to have many more sliding windows. However, the classifier worked a bit to well, identifying cars in more search windows than desired, and had some trouble in shadowed areas. 

With this result, I decided to create a 2-stage classifier. With the very fast inference speed of the convolutional neural network, I ran this first across the entire set of sliding windows. The resulting subset of "hot windows" were then run through the trained SVM classifier to obtain a high-confidence set of "hot windows". 

The SVM classifier is trained using the create_classifier function in utils.py [lines 216-158]. The final SVM classifier only used HOG features with color_space=YUV, orient=11, pixels_per_cell=(16,16) cells_per_block=(2,2). Using the GridSeachCV method with parameters {'kernel':('linear', 'rbf'), 'C':[1,10]}, the resulting trained SVM best parameters were {'kernel':('linear', 'rbf'), 'C':[1,10]}. These results were then saved for later use using sklearn.externals.joblib. The final classifier was pickled along with the X_scaler facture for use in later runs.

The Convolution Neural Network training was done entirely in a separate script, train_classifier.py. This script was largely pulled from my implementation of the Traffic Sign Classification project with some modifications to move from a Jupyter notebook to a stand-alone script. This classifier was then saved off using the session saving capabilities of TensorFlow to be used during vehicle detection.

Both classifiers were trained with the car/notcar dataset that was provided.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding windows were generated based on the code from the lectures. Each set of sliding windows was generated using the slide_windows function in utils.py[lines 141 to 186].

| window  | y start | y stop | overlap |
|:-------:|:-------:|:------:|:-------:|
| 64x64   |  400    |   480  |  90%    |
| 96x96   |  400    |   528  |  90%    |
| 128x128 |  400    |   560  |  90%    |
| 196x196 |  400    |   719  |  90%    |

This resulted in a total of 1834 windows, show here:

![alt text][windows]


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The sliding window search was the done on this set of windows using the search_windows_conv function in vehicles.py [lines 97 to 123]. The full set of windows is batched to the convolutional neural network for classification. This is completed very quickly using the GPU. The resulting windows which have a classification as 'car' are then passed to the SVM classifier for a second pass. All resulting 'car' classifications become the set of hot windows. Sample results are show below:

![alt text][hot_windows]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./video_out.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The full vehicle detection pipeline is show below:

![alt text][vehicle_detect]

The hot windows for each frame are saved (for a maximum of ten frames). From the positive detections I created a heatmap and then applied a threshold of 8 to that map to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from 6 consecutive frames of the project video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the frames:

#### Here are six frames and their corresponding heatmaps:

![alt text][all_heat]

#### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][labels]

#### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][final]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The challenges I faced primarily revolved around finding the best set of parameters for the HOG features, and the compute time required for both trianing and inference of the SVM classifier. When using just spatial, color and HOG features with a linear SVM, the trianing time was about 30 minutes, and the time required to render the entire project video was upwards of an hour. This also resulted in very messy results with a limited number of windows and quite a few false positives. I had some succes with tweaking the heatmap thresholds, but the results were not as good as I had expected.

Once I moved to a two-stage classifier using a convolutional neural network in conjuction with the SVM classifier using only HOG features, my results were better, and the time to render the entire project video dropped by an order of magnitude, and my final results were much cleaner.

I think a more robust system could be made by using a Region-based Convolution Neural Network (R-CNN) as described here: [Object Detection using Deep Learning for advanced users](https://medium.com/ilenze-com/object-detection-using-deep-learning-for-advanced-users-part-1-183bbbb08b19) and the newer [Fast R-CNN](https://arxiv.org/abs/1504.08083). These systems first identify regions of interest (RoI) using an object proposal algorith such as selective search, and pass those into a more traditional image classification CNN. The use of an algorithm like selective search to generate the RoIs vs. a blanket sliding window search provides much more efficient classification and better object localization. The other advantage of using a system like this, is you can easily identify different object types limited only by the CNN being used.

