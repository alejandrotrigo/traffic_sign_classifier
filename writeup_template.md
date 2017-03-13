#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: ./examples/chart.png "Visualization"
[image2]: ./examples/processed.png "Grayscaling"
[image3]: ./examples/augmentated.jpg "Augmentated"
[image4]: ./examples/g1.png "Traffic Sign 1"
[image5]: ./examples/g2.png "Traffic Sign 2"
[image6]: ./examples/g3.png "Traffic Sign 3"
[image7]: ./examples/g4.png "Traffic Sign 4"
[image8]: ./examples/g5.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the second code cell of the IPython notebook.  

I used some basic functions of python to get the number of training and test images, their shapes and the total number of different traffic signs.

At this point I knew these information about the dataset:

* The size of training set is 34799
* The size of validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in the third code cell of the IPython notebook.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed. We can see that there are some classes with less than 250 examples and others with more than 1500 examples. 

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in the fourth code cell of the IPython notebook.

As a first step, I decided to convert the images to grayscale because it seems to deal better with images of only 1 of depth
After that I dealed with the brightness with a equalizeHist function.
As a last step, I normalized the image data because the shapes were not fitting anymore because of the conversion to grayscale.

Here is an example of a traffic sign image after the preprocessing.

![alt text][image2]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

In the beggining I was ussing only the provided data in the dataset, but the test accuracy was pretty low, so  I realised that it was not enought for many classes, because of that I decided to augmentate the data by applying a random shift and a random rotation.

My final training set had 106226 number of images. My validation set and test set had 22110 and 12630 number of images.

Here is an example of an augmented image:

![alt text][image3]

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the seventh cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer					| Description									| 
|:---------------------:|:---------------------------------------------:| 
| Input 				| 32x32x1 grayscale image   					| 
| Convolution 1x1		| 1x1 stride, same padding, outputs 32x32x1 	|
| Convolution 5x5		| 5x5 stride, same padding, outputs 28x28x16	|
| relu 					| 												|
| pooling 2x2			| 2x2 kernel, 2x2 strides, output 14x14x16		|
| Convolution 5x5		| 5x5 stride, same padding, outputs 10x10x32	|
| relu   				| 												|
| pooling 2x2			| 2x2 kernel, 2x2 strides, output 5x5x32		|
| fully connected		| flatten to 800								|
| dropout				| 0.8 probability								|
| fully connected		| 128 neurons, relu								|
| fully connected		| 128 neurons, relu								|
| dropout				| 0.5 probability								|
| fully connected		| 43 neurons, relu								|
| softmax				|												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the eigth cell of the ipython notebook. 

To train the model, I used an AdamOptimizer with a learning rate of 0.001, 200 epochs and a batch size of 256. 

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:

* validation set accuracy of 93%
* test set accuracy of 95%

If an iterative approach was chosen:
* The first architecture I've chosen was the LeNet, It was easy because we had the LeNet exercise
* It was not good generalizing, I was reaching better performance in the validation set than in the test set
* The process is mostly of try and error, I've been reading a lot about convnets and I tried some of the iformation I read.One thing is the first conv layer. I also tried to make the net deeper by adding extra fully connected layers and two dropouts to prevent overfitting.
* I changed manny times the dropouts probability, the number of epochs and the learning rate, I also added dropouts but they were not helping.
* One of the most relevant changes I made was to add the first conv layer it was making the net generalize much better, also the dropouts. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first, second, third and fourth images might be easy to classify because the sign over they is not vissible and there is no stange stuff in the images.
The fifth could be more difficult because we can see more than one sign in the same image

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image											|     Prediction	        					| 
|:---------------------------------------------:|:---------------------------------------------:| 
| Speed Limit 20   								| Speed Limit 20								| 
| Double curve 									| Double curve									|
| No passing									| No passing									|
| End of no passing by vehicles over 3.5 tons	| End of no passing by vehicles over 3.5 tons	|
| Turn right ahead								| Slippery Road      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 95%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is completely sure that this is a Speed Limit (20) (probability about 1.0), and the image does contain a Speed Limit (20) sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Stop sign   									| 
| .0     				| 		 										|
| .0					| 												|
| .0	      			| 								 				|
| .0				    | 												|



For the second image, the model is not really sure that this is a Double curve (probability about 0.3), and the image does contain a  Double curve sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.3         			| Double curve 									| 
| 0.2     				| Right-of-way at the next intersection			|
| 0.2					| Wild animals crossing							|
| 0.1	      			| Roundabout mandatory			 				|
| 0.2				    | Others										|


For the third image, the model is completely sure that this is a No passing (probability about 0.9), and the image does contain a No passing sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9         			| No passing   									| 
| 0.1     				| Others		 								|
| 0.0					| 												|
| 0.0	      			| 								 				|
| 0.0				    | 												|


For the fourth image, the model is quite sure that this is a  End of no passing by vehicles over 3.5 tons (probability about 0.6), and the image does contain a End of no passing by vehicles over 3.5 tons sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.6         			| End of no passing by vehicles over 3.5 tons	| 
| 0.2     				| End of no passing								|
| 0.2					| Speed limit (100km/h)							|
| 0.0	      			| 								 				|
| 0.0				    | 												|


For the last image, the model is completely sure that this is a Turn Right Ahead (probability about 0.9), and the image does contain a Turn Right Ahead sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.9         			| Turn right ahead								| 
| .1     				| Others 										|
| .0					| 												|
| .0	      			| 								 				|
| .0				    | 												|