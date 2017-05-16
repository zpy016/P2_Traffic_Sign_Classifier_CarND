# **Traffic Sign Recognition** 


Pengyue Zhang 

Udacity account: zhangpengyue@gmail.com


---

## **Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


## Rubric Points
 Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---


### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. 

I have uploaded my files to github to link to the project repository:

1.1 The [Traffic_Sign_Classifier.ipynb](https://github.com/zpy016/P2_Traffic_Sign_Classifier_CarND/blob/master/Traffic_Sign_Classifier.ipynb) notebook file with all questions answered and all code cells executed and displaying output. Here is a link to my [project code].

1.2 A HTML export of the project notebook with the name [Traffic_Sign_Classifier.html](https://github.com/zpy016/P2_Traffic_Sign_Classifier_CarND/blob/master/Traffic_Sign_Classifier.html).

1.3 [Additional datasets or images](https://github.com/zpy016/P2_Traffic_Sign_Classifier_CarND/tree/master/Test_data_jpg) used for the project that are not from the German Traffic Sign Dataset. 

1.4 A writeup report as a markdown file with the name [R2_Report.md](https://github.com/zpy016/P2_Traffic_Sign_Classifier_CarND/tree/master/R2_Report.md).


### 2. Data Set Summary & Exploration

#### 2.1 Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 27839
* The size of the validation set is 6960
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2.2 Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. For a random image, Image Index ID, Class ID, sign name, the shape of the imagem, and the image itself are shown.

Image Index ID: 2245

Class ID: 3

Sign Name: Speed limit (60km/h)

The shape of the image is (32, 32, 3)

![alt text](https://raw.githubusercontent.com/zpy016/P2_Traffic_Sign_Classifier_CarND/master/images_in_report/sign60_data_explore.png "Visualization")

### 3.Design and Test a Model Architecture

#### 3.1 Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

I have explored three preprocessing method. Normalization, grayscalling and combined. I decided to try normalization so that the data has mean zero and equal variance. This method usually increases the validation rate. I also tried grayscalling because [this paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) suggested that this method had improved the accuracy of image classification.


Here is an example of a traffic sign image before and after pre-processing. The four images are original image, normalized image, gray image, normalized gray image.


![alt text](https://raw.githubusercontent.com/zpy016/P2_Traffic_Sign_Classifier_CarND/master/images_in_report/preprocessing.png "Pre-processing")

It turns out that all of these three pre-processing method improve the accuracy of image classification. Normalization is the best method.



#### 3.2 Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			    	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
|	RELU				|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 			    	|
|	Flatten				| outputs 400									|
| Fully connected		| outputs 180  									|
|	RELU				| dropout 0.6									|
| Fully connected		| outputs 90  									|
|	RELU				| dropout 0.6									|
| Fully connected		| outputs 43  									|
| Softmax				| tf.nn.softmax_cross_entropy_with_logits()     |
 


#### 3.3 Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the adam optimizer. Total epochs is 20 with batch size equals to 128. 

Hyperparameters includes learning rate, keep probablity, mu and sigma. Learning rate is 0.0006. Keep probablity is 0.6. Mu is 0 and sigma is 0.1.

Pre-process method is set to 'norm' which stands for normalization.

#### 3.4 Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.995
* validation set accuracy of 0.984 
* test set accuracy of 0.928

The LeNet model was choosen as a starting point for traffic sign classification bacause of its simplicity and proven effectiveness in classification of images that have similarities. The convolution layer can extract features and the fully connected layers can be trained to properly make decisions.

The original architecture does not include dropout which is effective in reducing overfitting. Thus, I have added dropout to fully connected layer which turned out to be effective. I also tuned hyperparameters of feature extraction layers.

I have tried all three pre-processing methods. The best one turned out to be normalization. Normalized grayscalling is the second best method.

Also, I adjusted the epochs to reduce reduce both underfitting and overfitting. I tunned the learning rate and found out that 0.005 to 0.001 yield very good results. 

 

### 5. Test a Model on New Images

#### 5.1 Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.


Here are the first five German traffic signs that I found on the web:

![alt text](https://raw.githubusercontent.com/zpy016/P2_Traffic_Sign_Classifier_CarND/master/images_in_report/test_images.png "VTraffic Sign Test Images")

These images might be difficult to classify because of low resolution，lighting conditions (e.g. low-contrast), stickers, sun glare and viewpoint variations.

#### 5.2 Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set.

Here are the results of the prediction:

| Image			                               |     Prediction	             				   | 
|:--------------------------------------------:|:---------------------------------------------:| 
| No passing for vehicles over 3.5 metric tons | No passing for vehicles over 3.5 metric tons  | 
| Keep right     		                       | Keep right									   |
| Roundabout mandatory                         | Roundabout mandatory		    			   |
| End of speed limit (80km/h)	               | End of speed limit (80km/h)				   |
| No passing		                           | No passing     							   |


The model was able to correctly predicted 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 12630 images which is 92.8%.

#### 5.3 Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

For all five images, the model is very certain about classification (probability > 99.99% for image 1,2,4 and 5). For the third image, the classifier is a bit less certain (probability of 98.2%) that the image is roundabout mandatory probably because of the stickers.

The following five tables shows the top five soft max probabilities for the five test images. The correct answer is in bold.

The First Image

| Probability         	|     Prediction	        				         | 
|:---------------------:|:--------------------------------------------------:| 
| 9.99981642e-01        | **No passing for vehicles over 3.5 metric tons**   | 
| 1.53403034e-05    	| No passing 									     |
| 1.55594182e-06	    | Slippery road 		                             |
| 1.00069497e-06	    | Speed limit (80km/h)							     |
| 2.67203291e-07		| End of no passing by vehicles over 3.5 metric tons |

The Second Image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00000000e+00        | **Keep right**   								| 
| 6.32613130e-14	    | Turn left ahead					    	    |
| 1.24744321e-15 		| Dangerous curve to the right					|
| 7.79721125e-19		| Go straight or right				 			|
| 2.58768834e-20	    | Roundabout mandatory   						|

The Third Image 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 9.82176661e-01		| **Roundabout mandatory**   					| 
| 1.78229362e-02		| Go straight or left					        |
| 3.60211232e-07 		| Keep left						                |
| 4.02539868e-08		| Turn right ahead	 				            |
| 3.73161324e-09	    | Speed limit (70km/h)					        |

The Fourth Image

| Probability         	|     Prediction	        					     | 
|:---------------------:|:--------------------------------------------------:| 
| 9.99999881e-01		| **End of speed limit (80km/h)**					 | 
| 7.47210436e-08		| End of no passing by vehicles over 3.5 metric tons |
| 8.57244054e-10		| End of all speed and passing limits				 |
| 5.72543957e-10		| Speed limit (80km/h)		 				         |
| 4.39701164e-10	    | Speed limit (60km/h)			                     |

The Fifth Image 

| Probability         	|     Prediction	        					    | 
|:---------------------:|:-------------------------------------------------:| 
| 9.99966145e-01		| **No passing**   								    | 
| 2.27319415e-05		| No passing for vehicles over 3.5 metric tons	    |
| 1.10898891e-05		| Vehicles over 3.5 metric tons prohibited		    |
| 2.15331308e-09		| End of no passing by vehicles over 3.5 metric tons|
| 1.72190950e-09	    | End of no passing 							    |

It is interesting to know that the second image is the easiest one to classifiy and the third image is the most difficult one.

### 6. Conclusions

The model architecture based on LeNet was proven to be effective in classifing real-world traffic signs. The most effective modifications of the model are normalizing input images and adding dropout to the fully connected layers. The prediction accuracy is 98.4% on the validation set and 92.8% on the test set.
