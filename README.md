# **Behavioral Cloning** 


**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: https://github.com/markroche92/Behavioural-Cloning-Project/blob/master/center_example.jpg "Center_Example"
[image2]: https://github.com/markroche92/Behavioural-Cloning-Project/blob/master/left_example.jpg "Left_Example"
[image3]: https://github.com/markroche92/Behavioural-Cloning-Project/blob/master/right_example.jpg "Right_Example"
[image4]: https://github.com/markroche92/Behavioural-Cloning-Project/blob/master/bad_bend.jpg "Bad_Bend_Example"
[image5]: https://github.com/markroche92/Behavioural-Cloning-Project/blob/master/center_track_two.jpg "Center_Track_Two"
[image6]: https://github.com/markroche92/Behavioural-Cloning-Project/blob/master/epochs.JPG "Epochs"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* load.py containing the code for loading the training and validation datasets
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

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I initially attempted training the LeNet model from the previous classes. The final model which I used is that which was provided by NVIDIA. The model uses a combination of convolutional and fully connected layers.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 16).


#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 22, 24, 26, 28). A dropout percentage of 20% was chosen.

The model was trained and validated on different data sets to ensure that the model was not overfitting (see load.py). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The number of training epochs was chosen such that the the validation loss was minimized (i.e. further training would have decreased training loss but increased validation loss. This to me would imply overfitting the training set).

#### 3. Model parameter tuning

The model used an adam optimizer, but the learning rate was tuned manually (model.py line 46). A learning rate of 0.0001 was chosen. The value was chosen based on trial and error. However, the premiss assumed was that a slower learning rate would yield a more accurate model (although longer training time, and higher number of epochs).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. After gathering an initial dataset, I assessed the performance of the model on the track. Based on my observations of failure in performance (i.e. which corners that model fails on, and how the model fails on these particular corners), additional data was gathered. 

In particular I gathered more data for the corner after the bridge on the training track. The type of data which I gathered was based around how to recover from different entry angles into the bend.
"

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The model which I have used is the NVIDIA "Parallel ForAll" model which is described here: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model to include Dropout layers between each Convolutional layer. 

Then I retrained the model for a large number of epochs. At a certain stage, I realised that the validation loss stopped decreasing although the training loss was still reducing. This again implied overfitting to me. Therefore, I reduced the number of epochs to the point where the validation loss stopped reducing.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track (bend after the bridge). To improve the driving behavior in these cases, I retreived more training data, as decribed in "4. Appropriate training data".

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

Normalisation and cropping layers are followed by 5 convolutional layers, and 3 fully connected layers.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded a lap on track one using center lane driving. Here is an example image of center lane driving (center camera):

![alt text][image1]

I also used images captured by the right and left cameras on the vehicle:

![alt text][image2]
![alt text][image3]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to. Specifically, I used this method of data gathering for the first bend after the bridge.

![alt text][image4]

I did not augment the dataset other than cropping. The positive impact in performance of the model was clear after adding cropping to the network input. I avoided adding flipped images to the training and validation dataset, as I realised that the cropping would also apply to these images, and the wrong information in the image would be cropped out. As I managed to get strong performance without adding flipped images, I am happy.

In order to help the model generalize, I gathered a single lap of data from the second track as shown below:

![alt text][image5]

After the collection process, I had 16091 training images and 4021 validation images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. I have stored the training/validation data split lists in a pickle file.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 14. The progression of training and validation loss is shown below:

![alt text][image6]

I used an adam optimizer with a learning rate of 0.0001. This learning rate allowed realtively quick training of the model to a sufficient level to drive around the track successfully.
