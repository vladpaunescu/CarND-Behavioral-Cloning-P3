#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./assets/crop_sample.png "Crop samples"
[image2]: ./assets/cnn-architecture-768x1095.png "NVIDIA CNN architecture"
[image3]: ./assets/center.png "Center sample"
[image4]: ./assets/left.png "Left sample"
[image5]: ./assets/right.png "Right sample"
[image6]: ./assets/original.png "Original image"
[image7]: ./assets/flipped.png "Flipped image"
[image8]: ./assets/brightness_augmentation.png "Brightness augmentation"
[image9]: ./assets/shift_augmentation.png "Shift augmentation"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* model_nvidia_generator_shift_t1.h5 the final NVIDIA model trained on track 1
* model_nvidia_generator_shift_t2.h5 the final NVIDIA model trained on track 2 - it doesn't succesfully drive the car across the full extent of the track
* track1.ogv video recorded on the first track

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 


```sh
 python drive.py model_nvidia_generator_shift_t1.h5
```
Due to the difficulty of the second track, I was unable to drive safely, so the colledcted training data is not good for the second track. The model for bonus track is not that good.

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

The model.py has 2 nets:
* LeNet
* NVIDIA self-driving car model - the model that is saved, and manages to successfully drive on the first track.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model of choice is the [NVIDIA Self-Driving Car model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/)

The model has been tested in real world circumstances.
The model consists of 3 layers of 5x5 convolutions with stride 2x2, and 24, 36 and 48 depth sizes, with an increase in depth space of 12 for each 2x2 subsampling in convolution.
These are followed 2 layers of 3x3 convolutions with stride 1x1, and a depth of 64 for both layer activations.

The convolutional part is followed by a fully connected part. There are 3 Fully connected layers of 100, 50, and 10 neurons. The output is a regression prediction, the vehicle steering angle.

In model.py you can find the keras model definition at lines 141-156:
 
```python

def NvidiaModel():
    model = Sequential()
    model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(160, 320, 3), output_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70, 25), (0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))

    return model


```
 
The model includes RELU activations after each convolutional layer, and fully connected layer. This introduces NonLinearity.
 
 The input has is normalized between -1 and 1, and has 0 mean.
 The input is cropped. The top 70 pixels, and bottom 25 pixels are removed.
 Thus, the area above the horizon and the bottom part, representing th front of the care are removed.
 
 A cropped image as seen by the network is shown below:
 
 ![cropped sample][image1]

####2. Attempts to reduce overfitting in the model


The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 217-220).

```python

    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    train_generator = train_generator(train_samples, batch_size=256)
    validation_generator = valid_generator(validation_samples, batch_size=32)

```
 
 The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 222).

Even the NVIDIA architecture doesn't specify any ReLU activations, I added them to aid learning.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of left, center and right images. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to implement the NVIDIA model architecture, and test it for the current challenge.
THe NVIDIA was designed with the autonomous driving goal in mind, and in order to predict a steering angle .

The model input is with wider images, from driving conditions.
On the other hand, LeNet design had the goal of predicting 32x32 square digits, so the context is not the same.

My first step was to use a convolution neural network model similar to the I thought this model might be appropriate because the driving conditions and regression learning in NVIDIA self-driving car architecture are very similar.

Otherwise, the NVIDIA architecture is fast, and ellegant.
The model trains very quickly on a large dataset, and on a modest GPU.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. Otherwise, the model was not able to capture edge cases. The car was leaving the track at the first road turn, or on the bridge.

To combat the overfitting, I modified the dataset.
With the provided dataset and augmentation, I almost managed to drive one full track. The only problem that persisted was the bridge, and the car was leasving the road there.

Dataset was augmenteted with the following strategies:

* brightness augmentation - random augmentation from a uniform distribution of mean 0.
* flip augmentation - randomly flip the input image since the track has a counter clockwise direction, and the network will be biased towards th left steering angle
* shift augmentation - randomly shift on horizontal and vertical axis. When shifting on horizontal, modify the steering angle accordingly. This prevents overfitting, helping the model generalize the steering angle
* input from all the 3 cameras, with steering angle adjusted (+/-0.2) for left and right cameras. This helps the model generalize the steering angle

Then I used the generator in keras. This input generator yields virtually an infinite amount of training samples. No to input images are the same. This stochasticity in input data acts similar to dropout, so the network never overfits.
 
 Standard augmentations - using random shuffling of training data after each epoch.
 
 The validation set was not augmented.
 
The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, even with all the augmentations and the improved model. The spot was the bridge. In order  to improve the driving behavior in these cases, I decided to collect my own data.
I drove the car for 3 tracks in counter-clockwise direction. 2 tracks were performed using keyboard steering, and one track using mouse steering.
Then I drove the car in clockwise direction for 2 tracks. One track was performed using keyboard steering, and one track using mouse steering.

There are a total of 11k samples (33k left right and center images).
The impact of keyboard steering is visible in the training model. The model is able to learn even more thant staying on the road. It even learns my style of driving - shaky one.

This is the main reason I was unable to train the model on the second track.
It was much more difficult for me to keep on the track when collecting training data. So the training data was very bad.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 143-158) consisted of a convolution neural network with the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image     						| 
| Normalization         | 160x320x3 RGB image     				        |
| Cropping 	            | 65x320x3 RGB image     				        |    
| Convolution 5x5     	| 2x2 stride, valid padding, 31x155x24 	        |
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, 14x76x36 	        |
| RELU					|												|
| Convolution 5x5     	| 2x2 stride, valid padding, 5x36x48 	        |
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, 3x34x64 	        |
| RELU					|												|
| Convolution 3x3     	| 1x1 stride, valid padding, 1x32x64 	        |
| RELU					|												|
| Flatten   	      	| input 1x32x64,  2048              			|
| Fully connected		| 100  									        |
| RELU          		|												|
| Fully connected		| 50  									        |
| RELU          		|												|
| Fully connected		| 10  									        |
| RELU          		|												|
| Fully connected		|  1  									        |
| MSE   				|												|


Here is a visualization of the architecture:


![Nvidia CNN architecture][image2]

####3. Creation of the Training Set & Training Process

**Left/Right/Centre Camera Augmentation**

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Because I couldn't drive on the left/right lane (keyboard steering), I used the left, right, and centre cameras to simulate and compensate for the steering angle. Below you can see examples of left, centre and right cameras for the same sample image.


+0.2 stering angle:

![left][image4]

no correction for steering angle:

![centre][image3]

-0.2 steering angle:

![right][image5]

**Flipping Augmentation**

To augment the data set, I also flipped images and angles thinking that this would improve the generalization due to the fact that the track is left biased. For example, here is an image that has then been flipped:

Original image. Steering angle alpha

![original image][image6]


Flipped image. Steering angle -alpha

![flipped image][image7]

**Brightness Augmentation**

The augmentation process involved visual augmentation of brightness in input pixels.
Here is an example of brightness augmentation:

![brightness augmentation][image8]

**Shift augmentation**

The shift augmentation is beneficial for 2 reasons:
* bumps and hills in the road generalization (vertical shift - no change in steering angle)
* edge cases and recovery from from the left and right sides of the road with the horizontal shift. Steering angle is adjusted with the shift amount. Each horizontal pixel accounts for a 0.3/60 = 0.005 adjustment in steering angle.

You can se a randomly shifted image below:

![shift augmentation][image9]

Since the data points are generated with a training sample generator, that augments the training data in a random fashion, the training data is virtually infinite.

The training/validation split is 0.2. I kept 20 % of the initial, unflipped images as vaildation samples. The validation samples are only randomlyu chosen from left, centre, and right cameras. No other data augmentation is performed.


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I trained the model for 7 epochs, with batch size 256 and 20000 samples per epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The number of epochs was chosen so that to stop when the validation error doesn't decrease any more. This is to prevent overfitting.
