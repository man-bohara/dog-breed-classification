# Udacity DLND Dog Breed Classification Project

## Overview
This is Udacity's DLND project which predicts the dog's breed using image classification techniques. 

## Instructions
1. Install Anaconda and create conda environment.

2. Install numpy, matplotlib, glob2, cv2, tqdm and pytorch.

3. Clone the repository
```
git clone https://github.com/man-bohara/dog-breed-classification.git
```

4. Run following command
```
jupyter notebook dog_app.ipynb
```

5. Follow instructions on the notebook. 

## Model Architecture
In this project, I have built two models.
1. One implemented from the scratch using 4 convolution layers with max pooling and batch normalization and 3 fully connected layers.
2. Another one implemented using pre-trained network (vgg16).

![alt_text](https://github.com/man-bohara/dog-breed-classification/blob/master/Dog_Breed_Classification_Scratch_Model.png)

## Training
Its recommended to train this model on GPU as it takes long time to train this model on large images. You can use utilize GPU provided by AWS cloud.
