# RGB image to Grayscale image

## Introduction
This repository will help you to train a AutoEncoder which can convert a RGB image to Grayscale image

## Get Started
### Installing Libraries
- Tensorflow
  * <code>pip install tensorflow</code>
- Numpy
  * <code>pip install numpy</code>
- OpenCV
  * <code>pip install opencv-python</code>
- Glob
  * <code>pip install glob2</code>
  
After installing libraries, run main.py to train the model using the images present in the data folder. Color_images contains all the colored 
images and gray_images contain all the corresponding grayscale images of colored images.
Once the model is trained, you can put any colored images of size 128X128 to convert into grayscale. Put the images in data/input_images. You 
will get the grayscale images in data/output_images.
