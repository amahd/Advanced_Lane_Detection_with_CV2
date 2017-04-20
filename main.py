# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 16:29:16 2017

@author: Aneeq Mahmood
email: aneeq.sdc@gmail.com
"""

# Importing regular modules
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#importing pipeline
from Pipeline import Pipeline



calib_image_dir = "camera_cal/"
test_image_dir = "test_images/"

NX = 9
NY = 6

# Make an instance of the pipeline
pipe = Pipeline(NX,NY) 

""" Collect all images for camera calibration"""

images = pipe.collectImages(calib_image_dir)
test_images = pipe.collectImages(test_image_dir)

#import sys
#sys.exit()

mat,coeff = pipe.cameraCalibMatrices(images)

output = []
#test_image = images[10]
for j in range(len(test_images)):
    output.append(pipe.cameraDistremove(test_images[j],mat,coeff,True))


out = pipe.abs_sobel_thresh( output[0], orient='x', sobel_kernel=3, thresh=(50, 255))
plt.figure()
plt.imshow(out)
