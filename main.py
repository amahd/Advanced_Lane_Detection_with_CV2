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
plt.close('all')
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
import os
if os.path.exists("Coeff.npy"):
    print("Using existing Coefficients ")
    coeff = np.load('Coeff.npy') 
    mat = np.load('Mat.npy')
else:
    print("Calculaitng new distortion Coefficients ")
    mat,coeff = pipe.cameraCalibMatrices(images)
    np.save('Coeff.npy',coeff)
    np.save('Mat.npy',mat)


output = []
#test_image = images[10]
for j in range(len(test_images)):
    output.append(pipe.cameraDistremove(test_images[j],mat,coeff,True))

VALX = 40
VALY = 40
VALM = 50
s = 3.0

absx = [5, [VALX,VALX*s]]
absy = [5, [VALY,VALY*s]]
mag = [5, [VALM,VALM*s]]
abs_dir = [25, [0.9,1.1]]

for im in range(len(output)):

    imge =  np.array(output[im])
    
    
    a,b,c,d = pipe.calcGradients(imge, absx, absy, mag, abs_dir )
    
    final = np.zeros_like(a)
    final[((a == 1) & (b == 1)) | ((c == 1) & (d == 1))] = 1
    
    thresh_col = [170,255]
    
    s_binary = pipe.colorThr(imge,thresh_col)
    
    
    
    
    combined_binary = np.zeros_like(final)
    combined_binary[(s_binary == 1) | (final == 1)] = 1
    
#    img_size = (output[2].shape[1], output[2].shape[0])
#    
#    src_x = [213,1093,561,720]
#    src_y = [711,711,475,475]
#    
#    dst_x = [213,1093,213,1093]
#    
#    dst_y = [711,711,475,475]
#    
#    
#    src = np.float32([[213,711], [1093,711], 
#                                     [561,475], 
#                                     [720,475]])
#    
#    dst = np.float32([[213,711], [1093,711], 
#                                     [213,475], 
#                                     [1093,475]])
#    
##    
#    M = cv2.getPerspectiveTransform(src, dst)
#        # Warp the image using OpenCV warpPerspective()
#    warped = cv2.warpPerspective(imge, M, img_size)
#    
#
#   
#    f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(24, 9))
#    f.tight_layout()
#    ax1.imshow(imge)
#    ax1.plot(src_x,src_y,'r-')
#    ax1.set_title('Sobel X')
#    ax2.imshow(warped)
    
    
    
    
    
    
    
#    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))
#    f.tight_layout()
#    ax1.imshow(a)
#    ax1.set_title('Sobel X')
#    ax2.imshow(b)
#    ax2.set_title('Sobel Y')
#    ax3.set_title('Mag threshold')
#    ax4.set_title('Direction Threshold')
#    ax3.imshow(c)
#    ax4.imshow(d)
#    plt.savefig('result_images/binary'+str(im)+'.jpg')
#
#    
#    
#    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 9))
#    f.tight_layout()
#    ax1.imshow(imge)
#    ax1.set_title('original')
#    ax2.imshow(final)
#    ax2.set_title('Binary Thresholds')
#    ax3.set_title('Color Threshold')
#    ax3.imshow(s_binary)
#    ax4.set_title('Final')
#    ax4.imshow(combined_binary)
#    plt.savefig('result_images/Final'+str(im)+'.jpg')    
#    













