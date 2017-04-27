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

plt.close('all')
#importing pipeline
from Pipeline_helper import Pipeline_helper,Line_Sanity, fitCoeff

import convol as con

from moviepy.editor import VideoFileClip

""" Directories for calibration and test images """

calib_image_dir = "camera_cal/"     #directory where clibration images are places
test_image_dir = "test_images/"     # dir for test images

""" Image calibration data """
NX = 9          # no of inner points on xaxis 
NY = 6          # No of inner points on y axis

""" Convolution window variables """
window_width = 50    # Convolution window width
window_height = 80   # Break image into 9 vertical layers since image height is 720
margin = 100          # How much to slide left and right for searching

# font to sue to write on top of video images
font = cv2.FONT_HERSHEY_SIMPLEX

""" Threshold for edge and color detection """

VALX = 40       # minimum threshold for Sobel on x axis
VALY = 40       # minimum threshold for Sobel on y axis
VALM = 50       # # minimum threshold for magnitude based threshold search
s = 3.0         # range for maximum threshold

absx = [5, [VALX,VALX*s]]   # Sobel size and thresholds for x axis derivative
absy = [5, [VALY,VALY*s]]   # Sobel size and thresholds for y axis derivative
mag = [5, [VALM,VALM*s]]    # Sobel size and thresholds for magnitude thresh detection
abs_dir = [25, [0.9,1.1]]   # Sobel size and thresholds for directional detection

thresh_col_s = [100,255]    # Color threshold for s channel

""" global variables to store image count and radius values """
indx = 0
radius = [0,0]


def pipeline(image):
    """
    Main pipeline for handling video images
    
    Input:
        image: vidoe image
    Ouput:
        n_img: processed image with lane lines
    """
    
    centre_x = []
    centre_y = []
    temp =  np.array(image)
    global indx,mat,coeff

    #undistort image
    imge = pipe.cameraDistremove( temp, mat,coeff,  True)
    
    # Caluclate all 4 gradients based threhsold detection
    a,b,c,d = pipe.calcGradients(imge, absx, absy, mag, abs_dir )
    
    #perform filtering using color threshold
    s_binary = pipe.colorThr_s(imge,thresh_col_s)

    # get final binary image   
    combined_binary = pipe.getBinaryImg( a,b,c,d, s_binary )
    
    # get the warped version of the image
    warped,img = pipe.getWarpedImg(imge, combined_binary)   

    # get initial lane points using convolution on warped image
    centre_x, centre_y = con.find_window_centroids(warped, window_width, window_height)
    
    # get all points for left and right lane
    leftx , lefty , rightx, righty = con.find_all_centroids(warped, centre_x, centre_y,
                                                            window_width, window_height, margin)
    
    
    # Sanity check if the convolved points are reasonable, returns True/False 
    val = line.checkLaneWidth(leftx,lefty,rightx,righty)    
        
    # If the queue is filled with all the correct points so far,
    # use thee mean coefficients  
    if (len(line.Leftx) == line.Leftx.maxlen):
        left_p = np.mean(np.array(line.Leftx),axis = 0)
        right_p = np.mean(np.array(line.Rightx),axis = 0)
     
    # Queue not full, check if the last set of points are sane, and use them   
    elif (val):
        left_p, right_p = fitCoeff(leftx, lefty,rightx,righty)
    
    # last set of points bad, use last sane  values, but first check if there are no sane points so far    
    elif (line.last_leftx is not None):   # if 
        left_p, right_p = fitCoeff(line.last_leftx, line.last_y,line.last_rightx, line.last_y)
    
    else:  # Total failure, skip this frame
        return imge
    
    # use coeffiecints to get lane lines    
    left_fitx, right_fitx, ploty = pipe.fitVector(warped,left_p,right_p)
   
    # get radii of left and right line
    l,r = pipe.getRadiusCurve(np.array(left_fitx), np.array(right_fitx),ploty)
   
    # get offset from centre
    center_offset_m = pipe.getCameraOffset(imge,left_fitx, right_fitx)
    
    #get radius via avraging
    radius[1] = float("{0:.2f}".format((l+r)/2.0))
 
    # unwarp the image to get the color image back
    n_img = pipe.warpToColor(imge, warped,left_fitx, right_fitx, ploty)
    
    # if radius exceeds sanity, use last  sane value
    if (radius[1] > 3000):
        radius[1] = radius[0]
    string1 = "Radius of Curvature is = "+str(radius[1])+" (m)"
    string2 = "Camera off centre by = "+str(center_offset_m)+" (m)"
    
    # Write radius and offset on every 10th frame
    if (indx//10 ):
        cv2.putText(n_img,string1,(300,150), font, 1,(255,255,255),2)
        cv2.putText(n_img,string2,(300,250), font, 1,(255,255,255),2)
  
    indx += 1
    radius[0] = radius[1]
    return n_img






# Make an instance of the pipeline
pipe = Pipeline_helper(NX,NY) 
line = Line_Sanity()

""" Collect all images for camera calibration"""

images = pipe.collectImages(calib_image_dir)
test_images = pipe.collectImages(test_image_dir)

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


# remove distortion on test images
output = []
for j in range(len(test_images)):
    output.append(pipe.cameraDistremove(test_images[j],mat,coeff,True))


white_output = 'output.mp4'

clip2 = VideoFileClip('project_video.mp4')#.subclip(0,2)
yellow_clip = clip2.fl_image(pipeline)
yellow_clip.write_videofile(white_output, audio=False)









